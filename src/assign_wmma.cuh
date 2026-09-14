// Tensor-core (WMMA) assignment kernel for fp16/bf16 inputs.
//
// The block keeps a 64-point x tile in shared memory and streams the centroid
// matrix in BK-centroid tiles (double buffered).  Each warp computes a 16x16
// tile of x . c^T with mma.m16n16k16 (fp16/bf16 inputs, fp32 accumulation);
// the epilogue converts the dot products into ||x - c||^2 scores and updates
// the running argmin/argmax per point.
#pragma once

#include "kmeans_common.cuh"

#if defined(__CUDACC__)
#include <mma.h>
#endif

namespace fk {

constexpr int kWmmaBlockN = 64;  // points per block (4 warps x 16 rows)

// Requirements: T is __half or __nv_bfloat16, D % 16 == 0 and the runtime
// dispatch checks sm_80+ (the body is compiled out on older targets).
template <typename T, int BK, bool MAXIMIZE>
__global__ void assign_kernel_wmma(const T* __restrict__ x,
                                   const float* __restrict__ xsq,
                                   const T* __restrict__ cent,
                                   const float* __restrict__ csq, int n, int d,
                                   int k, int* __restrict__ labels) {
  constexpr int BN = kWmmaBlockN;
  constexpr int WM = 16;  // warp tile rows

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  using namespace nvcuda;
  extern __shared__ unsigned char wmma_smem[];
  T* x_sh = reinterpret_cast<T*>(wmma_smem);                  // [BN][d]
  T* c_sh = x_sh + (std::size_t)BN * d;                       // [2][BK][d]
  float* scratch =
      reinterpret_cast<float*>(c_sh + (std::size_t)2 * BK * d);  // [BK][BN]
  float* xsq_sh = scratch + (std::size_t)BK * BN;              // [BN]

  const int b = blockIdx.y;
  const int p0 = blockIdx.x * BN;
  const int t = threadIdx.x;
  const int warp = t >> 5;
  const int d8 = d / 8;

  auto load_ctile = [&](int buf, int k0) {
    const int kcount = (k0 + BK <= k) ? BK : (k - k0);
    T* dst = c_sh + (std::size_t)buf * BK * d;
    for (int idx = t; idx < kcount * d8; idx += blockDim.x) {
      const int row = idx / d8;
      const int c8 = (idx - row * d8) * 8;
      const uint4 v = *reinterpret_cast<const uint4*>(
          cent + ((std::size_t)b * k + k0 + row) * d + c8);
      *reinterpret_cast<uint4*>(dst + (std::size_t)row * d + c8) = v;
    }
  };

  // ---- x tile (zero padded rows past the end) ----------------------------
  const T* xbase = x + ((std::size_t)b * n) * d;
  for (int idx = t; idx < BN * d8; idx += blockDim.x) {
    const int row = idx / d8;
    const int c8 = (idx - row * d8) * 8;
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (p0 + row < n) {
      v = *reinterpret_cast<const uint4*>(xbase + (std::size_t)(p0 + row) * d + c8);
    }
    *reinterpret_cast<uint4*>(x_sh + (std::size_t)row * d + c8) = v;
  }
  if (t < BN) {
    xsq_sh[t] = (p0 + t < n) ? xsq[(std::size_t)b * n + p0 + t] : 0.f;
  }
  load_ctile(0, 0);
  __syncthreads();

  float best = MAXIMIZE ? -FLT_MAX : FLT_MAX;
  int best_k = 0;
  int cur = 0;

  for (int k0 = 0; k0 < k; k0 += BK) {
    const int kcount = (k0 + BK <= k) ? BK : (k - k0);
    if (k0 + BK < k) {
      load_ctile(cur ^ 1, k0 + BK);  // overlaps with the mma below
    }
    const T* cbuf = c_sh + (std::size_t)cur * BK * d;

    // Each warp computes a 16-point x BK-centroid strip: BK/16 mma tiles.
    constexpr int KT = BK / 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[KT];
#pragma unroll
    for (int j = 0; j < KT; ++j) wmma::fill_fragment(acc[j], 0.f);
    for (int d0 = 0; d0 < d; d0 += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> a_frag;
      wmma::load_matrix_sync(a_frag, x_sh + (std::size_t)warp * WM * d + d0, d);
#pragma unroll
      for (int j = 0; j < KT; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> b_frag;
        wmma::load_matrix_sync(b_frag, cbuf + (std::size_t)j * 16 * d + d0, d);
        wmma::mma_sync(acc[j], a_frag, b_frag, acc[j]);
      }
    }
#pragma unroll
    for (int j = 0; j < KT; ++j) {
      wmma::store_matrix_sync(scratch + warp * WM + (std::size_t)j * 16 * BN,
                              acc[j], BN, wmma::mem_col_major);
    }
    __syncthreads();

    // ---- epilogue: one thread per point ----------------------------------
    if (t < BN) {
      const float x2 = xsq_sh[t];
      const float* srow = scratch + t;
      float bv = best;
      int bi = best_k;
      for (int kk = 0; kk < kcount; ++kk) {
        const float dot = srow[(std::size_t)kk * BN];
        const float score =
            MAXIMIZE ? dot : (x2 + csq[(std::size_t)b * k + k0 + kk] - 2.f * dot);
        const bool better = MAXIMIZE ? (score > bv) : (score < bv);
        if (better) {
          bv = score;
          bi = k0 + kk;
        }
      }
      best = bv;
      best_k = bi;
    }
    __syncthreads();
    cur ^= 1;
  }
  if (t < BN && p0 + t < n) {
    labels[(std::size_t)b * n + p0 + t] = best_k;
  }
#else
  (void)x;
  (void)xsq;
  (void)cent;
  (void)csq;
  (void)n;
  (void)d;
  (void)k;
  (void)labels;
#endif
}

// Returns true when the WMMA path launched.  `csq` may be null for MAXIMIZE
// modes; `xsq` may be null for MAXIMIZE modes.
template <typename T>
inline bool launch_assign_wmma(const T* x, const float* xsq, const T* cent,
                               const float* csq, std::int64_t B, std::int64_t N,
                               std::int64_t D, std::int64_t K, bool maximize,
                               int* labels, int device, cudaStream_t st) {
  if constexpr (std::is_same<T, __half>::value ||
                std::is_same<T, __nv_bfloat16>::value) {
    // Escape hatch for benchmarking / debugging: FK_DISABLE_WMMA=1 forces the
    // SIMT register kernel.
    static const bool disabled = [] {
      const char* env = std::getenv("FK_DISABLE_WMMA");
      return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    if (disabled) return false;
    if (D <= 0 || N <= 0 || K <= 0 || (D % 16) != 0) return false;
    // The SIMT register kernel is faster for small feature dimensions.
    if (D < 128) return false;
    constexpr std::int64_t kIntMax = 2147483647;
    if (B > kIntMax || N > kIntMax || D > kIntMax || K > kIntMax) return false;

    // mma.m16n16k16 with fp16/bf16 inputs requires Ampere and up.
    int major = 0;
    FK_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                    device));
    if (major < 8) return false;

    constexpr int BN = kWmmaBlockN;
    constexpr int BK = 32;
    const dim3 grid((unsigned)(((std::int64_t)N + BN - 1) / BN), (unsigned)B);
    const std::size_t smem = (std::size_t)BN * D * sizeof(T) +
                             (std::size_t)2 * BK * D * sizeof(T) +
                             (std::size_t)BK * BN * sizeof(float) +
                             BN * sizeof(float);
    int max_optin = 0;
    FK_CHECK(cudaDeviceGetAttribute(&max_optin,
                                    cudaDevAttrMaxSharedMemoryPerBlockOptin,
                                    device));
    if (smem > (std::size_t)max_optin) return false;  // fall back to SIMT
    if (smem > 48 * 1024) {
      if (maximize) {
        FK_CHECK(cudaFuncSetAttribute(
            assign_kernel_wmma<T, BK, true>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      } else {
        FK_CHECK(cudaFuncSetAttribute(
            assign_kernel_wmma<T, BK, false>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
      }
    }
    if (maximize) {
      assign_kernel_wmma<T, BK, true><<<grid, 128, smem, st>>>(
          x, xsq, cent, csq, (int)N, (int)D, (int)K, labels);
    } else {
      assign_kernel_wmma<T, BK, false><<<grid, 128, smem, st>>>(
          x, xsq, cent, csq, (int)N, (int)D, (int)K, labels);
    }
    return true;
  } else {
    (void)x;
    (void)xsq;
    (void)cent;
    (void)csq;
    (void)B;
    (void)N;
    (void)D;
    (void)K;
    (void)maximize;
    (void)labels;
    (void)device;
    (void)st;
    return false;
  }
}

}  // namespace fk
