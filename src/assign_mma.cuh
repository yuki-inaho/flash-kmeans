// Hand-written tensor-core assignment kernel (mma.m16n8k16).
//
// Compared to the WMMA path this removes the shared-memory scratch round trip
// and the extra barriers:
//   * each warp owns 16 points and keeps the A operand (an x strip) in
//     registers for the whole K loop,
//   * centroid tiles are staged with cp.async(double buffered) and the B
//     fragments are read directly from shared memory with a padded row stride
//     that makes the per-warp access pattern bank-conflict free,
//   * the argmin/argmax runs on the accumulator fragments in registers and is
//     reduced across the 4 lanes of each row group with shuffles.
#pragma once

#include "kmeans_common.cuh"

namespace fk {

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)

__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return (unsigned)__cvta_generic_to_shared(p);
}

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(
                   smem_addr(smem)),
               "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit() {
  asm volatile("cp.async.commit_group;\n");
}

__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.wait_all;\n");
}

__device__ __forceinline__ void mma_f16(float& c0, float& c1, float& c2, float& c3,
                                        unsigned a0, unsigned a1, unsigned a2,
                                        unsigned a3, unsigned b0, unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2,
                                         float& c3, unsigned a0, unsigned a1,
                                         unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

#endif  // __CUDA_ARCH__ >= 800

// BM = 64 points per block (4 warps x 16 rows), BN = BK centroids per tile.
// D must be a multiple of 16 and a template parameter.
template <typename T, int D, int BK, bool MAXIMIZE>
__global__ void assign_kernel_mma(const T* __restrict__ x,
                                  const float* __restrict__ xsq,
                                  const T* __restrict__ cent,
                                  const float* __restrict__ csq, int n, int k,
                                  int* __restrict__ labels) {
  constexpr int BM = 64;
  constexpr int PAD = 8;  // halves of shared padding per centroid row
  constexpr int DS = D / 16;
  constexpr int NT = BK / 8;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
  extern __shared__ unsigned char mma_smem[];
  T* c_sh = reinterpret_cast<T*>(mma_smem);  // [2][BK][D + PAD]

  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int g = lane >> 2;   // group id (row within the 16-row strip)
  const int t = lane & 3;    // thread id inside the group
  const int b = blockIdx.y;
  const int p0 = blockIdx.x * BM;
  const int row0 = p0 + warp * 16 + g;
  const int row1 = row0 + 8;

  // ---- A operands: the x strip lives in registers for the whole K loop ----
  unsigned a[DS][4];
  const T* x0 =
      x + ((std::size_t)b * n + p0 + warp * 16) * D;
#pragma unroll
  for (int ds = 0; ds < DS; ++ds) {
    const int col = ds * 16 + 2 * t;
    const bool ok0 = row0 < n;
    const bool ok1 = row1 < n;
    const unsigned* r0 =
        reinterpret_cast<const unsigned*>(x0 + (std::size_t)g * D + col);
    const unsigned* r1 =
        reinterpret_cast<const unsigned*>(x0 + (std::size_t)(g + 8) * D + col);
    a[ds][0] = ok0 ? r0[0] : 0u;
    a[ds][1] = ok1 ? r1[0] : 0u;
    a[ds][2] = ok0 ? r0[4] : 0u;  // cols +8..+9
    a[ds][3] = ok1 ? r1[4] : 0u;
  }

  const float x2_0 = (row0 < n) ? xsq[(std::size_t)b * n + row0] : 0.f;
  const float x2_1 = (row1 < n) ? xsq[(std::size_t)b * n + row1] : 0.f;
  float best0 = MAXIMIZE ? -FLT_MAX : FLT_MAX;
  float best1 = MAXIMIZE ? -FLT_MAX : FLT_MAX;
  int bi0 = 0;
  int bi1 = 0;

  const T* cbase = cent + ((std::size_t)b * k) * D;
  const int row_halves = D + PAD;
  const int chunks_per_row = (D * (int)sizeof(T)) / 16;

  auto issue_tile = [&](int st, int k0) {
    const int kcount = (k0 + BK <= k) ? BK : (k - k0);
    T* dst = c_sh + (std::size_t)st * BK * row_halves;
    const int total = kcount * chunks_per_row;
    for (int idx = threadIdx.x; idx < total; idx += blockDim.x) {
      const int r = idx / chunks_per_row;
      const int c = idx - r * chunks_per_row;
      cp_async16(dst + (std::size_t)r * row_halves + c * 8,
                 cbase + (std::size_t)(k0 + r) * D + c * 8);
    }
    cp_async_commit();
  };

  int st = 0;
  issue_tile(0, 0);
  for (int k0 = 0; k0 < k; k0 += BK) {
    cp_async_wait_all();
    __syncthreads();
    if (k0 + BK < k) {
      issue_tile(st ^ 1, k0 + BK);
    }

    float acc[NT][4];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      acc[nt][0] = acc[nt][1] = acc[nt][2] = acc[nt][3] = 0.f;
    }

    const T* tile = c_sh + (std::size_t)st * BK * row_halves;
#pragma unroll
    for (int ds = 0; ds < DS; ++ds) {
#pragma unroll
      for (int nt = 0; nt < NT; ++nt) {
        const unsigned* brow = reinterpret_cast<const unsigned*>(
            tile + (std::size_t)(nt * 8 + g) * row_halves + ds * 16 + 2 * t);
        const unsigned b0 = brow[0];  // k = 2t, 2t+1
        const unsigned b1 = brow[4];  // k = 2t+8, 2t+9
        if constexpr (std::is_same<T, __half>::value) {
          mma_f16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a[ds][0],
                  a[ds][1], a[ds][2], a[ds][3], b0, b1);
        } else {
          mma_bf16(acc[nt][0], acc[nt][1], acc[nt][2], acc[nt][3], a[ds][0],
                   a[ds][1], a[ds][2], a[ds][3], b0, b1);
        }
      }
    }

    // ---- epilogue: argmin/argmax reduced across each 4-lane row group ----
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
      const int kbase = k0 + nt * 8;
      if (kbase >= k) break;
      float v0, v1;
      int i0, i1;
      if (MAXIMIZE) {
        v0 = acc[nt][0];
        i0 = kbase + 2 * t;
        v1 = acc[nt][2];
        i1 = kbase + 2 * t;
        if (acc[nt][1] > v0) {
          v0 = acc[nt][1];
          i0 = kbase + 2 * t + 1;
        }
        if (acc[nt][3] > v1) {
          v1 = acc[nt][3];
          i1 = kbase + 2 * t + 1;
        }
      } else {
        const float c0 = csq[(std::size_t)b * k + kbase + 2 * t];
        const float c1 = csq[(std::size_t)b * k + kbase + 2 * t + 1];
        v0 = (x2_0 + c0) - 2.f * acc[nt][0];
        i0 = kbase + 2 * t;
        v1 = (x2_1 + c0) - 2.f * acc[nt][2];
        i1 = kbase + 2 * t;
        const float v01 = (x2_0 + c1) - 2.f * acc[nt][1];
        if (v01 < v0) {
          v0 = v01;
          i0 = kbase + 2 * t + 1;
        }
        const float v11 = (x2_1 + c1) - 2.f * acc[nt][3];
        if (v11 < v1) {
          v1 = v11;
          i1 = kbase + 2 * t + 1;
        }
      }
#pragma unroll
      for (int off = 1; off <= 2; off <<= 1) {
        const float ov0 = __shfl_xor_sync(0xffffffffu, v0, off);
        const float ov1 = __shfl_xor_sync(0xffffffffu, v1, off);
        const int oi0 = __shfl_xor_sync(0xffffffffu, i0, off);
        const int oi1 = __shfl_xor_sync(0xffffffffu, i1, off);
        if (MAXIMIZE ? (ov0 > v0) : (ov0 < v0)) {
          v0 = ov0;
          i0 = oi0;
        }
        if (MAXIMIZE ? (ov1 > v1) : (ov1 < v1)) {
          v1 = ov1;
          i1 = oi1;
        }
      }
      if (MAXIMIZE ? (v0 > best0) : (v0 < best0)) {
        best0 = v0;
        bi0 = i0;
      }
      if (MAXIMIZE ? (v1 > best1) : (v1 < best1)) {
        best1 = v1;
        bi1 = i1;
      }
    }
    __syncthreads();
    st ^= 1;
  }

  if (t == 0) {
    if (row0 < n) labels[(std::size_t)b * n + row0] = bi0;
    if (row1 < n) labels[(std::size_t)b * n + row1] = bi1;
  }
#else
  (void)x;
  (void)xsq;
  (void)cent;
  (void)csq;
  (void)n;
  (void)k;
  (void)labels;
#endif
}

// Dispatch helper: returns true when the MMA path handled the call.
template <typename T>
inline bool launch_assign_mma(const T* x, const float* xsq, const T* cent,
                              const float* csq, std::int64_t B, std::int64_t N,
                              std::int64_t D, std::int64_t K, bool maximize,
                              int* labels, int device, cudaStream_t st) {
  if constexpr (std::is_same<T, __half>::value ||
                std::is_same<T, __nv_bfloat16>::value) {
    static const bool disabled = [] {
      const char* env = std::getenv("FK_DISABLE_MMA");
      return env != nullptr && env[0] != '\0' && env[0] != '0';
    }();
    if (disabled) return false;
    if (N <= 0 || K <= 0 || B <= 0) return false;
    constexpr std::int64_t kIntMax = 2147483647;
    if (B > kIntMax || N > kIntMax || K > kIntMax) return false;

    int major = 0;
    FK_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                    device));
    if (major < 8) return false;

    constexpr int BM = 64;
    constexpr int BK = 32;
    const dim3 grid((unsigned)(((std::int64_t)N + BM - 1) / BM), (unsigned)B);
    const std::size_t smem =
        (std::size_t)2 * BK * (D + 8) * sizeof(T);  // 2 stages x padded rows

    auto launch = [&](auto d_tag) {
      constexpr int kD = decltype(d_tag)::value;
      if (smem > 48 * 1024) {
        if (maximize) {
          FK_CHECK(cudaFuncSetAttribute(
              assign_kernel_mma<T, kD, BK, true>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        } else {
          FK_CHECK(cudaFuncSetAttribute(
              assign_kernel_mma<T, kD, BK, false>,
              cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
        }
      }
      if (maximize) {
        assign_kernel_mma<T, kD, BK, true><<<grid, 128, smem, st>>>(
            x, xsq, cent, csq, (int)N, (int)K, labels);
      } else {
        assign_kernel_mma<T, kD, BK, false><<<grid, 128, smem, st>>>(
            x, xsq, cent, csq, (int)N, (int)K, labels);
      }
    };

    switch (D) {
      case 32:
        launch(std::integral_constant<int, 32>{});
        return true;
      case 64:
        launch(std::integral_constant<int, 64>{});
        return true;
      case 128:
        launch(std::integral_constant<int, 128>{});
        return true;
      case 256:
        launch(std::integral_constant<int, 256>{});
        return true;
      case 512:
        launch(std::integral_constant<int, 512>{});
        return true;
      default:
        return false;
    }
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
