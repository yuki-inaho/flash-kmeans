# Flash-KMeans (C++/CUDA + nanobind)

Fast, memory-efficient exact K-Means clustering. This branch is a rewrite of
the original Triton implementation: **all numerical work — the assignment
kernels, the centroid update, the k-means iteration loop and the large-N
streaming pipeline — is implemented in C++/CUDA** and exposed to Python through
[nanobind](https://github.com/wjakob/nanobind). The Python layer is a thin
marshalling shim that keeps the original public API (``batch_kmeans_Euclid``,
``FlashKMeans``, ``kmeans_largeN``, ...) unchanged.

| [**Paper (original)**](https://arxiv.org/abs/2603.09229) | [**Upstream repository**](https://github.com/svg-project/flash-kmeans) |
|---|---|

## Design

```
flash_kmeans/            # thin Python layer: pointer/shape/dtype marshalling
  ops.py                 #   batch_kmeans_*, assign/update wrappers
  interface.py           #   FlashKMeans estimator
src/
  flash_kmeans.h         # public C++ API
  kmeans_common.cuh      # CUDA kernels (assign, accumulate, finalize, rng, ...)
  assign_wmma.cuh        # tensor-core (WMMA) assignment for fp16/bf16
  kmeans_launch.cuh      # launch configuration (shared-memory aware)
  kmeans_impl.cu         # batched k-means host loop, assign / centroid update
  large_n.cu             # CPU -> GPU chunked streaming, optional multi-GPU
  bindings.cpp           # nanobind module: _flash_kmeans_cpp
```

The assignment kernel never materialises an ``(N, K)`` distance matrix: a
thread block stages an ``x`` tile in shared memory and streams over the
centroids, computing ``||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2`` with fp32
accumulation (vectorised for 2-byte dtypes, scalar for fp32 / odd ``D``).
Feature dimensions that do not fit in shared memory fall back to a
global-memory variant automatically; the launch configuration is derived from
the device's shared-memory limits at runtime.

### Supported hardware and dtypes

* Architectures: Pascal (GTX 1070, `sm_61`), Turing, Ampere, Ada (RTX 4090,
  `sm_89`), Blackwell (RTX 5090, `sm_120`), plus `sm_120` PTX for future GPUs.
  Override with the `CUDAARCHS` environment variable, e.g.
  `CUDAARCHS=89 pixi run build`.
* dtypes: `float32`, `float16`, `bfloat16` (bf16 uses portable bit-manipulation
  conversions so it also runs on pre-Ampere cards).
* `kmeans_largeN` streams CPU-resident data through the GPU in `BLOCK_N` chunks
  with double-buffered H2D transfers, and partitions the work across all visible
  GPUs (`device=None`) with a manual gather-reduce-broadcast AllReduce.

## Build

The repository ships a [pixi](https://pixi.sh) setup with two environments:

* `default` -- build toolchain only (CUDA 12.8 nvcc, CMake, ninja, nanobind,
  scikit-build-core).  The extension itself is torch-free.
* `equiv` -- adds the optional test dependencies (PyTorch cu128, Triton,
  numpy, pytest, tqdm) used for the Python marshalling layer, the functional
  equivalence checks against the original Triton implementation and the
  performance suite.

```bash
pixi install                     # create both environments
pixi run build                   # build the nanobind extension (default env)
pixi run -e equiv build          # build inside the equivalence/dev env
pixi run -e equiv test           # run the test suite
```

Or from source with a system CUDA toolkit (12.8+ for `sm_120`):

```bash
pip install .
# different targets:
CUDAARCHS="61-real;89-real;120-real" pip install .
```

> The GTX 1070 additionally needs a driver that supports the CUDA 12.x runtime
> (≥ 525). The kernels themselves are compiled to `sm_61` SASS.

## Performance

On an RTX 5090 with `B=32, N=74256, D=128, K=1000`, fp16, over 5 full
iterations (assignment + update + loop overhead):

| implementation                        | ms / iteration | TFLOP/s |
|---------------------------------------|---------------:|--------:|
| this rewrite (C++/CUDA, WMMA path)    |           ~15.2 |    40.0 |
| original Triton (`tl.dot`)            |            7.7  |    79.3 |

Assignment-only throughput by feature dimension (same shapes):

| D   | ms     | TFLOP/s | kernel                       |
|----:|-------:|--------:|------------------------------|
| 32  |   4.8  |   31.6  | SIMT register tiles          |
| 64  |   7.9  |   38.5  | SIMT register tiles          |
| 128 |  15.6  |   38.9  | tensor cores (WMMA)          |
| 256 |  35.1  |   34.6  | tensor cores (WMMA)          |
| 512 | 217.7  |   11.2  | SIMT register tiles          |

Two assignment backends are dispatched automatically:

* **Tensor cores** (`D % 16 == 0`, fp16/bf16, sm_80+ and the shared-memory
  budget fits): a 64-point x tile stays in shared memory while the centroid
  matrix is streamed in double-buffered 32-centroid tiles; each warp computes
  a 16-point x 32-centroid strip with `mma.m16n16k16` and the epilogue turns
  the dot products into `||x - c||^2` scores and updates the running argmin.
* **SIMT register tiles**: the x row is kept in fp32 registers (shared by
  `SPLIT` lanes per point) and centroid tiles are streamed through padded
  shared memory with conflict-free bank access.  This path covers every
  feature dimension up to 512 (rows are zero-padded to a compile-time width,
  so odd and non-power-of-two `D` are supported) and is also the fast path for
  fp32 input.

Larger dimensions use a shared-memory tile kernel with a global-memory
fallback.  `FK_DISABLE_WMMA=1` forces the SIMT path (useful for benchmarking
and for exercising both backends in the tests).

### Profiling and performance regression tests

```bash
FLASH_KMEANS_PROFILE=1 pixi run -e equiv python benchmarks/bench_kmeans.py
pixi run -e equiv bench              # timing table for standard configurations
pixi run -e equiv bench-update      # refresh tests/perf_baselines.json
pixi run -e equiv perf              # regression check against the baseline
```

`FLASH_KMEANS_PROFILE=1` enables CUDA-event instrumentation inside the C++
batch loop and prints a per-phase breakdown (`convert`, `csq`, `assign`,
`accumulate`, `finalize`, `sync`).  On the fp16 `D=128` benchmark the
assignment kernel accounts for ~95% of the iteration time.

## Usage

```python
import torch
from flash_kmeans import batch_kmeans_Euclid

x = torch.randn(32, 75600, 128, device="cuda", dtype=torch.float16)
cluster_ids, centers, n_iters = batch_kmeans_Euclid(
    x, n_clusters=1000, tol=1e-4, verbose=True
)
```

The `faiss`/`sklearn`-style estimator keeps its original surface:

```python
from flash_kmeans import FlashKMeans

km = FlashKMeans(d=128, k=1000, niter=25, dtype=torch.float16)
labels = km.fit_predict(x)          # (B, N) int32
km.predict(x)                       # assignment against the stored centroids
```

Large CPU-resident data (chunked streaming, all GPUs when `device=None`):

```python
import torch
from flash_kmeans import kmeans_largeN

x = torch.randn(100_000_000, 128, pin_memory=True)
labels, centroids = kmeans_largeN(x, n_clusters=8192, max_iters=100, tol=-1)
```

## Functional equivalence

The rewrite is validated against the original implementation with two test
suites:

* `tests/test_equivalence_triton.py` runs the **original Triton package in a
  subprocess** and compares, per mode (euclid/cosine/dot) and dtype:
  * single-iteration assignment agreement (kernel-level equivalence),
  * the clustering objective (inertia) after multiple iterations,
  * the converged partition overlap,
  * the assignment and centroid-update kernels in isolation,
  * the `kmeans_largeN` streaming path.
* `tests/test_equivalence_torch.py` compares against a pure-PyTorch reference
  and covers dimension edge cases (`D` odd, non-power-of-two, `D > 512`),
  empty clusters, convergence and seeded random initialization.
* `tests/test_golden.py` replays frozen outputs produced by the upstream
  Triton implementation (`tests/data/golden_*.pt`, regenerated with
  `tests/_generate_golden.py`), so equivalence stays covered even without the
  upstream checkout.
* `tests/test_perf_regression.py` guards the assignment/iteration timings
  against the records in `tests/perf_baselines.json` (run with `pixi run perf`,
  refresh with `pixi run bench-update`).

Point the Triton tests at any upstream checkout:

```bash
FLASH_KMEANS_REF_REPO=/path/to/flash-kmeans pixi run test
```

Notes on numerical behaviour:

* Random initialization is implemented in C++ (splitmix64 based) with a
  `seed` parameter, so runs are reproducible but *not bit-identical* to
  `torch.randint`-based initialization used by the original.
* Centroid accumulation uses fp32 `atomicAdd`; summation order may vary between
  runs, which is visible only at fp32 round-off level. Long k-means runs are
  chaotic, so equivalence is asserted on the reached objective and partition
  overlap rather than bit-exactness.
* `kmeans_largeN` returns the updated centroids together with the labels of the
  last assignment, matching the original semantics exactly.

## Citation

If you use this codebase, please cite the original work:

```bibtex
@article{yang2026flash,
  title={Flash-KMeans: Fast and Memory-Efficient Exact K-Means},
  author={Yang, Shuo and Xi, Haocheng and Zhao, Yilong and Li, Muyang and Fan, Xiaoze and Zhang, Jintao and Cai, Han and Lin, Yujun and Li, Xiuyu and Keutzer, Kurt and others},
  journal={arXiv preprint arXiv:2603.09229},
  year={2026}
}
```

## License

MIT — see [LICENSE.txt](LICENSE.txt).
