# LLMオンボーディングサマリー — flash-kmeans (C++/CUDA + nanobind)

> 新任LLMエージェント／開発者がこのリポジトリに参加する際の初期資料。
> 記入済みの数値・コマンドは実機（RTX 5090 / CUDA 12.8 / pixi 0.77）で確認した事実に基づく。
> 未確認の項目は `TBD` と明記する。

## 1. プロジェクト概要と目的

- **プロジェクト名称・領域:** `flash-kmeans`（upstream: `svg-project/flash-kmeans` のフォーク `yuki-inaho/flash-kmeans`）。
  高速・省メモリな exact K-Means クラスタリング GPU 実装。
- **最終成果物:**
  - C++/CUDA コア（`src/`）＋ nanobind 拡張 `flash_kmeans._flash_kmeans_cpp`。
  - API 互換の薄い Python 層（`flash_kmeans/`）と、元 Triton 実装・純 PyTorch 実装との同等性検証済みテスト群（`tests/`）。
- **ビジネス背景・価値:** 元実装（Python + Triton）のカーネル・反復ループ・大Nストリーミングを C++/CUDA に移植し、
  ランタイム依存を縮小しつつマルチアーキテクチャ（Pascal〜Blackwell）で動作させる。
- **現時点の進捗サマリ (2026-09-14):**
  - `main` にマージ済み（マージコミット `e922416`）。整理済み履歴は `cpp-nanobind-final`。
  - テスト: **79 passed / 3 skipped**（perf 3件は opt-in）。
  - 性能（RTX 5090, fp16, B=32 N=74256 D=128 K=1000）: **バッチ 15.2 ms/iter (40.0 TFLOP/s)**。
    元 Triton 実装は 7.7 ms/iter (79.3 TFLOP/s)。ASSIGN 単体は D=32/64/128/256/512 で 4.8 / 7.9 / 15.6 / 35.1 / 217.7 ms。

## 2. クリティカルな要求・制約

- **pixi 環境でビルド・検証する。** `default`（ビルド用: CUDA 12.8 nvcc / CMake / nanobind）と
  `equiv`（テスト用: torch cu128 / triton / pytest を追加）の2環境。torch・triton は **optional feature** であり、
  拡張本体（C++）は torch 非依存。
- **公開 API と戻り値セマンティクスは元実装互換。**
  - `batch_kmeans_*` は (labels int32, centroids, n_iters) を返し、labels は「最後の割当」に対応。
  - `kmeans_largeN` は「更新後 centroids + 最後の割当 labels」を返す（元実装と同じ非対称仕様）。
  - 空クラスタは旧 centroid を維持、cosine/dot は更新後に再正規化、累積和は fp32。
- **機能同等性は元 Triton 実装との比較で担保する。** 多反復 k-means はカオス的にずれるため、
  主判定は inertia（目的関数）の相対差 ≤1e-3 とし、ラベル一致率は sanity 閾値（≥0.8〜0.99）で見る。
- **カーネル変更時は MMA / WMMA / SIMT 全経路のテストを通す。** `FK_DISABLE_MMA=1` / `FK_DISABLE_WMMA=1` で下位経路を強制。
- **性能回帰はベースラインで担保。** `tests/perf_baselines.json`（GPU名キー, 許容1.3x）。更新は該当GPU実機で行う。
- **largeN は独自ストリームで動く。** 呼び出し前に `torch.cuda.synchronize()` で呼び出し側ストリームと同期する（`ops.py` 実装済み）。これを外すと、生成直後の CUDA テンソルを読み損ねて断続的に壊れる（過去に実際に踏んだ）。
- **数値の取り扱い:** fp32 atomic 加算の順序は非決定 → 同一シードでも runs 間で fp32 丸めレベルの差があり得る（仕様）。
- 乱数は C++ 実装（splitmix64）であり、元実装の `torch.randint` とビット一致しない（シード再現性は本実装内で担保）。

## 3. 参照すべき合意済み資料

| 種別 | ファイル/リンク | 概要・用途 |
|------|------------------|------------|
| 利用手順・性能 | `README.md` | ビルド、API、対応HW/dtype、性能表、プロファイリング |
| 公開 C++ API | `src/flash_kmeans.h` | `batch_kmeans` / `euclid_assign` / `centroid_update` / `kmeans_large_n[_assign]` |
| カーネル | `src/kmeans_common.cuh` | assign（レジスタ/共有メモリ/グローバル）, row_sq, normalize, rng_gather, accumulate, finalize |
| Tensor Core (mma) | `src/assign_mma.cuh` | 手書き `mma.m16n8k16` + レジスタ argmin + cp.async（fp16/bf16, D∈{32,64,128,256,512}, sm_80+）。最速経路 |
| Tensor Core (WMMA) | `src/assign_wmma.cuh` | WMMA 経路（fp16/bf16, D%16==0, sm_80+, 共有メモリ予算内）。mma 非対応次元用 |
| 起動構成 | `src/kmeans_launch.cuh` | デバイス共有メモリ量に基づくカーネル/タイル選択 |
| バッチループ | `src/kmeans_impl.cu` | euclid/cosine/dot 反復、収束判定、per-phase プロファイラ |
| 大N | `src/large_n.cu` | CPU→GPU チャンクストリーミング、マルチGPU gather-reduce-broadcast |
| バインディング | `src/bindings.cpp` | nanobind（生ポインタ＋CUDA stream を渡す設計） |
| Python 層 | `flash_kmeans/ops.py`, `flash_kmeans/interface.py` | マーシャリングのみ。旧モジュールパスは互換 shim |
| テスト | `tests/`（下記 §6） | 同等性・golden・性能回帰 |
| ベンチ | `benchmarks/bench_kmeans.py` | 計測表とベースライン生成 |
| 改修の詳細 | `git log --notes` | 各コミットの git notes に改修意図・性能を記録 |

## 4. タスク境界（任せること / 任せないこと）

### 任せるタスク
- カーネル最適化（プロファイル駆動）、対応次元拡張、テスト追加・修正。
- `README.md` / `docs/` の整合維持、pixi/CMake 設定の維持。
- 性能ベースラインの更新（実機がある場合のみ）。

### 任せないタスク
- upstream（`svg-project/flash-kmeans`）への push / PR 作成。
- 他ブランチ・他 worktree の履歴改変、dirty 差分の revert。
- GPU 実機のない環境での `perf_baselines.json` 更新（環境固有値のため）。
- `pyproject.toml` の依存関係変更（明示指示が必要）。

## 5. 環境セットアップ（pixi）

**前提:** NVIDIA ドライバが CUDA 12.8 ランタイム対応（RTX 50系は 570 以降）。CUDA 12.8 は sm_120 の nvcc に必須。

```bash
cd <repo root>

# 環境作成（default: ビルド用 / equiv: テスト用）
pixi install

# 拡張のビルド（default 環境）
pixi run build

# テスト環境でビルドしてテスト
pixi run -e equiv build
pixi run -e equiv test          # 79 passed, 3 skipped
```

- **アーキテクチャ指定:** 既定は `61-real;75-real;86-real;89-real;120-real;120-virtual`。
  変更は `CUDAARCHS="89-real;120-real" pixi run build` のように環境変数で上書き。
- **GTX 1070 等の実機:** ドライバが CUDA 12.x 対応（≥525）であること。SASS は sm_61 を含む。
- **`pixi run -e equiv ...` を忘れない:** テスト・ベンチ・`python -c` は equiv 環境で実行する。
- **cwd からの import シャドウに注意:** リポジトリ root から `python -c "import flash_kmeans"` すると
  ソースディレクトリが優先され拡張モジュールが見つからない。スクリプトを `/tmp` 等から実行するか、
  `pixi run -e equiv build-editable` を使う。

## 6. 動作確認（スモーク〜回帰）

```bash
# import スモーク（/tmp 等、リポジトリ外から）
pixi run -e equiv python - <<'PY'
import torch
from flash_kmeans import batch_kmeans_Euclid
x = torch.randn(2, 4096, 64, device="cuda", dtype=torch.float16)
labels, cent, iters = batch_kmeans_Euclid(x, 32, max_iters=3)
print(labels.shape, cent.shape, iters)
PY

# テスト構成
pixi run -e equiv pytest -q tests/test_equivalence_torch.py   # 純 torch リファレンス比較・次元網羅
pixi run -e equiv pytest -q tests/test_equivalence_triton.py  # 元 Triton 実装をサブプロセス実行して比較
pixi run -e equiv pytest -q tests/test_golden.py              # 凍結フィクスチャ（upstream 出力）
pixi run -e equiv pytest -q tests/test_api.py tests/test_large_n.py
FK_DISABLE_WMMA=1 pixi run -e equiv pytest -q tests/test_equivalence_torch.py  # SIMT 経路の確認

# 性能
FLASH_KMEANS_PROFILE=1 pixi run -e equiv python benchmarks/bench_kmeans.py  # フェーズ別内訳
pixi run -e equiv bench          # 計測表
pixi run -e equiv perf           # ベースライン回帰（FLASH_KMEANS_PERF=1 相当）
pixi run -e equiv bench-update   # ベースライン更新（実機のみ）
```

## 7. 運用ルール・変更管理

- **コミット粒度:** `build`（pixi/CMake/pyproject） / `core rewrite` / `optimization` / `tests` の4分割を踏襲。
  実例は `git log --oneline cpp-nanobind-final`（`fd415fc` → `498be2c` → `77485e9` → `876cf1c`）。
- **git notes 運用:** 各コミットに改修内容・プロファイル手法・性能を notes で記録している。
  `git log --notes` / `git notes show <hash>` で参照。notes は `refs/notes/commits` として push 済み。
- **性能ベースライン:** 更新は実機で `bench-update`。複数 GPU 分を1ファイルに蓄積する。
- **perf テスト:** 通常テストからはスキップ（`tests/conftest.py`）。`FLASH_KMEANS_PERF=1` で有効化。
- **生成物:** `.pixi/`, `build/`, `__pycache__/`, `*.so` はコミットしない（`.gitignore` / `.pixi/.gitignore`）。

## 8. 既知の落とし穴・トラブルシューティング

| 症状 | 原因 | 対処 |
|------|------|------|
| `ImportError: cannot import name '_flash_kmeans_cpp'`（cwd が repo root） | ソースの `flash_kmeans/` がインストール済みパッケージを shadow | repo 外から実行 or `build-editable` |
| import が書き換え版でなく元 Triton 版を解決してしまう | editable の import hook が PYTHONPATH より優先 | `tests/_triton_ref_runner.py` と同じく `sys.meta_path` から editable hook を除去 |
| WMMA 経路が使われない | D<128 / D%16≠0 / fp32 / sm<80 / 共有メモリ不足（D=512は約136KB必要で不可） | `FK_DISABLE_WMMA` 未設定でも自動で SIMT へフォールバック。設計上の閾値 |
| 長い k-means でラベルが僅かにずれる | fp32 atomic 加算の順序非決定 | 仕様。同等性は inertia で判定 |
| `CUDA error: misaligned address`（自作カーネル変更後） | 2byte 型の 16B ベクトルロードは `d % 8 == 0` が必要（`d % 4 == 0` では不十分） | `src/kmeans_common.cuh` の `row_vec_ok` 条件を確認 |
| Triton 参照テストが skip される | 参照リポジトリ未検出 / サブプロセス失敗 | `FLASH_KMEANS_REF_REPO=/path/to/flash-kmeans` を指定 |

### 付録: ディレクトリ構成

```
src/
  flash_kmeans.h       公開 C++ API
  kmeans_common.cuh    CUDA カーネル（assign 3種, row_sq, normalize, rng, accumulate, finalize）
  assign_wmma.cuh      Tensor Core 経路（fp16/bf16, D>=128, sm_80+）
  kmeans_launch.cuh    起動構成（共有メモリ量・タイル選択）
  kmeans_impl.cu       バッチループ（euclid/cosine/dot）＋ per-phase プロファイラ
  large_n.cu           CPU→GPU ストリーミング（マルチGPU 対応）
  bindings.cpp         nanobind バインディング
flash_kmeans/          Python 層（ops.py / interface.py / 互換 shim / torch_fallback）
tests/                 同等性（torch/triton）・golden・API・largeN・性能回帰
benchmarks/            bench_kmeans.py（計測・ベースライン生成）
docs/                  本ドキュメント
```

### 付録: 性能早見（RTX 5090, fp16, B=32 N=74256 K=1000）

| 項目 | ms | TFLOP/s |
|------|----:|--------:|
| バッチ（5反復平均/反復） | 4.6 | 130 |
| assign D=64（mma） | 3.3 | 93 |
| assign D=128（mma） | 4.7 | 129 |
| assign D=256（mma） | 9.0 | 136 |
| assign D=512（mma） | 21.8 | 112 |
| 参考: 元 Triton | 7.7 | 79.3 |

- D>512 は汎用カーネルへフォールバックし大幅に低速（D=1024 で約0.6 TFLOP/s）。
  改良するなら mma カーネルの A オペランドを D 方向にストリーム化（ldmatrix 化）する。

- 残課題（高速化）: D>512 の mma 対応（A オペランドの D ストリーミング / ldmatrix 化）。
- 残課題（検証）: マルチGPU largeN（本機は1GPUのため未検証）、sm_61/75/86/89 実機検証、他GPUの perf ベースライン追加。

> 本書は 2026-09-14 時点の `main`（`e922416`）に基づく。コマンドはすべて実機で確認済み。
> 未確認事項（マルチGPU・他アーキ実機）は TBD として残している。
