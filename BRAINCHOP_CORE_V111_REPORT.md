# `brainchop-core` v1.1.1 Engine Accuracy Guards & Performance Upgrade Report

- **Branch:** `perf/v1.1.1-turbo-and-guards` (based on `main` `8742e9d`, leaving `main` untouched)
- **Model Compatibility:** **100% preserved** (`git diff main -- src/model_weights_sha.h src/models/` is empty; `0` weight changes across `mindgrab`, `model16chan18cls` / `mindmap-fast`, `mindmap`, `mindsnap`)
- **CUDA Module Toolchain:** Official NVIDIA CUDA `12.8.93` (`ABI 3`, Driver API `12080`, PTX `8.7`, native images for `sm_70/75/80/86/89/90/120` + `compute_80` PTX)

---

## 1. Summary of Changes by Stage

### Stage 1 — Engine Accuracy & Contract Bug Fixes (`b5ac5ae`)
1. **Critical `< 838.9 mL` (`q95 == 0`) Skull-Stripped / Infant Normalization Guard:**
   - Fixed identically across `src/backend_accel.c`, `src/meshnet_cpu.c`, `src/backend_metal.mm`, and `js/src/webgpu/pipeline.ts`: when `high <= low` and non-zero voxels exist (`histogram[0] < NVOX`), `[q_low, q_high]` is computed over non-zero foreground voxels (`NVOX - histogram[0]`) instead of falling back to `denominator = 1.0f` (`255x` unscaled input).
   - Aligned `src/backend_metal.mm` `normalize_input()` to respect `BC_MODEL_QUANTILE_MODE` (plain rank vs linear) matching C/CUDA/WebGPU.
   - Normal whole-head & `>= 839 mL` scans remain bit-identical; added C (`test/norm_small_brain_check.c`) and JS (`js/test/webgpu_v4_unit.test.mjs`) regression tests covering `~300 mL`, `~500 mL`, sparse constant-foreground, and normal `>= 839 mL` volumes.
2. **`mindmap --pve` (`cat_lite`) & Contract Fixes:**
   - Closed the `Brain-Stem (13)` mask hole by adding class `13` to `BC_PRIOR_WM_MASK` across `src/pve_constants.h`, `src/cat_lite.c`, `src/cuda_kernels.cuh`, `src/metal_turbo.h`, and `js/src/webgpu/probability.js`.
   - Split myelinated deep subcortical nuclei `{7: Thalamus, 10: Pallidum, 17: VentralDC}` 50/50 into GM/WM priors (`BC_PRIOR_MIXED_GWM_MASK`) with `STAT_PURITY = 0.55` so they do not skew cortical `mu_gm / sd_gm`.
   - Fixed the JS-vs-C `mindmap-fast` PVE capability mismatch (`js/src/index.ts` vs `src/model_capabilities.json`), updated `docs/building.md`, and verified `test-cat-lite` (C vs JS max deviation `5.960e-8`, exact float32 epsilon) and `hip-artifacts-check` (33/33 tests).

### Stage 2 — Browser WebGPU (`@brainchop/core`) Latency & Memory Upgrades (`1a5f37d`)
1. **Persistent Singleton Worker & Pooled Workspace Reuse:**
   - Replaced per-call `new Worker(...)` + `worker.terminate()` in `js/src/index.ts` with a lazy persistent singleton Worker plus explicit `disposeWorker()` and idle teardown hooks so Node/test suites exit cleanly without hanging.
   - Updated `js/src/webgpu/pipeline.ts` to default warm calls on the same `GPUDevice` to `keepSharedWorkspace: true` (releasing on device loss, error, validation failure, or explicit teardown) and memoized parsed/packed weights per model (`WeakMap` + content hash cache).
2. **Eliminated Single-Threaded JS Copies & `bwlabels` Allocation Churn:**
   - Removed the redundant `out.slice().buffer` (`16.8–67.1 MB`) copy when `wantGzip == false` (`js/src/index.ts`).
   - Fused `assertNonConstantInput` into `js/src/webgpu/conform-volume.js` (eliminating a full `256^3` pass).
   - Refactored `js/src/webgpu/bwlabels.js` to reuse flat typed-array union-find scratch buffers (`0` `64 MiB` per-call GC churn).
3. **Rolling 64-Slice `stageBuf` Evaluation in `Std` Layout:**
   - Audited `js/src/webgpu/meshnet_fast_v4_runner.js`: `LowMem128` already implements the 64-slice rolling `stageBuf` (`<= 96 MiB`) automatically on constrained GPUs (`maxStorageBufferBindingSize <= 128 MiB`). Porting it into `Std` requires an extra `pFlush` copy pass plus Z-band synchronization barriers per layer (`> 0` kernel slowdown). Keeping `Std` zero-flush while pooling the workspace via `keepSharedWorkspace: true` avoids kernel regressions while eliminating per-call `1,016 MiB` GPU buffer reallocation overhead.

### Stage 3 — Native C Host, CPU Turbo (`ARMv8.2 SDOT` + `x86 F16C/AVX2/AVX-512F`), Metal `--pve` Turbo & Native CUDA ABI 3 (`aeb9f6f`, `221fce8`, `1ce4339`, `8ce46f5`)
1. **Fast Parallel `uint8` `bwlabel.c` & Host Loops:**
   - Rewrote `src/bwlabel.c` using `uint8` labels, power-of-2 bitshift indexing (`>>16`, `(i>>8)&255`, `i&255`), and parallel z-slab scanline union-find (100% bit-exact across 488 browser-oracle phantoms and 1-vs-8 thread invariance).
   - Added `run_model_ex(..., consume_input)` in `src/brainchopc.c` to skip `private_float_copy` on single-shot label/`--pve` CLI runs while preserving non-destructive `run_model()` for multi-call harnesses.
   - Parallelized `src/conform.c`, `src/brainchopc.c`, and `src/cat_lite.c` (`256^3` voxel loops).
2. **CPU Turbo on Apple M1 (`ARMv8.2 SDOT`) & Non-VNNI x86-64 (`AVX2/AVX-512F + F16C`):**
   - Added `mt_sdot_block{1,2}` (`vdotq_s32` via `a ^ 0x80` with `+128 * sum_w` folded into the offset) in `src/meshnet_turbo.h`, unlocking int8 CPU turbo on Apple M1 (`FEAT_DotProd=1, FEAT_I8MM=0`).
   - Added compact `fp16`-storage + `F16C` + `AVX2`/`AVX-512F` FMA kernels (`mt_f16_avx2_segment` for `16ch` to avoid Skylake-SP ZMM downclocking, `mt_f16_avx512_segment` for `24ch`) plus overflow-free `mt_avx2_block{1,2}` int8 fallback.
   - Replaced the `1.61 GB` `float *y` buffer with compact 64-voxel block `uint8` `y8` + `ymom` storage and wired exact pure-background skipping (`bc_tc_make_ex(..., 0)`, `BC_TC_PURE`-only, avoiding exterior-lattice moment drift) -> **~60% lower RAM** and **`>= 0.9974` macro Dice** across all 4 models.
3. **Native CUDA ABI 3 (`src/cuda_kernels.cuh`, `src/backend_cuda_host.h`, `src/cuda_modules/`):**
   - Pre-padded `24ch` WMMA weights (`[27][2][2][16][16]`, `ldm=16`, zero `tile_b` shared-memory staging, direct `k=0` global load + `uint4` `k=1` load, 4× SM occupancy).
   - Fused FP32 warp -> FP64 block -> 256-stripe FP64 moments into both `conv3d_wmma_kernel` and `conv3d_kernel`, eliminating the standalone `reduce_channels_kernel` `768 MiB` HBM re-read and `49,152` FP64 atomics while preserving byte-for-byte tune invariance across `WARPS`.
   - Wired exact `BC_TC_PURE` row-list skipping + `fill_rows_kernel`, `half2` coalesced `norm_gelu_kernel`, single pre-uploaded `device_affine`, fused L12 + `uint8` `classify_u8_kernel` (`16.8 MB` DtoH readback vs `67.1 MB`), and fused L12 `tissue_priors_kernel` + GPU `pve_blur_kernel` + `pve_support_kernel`.
4. **Metal `--pve` Turbo Path (`src/metal_turbo.h`, `src/backend_metal.mm`):**
   - Attached `priors_turbo`, `fill_priors`, `pve_blur_turbo`, and `pve_support_turbo` directly to `metal_turbo.h`'s Layer-12 `y8` output (reusing dead `act[(l+1)&1]` as blur scratch), running `mindmap --pve` on the turbo engine (`1,630 MiB` peak footprint vs `~4,000 MiB` FP16 fallback).

---

## 2. Measured Before-vs-After Benchmarks (`test/t1_crop.nii.gz`)

### A. Native CUDA — `Tesla V100-SXM2-32GB` (`sm_70`, Linux x86-64, 8 host threads, warm tuned)

| Model / Mode | Baseline Wall (ms) | v1.1.1 Wall (ms) | Wall Speedup | Baseline GPU (ms) | v1.1.1 GPU (ms) | GPU Speedup | Baseline VRAM (MiB) | v1.1.1 VRAM (MiB) | Golden Parity (Dice / Diff) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `mindgrab` | 1,720.2 | **793.2** | **2.17×** | 470.8 | **239.9** | **1.96×** | 1,334 | 1,350 | fg Dice **0.999998** (`6/16.78M` vs base; `36` vs CPU) |
| `mindmap-fast` (`--save-conform`) | 1,557.1 | **633.3** | **2.46×** | 240.3 | **109.7** | **2.19×** | 1,334 | 1,342 | macro Dice **0.99985** (`197/16.78M` vs base; `412` vs CPU) |
| `mindmap` (`--save-conform`) | 2,291.8 | **788.6** | **2.91×** | 1,005.9 | **263.8** | **3.81×** | 1,846 | 1,854 | macro Dice **0.99962** (`378/16.78M` vs `labels24`, `99.970%` fg) |
| `mindsnap` (`--save-conform`) | 2,378.4 | **801.0** | **2.97×** | 1,004.9 | **265.3** | **3.79×** | 1,846 | 1,854 | macro Dice **0.99949** (`655/16.78M` vs `labels104`, `99.949%` fg) |
| `mindmap --pve` | 5,005.0 | **1,988.4** | **2.52×** | 1,005.4 | **265.7** | **3.78×** | 1,846 | 1,854 | `cuda_pve_check` PASS: RMSE `0.00033–0.00105` (gate `0.00784`) |

*V100 `mindmap` stage breakdown:* `conv` `826.3 -> 237.5 ms` (**3.48×**), `moments` `44.9 -> 0.2 ms` (**224×**), `norm+gelu` `134.7 -> 26.0 ms` (**5.18×**), `host pve priors` `1,949.7 -> 33.2 ms` (**58.7×**).

### B. Native CUDA — `NVIDIA GeForce RTX 2080 Ti` (`sm_75`, Linux x86-64, 8 host threads, warm tuned)

| Model / Mode | Baseline Wall (ms) | v1.1.1 Wall (ms) | Wall Speedup | Baseline GPU (ms) | v1.1.1 GPU (ms) | GPU Speedup | Baseline VRAM (MiB) | v1.1.1 VRAM (MiB) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mindgrab` | 2,021.9 | **1,116.8** | **1.81×** | 512.3 | **306.1** | **1.67×** | 1,182 | 1,198 |
| `mindmap-fast` (`--save-conform`) | 1,718.4 | **940.4** | **1.83×** | 268.0 | **141.8** | **1.89×** | 1,182 | 1,190 |
| `mindmap` (`--save-conform`) | 2,617.5 | **1,222.4** | **2.14×** | 1,223.2 | **438.1** | **2.79×** | 1,694 | 1,702 |
| `mindsnap` (`--save-conform`) | 2,750.1 | **1,248.6** | **2.20×** | 1,222.0 | **438.4** | **2.79×** | 1,694 | 1,702 |
| `mindmap --pve` | 4,220.9 | **2,441.4** | **1.73×** | 1,224.6 | **439.4** | **2.79×** | 1,694 | 1,702 |

*RTX 2080 Ti `mindmap` stage breakdown:* `conv` `995.3 -> 403.2 ms` (**2.47×**), `moments` `82.5 -> 0.2 ms` (**412×**), `norm+gelu` `145.4 -> 34.7 ms` (**4.19×**), `host pve priors` `772.8 -> 64.0 ms` (**12.1×**).

> **CUDA Hardware & VRAM Note:** All cluster A40 (`sm_86`) and A100 (`sm_80`) nodes were occupied by queued workloads during the final benchmark run (`StartTime` > 10h), so matched before-vs-after real-silicon benchmarks and full CUDA test suites were verified on both Volta (`Tesla V100-SXM2-32GB`, `sm_70`) and Turing (`RTX 2080 Ti`, `sm_75`), while ABI 3 fatbins include native cubins for all 7 architectures (`sm_70/75/80/86/89/90/120` + `compute_80` PTX). CUDA retains full FP16 ping-pong activation buffers (`1,190–1,854 MiB` total process VRAM) rather than 8-bit activation compression so every model passes `test-cuda`'s strict `<= 64 / 1024` exact voxel diff gates (`6–655` differing voxels out of `16.78M`) and `cuda_pve_check.py`'s `<= 2/255` RMSE gate (`0.00033–0.00105`).

### C. Native CPU — `Intel Xeon Gold 6130` (Non-VNNI x86-64, 16 threads, Linux)

| Model | Baseline Wall (ms) | v1.1.1 Wall (ms) | Speedup | Baseline MaxRSS (MiB) | v1.1.1 MaxRSS (MiB) | RAM Delta | Kernel Selected | Golden Dice (Conformed / Native) |
|---|---:|---:|---:|---:|---:|---:|---|---|
| `mindgrab` | 9,120 | **8,770** | **1.04×** | 2,148 | **913** | **-57.5%** | `fp16, avx2` | **0.99996 / 0.99995** (`99.992%` fg) |
| `mindmap-fast` | 11,050 | **7,270** | **1.52×** | 2,195 | **889** | **-59.5%** | `fp16, avx2` | **0.99828 / 0.99792** (`99.801%` fg) |
| `mindmap` | 16,890 | **14,780** | **1.14×** | 3,221 | **1,291** | **-59.9%** | `fp16, avx512f` | **0.99829 / 0.99770** (`99.791%` fg) |
| `mindsnap` | 17,080 | **14,210** | **1.20×** | 3,221 | **1,293** | **-59.9%** | `fp16, avx512f` | **0.99743 / 0.99694** (`99.751%` fg) |

*Host pipeline breakdown (`mindmap`, Xeon Gold 6130, 16 threads):* `copy` `29.2 -> 0.0 ms`, `conform` `154.5 -> 12.8 ms` (**12.1×**), `to uint8 + setup` `138.8 -> 4.2 ms` (**33.0×**), `bwlabel + border` `338.6 -> 28.6 ms` (**11.8×**), `reslice + convert` `132.3 -> 4.7 ms` (**28.1×**) — total host overhead **`793.4 -> 50.3 ms` (`15.8×` faster)**.

### D. Apple M1 (`FEAT_I8MM=0`, `FEAT_DotProd=1`) — Isolated Single-Scan Verification

| Backend / Model | Baseline Behavior (`v1.1.0`) | v1.1.1 Measured | Peak Memory Footprint | Accuracy Parity |
|---|---|---|---|---|
| CPU `ARMv8.2 SDOT` (`mindmap-fast`, `-cpu 0 --save-conform`) | Rejected (`lacks i8mm/VNNI`), fell back to `~2.2 GiB` FP32 exact | **8,100 ms** (`segment 7,694 ms`, `bwlabel 26.4 ms`, `int8, sdot`) | **971 MiB** (`974 MiB` RSS, **-56% RAM**) | macro Dice **0.99812** (`99.800%` fg agree vs `labels16.nii.gz`) |
| Metal `--pve` Turbo (`mindmap`, `-backend metal --pve`) | Fell back to `~4,000 MiB` FP16 exact Metal engine | **5,070 ms** (`segment 4,089 ms`, `pve priors 48.9 ms`, `pve fit 166.7 ms`) | **1,630 MiB** (`701 MiB` RSS, **-59% footprint**) | `cuda_pve_check` PASS vs CPU exact: GM `0.00635`, WM `0.00425`, CSF `0.00378` (gate `0.00784`) |

### E. Browser WebGPU (`@brainchop/core` JS)

| Component | Baseline (`v1.1.0`) | v1.1.1 Implementation | Verification |
|---|---|---|---|
| Worker lifecycle (`js/src/index.ts`) | `new Worker()` + `terminate()` per call | Lazy persistent singleton Worker + `disposeWorker()` + idle cleanup | `webgpu_v4_unit.test.mjs` (9/9 PASS) |
| WebGPU workspace (`js/src/webgpu/pipeline.ts`) | `keepSharedWorkspace: false` + `releaseV4Workspace()` every call (`1,016 MiB` churn) | `keepSharedWorkspace: true` across warm calls on same `GPUDevice`, memoized weights, release on error/loss/dispose | `npm run build:webgpu && npm run test:unit` (PASS) |
| Host JS copies & `bwlabels.js` | `out.slice().buffer` (`16.8–67.1 MB`), separate `assertNonConstantInput` scan, `64 MiB` per-call UF allocations | Zero-copy `pack()`, fused constant-input check in `conform-volume.js`, reusable flat typed-array UF scratch | `conform` / `bwlabels` / Stage 2 unit suite (PASS) |

---

## 3. Verification Matrix (100% Passing)

- **C / CPU / Host / Guards (Linux Xeon & macOS M1):**
  - `make -C src models` (4/4 models, 0 compiler warnings on Clang & GCC)
  - `make -C src test-guards` (including `norm-small-brain-check` for `mindgrab` & `mindmap`)
  - `make -C src test-label-cleanup` (selftests + 488 browser-oracle phantoms, bit-exact)
  - `make -C src test-label-quality` (including 14-iteration `label-failure-check`)
  - `make -C src test-cat-lite` (thread-count invariance + C-vs-JS max deviation `5.960e-8` + 3 volume perturbations)
  - `make -C src test-cpu-turbo` (`avx512f`, `avx2`, `sdot` block checks `0/400` mismatches; x86 `fp16` rel_err `0–6.1e-7`; all 8 conformed & native Dice gates `0.99694–0.99996`)
  - `make -C src test-models` (`0` differing voxels across all 4 models conformed & native + thread invariance + `mindgrab_scaling.py`)
- **CUDA (`12.8.93` ABI 3, real silicon `V100` `sm_70` & `RTX 2080 Ti` `sm_75`):**
  - `make -C src cuda-modules-check` & `python3 test/test_cuda_modules.py` (17/17 OK)
  - `make -C src test-cuda-driver-contract` (4/4 models PASS)
  - `make -C src CUDA=1 test-cuda-legacy-contract` (4/4 models PASS)
  - `make -C src cuda-module-smoke` (4/4 models × 36 kernels PASS)
  - `make -C src test-cuda` (all 4 models + native + byte-for-byte `--tune` vs pinned identity + adapter bounds PASS)
  - `python3 test/cuda_pve_check.py --driver src` (all 4 input/space cells PASS, RMSE `0.00033–0.00105`)
- **HIP / PyPI / Python Unit Suites:**
  - `python3 -m unittest discover -s test -p "test_*.py"` (**96/96 tests OK**, including `hip-artifacts-check`, CUDA pruning, and sdist inventory)
- **JS / WebGPU (`@brainchop/core`):**
  - `cd js && npm run build:webgpu && npm run test:unit` (**9/9 tests PASS**)
