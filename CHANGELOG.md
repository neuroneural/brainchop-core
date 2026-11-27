# Changelog

All notable changes to BrainChop Core are documented in this file.

## [1.1.3] - 2026-11-27

Restores bundler-visible static module worker instantiation in `@brainchop/core` so Vite, Rollup, and Webpack 5 emit the persistent segmentation worker chunk reliably.

### Browser Package (`@brainchop/core`)
- **Static Worker URL Fix:** Restored the direct `new Worker(new URL(__BC_WORKER__, import.meta.url), { type: 'module' })` expression in `js/src/index.ts` while retaining the persistent singleton worker and pooled GPU workspace introduced in `1.1.1`.
- **Bundler Regression Guards:** Added a build-time check in `js/esbuild.config.mjs` (covering `dist/index.js` and all per-model entry points) and a unit assertion in `js/test/webgpu_v4_unit.test.mjs` verifying that every published entry preserves the literal `new Worker(new URL("./worker*.js", import.meta.url), { type: "module" })` call.

## [1.1.2] - 2026-11-27

Upgrades the embedded `mindmap-fast` model (`model16chan18cls`) to clinical-grade MindMap-16 v4.1 weights with the same 84K-parameter architecture and memory footprint, refreshes validation documentation, and standardizes runtime diagnostic log prefixes to `brainchop-core:`.

### Embedded Model Upgrade
- **MindMap-16 v4.1 (`mindmap-fast`):** Replaced legacy v3.5 weights with locked MindMap-16 v4.1 (`model.bin` SHA-256 `f6ea86f9...`, `model.safetensors` SHA-256 `cfc06473...`) at identical ~1.3 GiB peak memory and 2.3 s Apple M1 WebGPU runtime. Delivers clinical-cohort dementia morphometry (OASIS-3 AD vs CN hippocampal `%TBVV` AUROC `0.830`, HarP AD vs CN AUROC `0.881`, Mindboggle-101 Shared-11 Dice `0.822`), thick-slice robustness (`0.772` at 5 mm, `+5.75 pp` over v3.5), lower focal-lesion false positives (`40.0%` surgical resection cavity FP rate), and `2.83%` OASIS-TRT-20 scan–rescan AVD. Updated `src/model_weights_sha.h` and frozen CPU fixtures (`test/labels16*.nii.gz`).

### Diagnostics & Documentation
- **`brainchop-core:` Runtime Log Prefix:** Replaced legacy `brainchopc:` diagnostic, error, tuning, and profiling prefixes with `brainchop-core:` across C/C++/CUDA/HIP/Metal/OpenCL backends, browser WebGPU JS/TS (`@brainchop/core`), and unit/script assertions (preserving `BRAINCHOPC_*` environment variables for backward compatibility).
- **Documentation & Validation Ledger Alignment:** Refreshed `README.md` with compute backend summaries, `v1.1.1` native runtime highlights, locked-ledger validation metrics (including OASIS-3 and resection-cavity rows across all models), and updated SHA-256 weight provenance; aligned `js/README.md`, `docs/building.md`, and `THIRD_PARTY_NOTICES.md` with actual `mindmap-fast` and `mindsnap` CLI/browser capabilities.

## [1.1.1] - 2026-11-26

Non-weight-changing accuracy guards and performance upgrades across native C/CUDA/Metal/CPU engines and Browser WebGPU (`@brainchop/core`), preserving 100% weight compatibility with `1.1.0`.

### Accuracy Guards & PVE Fixes
- **Small-Brain (`< 838.9 mL`) Normalization Guard:** When `q_high <= q_low` on tightly skull-stripped pediatric, infant, or hemispherectomy brains (`histogram[0] >= 95%`), `[q_low, q_high]` now falls back to non-zero foreground quantiles identically across C, CUDA, Metal, and WebGPU (`>= 839 mL` volumes remain bit-identical). Aligned Metal's `normalize_input()` with `BC_MODEL_QUANTILE_MODE`.
- **`mindmap --pve` (`cat_lite`) & Contract Parity:** Included `Brain-Stem (13)` in `BC_PRIOR_WM_MASK` (closing the brainstem hole in `brain_wm` / `brain_brain`), split myelinated deep subcortical nuclei `{7: Thalamus, 10: Pallidum, 17: VentralDC}` 50/50 across GM/WM with `STAT_PURITY = 0.55` to prevent cortical `mu_gm/sd_gm` skew, and aligned the JS `mindmap-fast` PVE capability contract with C.
- **MindGrab Intensity Encoding:** Preserved scaled NIfTI intensities and hardened integer-encoding rules across C and WebGPU.

### Performance & Memory Upgrades
- **Native CUDA ABI 3 (`nvcc 12.8.93`, PTX `8.7`):** Pre-padded 24-channel WMMA weights, fused warp→block→256-stripe FP64 moments inside convolution kernels (preserving byte-for-byte `--tune` invariance), exact `BC_TC_PURE` background row skipping, coalesced `half2` `norm_gelu`, fused L12 + `uint8` classifier, and GPU PVE priors/blur/support (**2.8–3.8×** faster GPU compute, **2.1–3.0×** faster end-to-end CLI wall time on V100 / RTX 2080 Ti).
- **Apple Silicon Metal `--pve` Turbo:** Attached turbo PVE priors, blur, and support kernels directly to Layer-12 `y8` output (`~1.6 GiB` peak footprint vs `~4.0 GiB` FP16 fallback on Apple M1).
- **CPU Turbo on Apple M1 (`ARMv8.2 SDOT`) & Non-VNNI x86-64 (`F16C + AVX2 / AVX-512F`):** Unlocked `vdotq_s32` int8 CPU turbo on Apple M1 (`FEAT_DotProd=1, FEAT_I8MM=0`), added compact `fp16` + `F16C` + `AVX2`/`AVX-512F` kernels and compact 64-voxel block `y8` storage (`~60%` lower peak RAM) with exact `BC_TC_PURE` row skipping.
- **Parallel Native C Host Loops:** Rewrote `src/bwlabel.c` with `uint8` bitshift indexing and parallel z-slab scanline union-find (bit-exact vs 488 browser-oracle phantoms), skipped `private_float_copy` on label/`--pve` CLI runs, and parallelized `conform.c`, `brainchopc.c`, and `cat_lite.c` (**11–20×** faster host pre/post-processing).
- **Browser WebGPU (`@brainchop/core`):** Replaced per-call Worker creation/termination with a lazy persistent singleton Worker + `disposeWorker()`, defaulted warm calls on the same `GPUDevice` to pooled workspace reuse (`keepSharedWorkspace: true`), memoized parsed/packed weights, removed redundant uncompressed output buffer copies, fused constant-input validation into `conform-volume.js`, and reused flat typed-array scratch in `bwlabels.js`.

## [1.1.0] - 2026-11-23

### Breaking Changes

- **Default Model & Output Space (Python CLI, Python API & C Dispatcher):** Compared to `1.0.0` (which shipped on PyPI with `mindmap-fast` and conformed `256³` output by default in the Python CLI/API):
  - Default model is now `mindmap` (full-capacity atrophy-repaired 18-class model) instead of `mindmap-fast`. Pass `-m mindmap-fast` / `model="mindmap-fast"` to retain the faster lightweight model.
  - Default output space is now **native input space** across the C dispatcher, Python CLI, Python API (`save_conform=False`), and browser package. Pass `--save-conform` / `save_conform=True` to preserve the previous conformed `256³` output grid (`--inverse-conform` / `inverse_conform` remain accepted as compatibility aliases).

### Public Release Polish

- **NIfTI-1 Provenance Header:** Stamped version, model, 8-character `model.bin` SHA-256 weight prefix, backend, and precision/mode provenance into the NIfTI-1 `descrip` header (`brainchop <version> <model>/<sha256[:8]> <backend>[ <mode>]`, e.g., `brainchop 1.1.0 mindmap/58006d19 metal turbo`, `brainchop 1.1.0 mindsnap/995cdb52 cpu exact`, `brainchop 1.1.0 mindmap/58006d19 webgpu`) across native and browser outputs.
- **Input Validation & Actionable Diagnostics:** Reject constant / zero-contrast (e.g. all-zero) volumes explicitly in C/WASM and WebGPU instead of emitting hallucinated segmentations; report clear format errors for `.mgz`/`.mgh`/DICOM/NRRD CLI inputs and full dimensions for 4D+ NIfTI inputs; eliminate unused-static compiler warnings in `src/nifti_io.c`.
- **Model Cards, Checksums & Naming:** Added Model Cards with locked validation metrics, honest limitations, and SHA-256 weight provenance table in `README.md`; unified user-facing names (`mindmap`, `mindsnap`, `mindmap-fast`, `mindgrab`, and experimental GM/WM/CSF partial-volume tissue fractions `--pve`), dropping internal nicknames and unproven surface claims.
- **Browser Fallback Progress:** Emit `inference` progress stage when Emscripten CPU/WebGL2 fallback begins execution.
- **Repository & Documentation Hygiene:** Split `docs/building-and-testing.md` into `docs/building.md` and `docs/testing-and-wasm.md`, cleaned dangling internal pointers and archived test tarballs, clarified macOS `14_0` PEP 425 wheel vs `13.2+` native builds and portable wheel CPU ISAs, and updated `THIRD_PARTY_NOTICES.md`.

## [1.0.0] - 2026-11-22

First public release of **BrainChop Core** (`neuroneural/brainchop-core`), replacing the Python/tinygrad runtime (`brainchop==0.2.5`) with a zero-dependency native C/Metal/CUDA/HIP/CPU engine and WebGPU/WebGL2/WASM browser runtime.

### Highlights

- **Embedded Models:** Upgraded embedded checkpoints to **MindMap-24** (`mindmap`), **MindSnap-104** (`mindsnap`), and **MindMap-16** (`mindmap-fast`), alongside modality-agnostic **MindGrab** (`mindgrab`).
- **8-Bit Activation Turbo Mode & Apple M1 Determinism Fix:** Enabled fast 8-bit activation execution by default on Apple Metal, Apple SME, and x86-64 AVX-512 VNNI / AVX-VNNI CPUs (with exact fp32/fp16 available via `--no-turbo`), and fixed an Apple7/M1 `simdgroup_barrier` threadgroup-memory ordering bug in `metal_turbo.h` (`threadgroup_barrier`), ensuring bit-exact run-to-run Metal Turbo determinism.
- **Coil-Flare-Safe Conform & Parity Cleanup:** Unified `f_high = 0.98` conform quantile scaling, `NaN`/`±Inf` sanitization, singular-affine guard, 8-sample majority native-space reslicing, and class-aware connected-component cleanup (`bwlabel_cleanup18`, `bwlabel_cleanup104`) across native and browser runtimes.
- **Portable Prebuilt Wheels & Zero-Dep Python API:** Added prebuilt `win_amd64`, `macosx_arm64`, and `linux_x86_64` wheels with portable ABI checks and zero ML dependencies, plus a backward-compatible stdlib-only `import brainchop` Python API (`Volume`, `load`, `segment`, `save`, `list_models`).
- **Unified Native CLI Dispatcher:** Renamed `brainchop-16chan18cls` to `brainchop-mindmap-fast` and added a tiny zero-dependency C `brainchop` dispatcher for Homebrew, macOS `.pkg`, and standalone Linux/Windows archives.
- **Pure-JS WebGPU V4 Browser Package (`@brainchop/core`):** Replaced `-gpu.wasm` binaries with a pure-JS + WGSL WebGPU V4 runtime supporting `shader-f16`/subgroups/f32 fallback and automatic `LowMem128` + paced Z-chunking for mobile and integrated GPUs, alongside Emscripten WebGL2 and threaded-CPU WASM fallbacks.
- **Explicit-Only OpenCL:** Removed OpenCL from automatic backend selection to avoid slow/TDR-prone integrated-GPU paths; OpenCL remains available via explicit `--backend opencl`.
