# Changelog

All notable changes to BrainChop Core are documented in this file.

## [Unreleased]

### Public Release Polish

- **Unified Defaults:** Standardized default model to `mindmap` (full-capacity atrophy-repaired 18-class model) and default output space to native input space across the C dispatcher, Python CLI, Python API (`save_conform=False`), and browser package (`--save-conform` opts into conformed 256³ space; `--inverse-conform` / `inverse_conform` retained as compatibility aliases).
- **NIfTI-1 Provenance Header:** Stamped model, backend, and precision/mode provenance into the NIfTI-1 `descrip` header (e.g., `brainchop 1.0.0 mindmap metal turbo`, `brainchop 1.0.0 mindsnap cpu exact`, `brainchop mindmap webgpu`) across native and browser outputs.
- **Model Cards, Checksums & Naming:** Added Model Cards with locked validation metrics, honest limitations, and SHA-256 weight provenance table in `README.md`; unified user-facing names (`mindmap`, `mindsnap`, `mindmap-fast`, `mindgrab`, and experimental GM/WM/CSF partial-volume tissue fractions `--pve`), dropping internal nicknames and unproven surface claims.
- **Browser Fallback Progress:** Emit `inference` progress stage when Emscripten CPU/WebGL2 fallback begins execution.
- **Repository & Documentation Hygiene:** Split `docs/building-and-testing.md` into `docs/building.md` and `docs/testing-and-wasm.md`, cleaned dangling internal pointers and archived test tarballs, clarified macOS `14_0` PEP 425 wheel vs `13.2+` native builds, and updated `THIRD_PARTY_NOTICES.md`.

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
