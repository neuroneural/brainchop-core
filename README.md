<p align="center">
  <a href="https://brainchop.org">
    <img src="https://raw.githubusercontent.com/neuroneural/brainchop-core/main/docs/assets/brainchop-icon.png" alt="BrainChop" width="140" />
  </a>
</p>

<h1 align="center">BrainChop Core</h1>

<p align="center">
  <strong>Fast, zero-dependency 3D brain segmentation and skull stripping for desktops, HPC clusters, Python pipelines, and web browsers.</strong>
</p>

<p align="center">
  <a href="https://github.com/neuroneural/brainchop-core/actions/workflows/ci.yml"><img src="https://github.com/neuroneural/brainchop-core/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/brainchop/"><img src="https://img.shields.io/pypi/v/brainchop" alt="PyPI" /></a>
  <a href="https://www.npmjs.com/package/@brainchop/core"><img src="https://img.shields.io/npm/v/@brainchop/core" alt="npm" /></a>
  <a href="https://brainchop.org"><img src="https://img.shields.io/badge/web%20app-brainchop.org-blue" alt="brainchop.org" /></a>
  <a href="https://github.com/neuroneural/brainchop-core/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License: MIT" /></a>
</p>

Powered by four embedded 3D MeshNet models with **no PyTorch, TensorFlow, tinygrad, CUDA Toolkit, or Python ML stack required at runtime**.

- **Web App:** [brainchop.org](https://brainchop.org)
- **Repository:** [github.com/neuroneural/brainchop-core](https://github.com/neuroneural/brainchop-core)
- **License:** MIT (see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md))
- **Intended use:** Research and educational use; not cleared as a medical device for clinical diagnosis.

## Embedded Models & Model Cards

All four models are compiled directly into self-contained binaries (no separate weight downloads or cache directories):

| CLI / API name | Paper name | Executable | Params | Classes | Intended Input | Purpose & Notes |
| --- | --- | --- | ---: | ---: | --- | --- |
| `mindgrab` (`--skull-strip`) | MindGrab | `brainchop-mindgrab` | 145K | 2 | T1, T2, FLAIR, DWI, EPI, MRA, PDw, qT1, CT, PET | Modality-agnostic skull stripping & brain masking (`--mask`, `--border`, `--crop`) |
| `mindmap` *(default)* | MindMap-24 | `brainchop-mindmap` | 188K | 18 | 3D T1w MRI | Primary cortical & subcortical segmentation; optional experimental GM/WM/CSF tissue fractions (`--pve`) |
| `mindsnap` | MindSnap-104 | `brainchop-mindsnap` | 190K | 104 | 3D T1w MRI | Desikan-Killiany cortical + subcortical + 5-part corpus callosum parcellation |
| `mindmap-fast` | MindMap-16 (v4.1) | `brainchop-mindmap-fast` | 84K | 18 | 3D T1w MRI | Fast lightweight 18-class cortical & subcortical segmentation (`16chan18cls` alias); atrophy-repaired and clinical-scan-hardened |

*Training & label provenance:* `mindmap`, `mindsnap`, and `mindmap-fast` are trained on synthetic and T1w brain volumes supervised with FreeSurfer-convention anatomical parcellation labels; `mindgrab` is trained on multi-contrast synthetic and multi-modal neuroimaging volumes ([Fani et al., 2026](https://doi.org/10.1016/j.neuroimage.2026.122074)). The companion manuscript also evaluates a 50-class variant (`MindSnap-50`), omitted from this default runtime bundle to keep package footprints small.

### Validation Summary & Known Limitations

Metrics below are from the locked evaluation ledger (`v4.0` release weights for `mindmap` / `mindsnap` and `v4.1` release weights for `mindmap-fast`; exact checkpoint checksums in [Weight Provenance](#weight-provenance-sha-256)):

| Benchmark | `mindmap` | `mindsnap` | `mindmap-fast` | SynthSeg 2.0 | FastSurferVINN 2.4.2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 6 Manual-Label Cohorts Mean Dice (Mindboggle, IBSR18, LPBA40, HiSUB25, MALC-35, Hammers) | 0.792 | 0.790 | —\* | 0.801 | **0.803** |
| Mindboggle-101 Shared-11 Manual Dice | 0.829 | 0.826 | 0.822 | 0.827 | **0.843** |
| Mindboggle-101 Cortical DKT-62 Dice *(FS-initialized GT)* | — | 0.720 | — | 0.714 (`--parc`) | **0.824** |
| EADC-ADNI HarP (`N=135`) Hippocampal Asymmetry Pearson *r* | 0.794 | 0.684 | 0.640 | 0.792 | **0.811** |
| OASIS-3 AD vs CN (`N=1,043` cohort) Hippocampal `%TBVV` AUROC | 0.805 | **0.835** | 0.830 | 0.819 (`--parc`) | 0.749 |
| Surgical Resection Cavity (`N=50`) False-Positive Parenchyma Rate ↓ | 46.1% | 32.2% | 40.0% | 80.1% | **21.2%** |
| OASIS-TRT-20 Scan–Rescan Repeatability (Shared-11 AVD ↓) | **2.21%** | 2.26% | 2.83% | 2.91% | 3.38% |
| Ultra-Low-Field 64 mT MRI (`ds006557`, `N=23`, 2.34 mm vs paired 3T FastSurfer ref) | 0.636 | **0.649** | —† | 0.621 | 0.397 |
| Severe Acquisition Noise (10% head-masked noise, Mindboggle `N=20`, Shared-11 Dice) | **0.819** | 0.808 | 0.813 | 0.784 | 0.796 |
| Same-Laptop Apple M1 (16 GB) Runtime & Peak Memory *(WebGPU vs CPU / MPS)* | 3.8 s (1.6 GiB) | 3.9 s (1.6 GiB) | **2.3 s (1.3 GiB)** | 656 s (16.7 GiB, swaps) | 311 s (8.4 GiB) |

*(Bold indicates the best value in each row. On the 16 GB Apple M1 laptop, SynthSeg 2.0's 16.7 GiB peak footprint exceeds physical RAM and triggers macOS swap. \* `mindmap-fast` v4.1 was evaluated on Mindboggle `0.822`, IBSR-18 `0.824`, HiSUB25 `0.824`, LPBA40 `0.708`, MALC-15 `0.810`, and CANDI `0.804`; MALC-35 and Hammers were not re-run. † On 64 mT `2.34 mm` MRI (`ds006557`), `mindmap-fast` v4.1 achieves `0.631` self-consistency vs paired 3T (`mindmap`: `0.653`; `mindsnap`: `0.667`).)*

**Choosing between `mindmap` and `mindmap-fast`:** Both 18-class models are trained for clinical and dementia research cohorts. `mindmap-fast` (`v4.1`) delivers strong dementia morphometry (OASIS-3 AD vs CN `%TBVV` AUROC `0.830`, HarP AD vs CN AUROC `0.881`, Emory hippocampal Dice `0.810`), thick-slice robustness (`0.772` at 5 mm), and lower focal-lesion false positives on surgical resections (`40.0%` vs `46.1%`) and stroke (`58.7%` vs `74.7%`) than `mindmap` (`v4.0`) at ~1.3 GiB peak memory and 2.3 s on Apple M1 WebGPU. Choose `mindmap` (`v4.0`) when you need experimental `--pve` tissue fractions, finer sulcal detail, or higher HarP hippocampal asymmetry correlation (`r = 0.794` vs `0.640`).

**Known limitations:**
- **Thick-slice clinical scans (5 mm):** `mindmap-fast` (`v4.1`) is trained with anisotropic clinical degradation and reaches `0.772` Shared-11 Dice on 5 mm acquisitions (`+5.75 pp` over legacy v3.5), narrowing most of the gap to SynthSeg 2.0 (`0.791`) and FastSurferVINN (`0.789`), whereas isotropic-trained `mindmap` / `mindsnap` (`v4.0`) score `0.729` at 5 mm.
- **Sub-voxel grid shifts & large head tilts:** Inference runs directly on a single conformed 1 mm `256³` grid without affine registration or test-time augmentation, so sub-voxel padding/grid shifts or large head tilts can reduce boundary overlap compared to scan–rescan repeatability.
- **Experimental `--pve` tissue fractions:** Continuous GM/WM/CSF/Brain probability maps (`mindmap --pve` / `segmentTissues`) are experimental and have not yet been quantitatively benchmarked against CAT12 or FSL FAST.

### Weight Provenance (SHA-256)

Every binary and npm asset in this repository is built directly from the verified files below:

| Model | Paper Checkpoint | Shipped `model.bin` (SHA-256) | Shipped `model.safetensors` FP16 (SHA-256) | Locked `.pth` (SHA-256) |
| --- | --- | --- | --- | --- |
| `mindgrab` | MindGrab (NeuroImage 2026) | `c9a01490fe2d13acc1515fd019b69db176ebe6f40c9b84aeb77f25acbc2ee0f3` | `21bf4c811a4790fa2498a49f2aa9042f9108964111b54a006fde1194b0e0e7b9` | — |
| `mindmap` | MindMap-24 | `58006d1951212bf33695a8f4409823e872902a6e060b21e87d3fd65c1eb7c4e5` | `b86e11e7884061b81c2ac6d253c90933a3651e0accb7657b9c90f98afcee76ef` | `f6b60026daccabc689ff56bbadd5c1141a5645529f75eb998fe58426ac7f1b48` |
| `mindsnap` | MindSnap-104 | `995cdb522ba941b41ba7ab61d0484e8700f6c3d6084398ce28c6ce75d1ca2a1a` | `e485cac1ffc5bfc55a9cbca848d4fdae4a99a9f06cdec7e5d55e7801b9e44540` | `6b713d9d995bcce0ad5dd6c129f022aeb0241812f5acf64610f4a43ea6544bbc` |
| `mindmap-fast` | MindMap-16 (v4.1) | `f6ea86f9836d60a08b04b358ef5bc278c2f5f204f33fa12295e550333cf0afb3` | `cfc06473c083108de1df8b0d9f5ab2993753efeb4e90f1684060cbda64420591` | `a255782088a5843b8ae10433fb8d487dd36774f852e2427bb7c86b560d1a054c` |

## Quick Install

### Python / PyPI (Zero ML Dependencies)

```bash
pip install brainchop

# Or run immediately with uv (no virtualenv required)
uvx brainchop input.nii.gz -o labels.nii.gz
```

Prebuilt wheels (`win_amd64`, `macosx_14_0_arm64`, `manylinux_2_28_x86_64`) install in ~2 seconds with **zero runtime dependencies** (no PyTorch, no tinygrad, no CUDA Toolkit, no compiler needed). GPU backends (`Metal`, `CUDA`, `HIP`) run at full native speed in prebuilt wheels; for peak **CPU-only** throughput on newer hosts (e.g., Apple SME or x86-64 AVX-512 VNNI), build locally via `pip install --no-binary brainchop brainchop` or `make -C src` so the compiler targets your host CPU ISA rather than a portable baseline. On macOS 13.2–13.x Apple Silicon, pip falls back to a local source build (or install via Homebrew / the signed `.pkg` below), because PEP 425 wheel tags require minor version `0` on macOS 11+ (`macosx_14_0_arm64`). Source installs automatically detect your CPU/GPU (Apple Metal, NVIDIA CUDA Driver API, AMD ROCm HIP) and compile locally. See [docs/pypi.md](docs/pypi.md) for details.

### Homebrew (macOS & Linux)

```bash
brew install neuroneural/tap/brainchop
```

### Browser / JavaScript (npm)

```bash
npm install @brainchop/core
```

Pure-JS WebGPU engine with automatic low-memory staging, plus WebGL2 and multi-threaded WebAssembly CPU fallbacks. See [js/README.md](js/README.md) for browser API usage.

### Standalone Native Binaries (Zero Python)

Download signed macOS Apple Silicon `.pkg` installers (macOS 13.2+) or self-contained Linux / Windows archives from [GitHub Releases](https://github.com/neuroneural/brainchop-core/releases). Each archive includes the `brainchop` dispatcher plus all four standalone executables.

### Build from Source

```bash
make -C src -j4
```

Requires only a C11 compiler (`clang` or `gcc`) and system `zlib`. Builds all four model executables and the zero-dependency `brainchop` dispatcher.

## Usage

### Command Line

```bash
# Default 18-class cortical & subcortical segmentation (mindmap)
brainchop input.nii.gz -o labels.nii.gz

# 104-class Desikan-Killiany parcellation (mindsnap)
brainchop input.nii.gz -m mindsnap -o parcels.nii.gz

# Fast lightweight 18-class segmentation (mindmap-fast)
brainchop input.nii.gz -m mindmap-fast -o labels_fast.nii.gz

# Experimental continuous GM / WM / CSF / Brain partial-volume maps
brainchop input.nii.gz -m mindmap --pve -o tissues.nii.gz

# Skull stripping + binary brain mask (mindgrab)
brainchop input.nii.gz --skull-strip --mask brain_mask.nii.gz -o stripped.nii.gz

# Keep label output in conformed 256^3 1 mm space instead of input space
brainchop input.nii.gz -m mindmap --save-conform -o labels_conformed.nii.gz

# List embedded models or inspect GPU adapters
brainchop --list
brainchop-mindmap -a -1
```

> **Unified output space:** Every interface — the Python `brainchop` CLI, `import brainchop`, standalone native executables (`brainchop-mindmap`, etc.), and `@brainchop/core` — writes outputs in the **original input voxel grid by default**. Pass `--save-conform` (`save_conform=True` in Python, `saveConform: true` in JS) to keep conformed 256³ outputs. (`--inverse-conform` is still accepted as a no-op compatibility flag.) `mindgrab` always writes in input space.

### Python API

The `import brainchop` scripting interface requires **zero external packages** (`numpy` is optional; when installed, `vol.data` returns a `numpy.ndarray`):

```python
import brainchop as bc

print(bc.list_models())

vol = bc.load("input.nii.gz")
labels = bc.segment(vol, "mindmap")          # default: native input space
bc.save(labels, "labels.nii.gz")

# Optional conformed 256^3 output
labels_conf = bc.segment("input.nii.gz", "mindsnap", save_conform=True)

# Experimental GM/WM/CSF/Brain partial-volume tissue fractions
tissues = bc.segment("input.nii.gz", "mindmap", pve=True)
bc.save(tissues["gm"], "gm.nii.gz")
```

### Browser / JavaScript (WebGPU + WebGL2 + WASM CPU)

Install `@brainchop/core` via npm (`npm install @brainchop/core`) — see [js/README.md](js/README.md).

## Compute Backends, Precision & Provenance

`brainchop` automatically picks the fastest validated backend on your machine and falls back cleanly to multi-threaded CPU:

| Backend | Hardware | Auto-Selected | Notes |
| --- | --- | --- | --- |
| **CUDA** | NVIDIA GPUs (`sm70`–`sm120` + `compute_80` PTX, Driver API >= 12.8) | Yes | Embedded offline fatbins (ABI 3, ~0.6–0.8 s on V100 / ~0.9–1.2 s on RTX 2080 Ti) — no CUDA Toolkit or `nvcc` required |
| **HIP** | AMD Radeon (`gfx1151` / ROCm) | Yes | Embedded HSACO artifacts — no `hipcc` required |
| **Metal** | Apple Silicon (macOS 13.2+ native / 14+ wheel) | Yes | 8-bit Turbo (~0.8–1.2 s for `mindmap`/`mindsnap` on M-series; `--pve` turbo at ~1.6 GiB) & exact fp16 |
| **CPU** | Apple SME / NEON (`i8mm` & M1+ `ARMv8.2 SDOT`), x86-64 AVX-512 VNNI / AVX-VNNI / AVX-512F & AVX2+FMA+F16C | Yes | Compact int8/fp16 Turbo (~60% lower RAM) & bit-exact fp32 across compilers/architectures |
| **OpenCL** | Explicit opt-in (`--backend opencl`) | No | Available via `-backend opencl` when explicitly requested |

**Native CLI runtime highlights (`v1.1.1`, end-to-end wall time on `test/t1_crop.nii.gz`, see [BRAINCHOP_CORE_V111_REPORT.md](BRAINCHOP_CORE_V111_REPORT.md)):**

| Hardware / Backend | `mindgrab` | `mindmap-fast` | `mindmap` | `mindsnap` | `mindmap --pve` | Peak Memory |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| NVIDIA V100-SXM2 (`sm_70`, CUDA) | 0.79 s | 0.63 s | 0.79 s | 0.80 s | 1.99 s | 1.3–1.8 GiB VRAM |
| NVIDIA RTX 2080 Ti (`sm_75`, CUDA) | 1.12 s | 0.94 s | 1.22 s | 1.25 s | 2.44 s | 1.2–1.7 GiB VRAM |
| Apple M1 (`Metal` / `ARMv8.2 SDOT` CPU) | — | 8.10 s *(CPU `sdot`)* | — | — | 5.07 s *(Metal turbo)* | 1.0 GiB *(CPU)* / 1.6 GiB *(Metal PVE)* |
| Intel Xeon Gold 6130 (16t, non-VNNI CPU) | 8.77 s | 7.27 s | 14.78 s | 14.21 s | — | 0.9–1.3 GiB RAM |

**Cross-platform determinism, NIfTI headers & provenance:**
- Both the default 8-bit activation path (`turbo`) and GPU FP16 paths can differ at a small fraction of boundary voxels across GPU/CPU architectures (typical whole-brain voxel agreement >99.9%). For longitudinal or multi-site studies, pin a single backend across the cohort — or pass `-cpu 0 --no-turbo` for bit-exact FP32 CPU reproducibility across machines.
- Every output NIfTI-1 file stamps provenance into its 80-byte `descrip` header field in the form:
  ```text
  brainchop <version> <model>/<sha256[:8]> <backend>[ <mode>]
  ```
  where `<sha256[:8]>` is the first 8 lowercase hex characters of the shipped `model.bin` SHA-256 digest (matching [Weight Provenance](#weight-provenance-sha-256) above), so outputs uniquely identify both the exact weights and execution backend even when checkpoints are updated between version bumps (e.g., `brainchop 1.1.4 mindmap/58006d19 metal turbo`, `brainchop 1.1.4 mindsnap/995cdb52 cpu exact`, `brainchop 1.1.4 mindmap/58006d19 metal pve`, `brainchop 1.1.4 mindmap/58006d19 webgpu`).
- Output `intent_code` follows NIfTI-1 conventions: categorical label volumes (`mindmap`, `mindsnap`, `mindmap-fast`) and `mindgrab --mask` set `NIFTI_INTENT_LABEL` (`1002`); continuous partial-volume maps (`--pve`) and raw `--export-classes` maps set `NIFTI_INTENT_NONE` (`0`); `mindgrab` skull-stripped intensity outputs preserve the input volume's intent and datatype.

## Documentation

- [Building, CLI & Compute Backends](docs/building.md) — full option reference, macOS/Linux/Windows builds, `.pkg` notarization, and backend selection.
- [Testing, Benchmarks & WebAssembly](docs/testing-and-wasm.md) — verification suites, hardware gates, and browser WASM builds.
- [PyPI Packaging & Adaptive Hardware Selection](docs/pypi.md) — prebuilt wheels, source-first adaptive rebuilds, and offline installs.
- [Browser Package Documentation](js/README.md) — WebGPU, WebGL2, and threaded WASM CPU API.
- [CUDA Offline Modules](docs/cuda-modules.md) | [HIP Artifacts](docs/hip-artifacts.md) | [v1.1.1 Benchmarks](BRAINCHOP_CORE_V111_REPORT.md) | [Changelog](CHANGELOG.md) | [Contributing](CONTRIBUTING.md) | [Security](SECURITY.md)

## Citation

If you use BrainChop Core in published research, please cite the relevant papers below (see also [CITATION.cff](CITATION.cff)):

1. **MindGrab (`mindgrab` skull stripping):**
   Armina Fani, Mike Doan, Isabelle Le, Alex Fedorov, Malte Hoffmann, Chris Rorden, and Sergey Plis. *MindGrab: A spectrally-motivated architecture for accessible deep learning in neuroimaging.* **NeuroImage**, 338:122074, 2026.
   [DOI: 10.1016/j.neuroimage.2026.122074](https://doi.org/10.1016/j.neuroimage.2026.122074) · [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S1053811926003897) · [PubMed (PMID: 42331200)](https://pubmed.ncbi.nlm.nih.gov/42331200/)

2. **BrainChop Edge Ecosystem (primary BrainChop reference):**
   Sergey M. Plis, Mohamed Masoud, Farfalla Hu, Taylor Hanayik, Satrajit S. Ghosh, Chris Drake, Roger Newman-Norlund, and Christopher Rorden. *Brainchop: Providing an edge ecosystem for deployment of neuroimaging artificial intelligence models.* **Aperture Neuro**, 4, 2024.
   [DOI: 10.52294/001c.123059](https://doi.org/10.52294/001c.123059) · [Aperture Neuro](https://apertureneuro.org/article/123059-brainchop-providing-an-edge-ecosystem-for-deployment-of-neuroimaging-artificial-intelligence-models)

3. **Original BrainChop Software Paper:**
   Mohamed Masoud, Farfalla Hu, and Sergey Plis. *Brainchop: In-browser MRI volumetric segmentation and rendering.* **Journal of Open Source Software**, 8(83):5098, 2023.
   [DOI: 10.21105/joss.05098](https://doi.org/10.21105/joss.05098)

*(Citations for the MindMap and MindSnap segmentation models will be added here upon publication.)*

```bibtex
@article{fani2026mindgrab,
  title   = {MindGrab: A spectrally-motivated architecture for accessible deep learning in neuroimaging},
  author  = {Fani, Armina and Doan, Mike and Le, Isabelle and Fedorov, Alex and Hoffmann, Malte and Rorden, Chris and Plis, Sergey},
  journal = {NeuroImage},
  volume  = {338},
  pages   = {122074},
  year    = {2026},
  doi     = {10.1016/j.neuroimage.2026.122074}
}

@article{plis2024brainchop,
  title   = {Brainchop: Providing an edge ecosystem for deployment of neuroimaging artificial intelligence models},
  author  = {Plis, Sergey M. and Masoud, Mohamed and Hu, Farfalla and Hanayik, Taylor and Ghosh, Satrajit S. and Drake, Chris and Newman-Norlund, Roger and Rorden, Christopher},
  journal = {Aperture Neuro},
  volume  = {4},
  year    = {2024},
  doi     = {10.52294/001c.123059}
}

@article{masoud2023brainchop,
  title   = {Brainchop: In-browser MRI volumetric segmentation and rendering},
  author  = {Masoud, Mohamed and Hu, Farfalla and Plis, Sergey},
  journal = {Journal of Open Source Software},
  volume  = {8},
  number  = {83},
  pages   = {5098},
  year    = {2023},
  doi     = {10.21105/joss.05098}
}
```
