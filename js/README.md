<p align="center">
  <a href="https://brainchop.org">
    <img src="https://raw.githubusercontent.com/neuroneural/brainchop-core/main/docs/assets/brainchop-icon.png" alt="BrainChop" width="120" />
  </a>
</p>

# @brainchop/core

In-browser 3D MRI brain segmentation and skull stripping powered by WebGPU,
with automatic WebGL2 and multi-threaded WebAssembly CPU fallbacks. Pass a
NIfTI `.nii` / `.nii.gz` `ArrayBuffer` in, and get a segmented NIfTI
`ArrayBuffer` out — entirely on the client device with zero server uploads.

- **Web App:** [brainchop.org](https://brainchop.org)
- **Model cards, validation metrics & weight SHA-256 checksums:** [BrainChop Core README](https://github.com/neuroneural/brainchop-core#embedded-models--model-cards)
- **Intended use:** Research and educational use; not cleared as a medical device for clinical diagnosis.

```js
import { segment } from '@brainchop/core'

const input = await file.arrayBuffer()          // .nii or .nii.gz
const { image, backend, elapsedMs } = await segment(input, {
  model: 'mindmap',
  worker: true,
  onProgress: ({ stage, progress }) => {
    console.log(stage, Math.round(progress * 100) + '%')
  },
})
// Output preserves input geometry and gzip encoding by default.
```

## Models

| `model` | Description | Output |
| --- | --- | --- |
| `'mindgrab'` | Modality-agnostic skull stripping (T1, T2, FLAIR, DWI, CT, etc.) | input image with non-brain voxels floored (optional binary `mask`) |
| `'mindmap'` | Primary 18-class cortical and subcortical segmentation (3D T1w) | label volume (or tissue fractions via `segmentTissues`) |
| `'mindmap-omni'` | Contrast-agnostic 18-class segmentation (T2w, FLAIR, T1w, PDw, ULF 64 mT, FDG-PET) | label volume |
| `'mindsnap'` | 104-class Desikan-Killiany cortical + subcortical parcellation | label volume |
| `'mindmap-fast'` | Fast lightweight 18-class cortical and subcortical segmentation (`'16chan18cls'` alias) | label volume |

Each `MODELS` entry (and `listModels()`) carries `colormap`: `{ R, G, B, labels }`,
ready for viewers such as NiiVue.

`mindmap`, `mindmap-omni`, `mindmap-fast`, and `mindsnap` apply class-aware
connected-component cleanup with neighboring-label refill by default. Pass
`{ legacyCleanup: true }` for the original whole-brain-only 26-connected
cleanup. Categorical outputs reslice to the native input grid via midpoint
majority voting; masks use nearest-neighbor resampling; continuous tissue maps
use trilinear resampling. Invalid, empty, or severely fragmented categorical
outputs fail explicitly with `inference-failed`.

## Continuous tissue maps (experimental)

```js
import { segmentTissues } from '@brainchop/core'

const { tissues, backend } = await segmentTissues(input, {
  model: 'mindmap',
  worker: true,
})
// tissues.gm, tissues.wm, tissues.csf and tissues.brain (GM + WM) are NIfTI
// ArrayBuffers. Native geometry and input gzip encoding are preserved by default.
```

`segmentTissues` computes experimental GM/WM/CSF/Brain partial-volume tissue
fractions from `mindmap` across WebGPU, WebGL2, and CPU (`mindmap-omni`,
`mindmap-fast` / `'16chan18cls'`, `mindsnap`, and `mindgrab` do not support
`segmentTissues`).
`saveConform`, `gzipOutput`, `onProgress`, and `webgpuMode` work as for
`segment`; mask, border, and legacy categorical cleanup options are refused.

## Backends & Requirements

`auto` (the default) selects hardware `webgpu`, then hardware `webgl2`, then
multi-threaded `cpu` (on cross-origin isolated pages), reports the backend that
ran in `result.backend`, and throws `no-webgpu` if none are available. Every
output NIfTI-1 file stamps provenance into its 80-byte `descrip` header as
`brainchop <version> <model>/<sha256[:8]> <backend>[ <mode>]` (where
`<sha256[:8]>` is the first 8 hex characters of the `model.bin` SHA-256 digest,
e.g. `brainchop 1.1.2 mindmap/58006d19 webgpu` or
`brainchop 1.1.2 mindmap/58006d19 webgpu pve`). Software rasterisers
(SwiftShader, llvmpipe, lavapipe, WARP, fallback adapters) are never selected
automatically.

| Backend | Requirements | Assets loaded |
| --- | --- | --- |
| `'webgpu'` | WebGPU (`>= 128 MiB` storage buffers), secure context (`https` or `localhost`) | pure JS + `models/<name>.bin` (340–790 KiB; zero WebGPU WASM) |
| `'webgl2'` | WebGL2 with `EXT_color_buffer_float` (use `worker: true` to keep UI responsive) | `brainchop-<name>-gl.{js,wasm}` |
| `'cpu'` | Cross-origin isolated page (`COOP`/`COEP` headers) + shared memory | `brainchop-<name>.{js,wasm}` |

### Enabling the multi-threaded CPU fallback

Serve the **top-level document** with:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

```js
import { checkSupport, checkCpuSupport } from '@brainchop/core'

const report = await checkSupport()
if (!report.supported) console.warn(report.reasons.join('\n'))

const cpu = checkCpuSupport()
if (!cpu.supported) console.warn(cpu.reasons.join('\n'))
```

<details>
<summary>WebGPU implementation details (auto-detection, low-memory mode &amp; precision tiers)</summary>

On WebGPU, the entire pipeline (NIfTI-1 I/O, conform with `fHigh = 0.98`,
quantile normalization, 3D MeshNet WGSL inference, cleanup / PVE fitting, and
native-space reslicing) runs in pure JavaScript + WGSL directly from
`models/<name>.bin`.

Device adaptation uses three layers:
1. **Hardware hints:** Apple Silicon uses the single-encoder path; known
   shared-RAM iGPUs and `<= 128 MiB` storage-buffer devices (such as mobile
   WebGPU) select `LowMem128` (max single buffer `<= 80 MiB` for 15/16ch,
   `<= 120 MiB` for 24ch) with paced Z-chunking.
2. **Calibration probe:** Unrecognized GPUs run a brief 13.5 MiB micro-probe
   (cached for 30 days) to choose standard pipelined, standard paced, or
   `LowMem128` paced execution.
3. **Safe-mode retry:** If an un-paced run hits `device.lost` or OOM, safe mode
   is cached and the run retries once in `LowMem128` paced mode.

Precision tiers degrade automatically with no hard `shader-f16` requirement:
- **Tier 1:** `shader-f16` + `subgroups`
- **Tier 2:** `shader-f16` without `subgroups` (shared-memory epilogue)
- **Tier 3:** `f32` fallback when `shader-f16` is unavailable

</details>

## Options

```ts
segment(input: ArrayBuffer | ArrayBufferView, options: SegmentOptions)
```

| Option | Models | Meaning |
| --- | --- | --- |
| `model` | all | **required**: `'mindgrab'`, `'mindmap'`, `'mindmap-omni'`, `'mindsnap'`, `'mindmap-fast'` (or alias `'16chan18cls'`) |
| `ct` | all | input is CT; convert Hounsfield to Cormack units |
| `comply` | all | insert the compliance transform before conform |
| `saveConform` | label/PVE models | return the conformed 256³ volume instead of reslicing to the input grid |
| `legacyCleanup` | label models | restore whole-brain 26-connected cleanup without per-class refill |
| `mask` | `mindgrab` | also return a binary brain mask in `result.mask` |
| `borderMm` | `mindgrab` | grow the mask border, in mm; implies `mask` |
| `gzipOutput` | all | override gzip compression (defaults to matching input) |
| `backend` | all | `'auto'` (default), `'webgpu'`, `'webgl2'`, `'cpu'` |
| `webgpuMode` | all | WebGPU execution override: `'auto'` (default), `'standard'`, `'low-memory'`, `'recalibrate'` |
| `worker` | all | run segmentation in a Web Worker (**recommended**) |
| `device` | all | reuse an existing `GPUDevice` (implies `webgpu`) |
| `glContext` | all | reuse an existing `WebGL2RenderingContext` (implies `webgl2`) |
| `assetPath` | all | base URL for worker, `models/<name>.bin`, and fallback `.js`/`.wasm` when self-hosting `dist/` |
| `timeoutMs` | all | inference limit (default 120000; 900000 for `cpu` or `auto` in a worker) |
| `onLog` | all | receives diagnostic log lines |
| `onProgress` | all | receives monotonic `{ stage, progress, layer?, totalLayers?, message? }` updates |

Output is in the **input image's own space by default** for every model.

### Bundlers & per-model subpaths

Vite, Rollup, and webpack bundle assets automatically with no manual file
copying or `assetPath`. Vite needs:

```js
// vite.config.js
export default {
  optimizeDeps: { exclude: ['@brainchop/core'] },
  worker: { format: 'es' },
}
```

To deploy only one model's assets, import its subpath entrypoint:

```js
import { segment } from '@brainchop/core/mindgrab'      // or /mindmap, /mindmap-omni, /mindsnap, /mindmap-fast, /16chan18cls
```

## Errors

Every package error throws a `BrainchopError` with a machine-readable `code`:
`no-webgpu`, `no-adapter`, `device-too-small`, `no-f16`, `unsupported-option`,
`bad-input`, `initialization-failed`, `inference-failed`.

## Building & Testing

```sh
npm install
npm run build:webgpu                 # stage models/*.bin, emit types, bundle (no emcc needed)
npm run makeWasm && npm run build    # full build including WebGL2 + CPU WASM fallbacks (needs emcc)
npm run test:unit                    # Node unit tests
npm test                             # real-browser end-to-end tests
```

## Citation

If you use `@brainchop/core` in published research, please cite the relevant papers:

1. **MindGrab (`mindgrab` skull stripping):** Fani, A., Doan, M., Le, I., Fedorov, A., Hoffmann, M., Rorden, C., & Plis, S. (2026). *MindGrab: A spectrally-motivated architecture for accessible deep learning in neuroimaging.* **NeuroImage**, 338, 122074. [DOI: 10.1016/j.neuroimage.2026.122074](https://doi.org/10.1016/j.neuroimage.2026.122074) · [PubMed](https://pubmed.ncbi.nlm.nih.gov/42331200/)
2. **BrainChop Edge Ecosystem:** Plis, S. M., Masoud, M., Hu, F., Hanayik, T., Ghosh, S. S., Drake, C., Newman-Norlund, R., & Rorden, C. (2024). *Brainchop: Providing an edge ecosystem for deployment of neuroimaging artificial intelligence models.* **Aperture Neuro**, 4. [DOI: 10.52294/001c.123059](https://doi.org/10.52294/001c.123059)
3. **Original BrainChop Software Paper:** Masoud, M., Hu, F., & Plis, S. (2023). *Brainchop: In-browser MRI volumetric segmentation and rendering.* **Journal of Open Source Software**, 8(83), 5098. [DOI: 10.21105/joss.05098](https://doi.org/10.21105/joss.05098)

*(Full BibTeX entries are in the [BrainChop Core README](https://github.com/neuroneural/brainchop-core#citation); citations for MindMap and MindSnap will be added upon publication.)*
