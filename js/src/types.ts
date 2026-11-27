/** Canonical model identifiers. */
export type CanonicalModelName = 'mindgrab' | 'mindmap-fast' | 'mindmap' | 'mindmap-omni' | 'mindsnap'

/**
 * The models this package ships. `'16chan18cls'` is a full backward-compatible
 * alias for `'mindmap-fast'`.
 */
export type ModelName = CanonicalModelName | '16chan18cls'

/**
 * What runs the model.
 *
 * `webgpu` is preferred wherever it exists: it runs via the pure-JS V4 engine
 * (zero WASM binaries, `shader-f16` optional with automatic `f32` fallback,
 * `LowMem128` auto-scaling down to 128 MiB storage buffers). `webgl2` is the
 * WASM fallback for browsers without WebGPU and blocks the calling thread
 * (use `worker: true`). `cpu` is the last-resort pthreads WASM build requiring
 * cross-origin isolation.
 */
export type Backend = 'webgpu' | 'webgl2' | 'cpu'

/** Lifecycle stage reported through `onProgress`. */
export type ProgressStage =
  | 'load'
  | 'conform'
  | 'preprocess'
  | 'inference'
  | 'postprocess'
  | 'reslice'
  | 'complete'

/** Monotonic progress event emitted during segmentation or tissue estimation. */
export interface ProgressEvent {
  stage: ProgressStage
  /** Monotonic completion fraction in [0, 1]. */
  progress: number
  /** Current network layer (1-based) when available during paced inference. */
  layer?: number
  /** Total network layers when available. */
  totalLayers?: number
  /** Optional human-readable status message. */
  message?: string
}

/**
 * Optional override for WebGPU V4 execution layout/pacing auto-detection:
 * - `'auto'` (default): 3-layer hardware auto-detection (Apple fast path,
 *   known-weak iGPU hints, self-calibrating micro-probe, safe-mode retry).
 * - `'standard'`: force standard ZYCX split-plane layout when buffer limits allow.
 * - `'low-memory'`: force `LowMem128` (<=120 MiB max storage buffer) + paced Z-chunking.
 * - `'recalibrate'`: clear cached GPU calibration profile and re-run micro-probe.
 */
export type WebGpuMode = 'auto' | 'standard' | 'low-memory' | 'recalibrate'

/**
 * What each model actually accepts.
 *
 * This mirrors `capabilities` in the model's `model_meta.json`, which is also
 * what generates the CLI's own `--help`. Unsupported per-model options are
 * refused here before any weights or modules are fetched.
 */
export interface ModelCapabilities {
  /** Convert Hounsfield to Cormack units for CT input. */
  ct: boolean
  /** Insert the compliance transform before conform. */
  comply: boolean
  /** Write the conformed 256^3 volume instead of reslicing to the input grid. */
  saveConform: boolean
  /** Also produce a binary brain mask. */
  mask: boolean
  /** Grow the mask border, in mm. Only meaningful alongside `mask`. */
  border: boolean
}

export interface ModelInfo {
  name: ModelName
  /** What the model does, as the executable's own --help states it. */
  description: string
  /** 'mask' overwrites non-brain voxels of the input; 'labels' emits classes. */
  outputKind: 'mask' | 'labels'
  capabilities: ModelCapabilities
  /** Label names and 0-255 RGB per output value (mindgrab: its `mask`). Index 0 is background. */
  colormap: LabelColormap
}

export interface LabelColormap { R: number[]; G: number[]; B: number[]; labels: string[] }

export interface SegmentOptions {
  /** Which model to run. Required. */
  model: ModelName
  /** Treat the input as CT and convert Hounsfield to Cormack units. */
  ct?: boolean
  /** Insert the compliance transform before conform. */
  comply?: boolean
  /**
   * Return the conformed 256^3 volume rather than reslicing back to the input
   * grid. Off by default: output is in the input image's own space unless you
   * ask otherwise.
   */
  saveConform?: boolean
  /** Label models only: restore the original whole-brain cleanup, without per-class refill. */
  legacyCleanup?: boolean
  /**
   * Retained for backward compatibility. On the pure-JS WebGPU V4 backend the
   * V4 ZYCX engine is used unconditionally; ignored on WebGL2/CPU.
   */
  noTurbo?: boolean
  /** Also return a binary brain mask in `mask`. MindGrab only. */
  mask?: boolean
  /** Grow the mask border by this many mm. MindGrab only, implies `mask`. */
  borderMm?: number
  /**
   * gzip the returned buffers. Defaults to whatever the input was, so a
   * `.nii.gz` in gives a `.nii.gz` out.
   */
  gzipOutput?: boolean
  /**
   * What to run on. Defaults to `auto`: hardware WebGPU, otherwise hardware
   * WebGL2, otherwise the threaded CPU module (never a software rasteriser).
   * Naming one explicitly makes an unsupported choice an error rather than a
   * silent downgrade.
   */
  backend?: Backend | 'auto'
  /**
   * Optional override for WebGPU V4 layout & pacing auto-detection. Ignored on
   * non-WebGPU backends.
   */
  webgpuMode?: WebGpuMode
  /**
   * Run the segmentation in a Web Worker instead of on the calling thread.
   * Strongly recommended in a page that must stay responsive, and required to
   * avoid blocking the main thread on `webgl2`.
   *
   * Incompatible with `device` and `glContext`, which cannot cross a worker
   * boundary.
   */
  worker?: boolean
  /** Reuse a device you already own instead of requesting one. Implies webgpu. */
  device?: GPUDevice
  /**
   * Reuse a WebGL2 context you already own instead of creating one. Implies
   * webgl2. It must have been created with `EXT_color_buffer_float` available;
   * `acquireGlContext()` does that for you.
   */
  glContext?: WebGL2RenderingContext
  /**
   * Base URL the worker, model `.bin` weights, and fallback `.js`/`.wasm`
   * assets are served from when the host serves `dist/` itself. Unset, literal
   * `new URL(..., import.meta.url)` and `import(...)` specifiers are used.
   */
  assetPath?: string
  /**
   * Abort inference if it has not finished within this many ms. Default 120000,
   * except `cpu` (and `auto` inside a worker), which gets 900000.
   */
  timeoutMs?: number
  /** Receives diagnostic log lines as they are produced. */
  onLog?: (line: string) => void
  /** Receives monotonic progress updates across pipeline stages and V4 layers. */
  onProgress?: (event: ProgressEvent) => void
}

export interface SegmentResult {
  /** The segmentation, as a NIfTI file image. */
  image: ArrayBuffer
  /** The binary brain mask, present only when `mask` was requested. */
  mask?: ArrayBuffer
  /** Milliseconds spent inside the segmentation pipeline. */
  elapsedMs: number
  /** Which backend actually ran, which matters when `backend` was `auto`. */
  backend: Backend
  /** Whether it ran in a Web Worker. Follows `worker`; never silently different. */
  ranInWorker: boolean
}

/** Tissue fractions use the same input/output grid and compression options. */
export type TissueOptions = Omit<SegmentOptions, 'model' | 'mask' | 'borderMm' | 'legacyCleanup'> & {
  model: 'mindmap'
}
export interface TissueMaps {
  gm: ArrayBuffer; wm: ArrayBuffer; csf: ArrayBuffer
  /** GM + WM combined continuous tissue fraction map (experimental). */
  brain: ArrayBuffer
}
export interface TissueResult {
  tissues: TissueMaps
  elapsedMs: number
  backend: Backend
  ranInWorker: boolean
}
