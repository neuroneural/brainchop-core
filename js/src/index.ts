import { BrainchopError } from './error.js'
import {
  acquireDevice,
  canonicalModelName,
  checkSupport,
  WEBGPU_MIN_BUFFER_BYTES,
} from './device.js'
import { acquireGlContext, checkWebgl2Support, releaseGlContext } from './webgl2.js'
import { gunzip, gzip, isGzip } from './gzip.js'
import { assetUrl, CPU_TIMEOUT_MS, run } from './run.js'
import { MODEL_BINS, MODULES } from './modules.js'
import { COLORMAPS } from './colormaps.js'
import { runWebGpuSegment, runWebGpuTissues, disposeWebGpuResources } from './webgpu/pipeline.js'
import type {
  Backend, ModelInfo, ModelName, SegmentOptions, SegmentResult, TissueOptions, TissueResult, TissueMaps,
} from './types.js'

export { BrainchopError } from './error.js'
export type { BrainchopErrorCode } from './error.js'
export {
  checkSupport,
  acquireDevice,
  ACTIVATION_BYTES,
  activationBytes,
  WEBGPU_MIN_BUFFER_BYTES,
  v4StdBufferBytes,
  canonicalModelName,
} from './device.js'
export type { DeviceReport } from './device.js'
export {
  checkWebgl2Support, acquireGlContext, releaseGlContext, GL_ACTIVATION_BYTES,
} from './webgl2.js'
export type { Webgl2Report } from './webgl2.js'
export type {
  Backend,
  CanonicalModelName,
  ModelName,
  ModelInfo,
  ModelCapabilities,
  LabelColormap,
  ProgressStage,
  ProgressEvent,
  WebGpuMode,
  SegmentOptions,
  SegmentResult,
  TissueOptions,
  TissueResult,
  TissueMaps,
} from './types.js'

/**
 * Per-model capabilities, colormaps, and metadata.
 * `'16chan18cls'` is a full backward-compatible alias for `'mindmap-fast'`.
 */
export const MODELS: Record<ModelName, ModelInfo> = {
  mindgrab: {
    name: 'mindgrab',
    colormap: COLORMAPS['mindgrab'],
    description: 'skull stripping in any modality',
    outputKind: 'mask',
    capabilities: {
      ct: true, comply: true, saveConform: false, mask: true, border: true,
    },
  },
  'mindmap-fast': {
    name: 'mindmap-fast',
    colormap: COLORMAPS['mindmap-fast'],
    description: 'fast lightweight 18-class cortical and subcortical segmentation',
    outputKind: 'labels',
    capabilities: {
      ct: true, comply: true, saveConform: true, mask: false, border: false,
    },
  },
  '16chan18cls': {
    name: '16chan18cls',
    colormap: COLORMAPS['16chan18cls'],
    description: 'fast lightweight 18-class segmentation (alias for mindmap-fast)',
    outputKind: 'labels',
    capabilities: {
      ct: true, comply: true, saveConform: true, mask: false, border: false,
    },
  },
  mindmap: {
    name: 'mindmap',
    colormap: COLORMAPS['mindmap'],
    description: '18-class cortical and subcortical segmentation',
    outputKind: 'labels',
    capabilities: {
      ct: true, comply: true, saveConform: true, mask: false, border: false,
    },
  },
  mindsnap: {
    name: 'mindsnap',
    colormap: COLORMAPS['mindsnap'],
    description: '104-class Desikan-Killiany parcellation',
    outputKind: 'labels',
    capabilities: {
      ct: true, comply: true, saveConform: true, mask: false, border: false,
    },
  },
}

/** Set by esbuild per entry point: './worker.js', or './worker-<model>.js'. */
declare const __BC_WORKER__: string

/**
 * The models this entry point can run: all four canonical models on the full
 * entry, or one model for a per-model subpath import.
 */
export function listModels(): ModelInfo[] {
  const hasCanonicalFast = Object.hasOwn(MODEL_BINS, 'mindmap-fast')
  return Object.values(MODELS).filter((m) => {
    if (!Object.hasOwn(MODEL_BINS, m.name)) return false
    // Avoid listing both 'mindmap-fast' and its legacy alias '16chan18cls' in the
    // same list when 'mindmap-fast' is present.
    if (m.name === '16chan18cls' && hasCanonicalFast) return false
    return true
  })
}

function reject(model: ModelName, option: string, why: string): never {
  throw new BrainchopError('unsupported-option',
    `${model} does not support \`${option}\`: ${why}`)
}

function buildArgs(options: SegmentOptions): { args: string[]; maskPath?: string } {
  // Object.hasOwn, not truthiness: MODELS['__proto__'] / 'constructor' /
  // 'toString' are all truthy on an object literal, so a plain-JS caller could
  // walk straight past this guard and fail later with a bare TypeError.
  const model = Object.hasOwn(MODELS, options.model) ? MODELS[options.model] : undefined
  if (!model)
    throw new BrainchopError('unsupported-option',
      `unknown model '${options.model}'; expected one of ` +
      `${Object.keys(MODELS).join(', ')}`)
  // A per-model entry carries one model's assets; assetPath loads any by URL.
  const canonical = canonicalModelName(options.model)
  if (!options.assetPath && !Object.hasOwn(MODEL_BINS, options.model) && !Object.hasOwn(MODEL_BINS, canonical))
    throw new BrainchopError('unsupported-option',
      `${options.model} is not in this entry point; import '@brainchop/core' ` +
      `or '@brainchop/core/${options.model}'`)

  const args: string[] = []
  if (options.ct) args.push('--ct')
  if (options.comply) args.push('--comply')
  if (options.legacyCleanup) {
    if (model.outputKind !== 'labels')
      reject(options.model, 'legacyCleanup', 'this compatibility option is only for label models')
    args.push('--legacy-cleanup')
  }

  if (options.saveConform) {
    // MindGrab's output IS the input image with non-brain voxels floored, so it
    // is inherently in native space and the conformed grid is not on offer.
    if (!model.capabilities.saveConform)
      reject(options.model, 'saveConform',
        'its output is the input image with non-brain voxels floored, so it is ' +
        'inherently in the input space')
    args.push('--save-conform')
  }

  // borderMm is checked before mask, so that passing it to a labels model names
  // borderMm rather than the `mask` it happens to imply.
  if (options.borderMm !== undefined && !model.capabilities.border)
    reject(options.model, 'borderMm', 'a mask border is only meaningful for a mask model')
  if (options.mask && !model.capabilities.mask)
    reject(options.model, 'mask', 'a brain mask is only meaningful for a mask model')

  let maskPath: string | undefined
  if (options.mask || options.borderMm !== undefined) {
    maskPath = '/mask.nii'
    args.push('--mask', maskPath)
  }
  if (options.borderMm !== undefined) {
    if (!Number.isFinite(options.borderMm) || options.borderMm < 0)
      throw new BrainchopError('unsupported-option',
        `borderMm must be a non-negative number, got ${options.borderMm}`)
    args.push('--border', String(options.borderMm))
  }

  return { args, maskPath }
}

const WORKER_IDLE_MS = 120_000

interface PersistentWorkerEntry {
  key: string
  worker: Worker
  idleTimer: ReturnType<typeof setTimeout> | null
}

let singletonWorker: PersistentWorkerEntry | null = null
let workerQueue: Promise<unknown> = Promise.resolve()

function terminateSingletonWorker(): void {
  const entry = singletonWorker
  singletonWorker = null
  if (!entry) return
  if (entry.idleTimer !== null) clearTimeout(entry.idleTimer)
  entry.worker.onmessage = null
  entry.worker.onerror = null
  entry.worker.onmessageerror = null
  try { entry.worker.terminate() } catch { /* ignore */ }
}

function scheduleWorkerIdleDispose(entry: PersistentWorkerEntry): void {
  if (entry.idleTimer !== null) clearTimeout(entry.idleTimer)
  const timer = setTimeout(() => {
    if (singletonWorker === entry) terminateSingletonWorker()
  }, WORKER_IDLE_MS)
  if (typeof (timer as unknown as { unref?: () => void }).unref === 'function') {
    ;(timer as unknown as { unref: () => void }).unref()
  }
  entry.idleTimer = timer
}

/**
 * Explicitly terminate the persistent segmentation worker and release any
 * pooled WebGPU device/workspace buffers retained across warm calls.
 */
export function disposeWorker(): void {
  terminateSingletonWorker()
  disposeWebGpuResources()
}

/**
 * Run one segmentation in a lazily created persistent Web Worker.
 *
 * Reusing the worker preserves its WebGPU device, compiled WGSL pipelines,
 * parsed/packed weights, and pooled GPU workspace across warm calls. On timeout
 * or worker crash the worker is terminated immediately so cancellation is real;
 * an unref'd idle timer and `disposeWorker()` handle deterministic cleanup.
 */
async function runInWorker(
  input: Uint8Array,
  options: SegmentOptions,
  tissueMode = false,
): Promise<{ image: ArrayBuffer; mask?: ArrayBuffer; tissues?: TissueMaps; elapsedMs: number; backend: Backend }> {
  if (typeof Worker === 'undefined')
    throw new BrainchopError('unsupported-option',
      'this environment has no Worker; run with worker: false')

  const workerKey = options.assetPath
    ? `custom:${assetUrl(__BC_WORKER__, options.assetPath)}`
    : 'bundled'

  const executeJob = async (): Promise<{
    image: ArrayBuffer; mask?: ArrayBuffer; tissues?: TissueMaps; elapsedMs: number; backend: Backend
  }> => {
    if (singletonWorker && singletonWorker.key !== workerKey) {
      terminateSingletonWorker()
    }
    if (!singletonWorker) {
      // Keep `new Worker(new URL(__BC_WORKER__, import.meta.url), { type: 'module' })`
      // as a direct expression so Vite / Rollup / Webpack statically detect and
      // emit the worker bundle.
      const worker = options.assetPath
        ? new Worker(assetUrl(__BC_WORKER__, options.assetPath), { type: 'module' })
        : new Worker(new URL(__BC_WORKER__, import.meta.url), { type: 'module' })
      singletonWorker = { key: workerKey, worker, idleTimer: null }
    }
    const entry = singletonWorker
    if (entry.idleTimer !== null) {
      clearTimeout(entry.idleTimer)
      entry.idleTimer = null
    }
    const { worker } = entry
    const { device, glContext, onLog, onProgress, worker: _w, ...rest } = options
    void device; void glContext; void _w

    if (rest.assetPath !== undefined) rest.assetPath = assetUrl('', rest.assetPath)

    let timer: ReturnType<typeof setTimeout> | undefined
    let hardReset = false
    try {
      return await new Promise((resolve, reject) => {
        worker.onmessage = (event) => {
          const message = event.data
          if (message.type === 'log') { onLog?.(message.line); return }
          if (message.type === 'progress') { onProgress?.(message.event); return }
          if (message.type === 'done') {
            resolve({
              image: message.image, mask: message.mask, tissues: message.tissues,
              elapsedMs: message.elapsedMs, backend: message.backend,
            })
            return
          }
          reject(new BrainchopError(message.code, message.message, message.log))
        }
        worker.onerror = (event) => {
          hardReset = true
          reject(new BrainchopError('inference-failed',
            `the segmentation worker failed to start or threw: ${event.message ?? 'unknown'}`))
        }
        worker.onmessageerror = () => {
          hardReset = true
          reject(new BrainchopError('inference-failed',
            'the segmentation worker sent a message that could not be deserialized'))
        }

        const named = options.backend && options.backend !== 'auto' ? options.backend : null
        const limit = options.timeoutMs ?? (named && named !== 'cpu' ? 120_000 : CPU_TIMEOUT_MS)
        timer = setTimeout(() => {
          hardReset = true
          reject(new BrainchopError('inference-failed',
            `the segmentation did not finish within ${limit} ms`))
        }, limit)
        worker.postMessage({ input, options: rest, tissueMode })
      })
    } finally {
      clearTimeout(timer)
      if (hardReset) {
        if (singletonWorker === entry) terminateSingletonWorker()
        else { try { worker.terminate() } catch { /* ignore */ } }
      } else if (singletonWorker === entry) {
        worker.onmessage = null
        worker.onerror = null
        worker.onmessageerror = null
        scheduleWorkerIdleDispose(entry)
      }
    }
  }

  const run = workerQueue.then(executeJob, executeJob)
  workerQueue = run.then(() => {}, () => {})
  return run
}

/**
 * Decide which backend runs, and acquire or validate what it needs.
 */
async function acquire(options: SegmentOptions): Promise<
    { backend: Backend; device?: GPUDevice; glContext?: WebGL2RenderingContext
      ownsContext: boolean }> {
  const named = options.backend && options.backend !== 'auto' ? options.backend : null
  if (options.device && named && named !== 'webgpu')
    throw new BrainchopError('unsupported-option',
      `backend: '${named}' was requested but a GPUDevice was supplied; a device ` +
      'selects the webgpu backend. Drop the device, or drop the backend.')
  if (options.glContext && named && named !== 'webgl2')
    throw new BrainchopError('unsupported-option',
      `backend: '${named}' was requested but a WebGL2RenderingContext was ` +
      'supplied; a context selects the webgl2 backend. Drop the glContext, or ' +
      'drop the backend.')
  if (options.device) {
    const need = WEBGPU_MIN_BUFFER_BYTES
    const { maxStorageBufferBindingSize, maxBufferSize } = options.device.limits
    if (maxStorageBufferBindingSize < need || maxBufferSize < need)
      throw new BrainchopError('device-too-small',
        `the supplied device allows ${Math.round(maxStorageBufferBindingSize / 1048576)} MiB ` +
        `per storage binding and ${Math.round(maxBufferSize / 1048576)} MiB per buffer, but ` +
        `WebGPU V4 needs at least ${Math.round(need / 1048576)} MiB. ` +
        `Acquire one with acquireDevice('${options.model}').`)
    return { backend: 'webgpu', device: options.device, ownsContext: false }
  }
  if (options.glContext)
    return { backend: 'webgl2', glContext: options.glContext, ownsContext: false }

  const want = options.backend ?? 'auto'
  if (want === 'webgpu') {
    // Let pipeline.ts manage the cached owned device + Layer 3 safe-mode retry;
    // still run checkSupport() upfront when unavailable so explicit errors
    // ('no-webgpu', 'no-adapter', 'device-too-small') throw identically.
    const probe = await checkSupport(options.model)
    if (!probe.supported) {
      await acquireDevice(options.model) // throws exact BrainchopError code
    }
    return { backend: 'webgpu', ownsContext: false }
  }
  if (want === 'webgl2')
    return { backend: 'webgl2', glContext: acquireGlContext(), ownsContext: true }
  if (want === 'cpu') {
    const cpu = checkCpuSupport()
    if (!cpu.supported)
      throw new BrainchopError('no-webgpu', `the cpu backend cannot run here: ${cpu.reasons.join('; ')}`)
    return { backend: 'cpu', ownsContext: false }
  }

  const gpu = await checkSupport(options.model)
  if (gpu.supported && !gpu.software)
    return { backend: 'webgpu', ownsContext: false }

  const gl = checkWebgl2Support(options.model)
  if (gl.supported && gl.software === false)
    return { backend: 'webgl2', glContext: acquireGlContext(), ownsContext: true }

  const cpu = checkCpuSupport()
  if (cpu.supported) return { backend: 'cpu', ownsContext: false }

  if (gl.supported && !gl.software)
    return { backend: 'webgl2', glContext: acquireGlContext(), ownsContext: true }

  const soft = 'a software rasteriser, too slow and unreliable to run; make ' +
    'the page cross-origin isolated so the cpu backend can'
  throw new BrainchopError('no-webgpu',
    'no backend can run here.\n' +
    `  WebGPU: ${gpu.supported ? soft : gpu.reasons.join('; ')}\n` +
    `  WebGL2: ${gl.supported ? soft : gl.reasons.join('; ')}\n` +
    `  CPU:    ${cpu.reasons.join('; ')}`)
}

/** Whether the threaded CPU module can be instantiated at all. */
export function checkCpuSupport(): { supported: boolean; reasons: string[] } {
  const reasons: string[] = []
  if (typeof SharedArrayBuffer === 'undefined')
    reasons.push('this environment has no SharedArrayBuffer')
  else if (globalThis.crossOriginIsolated !== true)
    reasons.push('this page is not cross-origin isolated; serve it with ' +
      'Cross-Origin-Opener-Policy: same-origin and ' +
      'Cross-Origin-Embedder-Policy: require-corp')
  return { supported: reasons.length === 0, reasons }
}

/**
 * Segment one NIfTI volume.
 *
 * By default runs on hardware WebGPU via the pure-JS V4 engine (zero WASM),
 * falling back to hardware WebGL2 WASM or threaded CPU WASM on browsers
 * without WebGPU.
 *
 * @example
 * const stripped = await segment(await file.arrayBuffer(), { model: 'mindgrab' })
 */
export async function segment(
  input: ArrayBuffer | ArrayBufferView,
  options: SegmentOptions,
): Promise<SegmentResult> {
  return segmentInternal(input, options, false)
}

/** Return GM/WM/CSF/Brain NIfTI fraction maps, without a categorical label volume. */
export async function segmentTissues(
  input: ArrayBuffer | ArrayBufferView,
  options: TissueOptions,
): Promise<TissueResult> {
  const out = await segmentInternal(input, options, true)
  if (!out.tissues) throw new BrainchopError('inference-failed', 'missing tissue outputs')
  return { tissues: out.tissues, elapsedMs: out.elapsedMs, backend: out.backend, ranInWorker: out.ranInWorker }
}

async function segmentInternal(
  input: ArrayBuffer | ArrayBufferView,
  options: SegmentOptions,
  tissueMode: boolean,
): Promise<SegmentResult & { tissues?: TissueMaps }> {
  const raw = input instanceof ArrayBuffer
    ? new Uint8Array(input)
    : new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
  if (raw.byteLength === 0)
    throw new BrainchopError('bad-input', 'the input is empty')

  const { args, maskPath } = buildArgs(options)
  if (tissueMode) {
    const canonical = canonicalModelName(options.model)
    if (canonical !== 'mindmap' || options.mask || options.borderMm || options.legacyCleanup)
      reject(options.model, 'tissue fractions', 'partial-volume tissue fractions are supported only for mindmap without mask or categorical cleanup options')
    args.push('--pve')
  }

  if (options.worker && (options.device || options.glContext))
    throw new BrainchopError('unsupported-option',
      'worker cannot be combined with device or glContext: a GPUDevice and a ' +
      'WebGL2RenderingContext belong to the thread that created them and ' +
      'cannot cross into a worker')

  if (options.worker) {
    const out = await runInWorker(raw, options, tissueMode)
    const result: SegmentResult & { tissues?: TissueMaps } = {
      image: out.image, elapsedMs: out.elapsedMs, backend: out.backend,
      ranInWorker: true,
    }
    if (out.mask) result.mask = out.mask
    if (out.tissues) result.tissues = out.tissues
    return result
  }

  const compressed = isGzip(raw)
  const bytes = compressed ? await gunzip(raw) : raw
  const wantGzip = options.gzipOutput ?? compressed

  const { backend, device, glContext, ownsContext } = await acquire(options)

  const pack = async (data: Uint8Array): Promise<ArrayBuffer> => {
    const out = wantGzip ? await gzip(data) : data
    if (
      out.byteOffset === 0 &&
      out.byteLength === out.buffer.byteLength &&
      (typeof SharedArrayBuffer === 'undefined' || !(out.buffer instanceof SharedArrayBuffer))
    ) {
      return out.buffer as ArrayBuffer
    }
    return out.slice().buffer as ArrayBuffer
  }

  // --- Pure-JS WebGPU V4 path (zero WASM!) ---------------------------------
  if (backend === 'webgpu') {
    const req = {
      model: options.model,
      input: bytes,
      device,
      ct: options.ct,
      comply: options.comply,
      saveConform: options.saveConform,
      legacyCleanup: options.legacyCleanup,
      noTurbo: options.noTurbo,
      mask: options.mask,
      borderMm: options.borderMm,
      assetPath: options.assetPath,
      timeoutMs: options.timeoutMs,
      webgpuMode: options.webgpuMode,
      onLog: options.onLog,
      onProgress: options.onProgress,
    }
    if (tissueMode) {
      const res = await runWebGpuTissues(req)
      const [gm, wm, csf, brain] = await Promise.all([
        pack(res.gm),
        pack(res.wm),
        pack(res.csf),
        pack(res.brain),
      ])
      return {
        image: gm,
        tissues: { gm, wm, csf, brain },
        elapsedMs: res.elapsedMs,
        backend: 'webgpu',
        ranInWorker: false,
      }
    }
    const res = await runWebGpuSegment(req)
    const segmented: SegmentResult & { tissues?: TissueMaps } = {
      image: await pack(res.image),
      elapsedMs: res.elapsedMs,
      backend: 'webgpu',
      ranInWorker: false,
    }
    if (res.mask) segmented.mask = await pack(res.mask)
    return segmented
  }

  // --- WebGL2 / CPU WASM fallback paths ------------------------------------
  options.onProgress?.({ stage: 'load', progress: 0, message: `Loading ${backend} module` })
  let reportedFallbackInference = false
  let result
  try {
    result = await run({
      model: options.model,
      backend,
      glContext,
      input: bytes,
      args: options.noTurbo ? [...args, '--no-turbo'] : args,
      primaryOutput: tissueMode ? '/out_gm.nii' : undefined,
      extraOutputs: tissueMode ? ['/out_wm.nii', '/out_csf.nii', '/out_brain.nii'] : maskPath ? [maskPath] : [],
      assetPath: options.assetPath,
      timeoutMs: options.timeoutMs,
      onLog: (line) => {
        options.onLog?.(line)
        if (!reportedFallbackInference && /using\s+(cpu|webgl2)\s+backend/i.test(line)) {
          reportedFallbackInference = true
          options.onProgress?.({
            stage: 'inference',
            progress: 0.15,
            message: `Running ${options.model} (${backend})`,
          })
        }
      },
    })
  } finally {
    if (ownsContext && glContext) releaseGlContext(glContext)
  }
  options.onProgress?.({ stage: 'complete', progress: 1, message: 'Done' })

  const segmented: SegmentResult & { tissues?: TissueMaps } = {
    image: await pack(result.image),
    elapsedMs: result.elapsedMs,
    backend,
    ranInWorker: false,
  }
  const mask = maskPath ? result.extras.get(maskPath) : undefined
  if (mask) segmented.mask = await pack(mask)
  if (tissueMode) {
    const wm = result.extras.get('/out_wm.nii'), csf = result.extras.get('/out_csf.nii')
    const brain = result.extras.get('/out_brain.nii')
    if (!wm || !csf || !brain) throw new BrainchopError('inference-failed', 'missing WM/CSF/brain output')
    segmented.tissues = { gm: segmented.image, wm: await pack(wm), csf: await pack(csf), brain: await pack(brain) }
  }
  return segmented
}
