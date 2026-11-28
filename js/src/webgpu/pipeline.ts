// Pure-JS WebGPU V4 segmentation and continuous tissue-fraction (PVE) pipeline.
// Runs conform, V4 ZYCX WebGPU inference (from src/models/*/model.bin),
// connected-component cleanup / partial-volume tissue fitting, and native-space
// reslicing entirely in JS + WebGPU with zero WASM binaries.

import { BrainchopError } from '../error.js'
import { acquireDevice, canonicalModelName } from '../device.js'
import { MODEL_BINS } from '../modules.js'
import { assetUrl } from '../run.js'
import type {
  CanonicalModelName,
  ModelName,
  ProgressEvent,
  ProgressStage,
  WebGpuMode,
} from '../types.js'
import {
  readNifti1,
  writeNifti1,
  maskMindGrabVolume,
  isValidAffine4x4,
  NIFTI_DT,
  NIFTI_INTENT_LABEL,
  NIFTI_INTENT_NONE,
} from './nifti.js'
import { sha256Prefix8 } from './sha256.js'
import { conformVolumeData, INFERENCE_CONFORM_FHIGH } from './conform-volume.js'
import { resliceVolumeData, invertAffine4x4, mulAffine4x4 } from './reslice-volume.js'
import { BWLabeler, largestBinaryComponent26, growMaskBorderMm, releaseBwLabelScratch } from './bwlabels.js'
import { validateCategoricalSegmentation } from './segmentation-quality.js'
import {
  computeQuantiles,
  computeLinearQuantiles,
  linearScaleVolume,
} from './tensor-utils.js'
import { applyCatLitePartialVolume } from './cat-lite.js'
import {
  executeWithWebGpuSafeModeRetry,
  formatV4PlanDiagnostic,
} from './gpu-compatibility.js'
import { setupNet, releaseV4Workspace, type V4ExecuteFn } from './meshnet_fast_v4_runner.js'

/** Set by esbuild from js/package.json; falls back when run unbundled in tests. */
declare const __BC_VERSION__: string | undefined
const BC_VERSION = typeof __BC_VERSION__ === 'string' ? __BC_VERSION__ : '1.1.4'

const CONFORM_DIM = 256
const CONFORM_SHAPE: [number, number, number] = [CONFORM_DIM, CONFORM_DIM, CONFORM_DIM]
const CONFORM_NVOX = CONFORM_DIM * CONFORM_DIM * CONFORM_DIM
const DEFAULT_WEBGPU_TIMEOUT_MS = 120_000
const INITIALIZATION_TIMEOUT_MS = 120_000

const TARGET_CLASSES_18 = new Set([1, 2, 5, 6, 13])
const DK104_CORTICAL_FAMILY_OF = (cls: number): number =>
  cls >= 1 && cls <= 34 ? 1 : cls >= 35 && cls <= 68 ? 2 : 0

const PVE_MODEL_OPTIONS = {
  outputType: 'probability',
  softmaxTemperature: 2.0,
  brainSupportTemperature: 1.0,
  brainSupportPower: 1.0,
  partialVolumeSigma: 0.65,
  probabilityGroups: {
    grayMatter: [2, 6, 8, 9, 14, 15, 16],
    whiteMatter: [1, 5, 13],
    mixedGrayWhite: [7, 10, 17],
    csf: [3, 4, 11, 12],
  },
}

const CAT_LITE_OPTIONS = {
  catLitePurePriorPower: 3.0,
  catLitePriorStrength: 0.8,
  catLiteIntensityStrength: 0.75,
  catLiteMixelPrior: 0.35,
  catLiteCsfWmMixelPrior: 6.0,
  catLiteSpatialWeight: 0.25,
  catLiteMinSupport: 0.03,
  catLiteSigmaFloor: 0.025,
  catLiteBiasBlockSize: 16,
  catLiteBiasSmoothPasses: 3,
}

const MODEL_BIN_CACHE_MAX = 4
const rawModelBinCache = new Map<string, Uint8Array>()
const inflightModelBin = new Map<string, Promise<Uint8Array>>()

// Reuse one acquired GPUDevice and pooled V4 workspace per thread across warm
// calls so compiled WGSL pipelines and ~1 GiB activation buffers stay hot.
// Idle timeout, device loss, any error/validation failure, or explicit teardown
// releases every pooled GPU buffer deterministically.
const WORKSPACE_IDLE_MS = 60_000
let sharedOwnedDevice: GPUDevice | null = null
const workspaceIdleTimers = new WeakMap<GPUDevice, ReturnType<typeof setTimeout>>()
const workspaceLostHooked = new WeakSet<GPUDevice>()
const activeWorkspaceDevices = new Set<GPUDevice>()
const sharedBwScratch: Record<string, unknown> = { il: null, tt: null, runs: null, runLen: 0, dirty: false }

function clearWorkspaceTimer(dev: GPUDevice): void {
  const t = workspaceIdleTimers.get(dev)
  if (t !== undefined) {
    clearTimeout(t)
    workspaceIdleTimers.delete(dev)
  }
}

function releaseDeviceWorkspace(dev: GPUDevice | null | undefined): void {
  if (!dev) return
  clearWorkspaceTimer(dev)
  activeWorkspaceDevices.delete(dev)
  try { releaseV4Workspace(dev) } catch { /* ignore */ }
}

function scheduleWorkspaceRelease(dev: GPUDevice): void {
  clearWorkspaceTimer(dev)
  activeWorkspaceDevices.add(dev)
  if (!workspaceLostHooked.has(dev) && dev.lost && typeof dev.lost.then === 'function') {
    workspaceLostHooked.add(dev)
    dev.lost.then(() => {
      releaseDeviceWorkspace(dev)
      if (sharedOwnedDevice === dev) sharedOwnedDevice = null
    }).catch(() => {})
  }
  const timer = setTimeout(() => {
    releaseDeviceWorkspace(dev)
  }, WORKSPACE_IDLE_MS)
  if (typeof (timer as unknown as { unref?: () => void }).unref === 'function') {
    ;(timer as unknown as { unref: () => void }).unref()
  }
  workspaceIdleTimers.set(dev, timer)
}

/** Release pooled GPU workspaces, owned GPUDevice, and reusable CPU scratch buffers. */
export function disposeWebGpuResources(): void {
  for (const dev of Array.from(activeWorkspaceDevices)) {
    releaseDeviceWorkspace(dev)
  }
  if (sharedOwnedDevice) {
    resetOwnedDevice(sharedOwnedDevice)
  }
  sharedBwScratch.il = null
  sharedBwScratch.tt = null
  sharedBwScratch.runs = null
  sharedBwScratch.runLen = 0
  sharedBwScratch.dirty = false
  releaseBwLabelScratch()
}

async function getOwnedDevice(model: ModelName): Promise<GPUDevice> {
  if (sharedOwnedDevice && !(sharedOwnedDevice as GPUDevice & { __isLost?: boolean }).__isLost) {
    return sharedOwnedDevice
  }
  sharedOwnedDevice = null
  const dev = await acquireDevice(model)
  sharedOwnedDevice = dev
  return dev
}

function resetOwnedDevice(failedDevice: GPUDevice | null): void {
  if (failedDevice) {
    releaseDeviceWorkspace(failedDevice)
    if (!(failedDevice as GPUDevice & { __isLost?: boolean }).__isLost) {
      try { failedDevice.destroy() } catch { /* ignore */ }
    }
    if (sharedOwnedDevice === failedDevice) sharedOwnedDevice = null
  }
}

function resolveModelBinUrl(model: ModelName, assetPath?: string): string {
  const canonical = canonicalModelName(model)
  if (assetPath) return assetUrl(`models/${canonical}.bin`, assetPath)
  const resolver = MODEL_BINS[model] ?? MODEL_BINS[canonical]
  if (!resolver) {
    throw new BrainchopError(
      'unsupported-option',
      `${model} is not in this entry point; import '@brainchop/core' or '@brainchop/core/${model}'`,
    )
  }
  return resolver()
}

async function fetchModelBinBytes(
  model: ModelName,
  assetPath: string | undefined,
  timeoutMs: number,
  log: string[],
): Promise<Uint8Array> {
  const url = resolveModelBinUrl(model, assetPath)
  const cached = rawModelBinCache.get(url)
  if (cached) {
    rawModelBinCache.delete(url)
    rawModelBinCache.set(url, cached)
    return cached
  }

  let pending = inflightModelBin.get(url)
  if (!pending) {
    pending = (async () => {
      const response = await fetch(url)
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }
      const bytes = new Uint8Array(await response.arrayBuffer())
      rawModelBinCache.set(url, bytes)
      while (rawModelBinCache.size > MODEL_BIN_CACHE_MAX) {
        const oldest = rawModelBinCache.keys().next().value
        if (oldest !== undefined) rawModelBinCache.delete(oldest)
      }
      return bytes
    })().finally(() => {
      inflightModelBin.delete(url)
    })
    inflightModelBin.set(url, pending)
  }

  let timer: ReturnType<typeof setTimeout> | undefined
  try {
    return await Promise.race([
      pending,
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => {
          reject(new BrainchopError(
            'initialization-failed',
            `fetching weights for ${model} (${url}) timed out after ${timeoutMs} ms`,
            log,
          ))
        }, timeoutMs)
      }),
    ])
  } catch (err) {
    if (err instanceof BrainchopError) throw err
    throw new BrainchopError(
      'initialization-failed',
      `failed to load ${model} weights from ${url}: ${err instanceof Error ? err.message : String(err)}`,
      log,
    )
  } finally {
    clearTimeout(timer)
  }
}

function flatToNested4x4(m: number[]): number[][] {
  return [
    [m[0], m[1], m[2], m[3]],
    [m[4], m[5], m[6], m[7]],
    [m[8], m[9], m[10], m[11]],
    [0, 0, 0, 1],
  ]
}

function isSameGrid(
  srcDims: [number, number, number],
  srcAffine: number[],
  dstDims: [number, number, number],
  dstAffine: number[],
): boolean {
  if (srcDims[0] !== dstDims[0] || srcDims[1] !== dstDims[1] || srcDims[2] !== dstDims[2]) {
    return false
  }
  const srcInv = invertAffine4x4(srcAffine)
  if (!srcInv) return false
  const dstToSrc = mulAffine4x4(srcInv, dstAffine)
  for (let r = 0; r < 3; r++) {
    const r4 = r * 4
    let shift = Math.abs(dstToSrc[r4 + 3])
    for (let c = 0; c < 3; c++) {
      shift += Math.abs(dstToSrc[r4 + c] - (r === c ? 1 : 0)) * (dstDims[c] - 1)
    }
    if (shift >= 1e-3) return false
  }
  return true
}

function buildV4ModelEntry(
  canonical: CanonicalModelName,
  webgpuMode: WebGpuMode | undefined,
  tissueMode: boolean,
): Record<string, unknown> {
  const entry: Record<string, unknown> = {
    model: canonical,
    modelBinModel: canonical,
    quantize8Bit: true,
    enableTranspose: true,
    ...(tissueMode ? PVE_MODEL_OPTIONS : {}),
  }

  if (webgpuMode === 'low-memory') {
    entry.forceLowMem128 = true
    entry.webgpuExecMode = 'paced'
    entry.__v4PlanReasonOverride = 'override'
  } else if (webgpuMode === 'standard') {
    entry.forceLowMem128 = false
    entry.__v4PlanReasonOverride = 'override'
  } else if (webgpuMode === 'recalibrate') {
    entry.recalibrate = true
  }

  return entry
}

function createProgressReporter(onProgress?: (event: ProgressEvent) => void) {
  let lastProgress = 0
  return (
    stage: ProgressStage,
    rawProgress: number,
    extra?: { layer?: number; totalLayers?: number; message?: string },
  ) => {
    if (!onProgress) return
    const clamped = Math.max(0, Math.min(1, Number.isFinite(rawProgress) ? rawProgress : 0))
    const progress = Math.max(lastProgress, clamped)
    lastProgress = progress
    const ev: ProgressEvent = { stage, progress }
    if (extra?.layer !== undefined) ev.layer = extra.layer
    if (extra?.totalLayers !== undefined) ev.totalLayers = extra.totalLayers
    if (extra?.message !== undefined) ev.message = extra.message
    onProgress(ev)
  }
}

export interface WebGpuSegmentRequest {
  model: ModelName
  /** Uncompressed NIfTI-1 bytes. */
  input: Uint8Array
  /** Caller-supplied GPUDevice, or undefined to acquire/reuse one automatically. */
  device?: GPUDevice
  ct?: boolean
  comply?: boolean
  saveConform?: boolean
  legacyCleanup?: boolean
  noTurbo?: boolean
  mask?: boolean
  borderMm?: number
  assetPath?: string
  timeoutMs?: number
  webgpuMode?: WebGpuMode
  onLog?: (line: string) => void
  onProgress?: (event: ProgressEvent) => void
}

export interface WebGpuSegmentResult {
  image: Uint8Array
  mask?: Uint8Array
  elapsedMs: number
}

export interface WebGpuTissuesResult {
  gm: Uint8Array
  wm: Uint8Array
  csf: Uint8Array
  brain: Uint8Array
  elapsedMs: number
}

async function runV4Inference(params: {
  canonical: CanonicalModelName
  model: ModelName
  weightBytes: Uint8Array
  conformedImg: Uint8Array
  normOffset: number
  normRange: number
  normClamp: boolean
  tissueMode: boolean
  device?: GPUDevice
  webgpuMode?: WebGpuMode
  timeoutMs: number
  log: string[]
  record: (line: string) => void
  reportProgress: ReturnType<typeof createProgressReporter>
}): Promise<{ outputs: Array<Uint8Array | Float32Array>; device: GPUDevice }> {
  const {
    canonical,
    model,
    weightBytes,
    conformedImg,
    normOffset,
    normRange,
    normClamp,
    tissueMode,
    device: callerDevice,
    webgpuMode,
    timeoutMs,
    log,
    record,
    reportProgress,
  } = params

  const ownsDevice = !callerDevice
  const initialDevice = callerDevice ?? await getOwnedDevice(model)
  const modelEntry = buildV4ModelEntry(canonical, webgpuMode, tissueMode)
  const totalLayers = canonical === 'mindgrab' ? 25 : 13

  const runAttempt = async (dev: GPUDevice, entry: Record<string, unknown>) => {
    clearWorkspaceTimer(dev)
    const hasScopes =
      typeof dev.pushErrorScope === 'function' && typeof dev.popErrorScope === 'function'
    let oomScopeOpen = false
    let execute: V4ExecuteFn | null = null
    let timer: ReturnType<typeof setTimeout> | undefined
    let succeeded = false

    if (hasScopes) {
      dev.pushErrorScope('out-of-memory')
      oomScopeOpen = true
    }

    try {
      const work = (async () => {
        reportProgress('load', 0.12, { message: `Setting up WebGPU V4 for ${canonical}` })
        execute = await setupNet(dev, weightBytes, undefined, entry)
        const plan = typeof execute.__v4Plan === 'function' ? execute.__v4Plan() : null
        if (plan) record(formatV4PlanDiagnostic(plan))

        reportProgress('inference', 0.30, {
          layer: 0,
          totalLayers,
          message: tissueMode ? 'Estimating tissue priors' : `Running ${canonical} inference`,
        })

        const outputs = await execute(conformedImg, {
          gpuNativeInput: true,
          gpuNativeOutput: true,
          normOffset,
          normRange,
          normClamp,
          onProgress: (frac: number, stage: string) => {
            const clamped = Math.max(0, Math.min(1, Number(frac) || 0))
            const m = typeof stage === 'string' ? /^Layer (\d+)\/(\d+)$/.exec(stage) : null
            reportProgress('inference', 0.30 + 0.55 * clamped, {
              layer: m ? Number(m[1]) : undefined,
              totalLayers: m ? Number(m[2]) : totalLayers,
              message: stage || undefined,
            })
          },
        })

        reportProgress('inference', 0.85, {
          layer: totalLayers,
          totalLayers,
          message: 'Inference complete',
        })
        return outputs
      })()

      const outputs = await Promise.race([
        work,
        new Promise<never>((_, reject) => {
          timer = setTimeout(() => {
            reject(new BrainchopError(
              'inference-failed',
              `the ${model} WebGPU inference did not finish within ${timeoutMs} ms; the GPU device may have been lost`,
              log,
            ))
          }, timeoutMs)
        }),
      ])

      if (oomScopeOpen) {
        oomScopeOpen = false
        const oomError = await dev.popErrorScope()
        if (oomError) {
          const err = new Error(
            `WebGPU out of memory (${oomError.message || 'allocation failed'})`,
          ) as Error & { oomError?: unknown }
          err.oomError = oomError
          throw err
        }
      }

      succeeded = true
      return { outputs, device: dev }
    } catch (error) {
      if (oomScopeOpen) {
        oomScopeOpen = false
        try {
          const scopedOom = await dev.popErrorScope()
          if (scopedOom && error && typeof error === 'object' && !('oomError' in error)) {
            ;(error as { oomError?: unknown }).oomError = scopedOom
          }
        } catch { /* device may already be lost */ }
      }
      throw error
    } finally {
      clearTimeout(timer)
      const exec = execute as V4ExecuteFn | null
      if (exec) {
        try {
          exec.dispose({ keepSharedWorkspace: succeeded })
        } catch { /* ignore */ }
      }
      if (succeeded) {
        scheduleWorkspaceRelease(dev)
      } else {
        releaseDeviceWorkspace(dev)
      }
    }
  }

  try {
    return await executeWithWebGpuSafeModeRetry({
      device: initialDevice,
      modelEntry,
      runAttempt: (dev, entry) => runAttempt(dev, entry),
      requestFreshDevice: ownsDevice
        ? async ({ failedDevice }) => {
            resetOwnedDevice(failedDevice)
            try {
              return await getOwnedDevice(model)
            } catch {
              return null
            }
          }
        : null,
      onRetry: ({ failureKind, error }) => {
        const msg = `[WebGPU-V4] ${failureKind} during inference (${error instanceof Error ? error.message : String(error)}); retrying once in LowMem128 + paced`
        record(msg)
        reportProgress('load', 0.14, { message: 'Retrying in low-memory WebGPU mode' })
      },
    })
  } catch (err) {
    if (err instanceof BrainchopError) throw err
    const msg = err instanceof Error ? err.message : String(err)
    record(`brainchop-core: webgpu inference error: ${msg}`)
    throw new BrainchopError('inference-failed', msg, log)
  }
}

function quantizeFractionToU8(values: Float32Array): Uint8Array {
  const out = new Uint8Array(values.length)
  for (let i = 0; i < values.length; i++) {
    let v = values[i]
    if (!(v > 0)) v = 0
    else if (v > 1) v = 1
    out[i] = Math.floor(v * 255 + 0.5)
  }
  return out
}

function validateNonZeroFloat32(values: Float32Array, name: string, log: string[], record: (line: string) => void): void {
  let sum = 0
  for (let i = 0; i < values.length; i++) {
    const v = values[i]
    if (!Number.isFinite(v)) {
      const msg = `brainchop-core: ${name} tissue prior contains non-finite value at voxel ${i}`
      record(msg)
      throw new BrainchopError('inference-failed', msg, log)
    }
    sum += v
  }
  if (!(sum > 0)) {
    const msg = `brainchop-core: ${name} tissue prior is empty (all zeros)`
    record(msg)
    throw new BrainchopError('inference-failed', msg, log)
  }
}

/**
 * Execute full end-to-end categorical / skull-stripping segmentation on pure-JS WebGPU V4.
 */
export async function runWebGpuSegment(request: WebGpuSegmentRequest): Promise<WebGpuSegmentResult> {
  const started = performance.now()
  const log: string[] = []
  const record = (line: string) => {
    log.push(line)
    request.onLog?.(line)
  }
  const reportProgress = createProgressReporter(request.onProgress)
  const canonical = canonicalModelName(request.model)
  const limit = request.timeoutMs ?? DEFAULT_WEBGPU_TIMEOUT_MS
  const initLimit = Math.min(limit, INITIALIZATION_TIMEOUT_MS)

  reportProgress('load', 0.0, { message: `Loading ${canonical} weights` })
  if (request.noTurbo) {
    record(`[WebGPU-V4] running pure-JS V4 engine for ${canonical}`)
  }

  // Kick off weight fetch while parsing & conforming the input volume.
  const weightsPromise = fetchModelBinBytes(request.model, request.assetPath, initLimit, log)

  // 1. Parse NIfTI-1 input
  let orig
  try {
    orig = readNifti1(request.input)
  } catch (err) {
    await weightsPromise.catch(() => {})
    throw new BrainchopError(
      'bad-input',
      err instanceof Error ? err.message : String(err),
      log,
    )
  }

  // 2. Conform to 256^3 1mm LIA uint8 (fHigh = 0.98, optional CT Cormack & comply;
  //    constant / all-NaN rejection fused into the single decode pass)
  reportProgress('conform', 0.15, { message: 'Conforming volume to 256^3' })
  let conformed
  try {
    conformed = conformVolumeData({
      img: orig.data,
      datatypeCode: orig.header.datatypeCode,
      dims: orig.header.dims,
      affine: orig.header.affine,
      sclSlope: orig.header.sclSlope,
      sclInter: orig.header.sclInter,
      toRAS: false,
      isLinear: true,
      asFloat32: false,
      fHigh: INFERENCE_CONFORM_FHIGH,
      ct: Boolean(request.ct),
      comply: Boolean(request.comply),
      rejectConstant: true,
    })
  } catch (err) {
    await weightsPromise.catch(() => {})
    throw new BrainchopError(
      'bad-input',
      err instanceof Error ? err.message : String(err),
      log,
    )
  }
  const conformedImg = conformed.img as Uint8Array
  reportProgress('conform', 0.25)

  // 3. Normalization parameters (matching src/models/*/model_def.h)
  reportProgress('preprocess', 0.25, { message: 'Computing intensity quantiles' })
  let normOffset: number
  let normRange: number
  let normClamp: boolean
  if (canonical === 'mindgrab') {
    const { qmin, qmax } = computeLinearQuantiles(conformedImg, 0.02, 0.98)
    normOffset = qmin
    const denom = qmax - qmin + 0.001
    normRange = denom > 0 ? denom : 1
    normClamp = true
  } else {
    const { qmin, qmax } = computeQuantiles(conformedImg, 0.05, 0.95)
    normOffset = qmin
    const denom = qmax - qmin
    normRange = denom > 0 ? denom : 1
    normClamp = false
  }
  reportProgress('preprocess', 0.30)

  // 4. Fetch weights & run V4 WebGPU inference
  const weightBytes = await weightsPromise
  const segDescrip = `brainchop ${BC_VERSION} ${canonical}/${await sha256Prefix8(weightBytes)} webgpu`
  const { outputs, device: usedDevice } = await runV4Inference({
    canonical,
    model: request.model,
    weightBytes,
    conformedImg,
    normOffset,
    normRange,
    normClamp,
    tissueMode: false,
    device: request.device,
    webgpuMode: request.webgpuMode,
    timeoutMs: limit,
    log,
    record,
    reportProgress,
  })

  try {
    const rawOut = outputs[0]
    if (!rawOut || rawOut.length !== CONFORM_NVOX) {
      const msg = `brainchop-core: unexpected WebGPU output length (${rawOut?.length ?? 0})`
      record(msg)
      throw new BrainchopError('inference-failed', msg, log)
    }
    const rawLabels = rawOut instanceof Uint8Array ? rawOut : new Uint8Array(rawOut)

    // 5. Quality validation & connected-component postprocessing
    reportProgress('postprocess', 0.85, { message: 'Post-processing segmentation' })
    const origDims: [number, number, number] = [
      orig.header.dims[1],
      orig.header.dims[2],
      orig.header.dims[3],
    ]

    if (canonical === 'mindgrab') {
      let hasFg = false
      for (let i = 0; i < CONFORM_NVOX; i++) {
        if (rawLabels[i] > 0) { hasFg = true; break }
      }
      if (!hasFg) {
        const msg = 'mindgrab: the network found no brain; check the input contrast'
        record(msg)
        throw new BrainchopError('inference-failed', msg, log)
      }

      // 26-connected binary LCC (matches src/brainchopc.c:628)
      largestBinaryComponent26(rawLabels, CONFORM_SHAPE, true)

      // Optional Euclidean border dilation in mm (matches src/brainchopc.c:633-639)
      if (request.borderMm && request.borderMm > 0) {
        growMaskBorderMm(rawLabels, CONFORM_SHAPE, 1, 1, 1, request.borderMm)
      }
      reportProgress('reslice', 0.93, { message: 'Applying brain mask in native space' })

      if (!isValidAffine4x4(orig.header.stoXyz)) {
        const msg = 'brainchop-core: cannot reslice to the input grid; its sform is not a usable spatial transform'
        record(msg)
        throw new BrainchopError('bad-input', msg, log)
      }

      let nativeMask: Uint8Array
      if (isSameGrid(CONFORM_SHAPE, conformed.affine, origDims, orig.header.stoXyz)) {
        nativeMask = rawLabels
      } else {
        const resliced = resliceVolumeData({
          srcImg: rawLabels,
          srcDims: CONFORM_SHAPE,
          srcAffine: flatToNested4x4(conformed.affine),
          dstDims: origDims,
          dstAffine: flatToNested4x4(orig.header.stoXyz),
          isLabel: false,
          isProbability: false,
          dstTypedArrayKind: 'Uint8Array',
        })
        nativeMask = resliced.img as Uint8Array
      }

      // Optional binary mask NIfTI (matches write_mask in src/brainchopc.c:346-370)
      let maskBytes: Uint8Array | undefined
      if (request.mask || request.borderMm !== undefined) {
        const binMask = new Uint8Array(nativeMask.length)
        for (let i = 0; i < nativeMask.length; i++) binMask[i] = nativeMask[i] > 0 ? 1 : 0
        maskBytes = writeNifti1(orig.header, binMask, {
          datatypeCode: NIFTI_DT.UINT8,
          sclSlope: 1,
          sclInter: 0,
          calMin: 0,
          calMax: 0,
          intentCode: NIFTI_INTENT_LABEL,
          descrip: segDescrip,
        })
      }

      // Apply mask to native input volume, preserving native integer encoding and
      // NIfTI scaling (matches mask_integer_input / float path in src/brainchopc.c).
      const masked = maskMindGrabVolume(
        orig.data,
        orig.header.datatypeCode,
        orig.header.sclSlope,
        orig.header.sclInter,
        nativeMask,
      )
      const imageBytes = writeNifti1(orig.header, masked.data, {
        datatypeCode: orig.header.datatypeCode,
        sclSlope: masked.sclSlope,
        sclInter: masked.sclInter,
        descrip: segDescrip,
      })

      const elapsedMs = performance.now() - started
      reportProgress('complete', 1.0, { message: 'Done' })
      return { image: imageBytes, mask: maskBytes, elapsedMs }
    }

    // Categorical label models (mindmap-fast / 16chan18cls, mindmap, mindsnap)
    let foreground = 0
    for (let i = 0; i < CONFORM_NVOX; i++) {
      if (rawLabels[i] !== 0) foreground++
    }
    if (foreground === 0) {
      const msg = `brainchop-core: categorical foreground is 0/${CONFORM_NVOX} voxels; segmentation is empty`
      record(msg)
      throw new BrainchopError('inference-failed', msg, log)
    }
    try {
      validateCategoricalSegmentation(rawLabels)
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err)
      record(`brainchop-core: ${msg}`)
      throw new BrainchopError('inference-failed', msg, log)
    }

    let cleanedLabels: Uint8Array
    if (request.legacyCleanup) {
      // Still check noise fragmentation cap before legacy cleanup
      const bw = new BWLabeler()
      const [guardCount] = bw.bwlabel(rawLabels, CONFORM_SHAPE, 6, false, false, sharedBwScratch)
      const noiseCap = Math.max(100_000, Math.floor(CONFORM_NVOX * 0.01))
      if (guardCount > noiseCap) {
        const msg = `brainchop-core: categorical segmentation is excessively fragmented (${guardCount} six-connected regions, cap ${noiseCap})`
        record(msg)
        throw new BrainchopError('inference-failed', msg, log)
      }
      cleanedLabels = largestBinaryComponent26(rawLabels, CONFORM_SHAPE, false)
    } else {
      const bw = new BWLabeler()
      const [guardCount, guardLabels] = bw.bwlabel(rawLabels, CONFORM_SHAPE, 6, false, false, sharedBwScratch)
      const noiseCap = Math.max(100_000, Math.floor(CONFORM_NVOX * 0.01))
      if (guardCount > noiseCap) {
        const msg = `brainchop-core: categorical segmentation is excessively fragmented (${guardCount} six-connected regions, cap ${noiseCap})`
        record(msg)
        throw new BrainchopError('inference-failed', msg, log)
      }
      if (canonical === 'mindsnap') {
        const [, res] = bw.filterHierarchicalFromLabels(
          rawLabels,
          guardCount,
          guardLabels,
          CONFORM_SHAPE,
          'all',
          true,
          true,
          DK104_CORTICAL_FAMILY_OF,
          true,
        )
        cleanedLabels = res instanceof Uint8Array ? res : new Uint8Array(res)
      } else {
        const [, res] = bw.filter18ClassFromLabels(
          rawLabels,
          guardCount,
          guardLabels,
          CONFORM_SHAPE,
          TARGET_CLASSES_18,
          true,
          null,
          true,
        )
        cleanedLabels = res instanceof Uint8Array ? res : new Uint8Array(res)
      }
    }

    // 6. Reslice to native grid (default) or emit conformed 256^3 grid (--save-conform)
    reportProgress('reslice', 0.93, {
      message: request.saveConform ? 'Encoding conformed NIfTI' : 'Reslicing labels to native space',
    })

    let imageBytes: Uint8Array
    if (request.saveConform) {
      imageBytes = writeNifti1(orig.header, cleanedLabels, {
        dims: CONFORM_SHAPE,
        pixDims: [1, 1, 1],
        affine: conformed.affine,
        datatypeCode: NIFTI_DT.UINT8,
        sclSlope: 1,
        sclInter: 0,
        calMin: 0,
        calMax: 0,
        intentCode: NIFTI_INTENT_LABEL,
        descrip: segDescrip,
      })
    } else {
      if (!isValidAffine4x4(orig.header.stoXyz)) {
        const msg = 'brainchop-core: cannot reslice to the input grid; its sform is not a usable spatial transform'
        record(msg)
        throw new BrainchopError('bad-input', msg, log)
      }
      let nativeLabels: Uint8Array
      if (isSameGrid(CONFORM_SHAPE, conformed.affine, origDims, orig.header.stoXyz)) {
        nativeLabels = cleanedLabels
      } else {
        const resliced = resliceVolumeData({
          srcImg: cleanedLabels,
          srcDims: CONFORM_SHAPE,
          srcAffine: flatToNested4x4(conformed.affine),
          dstDims: origDims,
          dstAffine: flatToNested4x4(orig.header.stoXyz),
          isLabel: true,
          labelOutputType: 'uint8',
        })
        nativeLabels = resliced.img as Uint8Array
      }
      imageBytes = writeNifti1(orig.header, nativeLabels, {
        dims: origDims,
        affine: orig.header.stoXyz,
        preservePixDims: [orig.header.pixDims[1], orig.header.pixDims[2], orig.header.pixDims[3]],
        datatypeCode: NIFTI_DT.UINT8,
        sclSlope: 1,
        sclInter: 0,
        calMin: 0,
        calMax: 0,
        intentCode: NIFTI_INTENT_LABEL,
        descrip: segDescrip,
      })
    }

    const elapsedMs = performance.now() - started
    reportProgress('complete', 1.0, { message: 'Done' })
    return { image: imageBytes, elapsedMs }
  } catch (err) {
    // Release pooled GPU workspace on any post-inference failure so callers
    // holding a custom GPUDevice observe zero leaked live buffers.
    releaseDeviceWorkspace(usedDevice)
    throw err
  }
}

/**
 * Execute continuous GM/WM/CSF/Brain partial-volume tissue fraction estimation
 * on pure-JS WebGPU V4.
 */
export async function runWebGpuTissues(request: WebGpuSegmentRequest): Promise<WebGpuTissuesResult> {
  const started = performance.now()
  const log: string[] = []
  const record = (line: string) => {
    log.push(line)
    request.onLog?.(line)
  }
  const reportProgress = createProgressReporter(request.onProgress)
  const canonical = canonicalModelName(request.model)
  const limit = request.timeoutMs ?? DEFAULT_WEBGPU_TIMEOUT_MS
  const initLimit = Math.min(limit, INITIALIZATION_TIMEOUT_MS)

  reportProgress('load', 0.0, { message: `Loading ${canonical} weights` })
  const weightsPromise = fetchModelBinBytes(request.model, request.assetPath, initLimit, log)

  // 1. Parse NIfTI-1 input
  let orig
  try {
    orig = readNifti1(request.input)
  } catch (err) {
    await weightsPromise.catch(() => {})
    throw new BrainchopError(
      'bad-input',
      err instanceof Error ? err.message : String(err),
      log,
    )
  }

  // 2. Conform to 256^3 1mm LIA uint8 (fHigh = 0.98; constant rejection fused)
  reportProgress('conform', 0.15, { message: 'Conforming volume to 256^3' })
  let conformed
  try {
    conformed = conformVolumeData({
      img: orig.data,
      datatypeCode: orig.header.datatypeCode,
      dims: orig.header.dims,
      affine: orig.header.affine,
      sclSlope: orig.header.sclSlope,
      sclInter: orig.header.sclInter,
      toRAS: false,
      isLinear: true,
      asFloat32: false,
      fHigh: INFERENCE_CONFORM_FHIGH,
      ct: Boolean(request.ct),
      comply: Boolean(request.comply),
      rejectConstant: true,
    })
  } catch (err) {
    await weightsPromise.catch(() => {})
    throw new BrainchopError(
      'bad-input',
      err instanceof Error ? err.message : String(err),
      log,
    )
  }
  const conformedImg = conformed.img as Uint8Array
  reportProgress('conform', 0.25)

  // 3. Quantile normalization (0.05..0.95) + intensity copy for PVE fit
  reportProgress('preprocess', 0.25, { message: 'Computing intensity quantiles' })
  const { qmin, qmax } = computeQuantiles(conformedImg, 0.05, 0.95)
  const normOffset = qmin
  const normRange = qmax - qmin > 0 ? qmax - qmin : 1
  const catLiteIntensity = linearScaleVolume(conformedImg, normOffset, normRange)
  reportProgress('preprocess', 0.30)

  // 4. Fetch weights & run V4 WebGPU grouped tissue-prior inference
  const weightBytes = await weightsPromise
  const pveDescrip = `brainchop ${BC_VERSION} ${canonical}/${await sha256Prefix8(weightBytes)} webgpu pve`
  const { outputs, device: usedDevice } = await runV4Inference({
    canonical,
    model: request.model,
    weightBytes,
    conformedImg,
    normOffset,
    normRange,
    normClamp: false,
    tissueMode: true,
    device: request.device,
    webgpuMode: request.webgpuMode,
    timeoutMs: limit,
    log,
    record,
    reportProgress,
  })

  try {
    if (outputs.length !== 3) {
      const msg = `brainchop-core: PVE expected 3 tissue priors, got ${outputs.length}`
      record(msg)
      throw new BrainchopError('inference-failed', msg, log)
    }

    const priors = outputs.map((arr) =>
      arr instanceof Float32Array ? arr : Float32Array.from(arr),
    )
    validateNonZeroFloat32(priors[0], 'GM', log, record)
    validateNonZeroFloat32(priors[1], 'WM', log, record)
    validateNonZeroFloat32(priors[2], 'CSF', log, record)

    // 5. Partial-volume tissue fitting
    reportProgress('postprocess', 0.85, { message: 'Fitting partial-volume tissues' })
    const { tissues, stats } = applyCatLitePartialVolume(
      priors,
      catLiteIntensity,
      CONFORM_SHAPE,
      CAT_LITE_OPTIONS,
    )
    if (stats.applied && stats.tissueMeans && stats.tissueSigmas) {
      record(
        `[PVE] means CSF=${stats.tissueMeans.csf.toFixed(3)} GM=${stats.tissueMeans.gray.toFixed(3)} WM=${stats.tissueMeans.white.toFixed(3)}; ` +
        `sigmas CSF=${stats.tissueSigmas.csf.toFixed(3)} GM=${stats.tissueSigmas.gray.toFixed(3)} WM=${stats.tissueSigmas.white.toFixed(3)}`,
      )
    } else if (!stats.applied) {
      record(`[PVE] skipped: ${stats.reason ?? 'unknown'}`)
    }

    const [gmMap, wmMap, csfMap] = tissues
    const brainMap = new Float32Array(CONFORM_NVOX)
    for (let i = 0; i < CONFORM_NVOX; i++) {
      brainMap[i] = gmMap[i] + wmMap[i]
    }

    // 6. Encode (and optionally reslice trilinearly to native space)
    reportProgress('reslice', 0.93, {
      message: request.saveConform ? 'Encoding conformed tissue maps' : 'Reslicing tissue maps to native space',
    })

    const origDims: [number, number, number] = [
      orig.header.dims[1],
      orig.header.dims[2],
      orig.header.dims[3],
    ]
    if (!request.saveConform && !isValidAffine4x4(orig.header.stoXyz)) {
      const msg = 'brainchop-core: cannot reslice to the input grid; its sform is not a usable spatial transform'
      record(msg)
      throw new BrainchopError('bad-input', msg, log)
    }
    const sameGrid = !request.saveConform &&
      isSameGrid(CONFORM_SHAPE, conformed.affine, origDims, orig.header.stoXyz)

    const encodeTissue = (map: Float32Array): Uint8Array => {
      if (request.saveConform) {
        return writeNifti1(orig.header, quantizeFractionToU8(map), {
          dims: CONFORM_SHAPE,
          pixDims: [1, 1, 1],
          affine: conformed.affine,
          datatypeCode: NIFTI_DT.UINT8,
          sclSlope: 1 / 255,
          sclInter: 0,
          calMin: 0,
          calMax: 1,
          intentCode: NIFTI_INTENT_NONE,
          descrip: pveDescrip,
        })
      }
      const quantU8 = sameGrid
        ? quantizeFractionToU8(map)
        : (resliceVolumeData({
            srcImg: map,
            srcDims: CONFORM_SHAPE,
            srcAffine: flatToNested4x4(conformed.affine),
            dstDims: origDims,
            dstAffine: flatToNested4x4(orig.header.stoXyz),
            isLabel: false,
            isProbability: true,
          }).img as Uint8Array)
      return writeNifti1(orig.header, quantU8, {
        dims: origDims,
        affine: orig.header.stoXyz,
        preservePixDims: [orig.header.pixDims[1], orig.header.pixDims[2], orig.header.pixDims[3]],
        datatypeCode: NIFTI_DT.UINT8,
        sclSlope: 1 / 255,
        sclInter: 0,
        calMin: 0,
        calMax: 1,
        intentCode: NIFTI_INTENT_NONE,
        descrip: pveDescrip,
      })
    }

    const gm = encodeTissue(gmMap)
    const wm = encodeTissue(wmMap)
    const csf = encodeTissue(csfMap)
    const brain = encodeTissue(brainMap)

    const elapsedMs = performance.now() - started
    reportProgress('complete', 1.0, { message: 'Done' })
    return { gm, wm, csf, brain, elapsedMs }
  } catch (err) {
    releaseDeviceWorkspace(usedDevice)
    throw err
  }
}
