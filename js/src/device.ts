import { BrainchopError } from './error.js'
import type { CanonicalModelName, ModelName } from './types.js'

/*
 * WebGPU device acquisition and capability negotiation for the pure-JS V4 engine.
 *
 * With V4 ZYCX split-plane buffers + LowMem128 + Tier 3 f32 fallback:
 *   - `shader-f16` is OPTIONAL (Tier 1/2 use f16+Sterbenz, Tier 3 falls back to f32).
 *   - `subgroups` is OPTIONAL (Tier 1 uses sg32/sg64/sg32_64/sg_small, Tier 2 uses shared memory).
 *   - Minimum `maxStorageBufferBindingSize` / `maxBufferSize` is 128 MiB because
 *     LowMem128 keeps the largest single buffer <= 80 MiB (15/16ch) or <= 120 MiB (24ch).
 */

/** Normalize legacy model aliases (`'16chan18cls'` -> `'mindmap-fast'`). */
export function canonicalModelName(model: ModelName): CanonicalModelName {
  return model === '16chan18cls' ? 'mindmap-fast' : model
}

const MODEL_CHANNELS: Record<ModelName, number> = {
  mindgrab: 16,
  'mindmap-fast': 16,
  '16chan18cls': 16,
  mindmap: 24,
  'mindmap-omni': 24,
  mindsnap: 24,
}

/** Minimum WebGPU buffer / storage binding size required by V4 LowMem128 (128 MiB). */
export const WEBGPU_MIN_BUFFER_BYTES = 128 * 1024 * 1024

/** Largest single activation buffer used by V4 standard layout before LowMem128 kicks in. */
export function v4StdBufferBytes(model: ModelName = 'mindmap-fast'): number {
  const cs = MODEL_CHANNELS[model] ?? 16
  // DIM * DIM * (cs / 2) * X4_PAD(80) * 4 bytes: 160 MiB for 16ch, 240 MiB for 24ch.
  return 256 * 256 * (cs / 2) * 80 * 4
}

/** 256^3 voxels x the model's channel stride x 2 bytes: one f16 activation (used by WebGL2). */
export function activationBytes(model: ModelName): number {
  return 256 * 256 * 256 * MODEL_CHANNELS[model] * 2
}

/**
 * One f16 activation for a 16-channel model.
 *
 * @deprecated a per-model number; use `activationBytes(model)` or `WEBGPU_MIN_BUFFER_BYTES`.
 */
export const ACTIVATION_BYTES = activationBytes('16chan18cls')

const MiB = (n: number) => `${Math.round(n / (1024 * 1024))} MiB`

/*
 * CPU rasterisers behind a GPU API. They pass every capability probe, then run
 * ~100x slower than the CPU module, so `auto` never picks them (neuroneural/brainchop#78).
 */
export const SOFTWARE_RENDERER = /SwiftShader|llvmpipe|lavapipe|softpipe|Software|Basic Render/i

export interface DeviceReport {
  supported: boolean
  /** Why not, in the caller's terms. Empty when supported. */
  reasons: string[]
  maxBufferSize?: number
  maxStorageBufferBindingSize?: number
  hasShaderF16?: boolean
  hasSubgroups?: boolean
  /** True when the adapter's buffer limit requires V4's automatic LowMem128 layout. */
  lowMemoryRequired?: boolean
  /** A fallback or software adapter, which `auto` never picks. */
  software?: boolean
}

/**
 * Ask whether this browser can run the models on WebGPU V4, without running one.
 *
 * Cheap and side-effect free: requests an adapter but never creates a device.
 */
export async function checkSupport(model: ModelName = 'mindmap-fast'): Promise<DeviceReport> {
  if (typeof navigator === 'undefined' || !navigator.gpu)
    return {
      supported: false,
      reasons: [
        'navigator.gpu is undefined: either this browser has no WebGPU, or the ' +
        'page is not a secure context. about:blank and file:// are NOT secure ' +
        'contexts and look exactly like a browser without WebGPU; serve over ' +
        'https or localhost.',
      ],
    }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) return { supported: false, reasons: ['requestAdapter returned null'] }

  const reasons: string[] = []
  const hasShaderF16 = adapter.features.has('shader-f16')
  const hasSubgroups = adapter.features.has('subgroups')
  const { maxBufferSize, maxStorageBufferBindingSize } = adapter.limits

  const need = WEBGPU_MIN_BUFFER_BYTES
  if (maxStorageBufferBindingSize < need)
    reasons.push(
      `maxStorageBufferBindingSize is ${MiB(maxStorageBufferBindingSize)}, ` +
      `but WebGPU V4 requires at least ${MiB(need)}`)
  if (maxBufferSize < need)
    reasons.push(
      `maxBufferSize is ${MiB(maxBufferSize)}, but WebGPU V4 requires at least ${MiB(need)}`)

  const lowMemoryRequired = Math.min(maxBufferSize, maxStorageBufferBindingSize) < v4StdBufferBytes(model)

  // isFallbackAdapter moved from GPUAdapter to GPUAdapterInfo; read both.
  const info = adapter.info
  const software = !!(info?.isFallbackAdapter ??
    (adapter as { isFallbackAdapter?: boolean }).isFallbackAdapter) ||
    SOFTWARE_RENDERER.test(`${info?.vendor} ${info?.architecture} ${info?.description}`)

  return {
    supported: reasons.length === 0,
    reasons,
    maxBufferSize,
    maxStorageBufferBindingSize,
    hasShaderF16,
    hasSubgroups,
    lowMemoryRequired,
    software,
  }
}

/**
 * Acquire a WebGPU device suitable for the V4 engine, requesting optional
 * `shader-f16` and `subgroups` features and the adapter's reported limits.
 */
export async function acquireDevice(model: ModelName = 'mindmap-fast'): Promise<GPUDevice> {
  void model
  if (typeof navigator === 'undefined' || !navigator.gpu)
    throw new BrainchopError('no-webgpu',
      'navigator.gpu is undefined: either this browser has no WebGPU, or the ' +
      'page is not a secure context (https or localhost)')

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' })
  if (!adapter) throw new BrainchopError('no-adapter', 'requestAdapter returned null')

  const { maxBufferSize, maxStorageBufferBindingSize } = adapter.limits
  const need = WEBGPU_MIN_BUFFER_BYTES
  if (maxStorageBufferBindingSize < need || maxBufferSize < need)
    throw new BrainchopError('device-too-small',
      `this device allows ${MiB(maxStorageBufferBindingSize)} per storage ` +
      `binding and ${MiB(maxBufferSize)} per buffer, but WebGPU V4 needs at ` +
      `least ${MiB(need)}`)

  const requiredFeatures: GPUFeatureName[] = []
  if (adapter.features.has('shader-f16')) requiredFeatures.push('shader-f16')
  if (adapter.features.has('subgroups')) requiredFeatures.push('subgroups' as GPUFeatureName)

  const limitKeys = [
    'maxBufferSize',
    'maxStorageBufferBindingSize',
    'maxStorageBuffersPerShaderStage',
    'maxUniformBufferBindingSize',
    'maxComputeInvocationsPerWorkgroup',
    'maxComputeWorkgroupSizeX',
    'maxComputeWorkgroupSizeY',
    'maxComputeWorkgroupSizeZ',
    'maxComputeWorkgroupStorageSize',
    'maxComputeWorkgroupsPerDimension',
  ] as const
  const requiredLimits: Record<string, number> = {}
  const adapterLimits = adapter.limits as unknown as Record<string, unknown>
  for (const key of limitKeys) {
    const val = adapterLimits[key]
    if (typeof val === 'number' && Number.isFinite(val)) {
      requiredLimits[key] = val
    }
  }

  const device = await adapter.requestDevice({
    requiredFeatures,
    requiredLimits,
  })

  const tagged = device as GPUDevice & {
    __adapterInfo?: GPUAdapterInfo | null
    __isLost?: boolean
  }
  tagged.__adapterInfo = adapter.info ?? null
  tagged.__isLost = false
  if (device.lost && typeof device.lost.then === 'function') {
    void device.lost.then(() => {
      tagged.__isLost = true
    })
  }

  return device
}
