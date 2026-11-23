// Native Web Crypto SHA-256 prefix helper (first 8 lowercase hex chars) for
// stamping content-addressed model.bin weight provenance into NIfTI-1 descrip.
// WebGPU already requires a secure context where crypto.subtle is available
// (and Node 22+ provides globalThis.crypto.subtle natively).

const prefixCache = new WeakMap<Uint8Array, string>()

/**
 * Return the first 8 lowercase hex characters of the SHA-256 digest of `bytes`.
 * Results are memoized per Uint8Array instance (such as cached model.bin weights).
 */
export async function sha256Prefix8(bytes: Uint8Array): Promise<string> {
  const hit = prefixCache.get(bytes)
  if (hit !== undefined) return hit
  const copy = new Uint8Array(bytes.byteLength)
  copy.set(bytes)
  const digest = await globalThis.crypto.subtle.digest('SHA-256', copy.buffer)
  const head = new Uint8Array(digest, 0, 4)
  const prefix = Array.from(head, (b) => b.toString(16).padStart(2, '0')).join('')
  prefixCache.set(bytes, prefix)
  return prefix
}
