// Synchronous FIPS 180-4 SHA-256 prefix helper (first 8 lowercase hex chars)
// for stamping content-addressed model.bin weight provenance into NIfTI-1
// descrip headers without depending on Web Crypto availability on HTTP origins.

const INITIAL_STATE = new Uint32Array(8)
const ROUND_CONSTANTS = new Uint32Array(64)

{
  let found = 0
  for (let cand = 2; found < 64; cand++) {
    let prime = true
    for (let d = 2; d * d <= cand; d++) {
      if (cand % d === 0) { prime = false; break }
    }
    if (!prime) continue
    if (found < 8) {
      const r2 = Math.sqrt(cand)
      INITIAL_STATE[found] = Math.floor((r2 - Math.floor(r2)) * 0x1_0000_0000) >>> 0
    }
    const r3 = Math.cbrt(cand)
    ROUND_CONSTANTS[found] = Math.floor((r3 - Math.floor(r3)) * 0x1_0000_0000) >>> 0
    found++
  }
}

const prefixCache = new WeakMap<Uint8Array, string>()

function rotr32(v: number, bits: number): number {
  return ((v >>> bits) | (v << (32 - bits))) >>> 0
}

function compressBlock(
  st: Uint32Array,
  msg: Uint32Array,
  buf: Uint8Array,
  off: number,
): void {
  for (let i = 0; i < 16; i++) {
    const p = off + i * 4
    msg[i] = ((buf[p] << 24) | (buf[p + 1] << 16) | (buf[p + 2] << 8) | buf[p + 3]) >>> 0
  }
  for (let i = 16; i < 64; i++) {
    const a = msg[i - 15]
    const b = msg[i - 2]
    const p0 = (rotr32(a, 7) ^ rotr32(a, 18) ^ (a >>> 3)) >>> 0
    const p1 = (rotr32(b, 17) ^ rotr32(b, 19) ^ (b >>> 10)) >>> 0
    msg[i] = (msg[i - 16] + p0 + msg[i - 7] + p1) >>> 0
  }
  let v0 = st[0], v1 = st[1], v2 = st[2], v3 = st[3]
  let v4 = st[4], v5 = st[5], v6 = st[6], v7 = st[7]
  for (let i = 0; i < 64; i++) {
    const u1 = (rotr32(v4, 6) ^ rotr32(v4, 11) ^ rotr32(v4, 25)) >>> 0
    const ch = ((v4 & v5) ^ ((~v4) & v6)) >>> 0
    const t1 = (v7 + u1 + ch + ROUND_CONSTANTS[i] + msg[i]) >>> 0
    const u0 = (rotr32(v0, 2) ^ rotr32(v0, 13) ^ rotr32(v0, 22)) >>> 0
    const mj = ((v0 & v1) ^ (v0 & v2) ^ (v1 & v2)) >>> 0
    const t2 = (u0 + mj) >>> 0
    v7 = v6; v6 = v5; v5 = v4; v4 = (v3 + t1) >>> 0
    v3 = v2; v2 = v1; v1 = v0; v0 = (t1 + t2) >>> 0
  }
  st[0] = (st[0] + v0) >>> 0
  st[1] = (st[1] + v1) >>> 0
  st[2] = (st[2] + v2) >>> 0
  st[3] = (st[3] + v3) >>> 0
  st[4] = (st[4] + v4) >>> 0
  st[5] = (st[5] + v5) >>> 0
  st[6] = (st[6] + v6) >>> 0
  st[7] = (st[7] + v7) >>> 0
}

/**
 * Return the first 8 lowercase hex characters of the SHA-256 digest of `bytes`.
 * Results are memoized per Uint8Array instance (such as cached model.bin weights).
 */
export function sha256Prefix8(bytes: Uint8Array): string {
  const hit = prefixCache.get(bytes)
  if (hit !== undefined) return hit

  const st = new Uint32Array(INITIAL_STATE)
  const msg = new Uint32Array(64)
  const total = bytes.byteLength
  let pos = 0
  while (pos + 64 <= total) {
    compressBlock(st, msg, bytes, pos)
    pos += 64
  }

  const pad = new Uint8Array(64)
  const rem = total - pos
  if (rem > 0) pad.set(bytes.subarray(pos), 0)
  pad[rem] = 0x80
  if (rem >= 56) {
    compressBlock(st, msg, pad, 0)
    pad.fill(0)
  }

  const nbits = BigInt(total) * 8n
  for (let i = 0; i < 8; i++) {
    pad[63 - i] = Number((nbits >> BigInt(i * 8)) & 0xffn)
  }
  compressBlock(st, msg, pad, 0)

  const prefix = st[0].toString(16).padStart(8, '0')
  prefixCache.set(bytes, prefix)
  return prefix
}
