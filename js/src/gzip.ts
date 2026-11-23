import { BrainchopError } from './error.js'

/*
 * gzip lives here and nowhere else.
 *
 * The wasm module links no zlib at all -- a deliberate bundle-size decision --
 * and the C code refuses a `.nii.gz` cleanly rather than misreading it. So the module only ever sees raw NIfTI, and this file is the
 * whole of the compression contract, using the platform's own streams rather
 * than a bundled inflate.
 */

/** gzip's magic number. NIfTI-1's own first word is the header size, 348. */
const GZIP_MAGIC = [0x1f, 0x8b]

export function isGzip(bytes: Uint8Array): boolean {
  return bytes.length >= 2 && bytes[0] === GZIP_MAGIC[0] && bytes[1] === GZIP_MAGIC[1]
}

async function through(bytes: Uint8Array, stream: TransformStream<Uint8Array, Uint8Array>) {
  // A fresh Uint8Array, because a view into the wasm heap (or any SharedArrayBuffer)
  // is not something Response will accept.
  const copy = new Uint8Array(bytes)
  const body = new Blob([copy as BlobPart]).stream().pipeThrough(stream)
  return new Uint8Array(await new Response(body).arrayBuffer())
}

export async function gunzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof DecompressionStream === 'undefined')
    throw new BrainchopError('bad-input',
      'the input is gzipped but this environment has no DecompressionStream; ' +
      'decompress it before calling, or pass an uncompressed NIfTI')
  return through(bytes, new DecompressionStream('gzip') as unknown as
    TransformStream<Uint8Array, Uint8Array>)
}

export async function gzip(bytes: Uint8Array): Promise<Uint8Array> {
  if (typeof CompressionStream === 'undefined')
    throw new BrainchopError('bad-input',
      'gzip output was requested but this environment has no CompressionStream')
  return through(bytes, new CompressionStream('gzip') as unknown as
    TransformStream<Uint8Array, Uint8Array>)
}
