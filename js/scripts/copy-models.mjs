#!/usr/bin/env node
// Stage flat float32 model weights (`src/models/<dir>/model.bin`) into
// `js/dist/models/<canonical>.bin` for the pure-JS WebGPU V4 engine.
// Works without Emscripten (`emcc`) and keeps a single source of truth for
// weights in `src/models/` (zero duplicate weight files in git).

import { copyFileSync, existsSync, mkdirSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const srcModels = join(here, '..', '..', 'src', 'models')
const dstModels = join(here, '..', 'dist', 'models')

const MAP = [
  ['mindgrab', 'mindgrab'],
  ['model16chan18cls', 'mindmap-fast'],
  ['mindmap', 'mindmap'],
  ['mindmap_omni', 'mindmap-omni'],
  ['mindsnap', 'mindsnap'],
]

const missing = MAP
  .map(([srcDir]) => join(srcModels, srcDir, 'model.bin'))
  .filter((p) => !existsSync(p))

if (missing.length) {
  console.error(`copy-models: missing weight file(s):\n  ${missing.join('\n  ')}`)
  process.exit(1)
}

mkdirSync(dstModels, { recursive: true })
let totalBytes = 0
for (const [srcDir, dstName] of MAP) {
  const from = join(srcModels, srcDir, 'model.bin')
  const to = join(dstModels, `${dstName}.bin`)
  copyFileSync(from, to)
  totalBytes += statSync(to).size
}

console.log(
  `copy-models: staged ${MAP.length} model binaries into dist/models/ ` +
  `(${(totalBytes / 1048576).toFixed(2)} MiB total)`,
)
