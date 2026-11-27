#!/usr/bin/env node
// Stage the Emscripten WebGL2 and CPU fallback artifacts into dist/ beside the
// bundled index.js. WebGPU uses the pure-JS V4 engine + dist/models/*.bin (no
// `-gpu.wasm` binaries).

import { copyFileSync, existsSync, mkdirSync, readFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const from = join(here, '..', '..', 'dist', 'wasm')
const to = join(here, '..', 'dist')

/*
 * One version, two files that must carry it: src/version.h owns it, npm insists
 * on a literal in its own manifest. Checked here rather than trusted, because
 * silent drift would ship a package claiming a version its modules do not print.
 * Set BC_SKIP_VERSION_CHECK=1 only during staged cross-PR builds where
 * src/version.h is updated in a sibling PR.
 */
if (process.env.BC_SKIP_VERSION_CHECK !== '1') {
  const owner = join(here, '..', '..', 'src', 'version.h')
  const declared = /#define BC_VERSION "([^"]+)"/.exec(readFileSync(owner, 'utf8'))?.[1]
  const packaged = JSON.parse(readFileSync(join(here, '..', 'package.json'), 'utf8')).version
  if (declared !== packaged) {
    console.error(`copy-wasm: version drift -- src/version.h says ${declared}, ` +
      `js/package.json says ${packaged}`)
    process.exit(1)
  }
}

// WebGL2 (`-gl`) and threaded CPU (`''`) fallback modules for all 5 models.
const MODELS = ['mindgrab', 'mindmap-fast', 'mindmap', 'mindmap-omni', 'mindsnap']
const files = ['-gl', ''].flatMap((b) => MODELS.flatMap((m) =>
  ['.js', '.wasm'].map((e) => `brainchop-${m}${b}${e}`)))

const missing = files.filter((f) => !existsSync(join(from, f)))
if (missing.length) {
  console.error(`copy-wasm: missing ${missing.join(', ')} in ${from}`)
  console.error('copy-wasm: run `npm run makeWasm` first (or `npm run build:webgpu` for WebGPU-only builds)')
  process.exit(1)
}

mkdirSync(to, { recursive: true })
for (const f of files) copyFileSync(join(from, f), join(to, f))
console.log(`copy-wasm: staged ${files.length} files into dist/`)
