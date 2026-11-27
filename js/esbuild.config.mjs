import { build } from 'esbuild'
import { readFileSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'

// The emscripten glue is deliberately NOT in this bundle; scripts/copy-wasm.mjs
// stages it into dist/ beside it, and modules.js imports it by those literal
// relative specifiers, which `external` keeps verbatim. `platform: neutral`
// keeps esbuild from injecting node or browser shims into a plain ES module.
const common = {
  bundle: true,
  external: ['./brainchop-*'],
  format: 'esm',
  platform: 'neutral',
  target: ['es2022'],
  sourcemap: true,
  legalComments: 'inline',
}

// The worker is a SEPARATE, self-contained bundle rather than a module that
// imports ./index.js at run time. A host that bundles the package rewrites its
// own copy of index.js, and a worker reaching for a relative './index.js' would
// then find nothing. It is loaded by URL, so it must not be part of index's bundle.
//
// __BC_WORKER__ is inlined as a string literal so a host bundler still sees
// `new Worker(new URL('<literal>', import.meta.url))` and emits that worker.
const PKG_VERSION = JSON.parse(readFileSync('package.json', 'utf8')).version

const entry = (name, worker, plugins = []) => {
  const define = {
    __BC_WORKER__: JSON.stringify(`./${worker}.js`),
    __BC_VERSION__: JSON.stringify(PKG_VERSION),
  }
  return [
    build({ ...common, plugins, define, entryPoints: ['src/index.ts'], outfile: `dist/${name}.js` }),
    build({ ...common, plugins, define, entryPoints: ['src/worker.ts'], outfile: `dist/${worker}.js` }),
  ]
}

/*
 * Per-model entry points (`@brainchop/core/mindgrab`, `/mindmap-fast`,
 * `/16chan18cls`, `/mindmap`, `/mindmap-omni`, `/mindsnap`). A bundler emits every literal
 * `import()` and `new URL()` it sees, so these filter `modules.js` to a single
 * model's `.bin` weight URL and WebGL2/CPU fallback glue.
 */
const MODELS = Object.keys((await import('./src/modules.js')).MODEL_BINS)
const canonicalOf = (model) => (model === '16chan18cls' ? 'mindmap-fast' : model)
const modulesSource = readFileSync('src/modules.js', 'utf8')

const srcDir = resolve('src')
const oneModel = (model) => {
  const canonical = canonicalOf(model)
  const glueRe = new RegExp(`brainchop-${canonical}(-gl)?\\.js`)
  const binRe = new RegExp(`\\./models/${canonical}\\.bin`)
  return {
    name: 'one-model',
    setup(b) {
      b.onResolve({ filter: /^\.\.?\/modules\.js$/ }, () => ({
        path: model,
        namespace: 'one-model',
      }))
      b.onLoad({ filter: /.*/, namespace: 'one-model' }, () => ({
        resolveDir: srcDir,
        contents: modulesSource.split('\n').filter((line) => {
          if (line.includes("import('")) return glueRe.test(line)
          if (line.includes("new URL('./models/")) return binRe.test(line)
          return true
        }).join('\n'),
      }))
    },
  }
}

await Promise.all([
  ...entry('index', 'worker'),
  ...MODELS.flatMap((model) => entry(model, `worker-${model}`, [oneModel(model)])),
])

// Guard the point of the per-model entries: one stray import or model URL re-adds megabytes.
for (const model of MODELS) {
  const canonical = canonicalOf(model)
  for (const file of [`dist/${model}.js`, `dist/worker-${model}.js`]) {
    const code = readFileSync(file, 'utf8')
    const glue = new Set(code.match(/brainchop-[\w-]+?(?=(-gl)?\.js)/g))
    if (glue.size !== 1 || !glue.has(`brainchop-${canonical}`))
      throw new Error(`${file} references glue: ${[...glue].join(', ')}`)
    const bins = new Set(code.match(/\.\/models\/[\w-]+\.bin/g))
    if (bins.size !== 1 || !bins.has(`./models/${canonical}.bin`))
      throw new Error(`${file} references model bins: ${[...bins].join(', ')}`)
  }
}

/*
 * Make the shipped declarations self-sufficient: tsc 5.9.3 does not propagate
 * `/// <reference types="@webgpu/types" />` into emitted declaration files.
 */
const types = 'dist/index.d.ts'
const directive = '/// <reference types="@webgpu/types" />\n'
const declarations = readFileSync(types, 'utf8')
if (!declarations.startsWith(directive)) writeFileSync(types, directive + declarations)

console.log(`esbuild: wrote dist/index.js, dist/worker.js and ${MODELS.length} per-model entries`)
