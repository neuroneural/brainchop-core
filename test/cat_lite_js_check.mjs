/*
 * Run the upstream cat-lite.js on a BRAINCHOPC_PVE_DUMP file and write the
 * three tissue fractions as raw f32, so test/cat_lite_check.c's output can be
 * compared against it voxel for voxel.
 *
 * With a fourth argument it also compares that file (the C port's output) against
 * its own, to float32 epsilon, and exits nonzero on disagreement.
 *
 * usage: node cat_lite_js_check.mjs DUMP OUT [BRAINCHOP_TEST_DIR [C_OUTPUT]]
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const [dumpPath, outPath, treeArgument, comparePath] = process.argv.slice(2)
if (!dumpPath || !outPath) {
  console.error('usage: node cat_lite_js_check.mjs DUMP OUT [CAT_LITE_DIR [C_OUTPUT]]')
  process.exit(2)
}
const defaultDir = resolve(dirname(fileURLToPath(import.meta.url)), '../js/src/webgpu')
const tree = resolve(treeArgument ?? process.env.BC_CAT_LITE_JS_DIR ?? defaultDir)
const { applyCatLitePartialVolume } = await import(pathToFileURL(resolve(tree, 'cat-lite.js')))

const DIM = 256
const COUNT = DIM * DIM * DIM
const PLANES = 4

const raw = readFileSync(dumpPath)
if (raw.byteLength !== (PLANES + 1) * COUNT * 4) {
  console.error(`cat_lite_js_check: ${dumpPath} is ${raw.byteLength} bytes, expected ${(PLANES + 1) * COUNT * 4}`)
  process.exit(1)
}
const view = new Float32Array(raw.buffer, raw.byteOffset, (PLANES + 1) * COUNT)
const plane = (i) => view.subarray(i * COUNT, (i + 1) * COUNT).slice()
// applyCatLitePartialVolume mutates its priors in place, so hand it copies.
const volumes = [plane(0), plane(1), plane(2)]        // GM, WM, CSF
const intensity = view.subarray(PLANES * COUNT, (PLANES + 1) * COUNT)

// The model entry's CAT-lite constants. Everything else upstream of this call
// already happened in C and arrived through the dump.
const options = {
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

const { tissues, stats } = applyCatLitePartialVolume(
  volumes, intensity, [DIM, DIM, DIM], options
)
console.error('[js] stats', JSON.stringify(stats.applied ? {
  means: stats.tissueMeans, sigmas: stats.tissueSigmas,
} : stats))

const out = Buffer.alloc(3 * COUNT * 4)
const outView = new Float32Array(out.buffer, out.byteOffset, 3 * COUNT)
for (let t = 0; t < 3; t++) outView.set(tissues[t], t * COUNT)
writeFileSync(outPath, out)
console.error(`[js] wrote ${outPath}`)

// The comparison lives here rather than in a third script because this process
// already holds the reference result in memory. A numpy helper would be the
// only test in the repo needing it -- test/nii_diff.py, the established
// comparator, is stdlib-only -- and `make test` is the one gate that must run
// on a bare machine.
//
// The tolerance is float32 epsilon, not a fitted number: both sides store their
// result as float32, so 2^-24 is the smallest difference that can mean
// anything. A porting error shows up orders of magnitude above it; the one this
// gate caught on its first run read 5.5e-02.
if (comparePath) {
  const FLOAT32_EPSILON = 2 ** -24
  const raw = readFileSync(comparePath)
  if (raw.byteLength !== 3 * COUNT * 4) {
    console.error(`cat_lite_js_check: ${comparePath} is ${raw.byteLength} bytes, expected ${3 * COUNT * 4}`)
    process.exit(1)
  }
  const c = new Float32Array(raw.buffer, raw.byteOffset, 3 * COUNT)
  let worst = 0
  for (let tissue = 0; tissue < 3; tissue++) {
    let max = 0, sum = 0
    for (let i = tissue * COUNT; i < (tissue + 1) * COUNT; i++) {
      const d = Math.abs(c[i] - outView[i])
      if (d > max) max = d
      sum += d
    }
    if (max > worst) worst = max
    console.error(`  ${['gm', 'wm', 'csf'][tissue].padEnd(4)} max ${max.toExponential(3)}  mean ${(sum / COUNT).toExponential(3)}`)
  }
  console.error(`cat-lite C vs reference: max deviation ${worst.toExponential(3)} ` +
    `(float32 epsilon ${FLOAT32_EPSILON.toExponential(3)})`)
  if (worst > FLOAT32_EPSILON) {
    console.error('cat_lite_js_check: the C port and cat-lite.js disagree by more than float32 can represent')
    process.exit(1)
  }
}
