#!/usr/bin/env node
// The W3 gate: drive the built package in a real browser, through its public
// API, and hold it to the SAME voxel budgets `make test-wasm-gpu` uses.
//
// It deliberately does NOT re-test the kernels -- that is the C suite's job.
// What it tests is everything this package adds and the C does not: gzip on
// both sides, device acquisition, capability refusals, and the mask output.
//
//   node test/segment.test.mjs
//
// Headed, because headless Chromium on macOS has no GPU. Needs a checkout with
// playwright (BC_PLAYWRIGHT_ROOT, defaulting to ../../BrowserQC).

import { existsSync, mkdirSync, readFileSync, writeFileSync, mkdtempSync, copyFileSync, rmSync } from 'node:fs'
import { dirname, join, normalize, resolve, sep } from 'node:path'
import { tmpdir } from 'node:os'
import { createRequire } from 'node:module'
import { pathToFileURL, fileURLToPath } from 'node:url'
import { createServer } from 'node:http'
import { gunzipSync } from 'node:zlib'

const here = dirname(fileURLToPath(import.meta.url))
const repo = resolve(here, '..', '..')
const dist = join(here, '..', 'dist')

const root = [process.env.BC_PLAYWRIGHT_ROOT, join(repo, '..', 'BrowserQC')]
  .filter(Boolean).find((r) => existsSync(join(r, 'node_modules', 'playwright')))
if (!root) { console.error('playwright not found; set BC_PLAYWRIGHT_ROOT'); process.exit(2) }
const require = createRequire(pathToFileURL(join(root, 'package.json')))
const playwright = require('playwright')
// Selectable so this suite is the vehicle for validating a NEW platform: on a
// machine without WebGPU the webgpu cases will fail loudly, which is the
// correct report, and BC_TEST_BACKENDS can narrow it to the fallback.
const browserName = process.env.BC_TEST_BROWSER || 'chromium'

// Fixtures, and the budgets the Metal backend already carries.
const CASES = [
  { name: 'mindgrab', opts: { model: 'mindgrab' },
    input: 't1_crop.nii.gz', expect: 'bet.nii.gz', budget: 64 },
  { name: 'model16 --save-conform', opts: { model: '16chan18cls', saveConform: true },
    input: 't1_crop.nii.gz', expect: 'labels16.nii.gz', budget: 1024 },
  { name: 'model16 native 2mm', opts: { model: '16chan18cls' },
    input: 't1_2mm.nii.gz', expect: 'labels16_native_majority.nii.gz', expectCpu: 'labels16_native_majority_wasm.nii.gz', budget: 256 },
  // mindmap is 24 channels, so it exercises a different activation stride and
  // a WebGPU device limit the 16-channel models never reach. Same budgets:
  // they came from Metal and this model met them on both GPU backends.
  // Preserve the frozen inference references; new cleanup has a browser oracle.
  { name: 'mindmap --save-conform', opts: { model: 'mindmap', saveConform: true, legacyCleanup: true },
    input: 't1_crop.nii.gz', expect: 'labels24.nii.gz', budget: 1024 },
  { name: 'mindmap native 2mm', opts: { model: 'mindmap', legacyCleanup: true },
    input: 't1_2mm.nii.gz', expect: 'labels24_native_majority.nii.gz', budget: 256 },
  // 104 classes: src/Makefile's MS_TOL/MS_TOL_NATIVE, wider for a 104-way argmax.
  { name: 'mindsnap --save-conform', opts: { model: 'mindsnap', saveConform: true },
    input: 't1_crop.nii.gz', expect: 'labels104.nii.gz', expectCpu: 'labels104_wasm.nii.gz', budget: 2048 },
  { name: 'mindsnap native 2mm', opts: { model: 'mindsnap' },
    input: 't1_2mm.nii.gz', expect: 'labels104_native_majority.nii.gz', budget: 512 },
]

const stage = mkdtempSync(join(tmpdir(), 'bc-js-'))
/*
 * Registered HERE rather than with the server teardown below, so the
 * missing-dist check a few lines down is covered too -- it exits before that
 * point. A run that gets past it stages ~9 MB of modules and fixtures, and
 * nothing else would ever collect it. An exit handler covers process.exit();
 * rmSync is synchronous, so it is legal in one.
 */
process.on('exit', () => {
  try { rmSync(stage, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }) }
  catch (e) { console.error(`could not remove ${stage}: ${e.message}`) }
})
const BACKENDS = (process.env.BC_TEST_BACKENDS || 'webgpu,webgl2,cpu').split(',')

const stageFiles = [
  'index.js', 'worker.js', 'mindgrab.js', 'worker-mindgrab.js',
  'mindmap-fast.js', 'worker-mindmap-fast.js', '16chan18cls.js', 'worker-16chan18cls.js',
  'mindmap.js', 'worker-mindmap.js', 'mindmap-omni.js', 'worker-mindmap-omni.js',
  'mindsnap.js', 'worker-mindsnap.js',
  ...['mindgrab', 'mindmap-fast', 'mindmap', 'mindmap-omni', 'mindsnap'].map((m) => join('models', `${m}.bin`)),
  ...(BACKENDS.includes('webgl2')
    ? ['mindgrab', 'mindmap-fast', 'mindmap', 'mindmap-omni', 'mindsnap'].flatMap((m) =>
        ['.js', '.wasm'].map((e) => `brainchop-${m}-gl${e}`))
    : []),
  ...(BACKENDS.includes('cpu')
    ? ['mindgrab', 'mindmap-fast', 'mindmap', 'mindmap-omni', 'mindsnap'].flatMap((m) =>
        ['.js', '.wasm'].map((e) => `brainchop-${m}${e}`))
    : []),
]
mkdirSync(join(stage, 'models'), { recursive: true })
for (const f of stageFiles) {
  if (!existsSync(join(dist, f))) { console.error(`missing dist/${f}; run npm run build`); process.exit(2) }
  copyFileSync(join(dist, f), join(stage, f))
}
// Served gzipped, exactly as a user would hand over a downloaded .nii.gz.
for (const f of new Set(CASES.map((c) => c.input)))
  copyFileSync(join(repo, 'test', f), join(stage, f))
writeFileSync(join(stage, 'index.html'), '<!doctype html><title>brainchop</title>')

const types = { '.html': 'text/html', '.js': 'text/javascript',
  '.wasm': 'application/wasm', '.gz': 'application/gzip', '.bin': 'application/octet-stream' }

// Results come back by POST, never as a page.evaluate return value: serializing
// a 16.7M-element array over CDP costs 20+ seconds and would swamp the timings.
const received = new Map()
const server = createServer((req, res) => {
  if (req.method === 'POST') {
    const chunks = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', () => {
      received.set(req.url.slice(1), Buffer.concat(chunks)); res.writeHead(204); res.end()
    })
    return
  }
  // Normalize before joining: the raw request target is attacker-controlled by
  // anything that can reach this ephemeral port, and `/../../etc/passwd` would
  // otherwise escape the staging directory.
  // A malformed escape (`GET /%`) makes decodeURIComponent throw, and an
  // uncaught throw here kills the suite; `stage + sep` keeps a sibling temp
  // directory whose name merely extends this one out of reach.
  let raw
  try { raw = req.url === '/' ? '/index.html' : decodeURIComponent(req.url.split('?')[0]) }
  catch { res.writeHead(400); res.end(); return }
  const file = join(stage, normalize(raw).replace(/^(\.\.[/\\])+/, ''))
  if (!file.startsWith(stage + sep) || !existsSync(file)) { res.writeHead(404); res.end(); return }
  const ext = file.slice(file.lastIndexOf('.'))
  res.writeHead(200, {
    'Content-Type': types[ext] ?? 'application/octet-stream',
    // Cross-origin isolation, which the CPU backend cannot run without: its
    // module imports shared memory. Harmless to the GPU backends -- everything
    // here is same-origin -- and the host page of any deployment that wants the
    // CPU fallback has to send exactly these two.
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  })
  res.end(readFileSync(file))
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const port = server.address().port

// --enable-gpu/--ignore-gpu-blocklist matter on Linux, where a stock headless
// Chrome falls back to SwiftShader and would measure a software rasteriser.
// --use-angle=metal is macOS-only and would be rejected elsewhere.
const launchArgs = { headless: process.env.BC_WGSL_HEADLESS === '1' }
if (browserName === 'chromium') {
  launchArgs.args = ['--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu']
  if (process.env.BC_CHROME_CHANNEL) launchArgs.channel = process.env.BC_CHROME_CHANNEL
  if (process.env.BC_CHROME_EXECUTABLE) {
    delete launchArgs.channel
    launchArgs.executablePath = process.env.BC_CHROME_EXECUTABLE
  }
  launchArgs.args.push(...JSON.parse(process.env.BC_CHROME_ARGS || '[]'))
  if (process.platform === 'darwin' && !launchArgs.args.some((a) => a.startsWith('--use-angle=')))
    launchArgs.args.push('--use-angle=metal')
}
const browser = await playwright[browserName].launch(launchArgs)
const page = await browser.newPage()
page.on('pageerror', (e) => console.error('  pageerror:', e.message))
await page.goto(`http://127.0.0.1:${port}/index.html`, { waitUntil: 'domcontentloaded' })

// Belt and braces, and honestly no more than that: this file ends in an
// explicit process.exit(), which closes the socket anyway, and an 'exit'
// handler cannot rescue a run that hangs on an open handle: 'exit' fires only
// once the loop has drained or exit() was called, so it is the explicit
// server.close() at the end of the file, not this, that lets the process finish.
// Measured: drop that call as well as the exit() and the run hangs, handler
// never reached. It costs one line and covers a future edit that drops only the
// exit(). The staging directory has its own handler, registered at creation --
// see there for why, including the retry settings.
process.on('exit', () => { try { server.close() } catch {} })

// BACKENDS is declared near the top of this file before staging assets.

let failures = 0
const check = (ok, label, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? ` -- ${detail}` : ''}`)
  if (!ok) failures++
}

// --- support probe -----------------------------------------------------------
const support = await page.evaluate(async ({ origin }) => {
  const { checkSupport } = await import(`${origin}/index.js`)
  return checkSupport()
}, { origin: `http://127.0.0.1:${port}` })
// Only a FAILURE when webgpu is actually being tested. A machine without
// WebGPU is exactly where the WebGL2 fallback has to be validated, and
// `BC_TEST_BACKENDS=webgl2 npm test` is the documented way to do it -- a suite
// that reports a failure for the expected reason there is a suite nobody
// trusts. Reported either way, so the absence is never invisible.
if (BACKENDS.includes('webgpu'))
  check(support.supported, 'checkSupport reports this device can run the models',
    support.reasons.join('; '))
else
  console.log(`  --   webgpu not under test; checkSupport says supported=` +
    `${support.supported}${support.reasons.length ? ' (' + support.reasons.join('; ') + ')' : ''}`)

// --- the WebGL2 fallback probes independently --------------------------------
//
// Checked even on a machine where WebGPU works, because that is the only way
// this gets tested at all on a developer Mac -- and the fallback exists for
// machines nobody here runs.
const glSupport = await page.evaluate(async ({ origin }) => {
  const { checkWebgl2Support } = await import(`${origin}/index.js`)
  return checkWebgl2Support()
}, { origin: `http://127.0.0.1:${port}` })
check(glSupport.supported, 'checkWebgl2Support reports this device can run the models',
  glSupport.reasons.join('; ') ||
  `${glSupport.renderer}${glSupport.software ? ' (software)' : ''}, 3D<=${glSupport.max3dTextureSize}, ` +
  `${glSupport.maxDrawBuffers} draw buffers, allocates=${glSupport.allocates}`)

// --- the CPU fallback probes independently too -------------------------------
//
// This page IS cross-origin isolated (the server above sends both headers), so
// a false here means the module could not have been instantiated -- which is
// the failure a deployment discovers as "nothing happens".
if (BACKENDS.includes('cpu')) {
  const cpuSupport = await page.evaluate(async ({ origin }) => {
    const { checkCpuSupport } = await import(`${origin}/index.js`)
    return { ...checkCpuSupport(), isolated: globalThis.crossOriginIsolated }
  }, { origin: `http://127.0.0.1:${port}` })
  check(cpuSupport.supported && cpuSupport.isolated,
    'checkCpuSupport reports the threaded module can run on an isolated page',
    cpuSupport.reasons.join('; ') || `crossOriginIsolated=${cpuSupport.isolated}`)
}

// --- auto selection: hardware WebGPU, hardware WebGL2, then CPU -------------
{
  const picked = await page.evaluate(async ({ origin }) => {
    const { segment } = await import(`${origin}/index.js`)
    const bytes = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer()
    const out = await segment(bytes, { model: '16chan18cls' })
    return out.backend
  }, { origin: `http://127.0.0.1:${port}` })
  // The expectation follows the probe, not the platform. auto never picks a
  // software rasteriser, and this page is isolated, so CPU can run (brainchop#78).
  const want = support.supported && !support.software ? 'webgpu'
    : glSupport.supported && glSupport.software === false ? 'webgl2' : 'cpu'
  check(picked === want, `auto selection picks ${want} on this device`,
    `picked ${picked}`)
}

// --- capability refusals, before any GPU work --------------------------------
const refusals = await page.evaluate(async ({ origin }) => {
  const { segment } = await import(`${origin}/index.js`)
  const probe = async (opts) => {
    try { await segment(new Uint8Array([1, 2, 3, 4]).buffer, opts); return { code: 'no-error' } }
    catch (e) { return { code: e.code, message: e.message } }
  }
  return {
    conformOnMindgrab: await probe({ model: 'mindgrab', saveConform: true }),
    maskOnModel16: await probe({ model: '16chan18cls', mask: true }),
    borderOnModel16: await probe({ model: '16chan18cls', borderMm: 5 }),
    unknownModel: await probe({ model: 'nope' }),
    empty: await probe({ model: 'mindgrab' }),
  }
}, { origin: `http://127.0.0.1:${port}` })

check(refusals.conformOnMindgrab.code === 'unsupported-option',
  'mindgrab refuses saveConform', refusals.conformOnMindgrab.message)
check(refusals.maskOnModel16.code === 'unsupported-option',
  'model16 refuses mask', refusals.maskOnModel16.message)
check(refusals.borderOnModel16.code === 'unsupported-option',
  'model16 refuses borderMm', refusals.borderOnModel16.message)
check(refusals.unknownModel.code === 'unsupported-option',
  'an unknown model name is refused')

// --- the fixtures, on every backend ------------------------------------------
// WebGL2 uses fp16 voxel budgets and CPU is bit-exact (0); WebGPU runs the
// pure-JS V4 8-bit ZYCX engine and is gated via diceGate below.
for (const backend of BACKENDS.filter((b) => b !== 'webgpu'))
for (const [i, testCase] of CASES.entries()) {
  const key = `${backend}-r${i}`
  const info = await page.evaluate(async ({ origin, input, opts, key, backend }) => {
    const { segment } = await import(`${origin}/index.js`)
    const bytes = await (await fetch(`${origin}/${input}`)).arrayBuffer()
    const log = []
    // noTurbo: these are the fp16 budgets; turbo is gated separately below.
    const out = await segment(bytes, { ...opts, backend, noTurbo: true, onLog: (l) => log.push(l) })
    await fetch(`${origin}/${key}`, { method: 'POST', body: out.image })
    return {
      elapsedMs: Math.round(out.elapsedMs), bytes: out.image.byteLength,
      hasMask: !!out.mask, backend: out.backend, log,
    }
  }, { origin: `http://127.0.0.1:${port}`, input: testCase.input, opts: testCase.opts,
       key, backend })

  check(info.backend === backend,
    `${backend} ${testCase.name}: reported backend matches the request`,
    `got ${info.backend}`)
  const got = received.get(key)
  // The input was gzipped, so the output must be too -- that round trip is the
  // package's contract and is not exercised anywhere in the C suite.
  const gz = got[0] === 0x1f && got[1] === 0x8b
  check(gz, `${backend} ${testCase.name}: gzipped input produced gzipped output`)

  const plain = gz ? gunzipSync(got) : got
  const want = gunzipSync(readFileSync(join(repo, 'test', backend === 'cpu' && testCase.expectCpu ? testCase.expectCpu : testCase.expect)))
  let differ = 0
  if (plain.length !== want.length) differ = Infinity
  else for (let j = 352; j < want.length; j++) if (plain[j] !== want[j]) differ++
  // The new majority algorithm samples a model16 voxel whose native/wasm
  // arithmetic already differs in merged main. The wasm golden comes from
  // that unchanged build and the actual browser resampler, not this candidate.
  // Both CPU paths retain a zero-difference regression gate.
  // The GPU budgets are fp16 tolerances. The CPU module has nothing to tolerate:
  // it runs the same fp32 C as the native executables and is asserted bit-exact
  // in `make test-wasm`, so anything but zero here is a real change.
  const budget = backend === 'cpu' ? 0 : testCase.budget
  check(differ <= budget,
    `${backend} ${testCase.name}: ${differ} voxels differ (budget ${budget})`,
    `${info.elapsedMs} ms in the module`)
}

// --- WebGPU turbo (the default): src/Makefile's TURBO_GATE, as test-wasm-gpu --
function diceGate(ref, got) {
  const n = 256 * 256, inter = new Float64Array(n), nr = new Float64Array(n), ng = new Float64Array(n)
  let union = 0, same = 0
  for (let j = 352; j < ref.length; j++) {
    const r = ref[j], g = got[j]
    nr[r]++; ng[g]++
    if (r === g) inter[r]++
    if (r || g) { union++; if (r === g) same++ }
  }
  let sum = 0, classes = 0
  for (let c = 1; c < n; c++) if (nr[c] || ng[c]) { sum += 2 * inter[c] / (nr[c] + ng[c]); classes++ }
  return { dice: sum / Math.max(1, classes), fg: 100 * same / Math.max(1, union) }
}
if (BACKENDS.includes('webgpu'))
for (const [i, testCase] of CASES.entries()) {
  const key = `turbo-r${i}`
  const ms = await page.evaluate(async ({ origin, input, opts, key }) => {
    const { segment } = await import(`${origin}/index.js`)
    const bytes = await (await fetch(`${origin}/${input}`)).arrayBuffer()
    const out = await segment(bytes, { ...opts, backend: 'webgpu' })
    await fetch(`${origin}/${key}`, { method: 'POST', body: out.image })
    return Math.round(out.elapsedMs)
  }, { origin: `http://127.0.0.1:${port}`, input: testCase.input, opts: testCase.opts, key })
  const got = gunzipSync(received.get(key))
  const want = gunzipSync(readFileSync(join(repo, 'test', testCase.expect)))
  const [minDice, minFg] = testCase.name === 'mindgrab' ? [0.995, 99.5] : [0.988, 98.8]
  const { dice, fg } = got.length === want.length ? diceGate(want, got) : { dice: 0, fg: 0 }
  check(dice >= minDice && fg >= minFg,
    `webgpu V4 ${testCase.name}: Dice ${dice.toFixed(5)}, foreground agreement ${fg.toFixed(3)}%`,
    `gate ${minDice} / ${minFg}%, ${ms} ms`)
}

// --- repeated runs must not exhaust device memory -------------------------
// Exercises backend_webgpu.c's cleanup path on a cached module. Be precise
// about what this does and does not gate: the SUCCESS path always released the
// buffers, so this would not have failed before the cleanup rewrite -- it
// guards the module/layout/pipeline releases against future regression, and
// checks that repeated use neither degrades nor errors. The ten ERROR-path
// leaks, which stranded two 512 MiB activation buffers each, are NOT covered
// here because these runs succeed; inducing them needs a device that fails
// mid-run.
//
// It gates the WebGL2 path HARDER than the WebGPU one, and that is worth
// stating: backend_webgl2.c creates and destroys about 1.1 GiB of textures and
// 644 framebuffers on every call, all of it explicitly deleted in one cleanup
// label. Three runs on one page is a real test of that; leaking even one
// activation set would fail the third.
for (const backend of BACKENDS) {
  const repeat = await page.evaluate(async ({ origin, backend }) => {
    const { segment } = await import(`${origin}/index.js`)
    const bytes = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer()
    const ms = []
    for (let i = 0; i < 3; i++) {
      const out = await segment(bytes, { model: '16chan18cls', backend })
      ms.push(Math.round(out.elapsedMs))
    }
    return ms
  }, { origin: `http://127.0.0.1:${port}`, backend })
  check(repeat.length === 3,
    `${backend}: three back-to-back segmentations all completed`,
    `${repeat.join(' / ')} ms`)
}

// --- the worker path, and the responsiveness it exists for -------------------
//
// The claim under test is not "a worker works" but "the calling thread stays
// usable", so this measures it: a requestAnimationFrame ticker runs alongside
// the segmentation and any gap over 100 ms is a stall a user would see. In
// thread, webgl2 draws 3 frames and stalls for 2171 ms of a 2196 ms run.
for (const backend of BACKENDS) {
  const r = await page.evaluate(async ({ origin, backend }) => {
    const { segment } = await import(`${origin}/index.js`)
    const bytes = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer()
    const ticks = []
    let last = performance.now(), running = true
    const tick = () => {
      const now = performance.now(); ticks.push(now - last); last = now
      if (running) requestAnimationFrame(tick)
    }
    requestAnimationFrame(tick)
    const log = []
    const out = await segment(bytes, {
      model: '16chan18cls', backend, worker: true, noTurbo: true, onLog: (l) => log.push(l),
    })
    running = false
    const stalls = ticks.filter((g) => g > 100)
    await fetch(`${origin}/worker-${backend}`, { method: 'POST', body: out.image })
    return {
      backend: out.backend, ranInWorker: out.ranInWorker,
      elapsedMs: Math.round(out.elapsedMs), frames: ticks.length,
      longest: Math.round(Math.max(0, ...stalls)), logLines: log.length,
    }
  }, { origin: `http://127.0.0.1:${port}`, backend })

  check(r.ranInWorker && r.backend === backend,
    `${backend} in a worker: ran where it was asked to`,
    `backend=${r.backend} ranInWorker=${r.ranInWorker}`)
  // 100 ms is about six dropped frames: enough to be visible, loose enough not
  // to be flaky on a shared machine. The cpu backend gets 1000 ms, because it
  // is the one that saturates every core by design -- on a 2-core machine the
  // page's own rAF ticker is competing with the pool, and the js package
  // exposes no way to ask for fewer threads. It still catches the failure this
  // check is for, which is a backend blocking the calling thread outright.
  const stallLimit = backend === 'cpu' ? 1000 : 100
  check(r.longest < stallLimit,
    `${backend} in a worker: the calling thread stayed responsive`,
    `${r.frames} frames, longest gap ${r.longest} ms (limit ${stallLimit}), ` +
    `${r.elapsedMs} ms in the module`)
  check(r.logLines > 0, `${backend} in a worker: module logs reach onLog`,
    `${r.logLines} lines`)

  // Same volume, same budget as the in-thread run: a worker must not change
  // the answer, and this is the only place that is actually checked.
  const got = received.get(`worker-${backend}`)
  const plain = got[0] === 0x1f && got[1] === 0x8b ? gunzipSync(got) : got
  const want = gunzipSync(readFileSync(join(repo, 'test', backend === 'cpu' ? 'labels16_native_majority_wasm.nii.gz' : 'labels16_native_majority.nii.gz')))
  if (backend === 'webgpu') {
    const { dice, fg } = plain.length === want.length ? diceGate(want, plain) : { dice: 0, fg: 0 }
    check(dice >= 0.988 && fg >= 98.8,
      `webgpu in a worker: Dice ${dice.toFixed(5)}, fg ${fg.toFixed(3)}%`)
  } else {
    let differ = plain.length !== want.length ? Infinity : 0
    if (differ === 0) for (let j = 352; j < want.length; j++) if (plain[j] !== want[j]) differ++
    const wBudget = backend === 'cpu' ? 0 : 256
    check(differ <= wBudget,
      `${backend} in a worker: ${differ} voxels differ (budget ${wBudget})`)
  }
}

// --- a worker cannot be handed a device --------------------------------------
{
  const refused = await page.evaluate(async ({ origin }) => {
    const { segment, acquireGlContext } = await import(`${origin}/index.js`)
    try {
      await segment(new Uint8Array([1, 2, 3, 4]).buffer,
        { model: 'mindgrab', worker: true, glContext: acquireGlContext() })
      return { code: 'no-error' }
    } catch (e) { return { code: e.code, message: e.message } }
  }, { origin: `http://127.0.0.1:${port}` })
  check(refused.code === 'unsupported-option',
    'worker combined with glContext is refused', refused.message)
}

// --- the mask output ---------------------------------------------------------
{
  const info = await page.evaluate(async ({ origin }) => {
    const { segment } = await import(`${origin}/index.js`)
    const bytes = await (await fetch(`${origin}/t1_crop.nii.gz`)).arrayBuffer()
    const out = await segment(bytes, { model: 'mindgrab', mask: true, borderMm: 2 })
    return { hasMask: !!out.mask, maskBytes: out.mask?.byteLength ?? 0 }
  }, { origin: `http://127.0.0.1:${port}` })
  check(info.hasMask && info.maskBytes > 0,
    'mindgrab returns a mask when asked', `${info.maskBytes} bytes`)
}

// --- a silently empty GPU result must reject through both public transports ---
if (BACKENDS.includes('webgpu')) {
  const failed = await page.evaluate(async ({ origin }) => {
    const { segment, acquireDevice } = await import(`${origin}/index.js`)
    const input = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer()
    const device = await acquireDevice('16chan18cls')
    let live = 0
    const allocate = device.createBuffer.bind(device)
    device.createBuffer = (descriptor) => {
      const buffer = allocate(descriptor), destroy = buffer.destroy.bind(buffer)
      live += buffer.size
      let freed = false
      buffer.destroy = () => { if (!freed) { live -= buffer.size; freed = true } destroy() }
      return buffer
    }
    // Leave the fresh zero-filled buffers readable, but suppress every GPU
    // submission. This models a backend returning success with unusable labels.
    device.queue.submit = () => {}
    const log = []
    try {
      await segment(input, { model: '16chan18cls', backend: 'webgpu', device,
        onLog: (line) => log.push(line) })
      return { code: 'unexpected-success', live, log }
    } catch (e) { return { code: e.code, live, log } }
    finally { device.destroy() }
  }, { origin: `http://127.0.0.1:${port}` })
  check(failed.code === 'inference-failed' && failed.log.some(l => l.includes('segmentation is empty')),
    'empty GPU labels reject through segment()', JSON.stringify(failed))
  check(failed.live === 0, 'GPU failure releases every allocated buffer', `${failed.live} bytes remain`)

  // The same for a mask: an empty MindGrab result once returned success (brainchop#78).
  const blank = await page.evaluate(async ({ origin }) => {
    const { segment, acquireDevice } = await import(`${origin}/index.js`)
    const input = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer()
    const device = await acquireDevice('mindgrab'), log = []
    device.queue.submit = () => {}
    try {
      await segment(input, { model: 'mindgrab', backend: 'webgpu', device, onLog: (l) => log.push(l) })
      return { code: 'unexpected-success', log }
    } catch (e) { return { code: e.code, log } }
    finally { device.destroy() }
  }, { origin: `http://127.0.0.1:${port}` })
  check(blank.code === 'inference-failed' && blank.log.some(l => l.includes('found no brain')),
    'empty GPU mask rejects through segment()', JSON.stringify(blank))

  const workerPath = join(stage, 'worker.js'), originalWorker = readFileSync(workerPath, 'utf8')
  const prelude = `const originalAdapter = navigator.gpu.requestAdapter.bind(navigator.gpu);
    navigator.gpu.requestAdapter = async (...args) => {
      const adapter = await originalAdapter(...args);
      if (adapter) { const request = adapter.requestDevice.bind(adapter);
        adapter.requestDevice = async (...args) => { const device = await request(...args);
          device.queue.submit = () => {}; return device; }; }
      return adapter;
    };
`
  writeFileSync(workerPath, prelude + originalWorker)
  try {
    const failedWorker = await page.evaluate(async ({ origin }) => {
      const { segment, disposeWorker } = await import(`${origin}/index.js`)
      disposeWorker()
      const bytes = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer(), log = []
      try { await segment(bytes, { model: '16chan18cls', backend: 'webgpu', worker: true,
        onLog: (line) => log.push(line) }); return { code: 'unexpected-success', log } }
      catch (e) { return { code: e.code, log } }
    }, { origin: `http://127.0.0.1:${port}` })
    check(failedWorker.code === 'inference-failed' && failedWorker.log.some(l => l.includes('segmentation is empty')),
      'empty GPU labels reject through the worker', JSON.stringify(failedWorker))
  } finally {
    writeFileSync(workerPath, originalWorker)
    await page.evaluate(async ({ origin }) => {
      const { disposeWorker } = await import(`${origin}/index.js`)
      disposeWorker()
    }, { origin: `http://127.0.0.1:${port}` })
  }
}

// --- per-model entry: its own worker, one model, a refusal for the others ---
// The staged dir holds every glue, so a stray import would still load here; the
// one-model guarantee is esbuild.config.mjs's check. This gates that the entry
// and its worker actually segment.
for (const backend of BACKENDS) {
  const r = await page.evaluate(async ({ origin, backend }) => {
    const { segment, listModels } = await import(`${origin}/mindgrab.js`)
    const bytes = await (await fetch(`${origin}/t1_crop.nii.gz`)).arrayBuffer()
    let other
    try { await segment(bytes, { model: 'mindmap', backend }); other = 'no-error' }
    catch (e) { other = e.code }
    const out = await segment(bytes, { model: 'mindgrab', backend, worker: true, noTurbo: true })
    await fetch(`${origin}/entry-${backend}`, { method: 'POST', body: out.image })
    return { models: listModels().map((m) => m.name), other, ranInWorker: out.ranInWorker }
  }, { origin: `http://127.0.0.1:${port}`, backend })
  check(r.models.join() === 'mindgrab' && r.other === 'unsupported-option' && r.ranInWorker,
    `${backend}: mindgrab entry lists one model, refuses mindmap, runs in its worker`, JSON.stringify(r))
  const plain = gunzipSync(received.get(`entry-${backend}`))
  const want = gunzipSync(readFileSync(join(repo, 'test', 'bet.nii.gz')))
  if (backend === 'webgpu') {
    const { dice, fg } = plain.length === want.length ? diceGate(want, plain) : { dice: 0, fg: 0 }
    check(dice >= 0.995 && fg >= 99.5,
      `webgpu: mindgrab entry Dice ${dice.toFixed(5)}, fg ${fg.toFixed(3)}%`)
  } else {
    let differ = plain.length === want.length ? 0 : Infinity
    for (let j = 352; differ !== Infinity && j < want.length; j++) if (plain[j] !== want[j]) differ++
    const budget = backend === 'cpu' ? 0 : 64
    check(differ <= budget, `${backend}: mindgrab entry ${differ} voxels differ (budget ${budget})`)
  }
}

await browser.close()
server.close()
console.log(failures ? `\n${failures} check(s) failed` : '\nall checks passed')
process.exit(failures ? 1 : 0)
