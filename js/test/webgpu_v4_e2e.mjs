#!/usr/bin/env node
// Real Chrome WebGPU V4 end-to-end verification across all 4 models, CAT-lite
// tissues, main-thread & Web Worker execution, monotonic onProgress, forced
// LowMem128 + paced mode, per-model subpath entries, and zero-leak error paths.

import assert from 'node:assert/strict'
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

const pwRoot = [
  process.env.BC_PLAYWRIGHT_ROOT,
  join(repo, '..', 'brainchop-test'),
  join(repo, '..', 'BrowserQC'),
].filter(Boolean).find((r) => existsSync(join(r, 'node_modules', 'playwright')))

if (!pwRoot) {
  console.error('playwright not found; set BC_PLAYWRIGHT_ROOT')
  process.exit(2)
}
const require = createRequire(pathToFileURL(join(pwRoot, 'package.json')))
const playwright = require('playwright')

const stage = mkdtempSync(join(tmpdir(), 'bc-js-v4-e2e-'))
process.on('exit', () => {
  try { rmSync(stage, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }) } catch {}
})

mkdirSync(join(stage, 'models'), { recursive: true })
for (const f of [
  'index.js', 'worker.js',
  'mindgrab.js', 'worker-mindgrab.js',
  'mindmap-fast.js', 'worker-mindmap-fast.js',
  '16chan18cls.js', 'worker-16chan18cls.js',
  'mindmap.js', 'worker-mindmap.js',
  'mindmap-omni.js', 'worker-mindmap-omni.js',
  'mindsnap.js', 'worker-mindsnap.js',
  'models/mindgrab.bin', 'models/mindmap-fast.bin',
  'models/mindmap.bin', 'models/mindmap-omni.bin', 'models/mindsnap.bin',
]) {
  const src = join(dist, f)
  if (!existsSync(src)) {
    console.error(`missing dist/${f}; run npm run build:webgpu first`)
    process.exit(2)
  }
  copyFileSync(src, join(stage, f))
}
for (const f of ['t1_crop.nii.gz', 't1_2mm.nii.gz']) {
  copyFileSync(join(repo, 'test', f), join(stage, f))
}
writeFileSync(join(stage, 'index.html'), '<!doctype html><title>brainchop WebGPU V4 E2E</title>')

const mime = {
  '.html': 'text/html',
  '.js': 'text/javascript',
  '.bin': 'application/octet-stream',
  '.gz': 'application/gzip',
}

const received = new Map()
const server = createServer((req, res) => {
  if (req.method === 'POST') {
    const chunks = []
    req.on('data', (c) => chunks.push(c))
    req.on('end', () => {
      received.set(req.url.slice(1), Buffer.concat(chunks))
      res.writeHead(204)
      res.end()
    })
    return
  }
  let raw
  try { raw = req.url === '/' ? '/index.html' : decodeURIComponent(req.url.split('?')[0]) }
  catch { res.writeHead(400); res.end(); return }
  const file = join(stage, normalize(raw).replace(/^(\.\.[/\\])+/, ''))
  if (!file.startsWith(stage + sep) || !existsSync(file)) { res.writeHead(404); res.end(); return }
  const ext = file.slice(file.lastIndexOf('.'))
  res.writeHead(200, {
    'Content-Type': mime[ext] ?? 'application/octet-stream',
    'Cross-Origin-Opener-Policy': 'same-origin',
    'Cross-Origin-Embedder-Policy': 'require-corp',
  })
  res.end(readFileSync(file))
})
await new Promise((r) => server.listen(0, '127.0.0.1', r))
const origin = `http://127.0.0.1:${server.address().port}`
process.on('exit', () => { try { server.close() } catch {} })

const launchArgs = {
  headless: process.env.BC_WGSL_HEADLESS === '1',
  args: ['--enable-gpu', '--ignore-gpu-blocklist', '--enable-unsafe-webgpu'],
}
if (process.platform === 'darwin') {
  launchArgs.args.push('--use-angle=metal')
  const chromeApp = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'
  if (existsSync(chromeApp) && !process.env.BC_CHROME_EXECUTABLE) {
    launchArgs.executablePath = chromeApp
  }
}
if (process.env.BC_CHROME_EXECUTABLE) {
  launchArgs.executablePath = process.env.BC_CHROME_EXECUTABLE
}

const browser = await playwright.chromium.launch(launchArgs)
const page = await browser.newPage()
page.on('pageerror', (e) => console.error('  [pageerror]', e.message))
await page.goto(`${origin}/index.html`, { waitUntil: 'domcontentloaded' })

let failures = 0
const check = (ok, label, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label}${detail ? ` -- ${detail}` : ''}`)
  if (!ok) failures++
}

function macroDiceAndFg(refNii, gotNii, binaryMask = false) {
  assert.equal(gotNii.length, refNii.length, `byte length mismatch: got ${gotNii.length} vs ref ${refNii.length}`)
  if (binaryMask) {
    let inter = 0, nr = 0, ng = 0, diff = 0
    for (let i = 352; i < refNii.length; i++) {
      const r = refNii[i] > 0 ? 1 : 0
      const g = gotNii[i] > 0 ? 1 : 0
      if (r) nr++
      if (g) ng++
      if (r && g) inter++
      if (refNii[i] !== gotNii[i]) diff++
    }
    const dice = (2 * inter) / Math.max(1, nr + ng)
    const fg = (100 * inter) / Math.max(1, nr + ng - inter)
    return { dice, fg, diff }
  }
  const n = 256
  const inter = new Float64Array(n), nr = new Float64Array(n), ng = new Float64Array(n)
  let union = 0, same = 0, diff = 0
  for (let i = 352; i < refNii.length; i++) {
    const r = refNii[i], g = gotNii[i]
    nr[r]++; ng[g]++
    if (r === g) inter[r]++
    else diff++
    if (r || g) { union++; if (r === g) same++ }
  }
  let sum = 0, classes = 0
  for (let c = 1; c < n; c++) {
    if (nr[c] || ng[c]) {
      sum += (2 * inter[c]) / (nr[c] + ng[c])
      classes++
    }
  }
  return { dice: sum / Math.max(1, classes), fg: (100 * same) / Math.max(1, union), diff }
}

function isMonotonicProgress(events) {
  if (!Array.isArray(events) || events.length < 3) return false
  for (let i = 1; i < events.length; i++) {
    if (events[i].progress < events[i - 1].progress) return false
  }
  return events[0].progress >= 0 && events[events.length - 1].progress === 1
}

try {
  // 1. Capability probe
  const sup = await page.evaluate(async (o) => {
    const { checkSupport } = await import(`${o}/index.js`)
    return checkSupport('mindmap')
  }, origin)
  check(
    sup.supported && !sup.software,
    'checkSupport detects hardware WebGPU',
    `f16=${sup.hasShaderF16} subgroups=${sup.hasSubgroups} maxBuf=${Math.round((sup.maxBufferSize || 0) / 1048576)}MiB`,
  )

  // 2. All 4 models (conformed + native) + 16chan18cls alias + legacyCleanup checks
  const cases = [
    { name: 'mindgrab (t1_crop)', opts: { model: 'mindgrab', backend: 'webgpu' },
      input: 't1_crop.nii.gz', ref: 'bet.nii.gz', minDice: 0.99, binaryMask: true },
    { name: 'mindmap-fast conformed (t1_crop)', opts: { model: 'mindmap-fast', saveConform: true, backend: 'webgpu' },
      input: 't1_crop.nii.gz', ref: 'labels16.nii.gz', minDice: 0.985 },
    { name: '16chan18cls alias conformed (t1_crop)', opts: { model: '16chan18cls', saveConform: true, backend: 'webgpu' },
      input: 't1_crop.nii.gz', ref: 'labels16.nii.gz', minDice: 0.985 },
    { name: 'mindmap-fast native 2mm (t1_2mm)', opts: { model: 'mindmap-fast', backend: 'webgpu' },
      input: 't1_2mm.nii.gz', ref: 'labels16_native_majority.nii.gz', minDice: 0.985 },
    { name: 'mindmap conformed (t1_crop, legacyCleanup)', opts: { model: 'mindmap', saveConform: true, legacyCleanup: true, backend: 'webgpu' },
      input: 't1_crop.nii.gz', ref: 'labels24.nii.gz', minDice: 0.985 },
    { name: 'mindmap conformed (t1_crop, default cleanup)', opts: { model: 'mindmap', saveConform: true, backend: 'webgpu' },
      input: 't1_crop.nii.gz', ref: 'labels24.nii.gz', minDice: 0.985 },
    { name: 'mindmap native 2mm (t1_2mm, legacyCleanup)', opts: { model: 'mindmap', legacyCleanup: true, backend: 'webgpu' },
      input: 't1_2mm.nii.gz', ref: 'labels24_native_majority.nii.gz', minDice: 0.985 },
    { name: 'mindsnap conformed (t1_crop)', opts: { model: 'mindsnap', saveConform: true, backend: 'webgpu' },
      input: 't1_crop.nii.gz', ref: 'labels104.nii.gz', minDice: 0.98 },
    { name: 'mindsnap native 2mm (t1_2mm)', opts: { model: 'mindsnap', backend: 'webgpu' },
      input: 't1_2mm.nii.gz', ref: 'labels104_native_majority.nii.gz', minDice: 0.98 },
  ]

  for (const [idx, c] of cases.entries()) {
    const key = `case-${idx}`
    const info = await page.evaluate(async ({ origin, input, opts, key }) => {
      const { segment } = await import(`${origin}/index.js`)
      const buf = await (await fetch(`${origin}/${input}`)).arrayBuffer()
      const events = []
      const logs = []
      const res = await segment(buf, {
        ...opts,
        onProgress: (ev) => events.push(ev),
        onLog: (l) => logs.push(l),
      })
      await fetch(`${origin}/${key}`, { method: 'POST', body: res.image })
      return {
        elapsedMs: Math.round(res.elapsedMs),
        backend: res.backend,
        ranInWorker: res.ranInWorker,
        events,
        logs,
      }
    }, { origin, input: c.input, opts: c.opts, key })

    const gotRaw = received.get(key)
    const isGz = gotRaw[0] === 0x1f && gotRaw[1] === 0x8b
    const got = isGz ? gunzipSync(gotRaw) : gotRaw
    const ref = gunzipSync(readFileSync(join(repo, 'test', c.ref)))
    const { dice, fg, diff } = macroDiceAndFg(ref, got, c.binaryMask)
    const planLog = info.logs.find((l) => l.includes('[WebGPU-V4]')) || ''

    check(
      info.backend === 'webgpu' && isGz && dice >= c.minDice,
      `${c.name}: Dice=${dice.toFixed(5)} fg=${fg.toFixed(2)}% diff=${diff} (${info.elapsedMs} ms)`,
      planLog,
    )
    check(
      isMonotonicProgress(info.events),
      `${c.name}: onProgress monotonic (${info.events.length} events)`,
      `stages=[${[...new Set(info.events.map((e) => e.stage))].join(',')}]`,
    )
  }

  // 3. MindGrab mask + borderMm
  {
    const info = await page.evaluate(async ({ origin }) => {
      const { segment } = await import(`${origin}/index.js`)
      const buf = await (await fetch(`${origin}/t1_crop.nii.gz`)).arrayBuffer()
      const res = await segment(buf, { model: 'mindgrab', backend: 'webgpu', mask: true, borderMm: 2 })
      await fetch(`${origin}/mg-mask`, { method: 'POST', body: res.mask })
      return { elapsedMs: Math.round(res.elapsedMs), hasMask: !!res.mask }
    }, { origin })
    const maskNii = gunzipSync(received.get('mg-mask'))
    const intent = new DataView(maskNii.buffer, maskNii.byteOffset, 352).getInt16(68, true)
    let ones = 0
    for (let i = 352; i < maskNii.length; i++) if (maskNii[i] === 1) ones++
    check(
      info.hasMask && intent === 1002 && ones > 500_000,
      `mindgrab mask + borderMm=2: valid binary NIFTI_INTENT_LABEL (${ones} voxels, ${info.elapsedMs} ms)`,
    )
  }

  // 4. Web Worker execution (`worker: true`) + monotonic progress across worker boundary
  {
    const info = await page.evaluate(async ({ origin }) => {
      const { segment } = await import(`${origin}/index.js`)
      const buf = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer()
      const events = []
      const logs = []
      const res = await segment(buf, {
        model: 'mindmap-fast',
        backend: 'webgpu',
        worker: true,
        onProgress: (ev) => events.push(ev),
        onLog: (l) => logs.push(l),
      })
      await fetch(`${origin}/worker-fast`, { method: 'POST', body: res.image })
      return {
        backend: res.backend,
        ranInWorker: res.ranInWorker,
        elapsedMs: Math.round(res.elapsedMs),
        events,
        logs,
      }
    }, { origin })
    const got = gunzipSync(received.get('worker-fast'))
    const ref = gunzipSync(readFileSync(join(repo, 'test', 'labels16_native_majority.nii.gz')))
    const { dice, fg } = macroDiceAndFg(ref, got)
    check(
      info.ranInWorker && info.backend === 'webgpu' && dice >= 0.985 && isMonotonicProgress(info.events),
      `worker: true mindmap-fast native 2mm: Dice=${dice.toFixed(5)} fg=${fg.toFixed(2)}% (${info.elapsedMs} ms, ${info.events.length} progress events)`,
    )
  }

  // 5. Forced webgpuMode: 'low-memory' on mindmap (proves LowMem128 + paced + live layer progress)
  {
    const info = await page.evaluate(async ({ origin }) => {
      const { segment } = await import(`${origin}/index.js`)
      const buf = await (await fetch(`${origin}/t1_crop.nii.gz`)).arrayBuffer()
      const events = []
      const logs = []
      const res = await segment(buf, {
        model: 'mindmap',
        saveConform: true,
        legacyCleanup: true,
        backend: 'webgpu',
        webgpuMode: 'low-memory',
        worker: true,
        onProgress: (ev) => events.push(ev),
        onLog: (l) => logs.push(l),
      })
      await fetch(`${origin}/lowmem-mindmap`, { method: 'POST', body: res.image })
      return {
        elapsedMs: Math.round(res.elapsedMs),
        events,
        logs,
      }
    }, { origin })
    const got = gunzipSync(received.get('lowmem-mindmap'))
    const ref = gunzipSync(readFileSync(join(repo, 'test', 'labels24.nii.gz')))
    const { dice, fg } = macroDiceAndFg(ref, got)
    const planLog = info.logs.find((l) => l.includes('layout=LowMem128') && l.includes('exec=paced')) || ''
    const layerEvents = info.events.filter((e) => e.stage === 'inference' && typeof e.layer === 'number')
    check(
      Boolean(planLog) && dice >= 0.985 && isMonotonicProgress(info.events) && layerEvents.length >= 2,
      `forced webgpuMode='low-memory' mindmap (worker): Dice=${dice.toFixed(5)} fg=${fg.toFixed(2)}% (${info.elapsedMs} ms, ${layerEvents.length} layer events)`,
      planLog,
    )
  }

  // 6. segmentTissues (CAT-lite GM/WM/CSF/Brain) on mindmap (both main thread and worker)
  for (const useWorker of [false, true]) {
    const info = await page.evaluate(async ({ origin, useWorker }) => {
      const { segmentTissues } = await import(`${origin}/index.js`)
      const buf = await (await fetch(`${origin}/t1_crop.nii.gz`)).arrayBuffer()
      const events = []
      const logs = []
      const res = await segmentTissues(buf, {
        model: 'mindmap',
        backend: 'webgpu',
        worker: useWorker,
        onProgress: (ev) => events.push(ev),
        onLog: (l) => logs.push(l),
      })
      for (const k of ['gm', 'wm', 'csf', 'brain']) {
        await fetch(`${origin}/tissues-${useWorker}-${k}`, { method: 'POST', body: res.tissues[k] })
      }
      return {
        backend: res.backend,
        ranInWorker: res.ranInWorker,
        elapsedMs: Math.round(res.elapsedMs),
        events,
        logs,
      }
    }, { origin, useWorker })

    const maps = {}
    for (const k of ['gm', 'wm', 'csf', 'brain']) {
      const raw = gunzipSync(received.get(`tissues-${useWorker}-${k}`))
      const dv = new DataView(raw.buffer, raw.byteOffset, 352)
      assert.equal(dv.getInt16(70, true), 2, `${k}: expected DT_UINT8`)
      const slope = dv.getFloat32(112, true)
      assert.ok(Math.abs(slope - 1 / 255) < 1e-6, `${k}: expected scl_slope=1/255, got ${slope}`)
      maps[k] = raw.subarray(352)
    }
    let gmNz = 0, wmNz = 0, csfNz = 0, brainNz = 0, maxBrainDiff = 0
    for (let i = 0; i < maps.gm.length; i++) {
      const g = maps.gm[i] / 255
      const w = maps.wm[i] / 255
      const c = maps.csf[i] / 255
      const b = maps.brain[i] / 255
      if (g > 0) gmNz++
      if (w > 0) wmNz++
      if (c > 0) csfNz++
      if (b > 0) brainNz++
      const expectedB = Math.min(1, g + w)
      const d = Math.abs(b - expectedB)
      if (d > maxBrainDiff) maxBrainDiff = d
    }
    check(
      info.backend === 'webgpu' &&
      info.ranInWorker === useWorker &&
      gmNz > 100_000 && wmNz > 100_000 && csfNz > 50_000 && brainNz > 200_000 &&
      maxBrainDiff <= 2 / 255 &&
      isMonotonicProgress(info.events),
      `segmentTissues (mindmap, worker=${useWorker}): gm=${gmNz} wm=${wmNz} csf=${csfNz} brain=${brainNz} max|brain-(gm+wm)|=${maxBrainDiff.toFixed(4)} (${info.elapsedMs} ms)`,
    )
  }

  // 7. Empty GPU result refusal + zero buffer leak check (mirrors segment.test.mjs)
  {
    const failed = await page.evaluate(async ({ origin }) => {
      const { segment, acquireDevice } = await import(`${origin}/index.js`)
      const input = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer()
      const device = await acquireDevice('mindmap-fast')
      let live = 0
      const allocate = device.createBuffer.bind(device)
      device.createBuffer = (descriptor) => {
        const buffer = allocate(descriptor)
        const destroy = buffer.destroy.bind(buffer)
        live += buffer.size
        let freed = false
        buffer.destroy = () => {
          if (!freed) { live -= buffer.size; freed = true }
          destroy()
        }
        return buffer
      }
      device.queue.submit = () => {}
      const log = []
      try {
        await segment(input, { model: 'mindmap-fast', backend: 'webgpu', device, onLog: (l) => log.push(l) })
        return { code: 'unexpected-success', live, log }
      } catch (e) {
        return { code: e.code, live, log }
      } finally {
        device.destroy()
      }
    }, { origin })
    check(
      failed.code === 'inference-failed' && failed.log.some((l) => l.includes('segmentation is empty')) && failed.live === 0,
      'empty GPU labels reject with inference-failed and 0 leaked GPU bytes',
      `code=${failed.code} live=${failed.live}`,
    )

    const blank = await page.evaluate(async ({ origin }) => {
      const { segment, acquireDevice } = await import(`${origin}/index.js`)
      const input = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer()
      const device = await acquireDevice('mindgrab')
      device.queue.submit = () => {}
      const log = []
      try {
        await segment(input, { model: 'mindgrab', backend: 'webgpu', device, onLog: (l) => log.push(l) })
        return { code: 'unexpected-success', log }
      } catch (e) {
        return { code: e.code, log }
      } finally {
        device.destroy()
      }
    }, { origin })
    check(
      blank.code === 'inference-failed' && blank.log.some((l) => l.includes('found no brain')),
      'empty GPU MindGrab mask rejects with inference-failed',
      `code=${blank.code}`,
    )
  }

  // 8. Per-model subpath entries (mindgrab.js & mindmap-fast.js)
  {
    const sub = await page.evaluate(async ({ origin }) => {
      const mg = await import(`${origin}/mindgrab.js`)
      const mf = await import(`${origin}/mindmap-fast.js`)
      let mgOther = 'no-error'
      try { await mg.segment(new Uint8Array([1, 2, 3, 4]), { model: 'mindmap', backend: 'webgpu' }) }
      catch (e) { mgOther = e.code }
      const buf = await (await fetch(`${origin}/t1_2mm.nii.gz`)).arrayBuffer()
      const out = await mf.segment(buf, { model: 'mindmap-fast', backend: 'webgpu', worker: true })
      return {
        mgModels: mg.listModels().map((m) => m.name),
        mfModels: mf.listModels().map((m) => m.name),
        mgOther,
        mfWorker: out.ranInWorker,
        mfBytes: out.image.byteLength,
      }
    }, { origin })
    check(
      sub.mgModels.join() === 'mindgrab' &&
      sub.mfModels.join() === 'mindmap-fast' &&
      sub.mgOther === 'unsupported-option' &&
      sub.mfWorker && sub.mfBytes > 352,
      'per-model subpath entries isolate models and run via their own workers',
      JSON.stringify(sub),
    )
  }
} finally {
  await browser.close()
  server.close()
}

console.log(failures ? `\n${failures} check(s) failed` : '\nall WebGPU V4 E2E checks passed')
process.exit(failures ? 1 : 0)
