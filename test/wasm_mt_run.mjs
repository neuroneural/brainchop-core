#!/usr/bin/env node
// Prove the threaded CPU wasm module in real browsers.
//
//   node test/wasm_mt_run.mjs <module.js> <input.nii> <expected.nii>
//
// node already gates this module (`make test-wasm`), and node is the one host
// where none of the hard parts apply: it has SharedArrayBuffer unconditionally,
// no cross-origin isolation to establish, and no nested-worker rule. This
// harness exists for the three things only a browser can answer:
//
//   isolation  a -pthread module imports SHARED memory, which a page without
//              COOP/COEP cannot even instantiate -- so the failure mode is "no
//              CPU fallback at all", not "a slow one". The `noniso` row below
//              asserts that it fails, because a silent serial downgrade would
//              be the pleasant-looking version of a broken deployment.
//   growth     -sALLOW_MEMORY_GROWTH on shared memory needs the engine to
//              support a GROWABLE SharedArrayBuffer, and emcc warns about the
//              pairing. The CPU path allocates ~2.15 GB, so an engine without
//              it fails the `page` row -- growth is a property every row
//              exercises rather than a row of its own.
//   nesting    js/src/worker.ts runs segment() in a Worker, so in the shipping
//              topology emscripten spawns its pool from INSIDE a worker. The
//              `worker` row is the only one that tests that; `page` alone would
//              pass while the product's own arrangement did not.
//
// Every run is compared byte-for-byte against the same native fixture the node
// gate uses, so a browser that produces a plausible volume by a different route
// still fails.
import { existsSync, readFileSync, writeFileSync, mkdtempSync, copyFileSync, rmSync } from 'node:fs';
import { basename, dirname, join, normalize, resolve, sep } from 'node:path';
import { tmpdir } from 'node:os';
import { createRequire } from 'node:module';
import { pathToFileURL, fileURLToPath } from 'node:url';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';

const [modulePath, inputPath, expectedPath] = process.argv.slice(2);
if (!modulePath || !inputPath || !expectedPath) {
  console.error('usage: wasm_mt_run.mjs <module.js> <input.nii> <expected.nii>');
  process.exit(2);
}

const require = (() => {
  const root = [process.env.BC_PLAYWRIGHT_ROOT,
    resolve(dirname(fileURLToPath(import.meta.url)), '..', '..', 'BrowserQC')]
    .filter(Boolean).find((r) => existsSync(join(r, 'node_modules', 'playwright')));
  if (!root) { console.error('playwright not found; set BC_PLAYWRIGHT_ROOT'); process.exit(2); }
  return createRequire(pathToFileURL(join(root, 'package.json')));
})();
const playwright = require('playwright');

// The page and the worker run the SAME driver, so a difference between the two
// rows is a difference in topology and nothing else.
const RUN_TIMEOUT_MS = 600_000;
const DRIVE_JS = `
export async function drive(jsName, tag, initMs) {
  const RUN_TIMEOUT_MS = ${RUN_TIMEOUT_MS};
  const log = [];
  const bytes = new Uint8Array(await (await fetch('./input.nii')).arrayBuffer());
  const factory = (await import('./' + jsName)).default;
  let settle;
  const finished = new Promise((r) => { settle = r; });
  /*
   * The factory is RACED, not merely awaited. Without isolation emscripten's
   * pool spawn fails inside a promise nobody holds --
   *   DataCloneError: SharedArrayBuffer transfer requires self.crossOriginIsolated
   * -- so factory() neither resolves nor rejects and an await on it hangs for
   * good. That is the whole failure mode this harness exists to catch, and it
   * cost 25 minutes of a silent browser before the timeout was added.
   */
  const Module = await Promise.race([
    factory({
      noInitialRun: true,
      thisProgram: jsName.replace(/\\.js$/, ''),
      print: (l) => log.push(l),
      printErr: (l) => log.push(l),
      onExit: (code) => settle(code),
    }),
    new Promise((r) => setTimeout(() => r(null), initMs)),
  ]);
  if (!Module)
    return { status: -1, log: [...log, 'the module never finished initialising'],
      isolated: globalThis.crossOriginIsolated === true };
  Module.FS.writeFile('/in.nii', bytes);
  const t = performance.now();
  // PROXY_TO_PTHREAD: callMain returns before main() finishes. onExit is the end.
  try { Module.callMain(['-o', '/out.nii', '/in.nii']); }
  catch (e) {
    if (!(e && typeof e.status === 'number')) { log.push('threw: ' + (e && e.stack || e)); settle(-1); }
  }
  const status = await Promise.race([finished,
    new Promise((r) => setTimeout(() => r('timeout'), RUN_TIMEOUT_MS))]);
  const ran = Math.round(performance.now() - t);
  if (status === 0)
    await fetch('/result?tag=' + tag, { method: 'POST', body: Module.FS.readFile('/out.nii') });
  return { status, log, ran, isolated: globalThis.crossOriginIsolated === true,
    concurrency: navigator.hardwareConcurrency };
}
`;
const WORKER_JS = `
import { drive } from './drive.js';
self.onmessage = async (e) => {
  try { self.postMessage(await drive(e.data.jsName, e.data.tag, e.data.initMs)); }
  catch (err) { self.postMessage({ status: -1, log: ['worker: ' + (err && err.stack || err)] }); }
};
`;

const stage = mkdtempSync(join(tmpdir(), 'bc-mt-'));
/*
 * Removed in a handler, not at the end of the file: the failure paths below
 * leave through process.exit() or an uncaught rejection, and both would skip a
 * trailing rmSync. rmSync is synchronous, so it is legal in one, and swallowed
 * because a throw from an 'exit' handler is printed as an uncaught exception --
 * and on a natural exit it also turns 0 into 1.
 *
 * The retries are for Windows, where an open .wasm gives EBUSY; here removal
 * takes under a millisecond. They cost ~2 s when they do fire, which is
 * affordable only because this runs after every assertion.
 *
 * Signals are covered by accident, and only while a browser is open: playwright
 * installs its own SIGINT/SIGTERM handlers at launch, and its graceful close
 * rejects the pending evaluate, so the run ends through an ordinary exit that
 * DOES fire this handler -- measured, exit 1 and the directory removed. Before
 * launch and after close, bare node semantics return: no 'exit', and the
 * directory survives. Left that way on purpose; our own signal handler would
 * have to call process.exit() and would fabricate a status on a gate harness.
 *
 * Release mode is the exception that proves it: it launches no playwright
 * browser, so the bare case is the WHOLE run there, and Ctrl-C abandons this
 * AND the Firefox profile below, which is the expensive one.
 */
process.on('exit', () => {
  try { rmSync(stage, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); }
  catch (e) { console.error(`could not remove ${stage}: ${e.message}`); }
});

const jsName = basename(modulePath);
const wasmName = jsName.replace(/\.js$/, '.wasm');
copyFileSync(resolve(modulePath), join(stage, jsName));
copyFileSync(join(dirname(resolve(modulePath)), wasmName), join(stage, wasmName));
copyFileSync(resolve(inputPath), join(stage, 'input.nii'));
writeFileSync(join(stage, 'drive.js'), DRIVE_JS);
writeFileSync(join(stage, 'worker.js'), WORKER_JS);
writeFileSync(join(stage, 'index.html'), '<!doctype html><title>bc-mt</title>');
// A SEPARATE page, not a query parameter on index.html: the playwright rows
// below have been passing against that document, and a self-driving script
// inside it would be running in every one of them for the benefit of a mode
// they never take. It imports the same drive(), so the release run stays the
// same code path as the gate.
writeFileSync(join(stage, 'autorun.html'), `<!doctype html><title>bc-mt autorun</title>
<script type="module">
  const { drive } = await import('./drive.js');
  const record = await drive(${JSON.stringify(jsName)}, 'release-page', 120000);
  // POSTed, not just rendered into a headless document nobody reads.
  await fetch('/result?tag=release-record', { method: 'POST', body: JSON.stringify(record) });
  window.close();
</script>`);

const received = new Map();
const types = { '.html': 'text/html', '.js': 'text/javascript',
  '.wasm': 'application/wasm', '.nii': 'application/octet-stream' };

/*
 * Two servers, differing ONLY in the two headers. Both are needed and both go
 * on the top-level document: COOP alone leaves the page in a browsing-context
 * group it can be reached from, COEP alone does not isolate it, and putting
 * them on the .wasm response instead does nothing at all -- crossOriginIsolated
 * is a property of the document.
 */
function serve(isolate) {
  const server = createServer((req, res) => {
    if (req.method === 'POST' && req.url.startsWith('/result')) {
      const tag = new URL(req.url, 'http://x').searchParams.get('tag');
      const chunks = [];
      req.on('data', (c) => chunks.push(c));
      req.on('end', () => { received.set(tag, Buffer.concat(chunks)); res.writeHead(204); res.end(); });
      return;
    }
    // decodeURIComponent throws URIError on a malformed escape (`GET /%`), and
    // an uncaught throw in this handler takes the whole gate down. The
    // containment check keeps the separator: without it a sibling temp
    // directory whose name merely EXTENDS this one would be servable.
    let raw;
    try { raw = req.url === '/' ? '/index.html' : decodeURIComponent(req.url.split('?')[0]); }
    catch { res.writeHead(400); res.end(); return; }
    const file = join(stage, normalize(raw).replace(/^(\.\.[/\\])+/, ''));
    if (!file.startsWith(stage + sep) || !existsSync(file)) { res.writeHead(404); res.end(); return; }
    const headers = { 'Content-Type': types[file.slice(file.lastIndexOf('.'))] ?? 'application/octet-stream' };
    if (isolate) {
      headers['Cross-Origin-Opener-Policy'] = 'same-origin';
      headers['Cross-Origin-Embedder-Policy'] = 'require-corp';
    }
    res.writeHead(200, headers);
    res.end(readFileSync(file));
  });
  return new Promise((r) => server.listen(0, '127.0.0.1', () => r(server)));
}
const isolated = await serve(true);
const plain = await serve(false);
const url = (s, page = 'index.html') => `http://127.0.0.1:${s.address().port}/${page}`;

const expected = readFileSync(resolve(expectedPath));
let failed = 0;

function niftiEqualIgnoringDescrip(actual, ref) {
  if (!actual || actual.length !== ref.length || actual.length < 352) return false;
  const descrip = actual.subarray(148, 228).toString('ascii').split('\0', 1)[0];
  if (!descrip.startsWith('brainchop ')) return false;
  return actual.subarray(0, 148).equals(ref.subarray(0, 148)) &&
    actual.subarray(228).equals(ref.subarray(228));
}

// Printed as each case finishes rather than collected: a browser that hangs
// should leave behind which rows already passed.
const modes = (process.env.BC_MT_MODES ?? 'page,worker,noniso').split(',');
const firefoxPath = process.env.BC_FIREFOX_PATH;
const tier = process.env.BC_FIREFOX_WASM_TIER;
/*
 * Every knob is checked HERE, before any mode runs. They used to be validated
 * inside the playwright loop, below the release branch's process.exit -- so
 * `BC_MT_MODES=page BC_FIREFOX_PATH=...` silently ignored the modes, and a
 * misspelt tier left you measuring default tiering while believing otherwise.
 */
for (const mode of modes)
  if (!['page', 'worker', 'noniso', 'release'].includes(mode)) {
    console.error(`unknown BC_MT_MODES entry '${mode}'; expected page, worker, noniso or release`);
    process.exit(2);
  }
if (modes.includes('release')) {
  if (modes.length > 1) {
    console.error("BC_MT_MODES=release drives an installed Firefox directly and runs alone");
    process.exit(2);
  }
  if (!firefoxPath) {
    console.error('BC_MT_MODES=release needs BC_FIREFOX_PATH=/path/to/installed/firefox');
    process.exit(2);
  }
  // Neither reaches this mode: it does not go through playwright, so there is
  // no launch to give prefs to and no engine to choose. Accepting them would
  // leave you measuring default tiering in the browser you did not name --
  // which is the exact failure hoisting this validation was meant to end.
  for (const [name, value] of [['BC_FIREFOX_WASM_TIER', tier],
      ['BC_TEST_BROWSER', process.env.BC_TEST_BROWSER]])
    if (value) {
      console.error(`${name} has no effect with BC_MT_MODES=release, which drives ` +
        'BC_FIREFOX_PATH directly rather than through playwright');
      process.exit(2);
    }
}
if (tier && !['baseline', 'optimizing'].includes(tier)) {
  console.error("BC_FIREFOX_WASM_TIER must be 'baseline' or 'optimizing'");
  process.exit(2);
}
if (tier && (process.env.BC_TEST_BROWSER ?? 'chromium') !== 'firefox') {
  console.error('BC_FIREFOX_WASM_TIER is only valid with BC_TEST_BROWSER=firefox');
  process.exit(2);
}

const report = (r) => console.error(
  `  ${r.ok === undefined ? '   ' : r.ok ? 'ok  ' : 'FAIL'} ${r.tag.padEnd(18)}` +
  `${r.ms != null ? `${(r.ms / 1000).toFixed(1)}s ` : ''}${r.note ?? ''}`);

/*
 * The `release` mode: an INSTALLED Firefox, driven without playwright.
 *
 * OPT IN BY MODE, not by the presence of BC_FIREFOX_PATH, and that is a
 * correctness point rather than taste. This branch exits the process, so a
 * BC_FIREFOX_PATH merely EXPORTED in a shell -- which test/webgl2_spike/ also
 * reads, for a different purpose -- turned `make test-wasm-mt` from a three-row
 * gate into a one-row control that exits 0, taking the isolation row with it.
 * Naming the mode makes that impossible to do by accident.
 *
 * The browser cannot be handed to playwright as an executablePath: its Firefox
 * launches with `-juggler-pipe`, which a release build does not implement --
 * verified, it fails at launch. So the page drives ITSELF, through the same
 * drive() the gate rows use and against the same fixture, and POSTs both the
 * volume and its own result record back.
 */
if (modes.includes('release')) {
  const profile = mkdtempSync(join(tmpdir(), 'bc-firefox-profile-'));
  /*
   * Same shape as the staging directory above, for the same reason and one
   * more: measured 150 MB for the documented model16 control and 250 MB for the
   * longer mindgrab one -- it grows with run length -- and a throw anywhere
   * below would miss a cleanup call sitting in finish(). Retried because SIGTERM is asynchronous --
   * Firefox may still be writing, and a file appearing between readdir and
   * rmdir raises ENOTEMPTY, which thrown from a handler would turn a PASSING
   * run into an uncaught exception.
   */
  process.on('exit', () => {
    try { rmSync(profile, { recursive: true, force: true, maxRetries: 5, retryDelay: 200 }); }
    catch (e) { console.error(`could not remove ${profile}: ${e.message}`); }
  });
  const finish = (code) => { isolated.close(); plain.close(); process.exit(code); };
  const started = Date.now();
  const child = spawn(firefoxPath,
    ['--headless', '--no-remote', '--profile', profile,
      url(isolated, 'autorun.html')], { stdio: 'inherit' });
  child.on('error', (e) => {
    console.error(`could not launch BC_FIREFOX_PATH=${firefoxPath}: ${e.message}`);
    finish(2);
  });
  /*
   * Waits for the VOLUME, or for a record that reports failure -- drive() POSTs
   * the volume only on status 0 but the record always, so keying on the volume
   * alone meant a failed run sat here until the 720 s deadline with its own
   * diagnosis already in hand. Twelve minutes late is the wrong way to report
   * the failure this mode exists to investigate.
   */
  const failed = () => {
    if (!received.has('release-record')) return false;
    try { return JSON.parse(received.get('release-record').toString()).status !== 0; }
    catch { return true; }
  };
  const deadline = Date.now() + RUN_TIMEOUT_MS + 120000;
  while (!received.has('release-page') && !failed() &&
      child.exitCode === null && Date.now() < deadline)
    await new Promise((r) => setTimeout(r, 250));
  /*
   * A bounded drain for the record, which is NOT ordered against the volume:
   * drive() awaits the volume POST, whose response is sent from inside the same
   * handler that registers it, so the volume lands before the record is even
   * issued. A 250 ms poll usually covers the gap and nothing enforces it -- and
   * losing that race silently drops the module time, isolated= and
   * hardwareConcurrency=, i.e. the entire output of a mode that exists for
   * diagnosis. Bounded rather than required: if the page dies between the two
   * POSTs, waiting for both would hang to the 720 s deadline.
   */
  const recordBy = Date.now() + 5000;
  while (received.has('release-page') && !received.has('release-record') &&
      Date.now() < recordBy)
    await new Promise((r) => setTimeout(r, 50));
  const completed = Date.now();
  const bytes = received.get('release-page');
  // Bit-exact against the SAME fixture the gate rows use, so this control
  // cannot report a browser as healthy on a volume it got wrong.
  const ok = niftiEqualIgnoringDescrip(bytes, expected);
  // drive()'s own record, POSTed separately. Without it this mode -- the one
  // whose whole purpose is diagnosis -- printed "NO OUTPUT" and nothing else on
  // a failure, discarding the status and the module's log.
  let record = {};
  try { record = JSON.parse(received.get('release-record')?.toString() ?? '{}'); } catch { /* keep {} */ }
  console.error(`firefox release run (${firefoxPath}): ` +
    `${ok ? 'bit-exact' : bytes ? 'OUTPUT DIFFERS' : 'NO OUTPUT'}, ` +
    `${record.ran != null ? `${(record.ran / 1000).toFixed(1)}s in the module, ` : ''}` +
    `${((completed - started) / 1000).toFixed(1)}s including browser startup` +
    `${record.isolated != null ? `, isolated=${record.isolated}` : ''}` +
    `${record.concurrency != null ? `, hardwareConcurrency=${record.concurrency}` : ''}`);
  if (!ok) for (const l of record.log ?? []) console.error(`  ${l}`);
  /*
   * SIGTERM at once, and no pretence of a graceful shutdown. MEASURED on
   * Firefox 153.0.1 / macOS 26: window.close() does not close a window the
   * script did not open (nor with dom.allow_scripts_to_close_windows), and
   * SIGINT is ignored -- so the 30 s "give it time to flush
   * MOZ_PROFILER_SHUTDOWN" wait that used to be here always expired, always
   * ended in SIGTERM anyway, and never produced a capture. A profiler run needs
   * a clean quit; profiles/README.md has the --marionette recipe for that.
   */
  if (child.exitCode === null) child.kill('SIGTERM');
  finish(ok ? 0 : 1);
}

/*
 * Chromium by default, and Firefox deliberately NOT -- which is the opposite of
 * check-glsl, where running both is the point.
 *
 * PLAYWRIGHT'S Firefox -- a patched Nightly -- executes this module about 20x
 * slower than Chromium (125.8 s against 6.1 s for model16 on t1_2mm, same page,
 * same file), so a two-engine default would turn a one-minute gate into a
 * twenty-minute one and would fail on its timeouts rather than on anything
 * about the code.
 *
 * That is a property of the harness browser and NOT of Firefox: installed
 * Firefox 153.0.1 runs the same module and fixture bit-exact in 7.6 s, which is
 * what the BC_FIREFOX_PATH mode above measures. A symbol-mapped Gecko profile
 * shows all 14 pool workers busy and balanced in mn_conv_hidden_rows, so the
 * Nightly executes compute slowly rather than losing parallelism -- do not tune
 * production code against it. Known gaps has the state of it.
 *
 * `BC_TEST_BROWSER=firefox` re-runs the gate on the Nightly for whoever picks
 * that up -- the same variable js/test/segment.test.mjs already uses, rather
 * than a fourth spelling of "which browser".
 */
for (const engine of [process.env.BC_TEST_BROWSER ?? 'chromium']) {
  const name = engine.trim()
  const firefoxUserPrefs = tier ? {
    'javascript.options.wasm_baselinejit': tier === 'baseline',
    'javascript.options.wasm_optimizingjit': tier === 'optimizing',
    // Tiering requires both compilers. With one deliberately disabled, force
    // eager compilation by the remaining compiler rather than leaving Firefox
    // with an impossible baseline-to-optimizing transition.
    'javascript.options.wasm_lazy_tiering': false,
  } : undefined
  const browser = await playwright[name].launch({ firefoxUserPrefs });
  for (const mode of modes) {
    const tag = `${engine}-${mode}`;
    const page = await browser.newPage();
    page.on('pageerror', (e) => report({ tag, note: `pageerror: ${e.message}` }));
    if (process.env.BC_MT_BROWSER_CONSOLE === '1')
      page.on('console', (m) => report({ tag, note: `console.${m.type()}: ${m.text()}` }));
    await page.goto(url(mode === 'noniso' ? plain : isolated), { waitUntil: 'domcontentloaded' });
    // The unisolated page is expected to fail, so it gets a short leash; the
    // real runs get one long enough for a cold module on a slow machine.
    const initMs = mode === 'noniso' ? 20000 : 120000
    const result = await page.evaluate(async ({ jsName, tag, useWorker, initMs, RUN_TIMEOUT_MS }) => {
      try {
        if (!useWorker) return await (await import('./drive.js')).drive(jsName, tag, initMs);
        const w = new Worker('./worker.js', { type: 'module' });
        return await new Promise((r) => {
          w.onmessage = (e) => r(e.data);
          w.onerror = (e) => r({ status: -1, log: ['worker error: ' + e.message] });
          setTimeout(() => r({ status: -1, log: ['the worker never answered'] }), initMs + RUN_TIMEOUT_MS);
          w.postMessage({ jsName, tag, initMs });
        });
      } catch (e) { return { status: -1, log: ['outer: ' + (e && e.stack || e)] }; }
    }, { jsName, tag, useWorker: mode === 'worker', initMs, RUN_TIMEOUT_MS });
    await page.close();

    if (mode === 'noniso') {
      /*
       * The REQUIRED outcome is failure, FOR THE STATED REASON. "status !== 0"
       * alone would also be satisfied by a 404, a typo in the module name or a
       * page that never loaded -- so a broken harness would report this row
       * green and the one thing it exists to prove would go untested. The page
       * must additionally report itself unisolated and fail where the shared
       * memory is set up.
       */
      const why = (result.log ?? []).join(' | ');
      const refused = result.status !== 0 && !received.has(tag) && result.isolated === false &&
        /never finished initialising|SharedArrayBuffer|DataCloneError/.test(why);
      if (!refused) failed++;
      report({ tag, ok: refused, ms: null,
        note: refused ? `refused: ${why.slice(0, 80)}`
          : `expected a shared-memory failure on an unisolated page; got status ` +
            `${result.status}, isolated=${result.isolated}, ${why.slice(0, 60) || 'no output'}` });
      continue;
    }
    const bytes = received.get(tag);
    const ok = result.status === 0 && niftiEqualIgnoringDescrip(bytes, expected);
    if (!ok) { failed++; for (const l of result.log ?? []) console.error(`  ${tag}: ${l}`); }
    report({ tag, ok, ms: result.ran,
      note: ok ? `bit-exact, isolated=${result.isolated}, hardwareConcurrency=${result.concurrency}`
        : `status ${result.status}, ${bytes ? 'output differs' : 'no output'}` });
  }
  await browser.close();
}
isolated.close();
plain.close();

if (failed) { console.error(`wasm_mt_run: ${failed} case(s) failed`); process.exit(1); }
/*
 * Derived from `modes`, never fixed text. With BC_MT_MODES=page this used to
 * print "bit-exact in page and worker, refused without isolation" -- claiming
 * the isolation row, the one that exists to catch a silent serial downgrade,
 * had passed when it had not run at all.
 *
 * Keyed on the SET of modes, not on how many were listed: `page,worker,page`
 * is three entries and two rows, and a count would have printed no disclaimer
 * while the isolation row never ran. Same bug, one indirection along.
 */
const GATE_MODES = ['page', 'worker', 'noniso'];
const ranSet = new Set(modes);
const missing = GATE_MODES.filter((m) => !ranSet.has(m));
const ran = modes.map((m) => ({ page: 'bit-exact in a page',
  worker: 'bit-exact in a worker',
  noniso: 'refused without isolation' })[m]);
console.error(`wasm_mt_run: ${jsName} ${ran.join(', ')}` +
  (missing.length ? ` (BC_MT_MODES=${modes.join(',')}; ${missing.join(' and ')} did not run)` : ''));
