# Testing, Benchmarks & WebAssembly

## Tests

```sh
make test           # everything below that runs without a GPU
make test-models    # forced-CPU regressions, bit-exact, plus the CLI surface
make test-guards    # exit-code contracts only
make test-cpu-turbo # default CPU turbo: kernels vs scalar/fp32 oracles, CPU_TURBO_GATE
make test-cpu-sme   # exact SME vs NEON, float for float
make test-cat-lite  # the --pve port against the upstream cat-lite.js
make check-payloads # each model.bin's size matches its generated descriptor
make refresh-fixtures MODEL=<name>  # rewrite CPU fixtures after a deliberate weights change
make test-metal     # Metal/CPU parity on macOS: turbo, fp16 and mindmap --pve
make test-cuda      # CUDA/CPU parity when CUDA is built and a device is present
make test-opencl    # OpenCL/CPU parity when OpenCL is built and a device is present
make benchmark      # timing / memory / agreement matrix, every backend this host can run
```

The model regressions require only Python's standard library; there is no numpy
or other third-party dependency. `make test` additionally runs `test-cat-lite`,
which compares the `--pve` port against the reference implementation and so
needs **node and a `../brainchop-test` checkout** (with `npm ci` run in it; `test-label-resample` needs its `gl-matrix`) — it **skips** loudly,
naming `BC_BRAINCHOP_TEST`, rather than failing. Failing would take `make test`
— the whole safety net, as CI runs only by hand — down for every change
unrelated to PVE. A skip does retire the only gate the partial-volume tissue fit has, so anyone
relying on that gate must check it actually ran. A build whose model list
carries no PVE model says so and passes.

The fixtures are tracked, so the targets run on a fresh clone. Two inputs:
`test/t1_crop.nii.gz`, already 256³ at 1 mm, and `test/t1_2mm.nii.gz`, a real
96×128×128 2 mm INT16 acquisition. CPU reference outputs: `test/bet.nii.gz`
(MindGrab), `test/labels16.nii.gz` (model16 `--save-conform`),
`test/labels16_native_majority.nii.gz` (model16 default, resliced to the input grid), and
`test/labels24.nii.gz` / `test/labels24_native_majority.nii.gz`, the same two cases for
mindmap, and `test/labels104.nii.gz` / `test/labels104_native_majority.nii.gz`
for mindsnap (`labels104_wasm.nii.gz` on wasm, for one FMA-rounded tie).
Model16 uses `labels16_native_majority_wasm.nii.gz` on wasm and x86
for a pre-existing arithmetic difference; all CPU gates remain exact.
The paths are the `TEST_IN` / `TEST_2MM` / `TEST_MG` / `TEST_M16` /
`TEST_M16_NATIVE` / `TEST_MM` / `TEST_MM_NATIVE` / `TEST_MS` / `TEST_MS_NATIVE` variables at the top of the
test section.

Both inputs are needed: `t1_crop.nii.gz` is already on the grid `conform`
targets, so for it the conformed and native output spaces coincide and a test of
that distinction cannot fail. `test/labels16_native_majority.nii.gz` is therefore derived
from `t1_2mm.nii.gz`.

`test-models` also asserts the CLI surface: that each `--help` lists exactly the
options its model supports, and that an option the sibling executable offers is
refused with a nonzero exit rather than ignored.

The exact CPU path (`--no-turbo`) is asserted **bit-exact** against those references, because it is
what every GPU backend is compared against; a tolerance there would only mask CPU
regressions. That assertion has been verified to hold across compilers and
architectures — the references were produced by Apple Clang on arm64, and the
same three 16-channel cases pass bit-exact under GCC 13.3 and Clang 18.1.3 on
x86-64. MindMap's two fixtures also pass bit-exact under GCC 13.3 on the
NVIDIA cluster's x86-64 hosts (2026-09-20), and all seven cases pass under
Clang 22 on Windows/x86-64 (2026-09-30).
Output is also independent of the thread count.

The GPU targets carry tolerances. Metal and WebGPU turbo is gated by
`test/nii_dice.py` at brainchop-test's thresholds (`TURBO_GATE`: macro Dice
0.988, foreground agreement 98.9%; MindGrab 0.995 / 99.5%). The fp16 paths —
Metal and WebGPU under `--no-turbo`, WebGL2 and CUDA — share the same budgets: 64 voxels for MindGrab and 1024 for an
18-class model, out of 16777216. With MindGrab's trained final bias enabled
([PR #29](https://github.com/neuroneural/brainchop-core/pull/29), 2026-10-19),
the measured MindGrab differences are **Metal 35 on Apple M1 and 38 on M4 Pro**,
**WebGPU 33** and **WebGL2 29** on M1 (Chromium and Firefox for WebGL2), 31 and
31 on M4 Pro (2026-10-19), and
**CUDA 38–40** across eight NVIDIA GPU families, with scalar and tensor-core
paths tested. These use the updated `test/bet.nii.gz` reference.

The unchanged model16 measurements are Metal 654, WebGPU 682 and WebGL2 710
(Apple M4 Pro, 2026-08-05), and CUDA 469 (RTX 4070 Ti SUPER). The two browser
backends additionally check model16 in native 2 mm space against a budget of
256, measuring 95 (WebGPU) and 94. None of those budgets has been widened.

mindmap holds the same budgets at 24 channels, on every backend it has, and
the existing backends are gated: on the released weights **Metal 517 / 92**,
conformed and native. On the previous weights WebGPU read 564 / 88, WebGL2
610 / 86 and CUDA 490–505 / 65–74 across the tested GPU types.
`mindsnap` is a 104-way argmax, so it carries its own wider budget
(`MS_TOL` 2048, `MS_TOL_NATIVE` 512): **Metal 1248 / 171**, **WebGPU 1280 / 198**
and **WebGL2 1219 / 184**; **CUDA 919 / 169**. `make test-metal`,
`make test-wasm-gpu` and `make test-wasm-gl` each run seven cases now — the three
16-channel ones plus mindmap's two and mindsnap's two. Before MindGrab's bias
fix, the 16-channel cases remained at 33 / 710 / 94 on WebGL2 and 37 / 682 /
88 on WebGPU while the kernels were generalized beyond a hardcoded 16-wide
stride. Those historical counts verify that the stride change preserved the
16-channel paths; current MindGrab counts are listed above.

OpenCL is fp16 too and holds the same budgets (`OPENCL_TOL_MINDGRAB` 64,
`OPENCL_TOL_MODEL16` 1024). Bias-enabled MindGrab measures 37 differing voxels
on RTX 2080 Ti; model16's earlier measurement is 460 on RTX 4070 Ti SUPER.
Override the variables on the make command line if a different device
legitimately needs more room.

These budgets were set on `test/t1_crop.nii.gz` and are not a promise for every
brain. On three real-world T1s that are not 256³ 1 mm (from niivue-demo-images,
0.74–0.88 mm), GPU and CPU differed in 0.003–0.007% of conformed voxels with
mean label Dice ≥ 0.9991; one image read 1063–1225 for the 18-class models,
slightly above 1024. `TEST_IN=` and `TEST_2MM=` point the test targets at
other images — for example `make test-cat-lite-run TEST_IN=scan.nii.gz` runs the
`--pve` comparison against `cat-lite.js` on your own scan.

model16's and (since PR #41) mindmap's references are generated by this
implementation, so they detect change, not correctness. MindGrab's and
mindsnap's are corroborated: MindGrab's are bit-identical on both bundled fixtures to the
Python brainchop CLI at
[`31150c3`](https://github.com/neuroneural/brainchop-cli/commit/31150c39dae90dba3d18b00d05d32eba273056af)
or later (PR #60, which restores its trained final classifier bias). On the
pre-PR #41 mindmap weights both fixtures agreed with a tinygrad run at **zero**
differing voxels (current `labels24*` are regression-only until re-run) — which
required substituting this model's 5/95 rank normalization for the 2/98
MindGrab form that the reference's `qnormalize()` hardcodes for every model —
and both mindsnap fixtures agree with `scripts/oracle_mindsnap.sh` (tinygrad
`--custom`, this model's own 2/98 form) to 0 and 2 voxels, both fp32 ties.

Historical checks on three further T1s (2026-10-01, before MindGrab's final
bias was restored; niivue-demo-images `register/`: MNI152 1 mm and 2 mm,
ICBM152 nlin asym 55 ext), CPU vs tinygrad (fp32, NV device) differed in
mindgrab 0/0/0, mindmap (`--legacy-cleanup`, conformed) 1/1/1 and mindsnap
0/2/2 voxels. CUDA vs tinygrad: mindgrab 25/3/32, mindmap 877/1096/806,
mindsnap 1490/1630/1471, mean Dice ≥ 0.9989. Wall time: tinygrad 5.3–7.2 s,
CUDA 0.8–1.4 s (mindmap `--pve` 1.6–2.0 s), CPU 4.4–5.7 s; peak host RSS
~490 MB, 285–530 MB and 2.2–3.5 GB respectively.

`make test-cat-lite` is the `--pve` gate and is not a tolerance fitted after the
fact. Both implementations are driven from one `BRAINCHOPC_PVE_DUMP` file — the
blurred, support-gated priors and the normalized intensity, at exactly the
boundary `cat-lite.js` consumes — so the network, the normalization and fp16 are
all upstream of the comparison. The threshold is float32 epsilon because nothing
smaller can mean anything; measured agreement is 5.96e-08, the last bit of the
float the two sides store. It also asserts the fit is thread-count independent.
`BRAINCHOPC_PVE_DEBUG=1` prints the fitted tissue means and sigmas, which the
reference prints too, and is how a divergence is localised to the fit.

Beyond the make targets, `test/baseline.sh` replays a wider 41-case invocation
matrix (`test/baseline_cases.txt`) against a capture directory. It is not wired
to a make target; invoke it directly:

```sh
test/baseline.sh compare modular test/baseline \
  src/brainchop-mindgrab src/brainchop-mindmap-fast
```

The capture directory is untracked, so a fresh clone must `capture` before it can
`compare`. 15 of the 41 cases run over real acquisitions and skip unless
`BC_EXTRA_DIR` points at them. `baseline_cases.txt` names the two original
executables only; mindmap and `--pve` are not in that matrix.

`make test-cuda` and `make test-opencl` print a message and succeed when the
backend is not built, which is the case on macOS by default. A green run there
proves nothing about those backends; they need a Linux or Windows host with the
corresponding device.

`make ubsan` creates a serial CPU-only UndefinedBehaviorSanitizer build, and
`make asan` the same with `-fsanitize=address,undefined`, which on Linux also
enables LeakSanitizer. Run `make clean` afterwards: both leave instrumented
objects behind that a later plain `make` would relink without the sanitizer
runtime. Each is serial (`THREADS=0`) and takes on the order of half an hour per
model, so they are not part of `make test`.

`make tsan-threads` is a third sanitizer target and is unlike those two: it
builds the pool THREADED, because it exists to test the thread pool rather than
the engine, and it runs in seconds. (It does not read `THREADS` at all — the
recipe hardcodes `-DBC_THREADS`, so `THREADS=0` does not make it serial.) It builds `test/threads_race_check.c` under
ThreadSanitizer and asserts both directions — index-disjoint writes through
`bc_parallel_for` must come back clean, and a deliberately planted race must be
caught. The second half is what makes the first mean anything: a clean run is
equally consistent with the detector being unarmed, so the target fails if the
planted race goes unreported. Also not part of `make test`, since it needs its
own instrumented build.

### Benchmark

`make benchmark` runs a fixed matrix — both tracked inputs, every backend the
build has — and prints wall time, peak CPU resident memory, peak GPU memory and
differing voxels against the CPU result. `test/benchmark.py` names its models
literally and now covers all four executables:

```sh
make benchmark                  # print the table
make benchmark-capture          # write test/benchmark_reference.json
make benchmark-compare          # replay and diff against that reference
```

`benchmark-compare` fails only if voxel agreement got worse; timing deltas are
reported but never asserted, because they depend on machine load. Each GPU cell
runs one discarded warm-up (which absorbs the CUDA autotune) and reports the
median of `--repeats` timed runs; the two CPU rows are timed once. The reference
file records the CPU, GPU, compiler and commit it was captured on, since neither
the timings nor the GPU voxel counts are comparable across machines. Captures
for other hosts sit beside it and are selected with `BENCH_REFERENCE=`:
`test/benchmark_reference_m4pro.json` (Apple M4 Pro) and
`test/benchmark_reference_windows.json` (Windows 11, on the same Ryzen 9 7950X
and RTX 4070 Ti SUPER as the default Linux capture).

The matrix includes a `cpuN1` row, which is the CPU backend restricted to a
single thread (`-cpu 1`). Single-core throughput is the figure that
characterises the convolution kernel itself, separately from thread scaling and
memory bandwidth.

A backend that is not compiled in, or has no device, is skipped rather than
reported — so the table shows which backends are healthy on this host, and says
nothing at all about the ones that did not appear.

Header dependencies are tracked (`-MMD -MP` for the host compilers, `-MD` for
`nvcc`, which spells the feature differently and rejects an unknown option
fatally rather than passing it through), and the objects depend on a
`.build-config` stamp covering the compiler, CPU flags, backend selections and
library choices (`BUILD_CONFIG` in `src/Makefile`), so
changing a header, a backend selection or the zlib choice rebuilds what it
should instead of silently linking stale objects. Each model's objects live in
their own `src/build/<model>/` directory, since each is compiled with
different `-D` flags and cannot be shared.

The default release build already enables `-O3`, fast-math, link-time
optimization, dead-section elimination and the pthread pool. CUDA uses `-O3` and
`--use_fast_math`. The Objective-C++ Metal host runner uses `-O3` and LTO;
its generated MSL deliberately uses strict floating-point math for reference
parity.

## WebAssembly and the browser

The same C builds for the browser, with two GPU backends and a threaded CPU one:
**WebGPU** where it exists, a **WebGL2** fallback for the browsers it has not
reached — notably Linux Firefox — and a **pthreads CPU** module for a browser
with neither, which needs the page to be cross-origin isolated. All three are
separate modules, and a caller downloads only the one it can use.

```sh
make wasm-gpu          # the shipping WebGPU module, one .js + .wasm per model
make wasm-gl           # the WebGL2 fallback module, likewise
make wasm              # the threaded CPU module, likewise (THREADS=0 for serial)
make test-wasm-gpu     # seven cases: fp16 within the Metal budgets, turbo within its gates
make test-wasm-gl      # the same seven cases, same budgets, on WebGL2
make test-wasm         # asserts the CPU module bit-exact with the native build
make test-wasm-mt      # the CPU module in a browser: page, worker, and a
                       #   REQUIRED failure on a page without COOP/COEP
make test-wasm-guards  # asserts a .nii.gz input is refused, not misread
make check-wgsl        # compiles the WGSL in a real browser
make check-glsl        # compiles the GLSL in Chromium and Firefox
```

WebGPU is about twice as fast — on an Apple M4 Pro, 2026-08-04, MindGrab takes
1.63 s in the module against WebGL2's 3.35 s, and the 18-class model 0.91 s
against 1.84 s, both fp16 — and both meet the same fp16 voxel budgets as the
Metal backend, unwidened. WebGPU's default turbo path cuts mindmap's GPU time to
about 1.1 s from 1.9 s (`noTurbo` / `--no-turbo` restores fp16). End to end, including the host stages the module does not time, the
same machine measures 1.52 s for model16 on WebGPU and 2.50 s for mindmap,
against roughly 12.3 s for the JavaScript brainchop page this replaces on an M1.
The WebGPU convolution runs 64x2 workgroups, or 256x1 on an AMD adapter
(output bit-identical; `BC_WG_CONV_PRESET` forces either).

mindmap pays extra on WebGL2 specifically, and the reason is structural rather
than the extra arithmetic: a 24-wide channel stride puts the two-voxel blocking
out of reach, because two z slices at once would want twelve draw buffers
against the eight browsers report. It therefore runs one voxel per fragment —
5.5 s against model16's 2.0 s, an end-to-end pair measured together. Those
are macOS figures; the WebGL2 fallback is known to run on Linux Mesa stacks in
both Firefox and Chrome, but has not been timed there. WebGL2 also runs
**synchronously**, so it blocks the thread it is called on for the whole
segmentation; run it in a Web Worker (below) if the page must stay responsive.

The threaded CPU module runs MindGrab in about 10 s on 14 cores against ~1 s on
WebGPU, and is **bit-exact** with the native executables rather than held to a
voxel budget. The JS `auto` backend takes it after hardware WebGPU and
WebGL2, and never picks an identified software rasteriser (SwiftShader,
llvmpipe, lavapipe, WARP, a WebGPU fallback adapter), which is ~100× slower
still; a masked WebGL2 renderer runs only when CPU cannot, and with none of
them it throws `no-webgpu`. Every backend refuses an empty MindGrab mask
("the network found no brain"), as label models refuse empty labels. It needs
`Cross-Origin-Opener-Policy: same-origin` and
`Cross-Origin-Embedder-Policy: require-corp` on the page, because a pthreads
module imports shared memory. 20× slower than Chromium, but installed Firefox 153.0.1 ran the model16 /
`t1_2mm` case bit-exact in 7.6 s — level with Chromium once browser startup is
excluded.

mindmap and mindsnap are the models that strain the address space: 24
channels need about 3.2 GB of heap against wasm32's 4 GB ceiling, so each is
bit-exact under node on a machine that can supply that and fails at the memory
growth on one that cannot.

The module handles **uncompressed NIfTI only** — no zlib is linked, so gzip is
the wrapper's job. The wasm tests need `emcc`; `test-wasm-gpu`, `test-wasm-gl`,
`test-wasm-mt`, `check-wgsl` and `check-glsl` additionally need a checkout with
playwright installed (`BC_PLAYWRIGHT_ROOT`). The GPU ones run **headed**,
because headless Chromium on macOS has no GPU; `test-wasm-mt` needs no GPU and
runs headless. None of them are part of `make test`.

`js/` is published to npm as
[`@brainchop/core`](https://www.npmjs.com/package/@brainchop/core)
(`npm i @brainchop/core`), combining a pure-JS WebGPU V4 engine (`dist/models/*.bin`)
with Emscripten WebGL2 and threaded-CPU WASM fallbacks behind
`segment(buffer, {model}) -> {image, mask?, elapsedMs, backend, ranInWorker}`.
It probes for WebGPU, then WebGL2, then cross-origin isolation before fetching
anything, or `backend: 'webgpu' | 'webgl2' | 'cpu'` names one explicitly —
honoured or refused, never silently downgraded. `worker: true` runs the whole
segmentation in a Web Worker, which is the recommended way to call it from a
page whichever backend runs. It owns the things the WASM fallback modules deliberately do
not: gzip, device and context acquisition, the isolation probe, and the decision
to refuse. The worker, model binaries, and fallback modules are imported by literal specifiers, so Vite,
Rollup and webpack emit them and nothing needs copying.
A host running one model can import
`@brainchop/core/<model>` to deploy only that model's assets.
`MODELS[name].colormap` carries each model's label names and colours. Hosts that
serve `dist/` themselves can still point `assetPath` at it. See
[js/README.md](../js/README.md).

The package carries **all four** models on all three backends, and
`js/test/segment.test.mjs` gates seven fixtures on each of them. The
one thing 24 channels changed on the JavaScript side is the WebGPU device
request: it is now `max(1 GiB, 2 x one activation)` per model rather than a flat
1 GiB, which is unchanged at 16 channels and 1.6 GiB at 24. The flat request did
not report an error at 24 — the renderer died, and playwright said only that the
page had closed.

## Licensing

The project is MIT; see `LICENSE` and `THIRD_PARTY_NOTICES.md`, which also
covers libwinpthread (static in Windows executables) and the model weights.
MIT permits including BSD-2-Clause material as long as the notice travels with
it, which is what the niimath section of `THIRD_PARTY_NOTICES.md` is for: the
EDT in `preprocess.c` is a port of niimath's, and `conform.c` reproduces its
conform, comply and reference-grid reslice.
The current `bwlabel.c` is a small independent queue-based 26-connected flood
fill, not the Andersson/SPM labeling implementation in niimath. Therefore the
Andersson source notice is not required for this file. If that upstream
implementation is copied back in later, its BSD notice must accompany it.
