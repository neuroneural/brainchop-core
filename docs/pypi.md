# Python Packaging & Adaptive Hardware Selection

`brainchop` distributes prebuilt portable wheels (`win_amd64`, `macosx_arm64`,
`linux_x86_64`) plus a zero-dependency PEP 517 source package (`sdist`) that
builds the four native runtimes locally on any supported platform. Prebuilt
wheels deliver full native GPU performance (`Metal`, `CUDA`, `HIP`) while
compiling CPU kernels for portable ISAs (`AVX2+FMA` on x86-64, baseline `arm64`
on macOS); on CPU-only hosts with newer SIMD extensions (e.g., Apple SME or
AVX-512 VNNI), `pip install --no-binary brainchop brainchop` builds with
host-specific CPU flags for faster CPU inference. The distribution version
(`packaging/PYPI_VERSION`) and native runtime version (`src/version.h`)
are synchronized at **1.1.1**, succeeding `1.1.0`, `1.0.0`, and the legacy
Python/tinygrad `brainchop==0.2.5` release.

## Installation, shared caches and first use

A fresh source build detects usable devices, prunes the developer-prepared GPU
artifacts, and compiles the small C/C++ runtime locally. Metadata preparation
never probes hardware or invokes a compiler. Both the build backend and installed
adapter use only Python's standard library: no Python dependencies, ML frameworks,
nvcc, hipcc, BEAM search, kernel generation or build-time downloads.

**Pip can reuse a locally built wheel without running our build backend.**
Its wheel tags identify OS, CPU architecture and Python ABI, not GPU architecture
or AVX2/FMA. There is no standard post-install hook, and a package cannot tell
pip to reject a cache entry based on the GPU. This affects shared NFS home
caches, login-node installs, containers and hardware changes. See the official
[build interface](https://pip.pypa.io/en/stable/reference/build-system/) and
[cache rules](https://pip.pypa.io/en/stable/topics/caching/).

To handle this without special pip flags, the public launchers check current
hardware. If the installed runtime lacks a usable supported GPU target or is
incompatible with the CPU/libc, **first use automatically rebuilds from a bundled,
compressed source archive**, then runs the requested inference. This is the
explicitly approved change from the original installation-only-build proposal.
No network or GPU compiler is involved. An ordinary native toolchain must still
be available on that node. A failed rebuild produces an actionable error and
build-log path; it does not silently downgrade a supported GPU to CPU.

The cache is under `$XDG_CACHE_HOME/brainchop/runtimes` (normally
`~/.cache/brainchop/runtimes` on Linux, `~/Library/Caches/brainchop/runtimes` on
macOS, `%LOCALAPPDATA%/brainchop/runtimes` on Windows). Keys include source hash,
CPU capabilities, OS/libc and OS version, detected GPU architectures/driver
information and build policy; an OS version change therefore rebuilds once. Native tuning separately retains its device/driver/model/code
identity. OS file locks and atomic publication coordinate concurrent first
callers, including on NFS filesystems supporting advisory locks. Subsequent
calls verify and reuse the completed runtime without a compiler. Intact cache
hits also work when the cache is read-only. New hardware requires writable cache
space; missing build tools or unwritable storage are reported as errors.

Only applicable artifacts are embedded in executable files. The compressed
source archive necessarily retains the full candidate artifact inventory so a
later hardware change can rebuild offline. That is an explicit size/pruning
tradeoff, not a claim that unrelated bytes are absent from the whole wheel.
No training checkpoints or ML toolchains are retained. Cache entries from older
sources remain separate and are not automatically deleted while another process
may use them; users may remove unused `brainchop/runtimes` entries. Pip uninstall
removes installed files but, like other application caches, leaves this cache.

Driver loss and unsupported GPU architectures use the native error/fallback
behavior. Explicit GPU requests fail if unusable; automatic mode may use CPU
when no supported GPU is available. A supported GPU appearing after a CPU build
triggers adaptation. CPU-only machines do not receive repeated pip warnings.
Local Linux wheels use `linux_<machine>`, without a manylinux claim. The launcher
can rebuild for a different libc, including recompiling its baseline probe when
the old probe cannot execute. This does not make cross-OS or cross-CPU-family
wheels portable; pip's normal platform restrictions still apply.

### Optional fixed-target deployment builds

Ordinary `pip install brainchop` needs no settings. Deployments that know their
compute architecture can instead build a fixed wheel without seeing that GPU:

```sh
pip wheel . --config-settings brainchop.target=sm89
pip wheel . --config-settings brainchop.target=sm80,sm89
BRAINCHOP_TARGET_ARCH=gfx1151 pip wheel .
```

`brainchop.target` accepts `auto` (default), `cpu`, `all`, or a comma-separated
selection of `sm60`, `sm70`, `sm75`, `sm80`, `sm86`, `sm89`, `sm90`, `gfx1151`.
`all` intentionally embeds every available target for the current OS, including
OpenCL; it is an opt-in size tradeoff. CUDA targets require Linux/Windows and
HIP requires Linux. Fixed GPU selections **stay fixed at runtime**: moving an
sm89 wheel to sm75 will not retarget it. Its native unavailable-backend behavior
still applies. CPU/libc incompatibility may rebuild while preserving that fixed
GPU selection. Build settings affect builds, not already cached pip wheels;
use the resulting explicit wheel for deployment when fixing an architecture.

Fixed-target builds default to baseline CPU instructions for portability.
`brainchop.cpu=baseline` also requests that policy in automatic builds;
`brainchop.cpu=auto` enables AVX2/FMA when the build host supports them. A
config setting overrides `BRAINCHOP_TARGET_ARCH`. Unknown `brainchop.*` settings
are errors; unrelated frontend settings are ignored. No runtime environment
variable overrides a wheel's recorded policy.

## Prerequisites and hardware selection

64-bit Python 3.9+ and pip are required. Baseline x86-64 CPUs are supported;
AVX2/FMA is an optimization, not an installation requirement. Pip cannot install
system compilers or GPU drivers. Tools are needed during a source build and
again only if first use must adapt the native runtime.

| Platform | Native build prerequisites | Selection |
|---|---|---|
| macOS | Xcode Command Line Tools (C/C++, make, SDK, system zlib) | Metal on usable Apple Silicon devices; CPU otherwise |
| Linux | GNU make, GCC or Clang with LTO linker, binutils (`objdump`), libc/pthread/zlib development files (Ubuntu: `build-essential zlib1g-dev`) | CUDA Driver API, HIP gfx1151, OpenCL, CPU |
| Windows x64 | MSYS2 UCRT64 GCC and zlib, MSYS make; default root `C:\msys64`, or set `MSYS2_ROOT` for a custom installation | CUDA Driver API, OpenCL, CPU |

CUDA, HIP and OpenCL dynamically load driver entry points; no GPU SDK headers or
link libraries are needed. OpenCL supports MindGrab and MindMap Fast only.
Driver compilation of prepared OpenCL/Metal source and bounded launch
calibration remain native runtime work. OpenCL loader loss does not break CPU
or CUDA execution, and no duplicate executables are needed.

Detection uses a small baseline C probe with isolated vendor subprocesses;
CUDA/HIP primary contexts must be usable. A broken vendor ICD cannot suppress
another backend. All usable CUDA architectures are selected as a union. Native
adapter indices and `-a`/`-backend` remain available. HIP currently has only the
validated **gfx1151** profile; other AMD architectures are not declared tuned.

CUDA uses checked-in 12.8 cubins and driver API 12080 or newer. Selection retains
an exact target or highest compatible lower minor within the same major from
sm60/70/75/80/86/89/90. PTX and unrelated runnable images are removed byte-for-byte;
new major architectures are not assumed supported through PTX JIT.

Weights, inference math and cleanup remain unchanged. ARM64 uses its baseline
hardware FMA; capable x86 hosts use AVX2/FMA. Baseline x86 uses existing SSE2 code.
The strict FMA fixtures remain unchanged. The baseline MindSnap conformed test
allows the separately measured one-voxel rounding difference out of 16,777,216;
it does not reject an installation or change production inference. Other
baseline model fixtures remain exact. These are tested fixture budgets, not a
claim that every possible input will differ by at most one voxel.

CUDA/OpenCL's existing small automatic calibration and warm-cache reuse are
preserved. Pruned artifact hashes separate their calibration caches naturally.
HIP and Metal gain no new tuning stage.

## CLI contract

The `brainchop` CLI and Python API default to **`mindmap`** (18-class cortical
and subcortical segmentation), matching the standalone native dispatcher. It
ships only `mindgrab`, `mindmap`, `mindsnap`, and `mindmap-fast`.
`16chan18cls` and `model16chan18cls` are exact aliases for `mindmap-fast`.
Historical `tissue_fast`, `robust_tissue`, `subcortical`, `DKatlas`, `aparc50`,
and arbitrary custom checkpoints are rejected; sharing a class count is not
checkpoint equivalence.

```
brainchop scan.nii.gz
brainchop scan.nii.gz -m mindmap --save-conform
brainchop scan.nii.gz --skull-strip --mask mask.nii.gz
brainchop scan.nii.gz -m mindmap --pve
```

The supported historical spellings, batch naming, default mask filename,
skull-strip precedence, CT/compliance flags, masks/borders, raw-class filenames
and error exits are retained. Output defaults to the **original input voxel
grid** across both the Python wrapper and standalone native executables
(`--save-conform` / `save_conform=True` keeps conformed 256³ outputs;
`--inverse-conform` is still accepted as a no-op compatibility flag). `mindgrab`
always writes the input grid. Native label restoration uses the existing
majority resampling contract, whereas the historical Python CLI calls niimath
nearest-neighbor restoration; this adapter does not promise identical
inverse-resampling voxels to every legacy checkpoint. Frozen native fixtures
remain the numerical gate.

`--beam`, `--no-optimize` and `--batch-size` are accepted with an explanation:
there is no expensive user search, and multiple inputs run sequentially.
`--update` describes updating the package, because weights are embedded.
`mindmap`'s `--pve` writes experimental GM/WM/CSF/Brain partial-volume tissue
fractions, not a substitute for the retired tissue models. Unsupported
model/option combinations fail explicitly. A zero-dependency `import brainchop`
scripting API (`Volume`, `load`, `segment`, `save`, `list_models`,
`export_classes`) is included using only the Python standard library (`numpy`
optional).

Individual tools: `brainchop-mindgrab`, `brainchop-mindmap`,
`brainchop-mindsnap`, and `brainchop-mindmap-fast`. The package exposes only
`brainchop-mindmap-fast` for the lightweight checkpoint; the internal C build
identifier remains unchanged. Prebuilt macOS Apple Silicon wheels are tagged
`macosx_14_0_arm64` per PEP 425 (which requires minor version `0` on macOS 11+
and rounds `MACOSX_DEPLOYMENT_TARGET=13.2` up to `14_0`); macOS 13.2–13.x users
can install via Homebrew, the standalone `.pkg`, or a source wheel build. Thin
Python entry points locate only this environment's binaries and protect optional
CPU instructions before launch.

## Reproducible validation

```
python3 -m unittest discover -s test -p 'test_pypi*.py'
PYTHONPATH=packaging python3 -c 'import brainchop_build as b; print(b.build_sdist("dist"))'
python3 test/pypi_lifecycle.py --sdist "dist/brainchop-$(cat packaging/PYPI_VERSION).tar.gz" --output /tmp/bc-install --upgrade
/tmp/bc-install/cold/bin/python test/pypi_installed.py
/tmp/bc-install/cold/bin/python test/pypi_gpu.py --backend metal --output /tmp/bc-metal
```

The lifecycle harness provides an isolated localhost simple index and cache;
users do not need those test settings. It resolves the package by name, creates
fresh virtual environments, checks cold/warm installs, and uninstalls the warm
copy. The 0.2.5 upgrade uses `--no-deps` to test file ownership and replacement
without installing its retired ML dependencies. Pip does not automatically
remove orphaned dependencies already installed by an older release.

The sdist includes source and candidate artifacts for different build hosts.
The wheel contains the compressed source capsule, four selected model
executables, baseline hardware probe, adapter, build record and licenses.
Runtime cache entries contain only selected binaries and the build record.
Build records contain hashes, sizes, exact selections and diagnostics.
