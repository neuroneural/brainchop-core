# Offline HIP artifacts

BrainChop's AMD path separates a heavyweight maintainer compiler from a small
runtime.  Maintainers may use ROCm, MIGraphX, rocMLIR, tinygrad and a target GPU
to search for kernels.  A normal source build consumes only checked-in code
objects and must never invoke that toolchain.

`scripts/hip_artifacts.py` defines and verifies the boundary between those two
parts.  It uses only the Python standard library and needs neither ROCm nor a
GPU:

```sh
python3 scripts/hip_artifacts.py path/to/artifact-directory
make -C src hip-artifacts-check
```

The checked-in catalog contains the launch records consumed by the SDK-free
runtime. Schema/ABI 1 is the legacy label-only contract. Schema/ABI 2 adds an
explicit model PVE capability and common/label/PVE dispatch modes; the loader
continues to accept version 1 artifacts for models that remain label-only.

## Directory and manifest

One directory represents one model and one exact AMD architecture.  It holds a
`manifest.json`, a single `kernels.bin`, and a generated `catalog.h`.  The bundle
is a concatenation of complete HSA code objects; the manifest records each
object's offset, length and hash.  The catalog is the same launch metadata in a
small C-only form for the production loader.  This keeps generated binary bytes
out of C source, avoids one tracked file per kernel, and requires neither a JSON
parser nor Python in a normal build.

The manifest binds the bundle to:

- schema and runtime ABI versions;
- model descriptor hash (weights upload at run time and are not bound;
  legacy `weights_sha256` is accepted and ignored);
- fixed model shape, channels, layers, classes, affine and classifier choices;
- the exact `gfx` architecture and assumed compute-unit, chiplet and wave size;
- ROCm, MIGraphX and tinygrad versions and compile options;
- every source/input hash needed to reproduce the artifact;
- every device buffer and its byte extent;
- each code object's producer, entry point and kernarg ABI;
- dispatch dimensions, arguments, dynamic LDS and execution mode;
- explicit weight tensors, conversions, layouts and destinations.

Unknown fields are rejected.  Changing the contract requires a schema or ABI
decision rather than being silently ignored by an older runtime.

## Checks performed without a GPU

The verifier checks the whole bundle and every individual kernel hash, slice
bounds and overlap.  It decodes the AMDGPU MessagePack note inside each ELF code
object and compares its target, entry point, kernarg size, arguments and static
LDS with the manifest.  Dispatches must reference known kernels, match their
argument counts, and stay within declared buffers.  Weight uploads receive the
same bounds checks.

Static and dynamic LDS are deliberately separate:

- `static_lds_bytes` comes from `.group_segment_fixed_size` in the code object;
  the runtime allocates it automatically.
- `dynamic_lds_bytes` is the extra amount passed to
  `hipModuleLaunchKernel`.

Copying the first number into the second double-allocates LDS, which cut
occupancy and cost ~18% on the fp16 convolution; the distinction is part of
the ABI contract.

These checks establish integrity and launch-contract consistency.  They do not
establish numerical correctness or performance.  Promotion still requires
loading every code object on the named architecture, full-model parity, paired
timing, and memory measurements on physical hardware.

## Portability policy

An HSA code object is selected only for the exact base architecture recorded in
the manifest.  Feature suffixes reported by the driver may be normalized for
matching only after their compatibility is tested; the artifact is not a
forward-compatible intermediate representation.

Cross-compilation may produce a candidate for a device that is not present on
the build host.  Such a candidate is not called optimized or promoted until it
passes the same hardware gates as an on-device exhaustive search.

## Maintainer compiler

`scripts/build_hip_artifacts.py` is the developer side of this boundary.  It
discovers the architecture from `src/models/<name>/model.json` and
`model_meta.json`, so it accepts any model of this layer family and any exact
AMD `gfx` target the installed toolchains support.

For example, on a ROCm development machine:

```sh
python3 scripts/build_hip_artifacts.py \
  --model mindmap \
  --gfx gfx1151 --num-compute-units 40 \
  --output src/hip_artifacts/mindmap/gfx1151
```

MIGraphX 2.15 predates the cross-compilation flags.  When compiling on the
target GPU with that release, add `--native-target`; the target identity and
compute-unit count are still required and recorded in the manifest, but are
not passed to the driver.  Omit the option with a newer MIGraphX when producing
a cross-architecture candidate.

The maintainer environment needs PyTorch with ONNX export support, MIGraphX,
tinygrad 0.14, `hipcc`, ROCm's LLVM libraries, LLVM's `ld.lld`, and access to
the target GPU for tinygrad search and final tuning.  `--migraphx-driver`,
`--migraphx-python-path`, `--rocm-root`, and `--tinygrad-device` make non-default
installations explicit.  `--beam` controls tinygrad's search.  Use `--dry-run`
to inspect the resolved model and every command without importing any of those
packages.

The compiler runs independently cached stages:

1. export the checked-in fp32 `model.bin` and descriptor to network-only ONNX
   (this stage's cache key covers `model.bin`; the manifest does not);
2. compile fp16 convolutions with MIGraphX for the named architecture;
3. extract rocMLIR code objects and their launch descriptors;
4. search and extract tinygrad widen, moments and normalization/GELU kernels;
   label-only models compile `hip_classify.hip` offline with metadata-derived
   dimensions, channels, classes and bias; only PVE models retain the legacy
   full-logit classifier/argmax pipeline;
5. for an explicitly PVE-capable 18-class model, compile the shared
   class-logit-to-tissue-prior HIP kernel; and
6. pack `kernels.bin`, emit the complete manifest, and run the GPU-free
   verifier.

PyTorch export and MIGraphX extraction run in different processes because
PyTorch wheels can carry ROCm libraries that conflict with the system MIGraphX
bindings.  A content hash beside each intermediate makes an interrupted
15--30 minute compile resumable.  Changing the model, architecture, compiler
options, or relevant generator source invalidates only the affected stages.
Adding the PVE support stage does not invalidate cached convolution searches.

MIGraphX benchmark progress is enabled by default.  It reports the number of
candidate configurations and the selected winner for each operation without
dumping the full graph.  A quick-tune or exhaustive-tune problem cache in the
work directory is updated as searches finish, so a timed-out retry resumes
instead of repeating completed searches.  `--compile-timeout` defaults to one
hour per attempt; set it to `0` only for an intentionally unbounded maintainer
run.  `--migraphx-trace-benchmarking` controls progress detail, and
`--migraphx-problem-cache` selects an explicit shared cache.  Quick and
exhaustive searches use separate caches by default so a quick winner cannot
silently bypass a requested promotion search.

MIGraphX's `--gpu-arch`, `--gpu-num-cus`, and `--gpu-num-chiplets` options can
produce a cross-compiled candidate without that physical GPU. The tinygrad
network support-kernel search still executes kernels and therefore belongs on
the target machine. The PVE support kernel itself is a direct,
architecture-exact `hipcc` compile with a content-addressed cache and a
five-minute hard timeout. `--exhaustive-tune` is intentionally opt-in: use it for a
promotion build on matching hardware; a cross-compiled result remains a
candidate until the parity and timing gates below pass.

The extraction boundary uses MIGraphX's current `op().values()` representation,
which is not a stable public serialization API.  It is isolated in
`scripts/extract_hip_kernels.py`; a future MIGraphX change can break generation
but cannot break already checked-in artifacts or normal builds.

## Production runtime

Linux builds include `src/backend_hip_driver.c` when an artifact exists for the
selected model and `HIP_ARCH` (gfx1151 by default).  The code-object bundle is
embedded beside the model weights and the generated catalog is compiled as
ordinary C metadata.  The loader uses `dlopen` for `libamdhip64.so`; it includes
no ROCm headers and links no ROCm, MIGraphX, tinygrad, PyTorch, or Python
library.  A system without the HIP runtime or without an exact architecture
match simply reports the backend unavailable and continues automatic backend
selection.

The checked-in gfx1151 artifacts are measured rocMLIR/tinygrad hybrids. MindMap
also packages the small `hipcc` tissue-prior object and allocates a four-plane
float output only when `--pve` is selected; ordinary inference still runs the
argmax dispatch and downloads byte labels. Every packaged object must be a
linked `ET_DYN` HSA object. `make hip-artifacts-check` verifies manifests and
also regenerates each `catalog.h` in memory to catch drift.


## Bounded label classifier

Newly generated label-only artifacts use `bc_hip_classify` (checked in: mindsnap
only; mindgrab and model16 keep their earlier heads): each thread reads one voxel's
logical fp16 channels, accumulates each class in fp32 (bias first, ascending
channels, explicit FMA), and keeps only a winning score and uint8 class. Strict
`>` retains the first/lowest class on a tie; the initial best is negative
infinity. Features are contiguous **NCDHW**, `features[c * voxels + i]`, and
weights are **[class, channel]** after `tfjs-in-out_to_hip-out-in` upload.
Neither layout matches CUDA/WebGPU's voxel-major head. The unchanged tinygrad
norm/GELU producer constructs contiguous `[1,C,D,H,W]` fp16 output; the last
producer's output range must exactly match the head input when reusing a body.

There is no score buffer or head scratch. Device allocation depends on voxels
and hidden channels, plus the small class/channel weight matrix. The optional
`compile_options.bounded_head` contract uses existing schema/ABI 2: its verifier
requires the fused dispatch, zero head workspace, only the three declared
buffers, no old class-volume argument or obsolete code object, exact head
argument extents, and at most 4 GiB persistent memory. Schema/ABI 1 and ABI 2
artifacts without it remain supported, as do mindmap's PVE pipeline and the
legacy PVE upgrade.

To preserve a validated tuned body while replacing its full label tail:

```sh
python3 scripts/build_hip_artifacts.py --model mindsnap \
  --gfx gfx1151 --num-compute-units 40 --hipcc /opt/rocm/bin/hipcc \
  --work-dir "$PWD/dist/head-build" \
  --reuse-network-artifact /path/to/original/src/hip_artifacts/mindsnap/gfx1151 \
  --reuse-source-root /path/to/original \
  --output src/hip_artifacts/mindsnap/gfx1151
```

The original checkout is required to verify the reused artifact's provenance.
The packer discards the classifier/argmax objects, removes their scratch ranges,
compacts surviving arena intervals (preserving overlapping reduction views),
and copies body code objects byte for byte. It always repacks, so no cached
full-head artifact is reused. The separate head cache includes model/target,
compiler command/version, compiler script and HIP source hashes; the full
artifact source list also includes the head and compiler. The expensive tuned
network cache key stays unchanged.

`BC_HIP_MEMORY=1` prints the high-water sum of **successful actual** runtime
`hipMalloc` allocations (arena, input, selected output); this runtime allocates
all three before inference and frees them afterwards. Driver/private allocations
are excluded from that number. `test/hip_native_check.py` samples process DRM
fdinfo counters at 10 ms and records `/usr/bin/time -v` host maximum RSS
separately. DRM residency includes runtime/driver overhead and is not an exact
split of scratch versus private allocations or a global free-memory estimate.

Maintainer gates, with a disk-backed `BC_TMP` (the GPU host's `/tmp` is tmpfs):

```sh
make -C src CC=gcc CUDA=driver HIP=1 OPENCL=0 ZSTD=0 hip-artifacts-check test
make -C src CC=gcc CUDA=driver HIP=1 OPENCL=0 ZSTD=0 BC_TMP="$PWD/dist/tmp" test-hip
make -C src HIP=1 BC_TMP="$PWD/dist/tmp" test-hip-classifier
make -C src CC=gcc CUDA=driver HIP=1 OPENCL=0 ZSTD=0 BC_TMP="$PWD/dist/tmp" test-hip-no-runtime
make -C src BC_TMP="$PWD/dist/tmp" test-hip-no-toolchain
```

These tests use native executable names, unchanged fixtures/tolerances, real
HIP dispatch evidence, same-run CPU default cleanup/PVE references, exact
forced/auto equality, a warm-up and three timed runs. `t1_crop` is already
256³/1 mm: its native and conformed spaces coincide and both use the 2,048
full-resolution MindSnap budget. The established 512 native-voxel budget is
for `t1_2mm`. Synthetic real-GPU tests additionally
cover fp32 accumulation, negative scores, ties, bias, class 103, final launch
blocks and 15/16/24 logical channels with a poisoned padding plane.

Validated on `gfx1151`. No other AMD architecture is promoted.


## Compatibility tests and harness cleanup

The bounded contract is mandatory when the packaged entry is `bc_hip_classify`.
Legacy full-head artifacts remain valid upgrade inputs at every class count.
The bounded verifier also checks interior arena compaction and rejects private
memory/register spills in the fused head as a promotion guard. Schema-1
`readwrite` annotations on the final producer and head inputs are normalized
when upgrading to schema 2 without changing code objects or launch geometry.

The native harness checks the adapter listing, invalid adapter refusal without
output, and all four PVE maps (`gm`, `wm`, `csf`, `brain`). Each timed inference
runs in its own process session. On timeout or an exception, cleanup sends
SIGKILL to the owned process group and waits for the wrapper, stopping both
`/usr/bin/time` and its inference child. The GPU-free timeout regression uses
an actual wrapper/child pair that ignores SIGTERM. Those process-group tests
are POSIX-specific; the HIP hardware gate remains Linux-specific.


## Memory budgets and allocation errors

`scripts/hip_memory.py` discovers every artifact under `src/hip_artifacts`,
validates its ABI/hashes, and checks each supported execution mode. The same
policy runs on newly promoted builder outputs and on successful hardware
allocation reports. It permits only arena/input/output allocations and requires
actual totals to match the manifest. Legacy artifacts remain valid upgrade
inputs; promotion applies the current budget after replacing the tail.

The base budget is 256 bytes per voxel: 4 GiB at 256³, including weights and
all persistent buffers. A PVE-capable profile adds `classes * voxels * 4` bytes
for its shared logits. PVE execution also adds the difference between the four
float output planes and byte labels (`15 * voxels`). Thus MindMap's existing
labels/PVE modes have distinct budgets without a model-name allowlist. A future
artifact is checked automatically and must fit the policy before promotion.
This is a conservative engineering budget, not a claim about minimum hardware.

On allocation failure the C-only loader reports the buffer/request, total plan,
already allocated memory, Linux host MemAvailable, and an optional
`hipMemGetInfo` free/total snapshot. The host counter is separate because UMA
driver free-memory counters may not reflect host memory pressure.
The snapshot is not a reservation and may change on a shared device. An absent
or failed query does not prevent HIP availability. `BC_HIP_MEMORY=1` success
reports include model and mode so hardware tests can validate every plan.

`python3 test/hip_allocation_check.py --output "$RUN/allocation"` compiles
GPU-free C boundary tests for every artifact/mode and checks each allocation
failure with successful, failed and absent memory queries. On Linux/glibc,
`--real-hip` adds process-local OOM injection while retaining the real HIP
adapter/query path. It does not create device memory pressure. Normal source
builds require none of this test instrumentation.
