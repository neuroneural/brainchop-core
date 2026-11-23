CFLAGS ?= -O3
# Packaging selects only CPU instructions proved usable by its baseline probe.
CPUFLAGS ?=
CFLAGS += $(CPUFLAGS)
CFLAGS += -std=c11 -Wall -Wextra -Wpedantic -ffunction-sections -fdata-sections \
	-ffast-math -fno-finite-math-only -flto
CPPFLAGS += -DHAVE_ZLIB

# Dependency generation: emit a .d file per object listing its headers, so
# editing a header rebuilds the objects that include it. Without this, changing
# a struct relinks stale objects with a mismatched ABI, which is silent memory
# corruption.
#
# Kept OUT of CPPFLAGS, which is the one thing this must not be. CPPFLAGS is
# forwarded verbatim to the nvcc rule, and nvcc rejects an unknown option
# FATALLY rather than passing it to the host compiler:
#
#   nvcc fatal   : Unknown option '-MMD'
#
# so a flag that only gcc understands turns the whole CUDA build off. nvcc
# spells the same feature -MD (--generate-dependencies-with-compile) and has no
# -MP at all. Two variables rather than one, each given only to the compiler
# that accepts it.
#
# The missing -MP on the CUDA object costs one thing and it is worth naming:
# -MP emits a phony target per header, so that DELETING a header makes the
# stale .d harmless instead of a "No rule to make target" that can only be
# cleared by hand. backend_cuda.o has no such protection; `make clean` is the
# recovery if a header it references is ever removed.
BC_DEPFLAGS ?= -MMD -MP
BC_NVCC_DEPFLAGS ?= -MD
LDFLAGS += -flto

# zlib, and an OPT-IN faster one.
#
# The default is the system zlib (-lz), present on every platform this builds
# for: macOS ships /usr/lib/libz.1.dylib in the dyld shared cache, Linux has it,
# and MSYS2's UCRT64 provides it. Nothing needs installing, and nothing changes
# for anyone who does not set ZLIBNG_ROOT. This was never an availability
# question -- only a speed one.
#
# ZLIBNG_ROOT links a static zlib-ng instead, which is markedly faster at
# DEFLATE (inflate is unaffected) and emits a slightly smaller file. project notes's
# *Faster gzip, opt-in* OWNS the measurements and the size cost -- they are not
# restated here, because that file says in as many words not to, and an audit
# caught this block breaching that rule one commit after it was written.
#
# WHY IT IS NOT THE DEFAULT, which is the question a reader has at this ifdef:
# test/baseline.sh compares output files by SHA-256, so a different deflate
# implementation strands every .nii.gz case in a suite whose capture step is a
# deliberate refusal. `make test` is unaffected -- it compares decompressed
# content.
#
# FOUR THINGS THAT ARE NOT OBVIOUS AND BELONG HERE:
#
#  - IT MUST BE A ZLIB_COMPAT BUILD -- a drop-in libz.a plus zlib.h, which is
#    what the two wildcards below look for. An INSTALLED zlib-ng (Homebrew,
#    distro packages) does NOT work: those are the non-compat flavour with
#    zng_-prefixed symbols in zlib-ng.h, so they are not drop-in for
#    nifti_io.c's <zlib.h>. Build one with zlib-ng's own configure, no CMake:
#
#      cd /path/to/zlib-ng && ./configure --zlib-compat && make
#      cd /path/to/brainchopC && make -C src ZLIBNG_ROOT=/path/to/zlib-ng
#
#    Two separate checkouts; running `make -C src` from inside zlib-ng finds no
#    src/ and fails.
#
#  - ON macOS THE DEPLOYMENT TARGET MUST BE **EXPORTED**, not prefixed:
#
#      export MACOSX_DEPLOYMENT_TARGET=13.2
#      ./configure --zlib-compat && make
#
#    A `MACOSX_DEPLOYMENT_TARGET=13.2 ./configure ...` prefix binds to configure
#    alone; zlib-ng does not record it, so `make` then compiles at the host's
#    default. Measured: prefixed gives archive members at minos 26.0, exported
#    gives 13.2. This documentation previously gave the prefixed form while the
#    figures beside it had been taken with the exported one.
#
#    What that costs is WARNINGS ONLY, and this block previously claimed
#    otherwise: 50 x `ld: warning: object file was built for newer 'macOS'
#    version`, after which the link succeeds and the executable's own minos is
#    13.2, because the linker takes it from MACOSX_DEPLOYMENT_TARGET and not
#    from archive members. scripts/package_macos.sh's minos assertion therefore
#    PASSES -- measured. Nothing catches a mismatched archive, which is why the
#    warnings are worth heeding rather than filtering.
#
#  - THE SELECTION IS PART OF .build-config AND THE ARCHIVE IS A LINK
#    PREREQUISITE. Both are load-bearing rather than tidy: without the first,
#    `make ZLIBNG_ROOT=...` over an existing tree reports everything up to date
#    and silently keeps the system zlib; without the second, rebuilding zlib-ng
#    in place leaves the old code linked. The header needs neither -- -MMD puts
#    the resolved zlib.h in the .d file, because -I paths count as user headers.
#
#  - TWO SHARP EDGES LEFT UNFIXED, neither reachable by accident: the -I below
#    puts zlib-ng's source root on the include path for every translation unit,
#    so a future src/ header named deflate.h, inflate.h or zutil.h would be
#    shadowed silently (checked: no collision among zlib-ng's headers today);
#    and a ZLIBNG_ROOT containing a space breaks $(wildcard) and reports the
#    "no libz.a" error misleadingly.
#
ifdef ZLIBNG_ROOT
BC_ZNG_LIB := $(firstword $(wildcard $(ZLIBNG_ROOT)/libz.a $(ZLIBNG_ROOT)/lib/libz.a))
BC_ZNG_INC := $(firstword $(wildcard $(ZLIBNG_ROOT)/zlib.h $(ZLIBNG_ROOT)/include/zlib.h))
ifeq ($(BC_ZNG_LIB),)
$(error ZLIBNG_ROOT is set but no libz.a is under $(ZLIBNG_ROOT) or $(ZLIBNG_ROOT)/lib -- run `./configure --zlib-compat && make` in the zlib-ng source)
endif
ifeq ($(BC_ZNG_INC),)
$(error ZLIBNG_ROOT is set but no zlib.h is under $(ZLIBNG_ROOT) or $(ZLIBNG_ROOT)/include -- headers and libz.a must come from the same ZLIB_COMPAT build)
endif
# zlib.h alone is not enough -- it includes zconf.h, and copying just the two
# obvious files out of a zlib-ng tree otherwise fails with ~20 `unknown type
# name 'Z_EXTERN'` errors that name neither the missing file nor this option.
ifeq ($(wildcard $(dir $(BC_ZNG_INC))zconf.h),)
$(error ZLIBNG_ROOT: $(dir $(BC_ZNG_INC)) has zlib.h but no zconf.h beside it -- point ZLIBNG_ROOT at the zlib-ng build tree, not at copied files)
endif
# Catch a plain zlib pointed at by mistake. It has the same file names and the
# same API, so it would link happily and simply be no faster -- silent, which is
# the failure mode this project keeps getting caught by. zlib-ng's ZLIB_COMPAT
# zlib.h still identifies itself in its banner; stock zlib does not.
#
# Spelled `grep -q ... || echo no` rather than `grep -lc`: BSD grep prints "0"
# for a non-matching file rather than nothing, so the obvious `ifeq ($(shell
# grep -lc ...),)` is never true and the guard silently never fires. That was
# the first version of this check, and it took running it to notice.
ifneq ($(shell grep -q zlib-ng $(BC_ZNG_INC) 2>/dev/null || echo no),)
$(error ZLIBNG_ROOT: $(BC_ZNG_INC) does not look like zlib-ng -- a stock zlib would link and be no faster, so this is refused rather than silently accepted)
endif
# The archive is named by full path rather than -L/-lz, so there is no way for a
# system libz to be picked up instead. (An earlier comment here also claimed the
# include path "goes first so it wins over any system zlib.h" -- it does not,
# CPPFLAGS is appended to, and the claim was removed rather than made true
# because nothing in this build supplies a competing -I.)
CPPFLAGS += -I$(dir $(BC_ZNG_INC))
LDLIBS += $(BC_ZNG_LIB) -lm
# Named in .build-config below, so switching zlib forces a rebuild, and made a
# link prerequisite, so rebuilding zlib-ng in place forces a relink. Without the
# first, `make ZLIBNG_ROOT=...` over an existing tree reports everything current
# and silently keeps the system zlib -- measured, and the reason the option is
# not merely a flag.
BC_ZLIB_ID := $(BC_ZNG_LIB)
BC_ZLIB_DEP := $(BC_ZNG_LIB)
else
LDLIBS += -lz -lm
BC_ZLIB_ID := system
endif

# .nii.zst I/O (HAVE_ZSTD in nifti_io.c) when pkg-config finds libzstd; ZSTD=0
# disables. It links dynamically, so the release packagers pass ZSTD=0: their
# audits accept only system libraries. The resolved flags and version go into
# .build-config, so a different libzstd forces a rebuild.
ZSTD ?= $(if $(shell pkg-config --exists libzstd 2>/dev/null && echo 1),1,0)
ifeq ($(ZSTD),1)
BC_ZSTD_CFLAGS := $(shell pkg-config --cflags libzstd)
BC_ZSTD_LIBS := $(shell pkg-config --libs libzstd)
BC_ZSTD_ID := $(shell pkg-config --modversion libzstd) $(BC_ZSTD_CFLAGS) $(BC_ZSTD_LIBS)
CPPFLAGS += -DHAVE_ZSTD $(BC_ZSTD_CFLAGS)
LDLIBS += $(BC_ZSTD_LIBS)
endif

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# ------------------------------------------------------------------- Windows
#
# There is no Windows `uname`, so this is always being read through a POSIX
# shell layer, and which one it is decides what the toolchain can do:
#
#   MINGW64_NT / UCRT64_NT / CLANG64_NT   MSYS2's mingw shells. The compiler
#                                         emits NATIVE Windows binaries with no
#                                         runtime dependency on MSYS2. This is
#                                         the supported configuration.
#   MSYS_NT                               MSYS2's POSIX-emulation shell. gcc
#                                         there links msys-2.0.dll and produces
#                                         a Cygwin-like binary. It builds, but
#                                         it is not what to ship.
#   CYGWIN_NT                             Same story as MSYS_NT.
#
# Git Bash also reports MINGW64_NT, but ships no compiler -- it is a shell, not
# a toolchain -- so `make` there fails at CC, which is the correct outcome.
#
# Everything Windows needs from this file is a consequence of two facts: the
# linker appends .exe, and `/dev/null` is a shell fiction that a native compiler
# invoked from that shell cannot open as an output file.
# One % per pattern, and it must be the only one: make treats the FIRST % in a
# filter pattern as the wildcard and every later one as a literal percent sign,
# so the obvious-looking `MINGW%_NT%` asks for a name ending in '%' and matches
# nothing. That failure is silent -- BC_WINDOWS simply stays empty and every
# Windows branch below switches itself off -- which is why the patterns are cut
# short here rather than spelled out in full.
BC_WINDOWS := $(if $(filter MINGW% UCRT% CLANG% MSYS% CYGWIN%,$(UNAME_S)),1,)

# Native Windows linkers append .exe whether or not -o asked for it, so a target
# named without the suffix is remade on every invocation: make stats
# `brainchop-mindgrab`, never finds it, and relinks. Naming the target what the
# linker actually writes is what makes the dependency graph work at all. Empty
# everywhere else, so no other platform sees any change.
EXEEXT := $(if $(BC_WINDOWS),.exe,)

# Where the probes below put the executable they do not want. A REAL file, on
# every platform, because both of the obvious bit buckets fail on Windows and
# they fail in the way that costs the most: the probe reports "this compiler
# does not accept the flag" rather than "the probe itself is broken".
#
#   /dev/null   The probing compiler is a native Windows program that does not
#               go through MSYS2's path translation, so it cannot open it.
#   nul         Resolves fine -- and then mingw-w64's ld fails with
#               `final link failed: file truncated`, because it writes the
#               output, seeks back and rewrites headers, and NUL discards
#               writes while reporting a size of 0. MSVC's linker tolerates
#               this, which is why `nul` looks correct until it is tried.
#
# The cost of getting this wrong is silent and specific: -march=x86-64-v3 gets
# rejected, the build falls back to baseline SSE2, and the CPU backend runs at
# roughly half speed with only a warning to say so. That is the 2x this file
# spends fifty lines explaining, thrown away by a probe artifact.
#
# $(EXEEXT) so the linker writes the name we then delete. The second probe is
# no longer the OpenMP one -- pthreads needs no runtime package, so that reason
# is gone -- but a link probe is still required; see the CC block below.
BC_PROBE_ISA := .bc-probe-isa$(EXEEXT)
BC_PROBE_CC := .bc-probe-cc$(EXEEXT)

# Where the test recipes put their intermediates. On Unix this is /tmp, exactly
# as it has always been. On Windows it MUST be relative, and the reason is a
# genuinely silent trap rather than a matter of taste.
#
# MSYS2 rewrites arguments that look like absolute POSIX paths when it spawns a
# NATIVE child, so `-o /tmp/bc-mg-test.nii.gz` reaches brainchop-mindgrab.exe as
# C:\msys64\tmp\bc-mg-test.nii.gz. The very next line hands the same spelling to
# python as part of a `-c` script -- one long argument that does not look like a
# path, so it is passed through untouched, and native Python resolves /tmp
# against the current drive as C:\tmp. The executable writes one file and the
# assertion reads a different one, so a passing test would depend on the two
# directories happening to hold the same stale content.
#
# A relative path removes the question: no rewriting applies to it, and both
# programs resolve it against the same working directory. It also stops the
# suite scribbling into a shared /tmp, though that is a side benefit and not
# why it is here.
# The rule that creates it is deliberately NOT here: make takes its default goal
# from the first non-dot target it reads, so a rule this early would make `make`
# with no arguments build a directory.
BC_TMP := $(if $(BC_WINDOWS),build/tmp,/tmp)

METAL ?= $(if $(filter Darwin,$(UNAME_S)),1,0)
# Turbo (Metal, WebGPU) cone skip, turbo_cone.h; 0 computes every row.
TURBO_CONE ?= 1
TINYGRAD_PYTHON ?= python3
MINDGRAB_BEAM ?= 3
MODEL16_BEAM ?= 3
MINDMAP_BEAM ?= 3
MINDSNAP_BEAM ?= 3
METAL_MANIFEST_DIR ?= /tmp
# Linux/Windows builds embed the offline modules without probing the toolkit.
# CUDA=1 is the explicit nvcc/cudart validation path; CUDA=0 excludes CUDA.
NVCC ?= nvcc
CUDA ?= $(if $(or $(filter Linux,$(UNAME_S)),$(BC_WINDOWS)),driver,0)
ifeq ($(filter $(CUDA),0 1 driver),)
$(error CUDA must be 0, 1 (legacy nvcc), or driver)
endif
# HIP uses checked-in HSA code objects through dlopen. It needs no ROCm SDK at
# build time, so Linux can include it by default; a machine without
# libamdhip64.so simply reports no HIP adapters and continues backend selection.
HIP ?= $(if $(filter Linux,$(UNAME_S)),1,0)
HIP_ARCH ?= gfx1151
ifeq ($(filter $(HIP),0 1),)
$(error HIP must be 0 or 1)
endif
# CUDA_ARCH builds ONE architecture, which is what you want while iterating:
# roughly 2 s per model instead of 10, because every one of the convolution's
# 18 template variants is compiled once per architecture.
#
# Left empty, the build emits a fatbin instead, and that is the default because
# the alternative is worse than slow -- it is a binary that will not start. PTX
# JITs FORWARD only, so the old single `-arch=sm_75` default produced something
# that ran on Turing and newer and died on anything older with "no kernel image
# is available for execution on the device". Real cubins for sm_60/70/75/80/89
# cover Pascal through Ada (a cubin runs on later MINOR revisions of its own
# major, so sm_80 serves sm_86/87 and sm_60 serves sm_61/62), and compute_90
# PTX is the forward-compatible tail for Hopper and anything after it.
#
# sm_60 is the floor by code, not by choice: backend_cuda.cu uses
# atomicAdd(double *), which does not exist below it.
#
# The list is PROBED against nvcc, not hardcoded, for the same reason the CC
# rule below probes clang rather than trusting `command -v`: an unsupported
# -gencode is fatal, so a fixed list turns a working toolkit into a build
# failure. CUDA 13 removed Maxwell, Pascal and Volta, so a hardcoded
# compute_60/70 would fail to build on precisely the toolkit a brand-new card
# needs -- the opposite of the portability this is for. compute_89/90 likewise
# do not exist before CUDA 11.8. Intersecting our wish list with
# `nvcc --list-gpu-arch` keeps old and new toolkits both building, each
# covering as much hardware as it can.
CUDA_ARCH ?=
ifeq ($(CUDA),1)
CUDA_WANT_ARCH ?= 60 70 75 80 89 90 100 120 121
ifeq ($(strip $(CUDA_ARCH)),)
BC_NVCC_ARCH := $(subst compute_,,$(shell $(NVCC) --list-gpu-arch 2>/dev/null))
BC_CUDA_ARCH := $(filter $(CUDA_WANT_ARCH),$(BC_NVCC_ARCH))
# Highest supported architecture, kept as PTX so the build still runs on cards
# newer than the toolkit knows about. Sorted NUMERICALLY by the shell, which is
# neither of the two obvious shortcuts: make's own $(sort) is lexical, so a
# three-digit Blackwell arch would sort before the two-digit ones and 90 would
# win; and nvcc's listing order cannot be trusted either, because CUDA 13.0
# really does print compute_110 ahead of compute_103.
BC_CUDA_TOP := $(shell printf '%s\n' $(BC_CUDA_ARCH) | sort -n | tail -1)
ifeq ($(strip $(BC_CUDA_ARCH)),)
# The probe found nothing usable (very old nvcc without --list-gpu-arch, or a
# toolkit supporting none of the wish list). Fall back to nvcc's own default
# rather than guessing, so the build works and the user gets a plain binary.
CUDA_GENCODE ?=
else
CUDA_GENCODE ?= $(foreach a,$(BC_CUDA_ARCH),-gencode arch=compute_$(a),code=sm_$(a) ) \
	-gencode arch=compute_$(BC_CUDA_TOP),code=compute_$(BC_CUDA_TOP)
endif
else
CUDA_GENCODE ?= -arch=$(CUDA_ARCH)
endif
endif # legacy CUDA architecture probing
OPENCL ?= $(if $(filter Darwin,$(UNAME_S)),0,$(if $(shell pkg-config --exists OpenCL 2>/dev/null && echo 1),1,0))
ifeq ($(UNAME_S),Darwin)
MACOSX_DEPLOYMENT_TARGET ?= 13.2
export MACOSX_DEPLOYMENT_TARGET
endif
# Multithreading, via the pthread pool in bc_threads.c. THREADS=0 builds the
# whole engine serial, which is what the sanitizer targets want. This was called
# OMP while the pool was OpenMP; see the pthread block below for why it is not.
THREADS ?= 1

# Refuse the old spelling rather than ignoring it. `make OMP=0` was the
# documented way to get a serial build for two years and appears in older notes
# and shell history; silently handing back a THREADS=1 build would be the
# quietest possible way to invalidate someone's measurement. An error costs one
# line and names the replacement.
ifneq ($(origin OMP),undefined)
$(error OMP= is no longer used; threading is pthreads, not OpenMP. Use THREADS=$(OMP) instead)
endif

# `CC ?=` cannot work here: make predefines CC=cc as a default, which ?= treats
# as already set. Override only when the value is make's own default, so an
# explicit CC= on the command line or in the environment still wins.
#
# The OpenMP link probe that used to live here is gone; the project's *Compiler
# selection, and the CI break it caused* records what it was for and why
# removing it was safe. It also changed which compiler Windows picks, which is
# the part to know before editing this block.
#
# A LINK PROBE IS STILL REQUIRED, for a different reason than the OpenMP one.
# This build passes -flto, and on MSYS2/UCRT64 clang cannot link that without
# -fuse-ld=lld, because GNU ld's LTO plugin wants an LLVMgold the clang package
# does not install. A bare `command -v clang` would make plain `make` select a
# configuration KNOWN NOT TO LINK on Windows, and it fails at the very END of a
# build rather than at its start. So clang is probed, and one that cannot link
# falls back instead of being selected.
#
# -flto is what is probed because it is the flag that actually breaks. Written
# literally rather than referencing CFLAGS, which is assembled below this point;
# if -flto ever leaves the build this probe becomes harmlessly conservative
# rather than wrong.
#
# The output goes to a REAL FILE that is then deleted, never to `nul` or
# `/dev/null`: this file records that mingw-w64's ld writes the output, seeks
# back and rewrites headers, and NUL discards writes while reporting size 0, so
# the link fails with `file truncated` and the probe silently reports failure.
#
# The FALLBACK is probed too, for Windows' sake: MSYS2's mingw toolchains
# install `gcc` and no `cc` at all, so the unconditional `cc` this used to name
# was a command that does not exist there -- and because make's own default for
# CC is also `cc`, the failure surfaced as a bare "cc: command not found" that
# looked like a broken PATH rather than a Makefile assumption.
#
# THE SECOND PROBE adds -fuse-ld=lld and is the one that makes clang reachable
# on Windows. It runs ONLY when the first failed, which is what keeps it inert
# on Linux and macOS: there the bare -flto probe succeeds, so nothing below the
# first branch is ever evaluated and neither CC nor LDFLAGS moves. Apple clang
# ships no lld at all, so the flag must not be added unconditionally.
#
# It is written as a second probe rather than as -fuse-ld=lld on the first
# because the flag has to be EARNED: a clang whose LTO works with the default
# linker should keep using it, and a box with neither working must still fall
# back to gcc rather than be handed a linker it does not have.
# Require the POSIX clock/pthread ABI too: MSVC clang can link an empty
# main but cannot compile this runtime. Keep MinGW clang eligible.
ifeq ($(origin CC),default)
BC_CLANG_OK := $(shell command -v clang >/dev/null 2>&1 && \
	clang -flto -pthread compiler_probe.c -o $(BC_PROBE_CC) 2>/dev/null && \
	rm -f $(BC_PROBE_CC) && echo 1)
BC_CC_FALLBACK := $(if $(shell command -v cc 2>/dev/null),cc,gcc)
ifeq ($(BC_CLANG_OK),1)
CC := clang
else
BC_CLANG_LLD_OK := $(shell command -v clang >/dev/null 2>&1 && \
	clang -flto -fuse-ld=lld -pthread compiler_probe.c -o $(BC_PROBE_CC) 2>/dev/null && \
	rm -f $(BC_PROBE_CC) && echo 1)
ifeq ($(BC_CLANG_LLD_OK),1)
CC := clang
# Both phases: -flto makes the LINK the step that needs the plugin, and it is
# the link that was failing. Appended here rather than at the top of the file so
# that a toolchain which never needed it never sees it.
LDFLAGS += -fuse-ld=lld
else
CC := $(BC_CC_FALLBACK)
# Warn like the -march probe below, for the same reason: this fails for causes
# that are not "clang is absent", and landing silently on gcc costs 1.61x.
ifneq ($(shell command -v clang >/dev/null 2>&1 && echo 1),)
$(warning clang is installed but both POSIX/LTO probes failed, so building \
with $(CC). A compatible compiler and LTO linker are required -- see docs/building-and-testing.md)
endif
endif
endif
endif
# ponytail: an explicit `make CC=clang` skips the probe above and so gets no
# -fuse-ld=lld. Left alone: the failure is a link error naming the missing
# linker, and `make CC=gcc` or installing lld fixes it. Probe it too if that
# turns out to bite someone.

# ------------------------------------------------------------------- x86 SIMD
#
# The x86-64 baseline ISA is SSE2: 128-bit vectors and, decisively, NO fused
# multiply-add. MN_DEFINE_CONV is an FMA-bound kernel -- it is tuned against the
# NEON FMA roofline on arm64 -- so compiling it for baseline x86-64 emits every
# a*b+c as a separate mulps and addps at half the vector width. Measured on a
# Ryzen 9 7950X, alternating A/B, 32 threads, t1_crop: MindGrab 10.24 s -> 4.66 s
# (2.20x) and model16 6.17 s -> 3.18 s (1.94x) simply by raising the target.
# Instruction counts tell the same story without a clock: 0 vfmadd and 0 ymm
# references at baseline against 336 and 1570 at v3.
#
# x86-64-v3 is a portable ISA LEVEL, not -march=native: AVX2 + FMA + BMI, which
# means Haswell (2013) and Zen 1 (2017) and later. That is the floor this buys
# the speed with, and a binary built here will SIGILL on anything older, so
# X86_ISA= (empty) restores the portable baseline and X86_ISA=x86-64-v2 is the
# no-FMA middle ground. v4 (AVX-512) is deliberately not the default: narrow
# hardware support and it often costs clocks.
#
# Probed rather than assumed, because -march=x86-64-v3 needs GCC 11 / Clang 12;
# older compilers reject the name and would fail the build outright. This must
# sit after CC is chosen, since it is that compiler which has to accept it.
#
# Verified bit-exact: both models reproduce test/bet.nii.gz and
# test/labels16.nii.gz exactly with FMA contraction enabled. That is not
# guaranteed a priori -- an FMA rounds once where a multiply-then-add rounds
# twice -- so re-check `make test` if this default is changed.
X86_ISA ?= x86-64-v3
ifeq ($(UNAME_M),x86_64)
ifneq ($(strip $(X86_ISA)),)
BC_ISA_OK := $(shell printf 'int main(void){return 0;}' | \
	$(CC) -march=$(X86_ISA) -x c - -o $(BC_PROBE_ISA) 2>/dev/null && \
	rm -f $(BC_PROBE_ISA) && echo 1)
ifeq ($(BC_ISA_OK),1)
CFLAGS += -march=$(X86_ISA)
else
$(warning this compiler does not accept -march=$(X86_ISA); building for \
baseline x86-64, which has no FMA and is about 2x slower on the CPU backend)
endif
endif
endif

# Threading is pthreads, not OpenMP, and the difference is mostly this block.
#
# What this replaced, and what dropping it was worth in bytes, is in the project's
# *Threading is pthreads now*, which owns those figures -- they were restated
# here and in bc_threads.h and went stale in both. The short version: macOS
# needed Homebrew's libomp, so `git clone && make` did not work on a clean Mac,
# and pthreads is in libSystem.
#
# -pthread rather than -lpthread: it also sets the feature macros the platform
# wants at COMPILE time (_REENTRANT and friends), and on macOS it links nothing
# extra because libSystem already has it. Passed to both compile and link, since
# the flag means different things in each phase.
ifneq ($(THREADS),0)
CPPFLAGS += -DBC_THREADS
CFLAGS += -pthread
LDFLAGS += -pthread
endif

# ---------------------------------------------------------------------- models
#
# One block per model. Adding a model means adding a block here and dropping its
# directory into models/. The names live in the Makefile rather than being read
# out of model_meta.json so that an ordinary build still needs no Python.
#
# Each executable compiles the shared sources with -DBC_MODEL_IS_<NAME>, which
# binds that model's generated constants to the generic BC_MODEL_* names. Only
# the selected model's generated Metal program is compiled, which is where most
# of the per-model size saving comes from.
MODELS = mindgrab model16chan18cls mindmap mindsnap

# NOT every model in MODELS. scripts/build_cuda_modules.py carries its own
# hardcoded model list and regenerating a fatbin needs nvcc pinned to a specific
# version plus cuobjdump, so a model can legitimately exist here with no CUDA
# module. DERIVED from the checked-in artifacts rather than restated, because a
# second hand-maintained list is a second thing to forget: a model has a CUDA
# backend exactly when its fatbin is present.
CUDA_MODELS = $(basename $(notdir $(wildcard cuda_modules/*.fatbin)))
# Which models have an artifact for HIP_ARCH decides -DHAVE_HIP per model, so
# it belongs in .build-config: dropping a newly built artifact into an existing
# tree must rebuild that model, or backend.c keeps its table without bc_hip_backend
# and the executable links the driver while reporting the backend unavailable.
HIP_MODELS = $(notdir $(patsubst %/$(HIP_ARCH)/catalog.h,%,$(wildcard hip_artifacts/*/$(HIP_ARCH)/catalog.h)))

exe_mindgrab = brainchop-mindgrab$(EXEEXT)
metal_hdr_mindgrab = metal_mindgrab_generated.h
metal_sym_mindgrab = mindgrab
metal_uc_mindgrab = MINDGRAB
sel_mindgrab = BC_MODEL_IS_MINDGRAB

exe_model16chan18cls = brainchop-mindmap-fast$(EXEEXT)
metal_hdr_model16chan18cls = metal_model16_generated.h
metal_sym_model16chan18cls = model16
metal_uc_model16chan18cls = MODEL16
sel_model16chan18cls = BC_MODEL_IS_MODEL16CHAN18CLS

exe_mindmap = brainchop-mindmap$(EXEEXT)
metal_hdr_mindmap = metal_mindmap_generated.h
metal_sym_mindmap = mindmap
metal_uc_mindmap = MINDMAP
sel_mindmap = BC_MODEL_IS_MINDMAP

exe_mindsnap = brainchop-mindsnap$(EXEEXT)
metal_hdr_mindsnap = metal_mindsnap_generated.h
metal_sym_mindsnap = mindsnap
metal_uc_mindsnap = MINDSNAP
sel_mindsnap = BC_MODEL_IS_MINDSNAP

# The weight file each executable embeds, and the float count its generated
# descriptor says that file should hold. Read straight out of model_def.h with
# sed so an ordinary build still needs no Python; check-payloads asserts the two
# agree before anything is assembled, because .incbin will happily embed a file
# of the wrong size and the result is a plausible-looking wrong segmentation
# rather than a crash. wc -c rather than stat: stat's size flag differs between
# macOS and Linux. awk rather than sed: a sed script matching '^#define' would
# put a '#' in the recipe, which make reads as the start of a comment and which
# swallows the closing paren of the $(shell ...) call.
bin_mindgrab = models/mindgrab/model.bin
bin_model16chan18cls = models/model16chan18cls/model.bin
floats_mindgrab := $(shell awk '$$2 == "BC_MINDGRAB_WEIGHT_FLOATS" {print $$3; exit}' models/mindgrab/model_def.h)
floats_model16chan18cls := $(shell awk '$$2 == "BC_MODEL16_WEIGHT_FLOATS" {print $$3; exit}' models/model16chan18cls/model_def.h)
bin_mindmap = models/mindmap/model.bin
floats_mindmap := $(shell awk '$$2 == "BC_MINDMAP_WEIGHT_FLOATS" {print $$3; exit}' models/mindmap/model_def.h)
bin_mindsnap = models/mindsnap/model.bin
floats_mindsnap := $(shell awk '$$2 == "BC_MINDSNAP_WEIGHT_FLOATS" {print $$3; exit}' models/mindsnap/model_def.h)

# OpenCL's 16-wide ceiling is a per-model fact. CUDA uses a separate,
# model-specific module/geometry contract and now covers all three models.
chan_mindgrab := $(shell awk '$$2 == "BC_MINDGRAB_CHAN" {print $$3; exit}' models/mindgrab/model_def.h)
chan_model16chan18cls := $(shell awk '$$2 == "BC_MODEL16_CHAN" {print $$3; exit}' models/model16chan18cls/model_def.h)
chan_mindmap := $(shell awk '$$2 == "BC_MINDMAP_CHAN" {print $$3; exit}' models/mindmap/model_def.h)
chan_mindsnap := $(shell awk '$$2 == "BC_MINDSNAP_CHAN" {print $$3; exit}' models/mindsnap/model_def.h)
# 1 when the model fits the 16-wide OpenCL backend.
narrow_model = $(if $(filter 0,$(shell test $(chan_$(1)) -le 16 && echo 0)),1,0)

check-payloads:
	@$(foreach m,$(MODELS), \
		test -n "$(floats_$(m))" || { \
			echo "$(m): no WEIGHT_FLOATS in model_def.h" >&2; exit 1; }; \
		actual=$$(wc -c < $(bin_$(m)) | tr -d ' '); \
		expected=$$((4 * $(floats_$(m)))); \
		[ "$$actual" = "$$expected" ] || { \
			echo "$(bin_$(m)) is $$actual bytes, but model_def.h describes $$expected" >&2; \
			exit 1; }; \
		if command -v sha256sum >/dev/null 2>&1; then \
			got_sha=$$(sha256sum $(bin_$(m)) | cut -c1-8); \
		elif command -v shasum >/dev/null 2>&1; then \
			got_sha=$$(shasum -a 256 $(bin_$(m)) | cut -c1-8); \
		else got_sha=""; fi; \
		if [ -n "$$got_sha" ]; then \
			hdr_sha=$$(awk '/$(sel_$(m))/ {getline; gsub(/"/, "", $$3); print $$3; exit}' model_weights_sha.h); \
			[ "$$got_sha" = "$$hdr_sha" ] || { \
				echo "$(bin_$(m)) sha256 prefix is $$got_sha, but model_weights_sha.h has $$hdr_sha" >&2; \
				exit 1; }; \
		fi; \
		echo "payload $(m): $$actual bytes = $(floats_$(m)) floats"; \
	)

# Everything except the per-model weight-layout adapters.
# Every source is shared: there are no per-model adapters left. What differs
# between the executables is the -D flags they are compiled with and the payload
# they link.
COMMON_SOURCES = brainchopc.c cli.c preprocess.c backend.c backend_accel.c \
	backend_cpu.c bc_threads.c cat_lite.c conform.c meshnet_cpu.c model_specs.c bwlabel.c \
	nifti_io.c
ifeq ($(METAL),1)
CPPFLAGS += -DHAVE_METAL -DBC_TURBO_CONE=$(TURBO_CONE)
LDLIBS += -framework Foundation -framework Metal -lc++
endif
ifneq ($(filter 1 driver,$(CUDA)),)
CPPFLAGS += -DHAVE_CUDA
endif
ifeq ($(CUDA),driver)
ifeq ($(UNAME_S),Linux)
LDLIBS += -ldl
endif
endif
ifeq ($(HIP),1)
ifeq ($(UNAME_S),Linux)
LDLIBS += -ldl
endif
endif
ifeq ($(CUDA),1)
# Every substitution here is quoted, and on Windows that is load-bearing rather
# than tidy: nvcc's default location is
# "C:\Program Files\NVIDIA GPU Computing Toolkit\...", and an unquoted
# $(command -v nvcc) word-splits on the space. dirname then receives FIVE
# arguments instead of one and prints a line per argument, which $(shell) joins
# back together with spaces, so CUDA_HOME came out as the nonsense
#
#   / . . . Toolkit/CUDA/v13.3
#
# and the link failed naming a path nobody wrote. It degrades rather than
# erroring because dirname succeeds on every one of those five fragments.
CUDA_HOME ?= $(shell dirname "$$(dirname "$$(command -v $(NVCC))")")
ifeq ($(BC_WINDOWS),1)
# The host compiler nvcc drives, and the one that builds the two C files that go
# into the CUDA DLL beside it. Named rather than assumed so a different Build
# Tools install can be pointed at; it must be on PATH, which means this build
# has to be run from a vcvars64 environment (nvcc needs cl.exe regardless).
BC_MSVC_CC ?= cl
# Windows names the 64-bit import libraries lib/x64, not lib64, and there is no
# rpath: cudart64_NN.dll is found through PATH, which the toolkit installer
# already sets. CUDA_HOME here is nearly always
# "/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/vNN.N", so every use of it
# must be quoted -- make itself splits on the space, and an unquoted -L would
# silently become two bogus search paths rather than an error.
#
# Deliberately no -lstdc++ here. On Linux that pulls in the GNU C++ runtime the
# .cu object was built against; on Windows the .cu object is built by nvcc
# through MSVC, so it wants the MSVC C++ runtime and adding MinGW's would mix
# two of them. Whatever that object still needs is an ABI question, not a
# library-list one -- see the WINDOWS CUDA note below.
LDFLAGS += -L"$(CUDA_HOME)/lib/x64"
# The CUDA runtime is named by its FULL PATH rather than with -lcudart, and that
# is deliberate on Windows for two independent reasons.
#
# First, -l does not find it. The toolkit ships MSVC import libraries --
# lib/x64 holds cudart.lib and cudart_static.lib -- and `-lcudart` makes GNU ld
# search for libcudart.a / libcudart.dll.a, neither of which exists. Worse, this
# build links -static, so ld reports the miss as
#
#   cannot find -lcudart: No such file or directory
#   have you installed the static version of the cudart library ?
#
# which invites exactly the wrong fix. cudart_static.lib IS present, and linking
# it would drag the MSVC CRT into a MinGW executable to no purpose: the DLL is
# what the driver stack expects, and the installer already puts it on PATH.
#
# Second, naming the file sidesteps the -Wl,-Bdynamic / -Wl,-Bstatic pairing the
# OpenCL block needs. An explicit input file is not subject to the -B search
# mode at all, so there is no mode to restore afterwards and no risk of the
# static/dynamic split silently flipping for whatever gcc appends next.
#
# Quoted because CUDA_HOME contains a space ("Program Files") and make splits on
# it; unquoted, the -L above would become two bogus search paths rather than an
# error.
#
# It goes to the DLL LINK ONLY, not to LDLIBS. The executable contains no CUDA
# code on Windows -- backend_cuda.o lives in brainchop-cuda-<model>.dll instead --
# so adding cudart here would give every executable an import of
# cudart64_NN.dll that nothing uses, and Windows resolves imports EAGERLY at
# process start. A machine with no CUDA runtime would then fail to launch the
# binary at all, for a CPU-only or OpenCL run that never needed it.
BC_CUDART_LIB := "$(CUDA_HOME)/lib/x64/cudart.lib"
else
LDFLAGS += -L$(CUDA_HOME)/lib64
# Bake in the runtime search path when the toolkit ships its own libcudart,
# which is the case for anything installed under /usr/local/cuda-<version>.
# Distribution packaging (Ubuntu's nvidia-cuda-toolkit) puts libcudart on the
# system loader path instead and has no lib64 here, so this adds nothing there.
# Without it, building against a side-by-side toolkit links cleanly and then
# fails at startup with "libcudart.so.NN: cannot open shared object file",
# which reads as a broken install rather than a missing rpath.
ifneq ($(wildcard $(CUDA_HOME)/lib64/libcudart.so),)
LDFLAGS += -Wl,-rpath,$(CUDA_HOME)/lib64
endif
LDLIBS += -lcudart -lstdc++
endif
# Warning flags nvcc forwards to the HOST compiler, which is gcc or clang on
# Unix and cl.exe on Windows. cl reads a leading '-' as '/', so the Unix spelling
# does not merely fail to apply: -Wextra arrives as the unknown option /Wextra
# (warning D9002, ignored) while -Wall arrives as cl's /Wall, which is "every
# warning including the ones Microsoft's own headers trip" and buries the build
# in thousands of lines. W3 is what the Unix pair actually asks for.
#
# Spelled -W3 and NOT /W3, which matters and is not a style choice. MSYS2
# rewrites any argument that looks like an absolute POSIX path when it spawns a
# NATIVE child, and nvcc.exe is one, so `/W3` is helpfully translated into
# `C:/msys64/W3` before nvcc ever sees it. The failure surfaces from a compiler
# two levels down, naming a path nobody wrote:
#
#   c1xx: fatal error C1083: Cannot open source file: 'C:/msys64/W3'
#
# cl.exe accepts '-' and '/' interchangeably as its option prefix, so the dash
# form is the same flag with nothing for MSYS2 to mistake for a path. The same
# trap applies to every future /Zi, /GS-, /MD added here -- use the dash form.
ifeq ($(BC_WINDOWS),1)
NVCC_HOST_FLAGS ?= -Xcompiler=-W3
else
NVCC_HOST_FLAGS ?= -Xcompiler=-Wall,-Wextra
endif
endif
ifeq ($(OPENCL),1)
CPPFLAGS += -DHAVE_OPENCL
ifeq ($(UNAME_S),Linux)
LDLIBS += -ldl
endif
endif

# Dead-strip the per-function/per-data sections requested above. Without this
# the -ffunction-sections/-fdata-sections flags are inert.
ifeq ($(UNAME_S),Darwin)
LDFLAGS += -Wl,-dead_strip
else
LDFLAGS += -Wl,--gc-sections
endif

# A MinGW build otherwise resolves libwinpthread-1.dll and zlib1.dll out of the
# MSYS2 prefix, which is on PATH only inside the MSYS2 shell. The executable then
# works for whoever built it and fails for everyone else with "the code execution
# cannot proceed because libwinpthread-1.dll was not found" -- a message that
# describes a missing DLL rather than the missing PATH entry that is actually
# wrong, so it reads as a broken build. (libgomp-1.dll was a third until
# threading stopped being OpenMP.)
#
# The answer is the same one macOS reaches by a different route: the weights are
# already embedded, so these DLLs are the only thing standing between the build
# and a single file that can be copied anywhere. macOS gets there for free
# because pthreads and zlib both live in libSystem; Windows has to ask for it.
#
# WINDOWS_STATIC=0 gives a smaller executable that must stay beside its DLLs.
ifeq ($(BC_WINDOWS),1)
WINDOWS_STATIC ?= 1
ifeq ($(WINDOWS_STATIC),1)
LDFLAGS += -static -static-libgcc
endif
endif

.PHONY: clean test test-cpu-turbo refresh-fixtures test-guards test-metal test-cuda test-opencl ubsan tsan-threads FORCE \
	metal-beam metal-regenerate metal-regenerate-mindgrab metal-regenerate-model16 metal-regenerate-mindsnap \
	hip-artifacts-check hip-artifacts-build \
	all \
	baseline-capture baseline-compare check-generated generate-models \
	benchmark benchmark-capture benchmark-compare \
	models test-models check-payloads wasm wasm-gpu test-wasm test-wasm-gpu test-wasm-guards check-wgsl \
	test-wasm-mt \
	wasm-gl test-wasm-gl check-glsl \
	macos-pkg-adhoc macos-pkg macos-notarize macos-notary-profile macos-release \
	check-notary-profile \
	dispatcher \
	print-model-exes \
	print-release-exes
# Plain builds consume the checked-in BEAM-optimized headers and therefore
# retain the no-Python/no-tinygrad build. Use `make metal-beam` to retune them.
#
# `all` builds the four embedded-model executables plus the tiny zero-dep
# `brainchop` CLI dispatcher used by Homebrew, macOS .pkg, and native archives.
all: models dispatcher

# Make `all` the default rather than whatever rule happens to come first.
# check-payloads is defined near the top, so a bare `make` used to run only the
# payload size check -- printing two plausible lines, building nothing, exiting
# 0. README tells users `make` in four places and none of them built anything.
.DEFAULT_GOAL := all

# Each configuration compiles different -D flags, so a stale object built under
# a different one would link silently. Objects depend on a stamp that changes
# whenever the selected backends do. CUDA_MODELS is in it because adding a
# fatbin drops a model's -UHAVE_CUDA; without it the new module was ignored.
BUILD_CONFIG := CPUFLAGS=$(subst $() ,+,$(CPUFLAGS)) CC=$(CC) METAL=$(METAL) CUDA=$(CUDA) HIP=$(HIP) HIP_ARCH=$(HIP_ARCH) OPENCL=$(OPENCL) THREADS=$(THREADS) TURBO_CONE=$(TURBO_CONE) CUDA_GENCODE=$(CUDA_GENCODE) X86_ISA=$(X86_ISA) ZLIB=$(BC_ZLIB_ID) ZSTD=$(subst $() ,+,$(strip $(ZSTD) $(BC_ZSTD_ID))) WINDOWS_STATIC=$(WINDOWS_STATIC) MACOSX_DEPLOYMENT_TARGET=$(MACOSX_DEPLOYMENT_TARGET) CUDA_MODELS=$(subst $() ,+,$(sort $(CUDA_MODELS))) HIP_MODELS=$(subst $() ,+,$(sort $(HIP_MODELS)))
.build-config: FORCE
	@[ "$$(cat $@ 2>/dev/null)" = "$(BUILD_CONFIG)" ] || \
		{ echo "$(BUILD_CONFIG)" > $@; echo "configuration changed; rebuilding"; }
FORCE:

# --------------------------------------------------------------- WebAssembly
#
# A browser build. Deliberately NOT a variant of the native rules: it must not
# inherit HAVE_ZLIB or the native threading flags, and emcc links to .js + .wasm rather
# than an executable, so overloading the native template would obscure both.
#
# No zlib. The JS wrapper handles gzip with CompressionStream/DecompressionStream
# and the module only ever sees raw NIfTI, which is the same arrangement
# niimath's WASI reactor uses. Verified: a zlib-free build is bit-exact on
# uncompressed I/O and refuses .nii.gz with a clear message.
#
# No Metal/CUDA/OpenCL. The CPU engine is compiled in for every wasm target so
# that a build can be validated end to end without a GPU.
#
# -msimd128 because the CPU kernel is already written around 16 float
# accumulators, i.e. 4 x 128-bit vectors. -sMAXIMUM_MEMORY=4GB because CPU
# inference allocates ~2.15 GB of activations inside a 4 GiB address space --
# a constraint that disappears with the CPU path.
EMCC ?= emcc
WASM_DIR ?= ../dist/wasm
WASM_CFLAGS = -O3 -flto -ffast-math -fno-finite-math-only -msimd128 \
	-std=c11 -Wall -Wextra
WASM_LDFLAGS_BASE = -sMODULARIZE=1 -sEXPORT_ES6=1 -sSTACK_SIZE=4194304 \
	-sINVOKE_RUN=0 -sEXPORTED_RUNTIME_METHODS='["callMain","FS"]' \
	-sEXPORTED_FUNCTIONS='["_main","_malloc","_free"]'

# Standalone browser fitting adapter: shared by upstream WebGPU and WebGL2.
.PHONY: wasm-cat-lite
wasm-cat-lite:
	mkdir -p $(WASM_DIR)
	$(EMCC) $(WASM_CFLAGS) -I. -Imodels/mindmap -DBC_MODEL_IS_MINDMAP=1 \
		cat_lite.c bc_threads.c bwlabel.c --no-entry -sMODULARIZE=1 -sEXPORT_ES6=1 \
		-sEXPORTED_FUNCTIONS='["_bc_cat_lite_fit","_bc_cat_lite_fit_report","_malloc","_free"]' \
		-sEXPORTED_RUNTIME_METHODS='["HEAPU32","HEAPF32","HEAPF64"]' \
		-sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=2GB -sINITIAL_MEMORY=805306368 \
		-o $(WASM_DIR)/cat-lite.js

.PHONY: wasm-browser-benchmark
wasm-browser-benchmark:
	mkdir -p $(WASM_DIR)
	$(EMCC) $(WASM_CFLAGS) $(BC_BENCH_CFLAGS) -DBC_MODEL_IS_MINDMAP=1 -Imodels/mindmap -I. \
		-DHAVE_WEBGPU -DBC_WEBGPU_PROFILE=1 -DBC_PAYLOAD_PATH='"models/mindmap/model.bin"' \
		-DBC_PAYLOAD_SYM=bc_model_payload ../test/webgpu_benchmark.c \
		$(filter-out brainchopc.c,$(COMMON_SOURCES)) ../test/webgpu_benchmark_backend.c \
		webgpu_kernels.c payload.S $(WASM_LDFLAGS_BASE) --use-port=emdawnwebgpu \
		-sDEFAULT_TO_CXX -sASYNCIFY -sASYNCIFY_STACK_SIZE=32768 \
		-sINITIAL_MEMORY=805306368 -sEXIT_RUNTIME=1 -o $(WASM_DIR)/mindmap-benchmark.js

# The CPU module's own flags. SEPARATE VARIABLES, not additions to
# WASM_CFLAGS/WASM_LDFLAGS_BASE, which the WebGPU and WebGL2 rules also use:
# those modules run their arithmetic on the GPU, so pthreads would buy them
# nothing and cost them cross-origin isolation.
#
# Threading obeys THREADS=0 like the native targets do, so the serial control
# behind the patched-Nightly measurement in the build notes is `make wasm THREADS=0`
# rather than a hand-assembled flag set.
#
# bc_threads.c compiles unchanged: it is plain POSIX, and resolve_count() falls
# through to sysconf(_SC_NPROCESSORS_ONLN), which emscripten answers with
# navigator.hardwareConcurrency. the project's *Threading* owns the figure.
#
# -sPROXY_TO_PTHREAD is required rather than tuning: bc_parallel_for blocks on a
# condition variable, and blocking is illegal on a browser main thread. It moves
# main() onto a worker, which ALSO makes callMain return before main finishes --
# the trap ASYNCIFY already sets for the WebGPU module, arriving by another
# route. ../test/wasm_run.mjs waits on onExit, which is why EXIT_RUNTIME is
# here -- and why it is OUTSIDE the conditional. A proxied callMain returns 0
# without throwing, exactly as a serial one does after main has finished, so the
# return value cannot tell the two apart and onExit has to fire in both. A
# THREADS=0 module linked without it made the harness read /out.nii before it
# existed, which is the same ENOENT the -cpu 0 --no-turbo harness used to produce.
#
# The pool size is a JS expression emcc evaluates at run time, so the workers are
# preallocated at the machine's width and -cpu N never has to spawn one late.
WASM_CPU_CFLAGS = $(WASM_CFLAGS)
WASM_CPU_LDFLAGS = $(WASM_LDFLAGS_BASE) -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB \
	-sEXIT_RUNTIME=1
ifneq ($(THREADS),0)
WASM_CPU_CFLAGS += -pthread -DBC_THREADS
WASM_CPU_LDFLAGS += -pthread -sPROXY_TO_PTHREAD \
	-sPTHREAD_POOL_SIZE=navigator.hardwareConcurrency
endif

# WASM_RELAXED_SIMD=1 buys 1.4x and costs the bit-exactness guarantee.
#
# Wasm SIMD has no FMA, and that is essentially the whole gap to native: the
# default build emits 248 f32x4.mul against 233 f32x4.add where arm64 emits 324
# fmla, and runs 2x slower than the same code compiled natively. -mrelaxed-simd
# fuses 190 of those pairs into f32x4.relaxed_madd and takes MindGrab from
# 10.9/11.5 s to 7.6/8.0 s (alternating pairs, 14 CPUs, t1_crop).
#
# OFF by default because whether relaxed_madd actually fuses is the ENGINE's
# choice: it lowers to vfmadd on hardware with FMA and to a separate multiply
# and add on hardware without, so the same module can round two ways. It was
# bit-exact against test/bet.nii.gz on V8/arm64, which is evidence about one
# engine and one CPU, not the portable "one answer everywhere" that makes
# test-wasm a gate. Turn it on for a deployment measured on its own targets.
ifeq ($(WASM_RELAXED_SIMD),1)
WASM_CPU_CFLAGS += -mrelaxed-simd
endif

# Profiler-only name map. Emscripten keeps production wasm stripped, but writes
# `<module>.js.symbols` beside it so Firefox's `wasm-function[N]` frames can be
# mapped back to the C function that generated them. Opt-in rather than another
# shipped artifact.
#
# It does not alter the generated CODE -- an A/B disassembly found every opcode
# count equal and all 173 function bodies matching as an exact multiset -- but
# it DOES alter the file: 121,043 of 472,097 bytes differ, because function and
# type index order moves and the minified import names follow. That is why it is
# safe to profile against where -g2 would not be, and why it wants its own
# WASM_DIR: the default would leave dist/wasm holding bytes that are not the
# shipped ones, and `clean` does not remove the stray .js.symbols.
ifeq ($(WASM_CPU_SYMBOL_MAP),1)
WASM_CPU_LDFLAGS += --emit-symbol-map
endif

$(WASM_DIR):
	@mkdir -p $@

# Compile the WGSL kernels in a real browser and fail on any diagnostic.
#
# A browser's own compiler is the only authority that matters: what Chromium
# accepts is what the shipped module can use. Needs a checkout with playwright
# installed (BC_PLAYWRIGHT_ROOT, defaulting to ../../BrowserQC) and runs headed,
# because headless Chromium on macOS has no GPU. Not part of `make test` for
# that reason.
check-wgsl: | $(BC_TMP)
	@for m in $(MODELS); do \
		$(CC) -I. -Imodels/$$m -D$$(echo BC_MODEL_IS_$$m | tr a-z A-Z)=1 \
			-std=c11 ../test/dump_wgsl.c -o $(BC_TMP)/bc-dump-$$m || exit 1; \
		$(BC_TMP)/bc-dump-$$m > $(BC_TMP)/bc-kernels-$$m.wgsl || exit 1; \
		$(BC_TMP)/bc-dump-$$m wide > $(BC_TMP)/bc-kernels-$$m-wide.wgsl || exit 1; \
	done
	node ../test/wgsl_validate.mjs $(foreach m,$(MODELS),$(BC_TMP)/bc-kernels-$(m).wgsl $(BC_TMP)/bc-kernels-$(m)-wide.wgsl)

# The WebGL2 counterpart, and STRICTER than check-wgsl: linkProgram validates
# the MRT output locations and the uniform interface, which createShaderModule
# has no analogue for. Runs on Chromium AND Firefox, because they do not share a
# GLSL compiler (ANGLE vs Firefox's own translator) and Firefox is the browser
# this backend exists for.
check-glsl: | $(BC_TMP)
	@for m in $(MODELS); do \
		$(CC) -I. -Imodels/$$m -D$$(echo BC_MODEL_IS_$$m | tr a-z A-Z)=1 \
			-std=c11 ../test/dump_glsl.c -o $(BC_TMP)/bc-dump-glsl-$$m || exit 1; \
		$(BC_TMP)/bc-dump-glsl-$$m > $(BC_TMP)/bc-kernels-$$m.glsl || exit 1; \
	done
	node ../test/glsl_validate.mjs $(foreach m,$(MODELS),$(BC_TMP)/bc-kernels-$(m).glsl)

# The WebGPU variant. ASYNCIFY is needed for exactly one thing -- waiting on the
# buffer readback -- because WebGPU's map is asynchronous and bc_backend->segment
# is not. Device creation is NOT async here: JS does that and hands the device
# over, which is why ASYNCIFY's reach can stay small.
#
# FIXED memory, not ALLOW_MEMORY_GROWTH, and this is not a tuning choice.
# Growth makes the wasm heap a RESIZABLE ArrayBuffer, and TextDecoder refuses
# to decode one, so emdawnwebgpu cannot read the WGSL string view it is handed:
# createShaderModule fails with "The provided ArrayBuffer value must not be
# resizable". Fixed memory is the right shape here regardless -- the activations
# live on the GPU, so the heap only holds the image, the conformed volume and
# the packed weights, a few hundred MiB rather than the 2.15 GB the CPU path
# needs.
# Current emdawnwebgpu links C++ support; compile the .c sources as C as before.
WASM_GPU_LDFLAGS = --use-port=emdawnwebgpu -sDEFAULT_TO_CXX -sASYNCIFY \
	-sASYNCIFY_STACK_SIZE=32768 -sINITIAL_MEMORY=805306368 -sEXIT_RUNTIME=1

define BC_WASM_RULES
wasm-$(1): check-payloads | $$(WASM_DIR)
	$$(EMCC) $$(WASM_CPU_CFLAGS) -D$$(sel_$(1))=1 -Imodels/$(1) -I. \
		-DBC_PAYLOAD_PATH='"$$(bin_$(1))"' -DBC_PAYLOAD_SYM=bc_model_payload \
		$$(COMMON_SOURCES) payload.S \
		$$(WASM_CPU_LDFLAGS) -o $$(WASM_DIR)/$$(exe_$(1)).js

wasm-gpu-$(1): check-payloads | $$(WASM_DIR)
	$$(EMCC) $$(WASM_CFLAGS) -D$$(sel_$(1))=1 -Imodels/$(1) -I. -DHAVE_WEBGPU -DBC_TURBO_CONE=$$(TURBO_CONE) \
		-DBC_PAYLOAD_PATH='"$$(bin_$(1))"' -DBC_PAYLOAD_SYM=bc_model_payload \
		$$(COMMON_SOURCES) backend_webgpu.c webgpu_kernels.c payload.S \
		$$(WASM_LDFLAGS_BASE) $$(WASM_GPU_LDFLAGS) \
		-o $$(WASM_DIR)/$$(exe_$(1))-gpu.js
endef
$(foreach m,$(MODELS),$(eval $(call BC_WASM_RULES,$(m))))

# The WebGL2 variant -- the fallback for browsers without WebGPU.
#
# No ASYNCIFY and no emdawnwebgpu: glReadPixels blocks, so the whole backend is
# synchronous. -sGL_PREINITIALIZED_CONTEXT=1 takes the context from
# Module.preinitializedWebGLContext, mirroring how the WebGPU module is handed a
# device, so capability refusals can be phrased in TypeScript. Verified on
# Chromium and Firefox.
#
# MIN_WEBGL_VERSION=2 as well as MAX: without it emscripten also emits the
# WebGL1 code paths, which this backend can never use.
#
# FIXED memory, like the WebGPU module. The activations live on the GPU, so the
# heap holds the image, the conformed volume, the packed weights and -- in a
# WASM_GL_PHASE1 build -- the reference input and two slices. 512 MiB covers that with
# room; the shipping module should be re-measured and trimmed.
#
# BC_WEBGL2_PHASE1 adds the layer-0 verification: after conv_first, three slices
# are read back and compared against a CPU reference in f16 ULPs. It only ADDS
# that check -- an earlier version of this comment said segment() refuses
# without it, which was true only while the backend was incomplete.
#
# INITIAL_MEMORY is 768 MiB, including the four PVE planes and CPU fit
# scratch required by tissue mode. The label path holds a 67 MB
# f32 copy of the whole normalized volume so the reference sees exactly the
# values the GPU saw, plus three slice buffers, and it costs four blocking
# glReadPixels per checked slice. js/package.json builds the shipped module with
# WASM_GL_PHASE1=0 for that reason.
# WASM_GL_PROFILE=1 adds the per-stage breakdown. It is compile-time and not an
# environment variable because emscripten snapshots the environment into
# __environ during runtime init, so getenv cannot be flipped from JS -- the same
# trap BC_WEBGPU_PROFILE documents. It also inserts a real GPU sync between
# stages, which perturbs the total, so profile and shipping builds are separate
# artifacts by construction rather than by discipline.
WASM_GL_PHASE1 ?= 1
WASM_GL_PROFILE ?= 0
# Voxels per fragment in the hidden convolution; 1 is the A/B control.
WASM_GL_VOX ?= 2
# specialHTMLTargets is exported so the host can register its canvas under a
# namespaced key rather than needing a DOM element called "#canvas". See the
# comment on acquire_context() in backend_webgl2.c; this is what makes the
# backend usable from a Worker with an OffscreenCanvas.
WASM_GL_LDFLAGS = -sMAX_WEBGL_VERSION=2 -sMIN_WEBGL_VERSION=2 \
	-sGL_PREINITIALIZED_CONTEXT=1 -sINITIAL_MEMORY=805306368 -sEXIT_RUNTIME=1 \
	-sEXPORTED_RUNTIME_METHODS='["callMain","FS","specialHTMLTargets"]'

define BC_WASM_GL_RULES
wasm-gl-$(1): check-payloads | $$(WASM_DIR)
	$$(EMCC) $$(WASM_CFLAGS) -D$$(sel_$(1))=1 -Imodels/$(1) -I. -DHAVE_WEBGL2 \
		-DBC_WEBGL2_PHASE1=$$(WASM_GL_PHASE1) \
		-DBC_WEBGL2_PROFILE=$$(WASM_GL_PROFILE) -DBC_WGL_VOX=$$(WASM_GL_VOX) \
		-DBC_PAYLOAD_PATH='"$$(bin_$(1))"' -DBC_PAYLOAD_SYM=bc_model_payload \
		$$(COMMON_SOURCES) backend_webgl2.c webgl2_kernels.c payload.S \
		$$(WASM_LDFLAGS_BASE) $$(WASM_GL_LDFLAGS) \
		-o $$(WASM_DIR)/$$(exe_$(1))-gl.js
endef
$(foreach m,$(MODELS),$(eval $(call BC_WASM_GL_RULES,$(m))))
WASM_GL_TARGETS = $(foreach m,$(MODELS),wasm-gl-$(m))
.PHONY: $(WASM_GL_TARGETS)
wasm-gl: $(WASM_GL_TARGETS)

# The layer-0 gate: the first convolution on the GPU against the CPU reference.
#
# Two steps, and the first is what makes the second meaningful. The native
# check pins bc_webgl2_ref_conv_first_slice to mn_conv_first over the full
# volume at every dilation the models use; the browser run then compares the
# GPU against that reference. Without the first step the second is only this
# code agreeing with itself.
#
# Not part of `make test`: it needs a browser, and runs headed because headless
# Chromium on macOS has no GPU.
#
# bc_threads.c is in the native link line because webgl2_conv_first_check.c
# includes meshnet_cpu.c whole -- that is how it reaches the static
# mn_conv_first without disturbing MN_DEFINE_CONV -- and those loops call
# bc_parallel_for. It is compiled WITHOUT -DBC_THREADS, so the reference runs
# serial, which is what a reference should be: one fewer variable between this
# and the GPU it is pinning.
test-wasm-gl: wasm-gl | $(BC_TMP)
	@for m in $(MODELS); do \
		echo "  reference check: $$m"; \
		$(CC) -O2 -std=c11 -I. -Imodels/$$m \
			-D$$(echo BC_MODEL_IS_$$m | tr a-z A-Z)=1 \
			-ffast-math -fno-finite-math-only \
			../test/webgl2_conv_first_check.c model_specs.c bc_threads.c \
			-o $(BC_TMP)/bc-ref-check-$$m$(EXEEXT) -lm || exit 1; \
		$(BC_TMP)/bc-ref-check-$$m$(EXEEXT) > $(BC_TMP)/bc-ref-check-$$m.log || \
			{ cat $(BC_TMP)/bc-ref-check-$$m.log; exit 1; }; \
		tail -1 $(BC_TMP)/bc-ref-check-$$m.log; \
	done
	python3 -c 'import gzip,pathlib; pathlib.Path("$(BC_TMP)/bc-w-t1crop.nii").write_bytes(gzip.open("$(TEST_IN)","rb").read())'
	python3 -c 'import gzip,pathlib; pathlib.Path("$(BC_TMP)/bc-w-t12mm.nii").write_bytes(gzip.open("$(TEST_2MM)","rb").read())'
	node ../test/wasm_gl_run.mjs $(WASM_DIR)/$(exe_mindgrab)-gl.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-gl-mg.nii -backend webgl2
	python3 ../test/nii_diff.py $(TEST_MG) $(BC_TMP)/bc-gl-mg.nii 64
	node ../test/wasm_gl_run.mjs $(WASM_DIR)/$(exe_model16chan18cls)-gl.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-gl-m16.nii -backend webgl2 --save-conform
	python3 ../test/nii_diff.py $(TEST_M16) $(BC_TMP)/bc-gl-m16.nii 1024
	node ../test/wasm_gl_run.mjs $(WASM_DIR)/$(exe_model16chan18cls)-gl.js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-gl-nat.nii -backend webgl2
	python3 ../test/nii_diff.py $(TEST_M16_NATIVE) $(BC_TMP)/bc-gl-nat.nii 256
	node ../test/wasm_gl_run.mjs $(WASM_DIR)/$(exe_mindmap)-gl.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-gl-mm.nii -backend webgl2 --save-conform --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM) $(BC_TMP)/bc-gl-mm.nii 1024
	node ../test/wasm_gl_run.mjs $(WASM_DIR)/$(exe_mindmap)-gl.js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-gl-mmnat.nii -backend webgl2 --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM_NATIVE) $(BC_TMP)/bc-gl-mmnat.nii 256
	node ../test/wasm_gl_run.mjs $(WASM_DIR)/$(exe_mindsnap)-gl.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-gl-ms.nii -backend webgl2 --save-conform
	python3 ../test/nii_diff.py $(TEST_MS) $(BC_TMP)/bc-gl-ms.nii $(MS_TOL)
	node ../test/wasm_gl_run.mjs $(WASM_DIR)/$(exe_mindsnap)-gl.js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-gl-msnat.nii -backend webgl2
	python3 ../test/nii_diff.py $(TEST_MS_NATIVE) $(BC_TMP)/bc-gl-msnat.nii $(MS_TOL_NATIVE)
	@echo "test-wasm-gl: layer 0 exact, and all seven cases within the fp16 budgets"

WASM_TARGETS = $(foreach m,$(MODELS),wasm-$(m))
WASM_GPU_TARGETS = $(foreach m,$(MODELS),wasm-gpu-$(m))
# Declared HERE, not in the .PHONY block above: make expands a target line's
# prerequisites as it reads them, and these variables are defined below it, so
# the earlier reference silently expanded to nothing.
.PHONY: $(WASM_TARGETS) $(WASM_GPU_TARGETS)
wasm: $(WASM_TARGETS)
wasm-gpu: $(WASM_GPU_TARGETS)

# The WebGPU gate. Runs the modules in a real browser and holds them to the
# SAME voxel budgets the Metal backend carries -- these are fp16 GPU paths with
# the same tolerance argument, so a separate, looser budget would need its own
# justification rather than being assumed.
#
# Not part of `make test`: it needs a browser with WebGPU, runs headed because
# headless Chromium on macOS has no GPU, and depends on a checkout with
# playwright (BC_PLAYWRIGHT_ROOT).
test-wasm-gpu: wasm-gpu | $(BC_TMP)
	python3 -c 'import gzip,pathlib; pathlib.Path("$(BC_TMP)/bc-w-t1crop.nii").write_bytes(gzip.open("$(TEST_IN)","rb").read())'
	python3 -c 'import gzip,pathlib; pathlib.Path("$(BC_TMP)/bc-w-t12mm.nii").write_bytes(gzip.open("$(TEST_2MM)","rb").read())'
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindgrab)-gpu.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-g-mg.nii -backend webgpu --no-turbo
	python3 ../test/nii_diff.py $(TEST_MG) $(BC_TMP)/bc-g-mg.nii 64
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_model16chan18cls)-gpu.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-g-m16.nii -backend webgpu --no-turbo --save-conform
	python3 ../test/nii_diff.py $(TEST_M16) $(BC_TMP)/bc-g-m16.nii 1024
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_model16chan18cls)-gpu.js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-g-nat.nii -backend webgpu --no-turbo
	python3 ../test/nii_diff.py $(TEST_M16_NATIVE) $(BC_TMP)/bc-g-nat.nii 256
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindmap)-gpu.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-g-mm.nii -backend webgpu --no-turbo --save-conform --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM) $(BC_TMP)/bc-g-mm.nii 1024
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindmap)-gpu.js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-g-mmnat.nii -backend webgpu --no-turbo --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM_NATIVE) $(BC_TMP)/bc-g-mmnat.nii 256
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindsnap)-gpu.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-g-ms.nii -backend webgpu --no-turbo --save-conform
	python3 ../test/nii_diff.py $(TEST_MS) $(BC_TMP)/bc-g-ms.nii $(MS_TOL)
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindsnap)-gpu.js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-g-msnat.nii -backend webgpu --no-turbo
	python3 ../test/nii_diff.py $(TEST_MS_NATIVE) $(BC_TMP)/bc-g-msnat.nii $(MS_TOL_NATIVE)
	# Turbo (default): the same gates as test-metal.
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindgrab)-gpu.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-gt-mg.nii -backend webgpu
	python3 ../test/nii_dice.py $(TEST_MG) $(BC_TMP)/bc-gt-mg.nii $(TURBO_GATE_MG)
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_model16chan18cls)-gpu.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-gt-m16.nii -backend webgpu --save-conform
	python3 ../test/nii_dice.py $(TEST_M16) $(BC_TMP)/bc-gt-m16.nii $(TURBO_GATE)
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_model16chan18cls)-gpu.js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-gt-m16n.nii -backend webgpu
	python3 ../test/nii_dice.py $(TEST_M16_NATIVE) $(BC_TMP)/bc-gt-m16n.nii $(TURBO_GATE)
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindmap)-gpu.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-gt-mm.nii -backend webgpu --save-conform --legacy-cleanup
	python3 ../test/nii_dice.py $(TEST_MM) $(BC_TMP)/bc-gt-mm.nii $(TURBO_GATE)
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindmap)-gpu.js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-gt-mmn.nii -backend webgpu --legacy-cleanup
	python3 ../test/nii_dice.py $(TEST_MM_NATIVE) $(BC_TMP)/bc-gt-mmn.nii $(TURBO_GATE)
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindsnap)-gpu.js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-gt-ms.nii -backend webgpu --save-conform
	python3 ../test/nii_dice.py $(TEST_MS) $(BC_TMP)/bc-gt-ms.nii $(TURBO_GATE)
	node ../test/wasm_gpu_run.mjs $(WASM_DIR)/$(exe_mindsnap)-gpu.js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-gt-msn.nii -backend webgpu
	python3 ../test/nii_dice.py $(TEST_MS_NATIVE) $(BC_TMP)/bc-gt-msn.nii $(TURBO_GATE)
	@echo "test-wasm-gpu: fp16 within the Metal voxel budgets, turbo within its gates"

# The wasm gate. Runs the same fixtures the native suite uses, through the
# module under node, and asserts BIT-EXACT agreement with the native reference
# -- which held on the first run, so there is no tolerance here to erode.
#
# Uncompressed in and out: the module has no zlib, so the harness decompresses
# first, exactly as the browser wrapper will with DecompressionStream.
#
# Not part of `make test`, which needs no emcc. The three cases run at the
# module's default thread count -- what a browser will use -- and take about
# 30 s together on a 14-CPU machine; they took nine minutes when the module was
# built without -DBC_THREADS, which is what made them serial (the old `-cpu 0`
# on these lines never pinned anything: cli.c reads N <= 0 as "choose"). No case
# pins -cpu: the CPU path's thread-count independence is the native suite's job
# to assert, and asserting it again here would double the gate to re-test a
# property of code this build shares.
test-wasm: wasm | $(BC_TMP)
	python3 -c 'import gzip,pathlib; pathlib.Path("$(BC_TMP)/bc-w-t1crop.nii").write_bytes(gzip.open("$(TEST_IN)","rb").read())'
	python3 -c 'import gzip,pathlib; pathlib.Path("$(BC_TMP)/bc-w-t12mm.nii").write_bytes(gzip.open("$(TEST_2MM)","rb").read())'
	node ../test/wasm_run.mjs $(WASM_DIR)/$(exe_mindgrab).js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-w-mg.nii
	python3 ../test/nii_diff.py $(TEST_MG) $(BC_TMP)/bc-w-mg.nii 0
	node ../test/wasm_run.mjs $(WASM_DIR)/$(exe_model16chan18cls).js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-w-m16c.nii --save-conform
	python3 ../test/nii_diff.py $(TEST_M16) $(BC_TMP)/bc-w-m16c.nii 0
	node ../test/wasm_run.mjs $(WASM_DIR)/$(exe_model16chan18cls).js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-w-m16n.nii
	python3 ../test/nii_diff.py $(TEST_M16_NATIVE_WASM) $(BC_TMP)/bc-w-m16n.nii 0
	# mindmap under node needs ~3.2 GB of heap, against wasm32's 4 GB ceiling.
	# Bit-exact on this machine; a machine that cannot supply the heap fails at
	# the memory growth, and nothing here checks that failure is legible.
	node ../test/wasm_run.mjs $(WASM_DIR)/$(exe_mindmap).js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-w-mm.nii --save-conform --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM) $(BC_TMP)/bc-w-mm.nii 0
	node ../test/wasm_run.mjs $(WASM_DIR)/$(exe_mindsnap).js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-w-ms.nii --save-conform
	python3 ../test/nii_diff.py $(TEST_MS_WASM) $(BC_TMP)/bc-w-ms.nii 0
	node ../test/wasm_run.mjs $(WASM_DIR)/$(exe_mindsnap).js $(BC_TMP)/bc-w-t12mm.nii $(BC_TMP)/bc-w-msn.nii
	python3 ../test/nii_diff.py $(TEST_MS_NATIVE) $(BC_TMP)/bc-w-msn.nii 0
	@echo "test-wasm: all six cases bit-exact with the frozen CPU references"

# The browser half of the CPU gate, and the half node cannot stand in for: the
# threaded module imports SHARED memory, so it needs a cross-origin isolated
# page, a growable shared heap and -- in the shipping topology -- a pool spawned
# from inside a Worker. ../test/wasm_mt_run.mjs has the reasoning per row.
#
# Not part of `make test-wasm`, for the same reason test-wasm-gpu is not part of
# `make test`: it needs a playwright checkout (BC_PLAYWRIGHT_ROOT). It needs no
# GPU, so unlike the other browser gates it runs headless. `BC_TEST_BROWSER`
# picks the engine.
test-wasm-mt: wasm | $(BC_TMP)
	python3 -c 'import gzip,pathlib; pathlib.Path("$(BC_TMP)/bc-w-t1crop.nii").write_bytes(gzip.open("$(TEST_IN)","rb").read())'
	python3 -c 'import gzip,pathlib; pathlib.Path("$(BC_TMP)/bc-w-bet.nii").write_bytes(gzip.open("$(TEST_MG)","rb").read())'
	node ../test/wasm_mt_run.mjs $(WASM_DIR)/$(exe_mindgrab).js $(BC_TMP)/bc-w-t1crop.nii $(BC_TMP)/bc-w-bet.nii

# The module has no zlib, so a compressed input must be refused clearly rather
# than misread. This is the contract the JS wrapper depends on.
test-wasm-guards: wasm | $(BC_TMP)
	! node ../test/wasm_run.mjs $(WASM_DIR)/$(exe_mindgrab).js $(TEST_IN) $(BC_TMP)/bc-w-nope.nii 2>/dev/null
	@echo "test-wasm-guards: gzip input correctly refused"

# ------------------------------------------------------- per-model executables
#
# Objects cannot be shared between models -- they are compiled with different
# -D flags -- so each model builds into its own directory. That is also what
# makes the size comparison meaningful: build/<model>/ contains only what that
# model needs.
define BC_MODEL_RULES
$(1)_SRCS := $$(COMMON_SOURCES)
$(1)_OBJS := $$(patsubst %.c,build/$(1)/%.o,$$($(1)_SRCS))
$(1)_OBJS += build/$(1)/payload.o
ifeq ($$(METAL),1)
$(1)_OBJS += build/$(1)/backend_metal.o
endif
# A model with no offline module gets no CUDA backend at all, and must not be
# compiled with -DHAVE_CUDA either: backend.c would reference bc_cuda_backend
# and fail to link. CUDA defaults to `driver` on Linux and Windows, so without
# this a third model breaks `make models` on both -- and this machine is macOS,
# where CUDA defaults to 0 and the break is invisible.
$(1)_CPPFLAGS :=
$(1)_HIP_DIR := hip_artifacts/$(1)/$$(HIP_ARCH)
$(1)_HIP := $$(if $$(wildcard $$($(1)_HIP_DIR)/catalog.h),1,0)
ifeq ($$(HIP),1)
ifeq ($$($(1)_HIP),1)
$(1)_OBJS += build/$(1)/backend_hip_driver.o build/$(1)/hip_kernels.o
$(1)_CPPFLAGS += -DHAVE_HIP -DBC_HIP_CATALOG_HEADER='"$$($(1)_HIP_DIR)/catalog.h"'
endif
endif
$(1)_CUDA := $$(if $$(filter $(1),$$(CUDA_MODELS)),1,0)
ifeq ($$(CUDA),driver)
ifeq ($$($(1)_CUDA),1)
$(1)_OBJS += build/$(1)/backend_cuda_driver.o build/$(1)/cuda_module.o
else
$(1)_CPPFLAGS += -UHAVE_CUDA -DBC_CUDA_EXCLUDED=1
endif
endif
# CUDA=1 compiles backend_cuda.cu per model with no fatbin, but its geometry
# (cuda_geometry.h) exists only for the CUDA_MODELS, so the same gate applies.
ifeq ($$(CUDA)$$($(1)_CUDA),10)
$(1)_CPPFLAGS += -UHAVE_CUDA -DBC_CUDA_EXCLUDED=1
endif
ifeq ($$(CUDA)$$($(1)_CUDA),11)
ifeq ($$(BC_WINDOWS),1)
# On Windows the CUDA object is NOT linked into the executable. It cannot be:
# nvcc accepts only MSVC as its host compiler, so the object is MSVC-ABI and
# needs MSVC's CRT, and GNU ld cannot read MSVC's msvcrt.lib at all --
# "file format not recognized". See the CUDA DLL note above the rules below.
#
# One DLL PER MODEL, not one shared. The DLL contains backend_accel.c and
# model_specs.c compiled with -DBC_MODEL_IS_<NAME>, so a single brainchop_cuda.dll
# would carry one model's constants and silently mis-segment for the other.
$(1)_CUDA_DLL := brainchop-cuda-$(1).dll
$(1)_CUDA_IMPLIB := build/$(1)/brainchop-cuda-$(1).lib
$(1)_LINK_EXTRA := $$($(1)_CUDA_IMPLIB)
else
$(1)_OBJS += build/$(1)/backend_cuda.o
endif
endif
ifeq ($$(OPENCL),1)
ifeq ($$(call narrow_model,$(1)),1)
$(1)_OBJS += build/$(1)/backend_opencl.o
else
# BC_OPENCL_EXCLUDED lets `-backend opencl` say why, rather than reading like
# a machine with no OpenCL driver.
$(1)_CPPFLAGS += -UHAVE_OPENCL -DBC_OPENCL_EXCLUDED=1
endif
endif

build/$(1):
	@mkdir -p $$@

build/$(1)/msvc:
	@mkdir -p $$@

$$($(1)_OBJS): .build-config

build/$(1)/%.o: %.c | build/$(1)
	$$(CC) $$(CPPFLAGS) $$($(1)_CPPFLAGS) $$(BC_DEPFLAGS) -D$$(sel_$(1))=1 \
		-Imodels/$(1) $$(CFLAGS) -c $$< -o $$@

build/$(1)/payload.o: payload.S $$(bin_$(1)) | build/$(1)
	$$(CC) -I. -DBC_PAYLOAD_PATH='"$$(bin_$(1))"' \
		-DBC_PAYLOAD_SYM=bc_model_payload -c $$< -o $$@

build/$(1)/cuda_module.o: payload.S cuda_modules/$(1).fatbin | build/$(1)
	$$(CC) -I. -DBC_PAYLOAD_PATH='"cuda_modules/$(1).fatbin"' \
		-DBC_PAYLOAD_SYM=bc_cuda_module -c $$< -o $$@

build/$(1)/hip_kernels.o: payload.S $$($(1)_HIP_DIR)/kernels.bin | build/$(1)
	$$(CC) -I. -DBC_PAYLOAD_PATH='"$$($(1)_HIP_DIR)/kernels.bin"' \
		-DBC_PAYLOAD_SYM=bc_hip_kernels -c $$< -o $$@

build/$(1)/backend_metal.o: backend_metal.mm $$(metal_hdr_$(1)) | build/$(1)
	$$(CXX) $$(CPPFLAGS) $$(BC_DEPFLAGS) -D$$(sel_$(1))=1 -Imodels/$(1) \
		-DBC_METAL_HEADER='"$$(metal_hdr_$(1))"' \
		-DBC_METAL_SYM=$$(metal_sym_$(1)) -DBC_METAL_SYM_UC=$$(metal_uc_$(1)) \
		-O3 -std=c++17 -fobjc-arc -ffast-math -flto -Wall -Wextra -c $$< -o $$@

build/$(1)/backend_cuda.o: backend_cuda.cu backend_cuda_host.h cuda_kernels.cuh cuda_variants.h cuda_geometry.h cuda_geometry_mindmap.h pve_constants.h | build/$(1)
	$$(NVCC) $$(CPPFLAGS) $$(BC_NVCC_DEPFLAGS) -D$$(sel_$(1))=1 -Imodels/$(1) \
		-O3 --use_fast_math -std=c++11 \
		$$(CUDA_GENCODE) $$(NVCC_HOST_FLAGS) -c $$< -o $$@

build/$(1)/backend_opencl.o: backend_opencl.c | build/$(1)
	$$(CC) $$(CPPFLAGS) $$(BC_DEPFLAGS) -D$$(sel_$(1))=1 -Imodels/$(1) $$(CFLAGS) -c $$< -o $$@

# ------------------------------------------------- the Windows CUDA DLL
#
# Everything from here to the executable rule exists because of one fact:
# nvcc on Windows accepts ONLY MSVC as its host compiler, so backend_cuda.cu
# becomes an MSVC-ABI object, and this project's executable is MinGW/UCRT.
#
# The obvious fix -- hand ld the MSVC runtime as well -- was tried and does not
# work. GNU ld reads cudart.lib happily but rejects msvcrt.lib outright with
# "file format not recognized", and without it the link is short of
# __security_cookie, __security_check_cookie, __GSHandlerCheck,
# _Init_thread_header/footer/epoch, __local_stdio_printf_options, fprintf,
# snprintf and sscanf. Satisfying that list by hand means reimplementing pieces
# of the MSVC CRT, which would be wrong the first time a toolkit changed.
#
# So the ABI boundary is a DLL instead, which is where a boundary between two
# C++ ABIs belongs. MSVC links its own object against its own CRT, and exactly
# ONE symbol crosses: the bc_cuda_backend dispatch table, exported as DATA
# through cuda_exports.def. MinGW's auto-import handles the data reference on
# the other side, so no header needs a __declspec.
#
# backend_accel.c, model_specs.c and bc_threads.c are compiled INTO the DLL as
# well, because backend_cuda.cu calls bc_accel_spec/normalize_input/copy_weights,
# those in turn need brainchop_model_dilation, and backend_accel.c's two
# elementwise maps call bc_parallel_for. They are also still compiled into the
# executable, for the CPU and OpenCL paths. Two copies is safe here and not a
# smell: the first two are pure functions over a `static const` table generated
# from the same descriptor, so the copies cannot disagree.
#
# bc_threads.c is the third for a different reason, and it is the one to
# understand before touching this list. The DLL is compiled WITHOUT
# -DBC_THREADS, so what MSVC gets is the serial fallback -- no pthread.h, no
# pool, the range run whole on the calling thread. That is not a compromise
# forced by MSVC lacking pthreads; it is exactly what this configuration did
# under OpenMP, where the DLL was compiled without /openmp and the copy
# cuda_segment calls was already serial. The output is identical either way,
# which is what makes the duplicate safe -- see the note on bc_accel_narrow_input
# in backend_accel.h. A Win32 threading backend would change this line, not the
# reasoning.
#
# -TC forces C. nvcc compiles a .c input as C++, and these two are C11 --
# model_specs.c is a wall of _Static_assert, which C++ spells static_assert, so
# as C++ it fails with a syntax error that names neither the language nor the
# cause.
#
# Guarded, because $$($(1)_CUDA_DLL) is only defined for a Windows CUDA build.
# Left ungated, the rules below would have an EMPTY target and make would abort
# reading the file with "missing target" on every other platform.
ifeq ($$(CUDA)$$(BC_WINDOWS)$$($(1)_CUDA),111)
# Header prerequisites, spelled out because neither rule below has automatic
# dependency tracking: cl is invoked without /showIncludes, and the nvcc rule
# carries $(BC_NVCC_DEPFLAGS) only in the non-DLL form. This matters more here
# than it usually would. bc_backend and bc_inference_options are DEFINED in
# backend.h and cross the DLL boundary by value, so an edit to either that
# rebuilt one side and not the other would not fail to link -- it would produce
# two different memory layouts for the same struct, which is silent corruption
# rather than an error. That is exactly what happened to be at risk when
# --adapter added two function pointers to bc_backend.
# f16.h and accel_pack_padded.h are here because the two files below include
# them and this list is the ONLY dependency tracking the Windows DLL rules have.
# Omitting one is silent: backend_accel.c is compiled twice (gcc into the exe,
# cl into the DLL), so a stale DLL object would leave CUDA narrowing input with
# one f16 rounding rule while OpenCL, in the exe, uses another -- inside one
# binary, with no error. That is the divergence f16.h exists to prevent.
BC_CUDA_DLL_HEADERS := pve_constants.h cuda_geometry_mindmap.h backend_cuda_host.h backend.h backend_accel.h cuda_variants.h cuda_geometry.h model_specs.h f16.h \
	accel_pack_padded.h bc_threads.h
build/$(1)/msvc/%.obj: %.c .build-config $$(BC_CUDA_DLL_HEADERS) models/$(1)/model_def.h | build/$(1)/msvc
	$$(BC_MSVC_CC) -nologo -c -TC -std:c11 -O2 -DHAVE_CUDA \
		-D$$(sel_$(1))=1 -Imodels/$(1) -I. $$< -Fo$$@

# The import library is a SIDE EFFECT of linking the DLL, not a separate build
# step, so it gets a recipe-less rule that simply waits for the DLL. Without
# this make has no rule for the .lib the executable asks for.
$$($(1)_CUDA_IMPLIB): $$($(1)_CUDA_DLL)
	@:

# -std=c++11 is deliberately absent: MSVC rejects it and nvcc only warns, so it
# was never doing anything on this platform.
# .build-config, for the same reason $($(1)_OBJS) depends on it: CUDA_ARCH and
# CUDA_GENCODE are part of the configuration, and without this prerequisite
# `make CUDA=1 CUDA_ARCH=sm_89 && make CUDA=1` rebuilds the executable and
# leaves a STALE single-architecture DLL beside it. That fails at run time with
# cudaErrorNoKernelImageForDevice rather than at build time.
$$($(1)_CUDA_DLL): backend_cuda.cu cuda_kernels.cuh cuda_exports.def .build-config \
		$$(BC_CUDA_DLL_HEADERS) models/$(1)/model_def.h \
		build/$(1)/msvc/backend_accel.obj build/$(1)/msvc/model_specs.obj \
		build/$(1)/msvc/bc_threads.obj | build/$(1)
	$$(NVCC) -DHAVE_CUDA -D$$(sel_$(1))=1 -Imodels/$(1) -I. \
		-O3 --use_fast_math $$(CUDA_GENCODE) $$(NVCC_HOST_FLAGS) \
		-shared backend_cuda.cu \
		build/$(1)/msvc/backend_accel.obj build/$(1)/msvc/model_specs.obj \
		build/$(1)/msvc/bc_threads.obj \
		-o $$@ -Xlinker -DEF:cuda_exports.def \
		-Xlinker -IMPLIB:$$($(1)_CUDA_IMPLIB)
endif

# BC_ZLIB_DEP is empty for the system zlib and the archive path for zlib-ng, so
# replacing that archive relinks. It is a prerequisite only -- LDLIBS already
# names it on the command line.
$$(exe_$(1)): $$($(1)_OBJS) $$($(1)_LINK_EXTRA) $$(BC_ZLIB_DEP)
	$$(CC) $$(LDFLAGS) -o $$@ $$($(1)_OBJS) $$($(1)_LINK_EXTRA) $$(LDLIBS)

-include $$(patsubst %.o,%.d,$$($(1)_OBJS))
endef
$(foreach m,$(MODELS),$(eval $(call BC_MODEL_RULES,$(m))))

MODEL_EXES = $(foreach m,$(MODELS),$(exe_$(m)))
DISPATCHER_EXE = brainchop$(EXEEXT)
models: check-payloads $(MODEL_EXES)

$(DISPATCHER_EXE): brainchop_dispatch.c version.h .build-config
	$(CC) $(CFLAGS) -I. $(LDFLAGS) -o $@ brainchop_dispatch.c

dispatcher: $(DISPATCHER_EXE)

# Run from the repository root because the exporter deliberately resolves its
# model inputs there. The two recursive calls serialize GPU tuning even under
# `make -j`, avoiding simultaneous multi-gigabyte Metal workloads.
metal-regenerate:
	$(MAKE) metal-regenerate-mindgrab
	$(MAKE) metal-regenerate-model16
	$(MAKE) metal-regenerate-mindmap
	$(MAKE) metal-regenerate-mindsnap

metal-regenerate-mindgrab:
	cd .. && mkdir -p "$(METAL_MANIFEST_DIR)"
	cd .. && $(TINYGRAD_PYTHON) scripts/export_tinygrad_metal.py mindgrab \
		"$(METAL_MANIFEST_DIR)/brainchopc-mindgrab-beam$(MINDGRAB_BEAM).json" \
		--header src/metal_mindgrab_generated.h --beam $(MINDGRAB_BEAM)

metal-regenerate-model16:
	cd .. && mkdir -p "$(METAL_MANIFEST_DIR)"
	cd .. && $(TINYGRAD_PYTHON) scripts/export_tinygrad_metal.py model16chan18cls \
		"$(METAL_MANIFEST_DIR)/brainchopc-model16-beam$(MODEL16_BEAM).json" \
		--header src/metal_model16_generated.h --beam $(MODEL16_BEAM)

metal-regenerate-mindmap:
	cd .. && mkdir -p "$(METAL_MANIFEST_DIR)"
	cd .. && $(TINYGRAD_PYTHON) scripts/export_tinygrad_metal.py mindmap \
		"$(METAL_MANIFEST_DIR)/brainchopc-mindmap-beam$(MINDMAP_BEAM).json" \
		--header src/metal_mindmap_generated.h --beam $(MINDMAP_BEAM)

metal-regenerate-mindsnap:
	cd .. && mkdir -p "$(METAL_MANIFEST_DIR)"
	cd .. && $(TINYGRAD_PYTHON) scripts/export_tinygrad_metal.py mindsnap \
		"$(METAL_MANIFEST_DIR)/brainchopc-mindsnap-beam$(MINDSNAP_BEAM).json" \
		--header src/metal_mindsnap_generated.h --beam $(MINDSNAP_BEAM)

metal-beam: metal-regenerate
	$(MAKE) METAL=1 models

# Test fixtures: two inputs plus the CPU reference outputs they produce.
#
# TEST_IN is already 256^3 at 1 mm -- the grid conform targets -- so for it the
# conformed and native output spaces COINCIDE and --inverse-conform is a no-op.
# TEST_2MM is a real 96x128x128 2 mm INT16 acquisition, so conform must upsample
# into 256^3, the inverse reslice must undo it, and MindGrab must restore INT16.
# The modular refactor makes native space the DEFAULT for label models, and
# TEST_M16_NATIVE is what that new default must reproduce bit-exactly. Deriving
# it from TEST_IN instead would give a fixture identical to TEST_M16 and an
# assertion that cannot fail.
TEST_IN  = ../test/t1_crop.nii.gz
TEST_2MM = ../test/t1_2mm.nii.gz
TEST_MG  = ../test/bet.nii.gz
TEST_M16 = ../test/labels16.nii.gz
TEST_M16_NATIVE = ../test/labels16_native_majority.nii.gz
TEST_M16_NATIVE_WASM = ../test/labels16_native_majority_wasm.nii.gz
# The frozen x86 and wasm model16 results coincide; ARM differs at one
# conformed voxel newly sampled by majority voting. Both remain exact gates.
# See docs/nvidia-browser-parity-validation.md for baseline provenance.
ifeq ($(UNAME_M),x86_64)
TEST_M16_NATIVE = $(TEST_M16_NATIVE_WASM)
endif
# Frozen fixtures retain pre-cleanup semantics; replay with --legacy-cleanup.
TEST_MM  = ../test/labels24.nii.gz
TEST_MM_NATIVE = ../test/labels24_native_majority.nii.gz
# Corroborated against ../brainchop-cli (scripts/oracle_mindsnap.sh): t1_2mm agrees
# to 0 conformed voxels, t1_crop to 2 against tinygrad's METAL fp32 device.
TEST_MS = ../test/labels104.nii.gz
TEST_MS_NATIVE = ../test/labels104_native_majority.nii.gz
# wasm SIMD has no FMA, and one t1_crop voxel is a 4-ulp tie (57 vs 51,
# 5e-7 relative) that the different rounding flips. Same shape as model16's.
TEST_MS_WASM = ../test/labels104_wasm.nii.gz
# Metal, WebGPU and WebGL2 measured 1248/171, 1280/198 and 1219/184
# (conformed/native): a 104-way argmax has far more near-tied pairs than
# mindmap's 18-way (517/92 Metal on PR #41's weights).
MS_TOL = 2048
MS_TOL_NATIVE = 512
# Turbo: brainchop-test's V4 gates (macro Dice, foreground agreement %).
TURBO_GATE = 0.988 98.9
TURBO_GATE_MG = 0.995 99.5

# Scratch directory for every test recipe below; see BC_TMP at the top of this
# file for why it is not simply /tmp on Windows. A no-op on Unix, where /tmp
# already exists and so is never remade. Used as an ORDER-ONLY prerequisite
# (after the |) throughout: the directory's timestamp changes every time a test
# writes into it, and a normal prerequisite would make every test target look
# out of date on the next run.
$(BC_TMP):
	@mkdir -p $@

# Per-model constants are generated from models/<name>/model.json rather than
# transcribed, and the generator cross-checks the derived layout against the
# byte size of model.bin. check-generated fails if a checked-in model_def.h no
# longer matches its model -- the drift that left test/labels16.nii.gz stale for
# two commits. The headers are checked in, so an ordinary build needs no Python.
MODEL_DIRS = $(sort $(dir $(wildcard models/*/model.json)))

check-generated:
	$(TINYGRAD_PYTHON) ../scripts/gen_model_capabilities.py --check
	@for dir in $(MODEL_DIRS); do \
		$(TINYGRAD_PYTHON) ../scripts/gen_model_def.py "$$dir" --check || exit 1; \
	done

generate-models:
	$(TINYGRAD_PYTHON) ../scripts/gen_model_capabilities.py
	@for dir in $(MODEL_DIRS); do \
		$(TINYGRAD_PYTHON) ../scripts/gen_model_def.py "$$dir" || exit 1; \
	done

# The reference capture in test/baseline was taken in the LEGACY dialect from
# the pre-milestone-4 binary; no current build accepts --inverse-conform, so
# only compare (modular dialect) is replayable against today's executables.
BASELINE_DIR ?= ../test/baseline

baseline-capture:
	@echo "The baseline was captured from the pre-milestone-4 binary." >&2
	@echo "Recapturing needs that binary: git worktree add ../bc-main main" >&2
	@exit 2

baseline-compare: models
	../test/baseline.sh compare modular $(BASELINE_DIR) \
		./$(exe_mindgrab) ./$(exe_model16chan18cls)

# The CPU path is deterministic and is the reference every GPU backend is
# compared against, so it is asserted bit-exact. A tolerance here would only
# hide CPU regressions -- the fp16 GPU tolerances below are a separate matter.
test: models
	$(MAKE) check-generated
	$(MAKE) cuda-modules-check
	python3 ../test/test_cuda_modules.py
	$(MAKE) hip-artifacts-check
	$(MAKE) test-cuda-driver-contract
ifeq ($(CUDA),1)
	$(MAKE) test-cuda-legacy-contract
endif
	$(MAKE) test-models
	$(MAKE) test-cpu-turbo
	$(MAKE) test-cpu-sme
	$(MAKE) test-guards
	$(MAKE) test-label-cleanup
	$(MAKE) test-label-quality
	@if [ -f "$(BC_BRAINCHOP_TEST)/main.js" ] && command -v node >/dev/null 2>&1; then \
		$(MAKE) test-label-resample; \
	else echo "test-label-resample: browser oracle SKIPPED (requires node and BC_BRAINCHOP_TEST/main.js)"; fi
	$(MAKE) test-cat-lite

.PHONY: test-cpu-sme
test-cpu-sme: models | $(BC_TMP)
	for m in MINDMAP:mindmap MINDGRAB:mindgrab MODEL16CHAN18CLS:model16chan18cls; do set -- $$(echo $$m | tr : ' '); \
		$(CC) $(CPPFLAGS) $(CFLAGS) -DBC_MODEL_IS_$$1=1 -Imodels/$$2 -I. ../test/cpu_sme_check.c \
			bc_threads.c model_specs.c cat_lite.c $(LDFLAGS) -lm -o $(BC_TMP)/cpu-sme-check-$$2$(EXEEXT) && \
		$(BC_TMP)/cpu-sme-check-$$2$(EXEEXT) || exit 1; done

# CPU turbo (default): integer kernels match the scalar reference; SME uses an
# fp32 oracle. Both spaces share a quality gate, skipped loudly without
# i8mm/VNNI (mt_select gates SME behind i8mm too). Every bit-exact CPU case
# elsewhere passes --no-turbo.
CPU_TURBO_GATE = 0.995 99.5
test-cpu-turbo: models | $(BC_TMP)
	for m in MINDMAP:mindmap:24 MINDGRAB:mindgrab:16; do set -- $$(echo $$m | tr : ' '); \
		$(CC) $(CPPFLAGS) $(CFLAGS) -DBC_MODEL_IS_$$1=1 -Imodels/$$2 -I. ../test/cpu_turbo_check.c \
			bc_threads.c model_specs.c cat_lite.c $(LDFLAGS) -lm -o $(BC_TMP)/cpu-turbo-check-$$3$(EXEEXT) && \
		$(BC_TMP)/cpu-turbo-check-$$3$(EXEEXT) || exit 1; done
	./$(exe_mindmap) -cpu 0 --save-conform --legacy-cleanup -o $(BC_TMP)/bc-ct-mm.nii.gz $(TEST_IN) \
		> $(BC_TMP)/bc-ct.log 2>&1
	@if grep -q 'lacks i8mm' $(BC_TMP)/bc-ct.log; then \
		echo "test-cpu-turbo: gates SKIPPED (this CPU has no i8mm/VNNI)"; exit 0; fi; \
	grep -q 'CPU turbo (' $(BC_TMP)/bc-ct.log || { echo "test-cpu-turbo: turbo did not run"; exit 1; }; \
	set -ex; \
	python3 ../test/nii_dice.py $(TEST_MM) $(BC_TMP)/bc-ct-mm.nii.gz $(CPU_TURBO_GATE); \
	./$(exe_mindmap) -cpu 0 --legacy-cleanup -o $(BC_TMP)/bc-ct-mmn.nii.gz $(TEST_2MM); \
	python3 ../test/nii_dice.py $(TEST_MM_NATIVE) $(BC_TMP)/bc-ct-mmn.nii.gz $(CPU_TURBO_GATE); \
	./$(exe_mindsnap) -cpu 0 --save-conform -o $(BC_TMP)/bc-ct-ms.nii.gz $(TEST_IN); \
	python3 ../test/nii_dice.py $(TEST_MS) $(BC_TMP)/bc-ct-ms.nii.gz $(CPU_TURBO_GATE); \
	./$(exe_mindsnap) -cpu 0 -o $(BC_TMP)/bc-ct-msn.nii.gz $(TEST_2MM); \
	python3 ../test/nii_dice.py $(TEST_MS_NATIVE) $(BC_TMP)/bc-ct-msn.nii.gz $(CPU_TURBO_GATE); \
	./$(exe_model16chan18cls) -cpu 0 --save-conform -o $(BC_TMP)/bc-ct-m16.nii.gz $(TEST_IN); \
	python3 ../test/nii_dice.py $(TEST_M16) $(BC_TMP)/bc-ct-m16.nii.gz $(CPU_TURBO_GATE); \
	./$(exe_model16chan18cls) -cpu 0 -o $(BC_TMP)/bc-ct-m16n.nii.gz $(TEST_2MM); \
	python3 ../test/nii_dice.py $(TEST_M16_NATIVE) $(BC_TMP)/bc-ct-m16n.nii.gz $(CPU_TURBO_GATE); \
	: MindGrab keeps the input datatype, so its uint8 masks are gated; \
	for input in $(TEST_IN) $(TEST_2MM); do \
		./$(exe_mindgrab) -cpu 0 --no-turbo --mask $(BC_TMP)/bc-ct-mg-ref.nii.gz -o $(BC_TMP)/bc-ct-mg-img.nii.gz "$$input"; \
		./$(exe_mindgrab) -cpu 0 --mask $(BC_TMP)/bc-ct-mg.nii.gz -o $(BC_TMP)/bc-ct-mg-timg.nii.gz "$$input"; \
		python3 ../test/nii_dice.py $(BC_TMP)/bc-ct-mg-ref.nii.gz $(BC_TMP)/bc-ct-mg.nii.gz $(CPU_TURBO_GATE); \
	done; \
	: On SME machines also gate the int8 kernels end to end, at both strides; \
	if grep -q 'SME)' $(BC_TMP)/bc-ct.log; then \
		BRAINCHOPC_CPU_SME=0 ./$(exe_mindmap) -cpu 0 --legacy-cleanup -o $(BC_TMP)/bc-ct-mmn8.nii.gz $(TEST_2MM) \
			> $(BC_TMP)/bc-ct8.log 2>&1; \
		grep -q 'CPU turbo (int8' $(BC_TMP)/bc-ct8.log; \
		python3 ../test/nii_dice.py $(TEST_MM_NATIVE) $(BC_TMP)/bc-ct-mmn8.nii.gz $(CPU_TURBO_GATE); \
		BRAINCHOPC_CPU_SME=0 ./$(exe_model16chan18cls) -cpu 0 -o $(BC_TMP)/bc-ct-m16n8.nii.gz $(TEST_2MM); \
		python3 ../test/nii_dice.py $(TEST_M16_NATIVE) $(BC_TMP)/bc-ct-m16n8.nii.gz $(CPU_TURBO_GATE); \
	fi

# Rewrites one model's frozen CPU fixtures after a deliberate weights change,
# with the exact commands test-models replays. GPU budgets are not covered,
# and test-wasm also compares labels24 exactly: re-run those gates. model16 is
# x86-only: there TEST_M16_NATIVE is the shared x86/wasm fixture, while arm64
# output differs by one voxel from both.
fix_conf_model16chan18cls = $(TEST_M16)
fix_conf_mindmap = $(TEST_MM)
fix_conf_mindsnap = $(TEST_MS)
fix_native_mindgrab = $(TEST_MG)
fix_native_model16chan18cls = $(TEST_M16_NATIVE)
fix_native_mindmap = $(TEST_MM_NATIVE)
fix_native_mindsnap = $(TEST_MS_NATIVE)
fix_in_mindgrab = $(TEST_IN)
fix_flags_mindmap = --legacy-cleanup
refresh-fixtures: models
	@test -n "$(fix_native_$(MODEL))" || { echo "usage: make refresh-fixtures MODEL=<$(MODELS)>" >&2; exit 1; }
	$(if $(filter model16chan18cls,$(MODEL)),$(if $(filter x86_64,$(UNAME_M)),,@echo "refresh-fixtures: model16 fixtures are refreshed on x86-64 only" >&2; exit 1))
	$(if $(fix_conf_$(MODEL)),./$(exe_$(MODEL)) -cpu 0 --no-turbo --save-conform -o $(fix_conf_$(MODEL)) $(TEST_IN) $(fix_flags_$(MODEL)))
	./$(exe_$(MODEL)) -cpu 0 --no-turbo -o $(fix_native_$(MODEL)) $(or $(fix_in_$(MODEL)),$(TEST_2MM)) $(fix_flags_$(MODEL))

# Each executable must reproduce the references bit-exactly, offer exactly the
# options its model supports, and refuse every option it does not.
test-models: models $(TEST_MG) $(TEST_M16) $(TEST_M16_NATIVE) $(TEST_MM) $(TEST_MM_NATIVE) $(TEST_MS) $(TEST_MS_NATIVE) | $(BC_TMP)
	./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-mg-test.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(TEST_MG) $(BC_TMP)/bc-mg-test.nii.gz 0
	./$(exe_mindgrab) -cpu 3 --no-turbo -o $(BC_TMP)/bc-mg-threads.nii.gz $(TEST_IN)
	python3 -c 'import gzip; assert gzip.open("$(BC_TMP)/bc-mg-threads.nii.gz","rb").read() == gzip.open("$(BC_TMP)/bc-mg-test.nii.gz","rb").read(), "MindGrab output depends on the thread count"'
	./$(exe_model16chan18cls) -cpu 0 --no-turbo --save-conform -o $(BC_TMP)/bc-m16-conf-test.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(TEST_M16) $(BC_TMP)/bc-m16-conf-test.nii.gz 0
	./$(exe_model16chan18cls) -cpu 0 --no-turbo -o $(BC_TMP)/bc-m16-native-test.nii.gz $(TEST_2MM)
	python3 ../test/nii_diff.py $(TEST_M16_NATIVE) $(BC_TMP)/bc-m16-native-test.nii.gz 0
	# mindmap: the 24-channel sibling. Its fixtures are corroborated rather than
	# self-referential -- both agreed with tinygrad to ZERO voxels, run through
	# brainchop-cli with this model's 5/95 rank normalization substituted for
	# the 2/98 MindGrab form that qnormalize() hardcodes.
	./$(exe_mindmap) -cpu 0 --no-turbo --save-conform -o $(BC_TMP)/bc-mm-conf-test.nii.gz $(TEST_IN) --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM) $(BC_TMP)/bc-mm-conf-test.nii.gz 0
	./$(exe_mindmap) -cpu 0 --no-turbo -o $(BC_TMP)/bc-mm-native-test.nii.gz $(TEST_2MM) --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM_NATIVE) $(BC_TMP)/bc-mm-native-test.nii.gz 0
	./$(exe_mindmap) -cpu 5 --no-turbo --save-conform -o $(BC_TMP)/bc-mm-threads.nii.gz $(TEST_IN) --legacy-cleanup
	python3 -c 'import gzip; assert gzip.open("$(BC_TMP)/bc-mm-threads.nii.gz","rb").read() == gzip.open("$(BC_TMP)/bc-mm-conf-test.nii.gz","rb").read(), "mindmap output depends on the thread count"'
	./$(exe_mindsnap) -cpu 0 --no-turbo --save-conform -o $(BC_TMP)/bc-ms-conf-test.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(TEST_MS) $(BC_TMP)/bc-ms-conf-test.nii.gz 0
	./$(exe_mindsnap) -cpu 0 --no-turbo -o $(BC_TMP)/bc-ms-native-test.nii.gz $(TEST_2MM)
	python3 ../test/nii_diff.py $(TEST_MS_NATIVE) $(BC_TMP)/bc-ms-native-test.nii.gz 0
	./$(exe_mindsnap) -cpu 5 --no-turbo --save-conform -o $(BC_TMP)/bc-ms-threads.nii.gz $(TEST_IN)
	python3 -c 'import gzip; assert gzip.open("$(BC_TMP)/bc-ms-threads.nii.gz","rb").read() == gzip.open("$(BC_TMP)/bc-ms-conf-test.nii.gz","rb").read(), "mindsnap output depends on the thread count"'
	# Provenance stamped into NIfTI-1 descrip header across all four models,
	# including the 8-hex-char SHA-256 prefix of each embedded model.bin.
	python3 -c 'import hashlib, pathlib, sys; sys.path.insert(0, "../test"); import nii_diff; \
		sha8 = lambda m: hashlib.sha256(pathlib.Path("models/" + m + "/model.bin").read_bytes()).hexdigest()[:8]; \
		checks = [("$(BC_TMP)/bc-mg-test.nii.gz", "mindgrab/" + sha8("mindgrab") + " cpu exact"), \
			("$(BC_TMP)/bc-m16-conf-test.nii.gz", "mindmap-fast/" + sha8("model16chan18cls") + " cpu exact"), \
			("$(BC_TMP)/bc-mm-conf-test.nii.gz", "mindmap/" + sha8("mindmap") + " cpu exact"), \
			("$(BC_TMP)/bc-ms-conf-test.nii.gz", "mindsnap/" + sha8("mindsnap") + " cpu exact")]; \
		bad = [(p, nii_diff.descrip(nii_diff.read(p)), t) for p, t in checks \
			if not (nii_diff.descrip(nii_diff.read(p)).startswith("brainchop ") \
				and nii_diff.descrip(nii_diff.read(p)).endswith(" " + t))]; \
		assert not bad, f"unexpected NIfTI descrip provenance: {bad}"'
	# Options a sibling executable offers must be refused, not silently ignored:
	# a dropped flag produces a plausible file answering a different question.
	! ./$(exe_mindgrab) -m model16chan18cls -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	! ./$(exe_mindgrab) --list
	! ./$(exe_mindgrab) --save-conform -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	! ./$(exe_mindgrab) --legacy-cleanup -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	! ./$(exe_model16chan18cls) --crop -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	! ./$(exe_model16chan18cls) --mask $(BC_TMP)/bc-absent-mask.nii.gz -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	! ./$(exe_model16chan18cls) --border 5 -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	! ./$(exe_model16chan18cls) --pve -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	./$(exe_model16chan18cls) --pve $(TEST_IN) 2>&1 | grep -q "supported only for mindmap"
	! ./$(exe_mindsnap) --pve -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	! ./$(exe_mindsnap) --crop -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	! ./$(exe_mindgrab) --custom $(BC_TMP) -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	# Retired spellings name their replacement rather than saying "unrecognized".
	! ./$(exe_model16chan18cls) --inverse-conform -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	./$(exe_model16chan18cls) --inverse-conform $(TEST_IN) 2>&1 | grep -q -- --save-conform
	# --help advertises exactly what the executable honours.
	./$(exe_mindgrab) --help | python3 -c 'import sys; t=sys.stdin.read(); assert "--mask" in t and "--border" in t and "--crop" in t, "mindgrab help omits an option it supports"; assert "--save-conform" not in t and "--model" not in t and "--list" not in t and "--legacy-cleanup" not in t, "mindgrab help advertises an option it does not have"'
	./$(exe_model16chan18cls) --help | python3 -c 'import sys; t=sys.stdin.read(); assert "--save-conform" in t and "--legacy-cleanup" in t, "model16 help omits an option it supports"; assert "--mask" not in t and "--border" not in t and "--crop" not in t and "--model" not in t and "--pve" not in t, "model16 help advertises an option it does not have"'
	./$(exe_mindmap) --help | python3 -c 'import sys; t=sys.stdin.read(); assert "--pve" in t and "--legacy-cleanup" in t, "MindMap help omits an option it supports"'
	./$(exe_mindsnap) --help | python3 -c 'import sys; t=sys.stdin.read(); assert "--save-conform" in t and "--legacy-cleanup" in t, "mindsnap help omits an option it supports"; assert "--pve" not in t and "--crop" not in t, "mindsnap help advertises an option it does not have"'

# Link a test harness against model $(1)'s objects, minus its main().
# A link that takes $(CFLAGS) carries -flto, so it must also take $(LDFLAGS)
# (the probed -fuse-ld=lld) and $(1)_LINK_EXTRA (the CUDA DLL's import library
# under Windows CUDA=1). Omitting either breaks only a Windows build, which is
# how it was fixed recipe by recipe until it became this macro.
bc_link_model = $(CC) $(CPPFLAGS) $(CFLAGS) -I. -Imodels/$(1) -D$(sel_$(1))=1 \
	$(2) $(filter-out build/$(1)/brainchopc.o,$($(1)_OBJS)) \
	$($(1)_LINK_EXTRA) $(LDFLAGS) $(LDLIBS)

.PHONY: test-label-quality
test-label-quality: models | $(BC_TMP)
	$(CC) $(CFLAGS) -I. ../test/label_quality_check.c bwlabel.c $(LDFLAGS) -lm -o $(BC_TMP)/label-quality-check$(EXEEXT)
	$(BC_TMP)/label-quality-check$(EXEEXT)
	$(call bc_link_model,mindmap,../test/label_failure_check.c) \
		-o $(BC_TMP)/label-failure-check$(EXEEXT)
	$(BC_TMP)/label-failure-check$(EXEEXT) $(TEST_IN)

.PHONY: test-label-resample label-resample-check
label-resample-check: models | $(BC_TMP)
	$(call bc_link_model,mindmap,../test/label_resample_check.c) \
		-o $(BC_TMP)/label-resample-check$(EXEEXT)
test-label-resample: label-resample-check
	node ../test/label_resample_check.mjs $(BC_TMP)/label-resample-check$(EXEEXT) $(BC_BRAINCHOP_TEST)

# Exit-code contracts. These are the paths where a silent success is worst:
# a mistyped invocation that does nothing, or an unusable image that produces a
# full-size volume of garbage.
test-guards: models | $(BC_TMP)
	./$(exe_mindgrab) >/dev/null                 # bare invocation prints usage, exits 0
	./$(exe_mindgrab) --help >/dev/null
	! ./$(exe_mindgrab) --mask $(TEST_IN)        # optional-value flag ate the input
	! ./$(exe_mindgrab) --crop 101 $(TEST_IN)
	! ./$(exe_mindgrab) -cpu nope -o $(BC_TMP)/bc-invalid.nii.gz $(TEST_IN)
	! ./$(exe_mindgrab) -backend nonexistent -o $(BC_TMP)/bc-invalid.nii.gz $(TEST_IN)
	python3 -c 'import struct; h=bytearray(352); struct.pack_into("<i",h,0,348); struct.pack_into("<8h",h,40,3,8,8,8,1,1,1,1); struct.pack_into("<h",h,70,16); struct.pack_into("<h",h,72,32); struct.pack_into("<8f",h,76,0,1,1,1,1,1,1,1); struct.pack_into("<f",h,108,352.0); h[344:348]=b"n+1\0"; open("$(BC_TMP)/bc-nan-test.nii","wb").write(bytes(h)+struct.pack("<f",float("nan"))*512)'
	! ./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-nan-out.nii.gz $(BC_TMP)/bc-nan-test.nii
	python3 -c 'import struct; h=bytearray(352); struct.pack_into("<i",h,0,348); struct.pack_into("<8h",h,40,3,8,8,8,1,1,1,1); struct.pack_into("<h",h,70,16); struct.pack_into("<h",h,72,32); struct.pack_into("<8f",h,76,0,1,1,1,1,1,1,1); struct.pack_into("<f",h,108,352.0); h[344:348]=b"n+1\0"; open("$(BC_TMP)/bc-zero-test.nii","wb").write(bytes(h)+struct.pack("<f",0.0)*512)'
	! ./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-zero-out.nii.gz $(BC_TMP)/bc-zero-test.nii
	./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-zero-out.nii.gz $(BC_TMP)/bc-zero-test.nii 2>&1 | grep -q "constant"
	python3 -c 'import struct; h=bytearray(352); struct.pack_into("<i",h,0,348); struct.pack_into("<8h",h,40,4,8,8,8,2,1,1,1); struct.pack_into("<h",h,70,16); struct.pack_into("<h",h,72,32); struct.pack_into("<8f",h,76,0,1,1,1,1,1,1,1); struct.pack_into("<f",h,108,352.0); h[344:348]=b"n+1\0"; open("$(BC_TMP)/bc-4d-test.nii","wb").write(bytes(h)+struct.pack("<f",1.0)*1024)'
	! ./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-4d-out.nii.gz $(BC_TMP)/bc-4d-test.nii
	./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-4d-out.nii.gz $(BC_TMP)/bc-4d-test.nii 2>&1 | grep -q "4D (8x8x8x2)"
	! ./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-mgz-out.nii.gz $(BC_TMP)/nonexistent.mgz
	./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-mgz-out.nii.gz $(BC_TMP)/nonexistent.mgz 2>&1 | grep -q "unsupported format"
	python3 -c 'import struct; h=bytearray(352); struct.pack_into("<i",h,0,348); struct.pack_into("<8h",h,40,3,8,8,8,1,1,1,1); struct.pack_into("<h",h,70,16); struct.pack_into("<h",h,72,32); struct.pack_into("<8f",h,76,0,0,0,0,0,0,0,0); struct.pack_into("<f",h,108,352.0); struct.pack_into("<hh",h,252,0,1); struct.pack_into("<12f",h,280,1,0,0,0, 1,0,0,0, 0,0,0,0); h[344:348]=b"n+1\0"; open("$(BC_TMP)/bc-singular-test.nii","wb").write(bytes(h)+b"".join(struct.pack("<f",float(i)) for i in range(512)))'
	! ./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-singular-out.nii.gz $(BC_TMP)/bc-singular-test.nii
	./$(exe_mindgrab) -cpu 0 --no-turbo --crop -o $(BC_TMP)/bc-crop-test.nii.gz $(TEST_IN)

# RMSE <= 2/255 over the nonzero union, identical headers, per tissue map.
PVE_COMPARE = -c 'import sys; sys.path.insert(0, "../test"); from cuda_pve_check import compare; \
	d, a, b = sys.argv[1:]; [print(a, t, compare("%s/%s_%s.nii.gz" % (d, a, t), "%s/%s_%s.nii.gz" % (d, b, t))) \
	for t in ("gm", "wm", "csf", "brain")]'

test-metal: models | $(BC_TMP)
ifeq ($(METAL),1)
	./$(exe_mindgrab) -backend metal --no-turbo -o $(BC_TMP)/bc-metal-mg.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(TEST_MG) $(BC_TMP)/bc-metal-mg.nii.gz 64
	./$(exe_model16chan18cls) -cpu 0 --no-turbo --save-conform -o $(BC_TMP)/bc-m16-conf-test.nii.gz $(TEST_IN)
	./$(exe_model16chan18cls) -backend metal --no-turbo --save-conform -o $(BC_TMP)/bc-metal-m16.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(BC_TMP)/bc-m16-conf-test.nii.gz $(BC_TMP)/bc-metal-m16.nii.gz 1024
	# The DEFAULT returns to the input grid; assert that on an input whose grid
	# is not already the conformed one, since t1_crop cannot tell them apart.
	./$(exe_model16chan18cls) -backend metal --no-turbo -o $(BC_TMP)/bc-metal-m16-native.nii.gz $(TEST_2MM)
	python3 -c 'import gzip,struct; b=gzip.open("$(BC_TMP)/bc-metal-m16-native.nii.gz","rb").read(); i=gzip.open("$(TEST_2MM)","rb").read(); assert struct.unpack_from("<3h",b,42) == struct.unpack_from("<3h",i,42), "default output did not return to the input grid"; assert struct.unpack_from("<hh",b,68) == (1002,2)'
	# mindmap against its frozen CPU fixture, not a same-run CPU pass: the
	# fixture is the corroborated one, so this gates the Metal kernels against
	# the reference rather than against whatever the CPU did today.
	./$(exe_mindmap) -backend metal --no-turbo --save-conform -o $(BC_TMP)/bc-metal-mm.nii.gz $(TEST_IN) --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM) $(BC_TMP)/bc-metal-mm.nii.gz 1024
	./$(exe_mindmap) -backend metal --no-turbo -o $(BC_TMP)/bc-metal-mm-native.nii.gz $(TEST_2MM) --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM_NATIVE) $(BC_TMP)/bc-metal-mm-native.nii.gz 256
	# 104-way argmax: many more near-tied logit pairs than 18-way, hence the
	# wider budgets. See the project's budget table for the measured counts.
	./$(exe_mindsnap) -backend metal --no-turbo --save-conform -o $(BC_TMP)/bc-metal-ms.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(TEST_MS) $(BC_TMP)/bc-metal-ms.nii.gz $(MS_TOL)
	./$(exe_mindsnap) -backend metal --no-turbo -o $(BC_TMP)/bc-metal-ms-native.nii.gz $(TEST_2MM)
	python3 ../test/nii_diff.py $(TEST_MS_NATIVE) $(BC_TMP)/bc-metal-ms-native.nii.gz $(MS_TOL_NATIVE)
	# --pve against a same-run CPU pass: there is no frozen PVE fixture. The
	# native case runs under AUTO selection, asserting that it picks Metal.
	./$(exe_mindmap) -cpu 0 --no-turbo --pve --save-conform -o $(BC_TMP)/bc-pve-cpu.nii.gz $(TEST_IN)
	./$(exe_mindmap) -backend metal --pve --save-conform -o $(BC_TMP)/bc-pve-metal.nii.gz $(TEST_IN)
	./$(exe_mindmap) -cpu 0 --no-turbo --pve -o $(BC_TMP)/bc-pve-cpu-native.nii.gz $(TEST_2MM)
	# A log, not a pipe: `| grep -q` exits at the match and SIGPIPE killed the
	# run before it wrote the CSF map, with grep's status masking the failure.
	./$(exe_mindmap) --pve -o $(BC_TMP)/bc-pve-metal-native.nii.gz $(TEST_2MM) \
		2>$(BC_TMP)/bc-pve-auto.log
	grep -q "using metal backend" $(BC_TMP)/bc-pve-auto.log
	! grep -q "falling back" $(BC_TMP)/bc-pve-auto.log  # else CPU vs CPU passes
	for pair in "bc-pve-cpu bc-pve-metal" "bc-pve-cpu-native bc-pve-metal-native"; do \
		python3 $(PVE_COMPARE) $(BC_TMP) $$pair || exit 1; done
	# Turbo (8-bit, the default): brainchop-test's gates, macro Dice and
	# foreground agreement against the same fixtures, conformed and native.
	./$(exe_mindgrab) -backend metal -o $(BC_TMP)/bc-turbo-mg.nii.gz $(TEST_IN)
	python3 ../test/nii_dice.py $(TEST_MG) $(BC_TMP)/bc-turbo-mg.nii.gz $(TURBO_GATE_MG)
	./$(exe_model16chan18cls) -backend metal --save-conform -o $(BC_TMP)/bc-turbo-m16.nii.gz $(TEST_IN)
	python3 ../test/nii_dice.py $(TEST_M16) $(BC_TMP)/bc-turbo-m16.nii.gz $(TURBO_GATE)
	./$(exe_model16chan18cls) -backend metal -o $(BC_TMP)/bc-turbo-m16n.nii.gz $(TEST_2MM)
	python3 ../test/nii_dice.py $(TEST_M16_NATIVE) $(BC_TMP)/bc-turbo-m16n.nii.gz $(TURBO_GATE)
	./$(exe_mindmap) -backend metal --save-conform --legacy-cleanup -o $(BC_TMP)/bc-turbo-mm.nii.gz $(TEST_IN)
	python3 ../test/nii_dice.py $(TEST_MM) $(BC_TMP)/bc-turbo-mm.nii.gz $(TURBO_GATE)
	./$(exe_mindmap) -backend metal --legacy-cleanup -o $(BC_TMP)/bc-turbo-mmn.nii.gz $(TEST_2MM)
	python3 ../test/nii_dice.py $(TEST_MM_NATIVE) $(BC_TMP)/bc-turbo-mmn.nii.gz $(TURBO_GATE)
	./$(exe_mindsnap) -backend metal --save-conform -o $(BC_TMP)/bc-turbo-ms.nii.gz $(TEST_IN)
	python3 ../test/nii_dice.py $(TEST_MS) $(BC_TMP)/bc-turbo-ms.nii.gz $(TURBO_GATE)
	./$(exe_mindsnap) -backend metal -o $(BC_TMP)/bc-turbo-msn.nii.gz $(TEST_2MM)
	python3 ../test/nii_dice.py $(TEST_MS_NATIVE) $(BC_TMP)/bc-turbo-msn.nii.gz $(TURBO_GATE)
	python3 -c 'import hashlib, pathlib, sys; sys.path.insert(0, "../test"); import nii_diff; \
		sha8 = lambda m: hashlib.sha256(pathlib.Path("models/" + m + "/model.bin").read_bytes()).hexdigest()[:8]; \
		checks = [("$(BC_TMP)/bc-metal-mg.nii.gz", "mindgrab/" + sha8("mindgrab") + " metal exact"), \
			("$(BC_TMP)/bc-turbo-mg.nii.gz", "mindgrab/" + sha8("mindgrab") + " metal turbo"), \
			("$(BC_TMP)/bc-pve-metal_gm.nii.gz", "mindmap/" + sha8("mindmap") + " metal pve")]; \
		bad = [(p, nii_diff.descrip(nii_diff.read(p)), t) for p, t in checks \
			if not (nii_diff.descrip(nii_diff.read(p)).startswith("brainchop ") \
				and nii_diff.descrip(nii_diff.read(p)).endswith(" " + t))]; \
		assert not bad, f"unexpected Metal NIfTI descrip provenance: {bad}"'
else
	@echo "Metal backend is not enabled for this build"
endif

# Accelerator voxel budgets.
#
# These were fitted before any of this code had been RUN -- CUDA had never been
# compiled by anything, and the model16 figure of 16777 is exactly 0.1% of the
# voxel count, identical for two backends that supposedly differed 64x on
# MindGrab. Measured on an RTX 4070 Ti SUPER (driver 595.84, nvcc 12.0, OpenCL
# 3.0 CUDA 13.2.87) against the CPU reference, both models:
#
#   CUDA   MindGrab 0    model16 0
#   OpenCL MindGrab 0    model16 1
#
# So the old budgets could not fail. The values below keep real headroom over
# what was measured while being tight enough that an actual numeric break --
# which moves thousands of voxels, not tens -- fails instead of passing.
#
# Why the two models differ: MindGrab emits a binary mask, so a perturbed
# activation has to cross the threshold to change a voxel, whereas model16 is an
# 18-class argmax where two near-tied logits swap on the last bit. The label
# model legitimately needs more room than the mask model.
#
# Why OpenCL keeps more room than CUDA on MindGrab: backend_opencl.c still
# accumulates the GroupNorm moments in fp32, forming the variance as the
# cancellation-prone E[x^2]-E[x]^2, where CUDA and the CPU use fp64. That is
# unfixed. It does not bite here, but this device is the WRONG one to conclude
# from -- CUDA and OpenCL both run through the same NVIDIA driver stack, so
# nothing here tests an Intel or AMD ICD.
#
# If a future build moves these backends to fp16 -- which is the precision these
# models are designed for, and what Metal and WebGPU already use -- these
# budgets do not carry over. Use the fp16 figures instead: 64 for MindGrab and
# 1024 for model16, as test-metal and test-wasm-gpu do.
# The CUDA backend is fp16 as of the tensor-core work, so these are the FP16
# budgets that test-metal and test-wasm-gpu already carry, not the fp32 ones
# they replace (16 and 256). the optimization pass prescribed exactly this swap before the
# change was made. Measured on an RTX 4070 Ti SUPER, sm_89, tensor-core path:
# MindGrab 36 of 16.7M against 64, model16 469 against 1024 -- 56% and 46% of
# budget, comparable to Metal's 58%/64%.
#
# The split between them is unchanged and still derived rather than fitted:
# MindGrab emits a binary mask, so an activation must cross a threshold to move
# a voxel, while model16 is an 18-class argmax where two near-tied logits swap
# on the last bit.
CUDA_TOL_MINDGRAB ?= 64
CUDA_TOL_MODEL16 ?= 1024
CUDA_TOL_MINDMAP ?= 1024
CUDA_TOL_MINDMAP_NATIVE ?= 256
# OpenCL is fp16 as of the vload_half work, so these are the FP16 budgets that
# test-metal, test-wasm-gpu and now test-cuda carry, not the fp32 ones they
# replace (64 and 256). Measured on an RTX 4070 Ti SUPER: MindGrab 31 of 16.7M
# against 64, model16 460 against 1024.
OPENCL_TOL_MINDGRAB ?= 64
OPENCL_TOL_MODEL16 ?= 1024

test-cuda: models | $(BC_TMP)
ifneq ($(filter 1 driver,$(CUDA)),)
	./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-cpu-mg.nii.gz $(TEST_IN)
	./$(exe_mindgrab) -backend cuda -o $(BC_TMP)/bc-cuda-mg.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(BC_TMP)/bc-cpu-mg.nii.gz $(BC_TMP)/bc-cuda-mg.nii.gz $(CUDA_TOL_MINDGRAB)
	./$(exe_model16chan18cls) -cpu 0 --no-turbo --save-conform -o $(BC_TMP)/bc-cpu-m16.nii.gz $(TEST_IN)
	./$(exe_model16chan18cls) -backend cuda --save-conform -o $(BC_TMP)/bc-cuda-m16.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(BC_TMP)/bc-cpu-m16.nii.gz $(BC_TMP)/bc-cuda-m16.nii.gz $(CUDA_TOL_MODEL16)
	./$(exe_mindmap) -backend cuda --save-conform -o $(BC_TMP)/bc-cuda-mm.nii.gz $(TEST_IN) --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM) $(BC_TMP)/bc-cuda-mm.nii.gz $(CUDA_TOL_MINDMAP)
	./$(exe_mindmap) -backend cuda -o $(BC_TMP)/bc-cuda-mm-native.nii.gz $(TEST_2MM) --legacy-cleanup
	python3 ../test/nii_diff.py $(TEST_MM_NATIVE) $(BC_TMP)/bc-cuda-mm-native.nii.gz $(CUDA_TOL_MINDMAP_NATIVE)
	./$(exe_mindsnap) -backend cuda --save-conform -o $(BC_TMP)/bc-cuda-ms.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(TEST_MS) $(BC_TMP)/bc-cuda-ms.nii.gz $(MS_TOL)
	./$(exe_mindsnap) -backend cuda -o $(BC_TMP)/bc-cuda-ms-native.nii.gz $(TEST_2MM)
	python3 ../test/nii_diff.py $(TEST_MS_NATIVE) $(BC_TMP)/bc-cuda-ms-native.nii.gz $(MS_TOL_NATIVE)

	# --adapter has no other coverage anywhere. Both lines are cheap and neither
	# needs a second GPU: the listing must name at least one adapter for a
	# backend that is compiled in, and an absurd index must be REFUSED rather
	# than silently falling back to device 0 -- which is the failure mode that
	# would quietly run on the wrong GPU.
	#
	# 999, not 9999: cli.c's parser rejects anything above 1024 on sight, so a
	# larger number never reaches bc_backend_adapter_valid and would test the
	# bound instead of the backend. 999 parses, then gets refused by the backend.
	# The tuner's central claim, which nothing asserted until now: every launch
	# shape it can pick produces IDENTICAL output, so tuning changes speed and
	# never results. The search checks this with a device-side checksum, but that
	# is the tuner grading its own homework -- this compares finished
	# segmentations from two independently chosen shapes.
	#
	# XDG_CACHE_HOME is redirected so the test neither reads nor writes the
	# developer's real tune cache: without it `make test-cuda` would mutate state
	# outside the tree and its result would depend on what happened to be cached.
	#
	# The `chose` grep is the VACUITY guard. Without it this passes trivially when
	# the search fails -- the backend then prints "tuning failed; using the
	# compiled-in convolution shape" and both runs are the same shape. Note the
	# gate deliberately does NOT require the tuned shape to DIFFER from the
	# pinned one: on a device where the compiled-in default is genuinely optimal
	# that is the correct outcome, and demanding a difference would fail a
	# healthy machine. cmp is the assertion; the grep only proves a search ran.
	XDG_CACHE_HOME=$(BC_TMP)/tunecache BRAINCHOPC_TUNE=0 \
		./$(exe_mindgrab) -backend cuda -o $(BC_TMP)/bc-cu-pin.nii.gz $(TEST_IN)
	XDG_CACHE_HOME=$(BC_TMP)/tunecache ./$(exe_mindgrab) --tune -backend cuda \
		-o $(BC_TMP)/bc-cu-tuned.nii.gz $(TEST_IN) 2>$(BC_TMP)/bc-cu-tune.log
	grep -q "chose" $(BC_TMP)/bc-cu-tune.log
	cmp $(BC_TMP)/bc-cu-pin.nii.gz $(BC_TMP)/bc-cu-tuned.nii.gz
	@echo "test-cuda: a real search ran and its shape agrees byte-for-byte with the pinned one"
	./$(exe_mindgrab) -a -1 | grep -q .
	! ./$(exe_mindgrab) -backend cuda -a 999 -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
else
	@echo "CUDA backend is not enabled for this build"
endif

test-opencl: models | $(BC_TMP)
ifeq ($(OPENCL),1)
	./$(exe_mindgrab) -cpu 0 --no-turbo -o $(BC_TMP)/bc-cpu-mg.nii.gz $(TEST_IN)
	./$(exe_mindgrab) -backend opencl -o $(BC_TMP)/bc-ocl-mg.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(BC_TMP)/bc-cpu-mg.nii.gz $(BC_TMP)/bc-ocl-mg.nii.gz $(OPENCL_TOL_MINDGRAB)
	./$(exe_model16chan18cls) -cpu 0 --no-turbo --save-conform -o $(BC_TMP)/bc-cpu-m16.nii.gz $(TEST_IN)
	./$(exe_model16chan18cls) -backend opencl --save-conform -o $(BC_TMP)/bc-ocl-m16.nii.gz $(TEST_IN)
	python3 ../test/nii_diff.py $(BC_TMP)/bc-cpu-m16.nii.gz $(BC_TMP)/bc-ocl-m16.nii.gz $(OPENCL_TOL_MODEL16)
	# --adapter has no other coverage anywhere. Both lines are cheap and neither
	# needs a second GPU: the listing must name at least one adapter for a
	# backend that is compiled in, and an absurd index must be REFUSED rather
	# than silently falling back to device 0 -- which is the failure mode that
	# would quietly run on the wrong GPU.
	#
	# 999, not 9999: cli.c's parser rejects anything above 1024 on sight, so a
	# larger number never reaches bc_backend_adapter_valid and would test the
	# bound instead of the backend. 999 parses, then gets refused by the backend.
	# The tuner's central claim, which nothing asserted until now: every launch
	# shape it can pick produces IDENTICAL output, so tuning changes speed and
	# never results. The search checks this with a device-side checksum, but that
	# is the tuner grading its own homework -- this compares finished
	# segmentations from two independently chosen shapes.
	#
	# XDG_CACHE_HOME is redirected so the test neither reads nor writes the
	# developer's real tune cache: without it `make test-opencl` would mutate state
	# outside the tree and its result would depend on what happened to be cached.
	#
	# The `chose` grep is the VACUITY guard. Without it this passes trivially when
	# the search fails -- the backend then prints "tuning failed; using the
	# compiled-in convolution shape" and both runs are the same shape. Note the
	# gate deliberately does NOT require the tuned shape to DIFFER from the
	# pinned one: on a device where the compiled-in default is genuinely optimal
	# that is the correct outcome, and demanding a difference would fail a
	# healthy machine. cmp is the assertion; the grep only proves a search ran.
	XDG_CACHE_HOME=$(BC_TMP)/tunecache BRAINCHOPC_TUNE=0 \
		./$(exe_mindgrab) -backend opencl -o $(BC_TMP)/bc-ocl-pin.nii.gz $(TEST_IN)
	XDG_CACHE_HOME=$(BC_TMP)/tunecache ./$(exe_mindgrab) --tune -backend opencl \
		-o $(BC_TMP)/bc-ocl-tuned.nii.gz $(TEST_IN) 2>$(BC_TMP)/bc-ocl-tune.log
	grep -q "chose" $(BC_TMP)/bc-ocl-tune.log
	cmp $(BC_TMP)/bc-ocl-pin.nii.gz $(BC_TMP)/bc-ocl-tuned.nii.gz
	@echo "test-opencl: a real search ran and its shape agrees byte-for-byte with the pinned one"
	./$(exe_mindgrab) -a -1 | grep -q .
	! ./$(exe_mindgrab) -backend opencl -a 999 -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN)
	# A model too wide for OpenCL must REFUSE an explicit request, naming the
	# reason, never fall back to another backend. The `if` is the exit status
	# (`!` would be exempt from set -e); the grep is the message, which would
	# otherwise read like a machine with no OpenCL driver.
	set -e; $(foreach m,$(MODELS),$(if $(filter 0,$(call narrow_model,$(m))), \
		if ./$(exe_$(m)) -backend opencl -o $(BC_TMP)/bc-absent.nii.gz $(TEST_IN) \
			2>$(BC_TMP)/bc-ocl-refuse.log; then \
			echo "$(m): -backend opencl did not refuse" >&2; exit 1; fi; \
		grep -q "not built into this executable: .*16-channel" $(BC_TMP)/bc-ocl-refuse.log;))
else
	@echo "OpenCL backend is not enabled for this build"
endif


# ------------------------------------------------------------------ benchmark
#
# The fixed matrix in test/benchmark.py: both models x both tracked images x
# every backend this build has, reporting time, peak CPU and GPU memory, and
# voxel agreement against the CPU result.
#
# Deliberately NOT part of `make test`. It runs every cell --repeats times, so
# it takes minutes, and its timings are only meaningful on an otherwise-idle
# machine -- benchmarking records that background load inflates them ~2.5x. `make
# test` must stay a correctness gate that anyone can run at any time.
#
# benchmark-compare is the one to use in anger: it replays the matrix against a
# captured reference and FAILS if voxel agreement got worse, while only
# reporting time deltas, which move with machine load and cannot be asserted on.
BENCH_REFERENCE ?= ../test/benchmark_reference.json
BENCH_ARGS ?=

benchmark: models | $(BC_TMP)
	python3 ../test/benchmark.py --builddir . --cc "$(CC)" --outdir $(BC_TMP) $(BENCH_ARGS)

benchmark-capture: models | $(BC_TMP)
	python3 ../test/benchmark.py --builddir . --cc "$(CC)" --outdir $(BC_TMP) --json $(BENCH_REFERENCE) $(BENCH_ARGS)

benchmark-compare: models | $(BC_TMP)
	python3 ../test/benchmark.py --builddir . --cc "$(CC)" --outdir $(BC_TMP) --compare $(BENCH_REFERENCE) $(BENCH_ARGS)

# ThreadSanitizer over bc_parallel_for, with its own positive control.
#
# NOT part of `make test`: it needs a separate instrumented build of the pool,
# and `make test` must stay a fast correctness gate. It is also not `asan`'s or
# `ubsan`'s neighbour in spirit -- those two pin THREADS=0 and therefore say
# nothing whatever about threading.
#
# Both modes build and run inside ONE recipe -- see the project's sanitizer note
# for why two invocations report a false green.
#
# The CONTROL is the load-bearing half. `race` plants an unsynchronised counter
# and TSan must catch it; if it does not, the gate is not armed and the clean
# run proves nothing.
#
# Each mode is asserted on BOTH its exit status and the presence or absence of a
# report in its log, because neither alone is worth much: an exit code says a
# run failed but not of what, and a log line says a race was seen but not that
# anything failed.
#
# `abort_on_error=0` is the load-bearing TSAN_OPTION and took two tries to find.
# It DEFAULTS TO TRUE ON DARWIN, so TSan reaches Die() through abort(): the
# requested `exitcode=` never applies, the shell sees 134, and make's own /bin/sh
# prints "Abort trap: 6" for the CONTROL run -- a line that reads as a failure in
# the middle of a passing target. Turning it off makes the control exit a
# deterministic 66 with no signal, on both platforms.
#
# No libarcher, no OMPT tool, no suppressions -- the thing OpenMP needed and
# pthreads does not. If a future reader adds TSAN_OPTIONS suppressions here,
# that is the signal something real regressed.
#
# BC_NUM_THREADS=4 is pinned so the gate tests the pool rather than whatever is
# in the developer's environment: BC_NUM_THREADS=1 would give one participant
# and fail a correct pool, and a large value would spawn workers for nothing.
# Four is enough for the rendezvous to prove overlap.
#
# This does NOT compile the engine, only the pool: instrumenting a 256^3 MeshNet
# runs for tens of minutes, which is why the real-workload TSan run recorded in
# manual checking is a by-hand exercise and this is the repeatable gate.
#
# The | $(BC_TMP) is not decoration: BC_TMP is build/tmp on Windows, which does
# not exist in a clean tree, and every other target that writes there carries
# the same order-only prerequisite. Omitting it works on Unix, where BC_TMP is
# /tmp, and fails on exactly the platform nobody here can test.
tsan-threads: | $(BC_TMP)
	@$(CC) -O1 -g -std=c11 -Wall -Wextra -fsanitize=thread -pthread -DBC_THREADS -I. \
		../test/threads_race_check.c bc_threads.c \
		-o $(BC_TMP)/bc-tsan-threads$(EXEEXT) || exit 1
	@echo "  clean mode: TSan must report nothing"
	@BC_NUM_THREADS=4 \
		TSAN_OPTIONS="abort_on_error=0 halt_on_error=0 exitcode=66" \
		$(BC_TMP)/bc-tsan-threads$(EXEEXT) clean \
		>$(BC_TMP)/bc-tsan-clean.log 2>&1 || \
		{ echo "FAIL: clean mode did not pass; the log says why"; \
			cat $(BC_TMP)/bc-tsan-clean.log; exit 1; }
	@if grep -q "WARNING: ThreadSanitizer" $(BC_TMP)/bc-tsan-clean.log; then \
		echo "FAIL: a race was reported for index-disjoint writes"; \
		cat $(BC_TMP)/bc-tsan-clean.log; exit 1; fi
	@grep -q "range covered exactly once" $(BC_TMP)/bc-tsan-clean.log || \
		{ echo "FAIL: the range was not covered exactly once"; \
			cat $(BC_TMP)/bc-tsan-clean.log; exit 1; }
	@echo "  race mode (control): TSan must report the planted race"
	@BC_NUM_THREADS=4 \
		TSAN_OPTIONS="abort_on_error=0 halt_on_error=0 exitcode=66" \
		$(BC_TMP)/bc-tsan-threads$(EXEEXT) race \
		>$(BC_TMP)/bc-tsan-race.log 2>&1; \
		test $$? -eq 66 || { \
			echo "FAIL: the control did not exit 66. Either the planted race"; \
			echo "  went undetected -- in which case the clean run proved"; \
			echo "  nothing -- or the harness failed a precondition. The log"; \
			echo "  below says which:"; \
			cat $(BC_TMP)/bc-tsan-race.log; exit 1; }
	@grep -q "WARNING: ThreadSanitizer: data race" $(BC_TMP)/bc-tsan-race.log || \
		{ echo "FAIL: the control run failed without a data-race report, so it"; \
			echo "  died of something other than the race it exists to plant"; \
			cat $(BC_TMP)/bc-tsan-race.log; exit 1; }
	@echo "tsan-threads: disjoint writes clean, planted race caught"

ubsan: clean
	$(MAKE) CFLAGS="-O1 -g -std=c11 -Wall -Wextra -fsanitize=undefined -fno-sanitize-recover=all" \
		LDFLAGS="-fsanitize=undefined" THREADS=0 METAL=0 CUDA=0 OPENCL=0 models
	@echo "ubsan builds ready as ./$(exe_mindgrab) and ./$(exe_model16chan18cls);"
	@echo "run 'make clean' before a normal build"

# AddressSanitizer, which on Linux brings LeakSanitizer with it at no extra
# cost. That combination is the reason this target is worth having: LeakSan
# does not exist on Apple Silicon at all, so on a developer Mac the only leak
# check available is `leaks` against an ordinary optimized build. Here both run
# together, and detect_leaks defaults to on.
#
# THREADS=0 for the same reason ubsan uses it: it removes any thread-race
# explanation for a report. It is no longer needed to keep LeakSanitizer quiet
# -- that was an OpenMP problem, whose runtime allocated pools reported as
# still-reachable noise at exit. bc_threads.c joins its workers and frees its
# array from an atexit handler, so a threaded build is leak-clean; running the
# sanitizers serial is now a choice about signal, not a workaround.
#
# -fno-omit-frame-pointer so the reported stacks name functions rather than
# addresses; -O1 because -O0 is slower than the sanitizer overhead itself and
# -O2 inlines away the frames that make a report readable.
asan: clean
	$(MAKE) CFLAGS="-O1 -g -fno-omit-frame-pointer -std=c11 -Wall -Wextra -fsanitize=address,undefined -fno-sanitize-recover=all" \
		LDFLAGS="-fsanitize=address,undefined" THREADS=0 METAL=0 CUDA=0 OPENCL=0 models
	@echo "asan builds ready as ./$(exe_mindgrab) and ./$(exe_model16chan18cls);"
	@echo "run 'make clean' before a normal build"

VERSION ?= 1.0.0
MACOS_SIGN_IDENTITY ?=
MACOS_INSTALLER_IDENTITY ?=
NOTARY_PROFILE ?= brainchopc-notary
MACOS_PKG = ../dist/brainchop-$(VERSION)-macos-arm64.pkg

# package_macos.sh asks for this rather than repeating the model list, so the
# package cannot ship a different set of executables from the one just built.
print-model-exes:
	@echo $(MODEL_EXES)

print-release-exes:
	@echo $(DISPATCHER_EXE) $(MODEL_EXES)

# Native release archives contain the CUDA-capable executables, the zero-dep
# brainchop dispatcher, and notices, with embedded modules and no companion
# CUDA runtime. Audit before creating the archive.
.PHONY: native-package
native-package:
ifeq ($(CUDA),driver)
	$(MAKE) OPENCL=0 WINDOWS_STATIC=1 ZSTD=0 models dispatcher cuda-modules-check
	python3 ../scripts/package_native.py --src . --output ../dist --version "$(VERSION)" --dispatcher "$(DISPATCHER_EXE)" $(foreach m,$(CUDA_MODELS),$(exe_$(m)))
else
	@echo "native-package requires the default Linux/Windows CUDA=driver build" >&2
	@exit 1
endif

# Unsigned, for local testing. Gatekeeper will refuse it on another machine.
macos-pkg-adhoc:
	env MACOS_SIGN_IDENTITY=- ../scripts/package_macos.sh "$(VERSION)"

# Two DIFFERENT certificates, both from the same Apple Developer account:
# "Developer ID Application" signs the executables, "Developer ID Installer"
# signs the .pkg. Signing a package with the Application certificate produces a
# package the installer rejects, so both are required here rather than one.
macos-pkg:
	@test -n "$(MACOS_SIGN_IDENTITY)" || { \
		echo "Set MACOS_SIGN_IDENTITY to a Developer ID Application identity or SHA-1." >&2; \
		exit 2; \
	}
	@test -n "$(MACOS_INSTALLER_IDENTITY)" || { \
		echo "Set MACOS_INSTALLER_IDENTITY to a Developer ID Installer identity or SHA-1." >&2; \
		echo "List them with: security find-identity -v" >&2; \
		exit 2; \
	}
	env MACOS_SIGN_IDENTITY="$(MACOS_SIGN_IDENTITY)" \
		MACOS_INSTALLER_IDENTITY="$(MACOS_INSTALLER_IDENTITY)" \
		../scripts/package_macos.sh "$(VERSION)"

macos-notarize:
	../scripts/notarize_macos.sh "$(MACOS_PKG)" "$(NOTARY_PROFILE)"

macos-notary-profile:
	@test -n "$(APPLE_ID)" || { echo "Set APPLE_ID to your Developer Apple ID." >&2; exit 2; }
	@test -n "$(TEAM_ID)" || { echo "Set TEAM_ID to your 10-character Developer Team ID." >&2; exit 2; }
	../scripts/store_notary_credentials.sh "$(NOTARY_PROFILE)" "$(APPLE_ID)" "$(TEAM_ID)"

# Check the notarization credentials FIRST. Without this the missing-profile
# error arrives only after a clean rebuild and a signed package -- several
# minutes spent to fail on something knowable up front.
macos-release: check-notary-profile macos-pkg
	../scripts/notarize_macos.sh "$(MACOS_PKG)" "$(NOTARY_PROFILE)"
	../scripts/verify_macos_pkg.sh "$(MACOS_PKG)"

# ASK NOTARYTOOL, do not guess at a Keychain item.
#
# This probed `security find-generic-password -s com.apple.gke.notary.tool -a
# <profile>`, which current notarytool does not write -- credentials go to a
# data-protection keychain `security` cannot see. A profile that
# store-credentials had just validated therefore failed this check and blocked
# the release, which is the worst shape for a preflight: a false negative on the
# one path it exists to protect.
#
# The cost is a network round trip. That is the right trade here: the check runs
# before a multi-minute build and sign, and notarization needs the network
# anyway, so failing early on no connectivity loses nothing.
check-notary-profile:
	@bc_err=$$(xcrun notarytool history --keychain-profile "$(NOTARY_PROFILE)" \
			2>&1 >/dev/null) || { \
		echo "$$bc_err" >&2; \
		echo "" >&2; \
		echo "Profile '$(NOTARY_PROFILE)' did not resolve. Store it once with:" >&2; \
		echo "  make macos-notary-profile APPLE_ID='you@example.com' TEAM_ID='ABCDE12345'" >&2; \
		echo "That prompts for an APP-SPECIFIC password (appleid.apple.com ->" >&2; \
		echo "Sign-In and Security -> App-Specific Passwords), not your Apple ID password." >&2; \
		exit 2; \
	}
	@echo "notarization credentials valid for profile '$(NOTARY_PROFILE)'"

clean:
	rm -rf build
	rm -f $(MODEL_EXES) $(DISPATCHER_EXE) .build-config
	# The probes above delete their own output on success. These are the
	# leftovers from a probe that failed partway, which is exactly the case
	# where something is being debugged and a stale file would confuse it.
	rm -f $(BC_PROBE_ISA) $(BC_PROBE_CC)
	# The Windows CUDA DLLs sit BESIDE the executables rather than under build/,
	# because that is where Windows' loader looks for them at run time, so
	# removing build/ does not catch them.
	rm -f $(foreach m,$(MODELS),brainchop-cuda-$(m).dll brainchop-cuda-$(m).exp)

# Offline CUDA modules: default Linux/Windows builds embed these artifacts.
# Only generation and explicit CUDA=1 legacy validation need nvcc.
#
# The CAT-lite port, checked against the reference implementation on IDENTICAL
# priors: test/cat_lite_check.c and test/cat_lite_js_check.mjs both consume one
# BRAINCHOPC_PVE_DUMP file, so the network, the normalization and the fp16 path
# are upstream of the comparison and cannot hide a porting error inside a
# tolerance. Needs node and ../brainchop-test; skipped, loudly, without them.
BC_BRAINCHOP_TEST ?= ../../brainchop-test
.PHONY: test-cat-lite
test-cat-lite: models | $(BC_TMP)
ifneq ($(filter mindmap,$(MODELS)),)
	@if [ ! -f "$(BC_BRAINCHOP_TEST)/cat-lite.js" ]; then \
		echo "test-cat-lite: SKIPPED -- no $(BC_BRAINCHOP_TEST)/cat-lite.js." >&2; \
		echo "  The reference tree is a sibling checkout, not a dependency of this" >&2; \
		echo "  repo. Set BC_BRAINCHOP_TEST to run the gate. Failing here would" >&2; \
		echo "  break 'make test' -- the whole safety net -- for a missing sibling." >&2; \
	else $(MAKE) test-cat-lite-run; fi

.PHONY: test-cat-lite-run
test-cat-lite-run: | $(BC_TMP)
	BRAINCHOPC_PVE_DUMP=$(BC_TMP)/bc-pve-dump.bin ./$(exe_mindmap) -cpu 0 --no-turbo --pve \
		--save-conform -o $(BC_TMP)/bc-pve.nii.gz $(TEST_IN)
	$(CC) -std=c11 -O2 -Wall -Wextra -I. -Imodels/mindmap -D$(sel_mindmap)=1 \
		$(CPPFLAGS) ../test/cat_lite_check.c cat_lite.c bwlabel.c bc_threads.c \
		$(CFLAGS) $(LDFLAGS) -lm -o $(BC_TMP)/cat-lite-check$(EXEEXT)
	$(CC) $(CFLAGS) $(CPPFLAGS) -I. -Imodels/mindmap -D$(sel_mindmap)=1 \
		../test/cat_lite_cache_check.c bc_threads.c $(LDFLAGS) -lm -o $(BC_TMP)/cat-cache-check$(EXEEXT)
	$(BC_TMP)/cat-cache-check$(EXEEXT)
	node ../test/cat_lite_cases.mjs $(BC_TMP)/cat-lite-check$(EXEEXT) $(BC_BRAINCHOP_TEST)
	$(BC_TMP)/cat-lite-check$(EXEEXT) $(BC_TMP)/bc-pve-dump.bin $(BC_TMP)/bc-pve-c.bin
	# The fit parallelises supported voxels and folds its reductions serially, the same
	# shape MN_DEFINE_MOMENTS uses. Assert that, rather than assuming it.
	./$(exe_mindmap) -cpu 5 --no-turbo --pve --save-conform -o $(BC_TMP)/bc-pve5.nii.gz $(TEST_IN)
	@for tissue in gm wm csf; do \
		cmp -s $(BC_TMP)/bc-pve_$$tissue.nii.gz $(BC_TMP)/bc-pve5_$$tissue.nii.gz || { \
			echo "--pve $$tissue depends on the thread count" >&2; exit 1; }; \
	done
	@echo "test-cat-lite: --pve is thread-count independent"
	node --max-old-space-size=8192 ../test/cat_lite_js_check.mjs \
		$(BC_TMP)/bc-pve-dump.bin $(BC_TMP)/bc-pve-js.bin $(BC_BRAINCHOP_TEST) \
		$(BC_TMP)/bc-pve-c.bin
	node ../test/cat_lite_volume_cases.mjs $(BC_TMP)/cat-lite-check$(EXEEXT) \
		$(BC_BRAINCHOP_TEST) $(BC_TMP)/bc-pve-dump.bin $(BC_TMP)/cat-lite-volume-cases.json
else
	@echo "test-cat-lite: this build has no model with a PVE mode"
endif

.PHONY: cuda-modules-generate cuda-modules-check cuda-module-smoke-build cuda-module-smoke
hip-artifacts-check:
	python3 ../test/test_hip_evidence.py
	python3 ../scripts/hip_memory.py hip_artifacts
	python3 ../test/test_hip_native_check.py
	python3 ../test/test_hip_artifacts.py
	python3 ../test/test_build_hip_artifacts.py
	@set -e; for manifest in hip_artifacts/*/*/manifest.json; do \
		[ -f "$$manifest" ] || continue; \
		python3 ../scripts/hip_artifacts.py "$$(dirname "$$manifest")"; \
	done

# Native integration requires matching physical AMD hardware; never accept a
# CPU fallback as HIP parity. The classifier compiler is maintainer-only.
.PHONY: test-hip test-hip-classifier test-hip-no-runtime test-hip-no-toolchain test-hip-allocation
test-hip: models | $(BC_TMP)
	@run="$$(mktemp -d "$(BC_TMP)/hip-native.XXXXXX")"; \
		python3 ../test/hip_native_check.py --binaries . --output "$$run/results"

test-hip-classifier: | $(BC_TMP)
	@run="$$(mktemp -d "$(BC_TMP)/hip-classifier.XXXXXX")"; \
		python3 ../test/hip_classifier_check.py --gfx $(HIP_ARCH) --output "$$run/results"

test-hip-no-runtime: models | $(BC_TMP)
	@run="$$(mktemp -d "$(BC_TMP)/hip-no-runtime.XXXXXX")"; \
		python3 ../test/hip_native_check.py --binaries . --hide-runtime --output "$$run/results"

test-hip-no-toolchain: | $(BC_TMP)
	@run="$$(mktemp -d "$(BC_TMP)/hip-no-toolchain.XXXXXX")"; \
		python3 ../test/hip_no_toolchain.py --output "$$run/results"

test-hip-allocation: models | $(BC_TMP)
	@run="$$(mktemp -d "$(BC_TMP)/hip-allocation.XXXXXX")"; \
		python3 ../test/hip_allocation_check.py --real-hip --output "$$run/results"

hip-artifacts-build:
	@test -n "$(HIP_MODEL)" || { echo "set HIP_MODEL (for example mindmap)" >&2; exit 2; }
	@test -n "$(HIP_GFX)" || { echo "set HIP_GFX (for example gfx1151)" >&2; exit 2; }
	@test -n "$(HIP_NUM_CUS)" || { echo "set HIP_NUM_CUS for the target GPU" >&2; exit 2; }
	@test -n "$(HIP_ARTIFACT_OUT)" || { echo "set HIP_ARTIFACT_OUT" >&2; exit 2; }
	python3 ../scripts/build_hip_artifacts.py --model "$(HIP_MODEL)" \
		--gfx "$(HIP_GFX)" --num-compute-units "$(HIP_NUM_CUS)" \
		--output "$(HIP_ARTIFACT_OUT)" $(HIP_BUILD_FLAGS)

cuda-modules-generate:
	python3 ../scripts/build_cuda_modules.py generate --nvcc "$(NVCC)"

cuda-modules-check:
	python3 ../scripts/build_cuda_modules.py check

cuda-module-smoke-build: cuda-modules-check | $(BC_TMP)
	@set -e; for m in $(CUDA_MODELS); do \
		$(CC) -I. -DBC_PAYLOAD_PATH=\"cuda_modules/$$m.fatbin\" \
			-DBC_PAYLOAD_SYM=bc_cuda_module -c payload.S -o "$(BC_TMP)/cuda-module-$$m.o"; \
		$(CC) -std=c11 -O2 -Wall -Wextra -Wpedantic -I. -Imodels/$$m \
			-DBC_MODEL_IS_$$(printf "%s" "$$m" | tr a-z A-Z)=1 ../test/cuda_module_smoke.c \
			"$(BC_TMP)/cuda-module-$$m.o" -lm $(if $(filter Linux,$(UNAME_S)),-ldl,) \
			-o "$(BC_TMP)/cuda-module-smoke-$$m$(EXEEXT)"; \
	done

CUDA_MODULE_ADAPTER ?= 0
.PHONY: test-cuda-legacy-contract
test-cuda-legacy-contract: models | $(BC_TMP)
ifeq ($(CUDA)$(BC_WINDOWS),1)
	@set -e; $(foreach m,$(CUDA_MODELS), \
		$(NVCC) -O3 --use_fast_math -std=c++11 $(CUDA_GENCODE) $(NVCC_HOST_FLAGS) \
			-DHAVE_CUDA -D$(sel_$(m))=1 -I. -Imodels/$(m) \
			../test/cuda_legacy_contract.cu build/$(m)/backend_accel.o \
			build/$(m)/model_specs.o build/$(m)/bc_threads.o -Xcompiler=-pthread \
			-o "$(BC_TMP)/cuda-legacy-contract-$(m)"; \
		"$(BC_TMP)/cuda-legacy-contract-$(m)";)
else ifeq ($(CUDA)$(BC_WINDOWS),11)
	# nvcc's host compiler here is MSVC, which cannot read the clang/LTO objects
	# above, so link the MSVC-ABI objects the CUDA DLL is built from instead --
	# the same three sources, compiled by the same rule, that ship in the DLL.
	@set -e; $(foreach m,$(CUDA_MODELS), \
		$(NVCC) -O3 --use_fast_math $(CUDA_GENCODE) $(NVCC_HOST_FLAGS) \
			-DHAVE_CUDA -D$(sel_$(m))=1 -I. -Imodels/$(m) \
			../test/cuda_legacy_contract.cu build/$(m)/msvc/backend_accel.obj \
			build/$(m)/msvc/model_specs.obj build/$(m)/msvc/bc_threads.obj \
			-o "$(BC_TMP)/cuda-legacy-contract-$(m)$(EXEEXT)"; \
		"$(BC_TMP)/cuda-legacy-contract-$(m)$(EXEEXT)";)
else
	@echo "Legacy dispatch test requires CUDA=1"
endif

.PHONY: test-cuda-driver-contract
test-cuda-driver-contract: cuda-module-smoke-build
	@set -e; $(foreach m,$(CUDA_MODELS), \
		$(CC) -std=c11 -O2 -Wall -Wextra -I. -I models/$(m) -D$(sel_$(m))=1 \
			../test/cuda_driver_contract.c backend_accel.c model_specs.c bc_threads.c \
			"$(BC_TMP)/cuda-module-$(m).o" -lm $(if $(filter Linux,$(UNAME_S)),-ldl,) \
			-o "$(BC_TMP)/cuda-driver-contract-$(m)$(EXEEXT)"; \
		"$(BC_TMP)/cuda-driver-contract-$(m)$(EXEEXT)";)

cuda-module-smoke: cuda-module-smoke-build
	@set -e; for m in $(CUDA_MODELS); do \
		"$(BC_TMP)/cuda-module-smoke-$$m$(EXEEXT)" $(CUDA_MODULE_ADAPTER); \
	done

# Browser-oracle gate is explicit about its external source dependency.
.PHONY: test-label-cleanup
test-label-cleanup: | $(BC_TMP)
	$(CC) -std=c11 -O2 -I. ../test/label_cleanup_check.c bwlabel.c -lm -o $(BC_TMP)/label-cleanup-check$(EXEEXT)
	$(BC_TMP)/label-cleanup-check$(EXEEXT) --selftest
	@if [ -f "$(BC_BRAINCHOP_TEST)/bwlabels.js" ] && command -v node >/dev/null 2>&1; then \
		node ../test/label_cleanup_check.mjs $(BC_TMP)/label-cleanup-check$(EXEEXT) $(BC_BRAINCHOP_TEST); \
	else echo "test-label-cleanup: browser oracle SKIPPED (requires node and BC_BRAINCHOP_TEST/bwlabels.js)"; fi

# Select cuda to exercise actual device inference and the default cleanup.
CLEANUP_BACKEND ?= cpu
.PHONY: test-mindmap-cleanup
test-mindmap-cleanup: models test-label-cleanup
	@test -f "$(BC_BRAINCHOP_TEST)/bwlabels.js" && command -v node >/dev/null
	@set -e; for fixture in $(TEST_IN) $(TEST_2MM); do \
		BRAINCHOPC_LABEL_DUMP=$(BC_TMP)/bc-cleanup-raw.bin \
			./$(exe_mindmap) -backend $(CLEANUP_BACKEND) --save-conform \
			-o $(BC_TMP)/bc-cleanup.nii.gz "$$fixture"; \
		node ../test/label_cleanup_check.mjs $(BC_TMP)/label-cleanup-check$(EXEEXT) \
			$(BC_BRAINCHOP_TEST) $(BC_TMP)/bc-cleanup-raw.bin $(BC_TMP)/bc-cleanup.nii.gz; \
	done
	@rm -f $(BC_TMP)/bc-cleanup-raw.bin
