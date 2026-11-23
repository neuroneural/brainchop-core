brainchop-core: Linux / Windows command-line distribution

Each model executable contains its embedded weights and CUDA kernels. Extract
the archive and run `brainchop` (unified dispatcher) or a model executable
directly (`brainchop-mindgrab`, `brainchop-mindmap-fast`, `brainchop-mindmap`,
or `brainchop-mindsnap`, with `.exe` on Windows):

  brainchop -o labels.nii.gz input.nii.gz
  brainchop -m mindgrab -o stripped.nii.gz input.nii.gz
  brainchop-mindmap --pve -o brain.nii.gz input.nii.gz
  brainchop-mindsnap -o parcels.nii.gz input.nii.gz

`mindmap` supports experimental GM/WM/CSF partial-volume tissue fractions
(`--pve`) on CPU and CUDA. CUDA accelerates the network and priors; the
partial-volume fit runs on CPU. All outputs return to native input space by
default (pass `--save-conform` for conformed 256^3 space).

NVIDIA execution requires a driver exposing CUDA Driver API 12080 or newer and
a supported GPU (compute capability 6.0 or newer). Native kernels cover
sm60/70/75/80/86/89/90; compute90 PTX is included for forward JIT. No CUDA
Toolkit, nvcc, cudart, Python or separate model/kernel files are needed to run.
CPU execution remains available without NVIDIA hardware or a driver:

  brainchop-mindgrab -backend cpu -o mask.nii.gz input.nii.gz

Use -a -1 to inspect devices, -backend cuda to require NVIDIA, and
--tune to retune. Automatic selection uses CPU when no supported driver/device
is discovered. An explicit -backend cuda request fails if CUDA is unavailable
or inference fails.

Windows: executables require only Windows system DLLs. No MSYS2, MSVC, CUDA
runtime or brainchop-cuda DLL needs to be copied alongside them. These archives
are unsigned; signing and any organizational deployment policy are separate.

Linux: system glibc and zlib are required. The archive is built for its host's
glibc and is not a universal Linux binary. Build on the oldest distribution you
intend to support. OpenCL is excluded from these archives to avoid requiring
its loader merely to start the executable.

See build.json for hashes, dependencies and build flags. X86_ISA=x86-64-v3
requires AVX2/FMA; an empty X86_ISA selects the baseline CPU instruction set.
The accompanying .sha256 file verifies the archive download. See LICENSE and
THIRD_PARTY_NOTICES.md for redistribution terms.
