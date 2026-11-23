#!/usr/bin/env python3
"""Fixed benchmark matrix: every model x image x backend, one table.

usage: benchmark.py [--backends cpu,cuda,...] [--repeats N]
                    [--json OUT] [--compare REF] [--builddir DIR]

The point of this file is to be a HARD reference that a future change is
compared against, so the matrix is fixed rather than discovered: the two
tracked acquisitions (t1_crop, already 256^3 at 1 mm; t1_2mm, a real
96x128x128 2 mm INT16 volume) crossed with both models. Those two inputs are
not interchangeable -- for t1_crop the conformed and native output grids
COINCIDE, so only t1_2mm exercises the inverse reslice.

Four numbers per cell:

  time      wall-clock milliseconds: MEDIAN of --repeats timed runs, after one
            discarded warm-up run. The warm-up is not decoration -- the CUDA
            backend autotunes its convolution on the first run against a device
            it has no cached entry for, which costs seconds and happens exactly
            once. Timing that run would measure the tuner, and discarding it is
            cheaper and more honest than suppressing the tuning.

            Median rather than min, because with the warm-up gone the remaining
            spread is ordinary scheduling noise in both directions rather than a
            one-sided tax on a clean floor. the project's warning still applies --
            a loaded machine inflates these ~2.5x, so only compare captures
            taken under the same conditions.

            The CPU rows are timed once. They run for minutes, they are not
            sensitive to the launch-shape details this protocol exists to
            stabilise, and repeating them spends minutes to learn nothing.
  peak CPU  peak resident set size of the child process, via os.wait4, which
            reports the rusage of THAT child rather than a running maximum
            over all children the way RUSAGE_CHILDREN does.
  peak GPU  sampled from nvidia-smi for this pid. Reported as "-" where it is
            not a meaningful number: the CPU backend allocates none, and a
            unified-memory backend (Metal) has no separate GPU pool to report.
  voxels    differing voxels against the CPU result for the SAME model and
            image, which is the reference every accelerator is held to. CPU
            rows are that reference, so they read "ref" rather than 0 -- a 0
            there would suggest a comparison happened when none did.

Stdlib only, and the datatype table is imported from nii_diff rather than
restated, so the two helpers cannot disagree about how to read a volume.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nii_diff  # noqa: E402  (path must be set first)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Windows linkers append .exe whether or not -o asked for it, so the file on
# disk is not the name the Makefile writes. os.name rather than sys.platform:
# this must be true for a native Python run from an MSYS2 shell, where
# sys.platform is still "win32" but the shell makes it easy to assume otherwise.
EXE = ".exe" if os.name == "nt" else ""

# (label, executable) -- the executables are per-model by design; there is no
# --model flag to vary.
MODELS = [
    ("mindgrab", "brainchop-mindgrab" + EXE),
    ("model16", "brainchop-mindmap-fast" + EXE),
    ("mindmap", "brainchop-mindmap" + EXE),
    ("mindsnap", "brainchop-mindsnap" + EXE),
]

# (label, fixture). Default invocation for both models, i.e. native output
# space, because that is what a user actually gets.
IMAGES = [
    ("t1_crop", "t1_crop.nii.gz"),
    ("t1_2mm", "t1_2mm.nii.gz"),
]

CC = "cc"  # overwritten from --cc in main()

# Label -> the arguments that select it. Not just "-backend LABEL", because
# cpuN1 is not a backend: it is the CPU backend pinned to one thread. Single-core
# throughput is what actually characterises the convolution kernel -- the
# all-threads number blends it with scheduling and memory bandwidth -- and it is
# the figure to watch when tuning MN_DEFINE_CONV, so it earns a row of its own.
# The exact engine: the reference for voxel counts and MN_DEFINE_CONV timing.
BACKEND_ARGS = {
    "cpuN1": ["-backend", "cpu", "-cpu", "1", "--no-turbo"],
    "cpu": ["-backend", "cpu", "--no-turbo"],
    "cuda": ["-backend", "cuda"],
    "opencl": ["-backend", "opencl"],
    "metal": ["-backend", "metal"],
    "webgpu": ["-backend", "webgpu"],
}
ALL_BACKENDS = list(BACKEND_ARGS)

# The CPU result every other row is compared against. Run first regardless of
# the order requested, so the reference exists before anything needs it.
REFERENCE_BACKEND = "cpu"

# Rows timed once, with no warm-up. Both CPU rows qualify: they take minutes,
# they have no per-device tuning to warm up, and their variance is small next to
# their duration -- a MindGrab cpuN1 cell is ~3 min, so the full four-run
# protocol would spend nine extra minutes to confirm a number that barely moves.
SINGLE_RUN_BACKENDS = {"cpuN1", "cpu"}

# Runs whose time is thrown away before the timed ones begin. One is enough for
# what this is here for: the CUDA backend's first-run autotune, which is a
# once-per-device cost that must not land in a reported figure.
WARMUP_RUNS = 1

# ru_maxrss is kilobytes on Linux and bytes on macOS/BSD. Nothing in the
# struct says which, so it has to be branched on the platform.
RSS_TO_MB = 1.0 / 1024.0 if sys.platform != "darwin" else 1.0 / (1024.0 * 1024.0)

# Rows with no separate GPU allocation to report: both CPU rows have none, and
# Metal/WebGPU on Apple silicon share the CPU's memory, so a "GPU MB" column
# there would either duplicate the CPU number or read 0. cpuN1 has to be listed
# alongside cpu -- omitting it printed a measured 0.0 for a row that never
# touches a GPU, which reads as "measured none" rather than "not applicable".
NO_GPU_POOL = {"cpu", "cpuN1", "metal", "webgpu"}


class GpuSampler(threading.Thread):
    """Poll nvidia-smi for one pid's compute memory until told to stop.

    Sampling rather than instrumenting the backend: this has to work against
    an unmodified executable, and every run here is seconds long, so a 50 ms
    period cannot miss the allocation plateau.
    """

    def __init__(self, pid, period=0.05):
        super().__init__(daemon=True)
        self.pid = pid
        self.period = period
        self.peak_mb = 0.0
        # Whether any sample ever yielded a NUMBER, as opposed to the query
        # working but returning nothing usable. Windows needs this: under the
        # WDDM driver model nvidia-smi lists the pid but reports its memory as
        # "[N/A]", so every sample fails to parse and the peak stays at its
        # initial 0.0 -- which the table then prints as a measured zero for a
        # backend that in fact allocated ~2 GB. Unknown and zero must not look
        # the same in a reference capture.
        self.measured = False
        # Not _stop: threading.Thread already uses self._stop() internally, and
        # shadowing it with an Event makes join() raise "not callable".
        self._done = threading.Event()

    def run(self):
        query = [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ]
        while not self._done.is_set():
            try:
                out = subprocess.run(
                    query, capture_output=True, text=True, timeout=5
                ).stdout
            except (OSError, subprocess.SubprocessError):
                return  # no nvidia-smi: leave peak at 0, reported as "-"
            for line in out.splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 2 and parts[0] == str(self.pid):
                    try:
                        self.peak_mb = max(self.peak_mb, float(parts[1]))
                        self.measured = True
                    except ValueError:
                        pass  # "[N/A]" on Windows/WDDM; self.measured stays False
            self._done.wait(self.period)

    def stop(self):
        self._done.set()


if os.name == "nt":
    import ctypes
    import ctypes.wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        """PROCESS_MEMORY_COUNTERS, for GetProcessMemoryInfo."""

        _fields_ = [
            ("cb", ctypes.wintypes.DWORD),
            ("PageFaultCount", ctypes.wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]


def _run_once_windows(argv, sample_gpu):
    """Windows equivalent of the posix_spawn path below.

    There is no wait4 here, so peak memory comes from GetProcessMemoryInfo's
    PeakWorkingSetSize. The handle stays valid after the child exits -- Popen
    holds it until garbage collection -- and the counters keep their final
    values, so this reads the true peak rather than a sample.

    PeakWorkingSetSize is the closest Windows analogue of ru_maxrss: both are
    the high-water mark of resident physical memory. It is not identical, so a
    CPU-memory figure is comparable between two Windows captures and only
    roughly so against a Linux one.
    """
    start = time.perf_counter()
    proc = subprocess.Popen(
        argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    sampler = None
    if sample_gpu:
        sampler = GpuSampler(proc.pid)
        sampler.start()
    proc.wait()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if sampler is not None:
        sampler.stop()
        sampler.join(timeout=2)

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    peak_mb = 0.0
    if ctypes.windll.psapi.GetProcessMemoryInfo(
        int(proc._handle), ctypes.byref(counters), counters.cb
    ):
        peak_mb = counters.PeakWorkingSetSize / (1024.0 * 1024.0)
    return (
        elapsed_ms,
        peak_mb,
        sampler.peak_mb if sampler is not None and sampler.measured else None,
        proc.returncode,
    )


def run_once(argv, sample_gpu):
    """Run argv to completion; return (ms, peak_cpu_mb, peak_gpu_mb, status).

    posix_spawn + wait4 rather than subprocess.run because wait4 is the only
    stdlib call that returns the peak RSS of one specific child. Neither exists
    on Windows, which takes the branch above.
    """
    if os.name == "nt":
        return _run_once_windows(argv, sample_gpu)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        actions = [
            (os.POSIX_SPAWN_DUP2, devnull, 1),
            (os.POSIX_SPAWN_DUP2, devnull, 2),
        ]
        start = time.perf_counter()
        pid = os.posix_spawn(argv[0], argv, os.environ, file_actions=actions)
        sampler = None
        if sample_gpu:
            sampler = GpuSampler(pid)
            sampler.start()
        _, status, usage = os.wait4(pid, 0)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
    finally:
        os.close(devnull)
    if sampler is not None:
        sampler.stop()
        sampler.join(timeout=2)
    return (
        elapsed_ms,
        usage.ru_maxrss * RSS_TO_MB,
        sampler.peak_mb if sampler is not None and sampler.measured else None,
        status,
    )


def describe_host():
    """What the capture was taken on.

    Without this a reference file is close to useless: the timings are only
    comparable against the same machine, and the GPU voxel counts are a
    property of a particular device and driver, not of the source tree.
    """
    def first(cmd, pattern=None):
        try:
            done = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        # A failing tool can still print to stdout: nvidia-smi exits 255 with
        # "Failed to initialize NVML: Unknown Error" there (seen when
        # ProgramFiles is absent from the environment), and that line was
        # recorded as the GPU's name.
        if done.returncode != 0:
            return None
        out = done.stdout.strip()
        if pattern:
            for line in out.splitlines():
                if pattern in line:
                    return line.split(":", 1)[-1].strip()
            return None
        return out.splitlines()[0].strip() if out else None

    def cpu_name():
        """The CPU's marketing name, by whatever means the platform offers.

        Windows has neither lscpu nor sysctl, and platform.processor() there
        returns "AMD64 Family 25 Model 97..." -- a family/model string, not a
        name anyone can compare two captures with. The registry holds the real
        one, and winreg is stdlib, so this keeps benchmark.py's no-dependency
        rule. Left empty, a Windows capture recorded a blank cpu field, which
        defeats the point of recording the host at all.
        """
        if os.name == "nt":
            try:
                import winreg

                key = winreg.OpenKey(
                    winreg.HKEY_LOCAL_MACHINE,
                    r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
                )
                with key:
                    return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
            except OSError:
                return platform.processor() or None
        return (
            first(["lscpu"], "Model name")
            or first(["sysctl", "-n", "machdep.cpu.brand_string"])
        )

    return {
        "captured": time.strftime("%Y-%m-%d"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu_name(),
        "gpu": first(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]
        ),
        # Told, not guessed. The Makefile PROBES for a usable clang and may
        # select one that neither $CC nor `cc` names, so reading the
        # environment here reported gcc for binaries clang had built.
        "cc": first([CC, "--version"]),
        "commit": first(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"]),
    }


def voxel_diff(path_a, path_b):
    """Differing voxel count, or None if the two are not comparable."""
    blob_a, blob_b = nii_diff.read(path_a), nii_diff.read(path_b)
    if not nii_diff.header_matches(blob_a, blob_b):
        return None
    a = nii_diff.voxels(blob_a, path_a)
    b = nii_diff.voxels(blob_b, path_b)
    if len(a) != len(b):
        return None
    return sum(x != y for x, y in zip(a, b))


def measure(exe, image, backend, repeats, outdir):
    # The image label is in the name because the matrix runs both images
    # through the same (exe, backend): without it the second image overwrites
    # the first, and anything reading these afterwards silently diffs the
    # wrong pair. benchmark.py itself compares before overwriting, so this is
    # about what is left on disk for a human.
    out = os.path.join(
        outdir,
        f"bench-{os.path.basename(exe)}-{os.path.basename(image).split('.')[0]}"
        f"-{backend}.nii.gz",
    )
    argv = [exe] + BACKEND_ARGS[backend] + ["-o", out, image]
    sample_gpu = backend not in NO_GPU_POOL
    warmups = 0 if backend in SINGLE_RUN_BACKENDS else WARMUP_RUNS
    if backend in SINGLE_RUN_BACKENDS:
        repeats = 1
    repeats = max(1, repeats)  # --repeats 0 would leave median() nothing to take

    samples = []
    peak_cpu = 0.0
    peak_gpu = 0.0
    # Whether GPU memory was ever actually read. Kept separate from peak_gpu so
    # that "the platform cannot tell us" reports as "-" rather than as 0.0; see
    # GpuSampler.measured.
    gpu_measured = False
    for index in range(warmups + repeats):
        ms, cpu_mb, gpu_mb, status = run_once(argv, sample_gpu)
        if status != 0:
            return None  # backend unavailable, or the run failed
        if index >= warmups:
            samples.append(ms)
        # Memory is a maximum over every run including the warm-up: the peak is
        # a property of the allocation pattern, not of the timing protocol.
        peak_cpu = max(peak_cpu, cpu_mb)
        if gpu_mb is not None:
            peak_gpu = max(peak_gpu, gpu_mb)
            gpu_measured = True
    return {
        "ms": statistics.median(samples),
        "cpu_mb": peak_cpu,
        "gpu_mb": peak_gpu if (sample_gpu and gpu_measured) else None,
        "output": out,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backends", default="cpuN1,cpu,cuda,opencl,metal,webgpu")
    # Timed runs per cell. One warm-up runs before these and is discarded, so
    # the default costs four executions and reports the median of three.
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--json", dest="json_out")
    parser.add_argument("--compare")
    parser.add_argument("--builddir", default=os.path.join(ROOT, "src"))
    parser.add_argument("--outdir", default="/tmp")
    # The Makefile passes its resolved $(CC); see describe_host.
    parser.add_argument("--cc", default=os.environ.get("CC", "cc"))
    args = parser.parse_args()
    global CC
    CC = args.cc

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    for backend in backends:
        if backend not in ALL_BACKENDS:
            sys.exit(f"benchmark: unknown backend {backend!r}")
    if REFERENCE_BACKEND not in backends:
        # Every voxel count in the table is defined against the CPU result, so
        # dropping it would silently turn the last column into "-".
        sys.exit("benchmark: the cpu backend is the reference and cannot be omitted")
    # Reference first, whatever order was asked for: cpuN1 is listed before cpu
    # by default, and comparing against a reference that does not exist yet
    # would report every row as incomparable.
    ordered = [REFERENCE_BACKEND] + [b for b in backends if b != REFERENCE_BACKEND]

    rows = []
    for model_label, exe_name in MODELS:
        exe = os.path.join(args.builddir, exe_name)
        if not os.path.exists(exe):
            sys.exit(f"benchmark: {exe} not built; run `make -C src models` first")
        for image_label, fixture in IMAGES:
            image = os.path.join(HERE, fixture)
            reference = None
            group = {}
            for backend in ordered:
                result = measure(exe, image, backend, args.repeats, args.outdir)
                if result is None:
                    continue  # backend not available in this build
                if backend == REFERENCE_BACKEND:
                    reference = result["output"]
                    differing = "ref"
                else:
                    count = voxel_diff(reference, result["output"])
                    differing = "incomparable" if count is None else count
                group[backend] = {
                    "model": model_label,
                    "image": image_label,
                    "backend": backend,
                    "ms": round(result["ms"], 1),
                    "cpu_mb": round(result["cpu_mb"], 1),
                    "gpu_mb": (
                        None
                        if result["gpu_mb"] is None
                        else round(result["gpu_mb"], 1)
                    ),
                    "voxels_differ": differing,
                }
                print(f"  ran {model_label:9s} {image_label:8s} {backend}", file=sys.stderr)
            # Emitted in the order the caller ASKED for, not the order they had
            # to run in: the reference has to go first to exist, but cpuN1
            # belongs next to cpu in the table.
            rows.extend(group[b] for b in backends if b in group)

    header = ["model", "image", "backend", "time (ms)", "peak CPU (MB)",
              "peak GPU (MB)", "voxels differ"]
    body = [
        [
            row["model"], row["image"], row["backend"],
            f"{row['ms']:.1f}", f"{row['cpu_mb']:.1f}",
            "-" if row["gpu_mb"] is None else f"{row['gpu_mb']:.1f}",
            str(row["voxels_differ"]),
        ]
        for row in rows
    ]
    widths = [max(len(r[i]) for r in [header] + body) for i in range(len(header))]

    def emit(cells):
        print("| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |")

    emit(header)
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for cells in body:
        emit(cells)

    if args.json_out:
        with open(args.json_out, "w") as handle:
            json.dump(
                {
                    "host": describe_host(),
                    "repeats": args.repeats,
                    # What the "time" column MEANS, not just how many samples
                    # fed it. Without this a median-after-warmup capture and the
                    # older min-of-3 capture are indistinguishable in the file,
                    # which is the same trap the host block exists to close.
                    "protocol": {
                        "statistic": "median",
                        "warmups": WARMUP_RUNS,
                        "single_run": sorted(SINGLE_RUN_BACKENDS),
                    },
                    "rows": rows,
                },
                handle,
                indent=2,
            )
            handle.write("\n")
        print(f"\nwrote {args.json_out}", file=sys.stderr)

    if args.compare:
        with open(args.compare) as handle:
            reference = json.load(handle)
        host = reference.get("host", {})
        reference_rows = {
            (r["model"], r["image"], r["backend"]): r
            for r in reference.get("rows", [])
        }
        print("\ncompared against", args.compare)
        # Say what the reference was captured on. Timings from a different
        # machine are not a regression signal, and silently comparing them as
        # if they were is the main way a benchmark starts lying.
        if host:
            print(f"  reference host: {host.get('cpu')} / {host.get('gpu')}"
                  f" / {host.get('captured')} / {host.get('commit')}")
        regressed = False
        for row in rows:
            key = (row["model"], row["image"], row["backend"])
            was = reference_rows.get(key)
            if was is None:
                print(f"  NEW      {key}")
                continue
            # Voxel counts are a correctness contract: any increase is a
            # regression. Times are not -- they move with machine load -- so
            # they are reported, never failed on.
            delta_ms = row["ms"] - was["ms"]
            note = f"  {key} {was['ms']:.0f} -> {row['ms']:.0f} ms ({delta_ms:+.0f})"
            if row["voxels_differ"] != was["voxels_differ"]:
                note += (f"  VOXELS {was['voxels_differ']} ->"
                         f" {row['voxels_differ']}")
                if isinstance(row["voxels_differ"], int) and isinstance(
                    was["voxels_differ"], int
                ):
                    regressed = regressed or row["voxels_differ"] > was["voxels_differ"]
                else:
                    regressed = True
            print(note)
        if regressed:
            sys.exit("benchmark: voxel agreement regressed against the reference")


if __name__ == "__main__":
    main()
