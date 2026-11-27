"""Compatibility adapter for brainchop-core's embedded model executables.

Only the five embedded checkpoints are supported. Legacy names with different
weights are rejected rather than mapped to a different checkpoint.
"""

import gzip
import json
import os
from pathlib import Path
import shutil
import subprocess


NATIVE_MODELS = {
    "mindgrab": "Skull stripping in any modality (native)",
    "mindmap": "18-class cortical and subcortical segmentation (native, default)",
    "mindmap-omni": "Contrast-agnostic 18-class segmentation: T2w, FLAIR, T1w, ULF, PET (native)",
    "mindsnap": "104-class Desikan-Killiany cortical and subcortical parcellation (native)",
    "mindmap-fast": "Fast lightweight 18-class segmentation (native)",
}
MODEL_ALIASES = {
    "16chan18cls": "mindmap-fast",
    "model16chan18cls": "mindmap-fast",
    "mindmap_omni": "mindmap-omni",
}
CLASS_COUNTS = {"mindgrab": 2, "mindmap-fast": 18, "mindmap": 18, "mindmap-omni": 18, "mindsnap": 104}
_LEGACY_HINTS = {
    "subcortical": "Try 'mindmap' (18-class) or 'mindmap-fast' (fast lightweight).",
    "subcortical_mini": "Try 'mindmap-fast' (fast lightweight 18-class).",
    "tissue_fast": "Try 'mindmap-fast' (18-class) or 'mindmap' with --pve (GM/WM/CSF fractions).",
    "tissue": "Try 'mindmap' with --pve for GM/WM/CSF tissue fractions.",
    "dkatlas": "Try 'mindsnap' (104-class Desikan-Killiany parcellation).",
    "aparc": "Try 'mindsnap' (104-class Desikan-Killiany parcellation).",
}


def unknown_model_message(model_name):
    hint = _LEGACY_HINTS.get(str(model_name).lower(), "")
    extra = f" {hint}" if hint else ""
    return (
        f"{model_name!r} is unavailable in this native release; "
        f"select mindgrab, mindmap-fast, mindmap, mindmap-omni or mindsnap.{extra} "
        "Use brainchop==0.2.5 if you need the retired Python models."
    )


def find_executable(model_name, adapt=True):
    """Prefer this Python environment's scripts over another environment's PATH."""
    name = "brainchop-" + model_name
    package=Path(__file__).parent
    manifest=json.loads((package/'build.json').read_text()) if (package/'build.json').exists() else {}
    if adapt and manifest:
        from .adaptive import select
        package,manifest=select(package,manifest)
    candidate = package / 'bin' / (name + ('.exe' if os.name == 'nt' else ''))
    if not candidate.is_file():
        raise ValueError(f"{name} is missing from this installation; reinstall brainchop.")
    if manifest.get('cpu_target')=='avx2-fma':
        from .adaptive import parse_probe
        try:
            result=subprocess.run([str(package/'bin'/('hardware-probe'+('.exe' if os.name=='nt' else ''))),'--cpu-only'],capture_output=True,text=True,timeout=10)
            cpu=parse_probe(result.stdout)[0] if result.returncode==0 else None
        except (OSError,subprocess.TimeoutExpired): cpu=None
        if cpu!='avx2-fma':
            raise ValueError('this build requires AVX2 and FMA on x86-64; the current CPU cannot be verified as compatible. No inference was run.')
    return str(candidate)


def validate_options(args, model_name):
    """Validate the entire request before launching or writing any volume."""
    if model_name not in NATIVE_MODELS:
        raise ValueError(unknown_model_message(model_name))
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    # Model/backend/flag conflicts are rejected by the native CLI before any
    # output is written; check here only what the adapter itself translates.
    if args.save_conform and model_name == "mindgrab":
        raise ValueError("MindGrab always writes in input space; omit --save-conform.")
    if args.threads is not None and args.threads < 0:
        raise ValueError("--threads must be nonnegative (0 selects automatically).")
    if args.threads is not None and args.native_backend not in (None, "auto", "cpu"):
        raise ValueError("--threads forces CPU execution; omit --backend or select cpu.")
    for path in args.input:
        if not Path(path).is_file():
            raise ValueError(f"input does not exist: {path}")
    if args.export_classes and not args.output.endswith((".nii", ".nii.gz")):
        raise ValueError("--export-classes requires a .nii or .nii.gz output path.")


def command(args, model_name, executable, input_path, output_path):
    """Build native executable argv; input space is the default for all models."""
    cmd = [executable, "-o", output_path]
    if model_name != "mindgrab" and args.save_conform:
        cmd.append("--save-conform")
    for name in ("ct", "comply", "export_classes", "pve", "no_turbo"):
        if getattr(args, name):
            cmd.append("--" + name.replace("_", "-"))
    if args.crop is not None:
        cmd.extend(["--crop", str(args.crop)])
    if args.mask is not None:
        cmd.extend(["--mask", args.mask])
    if args.border:
        cmd.extend(["--border", str(args.border)])
    if args.threads is not None:
        cmd.extend(["-cpu", str(args.threads)])
    elif args.native_backend is not None:
        cmd.extend(["-backend", args.native_backend])
    # Absolute input paths also prevent leading '-' from becoming an option.
    cmd.append(os.path.abspath(input_path))
    return cmd


def export_legacy_filenames(output_path, model_name):
    """The Python CLI always exports uncompressed <base>_cN.nii raw maps."""
    if not output_path.endswith(".nii.gz"):
        return
    base = output_path[:-7]
    for index in range(CLASS_COUNTS[model_name]):
        source = Path(f"{base}_c{index}.nii.gz")
        destination = source.with_suffix("")
        with gzip.open(source, "rb") as reader, destination.open("wb") as writer:
            shutil.copyfileobj(reader, writer)
        source.unlink()


def run(args, model_name, output_path_for):
    validate_options(args, model_name)
    executable = find_executable(model_name)
    for index, input_path in enumerate(args.input):
        output_path = output_path_for(args, model_name, index, input_path)
        print(f"brainchop :: [{index + 1}/{len(args.input)}] {input_path}", flush=True)
        try:
            subprocess.run(command(args, model_name, executable, input_path, output_path),
                           check=True)
        except KeyboardInterrupt:
            raise SystemExit(130) from None
        except subprocess.CalledProcessError as exc:
            # Native stderr is inherited, and a failure never triggers a different model.
            raise SystemExit(exc.returncode if exc.returncode > 0 else 128 - exc.returncode)
        if args.export_classes:
            export_legacy_filenames(output_path, model_name)
        kind = "partial-volume maps for " if args.pve else ""
        print(f"brainchop :: Saved {kind}{output_path}", flush=True)
    print("brainchop :: Done!", flush=True)
