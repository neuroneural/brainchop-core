"""The historical brainchop command, backed by brainchop-core's embedded models."""

# CLI spelling/output naming follows brainchop-cli (MIT, copyright 2024 spikedoanz).

import argparse
from importlib.metadata import PackageNotFoundError, version as _dist_version
from pathlib import Path
import sys

from brainchop_cli import native


def _package_version() -> str:
    try:
        return _dist_version("brainchop")
    except PackageNotFoundError:
        ver_file = Path(__file__).resolve().parents[1] / "PYPI_VERSION"
        return ver_file.read_text().strip() if ver_file.is_file() else "1.1.3"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="brainchop",
        description="BrainChop: native brain segmentation and skull stripping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  brainchop input.nii.gz -o output.nii.gz
  brainchop input.nii.gz -m mindmap-fast -o output_fast.nii.gz
  brainchop input.nii.gz --skull-strip --mask brain_mask.nii.gz
  brainchop input.nii.gz -m mindmap --pve -o tissues.nii.gz

All models write output in the original input voxel grid by default
(matching the standalone native binaries). Pass --save-conform to keep
conformed 256^3 label/PVE outputs.
""",
    )
    parser.add_argument("input", nargs="*", help="Input NIfTI file(s)")
    parser.add_argument("-o", "--output", default="output.nii.gz",
                        help="Output path (default: derive a name from input and model)")
    parser.add_argument("-m", "--model", default="mindmap",
                        help="mindgrab, mindmap (default; 18-class), mindsnap (104-class), or mindmap-fast (fast lightweight)")
    parser.add_argument("-l", "--list", action="store_true", help="List embedded models")
    parser.add_argument("-v", "--version", action="version", version=f"%(prog)s {_package_version()}",
                        help="Show version and exit")
    parser.add_argument("-u", "--update", action="store_true",
                        help="Show how to update the embedded models")
    parser.add_argument("--custom", metavar="DIR",
                        help="Retired: native models are embedded in the executables")
    parser.add_argument("--skull-strip", action="store_true",
                        help="Skull strip mode (alias for --model mindgrab)")
    parser.add_argument("--ct", action="store_true",
                        help="CT scan: convert Hounsfield to Cormack units")
    parser.add_argument("--crop", nargs="?", type=float, const=2.0, metavar="PCT",
                        help="MindGrab: crop to brain bounding box (default percentile: 2.0)")
    parser.add_argument("--comply", action="store_true",
                        help="Insert compliance transform before conform")
    space = parser.add_mutually_exclusive_group()
    space.add_argument("--inverse-conform", action="store_true",
                       help="Reslice output to original input space (default; kept for CLI compatibility)")
    space.add_argument("--save-conform", action="store_true",
                       help="Write label/PVE output in conformed 256^3 space instead of the input grid")
    parser.add_argument("--export-classes", action="store_true",
                        help="Export per-class raw maps as <output>_cN.nii (CPU)")
    parser.add_argument("--mask", nargs="?", const="mask.nii.gz", metavar="PATH",
                        help="Save binary brain mask (mindgrab only; default: mask.nii.gz)")
    parser.add_argument("--border", type=int, default=0, metavar="MM",
                        help="Grow mask border in mm (mindgrab only; default: 0)")
    parser.add_argument("--batch-size", type=int, default=1, metavar="N",
                        help="Compatibility option; inputs are processed sequentially")
    parser.add_argument("--beam", type=int, default=None, metavar="N",
                        help="Compatibility option; native models need no BEAM compilation")
    parser.add_argument("--no-optimize", action="store_true",
                        help="Compatibility option; native models need no first-run compilation")
    parser.add_argument("--backend", "--native-backend", dest="native_backend",
                        choices=("auto", "cpu", "metal", "cuda", "hip", "opencl"),
                        help="Native backend (default: auto; availability depends on the installation build)")
    parser.add_argument("--threads", "-cpu", type=int, metavar="N",
                        help="Force CPU execution with N threads (0: automatic)")
    parser.add_argument("--no-turbo", action="store_true",
                        help="Exact CPU / fp16 GPU instead of the faster, approximate default")
    parser.add_argument("--pve", action="store_true",
                        help="mindmap: write experimental GM/WM/CSF tissue fractions instead of labels")
    return parser


def _output_path(args, model_name: str, index: int, input_path: str) -> str:
    """Preserve the historical single-input and multi-input naming rules."""
    if len(args.input) == 1 and args.output != "output.nii.gz":
        return args.output
    base = Path(input_path).stem.replace(".nii", "")
    return f"{base}_{model_name}_{index + 1}.nii.gz"


def main():
    parser = get_parser()
    args = parser.parse_args()
    if args.update:
        print("Models are embedded in brainchop; update with: pip install --upgrade brainchop")
        return
    if args.list:
        for name, description in native.NATIVE_MODELS.items():
            print(f"  {name}: {description}")
        return
    if not args.input:
        parser.print_help()
        return
    if args.custom and not args.skull_strip:
        parser.error("--custom is unavailable: this release supports only the four embedded models")
    model_name = "mindgrab" if args.skull_strip else args.model
    canonical = native.MODEL_ALIASES.get(model_name, model_name)
    if args.beam is not None or args.no_optimize or args.batch_size != 1:
        print("brainchop: --beam, --no-optimize and --batch-size are compatibility options; "
              "native models are precompiled and inputs run sequentially", file=sys.stderr)
    try:
        native.run(args, canonical,
                   lambda parsed, _canonical, i, path: _output_path(parsed, model_name, i, path))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
    except (ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
