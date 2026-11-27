"""
brainchop - Native brain segmentation and skull stripping.

Example:
    from brainchop import load, segment, save, list_models

    vol = load("input.nii.gz")
    result = segment(vol, "mindmap")
    save(result, "output.nii.gz")
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _dist_version

from brainchop.api import (
    RawClassMaps,
    Volume,
    export,
    export_classes,
    list_models,
    load,
    optimize,
    save,
    segment,
)

try:
    __version__ = _dist_version("brainchop")
except PackageNotFoundError:
    from pathlib import Path as _Path

    _ver_file = _Path(__file__).resolve().parents[1] / "PYPI_VERSION"
    __version__ = _ver_file.read_text().strip() if _ver_file.is_file() else "1.1.3"

__all__ = [
    "__version__",
    "RawClassMaps",
    "Volume",
    "load",
    "save",
    "segment",
    "list_models",
    "export",
    "optimize",
    "export_classes",
]
