"""
brainchop.api - Zero-dependency Python scripting interface backed by native executables.

Backward-compatible with the historical ``brainchop-cli`` scripting surface
(``Volume``, ``load``, ``segment``, ``save``, ``list_models``, ``optimize``,
``export``, ``export_classes``), using only the Python standard library.
When ``numpy`` is installed, ``Volume.data`` returns a ``numpy.ndarray``;
otherwise it returns a lightweight pure-Python buffer view.
"""

from __future__ import annotations

import array
import gzip
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from typing import Any

from brainchop_cli import native

_NIFTI_DTYPES: dict[int, tuple[str, str, int]] = {
    2: ("uint8", "B", 1),
    4: ("int16", "h", 2),
    8: ("int32", "i", 4),
    16: ("float32", "f", 4),
    64: ("float64", "d", 8),
    256: ("int8", "b", 1),
    512: ("uint16", "H", 2),
    768: ("uint32", "I", 4),
}

_GZIP_MAGIC = b"\x1f\x8b"


def _read_nifti_bytes(path: str | Path) -> bytes:
    raw = Path(path).read_bytes()
    if raw[:2] == _GZIP_MAGIC:
        return gzip.decompress(raw)
    return raw


def _read_nifti_header(path: str | Path) -> bytes:
    file_path = Path(path)
    with file_path.open("rb") as handle:
        magic = handle.read(2)
    if magic == _GZIP_MAGIC:
        with gzip.open(file_path, "rb") as handle:
            hdr = handle.read(352)
    else:
        with file_path.open("rb") as handle:
            hdr = handle.read(352)
    if len(hdr) < 348:
        raise ValueError(f"not a valid NIfTI-1 file (too short): {path}")
    sizeof_le = struct.unpack_from("<i", hdr, 0)[0]
    sizeof_be = struct.unpack_from(">i", hdr, 0)[0]
    if sizeof_le != 348 and sizeof_be != 348:
        raise ValueError(f"not a valid NIfTI-1 file (unexpected sizeof_hdr): {path}")
    return hdr if len(hdr) == 352 else hdr + b"\x00" * (352 - len(hdr))


def _header_endian(header: bytes) -> str:
    if len(header) < 348:
        raise ValueError("NIfTI-1 header must be at least 348 bytes")
    if struct.unpack_from("<i", header, 0)[0] == 348:
        return "<"
    if struct.unpack_from(">i", header, 0)[0] == 348:
        return ">"
    raise ValueError("invalid NIfTI-1 header (sizeof_hdr != 348)")


def _parse_header_meta(header: bytes) -> dict[str, Any]:
    end = _header_endian(header)
    dims = struct.unpack_from(end + "8h", header, 40)
    ndim = max(1, min(dims[0], 7))
    shape = tuple(int(dims[i]) for i in range(1, ndim + 1))
    # Strip trailing singleton dims above 3D so (X, Y, Z, 1) -> (X, Y, Z)
    while len(shape) > 3 and shape[-1] == 1:
        shape = shape[:-1]
    datatype = struct.unpack_from(end + "h", header, 70)[0]
    vox_offset = int(struct.unpack_from(end + "f", header, 108)[0])
    if vox_offset < 352:
        vox_offset = 352
    scl_slope, scl_inter = struct.unpack_from(end + "2f", header, 112)
    return {
        "endian": end,
        "shape": shape,
        "datatype": datatype,
        "vox_offset": vox_offset,
        "scl_slope": scl_slope,
        "scl_inter": scl_inter,
    }


def _set_intent_label(header: bytes) -> bytes:
    end = _header_endian(header)
    ba = bytearray(header[:352])
    if len(ba) < 352:
        ba.extend(b"\x00" * (352 - len(ba)))
    struct.pack_into(end + "h", ba, 68, 1002)  # NIFTI_INTENT_LABEL
    return bytes(ba)


def _decode_nifti_payload(raw_nii: bytes, header: bytes) -> Any:
    meta = _parse_header_meta(header)
    shape = meta["shape"]
    datatype = meta["datatype"]
    if datatype not in _NIFTI_DTYPES:
        raise ValueError(f"unsupported NIfTI datatype code: {datatype}")
    dtype_name, fmt_char, itemsize = _NIFTI_DTYPES[datatype]
    nvox = 1
    for d in shape:
        nvox *= d
    start = meta["vox_offset"]
    payload = raw_nii[start : start + nvox * itemsize]
    if len(payload) < nvox * itemsize:
        raise ValueError("NIfTI payload shorter than header dimensions")
    try:
        import numpy as np  # type: ignore[import-not-found]

        np_dtype = np.dtype(meta["endian"] + np.dtype(dtype_name).str[1:])
        return np.frombuffer(payload, dtype=np_dtype, count=nvox).reshape(shape, order="F").copy()
    except ImportError:
        return _StdlibVolumeData(payload, shape, dtype_name, fmt_char, meta["endian"])


class _StdlibVolumeData:
    """Minimal pure-stdlib array view used when ``numpy`` is not installed."""

    def __init__(self, payload: bytes, shape: tuple[int, ...], dtype: str, fmt_char: str, endian: str):
        self._payload = payload
        self.shape = shape
        self.dtype = dtype
        self._fmt_char = fmt_char
        self._endian = endian
        size = 1
        for d in shape:
            size *= d
        self.size = size

    def tobytes(self) -> bytes:
        return self._payload

    def tolist(self) -> list[Any]:
        arr = array.array(self._fmt_char)
        arr.frombytes(self._payload)
        sys_endian = "<" if sys.byteorder == "little" else ">"
        if self._endian != sys_endian and arr.itemsize > 1:
            arr.byteswap()
        return arr.tolist()

    def __len__(self) -> int:
        return self.shape[0] if self.shape else 0

    def __repr__(self) -> str:
        return f"_StdlibVolumeData(shape={self.shape}, dtype={self.dtype!r})"


class Volume:
    """A brain volume paired with its 352-byte NIfTI-1 header.

    Compatible with ``brainchop-cli``'s ``Volume(data, header, crop_coords=None)``.
    When constructed via ``load(path)``, voxel decoding is deferred until
    ``.data`` is accessed so ``load() -> segment() -> save()`` avoids unnecessary
    in-memory decoding overhead.
    """

    def __init__(
        self,
        data: Any = None,
        header: bytes = b"",
        crop_coords: tuple[int, ...] | None = None,
        *,
        _source_path: Path | None = None,
        _raw_nii: bytes | None = None,
        _load_opts: dict[str, Any] | None = None,
        _is_label: bool = False,
    ) -> None:
        self._data: Any = data
        self.header: bytes = header
        self.crop_coords: tuple[int, ...] | None = crop_coords
        self._source_path: Path | None = _source_path
        self._raw_nii: bytes | None = _raw_nii
        self._load_opts: dict[str, Any] = dict(_load_opts or {})
        self._is_label: bool = _is_label

    @property
    def data(self) -> Any:
        if self._data is None:
            if self._raw_nii is not None:
                self._data = _decode_nifti_payload(self._raw_nii, self.header)
            elif self._source_path is not None:
                raw = _read_nifti_bytes(self._source_path)
                self._raw_nii = raw
                if not self.header:
                    self.header = raw[:352]
                self._data = _decode_nifti_payload(raw, self.header)
        return self._data

    @data.setter
    def data(self, value: Any) -> None:
        self._data = value
        self._raw_nii = None

    @property
    def shape(self) -> tuple[int, ...]:
        if self._data is not None and hasattr(self._data, "shape"):
            return tuple(int(d) for d in self._data.shape)
        if self.header:
            return _parse_header_meta(self.header)["shape"]
        return ()

    @property
    def dtype(self) -> str:
        if self._data is not None and hasattr(self._data, "dtype"):
            return str(self._data.dtype)
        if self.header:
            code = _parse_header_meta(self.header)["datatype"]
            return _NIFTI_DTYPES.get(code, (f"code_{code}", "", 0))[0]
        return "unknown"

    def _serialize_nii(self, *, force_label_intent: bool = False) -> bytes:
        if self._raw_nii is not None and self._data is None:
            if force_label_intent:
                return _set_intent_label(self._raw_nii[:352]) + self._raw_nii[352:]
            return self._raw_nii
        if not self.header:
            raise ValueError("Volume has no NIfTI header")
        header = _set_intent_label(self.header) if force_label_intent else self.header[:352]
        if len(header) < 352:
            header = header + b"\x00" * (352 - len(header))
        data = self.data
        if data is None:
            raise ValueError("Volume has no data")
        if hasattr(data, "numpy") and callable(data.numpy):
            data = data.numpy()
        if isinstance(data, _StdlibVolumeData):
            payload = data.tobytes()
        else:
            try:
                import numpy as np  # type: ignore[import-not-found]

                arr = np.asarray(data)
                meta = _parse_header_meta(header)
                dtype_info = _NIFTI_DTYPES.get(meta["datatype"])
                if dtype_info is not None:
                    np_dtype = np.dtype(meta["endian"] + np.dtype(dtype_info[0]).str[1:])
                    arr = arr.astype(np_dtype, copy=False)
                payload = arr.tobytes(order="F")
            except ImportError:
                if hasattr(data, "tobytes"):
                    payload = data.tobytes()
                elif isinstance(data, (bytes, bytearray, memoryview)):
                    payload = bytes(data)
                else:
                    raise TypeError(f"cannot serialize volume data of type {type(data).__name__}")
        # Reset vox_offset to 352 so the written NIfTI payload aligns immediately.
        end = _header_endian(header)
        hdr_ba = bytearray(header)
        struct.pack_into(end + "f", hdr_ba, 108, 352.0)
        return bytes(hdr_ba) + payload

    def __repr__(self) -> str:
        return f"Volume(shape={self.shape}, dtype={self.dtype!r})"


class RawClassMaps:
    """Container for per-class raw maps returned by ``segment(..., return_raw=True)``."""

    def __init__(self, class_nii_bytes: list[bytes]) -> None:
        self._class_nii = class_nii_bytes
        self.num_classes = len(class_nii_bytes)

    def __len__(self) -> int:
        return self.num_classes

    def numpy(self) -> Any:
        import numpy as np  # type: ignore[import-not-found]

        maps = []
        for raw in self._class_nii:
            vol = _decode_nifti_payload(raw, raw[:352])
            # Match legacy (1, C, D, H, W) convention from brainchop-cli
            maps.append(np.transpose(vol, (2, 1, 0)))
        return np.stack(maps, axis=0)[None, ...]

    def __array__(self, dtype: Any = None) -> Any:
        arr = self.numpy()
        return arr if dtype is None else arr.astype(dtype)


def list_models() -> dict[str, str]:
    """Return available embedded models as ``{name: description}``."""
    return dict(native.NATIVE_MODELS)


def _load_single(
    path: str | Path,
    *,
    crop: float | None = None,
    ct: bool = False,
    comply: bool = False,
) -> Volume:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"NIfTI input does not exist: {file_path}")
    header = _read_nifti_header(file_path)
    return Volume(
        data=None,
        header=header,
        crop_coords=None,
        _source_path=file_path.resolve(),
        _load_opts={"crop": crop, "ct": ct, "comply": comply},
    )


def load(
    path: str | Path | list[str | Path],
    *,
    crop: float | None = None,
    ct: bool = False,
    comply: bool = False,
) -> Volume | list[Volume]:
    """Load NIfTI file(s) as :class:`Volume` object(s).

    Conform/rescaling is executed natively inside :func:`segment`, avoiding
    extra Python dependencies and keeping full native preprocessing parity.
    """
    if isinstance(path, list):
        return [_load_single(p, crop=crop, ct=ct, comply=comply) for p in path]
    return _load_single(path, crop=crop, ct=ct, comply=comply)


def save(volume: Volume, path: str | Path) -> None:
    """Save a :class:`Volume` to ``.nii`` or ``.nii.gz`` using the stdlib."""
    out_path = Path(path)
    raw = volume._serialize_nii(force_label_intent=volume._is_label)
    if str(out_path).endswith(".gz"):
        out_path.write_bytes(gzip.compress(raw, compresslevel=6))
    else:
        out_path.write_bytes(raw)


def _resolve_model(model: str) -> str:
    canonical = native.MODEL_ALIASES.get(model, model)
    if canonical not in native.NATIVE_MODELS:
        raise ValueError(native.unknown_model_message(model))
    return canonical


def optimize(
    model: str,
    *,
    batch_size: int = 1,
    beam: int = 2,
) -> None:
    """Compatibility no-op: native models are already compiled and need no BEAM tuning."""
    _ = (batch_size, beam)
    _resolve_model(model)
    print(
        "brainchop :: Native models are precompiled; optimize() is a compatibility no-op.",
        file=sys.stderr,
    )


def export(
    model: str,
    output_dir: str,
    *,
    target: str = "webgpu",
    beam: int = 0,
    tta: bool = False,
    flip_axis: str = "sagittal",
    force: bool = False,
) -> tuple[str, str]:
    """Retired tinygrad WebGPU export hook (kept for clear migration errors)."""
    _ = (model, output_dir, target, beam, tta, flip_axis, force)
    raise NotImplementedError(
        "brainchop.export() (tinygrad WebGPU export) is retired in brainchop 1.0. "
        "For browser inference, use the @brainchop/core JS WebGPU package; "
        "install brainchop==0.2.5 if you need the legacy tinygrad exporter."
    )


def export_classes(raw_output: Any, header: bytes, output_path: str | Path) -> list[str]:
    """Write per-class probability/logit maps as ``<base>_c{i}.nii``."""
    base = str(output_path)
    for ext in (".nii.gz", ".nii"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break

    saved_paths: list[str] = []
    if isinstance(raw_output, RawClassMaps):
        for i, nii_bytes in enumerate(raw_output._class_nii):
            out_fname = f"{base}_c{i}.nii"
            Path(out_fname).write_bytes(nii_bytes)
            saved_paths.append(out_fname)
        return saved_paths

    import numpy as np  # type: ignore[import-not-found]

    arr = raw_output.numpy() if hasattr(raw_output, "numpy") else np.asarray(raw_output)
    if arr.ndim == 5:
        arr = arr[0]
    if arr.ndim != 4:
        raise ValueError(f"expected 4D (C,D,H,W) or 5D (1,C,D,H,W) raw_output, got shape {arr.shape}")
    end = _header_endian(header)
    hdr_ba = bytearray(header[:352])
    if len(hdr_ba) < 352:
        hdr_ba.extend(b"\x00" * (352 - len(hdr_ba)))
    struct.pack_into(end + "h", hdr_ba, 70, 16)  # float32
    struct.pack_into(end + "h", hdr_ba, 72, 32)  # bitpix
    struct.pack_into(end + "f", hdr_ba, 108, 352.0)
    hdr_bytes = bytes(hdr_ba)
    for i in range(arr.shape[0]):
        chan = np.transpose(arr[i], (2, 1, 0)).astype(end + "f4", copy=False)
        out_fname = f"{base}_c{i}.nii"
        Path(out_fname).write_bytes(hdr_bytes + chan.tobytes(order="F"))
        saved_paths.append(out_fname)
    return saved_paths


def segment(
    volume: Volume | str | Path | list[Volume | str | Path],
    model: str = "mindmap",
    *,
    shard_size: int = 1,
    beam: int = 0,
    return_raw: bool = False,
    tta: bool = False,
    backend: str | None = None,
    threads: int | None = None,
    save_conform: bool = False,
    inverse_conform: bool | None = None,
    no_turbo: bool = False,
    pve: bool = False,
    mask: bool = False,
    border: int = 0,
    ct: bool | None = None,
    comply: bool | None = None,
    crop: float | None = None,
) -> Any:
    """Segment one or more brain volumes with a native embedded model.

    Accepts a :class:`Volume` (from :func:`load`), a file path (``str``/``Path``),
    or a list of them. Output defaults to the original input voxel grid for all
    models (matching standalone binaries and ``@brainchop/core``); pass
    ``save_conform=True`` to keep conformed 256^3 outputs for label/PVE models.
    ``inverse_conform`` is accepted for backward compatibility.
    """
    canonical = _resolve_model(model)
    if tta:
        raise NotImplementedError(
            "Test-time augmentation (tta=True) is not available in the native runtime."
        )
    if save_conform and inverse_conform is True:
        raise ValueError("save_conform and inverse_conform cannot both be True.")
    if save_conform and canonical == "mindgrab":
        raise ValueError("mindgrab always writes in input space; omit save_conform=True.")
    if pve and canonical != "mindmap":
        raise ValueError("pve=True is supported only for model='mindmap'.")
    if pve and return_raw:
        raise ValueError("pve=True cannot be combined with return_raw=True.")
    if (mask or border) and canonical != "mindgrab":
        raise ValueError("mask and border options are supported only for model='mindgrab'.")
    if threads is not None and threads < 0:
        raise ValueError("threads must be nonnegative (0 selects automatically).")
    if threads is not None and backend not in (None, "auto", "cpu"):
        raise ValueError("threads forces CPU execution; omit backend or set backend='cpu'.")
    if shard_size < 1:
        raise ValueError("shard_size must be positive.")
    if beam != 0 or shard_size != 1:
        print(
            "brainchop :: beam and shard_size are compatibility options; "
            "native models are precompiled and run sequentially.",
            file=sys.stderr,
        )

    single_input = not isinstance(volume, list)
    items: list[Volume | str | Path] = [volume] if single_input else list(volume)
    executable = native.find_executable(canonical)

    results: list[Any] = []
    raw_outputs: list[RawClassMaps] = []

    with tempfile.TemporaryDirectory(prefix="brainchop-api-") as tmp:
        tmp_dir = Path(tmp)
        for idx, item in enumerate(items):
            if isinstance(item, (str, Path)):
                vol = _load_single(item)
            elif isinstance(item, Volume):
                vol = item
            else:
                raise TypeError(f"expected Volume or file path, got {type(item).__name__}")

            if vol._source_path is not None and vol._data is None:
                in_path = vol._source_path
            else:
                in_path = tmp_dir / f"in_{idx}.nii"
                in_path.write_bytes(vol._serialize_nii())

            out_path = tmp_dir / f"out_{idx}.nii"
            cmd = [executable, "-o", str(out_path)]
            if canonical != "mindgrab" and save_conform:
                cmd.append("--save-conform")

            use_ct = vol._load_opts.get("ct", False) if ct is None else ct
            use_comply = vol._load_opts.get("comply", False) if comply is None else comply
            use_crop = vol._load_opts.get("crop") if crop is None else crop

            if use_ct:
                cmd.append("--ct")
            if use_comply:
                cmd.append("--comply")
            if use_crop is not None:
                cmd.extend(["--crop", str(use_crop)])
            if no_turbo:
                cmd.append("--no-turbo")
            if return_raw:
                cmd.append("--export-classes")
            if pve:
                cmd.append("--pve")
            mask_path = tmp_dir / f"mask_{idx}.nii"
            if mask:
                cmd.extend(["--mask", str(mask_path)])
            if border:
                cmd.extend(["--border", str(border)])
            if threads is not None:
                cmd.extend(["-cpu", str(threads)])
            elif backend is not None:
                cmd.extend(["-backend", backend])
            cmd.append(os.path.abspath(str(in_path)))

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                detail = (proc.stderr or proc.stdout or "").strip()
                raise RuntimeError(
                    f"brainchop-{canonical} failed (exit {proc.returncode})"
                    + (f": {detail}" if detail else "")
                )

            if pve:
                tissue_vols: dict[str, Volume] = {}
                for tissue in ("gm", "wm", "csf", "brain"):
                    tpath = tmp_dir / f"out_{idx}_{tissue}.nii"
                    tbytes = tpath.read_bytes()
                    tissue_vols[tissue] = Volume(
                        data=None, header=tbytes[:352], _raw_nii=tbytes, _is_label=False
                    )
                results.append(tissue_vols)
            else:
                out_bytes = out_path.read_bytes()
                out_vol = Volume(
                    data=None,
                    header=out_bytes[:352],
                    _raw_nii=out_bytes,
                    _is_label=(canonical != "mindgrab"),
                )
                if mask:
                    mask_bytes = mask_path.read_bytes()
                    mask_vol = Volume(
                        data=None, header=mask_bytes[:352], _raw_nii=mask_bytes, _is_label=True
                    )
                    results.append((out_vol, mask_vol))
                else:
                    results.append(out_vol)

            if return_raw:
                n_cls = native.CLASS_COUNTS[canonical]
                raw_list = [
                    (tmp_dir / f"out_{idx}_c{c}.nii").read_bytes() for c in range(n_cls)
                ]
                raw_outputs.append(RawClassMaps(raw_list))

    final_result = results[0] if single_input else results
    if return_raw:
        final_raw = raw_outputs[0] if single_input else raw_outputs
        return final_result, final_raw
    return final_result
