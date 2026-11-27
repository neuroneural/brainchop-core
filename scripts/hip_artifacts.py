#!/usr/bin/env python3
"""Validate BrainChop HIP kernel bundles without ROCm, Python ML packages, or a GPU.

Generation lives in a separate maintainer tool.  This module owns the stable
on-disk contract shared by that tool and the small production HIP loader.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

SCHEMA_VERSION = 2
ABI_VERSION = 2
SUPPORTED_VERSIONS = {(1, 1), (SCHEMA_VERSION, ABI_VERSION)}
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
ID_RE = re.compile(r"[a-z][a-z0-9_.-]*\Z")
ARCH_RE = re.compile(r"gfx[0-9a-f]+\Z")
DTYPES = {"f16", "f32", "u8", "u32", "i32"}
ACCESS = {"read", "write", "readwrite"}
PRODUCERS = {"rocmlir", "tinygrad", "hipcc"}
DISPATCH_MODES = {"always", "labels", "pve"}


def _legacy_evolved_source(name):
    """Schema 1 predates capability/versioned-dispatch metadata.

    Its checked-in code objects remain valid when only the developer packer,
    verifier, generated capability header or model metadata evolves. Topology
    and weights are still checked strictly, so inference-affecting drift cannot
    pass under this compatibility rule.
    """
    return name in {
        "scripts/build_hip_artifacts.py", "scripts/export_hip_onnx.py",
        "scripts/hip_artifacts.py",
    }


# Checked-in gfx1151 bundles compiled with ROCm 7.2.2 against v1.0.0 prior
# masks remain valid for label inference; backend_hip_driver.c gates --pve
# on BC_HIP_PVE_REVISION until gfx1151 PVE support is rebuilt on ROCm.
_PRE_V111_PINNED_SOURCES = {
    "scripts/hip_artifacts.py":
        "325fc3973fbb4ea90fe5aa1ddcab31cdeee69e1f445d853b76c1af7bfbfff951",
    "src/pve_constants.h":
        "2bcbd8fceab885aba4b04cf0f61a4c878d6c5cab45422157af94e4e090d5ac4c",
    "scripts/hip_tissue_priors.hip":
        "7fad23fb52ac39de8b4cbf7c76f221d7bcbf2e0f175043559800f36119b59661",
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def _need_dict(value, where):
    if not isinstance(value, dict):
        raise ValueError(f"{where} must be an object")
    return value


def _need_list(value, where):
    if not isinstance(value, list):
        raise ValueError(f"{where} must be an array")
    return value


def _need_int(value, where, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{where} must be an integer >= {minimum}")
    return value


def _need_string(value, where, pattern=None):
    if not isinstance(value, str) or not value or (pattern and not pattern.fullmatch(value)):
        raise ValueError(f"invalid {where}")
    return value


def _keys(value, required, optional, where):
    value = _need_dict(value, where)
    missing = set(required) - set(value)
    extra = set(value) - set(required) - set(optional)
    if missing or extra:
        raise ValueError(f"{where} keys: missing={sorted(missing)}, extra={sorted(extra)}")
    return value


def _unpack(data, offset=0):
    """Small, dependency-free MessagePack reader for AMDGPU metadata notes."""
    if offset >= len(data):
        raise ValueError("truncated AMDGPU metadata")
    tag = data[offset]
    offset += 1
    if tag <= 0x7f:
        return tag, offset
    if tag >= 0xe0:
        return tag - 256, offset
    if 0xa0 <= tag <= 0xbf:
        size = tag & 0x1f
        end = offset + size
        return data[offset:end].decode("utf-8"), end
    if 0x90 <= tag <= 0x9f:
        count = tag & 0x0f
        out = []
        for _ in range(count):
            value, offset = _unpack(data, offset)
            out.append(value)
        return out, offset
    if 0x80 <= tag <= 0x8f:
        count = tag & 0x0f
        out = {}
        for _ in range(count):
            key, offset = _unpack(data, offset)
            value, offset = _unpack(data, offset)
            out[key] = value
        return out, offset
    if tag in (0xc2, 0xc3):
        return tag == 0xc3, offset
    if tag == 0xc0:
        return None, offset
    widths = {0xcc: (1, ">B"), 0xcd: (2, ">H"), 0xce: (4, ">I"),
              0xcf: (8, ">Q"), 0xd0: (1, ">b"), 0xd1: (2, ">h"),
              0xd2: (4, ">i"), 0xd3: (8, ">q")}
    if tag in widths:
        width, fmt = widths[tag]
        end = offset + width
        if end > len(data):
            raise ValueError("truncated AMDGPU metadata integer")
        return struct.unpack(fmt, data[offset:end])[0], end
    string_widths = {0xd9: 1, 0xda: 2, 0xdb: 4}
    if tag in string_widths:
        width = string_widths[tag]
        if offset + width > len(data):
            raise ValueError("truncated AMDGPU metadata string")
        size = int.from_bytes(data[offset:offset + width], "big")
        offset += width
        end = offset + size
        return data[offset:end].decode("utf-8"), end
    collection_widths = {0xdc: (2, False), 0xdd: (4, False),
                         0xde: (2, True), 0xdf: (4, True)}
    if tag in collection_widths:
        width, is_map = collection_widths[tag]
        if offset + width > len(data):
            raise ValueError("truncated AMDGPU metadata collection")
        count = int.from_bytes(data[offset:offset + width], "big")
        offset += width
        out = {} if is_map else []
        for _ in range(count):
            if is_map:
                key, offset = _unpack(data, offset)
                value, offset = _unpack(data, offset)
                out[key] = value
            else:
                value, offset = _unpack(data, offset)
                out.append(value)
        return out, offset
    raise ValueError(f"unsupported AMDGPU metadata MessagePack tag 0x{tag:02x}")


def code_object_metadata(blob):
    """Return the target and single-kernel ABI from an AMDGPU HSA code object."""
    if len(blob) < 64 or blob[:4] != b"\x7fELF" or blob[4:6] != b"\x02\x01":
        raise ValueError("kernel slice is not a 64-bit little-endian ELF object")
    if int.from_bytes(blob[16:18], "little") != 3:
        raise ValueError("kernel slice is not a loadable ELF shared object")
    marker = blob.find(b"AMDGPU\x00")
    if marker < 0:
        raise ValueError("kernel slice has no AMDGPU metadata note")
    start = marker + len(b"AMDGPU\x00")
    metadata = None
    for candidate in range(start, min(start + 8, len(blob))):
        if 0x80 <= blob[candidate] <= 0x8f or blob[candidate] in (0xde, 0xdf):
            try:
                value, _ = _unpack(blob, candidate)
            except (ValueError, UnicodeDecodeError, struct.error):
                continue
            if isinstance(value, dict) and "amdhsa.kernels" in value:
                metadata = value
                break
    if metadata is None:
        raise ValueError("cannot decode AMDGPU metadata note")
    target = metadata.get("amdhsa.target")
    kernels = metadata.get("amdhsa.kernels")
    if not isinstance(target, str) or not isinstance(kernels, list) or len(kernels) != 1:
        raise ValueError("each kernel slice must describe exactly one AMDGPU kernel")
    kernel = _need_dict(kernels[0], "AMDGPU kernel metadata")
    if kernel.get(".name") == "bc_hip_classify" and (
            kernel.get(".private_segment_fixed_size", 0) or
            kernel.get(".vgpr_spill_count", 0) or
            kernel.get(".sgpr_spill_count", 0)):
        raise ValueError("bounded classifier kernel must not spill to private memory")
    args = kernel.get(".args", [])
    if not isinstance(args, list) or any(not isinstance(a, dict) for a in args):
        raise ValueError("invalid AMDGPU kernel argument metadata")
    return {
        "target": target.rsplit("--", 1)[-1].split(":", 1)[0],
        "entry": kernel.get(".name"),
        "symbol": kernel.get(".symbol"),
        "kernarg_segment_size": kernel.get(".kernarg_segment_size"),
        "static_lds_bytes": kernel.get(".group_segment_fixed_size", 0),
        "arguments": [{"offset": a.get(".offset"), "size": a.get(".size"),
                       "value_kind": a.get(".value_kind")} for a in args],
    }


def _validate_model(model, schema):
    required = ("name", "descriptor_sha256", "dimension",
                "channels", "hidden_layers", "classes", "has_affine",
                "classifier_bias")
    if schema >= 2:
        required += ("supports_pve",)
    model = _keys(model, required, ("weights_sha256",), "model")
    # Legacy: weights upload at run time, so artifacts no longer bind them.
    # Dropped in place so verify() callers compare manifests without it.
    model.pop("weights_sha256", None)
    _need_string(model["name"], "model.name", ID_RE)
    _need_string(model["descriptor_sha256"], "model.descriptor_sha256", SHA256_RE)
    for key in ("dimension", "channels", "hidden_layers", "classes"):
        _need_int(model[key], f"model.{key}", 1)
    for key in ("has_affine", "classifier_bias"):
        if not isinstance(model[key], bool):
            raise ValueError(f"model.{key} must be boolean")
    if schema >= 2:
        if not isinstance(model["supports_pve"], bool):
            raise ValueError("model.supports_pve must be boolean")
        if model["supports_pve"] and model["classes"] != 18:
            raise ValueError("PVE capability requires an 18-class model")


def _verify_bounded_head(manifest):
    contract = manifest["toolchain"]["compile_options"].get("bounded_head")
    if contract is None:
        # Historical full-head artifacts remain valid upgrade inputs at any
        # class count. Packaging the fused kernel is what requires its contract.
        if any(k["entry"] == "bc_hip_classify" for k in manifest["kernels"]):
            raise ValueError("bounded classifier contract is required")
        return  # Schema/ABI 1 and previously published ABI 2 remain readable.
    if manifest["schema_version"] < 2 or manifest["abi_version"] < 2:
        raise ValueError("bounded classifier requires schema/ABI 2")
    model = manifest["model"]
    channels, classes, voxels = model["channels"], model["classes"], model["dimension"] ** 3
    expected = {"layout": "NCDHW", "weight_layout": "class-channel",
                "accumulation": "f32", "workspace_bytes": 0,
                "channels": channels, "classes": classes,
                "classifier_bias": model["classifier_bias"]}
    if contract != expected or model.get("supports_pve"):
        raise ValueError("invalid bounded classifier contract")
    if [(b["id"], b["kind"]) for b in manifest["buffers"]] != [
            ("arena", "arena"), ("input", "input"), ("output", "output")]:
        raise ValueError("bounded classifier has undeclared allocations")
    if sum(b["bytes"] for b in manifest["buffers"]) > 4 * 1024 ** 3:
        raise ValueError("bounded labels exceed 4 GiB persistent device memory")
    kernels = {k["id"]: k for k in manifest["kernels"]}
    dispatches = manifest["dispatches"]
    if set(kernels) != {d["kernel"] for d in dispatches}:
        raise ValueError("bounded artifact contains obsolete kernels")
    labels = [d for d in dispatches if d["mode"] == "labels"]
    if len(labels) != 1 or labels[0] is not dispatches[-1] or \
            kernels[labels[0]["kernel"]]["entry"] != "bc_hip_classify":
        raise ValueError("bounded artifact must end with one fused label dispatch")
    args = labels[0]["args"]
    sizes = [channels * voxels * 2, channels * classes * 2] + (
        [classes * 2] if model["classifier_bias"] else []) + [voxels]
    if [a.get("bytes") for a in args] != sizes or any(
            a.get("buffer") != "arena" or a.get("access") != "read" for a in args[:-1]) or \
            args[-1] != {"kind": "buffer", "buffer": "output", "offset": 0,
                         "bytes": voxels, "access": "write"}:
        raise ValueError("invalid bounded classifier argument ranges")
    if manifest["buffers"][1]["bytes"] != voxels * 2:
        raise ValueError("bounded classifier input size mismatch")
    uploads = [w for w in manifest["weights"] if w["layout"] == "tfjs-in-out_to_hip-out-in"]
    if len(uploads) != 1:
        raise ValueError("bounded classifier requires one class-channel weight upload")
    weight = uploads[0]
    for a, w in [(args[1], weight)] + ([
            (args[2], next((w for w in manifest["weights"]
                           if w["tensor"] == weight["tensor"][:-len("kernel")] + "bias"), {}))]
            if model["classifier_bias"] else []):
        if w.get("target_dtype") != "f16" or any(a.get(k) != w.get(k)
                for k in ("buffer", "offset", "bytes")):
            raise ValueError("bounded classifier argument does not match its weight upload")
    if len(dispatches) < 2 or not any(
            a.get("access") == "write" and all(a.get(k) == args[0].get(k)
                for k in ("buffer", "offset", "bytes")) for a in dispatches[-2]["args"]):
        raise ValueError("bounded classifier must consume the final hidden producer")
    # Reject the obsolete producer by semantics, not byte-size coincidence:
    # when classes == channels, the tuned convolution's required fp32 sink has
    # the same extent as a class volume, but is not classifier workspace.
    if any(k["entry"] in ("mlir_convolution_convert", "mlir_convolution_broadcast_add_convert")
           for k in manifest["kernels"]):
        raise ValueError("bounded artifact retains the full-logit classifier")
    intervals = sorted(
        [(a["offset"], a["offset"] + a["bytes"]) for d in dispatches for a in d["args"]
         if a.get("kind") == "buffer" and a.get("buffer") == "arena"] +
        [(w["offset"], w["offset"] + w["bytes"]) for w in manifest["weights"]])
    alignment = manifest["buffers"][0]["alignment"]
    cursor = 0
    for lo, hi in intervals:
        if lo > cursor:
            if lo != -(-cursor // alignment) * alignment:
                raise ValueError("bounded arena includes uncompacted interior storage")
            cursor = lo
        cursor = max(cursor, hi)
    if manifest["buffers"][0]["bytes"] != cursor:
        raise ValueError("bounded arena includes unused tail storage")


def verify(directory, source_root=None):
    directory = Path(directory)
    source_root = Path(source_root) if source_root is not None else Path(__file__).resolve().parents[1]
    manifest = json.loads((directory / "manifest.json").read_text())
    manifest = _keys(manifest,
        ("schema_version", "abi_version", "model", "target", "toolchain", "sources",
         "bundle", "buffers", "kernels", "dispatches", "weights"), (), "manifest")
    version = (manifest["schema_version"], manifest["abi_version"])
    if version not in SUPPORTED_VERSIONS:
        raise ValueError("unsupported HIP artifact schema or ABI")
    schema = version[0]
    _validate_model(manifest["model"], schema)

    target = _keys(manifest["target"],
        ("architecture", "num_compute_units", "num_chiplets", "wavefront_size"), (), "target")
    architecture = _need_string(target["architecture"], "target.architecture", ARCH_RE)
    for key in ("num_compute_units", "num_chiplets", "wavefront_size"):
        _need_int(target[key], f"target.{key}", 1)

    toolchain = _keys(manifest["toolchain"],
        ("rocm", "migraphx", "tinygrad", "compile_options"), (), "toolchain")
    for key in ("rocm", "migraphx", "tinygrad"):
        _need_string(toolchain[key], f"toolchain.{key}")
    _need_dict(toolchain["compile_options"], "toolchain.compile_options")
    sources = _need_dict(manifest["sources"], "sources")
    if not sources:
        raise ValueError("sources must not be empty")
    for name, value in sources.items():
        _need_string(name, "source path")
        _need_string(value, f"sources[{name}]", SHA256_RE)
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"invalid source path: {name}")
        try:
            data = (source_root / path).read_bytes()
        except OSError as error:
            raise ValueError(f"missing artifact source: {name}") from error
        if digest(data) != value and not (
            (schema == 1 and _legacy_evolved_source(name)) or
            _PRE_V111_PINNED_SOURCES.get(name) == value
        ):
            raise ValueError(f"stale artifact source: {name}")

    bundle = _keys(manifest["bundle"], ("file", "size", "sha256"), (), "bundle")
    filename = _need_string(bundle["file"], "bundle.file")
    if Path(filename).name != filename:
        raise ValueError("bundle.file must be a basename")
    blob = (directory / filename).read_bytes()
    if bundle["size"] != len(blob) or bundle["sha256"] != digest(blob):
        raise ValueError("corrupt HIP kernel bundle")

    buffers = {}
    for index, item in enumerate(_need_list(manifest["buffers"], "buffers")):
        item = _keys(item, ("id", "bytes", "alignment", "kind"), (), f"buffers[{index}]")
        name = _need_string(item["id"], f"buffers[{index}].id", ID_RE)
        size = _need_int(item["bytes"], f"buffers[{index}].bytes", 1)
        alignment = _need_int(item["alignment"], f"buffers[{index}].alignment", 1)
        if alignment & (alignment - 1) or item["kind"] not in ("arena", "input", "output"):
            raise ValueError(f"invalid buffers[{index}] alignment or kind")
        if name in buffers:
            raise ValueError("duplicate buffer id")
        buffers[name] = size
    if [b["kind"] for b in manifest["buffers"]].count("input") != 1 or \
       [b["kind"] for b in manifest["buffers"]].count("output") != 1:
        raise ValueError("exactly one input and one output buffer are required")

    kernels = {}
    occupied = []
    kernel_items = _need_list(manifest["kernels"], "kernels")
    if not kernel_items:
        raise ValueError("kernels must not be empty")
    for index, item in enumerate(kernel_items):
        item = _keys(item,
            ("id", "offset", "size", "sha256", "entry", "symbol", "producer",
             "target", "kernarg_segment_size", "static_lds_bytes", "arguments"), (),
            f"kernels[{index}]")
        name = _need_string(item["id"], f"kernels[{index}].id", ID_RE)
        offset = _need_int(item["offset"], f"kernels[{index}].offset")
        size = _need_int(item["size"], f"kernels[{index}].size", 1)
        if offset + size > len(blob) or any(offset < end and start < offset + size for start, end in occupied):
            raise ValueError(f"kernels[{index}] is outside or overlaps the bundle")
        occupied.append((offset, offset + size))
        part = blob[offset:offset + size]
        if digest(part) != item["sha256"]:
            raise ValueError(f"corrupt kernel slice: {name}")
        if item["producer"] not in PRODUCERS or item["target"] != architecture:
            raise ValueError(f"invalid producer or target for kernel {name}")
        metadata = code_object_metadata(part)
        expected = {key: item[key] for key in
                    ("target", "entry", "symbol", "kernarg_segment_size",
                     "static_lds_bytes", "arguments")}
        if metadata != expected:
            raise ValueError(f"AMDGPU metadata mismatch for kernel {name}")
        if name in kernels:
            raise ValueError("duplicate kernel id")
        kernels[name] = item

    dispatch_items = _need_list(manifest["dispatches"], "dispatches")
    if not dispatch_items:
        raise ValueError("dispatches must not be empty")
    for index, item in enumerate(dispatch_items):
        required = ("kernel", "grid", "block", "dynamic_lds_bytes", "args")
        if schema >= 2:
            required += ("mode",)
        item = _keys(item, required, (),
                     f"dispatches[{index}]")
        if item["kernel"] not in kernels:
            raise ValueError(f"dispatches[{index}] names an unknown kernel")
        for key in ("grid", "block"):
            values = _need_list(item[key], f"dispatches[{index}].{key}")
            if len(values) != 3:
                raise ValueError(f"dispatches[{index}].{key} must have three dimensions")
            for dimension in values:
                _need_int(dimension, f"dispatches[{index}].{key} dimension", 1)
        _need_int(item["dynamic_lds_bytes"], f"dispatches[{index}].dynamic_lds_bytes")
        if schema >= 2 and item["mode"] not in DISPATCH_MODES:
            raise ValueError(f"dispatches[{index}] has an invalid mode")
        args = _need_list(item["args"], f"dispatches[{index}].args")
        metadata_args = kernels[item["kernel"]]["arguments"]
        if schema >= 2:
            # HIP language objects describe implicit launch geometry after the
            # explicit kernargs. hipModuleLaunchKernel supplies those hidden
            # values; callers must pass only the non-hidden parameters.
            metadata_args = [arg for arg in metadata_args
                             if not str(arg["value_kind"]).startswith("hidden_")]
        if len(args) != len(metadata_args):
            raise ValueError(f"dispatches[{index}] argument count does not match the kernel ABI")
        for arg_index, arg in enumerate(args):
            arg = _need_dict(arg, f"dispatches[{index}].args[{arg_index}]")
            kind = arg.get("kind")
            value_kind = metadata_args[arg_index]["value_kind"]
            if kind == "buffer":
                _keys(arg, ("kind", "buffer", "offset", "bytes", "access"), (),
                      f"dispatches[{index}].args[{arg_index}]")
                buffer = arg["buffer"]
                offset = _need_int(arg["offset"], "buffer argument offset")
                size = _need_int(arg["bytes"], "buffer argument bytes", 1)
                if value_kind != "global_buffer" or buffer not in buffers or \
                   offset + size > buffers[buffer] or arg["access"] not in ACCESS:
                    raise ValueError(f"dispatches[{index}] has an invalid buffer argument")
            elif kind == "scalar":
                _keys(arg, ("kind", "dtype", "value"), (), f"dispatches[{index}].args[{arg_index}]")
                if value_kind == "global_buffer" or arg["dtype"] not in ("u32", "i32", "u64") or \
                   not isinstance(arg["value"], int):
                    raise ValueError(f"dispatches[{index}] has an invalid scalar argument")
            else:
                raise ValueError(f"dispatches[{index}] has an unknown argument kind")

    if schema >= 2:
        modes = [item["mode"] for item in dispatch_items]
        supports_pve = manifest["model"]["supports_pve"]
        if "labels" not in modes or (supports_pve != ("pve" in modes)):
            raise ValueError("dispatch modes do not match the model PVE capability")
        output = next(item for item in manifest["buffers"]
                      if item["kind"] == "output")
        voxels = manifest["model"]["dimension"] ** 3
        expected = 4 * voxels * 4 if supports_pve else voxels
        if output["bytes"] != expected:
            raise ValueError("output buffer does not match the model PVE capability")

    tensors = set()
    for index, item in enumerate(_need_list(manifest["weights"], "weights")):
        item = _keys(item, ("tensor", "buffer", "offset", "bytes", "source_dtype", "target_dtype", "layout"), (),
                     f"weights[{index}]")
        tensor = _need_string(item["tensor"], f"weights[{index}].tensor", ID_RE)
        if tensor in tensors:
            raise ValueError("duplicate weight tensor")
        tensors.add(tensor)
        if item["buffer"] not in buffers or item["source_dtype"] not in DTYPES or item["target_dtype"] not in DTYPES:
            raise ValueError(f"invalid weights[{index}]")
        offset = _need_int(item["offset"], f"weights[{index}].offset")
        size = _need_int(item["bytes"], f"weights[{index}].bytes", 1)
        if offset + size > buffers[item["buffer"]] or not isinstance(item["layout"], str):
            raise ValueError(f"weights[{index}] is outside its buffer")
    _verify_bounded_head(manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory containing manifest.json and its bundle")
    args = parser.parse_args()
    try:
        manifest = verify(args.directory)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        parser.exit(1, str(error) + "\n")
    print(f"HIP artifact: {manifest['model']['name']} {manifest['target']['architecture']} verified")


if __name__ == "__main__":
    main()
