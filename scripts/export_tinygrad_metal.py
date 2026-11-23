#!/usr/bin/env python3
"""Export tinygrad's fixed-shape Metal program as a native-runner manifest.

This is a development-time generator.  The resulting manifest contains MSL,
launch dimensions, buffer sizes, and state-dictionary names; brainchopc does
not need Python or tinygrad at runtime.
"""

import argparse
import json
import os
import re
import tempfile
from pathlib import Path


NVOX = 256 ** 3
ACTIVATION_OPS = re.compile(r"\b(?:exp2?|tanh)\(")


def classifier_boundary(data: dict) -> tuple:
    """First classifier dispatch and the arena offset of the activation it reads.

    A device head (--pve) replaces every dispatch from the classifier on and
    reads that activation, so a wrong answer here is a brain-shaped wrong
    answer at exit 0. Refuse rather than emit anything unproven: the input must
    be one 24xNVOX-style fp16 arena tensor, written by a kernel that applies
    the GELU, and the classifier itself must apply none (a fused GELU would
    hand the head un-activated values).
    """
    cw = data["weights"]["seq_conv_argmax.weight"]
    weight_slots = {(w["buffer"], w["offset"]) for w in data["weights"].values()}
    statements = data["statements"]
    at = next(i for i, s in enumerate(statements)
              if any(a["buffer"] == cw["buffer"] and
                     cw["offset"] <= a["offset"] < cw["offset"] + cw["size"]
                     for a in s["args"]))
    classifier = statements[at]
    inputs = [(p, (a["buffer"], a["offset"]))
              for p, a in enumerate(classifier["args"]) if p and
              (a["buffer"], a["offset"]) not in weight_slots]
    if len(inputs) != 1 or inputs[0][1][0] != "buf_0":
        raise RuntimeError(f"classifier inputs are not one arena tensor: {inputs}")
    position, activation = inputs[0]
    source = data["functions"][classifier["kernel"]]
    params = re.findall(r"device (\w+)\* data\d+_(\d+)", source)
    if params[position] != ("half", str(data["channels"] * NVOX)):
        raise RuntimeError(f"classifier input is {params[position]}, not "
                           f"{data['channels']}xNVOX halves")
    producer = next(s for s in reversed(statements[:at])
                    if (s["args"][0]["buffer"], s["args"][0]["offset"]) == activation)
    if not ACTIVATION_OPS.search(data["functions"][producer["kernel"]]) or \
            ACTIVATION_OPS.search(source):
        raise RuntimeError("GELU is not the last op before the classifier")
    return at, activation[1]


def add_mindgrab_head_bias(data: dict) -> None:
    """Add a bias argument to the captured head; preserve every tuned conv."""
    if "seq_conv_argmax.bias" in data["weights"]:
        return
    at, _ = classifier_boundary(data)
    statement = data["statements"][at]
    source = data["functions"][statement["kernel"]]
    # The two-class head uses its y group index as the output class. Refuse
    # an unfamiliar export instead of quietly modifying a different kernel.
    if "int gidx1 = gid.y; /* 2 */" not in source:
        raise RuntimeError("MindGrab classifier class axis changed")
    source, params = re.subn(r", uint3 gid", ", device float* classifier_bias, uint3 gid", source, count=1)
    source, stores = re.subn(r"(  \*\(data0_\d+[^\n]+ = )([^\n]+);",
                            r"\1(\2) + classifier_bias[gidx1];", source)
    if params != 1 or stores != 1:
        raise RuntimeError("MindGrab classifier signature/store changed")
    offset = (data["buffers"]["buf_0"]["size"] + 3) // 4 * 4
    data["buffers"]["buf_0"]["size"] = offset + 8
    data["weights"]["seq_conv_argmax.bias"] = {
        "buffer": "buf_0", "offset": offset, "size": 8, "dtype": "float"}
    statement["args"].append({"buffer": "buf_0", "offset": offset})
    data["functions"][statement["kernel"]] = source


def write_header(path: Path, data: dict, model_name: str) -> None:
    kernel_names = list(data["functions"])
    kernel_index = {name: i for i, name in enumerate(kernel_names)}
    buffer_index = {"buf_0": 0, "input0": 1, "output0": 2}
    args = []
    dispatches = []
    for statement in data["statements"]:
        first = len(args)
        for arg in statement["args"]:
            args.append((buffer_index[arg["buffer"]], arg["offset"]))
        dispatches.append((
            kernel_index[statement["kernel"]], first, len(statement["args"]),
            *statement["global_size"], *statement["local_size"],
        ))

    if model_name == "mindgrab":
        weight_order = [f"model.{3 * layer}.weight" for layer in range(25)]
        weight_order.extend(("seq_conv_argmax.weight", "seq_conv_argmax.bias"))
        symbol = "mindgrab"
    else:
        weight_order = []
        for layer in range(13):
            weight_order.extend((f"model.{3 * layer}.weight",
                                 f"model.{3 * layer + 1}.weight",
                                 f"model.{3 * layer + 1}.bias"))
        weight_order.extend(("seq_conv_argmax.weight", "seq_conv_argmax.bias"))
        symbol = model_name if model_name in ("mindmap", "mindsnap") else "model16"
    weights = [data["weights"][name] for name in weight_order]
    # bc_metal_arg/bc_metal_weight offsets are uint32_t: past 4 GiB they would
    # truncate silently and alias live buffers.
    if data["buffers"]["buf_0"]["size"] >= 2**32:
        raise RuntimeError("arena exceeds the uint32 offsets backend_metal.mm uses")
    # Each tinygrad program is valid as a standalone Metal library. BEAM can
    # add identically named WMMA helpers to several programs, however, and the
    # generated header combines every program into one library. Namespace those
    # private helpers per program so the combined source has no redefinitions.
    sources = []
    for index, function_source in enumerate(data["functions"].values()):
        sources.append(re.sub(
            r"\b(__WMMA_[A-Za-z0-9_]+)\b",
            lambda match: f"{match.group(1)}_bc{index}",
            function_source,
        ))
    source = "\n".join(sources)
    boundary = classifier_boundary(data)
    if ")BCMSL\"" in source:
        raise RuntimeError("MSL contains the generated raw-string delimiter")

    lines = [
        f"/* Generated by scripts/export_tinygrad_metal.py with BEAM={data['beam']}; do not edit. */",
        f"#ifndef BRAINCHOPC_METAL_{symbol.upper()}_GENERATED_H",
        f"#define BRAINCHOPC_METAL_{symbol.upper()}_GENERATED_H",
        "",
        f"#define BC_METAL_{symbol.upper()}_BEAM {data['beam']}",
        # The channel count the EXPORTED KERNELS were built for, which is not
        # necessarily the model's logical channel count: MindGrab is a
        # 15-channel network exported at 16 so its convolutions can reach the
        # simdgroup matrix units (see load_mindgrab_padded). backend_metal.mm
        # packs weights to this width and zero-fills the difference, so the
        # padding is declared here rather than assumed on both sides.
        f"#define BC_METAL_{symbol.upper()}_CHAN {data['channels']}",
        f"#define BC_METAL_{symbol.upper()}_ARENA_BYTES {data['buffers']['buf_0']['size']}ULL",
        f"#define BC_METAL_{symbol.upper()}_INPUT_BYTES {data['buffers']['input0']['size']}ULL",
        f"#define BC_METAL_{symbol.upper()}_OUTPUT_BYTES {data['buffers']['output0']['size']}ULL",
        # Where a device head (--pve) takes over: dispatches from
        # CLASSIFIER_DISPATCH on are skipped, and the post-GELU activation is
        # read from the arena. The stride is tinygrad's contiguous NCDHW
        # layout, channel-planar; classifier_boundary() proves the rest.
        f"#define BC_METAL_{symbol.upper()}_CLASSIFIER_DISPATCH {boundary[0]}",
        f"#define BC_METAL_{symbol.upper()}_CLASSIFIER_INPUT_OFFSET {boundary[1]}ULL",
        f"#define BC_METAL_{symbol.upper()}_CLASSIFIER_INPUT_CHAN_STRIDE {NVOX}",
        "",
        f'static const char bc_metal_{symbol}_source[] = R"BCMSL(',
        source,
        ')BCMSL";',
        "",
        f"static const char *const bc_metal_{symbol}_kernel_names[] = {{",
    ]
    lines.extend(f'\t"{name}",' for name in kernel_names)
    lines.extend(["};", "", f"static const bc_metal_arg bc_metal_{symbol}_args[] = {{"])
    lines.extend(f"\t{{{buffer}, {offset}U}}," for buffer, offset in args)
    lines.extend(["};", "", f"static const bc_metal_dispatch bc_metal_{symbol}_dispatches[] = {{"])
    lines.extend(
        "\t{%d, %d, %d, {%d, %d, %d}, {%d, %d, %d}}," % dispatch
        for dispatch in dispatches
    )
    lines.extend(["};", "", f"static const bc_metal_weight bc_metal_{symbol}_weights[] = {{"])
    lines.extend(
        f"\t{{{weight['offset']}U, {weight['size']}U, "
        f"{1 if weight['dtype'] == 'half' else 0}}},"
        for weight in weights
    )
    lines.extend(["};", "", "#endif", ""])
    path.write_text("\n".join(lines))


def replan_by_lifetime(captured, weight_ids) -> None:
    """CapturedJit.replan_buffers_memory_layout() with real lifetimes.

    tinygrad 0.12 hands its planner ONE step holding every buffer, so every
    intermediate looks live at once and nothing is reused: 6.78 GB of arena
    for the 104-class model. One step per dispatch lets it reuse. Weights are
    uploaded once before the first dispatch, so they are pinned to the first
    and last step rather than planned by use.
    """
    from dataclasses import replace
    from tinygrad.engine.memory import _internal_memory_planner
    from tinygrad.nn.state import get_parameters

    outputs = {id(t.uop.buffer) for t in get_parameters(captured.ret)}
    cache = captured.jit_cache
    steps = [[b for b in item.bufs if b is not None and id(b.base) not in outputs]
             for item in cache]
    weights = [b for step in steps for b in step if id(b.base) in weight_ids]
    steps[0] += weights
    steps[-1] += weights
    assignment = _internal_memory_planner(steps, ignore_checks=True)
    captured.jit_cache = [replace(item, bufs=[assignment.get(b, b) if b is not None else None
                                              for b in item.bufs]) for item in cache]
    for old, new in assignment.items():
        if old.is_allocated():
            new.ensure_allocated().copyin(old.as_buffer())
    captured.__post_init__()


# MindGrab is a 15-channel network. It is built here as a SIXTEEN-channel one
# with the sixteenth channel zeroed, and that is a performance fix, not a
# cosmetic one.
#
# tinygrad names its kernels by loop extent, and at 15 channels MindGrab's
# convolutions came out as `r_..._15_4_15_3_3_3`: the channel count survives as
# a literal, indivisible bound. 15 does not tile into the 8x8x8 operands of
# `simdgroup_multiply_accumulate`, so BEAM cannot reach the matrix units at all
# -- measured, every one of the five hidden-conv kernels contained zero
# `__WMMA`, against 33-97 for model16, whose channel count is 16. That is the
# whole of MindGrab's 148 ms/layer against model16's 59 ms.
#
# WHY THE PADDING IS INERT, rather than merely "small". The normalization is
# `nn.GroupNorm(num_groups=out_channels, num_channels=out_channels)` -- one
# group per channel, so channels never mix in the norm. A zeroed output channel
# therefore has mean 0 and variance 0, normalises to (0-0)/sqrt(0+eps) = 0, and
# MindGrab declares affine=False so nothing adds a bias to it. gelu(0) = 0. The
# channel stays identically zero for all 25 layers and contributes nothing
# through any later convolution or the classifier. The result is unchanged
# arithmetic, not an approximation.
#
# This is the same trick the rest of the project already relies on:
# MESHNET_CPU_CHANNEL_STRIDE and BC_WG_CS are both 16 for exactly this reason,
# and benchmarking records that bounding the CPU loops by 15 instead made Clang
# emit them scalar and cost ~10%.
PAD_CHANNELS = 16


def load_mindgrab_padded(MeshNet, load_state_dict, get_state_dict, Tensor):
    """Build MindGrab at PAD_CHANNELS channels from its 15-channel checkpoint."""
    from tinygrad.nn.state import torch_load
    from brainchop.api import _load_model  # noqa: F401  (keeps the dependency honest)
    from brainchop.utils import AVAILABLE_MODELS  # noqa: F401

    cache = Path.home() / ".cache" / "brainchop" / "models" / "mindgrab"
    config_fn, weights_fn = cache / "model.json", cache / "model.pth"
    if not config_fn.exists() or not weights_fn.exists():
        # Let brainchop fetch them, then use its cache.
        _load_model("mindgrab")
    if not config_fn.exists() or not weights_fn.exists():
        raise FileNotFoundError(f"mindgrab config/weights not found under {cache}")

    # MeshNet's own set_channel_num() rewrites every hidden layer to `channels`,
    # leaving the 1-channel input and 2-class output alone.
    model = MeshNet(in_channels=1, n_classes=2, channels=PAD_CHANNELS,
                    config_file=str(config_fn))
    target = get_state_dict(model)
    saved_state = torch_load(str(weights_fn))
    head_bias = saved_state.pop("model.25.bias")
    if head_bias.shape != (2,):
        raise RuntimeError("MindGrab checkpoint head bias must have two classes")
    if len(saved_state) != len(target):
        raise RuntimeError("MindGrab checkpoint backbone parameter count changed")
    # Map before padding; convert_keys() correctly rejects mismatched shapes
    # in ordinary inference, whereas this export deliberately widens 15 to 16.
    saved = dict(zip(target, saved_state.values()))

    padded = {}
    for name, tensor in saved.items():
        want = target[name].shape
        have = tensor.shape
        if want == have:
            padded[name] = tensor
            continue
        # Zero-pad the leading (output) and second (input) channel axes only;
        # the 3x3x3 spatial extent must never change.
        if len(want) != len(have) or any(w < h for w, h in zip(want, have)) \
                or want[2:] != have[2:]:
            raise RuntimeError(f"{name}: cannot pad {have} to {want}")
        padded[name] = tensor.pad(
            tuple((0, w - h) for w, h in zip(want, have))).contiguous()

    load_state_dict(model, padded, strict=True, verbose=False)
    model.init_seq_conv_argmax()
    # load_meshnet() halves the weights when FP16 is set, and the checked-in
    # header was generated that way (bc_metal_weight .half = 1). Constructing
    # the model by hand skips that, so do it explicitly -- otherwise this change
    # would silently alter the weight dtype as well as the channel count.
    if os.environ.get("FP16"):
        model = model.half()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model",
                        choices=("mindgrab", "model16chan18cls", "mindmap",
                                 "mindsnap"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--header", type=Path)
    parser.add_argument(
        "--beam",
        type=int,
        default=0,
        help="tinygrad BEAM search width; 0 disables BEAM search",
    )
    parser.add_argument("--fuse-chunk", type=int, default=8,
                        help="classifier channels per fused conv+argmax chunk "
                             "(mindsnap only)")
    args = parser.parse_args()
    if args.beam < 0:
        parser.error("--beam must be nonnegative")

    # Pin generator settings so ambient developer shell variables cannot alter
    # the checked-in program. BEAM remains opt-in through the explicit flag.
    os.environ.pop("DEBUG", None)
    os.environ["METAL"] = "1"
    os.environ["FP16"] = "1"
    os.environ["BEAM"] = str(args.beam)

    from tinygrad.tensor import Tensor
    from tinygrad.device import Device
    from tinygrad.helpers import Context
    from tinygrad.nn.state import get_state_dict, load_state_dict, safe_load
    from brainchop.api import _load_model
    from brainchop.tiny_meshnet import MeshNet
    from brainchop import export_model as exporter

    if "METAL" not in exporter.EXPORT_SUPPORTED_DEVICE:
        exporter.EXPORT_SUPPORTED_DEVICE.append("METAL")
    if Device.DEFAULT != "METAL":
        raise RuntimeError(f"expected METAL, got {Device.DEFAULT}")

    chan = 24 if args.model in ("mindmap", "mindsnap") else 16
    nclass = 104 if args.model == "mindsnap" else 18
    if args.model == "mindgrab":
        model = load_mindgrab_padded(
            MeshNet, load_state_dict, get_state_dict, Tensor)
    else:
        # The two affine models share a topology and differ only in width, so
        # one branch covers both. The fp16 weights are RESCALED relative to
        # model.bin -- a positive per-layer factor, which GroupNorm absorbs
        # exactly -- so the kernels must be built from model.safetensors, not
        # from the payload the CPU engine compiles in.
        dilations = (1, 3, 5, 7, 13, 19, 31, 19, 13, 7, 5, 3, 1)
        layers = [
            {"in_channels": 1 if i == 0 else chan, "out_channels": chan,
             "kernel_size": 3, "stride": 1, "padding": dilation,
             "dilation": dilation}
            for i, dilation in enumerate(dilations)
        ]
        layers.append({"in_channels": chan, "out_channels": nclass,
                       "kernel_size": 1, "stride": 1, "padding": 0,
                       "dilation": 1, "bias": True})
        config = {"dropout_p": 0, "bnorm": True, "gelu": True,
                  "affine": True, "bias": False, "layers": layers}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as config_file:
            json.dump(config, config_file)
            config_file.flush()
            model = MeshNet(1, nclass, chan, config_file.name)
        saved = safe_load(f"src/models/{args.model}/model.safetensors")
        saved = {name.removeprefix("m."): tensor for name, tensor in saved.items()
                 if not name.startswith("m.seq_conv_argmax.")}
        load_state_dict(model, saved, strict=True, verbose=False)
        model.init_seq_conv_argmax()
    # Match the reference fp16 profiler and exported browser runner. The normal
    # CLI currently leaves this tensor fp32 by accident, making only its first
    # convolution fp32 before all later activations are cast back to fp16.
    dummy = Tensor.empty(1, 1, 256, 256, 256, dtype="float16").contiguous().realize()
    with Context(JIT=2):
        # 104 classes in one chunk materialise 104 fp16 logit volumes (3.5 GB
        # of arena); chunking the fused conv+argmax keeps FUSE_CHUNK of them.
        forward = model if nclass == 18 else (
            lambda x: model(x, fuse_chunk=args.fuse_chunk))
        run, special_names = exporter.jit_model(forward, dummy)

    state = get_state_dict(model)
    state_names = {
        id(t.uop.base.realized): name
        for name, t in state.items()
        if t.uop.base.realized is not None
    }
    weight_slots = {
        (item_index, arg_index): state_names[id(buffer)]
        for item_index, item in enumerate(run.jit_cache)
        for arg_index, buffer in enumerate(item.bufs)
        if id(buffer) in state_names
    }

    # Exporting does not execute this captured object again, so plan the
    # intermediate buffers explicitly. This collapses per-layer temporaries
    # into a small reusable working set while preserving inputs, outputs, and
    # model parameters.
    assert run.captured is not None
    replan_by_lifetime(run.captured, weight_ids=set(state_names))
    special_names = {}
    for (j, i), input_index in run.input_replace.items():
        realized = dummy.uop.base.realized
        run.jit_cache[j].bufs[i] = realized
        special_names[id(realized.base)] = f"input{input_index}"
    for output_index, output in enumerate(run.captured.ret):
        special_names[id(output.uop.base.realized.base)] = f"output{output_index}"

    functions = {}
    physical = {}
    dispatches = []

    def physical_name(buffer):
        base = buffer.base
        key = id(base)
        if key not in physical:
            physical[key] = {
                "name": special_names.get(key, f"buf_{len(physical)}"),
                "size": base.nbytes,
                "dtype": base.dtype.name,
                "buffer": base,
            }
        return physical[key]["name"]

    for item in run.jit_cache:
        item.lower()
        program = item.prg.p
        functions[program.function_name] = program.src
        dispatches.append({
            "kernel": program.function_name,
            "args": [
                {"buffer": physical_name(buffer), "offset": buffer.offset}
                for buffer in item.bufs
            ],
            "global_size": program.global_size,
            "local_size": program.local_size,
        })

    buffers = {
        entry["name"]: {
            "size": entry["size"], "dtype": entry["dtype"],
        }
        for key, entry in physical.items()
    }
    weights = {}
    for (item_index, arg_index), state_name in weight_slots.items():
        buffer = run.jit_cache[item_index].bufs[arg_index]
        weights[state_name] = {
            "buffer": physical_name(buffer), "offset": buffer.offset,
            "size": buffer.nbytes, "dtype": buffer.dtype.name,
        }
    input_names = [name for name in buffers if name.startswith("input")]
    output_names = [name for name in buffers if name.startswith("output")]
    # The width the kernels were built for. MindGrab's logical 15 is exported
    # padded to PAD_CHANNELS; model16 is natively 16.
    channels = PAD_CHANNELS if args.model == "mindgrab" else chan
    data = {
        "backend": "METAL",
        "beam": args.beam,
        "channels": channels,
        "functions": functions,
        "statements": dispatches,
        "buffers": buffers,
        "weights": weights,
        "input_sizes": {name: buffers[name]["size"] for name in input_names},
        "output_sizes": {name: buffers[name]["size"] for name in output_names},
    }
    data["state"] = {
        name: {"shape": list(tensor.shape), "dtype": tensor.dtype.name}
        for name, tensor in state.items()
    }
    if args.model == "mindgrab":
        add_mindgrab_head_bias(data)
    args.output.write_text(json.dumps(data, separators=(",", ":")))
    if args.header is not None:
        write_header(args.header, data, args.model)
    print(
        f"wrote {args.output}: {len(data['functions'])} kernels, "
        f"{len(data['statements'])} dispatches, {len(data['buffers'])} allocations"
    )


if __name__ == "__main__":
    main()
