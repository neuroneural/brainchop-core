#!/usr/bin/env python3
"""GPU-free tests for the model-generic HIP developer compiler."""
import argparse
import copy
import json
import os
import shutil
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import build_hip_artifacts as builder  # noqa: E402
from export_hip_onnx import model_spec  # noqa: E402
from hip_memory import check_memory
from hip_artifacts import verify  # noqa: E402


def msgpack(value):
    if isinstance(value, int):
        if value < 128:
            return bytes([value])
        if value < 256:
            return b"\xcc" + bytes([value])
        if value < 65536:
            return b"\xcd" + value.to_bytes(2, "big")
        return b"\xce" + value.to_bytes(4, "big")
    if isinstance(value, str):
        data = value.encode()
        return (bytes([0xa0 + len(data)]) if len(data) < 32 else
                b"\xd9" + bytes([len(data)])) + data
    if isinstance(value, list):
        return bytes([0x90 + len(value)]) + b"".join(msgpack(item) for item in value)
    if isinstance(value, dict):
        return bytes([0x80 + len(value)]) + b"".join(
            msgpack(key) + msgpack(item) for key, item in value.items())
    raise TypeError(type(value))


def code_object(entry, arguments, target="gfx1151", static_lds=0):
    args = [{".offset": index * 8, ".size": 8,
             ".value_kind": "global_buffer"} for index in range(arguments)]
    metadata = {
        "amdhsa.kernels": [{".args": args, ".group_segment_fixed_size": static_lds,
                            ".kernarg_segment_size": arguments * 8,
                            ".name": entry, ".symbol": entry + ".kd"}],
        "amdhsa.target": "amdgcn-amd-amdhsa--" + target,
        "amdhsa.version": [1, 2],
    }
    elf = bytearray(b"\x7fELF\x02\x01" + bytes(58))
    elf[16:18] = (3).to_bytes(2, "little")
    return bytes(elf) + b"AMDGPU\x00" + msgpack(metadata)


def write_group(directory, entries):
    directory.mkdir(parents=True)
    manifest = []
    for index, (symbol, nargs, roles) in enumerate(entries):
        filename = f"kernel_{index}.hsaco"
        (directory / filename).write_bytes(code_object(symbol if symbol == "bc_hip_classify" else f"entry_{index}", nargs,
                                                       static_lds=64))
        item = {"file": filename, "symbol": symbol,
                "grid": [1, 1, 1], "block": [64, 1, 1]}
        if roles is None:
            item.update({"global": 64, "local": 64, "output_arg": nargs - 1,
                         "inputs": []})
        else:
            item.update({"dynamic_lds_bytes": 0,
                         "args": [{"role": role, "bytes": size}
                                  for role, size in roles]})
        manifest.append(item)
    (directory / "manifest.json").write_text(json.dumps(manifest))


class CompilerTests(unittest.TestCase):
    def test_all_checked_in_models_are_discovered(self):
        expected = {
            "mindgrab": (15, 25, 2, False, True),
            "model16chan18cls": (16, 13, 18, True, True),
            "mindmap": (24, 13, 18, True, True),
            "mindmap_omni": (24, 13, 18, True, True),
            "mindsnap": (24, 13, 104, True, True),
        }
        for name, values in expected.items():
            with self.subTest(model=name):
                spec = model_spec(name)
                self.assertEqual((spec["channels"], spec["hidden_layers"],
                                  spec["classes"], spec["has_affine"],
                                  spec["classifier_bias"]), values)
                self.assertEqual(spec["supports_pve"], name == "mindmap")

    def arguments(self, **overrides):
        values = dict(model="mindmap", gfx="gfx1151", num_compute_units=16,
                      num_chiplets=1, trace_dim=32, opset=18,
                      migraphx_driver="migraphx-driver", exhaustive_tune=True,
                      native_target=False,
                      hipcc="hipcc",
                      backend_options="{convolution_layout:channels_last}")
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_plan_targets_the_requested_architecture_and_model(self):
        spec = model_spec("mindmap")
        plan = builder.command_plan(self.arguments(), spec, Path("/tmp/work"))
        command = plan["compile"]
        self.assertEqual(command[command.index("--gpu-arch") + 1], "gfx1151")
        self.assertEqual(command[command.index("--gpu-num-cus") + 1], "16")
        start = command.index("--input-dim") + 1
        self.assertEqual(command[start:start + 6],
                         ["@volume", "1", "1", "256", "256", "256"])
        self.assertIn("--exhaustive-tune", command)
        norm = plan["extract_tinygrad"][2]
        self.assertEqual(norm[norm.index("--channels") + 1], "24")
        self.assertIn("--affine", norm)
        self.assertIsNotNone(plan["compile_support"])
        self.assertIsNone(plan["compile_classify"])
        self.assertIn("gfx1151", plan["compile_support"])

        model16 = builder.command_plan(
            self.arguments(model="model16chan18cls"),
            model_spec("model16chan18cls"), Path("/tmp/work-model16"))
        self.assertIsNone(model16["compile_support"])
        self.assertIsNotNone(model16["compile_classify"])

    def test_native_plan_omits_newer_cross_compile_options(self):
        spec = model_spec("mindmap")
        plan = builder.command_plan(
            self.arguments(native_target=True), spec, Path("/tmp/work"))
        command = plan["compile"]
        self.assertNotIn("--gpu-arch", command)
        self.assertNotIn("--gpu-num-cus", command)
        self.assertNotIn("--gpu-num-chiplets", command)

    def test_stage_cache_skips_a_completed_stage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "out"
            command = [sys.executable, "-c",
                       "from pathlib import Path; Path(r'%s').write_text('ok')" % output]
            builder.run_stage("small", command, [output], {"input": 1}, root)
            output.write_text("cached")
            builder.run_stage("small", ["this-command-does-not-exist"], [output],
                              {"input": 1}, root)
            self.assertEqual(output.read_text(), "cached")

    def test_compile_timeout_preserves_problem_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cache = root / "problem-cache.json"
            command = [
                sys.executable, "-c",
                "import os,time; open(os.environ['MIGRAPHX_PROBLEM_CACHE'], "
                "'w').write('checkpoint'); time.sleep(5)",
            ]
            env = os.environ.copy()
            env["MIGRAPHX_PROBLEM_CACHE"] = str(cache)
            with self.assertRaisesRegex(RuntimeError, "can resume"):
                builder.run_stage("compile", command, [root / "model.mxr"],
                                  {"input": 1}, root, env=env, timeout=0.05)
            self.assertEqual(cache.read_text(), "checkpoint")

    def test_mindmap_pack_is_contract_valid(self):
        spec = model_spec("mindmap")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            migraphx = root / "migraphx"
            entries = [("mlir_convert_convolution_reshape_convert", 4, None)]
            entries += [("mlir_convolution_reshape_convert", 4, None)] * 12
            entries += [("mlir_convolution_broadcast_add_convert", 4, None)]
            write_group(migraphx, entries)
            tinygrad = root / "tinygrad"
            write_group(tinygrad / "widen", [
                ("widen", 2, [("out", 256 ** 3 * 4), ("x", 256 ** 3 * 2)])])
            write_group(tinygrad / "moments", [
                ("moments", 2, [("out", 24 * 2 * 4),
                                  ("x", 24 * 256 ** 3 * 2)])])
            write_group(tinygrad / "norm_gelu", [
                ("norm", 6, [("out", 24 * 256 ** 3 * 2),
                              ("x", 24 * 256 ** 3 * 2),
                              ("mean", 24 * 4), ("inv", 24 * 4),
                              ("scale", 24 * 4), ("bias", 24 * 4)])])
            write_group(tinygrad / "argmax", [
                ("argmax", 2, [("out", 256 ** 3),
                                ("logits", 18 * 256 ** 3 * 4)])])
            support = root / "support"
            write_group(support, [
                ("bc_hip_tissue_priors", 2,
                 [("logits", 18 * 256 ** 3 * 4),
                  ("out", 4 * 256 ** 3 * 4)])])
            output = root / "artifact"
            manifest = builder.pack_artifact(
                spec, "gfx1151", 16, 1, 32, migraphx, tinygrad, support, output,
                {"rocm": "7.2.2", "migraphx": "2.18", "tinygrad": "0.14.0"},
                {"fp16": True, "exhaustive_tune": True})
            self.assertEqual(manifest["model"]["channels"], 24)
            self.assertTrue(manifest["model"]["supports_pve"])
            self.assertEqual(len(manifest["dispatches"]), 43)
            self.assertEqual(manifest["dispatches"][-2]["mode"], "labels")
            self.assertEqual(manifest["dispatches"][-1]["mode"], "pve")
            self.assertTrue(all(item["dynamic_lds_bytes"] == 0
                                for item in manifest["dispatches"]))
            self.assertGreater(manifest["buffers"][0]["bytes"], 3_000_000_000)
            catalog = (output / "catalog.h").read_text()
            self.assertIn('"mindmap"', catalog)
            self.assertIn('"gfx1151"', catalog)
            self.assertIn("BC_HIP_F16", catalog)
            self.assertIn("BC_HIP_DISPATCH_PVE", catalog)

    def test_checked_in_artifacts_and_runtime_catalogs(self):
        artifacts = sorted((ROOT / "src" / "hip_artifacts").glob("*/*"))
        self.assertIn("mindsnap", [path.parent.name for path in artifacts])
        self.assertTrue(all((ROOT / "src/models" / path.parent.name).is_dir()
                            for path in artifacts))
        for directory in artifacts:
            with self.subTest(artifact=str(directory.relative_to(ROOT))):
                manifest = verify(directory, ROOT)
                for mode in (("labels", "pve") if manifest["model"].get("supports_pve") else ("labels",)):
                    check_memory(manifest, mode)
                self.assertEqual((directory / "catalog.h").read_text(),
                                 builder.render_catalog(manifest))
                if manifest["schema_version"] >= 2 and not manifest["model"]["supports_pve"]:
                    # The native loader requires zero PVE storage for a
                    # labels-only profile, even though its output is nonempty.
                    self.assertIn("\t0,\n\t0ULL\n};", builder.render_catalog(manifest))

    def test_every_discovered_model_rejects_memory_regressions(self):
        for directory in sorted((ROOT / "src/hip_artifacts").glob("*/*")):
            original = json.loads((directory / "manifest.json").read_text())
            for mode in (("labels", "pve") if original["model"].get("supports_pve") else ("labels",)):
                with self.subTest(model=original["model"]["name"], mode=mode):
                    plan = check_memory(original, mode)
                    check_memory(original, mode, observed_bytes=plan["persistent_bytes"])
                    with self.assertRaisesRegex(ValueError, "do not match"):
                        check_memory(original, mode, observed_bytes=plan["persistent_bytes"] + 1)
                    mutated = copy.deepcopy(original)
                    mutated["buffers"][0]["bytes"] += plan["budget_bytes"] - plan["persistent_bytes"] + 1
                    with self.assertRaisesRegex(ValueError, "exceeds budget"):
                        check_memory(mutated, mode)
                    mutated = copy.deepcopy(original)
                    mutated["buffers"].append({"id": "hidden", "kind": "arena", "bytes": 1})
                    with self.assertRaisesRegex(ValueError, "only arena"):
                        check_memory(mutated, mode)

    def test_label_packer_has_no_class_volume_or_argmax_scratch(self):
        for model, classes in (("mindgrab", 2), ("model16chan18cls", 18),
                               ("model16chan18cls", 16), ("mindsnap", 104)):
            for bias in (False, True):
                with self.subTest(model=model, bias=bias), tempfile.TemporaryDirectory() as tmp:
                    spec = model_spec(model)
                    spec["classifier_bias"] = bias
                    spec["classes"] = classes
                    root = Path(tmp)
                    c, k, n = spec["channels"], spec["classes"], 256 ** 3
                    entries = [("mlir_convert_convolution_reshape_convert", 4, None)]
                    entries += [("mlir_convolution_reshape_convert", 4, None)] * (spec["hidden_layers"] - 1)
                    entries += [("mlir_convolution_broadcast_add_convert" if bias else
                                 "mlir_convolution_convert", 4 if bias else 3, None)]
                    write_group(root / "mx", entries)
                    write_group(root / "tg/widen", [("widen", 2, [("out", n * 4), ("x", n * 2)])])
                    write_group(root / "tg/moments", [("moments", 3,
                        [("out", c * 8), ("x", c * n * 2), ("scratch", c * 4096)])])
                    roles = [("out", c * n * 2), ("x", c * n * 2),
                             ("mean", c * 4), ("inv", c * 4)]
                    if spec["has_affine"]:
                        roles += [("scale", c * 4), ("bias", c * 4)]
                    write_group(root / "tg/norm_gelu", [("norm", len(roles), roles)])
                    roles = [("activation", c * n * 2), ("weight", c * k * 2)]
                    if bias:
                        roles += [("bias", k * 2)]
                    roles += [("out", n)]
                    write_group(root / "head", [("bc_hip_classify", len(roles), roles)])
                    m = builder.pack_artifact(spec, "gfx1151", 40, 1, 32,
                        root / "mx", root / "tg", None, root / "out",
                        {"rocm": "test", "migraphx": "test", "tinygrad": "test"}, {},
                        classify_directory=root / "head")
                    self.assertLessEqual(sum(b["bytes"] for b in m["buffers"]), 4 * 1024 ** 3)
                    self.assertEqual(m["toolchain"]["compile_options"]["bounded_head"]["workspace_bytes"], 0)
                    # No score storage is added even when class and hidden
                    # counts coincide with the convolution's fp32 sink extent.
                    self.assertEqual(len(m["dispatches"][-1]["args"]), 4 if bias else 3)
                    self.assertEqual(m["dispatches"][-1]["args"][-1]["bytes"], n)
                    plan = builder.command_plan(self.arguments(model=model), spec, root / "work")
                    self.assertEqual(len(plan["extract_tinygrad"]), 3)
                    self.assertNotIn("argmax", str(plan["extract_tinygrad"]))

    def test_replace_tail_preserves_body_and_compacts_scratch(self):
        spec = model_spec("mindmap")
        spec["supports_pve"] = False  # Construct a legacy labels-only source.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            shutil.copytree(ROOT / "src/hip_artifacts/mindmap/gfx1151", source)
            m = json.loads((source / "manifest.json").read_text())
            pve = m["dispatches"].pop()
            m["kernels"] = [k for k in m["kernels"] if k["id"] != pve["kernel"]]
            m["model"]["supports_pve"] = False
            m["buffers"][2]["bytes"] = 256 ** 3
            (source / "manifest.json").write_text(json.dumps(m))
            verify(source, ROOT)
            n = 256 ** 3
            write_group(root / "head", [("bc_hip_classify", 4,
                [("activation", 24 * n * 2), ("weight", 24 * 18 * 2),
                 ("bias", 18 * 2), ("out", n)])])
            args = argparse.Namespace(gfx="gfx1151", num_compute_units=40,
                num_chiplets=1, wavefront_size=32, hipcc="missing-hipcc")
            result = builder.replace_label_tail(spec, source, ROOT, root / "head", root / "out", args)
            used = {d["kernel"] for d in m["dispatches"][:-3]}
            before = [k["sha256"] for k in m["kernels"] if k["id"] in used]
            after = [k["sha256"] for k in result["kernels"][:-1]]
            self.assertEqual(before, after)
            self.assertLessEqual(result["buffers"][0]["bytes"],
                                 m["buffers"][0]["bytes"] - 18 * n * 4 - 64 * 1024 ** 2)
            self.assertEqual(len(result["dispatches"]), len(m["dispatches"]) - 2)
            # Schema 1 full-head artifacts (where access was recorded as
            # readwrite) must also upgrade cleanly to a bounded Schema 2 head.
            spec16 = model_spec("model16chan18cls")
            write_group(root / "head16", [("bc_hip_classify", 4,
                [("activation", 16 * n * 2), ("weight", 16 * 18 * 2),
                 ("bias", 18 * 2), ("out", n)])])
            upgraded16 = builder.replace_label_tail(
                spec16, ROOT / "src/hip_artifacts/model16chan18cls/gfx1151",
                ROOT, root / "head16", root / "out16", args)
            self.assertEqual(upgraded16["schema_version"], 2)
            self.assertLess(upgraded16["buffers"][0]["bytes"], 2_300_000_000)

    def test_promoted_mindsnap_memory_contract_rejects_regressions(self):
        source = ROOT / "src/hip_artifacts/mindsnap/gfx1151"
        original = json.loads((source / "manifest.json").read_text())
        self.assertEqual(original["toolchain"]["compile_options"]["bounded_head"]["workspace_bytes"], 0)
        # Freeze the tuned PR #34 body independently of the new head and arena.
        signature = builder.digest("\n".join(k["sha256"] for k in original["kernels"][:-1]).encode())
        self.assertEqual(signature, "73f9e2544a0a9c1fceadc4efe49a586273dc6c85e07f0ce3ead1beaaac578938")
        def add_interior_hole(m):
            w_start = min(w["offset"] for w in m["weights"])
            for w in m["weights"]:
                w["offset"] += 4096
            for d in m["dispatches"]:
                for a in d["args"]:
                    if a.get("kind") == "buffer" and a.get("buffer") == "arena" and a["offset"] >= w_start:
                        a["offset"] += 4096
            m["buffers"][0]["bytes"] += 4096
        mutations = [
            lambda m: m["buffers"][0].update(bytes=10_347_823_104),
            lambda m: m["buffers"].append({"id": "hidden", "bytes": 16, "alignment": 16, "kind": "arena"}),
            lambda m: m["dispatches"][-1]["args"][0].update(bytes=104 * 256 ** 3 * 4),
            lambda m: m["dispatches"][-1]["args"][1].update(offset=m["buffers"][0]["bytes"]),
            lambda m: m["dispatches"][-1]["args"][1].update(offset=m["dispatches"][-1]["args"][1]["offset"] + 2),
            lambda m: m["dispatches"][-1]["args"][0].update(offset=m["dispatches"][-1]["args"][0]["offset"] + 2),
            lambda m: m["toolchain"]["compile_options"]["bounded_head"].update(workspace_bytes=1),
            lambda m: m["toolchain"]["compile_options"].pop("bounded_head"),
            add_interior_hole,
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(source / "kernels.bin", root / "kernels.bin")
            for mutate in mutations:
                m = copy.deepcopy(original)
                mutate(m)
                (root / "manifest.json").write_text(json.dumps(m))
                with self.assertRaises(ValueError):
                    verify(root, ROOT)

    def test_mindmap_upgrade_reuses_label_bundle(self):
        spec = model_spec("mindmap")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            shutil.copytree(ROOT / "src/hip_artifacts/mindmap/gfx1151", source)
            source_manifest = json.loads((source / "manifest.json").read_text())
            if source_manifest["schema_version"] == 2:
                support_info = source_manifest["toolchain"]["compile_options"]["pve_support"]
                label_size = support_info["reused_label_bundle_size"]
                label_blob = (source / "kernels.bin").read_bytes()[:label_size]
                pve_kernel = source_manifest["dispatches"][-1]["kernel"]
                source_manifest["kernels"] = [item for item in source_manifest["kernels"]
                                               if item["id"] != pve_kernel]
                source_manifest["dispatches"] = [item for item in source_manifest["dispatches"]
                                                  if item["mode"] != "pve"]
                for item in source_manifest["dispatches"]:
                    item.pop("mode")
                source_manifest["model"].pop("supports_pve")
                # Pre-decoupling schema-1 artifacts still carry the weights hash.
                source_manifest["model"]["weights_sha256"] = builder.digest(
                    (ROOT / "src/models/mindmap/model.bin").read_bytes())
                source_manifest["buffers"][2]["bytes"] = 256 ** 3
                source_manifest.update(schema_version=1, abi_version=1)
                source_manifest["bundle"].update(
                    size=label_size, sha256=builder.digest(label_blob))
                (source / "kernels.bin").write_bytes(label_blob)
                (source / "manifest.json").write_text(
                    json.dumps(source_manifest, indent=2) + "\n")
            original = (source / "kernels.bin").read_bytes()
            support = root / "support"
            write_group(support, [
                ("bc_hip_tissue_priors", 2,
                 [("logits", 18 * 256 ** 3 * 4),
                  ("out", 4 * 256 ** 3 * 4)])])
            output = root / "output"
            args = argparse.Namespace(
                gfx="gfx1151", num_compute_units=40, num_chiplets=1,
                wavefront_size=32, hipcc="missing-hipcc")
            manifest = builder.upgrade_label_artifact(
                spec, source, support, output, args)
            self.assertTrue((output / "kernels.bin").read_bytes().startswith(original))
            self.assertEqual(manifest["schema_version"], 2)
            self.assertEqual(manifest["dispatches"][-2]["mode"], "labels")
            self.assertEqual(manifest["dispatches"][-1]["mode"], "pve")
            self.assertEqual(manifest["kernels"][-1]["producer"], "hipcc")


if __name__ == "__main__":
    unittest.main()
