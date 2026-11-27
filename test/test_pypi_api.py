#!/usr/bin/env python3
"""Zero-dependency Python API compatibility contracts for `import brainchop`."""

from contextlib import redirect_stderr
import gzip
from io import StringIO
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packaging"))

import brainchop
from brainchop import (
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
from brainchop_cli import native


def make_tiny_nifti(shape=(4, 5, 6), datatype=2, fill=7) -> bytes:
    """Build a valid minimal little-endian NIfTI-1 file in memory."""
    hdr = bytearray(352)
    struct.pack_into("<i", hdr, 0, 348)
    struct.pack_into("<8h", hdr, 40, 3, shape[0], shape[1], shape[2], 1, 1, 1, 1)
    struct.pack_into("<h", hdr, 70, datatype)
    bitpix = {2: 8, 4: 16, 16: 32}[datatype]
    struct.pack_into("<h", hdr, 72, bitpix)
    struct.pack_into("<8f", hdr, 76, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
    struct.pack_into("<f", hdr, 108, 352.0)
    struct.pack_into("<2f", hdr, 112, 1.0, 0.0)
    hdr[344:348] = b"n+1\x00"
    nvox = shape[0] * shape[1] * shape[2]
    if datatype == 2:
        payload = bytes([fill]) * nvox
    elif datatype == 4:
        payload = struct.pack("<" + "h" * nvox, *([fill] * nvox))
    elif datatype == 16:
        payload = struct.pack("<" + "f" * nvox, *([float(fill)] * nvox))
    else:
        raise ValueError(datatype)
    return bytes(hdr) + payload


class BrainchopApiTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.nii_bytes = make_tiny_nifti((4, 5, 6), datatype=2, fill=3)
        self.in_gz = self.root / "input.nii.gz"
        self.in_gz.write_bytes(gzip.compress(self.nii_bytes))
        self.in_nii = self.root / "input.nii"
        self.in_nii.write_bytes(self.nii_bytes)

    def tearDown(self):
        self.tmp.cleanup()

    def test_public_exports_and_list_models(self):
        for sym in ("Volume", "load", "save", "segment", "list_models",
                    "export", "optimize", "export_classes", "__version__"):
            self.assertTrue(hasattr(brainchop, sym), f"missing export {sym}")
        models = list_models()
        self.assertEqual(set(models), {"mindgrab", "mindmap-fast", "mindmap", "mindmap-omni", "mindsnap"})

    def test_load_and_save_without_numpy(self):
        with patch.dict(sys.modules, {"numpy": None}):
            vol = load(self.in_gz)
            self.assertIsInstance(vol, Volume)
            self.assertEqual(vol.shape, (4, 5, 6))
            self.assertEqual(vol.dtype, "uint8")
            self.assertEqual(len(vol.header), 352)
            # Accessing .data falls back to stdlib view when numpy is absent
            data = vol.data
            self.assertEqual(data.shape, (4, 5, 6))
            self.assertEqual(data.dtype, "uint8")
            self.assertEqual(len(data.tobytes()), 4 * 5 * 6)

            out_gz = self.root / "out.nii.gz"
            out_nii = self.root / "out.nii"
            save(vol, out_gz)
            save(vol, out_nii)
            self.assertEqual(gzip.decompress(out_gz.read_bytes()), self.nii_bytes)
            self.assertEqual(out_nii.read_bytes(), self.nii_bytes)

    def test_load_batch_and_errors(self):
        vols = load([self.in_gz, self.in_nii], ct=True, comply=True)
        self.assertEqual(len(vols), 2)
        self.assertTrue(vols[0]._load_opts["ct"])
        with self.assertRaises(FileNotFoundError):
            load(self.root / "does_not_exist.nii.gz")
        bad = self.root / "bad.nii"
        bad.write_bytes(b"not a nifti file")
        with self.assertRaises(ValueError):
            load(bad)

    def test_segment_translates_flags_and_returns_volume(self):
        label_nii = make_tiny_nifti((4, 5, 6), datatype=2, fill=5)
        captured_cmds = []

        def fake_run(cmd, capture_output=True, text=True):
            captured_cmds.append(list(cmd))
            out_idx = cmd.index("-o") + 1
            Path(cmd[out_idx]).write_bytes(label_nii)
            if "--mask" in cmd:
                mask_idx = cmd.index("--mask") + 1
                Path(cmd[mask_idx]).write_bytes(make_tiny_nifti((4, 5, 6), datatype=2, fill=1))
            if "--export-classes" in cmd:
                base = cmd[out_idx][:-4]
                for c in range(2):
                    Path(f"{base}_c{c}.nii").write_bytes(
                        make_tiny_nifti((4, 5, 6), datatype=16, fill=c + 1)
                    )
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.object(native, "find_executable", return_value="/fake/brainchop-mindmap"), \
             patch("brainchop.api.subprocess.run", side_effect=fake_run):
            vol = load(self.in_gz, ct=True)
            res = segment(vol, "mindmap", threads=4, no_turbo=True)
            self.assertIsInstance(res, Volume)
            self.assertEqual(res.shape, (4, 5, 6))
            cmd = captured_cmds[-1]
            self.assertNotIn("--save-conform", cmd)
            self.assertIn("--ct", cmd)
            self.assertIn("--no-turbo", cmd)
            self.assertEqual(cmd[cmd.index("-cpu") + 1], "4")
            self.assertEqual(cmd[-1], str(self.in_gz.resolve()))

            out_path = self.root / "seg.nii.gz"
            save(res, out_path)
            saved = gzip.decompress(out_path.read_bytes())
            # NIFTI_INTENT_LABEL (1002) is tagged on label outputs
            self.assertEqual(struct.unpack_from("<h", saved, 68)[0], 1002)

            # Direct path + save_conform=True + alias
            res2 = segment(self.in_nii, "16chan18cls", save_conform=True)
            self.assertIn("--save-conform", captured_cmds[-1])
            self.assertIsInstance(res2, Volume)

            # Backward-compatible inverse_conform=True still yields native output
            res3 = segment(self.in_nii, "16chan18cls", inverse_conform=True)
            self.assertNotIn("--save-conform", captured_cmds[-1])
            self.assertIsInstance(res3, Volume)

            # Batch input returns list
            batch_res = segment([self.in_gz, self.in_nii], "mindmap")
            self.assertIsInstance(batch_res, list)
            self.assertEqual(len(batch_res), 2)

            # MindGrab with mask + return_raw + export_classes
            (stripped, mask_vol), raw = segment(
                self.in_gz, "mindgrab", mask=True, border=2, return_raw=True
            )
            self.assertIsInstance(stripped, Volume)
            self.assertIsInstance(mask_vol, Volume)
            self.assertIsInstance(raw, RawClassMaps)
            self.assertEqual(len(raw), 2)
            self.assertNotIn("--save-conform", captured_cmds[-1])
            self.assertIn("--export-classes", captured_cmds[-1])
            exported = export_classes(raw, vol.header, self.root / "classes.nii.gz")
            self.assertEqual(len(exported), 2)
            for p in exported:
                self.assertTrue(Path(p).is_file())

    def test_segment_pve_returns_four_tissue_volumes(self):
        def fake_run(cmd, capture_output=True, text=True):
            out_base = cmd[cmd.index("-o") + 1][:-4]
            for tissue in ("gm", "wm", "csf", "brain"):
                Path(f"{out_base}_{tissue}.nii").write_bytes(
                    make_tiny_nifti((4, 5, 6), datatype=2, fill=100)
                )
            return subprocess.CompletedProcess(cmd, 0, "", "")

        with patch.object(native, "find_executable", return_value="/fake/brainchop-mindmap"), \
             patch("brainchop.api.subprocess.run", side_effect=fake_run):
            tissues = segment(self.in_gz, "mindmap", pve=True)
        self.assertEqual(set(tissues), {"gm", "wm", "csf", "brain"})
        for vol in tissues.values():
            self.assertIsInstance(vol, Volume)
            self.assertEqual(vol.shape, (4, 5, 6))

    def test_compatibility_guards_and_legacy_hints(self):
        with redirect_stderr(StringIO()) as err:
            self.assertIsNone(optimize("mindmap", beam=2))
        self.assertIn("precompiled", err.getvalue())

        with self.assertRaises(NotImplementedError):
            export("mindmap", str(self.root))

        with self.assertRaises(NotImplementedError):
            segment(self.in_gz, "mindmap", tta=True)

        with self.assertRaisesRegex(ValueError, "mindmap"):
            segment(self.in_gz, "subcortical")

        with self.assertRaisesRegex(ValueError, "mindsnap"):
            segment(self.in_gz, "DKatlas")

    def test_python_m_brainchop_and_version_flag(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT / "packaging")}
        for args in (["-S", "-m", "brainchop", "--version"],
                     ["-S", "-m", "brainchop", "--list"],
                     ["-S", "-m", "brainchop_cli", "--version"]):
            proc = subprocess.run([sys.executable, *args], env=env, capture_output=True, text=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertTrue(proc.stdout.strip())
        # Verify stdlib-only import with -S (no site-packages)
        proc = subprocess.run(
            [sys.executable, "-S", "-c", "import brainchop; print(sorted(brainchop.list_models()))"],
            env=env, capture_output=True, text=True,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("mindmap-fast", proc.stdout)


if __name__ == "__main__":
    unittest.main()
