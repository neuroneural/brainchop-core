#!/usr/bin/env python3
"""Artifact drift and ABI checks; requires neither nvcc nor a GPU."""
import copy
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('modules', ROOT / 'scripts/build_cuda_modules.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.out = self.root / 'src/cuda_modules'
        shutil.copytree(ROOT / 'src/cuda_modules', self.out)
        for path in m.source_hashes():
            dst = self.root / path
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / path, dst)

    def verify(self):
        return m.verify(self.out, self.root)

    def test_release_artifacts(self):
        self.verify()

    def test_corrupted_module(self):
        p = self.out / 'mindgrab.fatbin'
        data = bytearray(p.read_bytes()); data[-1] ^= 1; p.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'corrupt CUDA module'):
            self.verify()

    def test_changed_kernel_source(self):
        p = self.root / 'src/cuda_kernels.cuh'; p.write_text(p.read_text() + '\n/* changed */\n')
        with self.assertRaisesRegex(ValueError, 'sources'):
            self.verify()

    def test_weights_not_bound(self):
        # Weights upload at run time; a weights-only change must not stale modules.
        data = bytearray((ROOT / 'src/models/mindgrab/model.bin').read_bytes()); data[0] ^= 1
        (self.root / 'src/models/mindgrab/model.bin').write_bytes(data)
        self.verify()

    def test_stale_catalog(self):
        p = self.out / 'catalog.h'; p.write_text(p.read_text().replace('scalar', 'wrong') + '\n')
        with self.assertRaisesRegex(ValueError, 'catalog'):
            self.verify()

    def test_geometry_drift(self):
        p = self.root / 'src/cuda_geometry.h'
        p.write_text(p.read_text().replace('REDUCE_THREADS 256', 'REDUCE_THREADS 128'))
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.verify()
        self.assertIn('#define BC_CUDA_MODULE_REDUCE_THREADS 128', m.catalog(self.root))

    def test_geometry_metadata_corruption(self):
        p = self.out / 'manifest.json'
        data = json.loads(p.read_text()); data['geometry']['mindmap']['REDUCE_THREADS'] = 128
        p.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.verify()

    def test_model_geometry(self):
        self.assertEqual(m.geometry(self.root, 'mindgrab')['CS'], 16)
        self.assertEqual(m.geometry(self.root, 'model16chan18cls')['CS'], 16)
        self.assertEqual(m.geometry(self.root, 'mindmap')['CS'], 24)
        self.assertEqual(m.geometry(self.root, 'mindmap')['WMMA_SCRATCH'], 1024)

    def test_mindmap_geometry_drift(self):
        p = self.root / 'src/cuda_geometry_mindmap.h'
        p.write_text(p.read_text().replace('CS 24', 'CS 32'))
        with self.assertRaisesRegex(ValueError, 'geometry'):
            self.verify()

    def test_prior_contract_drift(self):
        p = self.root / 'src/pve_constants.h'
        p.write_text(p.read_text().replace('TEMPERATURE 2.0f', 'TEMPERATURE 1.0f'))
        with self.assertRaisesRegex(ValueError, 'sources'):
            self.verify()

    def test_manifest_contract_mutations(self):
        p = self.out / 'manifest.json'; original = json.loads(p.read_text())
        changes = [('abi_version', 99), ('architectures', [75]), ('nvcc_version', '0'),
                   ('minimum_driver_api', 0), ('kernels', []), ('models', {}), ('ptx_architecture', 60)]
        for key, value in changes:
            with self.subTest(key=key):
                bad = copy.deepcopy(original); bad[key] = value
                p.write_text(json.dumps(bad))
                with self.assertRaises(ValueError):
                    self.verify()
        p.write_text(json.dumps(original))


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.kernels = m.kernels()
        self.ptx = f'.version {m.PTX_VERSION}\n.target sm_90\n' + '\n'.join(
            '.visible .entry ' + k['symbol'] + '(' + ','.join(
                f'.param .{"u32" if t == "int" else "u64"} a{i}'
                for i, t in enumerate(k['parameters'])) + ')' for k in self.kernels)
        self.sass = '\n'.join('arch = sm_' + str(a) + '\n' + '\n'.join(
            'Function : ' + k['symbol'] for k in self.kernels) for a in m.ARCHES)

    def test_complete_inventory(self):
        m.validate_exports(self.ptx, self.sass, self.kernels)

    def test_ptx_header_in_sass_dump(self):
        m.validate_exports(self.ptx, self.sass + "\nFatbin ptx code:\narch = sm_90\n", self.kernels)

    def test_wrong_argument_width(self):
        with self.assertRaisesRegex(ValueError, 'ABI mismatch'):
            m.validate_exports(self.ptx.replace('.param .u64', '.param .u32', 1), self.sass, self.kernels)

    def test_missing_native_kernel(self):
        with self.assertRaisesRegex(ValueError, 'native image'):
            m.validate_exports(self.ptx, self.sass.replace('Function : ' + self.kernels[0]['symbol'], '', 1), self.kernels)

    def test_unresolved_math(self):
        with self.assertRaisesRegex(ValueError, 'unresolved'):
            m.validate_exports(self.ptx + '\n.extern .func __nv_missing();', self.sass, self.kernels)

    def test_duplicate_variants(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / 'src').mkdir()
            text = (ROOT / 'src/cuda_variants.h').read_text().replace('X(1, 256)', 'X(1, 128)')
            (root / 'src/cuda_variants.h').write_text(text)
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                m.variants(root)


if __name__ == '__main__':
    unittest.main()
