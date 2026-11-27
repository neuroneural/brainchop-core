#!/usr/bin/env python3
"""Package contracts, with simulated selection explicitly separate from GPU tests."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'packaging'))
import brainchop_build as build
import brainchop_hardware as hardware

class BuildTests(unittest.TestCase):
    def test_metadata_does_not_probe_or_compile(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(build,'detect',side_effect=AssertionError('probe in metadata')), patch.object(build.subprocess,'run',side_effect=AssertionError('compile in metadata')):
            name=build.prepare_metadata_for_build_wheel(temp)
            meta_bytes=(Path(temp)/name/'METADATA').read_bytes()
            self.assertIn('Version: '+build.VERSION,meta_bytes.decode('utf-8'))
            commands=(Path(temp)/name/'entry_points.txt').read_text(encoding='utf-8')
            self.assertIn('brainchop-mindmap-fast =',commands)
            self.assertIn('brainchop-mindmap-omni =',commands)
            self.assertNotIn('brainchop-16chan18cls =',commands)
            self.assertEqual(build.get_requires_for_build_wheel(),[])
            self.assertEqual(build.get_requires_for_build_sdist(),[])

    def test_macos_tag_does_not_claim_older_deployment_targets(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('MACOSX_DEPLOYMENT_TARGET', None)
            os.environ.pop('BRAINCHOP_PLATFORM_TAG', None)
            for version,machine,expected in [('10.15.7','x86_64','macosx_10_15_x86_64'),('26.1','arm64','macosx_26_0_arm64')]:
                with patch.object(build.platform,'system',return_value='Darwin'), patch.object(build.platform,'mac_ver',return_value=(version,(),machine)), patch.object(build.platform,'machine',return_value=machine):
                    self.assertEqual(build.platform_tag(),expected)
        with patch.dict(os.environ, {'MACOSX_DEPLOYMENT_TARGET': '13.2'}, clear=False):
            os.environ.pop('BRAINCHOP_PLATFORM_TAG', None)
            with patch.object(build.platform,'system',return_value='Darwin'), patch.object(build.platform,'machine',return_value='arm64'):
                self.assertEqual(build.platform_tag(),'macosx_14_0_arm64')
        with patch.dict(os.environ, {'MACOSX_DEPLOYMENT_TARGET': '14.0'}, clear=False):
            os.environ.pop('BRAINCHOP_PLATFORM_TAG', None)
            with patch.object(build.platform,'system',return_value='Darwin'), patch.object(build.platform,'machine',return_value='arm64'):
                self.assertEqual(build.platform_tag(),'macosx_14_0_arm64')
        with patch.dict(os.environ, {'BRAINCHOP_PLATFORM_TAG': 'manylinux_2_28_x86_64'}, clear=False):
            self.assertEqual(build.platform_tag(),'manylinux_2_28_x86_64')

    def test_source_capsule_is_reproducible_after_extraction(self):
        from brainchop_cli.adaptive import unpack_source
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'source.tar.gz'; original=build.source_tarball(); path.write_bytes(original)
            source=unpack_source(path,Path(temp)/'extract')
            with patch.object(build,'ROOT',source): self.assertEqual(build.source_tarball(),original)

    def test_explicit_targets_and_frontend_configuration(self):
        with patch.dict(os.environ,{},clear=True):
            self.assertEqual(build.build_policy({'unrelated-setting':'value'})['target'],'auto')
            policy=build.build_policy({'brainchop.target':['sm75,sm89']})
            self.assertEqual(policy['cpu'],'baseline')
            selected=build.target_hardware(dict(system='Linux',machine='x86_64',devices=[],cpu='baseline'),policy,dict(cuda_architectures=[75,89],cuda_minimum_driver=12080))
            self.assertEqual([d['arch'] for d in selected['devices']],[75,89])
            for settings in ({'brainchop.target':'sm999'},{'brainchop.unknown':'yes'},{'brainchop.cpu':'avx512'}):
                with self.assertRaises(ValueError): build.build_policy(settings)

    def test_baseline_cpu_is_supported(self):
        def fake_run(command,**kwargs): return subprocess.CompletedProcess(command,0,'cpu\tavx2-fma\t0\n','')
        with tempfile.TemporaryDirectory() as tmp, patch.object(hardware.platform,'system',return_value='Linux'), patch.object(hardware.platform,'machine',return_value='x86_64'), patch.object(hardware.shutil,'which',return_value='/usr/bin/cc'), patch.object(hardware.subprocess,'run',side_effect=fake_run):
            self.assertEqual(hardware.detect(build.ROOT,Path(tmp),dict(PATH='/usr/bin'))['cpu'],'baseline')

    def test_sdist_is_self_contained_source_inventory(self):
        with tempfile.TemporaryDirectory() as temp:
            name=build.build_sdist(temp)
            with tarfile.open(Path(temp)/name) as tar:
                names=tar.getnames()
                self.assertTrue(any(n.endswith('/src/models/mindmap_omni/model.bin') for n in names))
                self.assertTrue(any(n.endswith('/src/models/mindsnap/model.bin') for n in names))
                self.assertFalse(any('.git/' in n or n.endswith(('.safetensors','.o','.exe','.pyc')) or '/test/' in n for n in names))
                self.assertEqual(sum(n.endswith('/model.bin') for n in names),5)

    def test_cuda_pruning_preserves_only_selected_native_images(self):
        for path in (build.ROOT/'src/cuda_modules').glob('*.fatbin'):
            blob=path.read_bytes(); original=hardware.fatbin_images(blob)
            self.assertEqual({arch for kind,arch,_ in original if kind==2},{60,70,75,80,86,89,90})
            for wanted,expected in [([61],[60]),([75],[75]),([87],[86]),([75,89],[75,89]),([100],[])]:
                selected,arches=hardware.select_cuda(blob,wanted)
                self.assertEqual(arches,expected)
                if not selected: continue
                images=hardware.fatbin_images(selected)
                self.assertEqual([(kind,arch) for kind,arch,_ in images],[(2,a) for a in expected])
                for image in images: self.assertIn(image,original)
            for bad in (blob[:-1],b'bad magic'+blob[9:]):
                with self.assertRaises(ValueError): hardware.fatbin_images(bad)

    def test_selection_is_target_only_and_reports_unsupported_devices(self):
        for devices,expected in [([],[]),([{'backend':'hip','arch':'gfx1151','index':0}],['hip_artifacts']),([{'backend':'cuda','arch':89,'driver':12080,'index':0}],['cuda_modules']),([{'backend':'hip','arch':'gfx1100','index':0}],[])]:
            with tempfile.TemporaryDirectory() as temp:
                src=Path(temp)/'src'
                shutil.copytree(build.ROOT/'src',src,ignore=shutil.ignore_patterns('build','brainchop-*','*.o'))
                options,artifacts,diagnostics=hardware.prune(src,dict(devices=devices))
                self.assertEqual(sorted({p.split('/')[0] for p in artifacts}),expected)
                self.assertFalse(list(src.glob('metal_*_generated.h')))
                self.assertIn('OPENCL=0',options)
                if not expected: self.assertTrue(diagnostics)
                for record in artifacts.values(): self.assertEqual(len(record['sha256']),64)

    def test_hip_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            src=Path(temp)/'src'; shutil.copytree(build.ROOT/'src',src,ignore=shutil.ignore_patterns('build','brainchop-*','*.o'))
            (src/'hip_artifacts/mindsnap/gfx1151/kernels.bin').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'HIP bundle'):
                hardware.prune(src,dict(devices=[dict(backend='hip',arch='gfx1151')]))

    def test_failed_vendor_probe_does_not_hide_cuda(self):
        def fake_run(command,**kwargs):
            if '--hip' in command: raise subprocess.TimeoutExpired(command,30)
            if '--opencl' in command: return subprocess.CompletedProcess(command,-11,'','ICD crashed')
            output='cpu\tavx2-fma\t1\n'
            if '--cuda' in command: output+='cuda\t0\t75\t13000\n'
            return subprocess.CompletedProcess(command,0,output,'')
        with tempfile.TemporaryDirectory() as tmp, patch.object(hardware.platform,'system',return_value='Linux'), patch.object(hardware.platform,'machine',return_value='x86_64'), patch.object(hardware.shutil,'which',return_value='/usr/bin/cc'), patch.object(hardware.subprocess,'run',side_effect=fake_run):
            found=hardware.detect(build.ROOT,Path(tmp),dict(PATH='/usr/bin'))
        self.assertEqual(found['devices'],[dict(backend='cuda',index=0,arch=75,driver=13000)])
        self.assertEqual(len(found['diagnostics']),2)

    def test_make_environment_cannot_inherit_parent_build_flags(self):
        with patch.dict(os.environ,{'MAKEFLAGS':'-e CPUFLAGS=-march=native','CFLAGS':'-march=native'}):
            env=hardware.tool_environment()
            self.assertNotIn('MAKEFLAGS',env); self.assertNotIn('CFLAGS',env)

if __name__=='__main__': unittest.main()
