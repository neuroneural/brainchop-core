#!/usr/bin/env python3
"""GPU-free compiled CUDA package check; detection is explicitly simulated.

For CI runners without NVIDIA hardware. It checks the pruned SDK-free driver
build, installed image contents, unavailable-driver errors and CPU fallback.
It does not validate CUDA execution or claim this runner has an NVIDIA GPU.
"""
import argparse
import gzip
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import nii_diff

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'packaging'))
import brainchop_build as build

p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=False)
selection=dict(system='Windows' if os.name=='nt' else 'Linux',machine='x86_64',cpu='baseline',opencl_build=True,devices=[dict(backend='cuda',index=0,arch=89,driver=12080)])
# Still compile the real baseline probe; only the reported GPU selection is simulated.
real_detect=build.detect
def simulated(root,work,env):
    actual=real_detect(root,work,env)
    assert not any(d['backend']=='cuda' for d in actual['devices']), 'this is a GPU-free test; use the real CUDA gate on a GPU host'
    return dict(selection,cpu=actual['cpu'])
with patch.object(build,'detect',side_effect=simulated):
    wheel=build.build_wheel(a.output)
venv=a.output/'venv'; subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
scripts=venv/('Scripts' if os.name=='nt' else 'bin'); python=scripts/('python.exe' if os.name=='nt' else 'python')
subprocess.run([str(python),'-m','pip','install',str(a.output/wheel)],check=True)
subprocess.run([str(python),str(ROOT/'test/pypi_audit.py'),'--output',str(a.output/'audit.json')],check=True)
suffix='.exe' if os.name=='nt' else ''
for model in build.EXES:
    binary=scripts/('brainchop-'+('mindmap-fast' if model=='16chan18cls' else model)+suffix)
    listing=subprocess.run([str(binary),'-a','-1'],capture_output=True,text=True,check=True)
    assert 'cuda: no adapters found' in listing.stdout+listing.stderr,listing
    output=a.output/(model+'-refused.nii.gz')
    forced=subprocess.run([str(binary),'-backend','cuda','-o',str(output),str(ROOT/'test/t1_crop.nii.gz')],capture_output=True,text=True)
    assert forced.returncode!=0 and not output.exists(),forced
    if model in ('mindgrab','16chan18cls'):
        missing=subprocess.run([str(binary),'-backend','opencl','-o',str(output),str(ROOT/'test/t1_crop.nii.gz')],capture_output=True,text=True)
        assert missing.returncode!=0 and not output.exists(),missing
output=a.output/'auto-cpu.nii.gz'
result=subprocess.run([str(scripts/('brainchop-mindgrab'+suffix)),'--no-turbo','-o',str(output),str(ROOT/'test/t1_crop.nii.gz')],capture_output=True,text=True,env=dict(os.environ,BC_NUM_THREADS='4'),check=True)
assert 'using cpu backend' in result.stderr,result.stderr
actual=gzip.decompress(output.read_bytes())
assert nii_diff.descrip(actual).startswith('brainchop '), nii_diff.descrip(actual)
assert nii_diff.equal_ignoring_descrip(actual,gzip.decompress((ROOT/'test/bet.nii.gz').read_bytes()))
print('PASS: simulated sm89/OpenCL selection compiled and installed without SDK headers; only sm89 images; missing CUDA refuses; CPU fallback bit-exact')
