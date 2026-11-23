#!/usr/bin/env python3
"""Installed CUDA survives OpenCL loader loss; driver loss uses CPU or refuses."""
import argparse
import gzip
import json
import os
from pathlib import Path
import subprocess
import sysconfig
import brainchop_cli
import nii_diff
p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=False)
scripts=Path(sysconfig.get_path('scripts')); fixtures=Path(__file__).resolve().parent
manifest=json.loads((Path(brainchop_cli.__file__).parent/'build.json').read_text())
assert 'CUDA=driver' in manifest['options']
source=a.output/'hidden.c'; source.write_text('int hidden_driver(void){return 0;}\n')
for name in ('no-opencl','no-drivers'):
    directory=a.output/name; directory.mkdir()
    for lib in ('libOpenCL.so.1',) if name=='no-opencl' else ('libOpenCL.so.1','libcuda.so.1'):
        subprocess.run(['cc','-shared','-fPIC',str(source),'-o',str(directory/lib)],check=True)
def run(label,model,extra=(),hidden=None):
    out=a.output/(label+'.nii.gz')
    env=dict(os.environ,BC_NUM_THREADS='4',XDG_CACHE_HOME=str(a.output/'cache'))
    if hidden: env['LD_LIBRARY_PATH']=str(a.output/hidden)+os.pathsep+env.get('LD_LIBRARY_PATH','')
    proc=subprocess.run([str(scripts/('brainchop-'+('mindmap-fast' if model=='16chan18cls' else model))),*extra,'-o',str(out),str(fixtures/'t1_crop.nii.gz')],env=env,capture_output=True,text=True,timeout=600)
    log=proc.stdout+proc.stderr; (a.output/(label+'.log')).write_text(log)
    return proc.returncode,log,out
for model in ('mindgrab','16chan18cls','mindmap','mindsnap'):
    rc,log,base=run(model+'-baseline',model)
    assert rc==0 and 'using cuda backend' in log and 'falling back' not in log,log
    rc,log,out=run(model+'-no-opencl',model,hidden='no-opencl')
    assert rc==0 and 'using cuda backend' in log and 'falling back' not in log,log
    assert gzip.decompress(base.read_bytes())==gzip.decompress(out.read_bytes())
    rc,log,out=run(model+'-forced-unavailable',model,('-backend','cuda'),hidden='no-drivers')
    assert rc!=0 and not out.exists(),log
rc,log,out=run('cpu-fallback','mindgrab',('--no-turbo',),hidden='no-drivers')
assert rc==0 and 'using cpu backend' in log,log
actual=gzip.decompress(out.read_bytes())
assert nii_diff.descrip(actual).startswith('brainchop '), nii_diff.descrip(actual)
assert nii_diff.equal_ignoring_descrip(actual,gzip.decompress((fixtures/'bet.nii.gz').read_bytes()))
(a.output/'results.json').write_text(json.dumps(dict(opencl_built='OPENCL=1' in manifest['options'],cuda_survives_opencl_loss=True,unavailable_cuda_refuses_all_models=True,cpu_fallback_exact=True),indent=2)+'\n')
print('PASS: CUDA survives OpenCL loss; missing CUDA refuses all four models; automatic CPU fallback exact')
