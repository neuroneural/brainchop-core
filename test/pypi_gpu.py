#!/usr/bin/env python3
"""Installed Metal/CUDA numerical gate: forced GPU with dispatch evidence and auto equality."""
import argparse
import json
import os
import platform
from pathlib import Path
import re
import subprocess
import sys
import sysconfig
import tempfile
from hip_native_check import compare

p=argparse.ArgumentParser(); p.add_argument('--backend',choices=('metal','cuda'),required=True); p.add_argument('--output',type=Path,required=True)
p.add_argument('--adapter',type=int,help='CUDA-visible ordinal; omit to test automatic selection'); p.add_argument('--scalar',action='store_true')
a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False)
scripts=Path(sysconfig.get_path('scripts')); fixtures=Path(__file__).resolve().parent
rows=[]
env=dict(os.environ,BC_TIME_GPU='1',BC_NUM_THREADS='4',XDG_CACHE_HOME=str(a.output.resolve()/'cache'))
for key in ('BRAINCHOPC_TUNE','BRAINCHOPC_CUDA_WMMA','BRAINCHOPC_CUDA_LAYERS','CUDA_FORCE_PTX_JIT'):
    env.pop(key,None)
if a.backend=='cuda':
    env['BRAINCHOPC_CUDA_PROFILE']='1'
    env['CUDA_DISABLE_PTX_JIT']='1'
    if a.scalar: env['BRAINCHOPC_CUDA_WMMA']='0'
    assert not (a.output/'cache').exists()
native16='labels16_native_majority_wasm.nii.gz' if platform.machine().lower() in ('x86_64','amd64') else 'labels16_native_majority.nii.gz'

for model,refs,budgets in [('mindgrab',('bet.nii.gz',None),(64,64)),('16chan18cls',('labels16.nii.gz',native16),(1024,256)),('mindmap',('labels24.nii.gz','labels24_native_majority.nii.gz'),(1024,256)),('mindsnap',('labels104.nii.gz','labels104_native_majority.nii.gz'),(2048,512))]:
    for index,fixture in enumerate(('t1_crop.nii.gz','t1_2mm.nii.gz')):
        label=model+'-'+str(index); output=a.output/(label+'.nii.gz')
        extra=['--legacy-cleanup'] if model=='mindmap' else []
        if a.adapter is not None: extra+=['-a',str(a.adapter)]
        # The budgets are fp16; Metal's default 8-bit path is gated after the loop body.
        if a.backend=='metal': extra+=['--no-turbo']
        if index==0 and model!='mindgrab': extra+=['--save-conform']
        cmd=[str(scripts/('brainchop-'+('mindmap-fast' if model=='16chan18cls' else model))),*extra,'-o',str(output),str(fixtures/fixture)]
        proc=subprocess.run(cmd+['-backend',a.backend],capture_output=True,text=True,env=env,timeout=600)
        log=proc.stdout+proc.stderr; (a.output/(label+'.log')).write_text(log)
        assert proc.returncode==0 and f'using {a.backend} backend' in log and 'falling back' not in log, log
        # Metal has no BC_TIME_GPU counter; forced backend plus no fallback is
        # its existing gate. CUDA additionally reports synchronized GPU timing.
        if a.backend=='cuda':
            timing=re.search(r'brainchopc:\s+total\s+([0-9.]+) ms',log)
            assert 'CUDA stages (' in log and timing and float(timing[1])>0,log
            if index==0: assert 'tuning the ' in log and 'caching the result' in log,log
            else: assert 'tuning the ' not in log,log
            assert 'tuning failed' not in log and 'could not save' not in log,log
        reference=fixtures/refs[index] if refs[index] else a.output/(label+'-cpu.nii.gz')
        if refs[index] is None:
            subprocess.run([str(scripts/('brainchop-'+('mindmap-fast' if model=='16chan18cls' else model))),'-cpu','4','--no-turbo','-o',str(reference),str(fixtures/fixture)],capture_output=True,env=env,check=True)
        different=compare(reference,output,budgets[index])
        auto=a.output/(label+'-auto.nii.gz')
        proc=subprocess.run([str(scripts/('brainchop-'+('mindmap-fast' if model=='16chan18cls' else model))),*extra,'-o',str(auto),str(fixtures/fixture)],capture_output=True,text=True,env=env,timeout=600)
        (a.output/(label+'-auto.log')).write_text(proc.stdout+proc.stderr)
        assert proc.returncode==0 and f'using {a.backend} backend' in proc.stderr and 'falling back' not in proc.stderr,proc.stderr
        if a.backend=='cuda': assert 'tuning the ' not in proc.stderr,proc.stderr
        assert compare(output,auto,0)==0
        row=dict(model=model,fixture=fixture,different_voxels=different,limit=budgets[index],auto_equal=True,adapter=a.adapter,scalar=a.scalar,cold_calibrated=a.backend=='cuda' and index==0,warm_no_retune=a.backend=='cuda')
        if a.backend=='metal':
            turbo=a.output/(label+'-turbo.nii.gz')
            exe=str(scripts/('brainchop-'+('mindmap-fast' if model=='16chan18cls' else model)))
            fast=[e for e in extra if e!='--no-turbo']
            gate=['0.995','99.5'] if model=='mindgrab' else ['0.988','98.9']
            ref,got=reference,turbo
            if model=='mindgrab':
                # Its image keeps the input datatype (int16 for t1_2mm); gate the uint8 masks.
                ref,got=a.output/(label+'-cpu-mask.nii.gz'),a.output/(label+'-turbo-mask.nii.gz')
                subprocess.run([exe,'-cpu','4','--no-turbo','--mask',str(ref),'-o',str(a.output/(label+'-cpu-img.nii.gz')),str(fixtures/fixture)],capture_output=True,env=env,check=True,timeout=600)
                fast+=['--mask',str(got)]
            subprocess.run([exe,*fast,'-backend','metal','-o',str(turbo),str(fixtures/fixture)],capture_output=True,env=env,check=True,timeout=600)
            subprocess.run([sys.executable,str(fixtures/'nii_dice.py'),str(ref),str(got),*gate],check=True)
            row['turbo_gate']=gate
        rows.append(row); print(json.dumps(row),flush=True)
        (a.output/'results.json').write_text(json.dumps(rows,indent=2)+'\n')

# Default unified command selects mindmap without a backend flag.
if not a.scalar and a.adapter is None:
    output=a.output/'compat-default.nii.gz'
    direct=a.output/'compat-direct.nii.gz'
    proc=subprocess.run([str(scripts/'brainchop'),str(fixtures/'t1_crop.nii.gz'),'-o',str(output)],env=env,capture_output=True,text=True,timeout=600)
    (a.output/'compat-default.log').write_text(proc.stdout+proc.stderr)
    assert proc.returncode==0 and f'using {a.backend} backend' in proc.stderr,proc.stderr
    subprocess.run([str(scripts/'brainchop-mindmap'),'-o',str(direct),str(fixtures/'t1_crop.nii.gz')],env=env,capture_output=True,text=True,timeout=600,check=True)
    assert compare(direct,output,0)==0
