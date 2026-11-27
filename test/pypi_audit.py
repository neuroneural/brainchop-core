#!/usr/bin/env python3
"""Inspect an installed package against the source artifact inventory (no ML imports)."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
import brainchop_cli
sys.path.insert(0,str(ROOT/'packaging'))
from brainchop_hardware import fatbin_images

p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--runtime',type=Path); a=p.parse_args()
package=a.runtime or Path(brainchop_cli.__file__).parent
manifest=json.loads((package/'build.json').read_text()); rows=[]
assert not importlib.metadata.requires('brainchop')
for entry in importlib.metadata.files('brainchop'):
    path=str(entry)
    assert not path.endswith(('.fatbin','.safetensors','.o')) and 'kernels.bin' not in path, path
for binary in (package/'bin').rglob('*'):
    if not binary.is_file(): continue
    data=binary.read_bytes(); selected=[]
    baseline=binary.name.startswith('hardware-probe')
    model=binary.name.removeprefix('brainchop-').removesuffix('.exe')
    model='model16chan18cls' if model in ('16chan18cls','mindmap-fast') else ('mindmap' if model=='mindmap-omni' else model)
    for path in (ROOT/'src/cuda_modules').glob('*.fatbin'):
        for kind,arch,image in fatbin_images(path.read_bytes()):
            present=image in data
            record=manifest['artifacts'].get(path.relative_to(ROOT/'src').as_posix(),{})
            expected=not baseline and path.stem==model and kind==2 and arch in record.get('architectures',[])
            assert present==expected,(binary.name,path.name,kind,arch,present,expected)
            if present: selected.append(f'cuda:sm{arch}')
    for path in (ROOT/'src/hip_artifacts').glob('*/*/kernels.bin'):
        present=path.read_bytes() in data
        expected=not baseline and path.parent.parent.name==model and path.relative_to(ROOT/'src').as_posix() in manifest['artifacts']
        assert present==expected,(binary.name,str(path),present,expected)
        if present: selected.append('hip:'+path.parent.name)
    if platform.system()=='Darwin':
        links=subprocess.check_output(['otool','-L',str(binary)],text=True)
        for line in links.splitlines()[1:]:
            assert line.strip().startswith(('/usr/lib/','/System/Library/')),line
    elif platform.system()=='Linux':
        links=subprocess.check_output(['objdump','-p',str(binary)],text=True)
        deps=re.findall(r'^\s*NEEDED\s+(\S+)',links,re.M)
        for dep in deps:
            assert re.fullmatch(r'lib(?:c|m|dl|pthread|rt|z)\.so(?:\.[0-9]+)*|ld-linux[^/]*\.so\.[0-9]+',dep),dep
        assert not re.search(r'^\s*(RPATH|RUNPATH)\s+',links,re.M)
        links='\n'.join(deps)
    else:
        from brainchop_build import audit_linkage
        from brainchop_hardware import tool_environment
        links='\n'.join(audit_linkage(binary,tool_environment()))
    rows.append(dict(binary=str(binary.relative_to(package)),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),artifacts=selected,linkage=links))
a.output.write_text(json.dumps(rows,indent=2)+'\n')
print('PASS: installed artifact images and runtime linkage match selection; no unrelated GPU artifacts or Python dependencies')
