#!/usr/bin/env python3
"""Run extracted default-build executables on a host without an NVIDIA driver."""
import argparse
import gzip
import os
from pathlib import Path
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('directory', type=Path)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
suffix = '.exe' if os.name == 'nt' else ''
env = dict(os.environ, BC_NUM_THREADS='2')
# Windows must also work outside MSYS2, without its DLL search paths.
if os.name == 'nt':
    env['PATH'] = str(Path(os.environ['SystemRoot']) / 'System32')
with tempfile.TemporaryDirectory(prefix='bc-no-driver-') as tmp:
    for model in ('mindgrab', 'mindmap-fast', 'mindmap', 'mindmap-omni', 'mindsnap'):
        binary = (a.directory / ('brainchop-' + model + suffix)).resolve()
        def run(args):
            return subprocess.run([str(binary), *map(str, args)], cwd=tmp, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
        listing = run(['-a', '-1'])
        assert 'cuda: no adapters found' in listing.stdout, listing.stdout
        for mode in ('auto', 'cpu'):
            out = Path(tmp) / (mode + '.nii.gz')
            proc = run(['-backend', mode, '-o', out, root / 'test/t1_2mm.nii.gz'])
            assert proc.returncode == 0 and 'using cpu backend' in proc.stdout, proc.stdout
        assert gzip.decompress((Path(tmp) / 'auto.nii.gz').read_bytes()) == gzip.decompress((Path(tmp) / 'cpu.nii.gz').read_bytes())
        rejected = Path(tmp) / 'rejected.nii.gz'
        proc = run(['-backend', 'cuda', '-o', rejected, root / 'test/t1_2mm.nii.gz'])
        assert proc.returncode != 0 and not rejected.exists(), proc.stdout
        print(f'PASS: {model}: auto/CPU exact; forced unavailable CUDA fails without output', flush=True)
