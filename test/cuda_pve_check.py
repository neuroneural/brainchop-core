#!/usr/bin/env python3
"""Real-device PVE integration gate, using only the Python standard library.

Compare CPU/CUDA fractions in both output spaces on both tracked inputs. The
engineering gate is RMSE <= two uint8 fraction quanta over the nonzero union,
with matching NIfTI headers. Report maximum errors too. This is numerical
regression coverage, not a clinical accuracy assessment. An optional legacy
binary directory must produce byte-identical CUDA results.
"""
import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

import nii_diff


def compare(reference, candidate):
    a, b = nii_diff.read(str(reference)), nii_diff.read(str(candidate))
    if not nii_diff.header_matches(a, b) or len(a) != len(b):
        raise AssertionError('PVE NIfTI geometry, datatype or scaling differs')
    if struct.unpack_from('<h', a, 70)[0] != 2:
        raise AssertionError('PVE output must be uint8')
    slope = struct.unpack_from('<f', a, 112)[0]
    if abs(slope - 1 / 255) > 1e-9:
        raise AssertionError('PVE scaling must represent fractions in [0,1]')
    histogram = [0] * 256
    active = 0
    for x, y in zip(a[352:], b[352:]):
        histogram[abs(x - y)] += 1
        active += bool(x or y)
    rmse = math.sqrt(sum(n * d * d for d, n in enumerate(histogram)) / max(1, active)) / 255
    result = dict(active_voxels=active, differing_voxels=sum(histogram[1:]),
                  max_absolute=max(d for d, n in enumerate(histogram) if n) / 255,
                  rmse_nonzero_union=rmse, rmse_limit=2 / 255)
    if not active or rmse > 2 / 255:
        raise AssertionError(f'PVE numerical gate failed: {result}')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--driver', type=Path, required=True)
    p.add_argument('--legacy', type=Path)
    p.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parent)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--adapter', type=int, default=0)
    variant = p.add_mutually_exclusive_group()
    variant.add_argument('--schedule', help='validate an offline CUDA layer schedule')
    variant.add_argument('--scalar', action='store_true', help='validate scalar CUDA inference')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, BRAINCHOPC_TUNE='0', BC_NUM_THREADS='8',
               XDG_CACHE_HOME=str(a.output.resolve() / 'cache'))
    for key in ('BRAINCHOPC_CUDA_LAYERS', 'BRAINCHOPC_CUDA_WMMA', 'BRAINCHOPC_PVE_DUMP'):
        env.pop(key, None)
    if a.schedule:
        env['BRAINCHOPC_CUDA_LAYERS'] = a.schedule
    if a.scalar:
        env['BRAINCHOPC_CUDA_WMMA'] = '0'
    rows = []
    try:
        for model in ('mindmap',):
            for fixture in ('t1_crop', 't1_2mm'):
                for conform in (False, True):
                    cell = f'{model}-{fixture}-' + ('conformed' if conform else 'native')
                    outputs = {}
                    for kind in ('cpu', 'driver', 'legacy') if a.legacy else ('cpu', 'driver'):
                        directory = a.legacy if kind == 'legacy' else a.driver
                        base = a.output / (cell + '-' + kind)
                        command = [str((directory / ('brainchop-' + model)).resolve()), '--pve',
                                   '-o', str(base) + '.nii.gz']
                        command += ['-cpu', '8'] if kind == 'cpu' else ['-backend', 'cuda', '-a', str(a.adapter)]
                        if conform:
                            command.append('--save-conform')
                        command.append(str(a.fixtures / (fixture + '.nii.gz')))
                        started = time.perf_counter()
                        proc = subprocess.run(command, env=env, capture_output=True, text=True, timeout=900)
                        log = proc.stdout + proc.stderr
                        base.with_suffix('.log').write_text(log)
                        if proc.returncode or ('using cpu backend' if kind == 'cpu' else 'using cuda backend') not in log:
                            raise AssertionError(f'{cell}-{kind}: execution failed; see log')
                        outputs[kind] = {t: Path(str(base) + '_' + t + '.nii.gz') for t in ('gm', 'wm', 'csf')}
                        rows.append(dict(cell=cell, kind=kind, schedule=a.schedule, scalar=a.scalar,
                                         seconds=time.perf_counter() - started,
                                         sha256={t: hashlib.sha256(gzip.decompress(f.read_bytes())).hexdigest()
                                                 for t, f in outputs[kind].items()}))
                    numerical = {t: compare(outputs['cpu'][t], outputs['driver'][t]) for t in ('gm', 'wm', 'csf')}
                    if a.legacy:
                        assert rows[-1]['sha256'] == rows[-2]['sha256'], cell + ': legacy/driver PVE differs'
                    rows.append(dict(cell=cell, numerical=numerical))
                    print(json.dumps(rows[-1]), flush=True)
                    for files in outputs.values():
                        for path in files.values():
                            path.unlink()
        print('PASS: MindMap PVE CPU/CUDA, both inputs and spaces; optional legacy equality', flush=True)
    finally:
        (a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
