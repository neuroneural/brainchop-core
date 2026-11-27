#!/usr/bin/env python3
"""Compare isolated legacy/Driver API builds; needs a CUDA GPU, no Python packages.

Each executable directory must contain all three brainchop model executables. Match
their host compiler/flags and native CUDA architecture profile before timing.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time


EXES = {'mindgrab': 'brainchop-mindgrab', 'model16': 'brainchop-mindmap-fast', 'mindmap': 'brainchop-mindmap',
        'mindsnap': 'brainchop-mindsnap'}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--legacy', type=Path, required=True)
    p.add_argument('--driver', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='new, isolated evidence directory')
    p.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[1] / 'test')
    p.add_argument('--pairs', type=int, default=0, help='alternating warm pairs per input/space cell')
    p.add_argument('--adapter', type=int, default=0)
    a = p.parse_args()
    if a.pairs < 0 or a.adapter < 0:
        p.error('pairs and adapter must be nonnegative')
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, BC_NUM_THREADS='8', BRAINCHOPC_TUNE='0', BRAINCHOPC_CUDA_WMMA='1')
    env.pop('BRAINCHOPC_CUDA_PROFILE', None)
    env['XDG_CACHE_HOME'] = str(a.output.resolve() / 'cache')
    records = []

    def run(kind, model, fixture, conform, tag, overrides=None, tune=False):
        binary = (getattr(a, kind) / EXES[model]).resolve()
        target = a.output / (tag + '.nii.gz')
        command = [str(binary), '-backend', 'cuda', '-a', str(a.adapter), '-o', str(target)]
        if conform:
            command.append('--save-conform')
        if tune:
            command.append('--tune')
        command.append(str(a.fixtures / (fixture + '.nii.gz')))
        started = time.perf_counter()
        proc = subprocess.run(command, env=dict(env, **(overrides or {})), stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=600)
        seconds = time.perf_counter() - started
        log = proc.stdout.decode(errors='replace')
        (a.output / (tag + '.log')).write_text(log)
        if proc.returncode or 'using cuda backend' not in log:
            raise RuntimeError(f'{tag}: CUDA execution failed (return code {proc.returncode}); see log')
        if kind == 'driver' and 'CUDA Driver API,' not in log:
            raise RuntimeError(f'{tag}: wrong backend implementation')
        data = gzip.decompress(target.read_bytes())
        digest = hashlib.sha256(data).hexdigest()
        records.append(dict(tag=tag, seconds=seconds, sha256=digest))
        target.unlink()  # hashes and logs retain evidence without gigabytes of duplicate labels
        return digest, seconds, log

    binary_hashes = {kind: {model: hashlib.sha256((getattr(a, kind) / name).read_bytes()).hexdigest()
        for model, name in EXES.items()}
        for kind in ('legacy', 'driver')}
    summary = []
    try:
        for model in EXES:
            for fixture in ('t1_crop', 't1_2mm'):
                for conform in ((False,) if model == 'mindgrab' else (False, True)):
                    cell = f'{model}-{fixture}-' + ('conformed' if conform else 'native')
                    old, _, _ = run('legacy', model, fixture, conform, cell + '-legacy-warmup')
                    new, _, _ = run('driver', model, fixture, conform, cell + '-driver-warmup')
                    assert old == new, f'{cell}: fixed WMMA output differs'
                    ratios = []
                    for pair in range(a.pairs):
                        times = {}
                        for kind in (('legacy', 'driver') if pair % 2 == 0 else ('driver', 'legacy')):
                            digest, times[kind], _ = run(kind, model, fixture, conform, f'{cell}-{pair}-{kind}')
                            assert digest == old, f'{cell}: unstable output'
                        ratios.append(times['driver'] / times['legacy'])
                    row = dict(cell=cell, sha256=old, driver_over_legacy=ratios,
                               median_ratio=statistics.median(ratios) if ratios else None)
                    summary.append(row)
                    print(json.dumps(row), flush=True)

            scalar = {}
            for kind in ('legacy', 'driver'):
                scalar[kind], _, _ = run(kind, model, 't1_crop', False, f'{model}-{kind}-scalar',
                                        {'BRAINCHOPC_CUDA_WMMA': '0'})
            assert scalar['legacy'] == scalar['driver'], f'{model}: fixed scalar output differs'
            for kind in ('legacy', 'driver'):
                for wmma in ('0', '1'):
                    tag = f'{model}-{kind}-tune-wmma{wmma}'
                    tuned, _, log = run(kind, model, 't1_crop', False, tag,
                                       {'BRAINCHOPC_CUDA_WMMA': wmma}, tune=True)
                    assert 'brainchop-core: chose ' in log and 'tuning failed' not in log, tag
                    cached, _, log = run(kind, model, 't1_crop', False, tag + '-cached',
                        {'BRAINCHOPC_TUNE': '2', 'BRAINCHOPC_CUDA_WMMA': wmma})
                    assert 'tuning the ' not in log and cached == tuned, tag + ': cache reuse failed'
                    expected = scalar[kind] if wmma == '0' else next(
                        r['sha256'] for r in summary if r['cell'] == f'{model}-t1_crop-native')
                    assert tuned == expected, tag + ': tuned output differs from fixed output'
        print('PASS: all models, both inputs, output spaces, scalar/WMMA, forced/cached tuning', flush=True)
    finally:
        (a.output / 'results.json').write_text(json.dumps(dict(binary_hashes=binary_hashes,
            environment={k: env.get(k) for k in ('BC_NUM_THREADS', 'CUDA_FORCE_PTX_JIT', 'CUDA_DISABLE_PTX_JIT')},
            cells=summary, runs=records), indent=2) + '\n')


if __name__ == '__main__':
    main()
