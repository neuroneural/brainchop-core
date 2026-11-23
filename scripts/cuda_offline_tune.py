#!/usr/bin/env python3
"""Offline MindMap search: compile register-cap candidates, measure every layer,
then validate mixed schedules and report warm whole-model timing and GPU memory.

Uses isolated copies and the legacy nvcc adapter for compiler experiments. The
selected schedule must be revalidated with the release Driver API artifact.
No production source, artifact, cache or default is changed by this script.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'test'))
import nii_diff


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def label_difference(reference, candidate):
    a, b = nii_diff.read(str(reference)), nii_diff.read(str(candidate))
    if not nii_diff.header_matches(a, b) or len(a) != len(b):
        raise ValueError('output geometry/datatype differs from CPU reference')
    return sum(x != y for x, y in zip(nii_diff.voxels(a, str(reference)),
                                     nii_diff.voxels(b, str(candidate))))


def sample_memory(pid, stop, samples):
    while not stop.is_set():
        try:
            proc = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                   '--format=csv,noheader,nounits'], capture_output=True,
                                  text=True, timeout=3)
            for line in proc.stdout.splitlines():
                fields = line.split(',')
                if len(fields) == 2 and fields[0].strip() == str(pid):
                    samples.append(int(fields[1].strip()))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        stop.wait(0.1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, help='existing MindMap binary; skips compilation')
    p.add_argument('--source', type=Path, default=ROOT)
    p.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
    p.add_argument('--arch', help='explicit target for compilation, e.g. sm_86')
    p.add_argument('--register-caps', default='0,64,96,128', help='0 keeps nvcc defaults')
    p.add_argument('--output', type=Path, required=True, help='new evidence/build directory')
    p.add_argument('--input', type=Path, action='append', help='repeat for independent inputs')
    p.add_argument('--adapter', type=int, default=0)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--memory-budget-mib', type=int, help='optional sampled process-VRAM ceiling')
    p.add_argument('--max-label-differences', type=int, required=True,
                   help='explicit accuracy gate, never inferred from candidate results')
    p.add_argument('--candidates', help='comma-separated sVxT/wW tokens; default all portable shapes')
    a = p.parse_args()
    if a.repeats < 2 or a.adapter < 0 or a.max_label_differences < 0:
        p.error('need repeats >= 2, adapter >= 0, and a nonnegative accuracy gate')
    if not a.binary and (not a.arch or not re.fullmatch(r'sm_[0-9]+', a.arch)):
        p.error('compilation requires an explicit --arch sm_NN')
    caps = [int(x) for x in a.register_caps.split(',')]
    if any(x < 0 or x > 255 for x in caps):
        p.error('register caps must be between 0 and 255')
    inputs = [x.resolve() for x in (a.input or [ROOT / 'test/t1_crop.nii.gz', ROOT / 'test/t1_2mm.nii.gz'])]
    if not all(x.is_file() for x in inputs):
        p.error('an input file is missing')
    candidates = (a.candidates.split(',') if a.candidates else
                  [f's{v}x{t}' for v in (1, 2, 3, 4, 6, 8) for t in (128, 256, 512)] +
                  [f'w{w}' for w in (2, 4, 8, 16)])
    if not candidates or any(not re.fullmatch(r's[0-9]+x[0-9]+|w[0-9]+', x) for x in candidates):
        p.error('invalid candidate token')
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, BRAINCHOPC_TUNE='0', XDG_CACHE_HOME=str(a.output / 'cache'))
    for key in ('BRAINCHOPC_CUDA_LAYERS', 'BRAINCHOPC_CUDA_PROFILE', 'BRAINCHOPC_CUDA_WMMA',
                'NVCC_PREPEND_FLAGS', 'NVCC_APPEND_FLAGS'):
        env.pop(key, None)
    records, builds, finalists = [], {}, []
    report = {'arguments': {k: str(v) for k, v in vars(a).items()}, 'runs': records,
              'builds': builds, 'finalists': finalists,
              'note': 'Memory is sampled process VRAM, not an exact allocation trace; profiling times are not end-to-end timings.'}
    inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total,power.limit',
                                '--format=csv'], text=True, capture_output=True, check=True)
    report['gpu_inventory'] = inventory.stdout
    report['inputs'] = {str(x): digest(x) for x in inputs}

    def save():
        (a.output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

    def run(binary, fixture, tag, schedule=None, profile=False, cpu=False):
        out = a.output / (tag + '.nii.gz')
        log = a.output / (tag + '.log')
        command = [str(binary), '--save-conform', '--no-turbo', '-o', str(out)]
        command += ['-cpu', '0'] if cpu else ['-backend', 'cuda', '--adapter', str(a.adapter)]
        command.append(str(fixture))
        run_env = dict(env)
        if schedule:
            run_env['BRAINCHOPC_CUDA_LAYERS'] = schedule
        if profile:
            run_env['BRAINCHOPC_CUDA_PROFILE'] = '1'
        samples, stop = [], threading.Event()
        start = time.perf_counter()
        # A pipe lets communicate observe process completion without wait(timeout)'s
        # polling backoff, which can add up to 50 ms to these short CLI timings.
        proc = subprocess.Popen(command, env=run_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        sampler = None
        if not cpu:
            sampler = threading.Thread(target=sample_memory, args=(proc.pid, stop, samples))
            sampler.start()
        try:
            captured, _ = proc.communicate(timeout=900)
            seconds = time.perf_counter() - start
            code = proc.returncode
            log.write_bytes(captured)
        except subprocess.TimeoutExpired:
            proc.kill()
            captured, _ = proc.communicate()
            log.write_bytes(captured)
            raise
        finally:
            stop.set()
            if sampler:
                sampler.join()
        text = log.read_text()
        row = {'tag': tag, 'schedule': schedule, 'profile': profile, 'seconds': seconds,
               'peak_gpu_mib': max(samples) if samples else None, 'exit_code': code}
        records.append(row)
        if code or ('using cpu backend' if cpu else 'using cuda backend') not in text:
            row['rejected'] = 'execution failed; see log'; save()
            raise ValueError(row['rejected'])
        row['sha256'] = hashlib.sha256(gzip.decompress(out.read_bytes())).hexdigest()
        row['layers_ms'] = {int(layer): float(ms) for layer, ms in
                            re.findall(r'CUDA layer=(\d+) dilation=\d+ conv_ms=([0-9.]+)', text)}
        save()
        return out, row

    try:
        if a.binary:
            builds['existing'] = {'binary': str(a.binary.resolve()), 'sha256': digest(a.binary)}
        else:
            report['nvcc'] = subprocess.check_output([a.nvcc, '--version'], text=True)
            # Copy source files, not build products, untracked personal files or
            # old experiments. New headers in the working source are included.
            for cap in caps:
                name = f'reg{cap}'
                directory = a.output / name
                source = directory / 'src'
                source.mkdir(parents=True)
                for path in (a.source / 'src').iterdir():
                    if path.is_file() and (path.suffix in ('.c', '.h', '.cu', '.cuh', '.S', '.def') or path.name == 'Makefile'):
                        shutil.copy2(path, source / path.name)
                shutil.copytree(a.source / 'src/models', source / 'models')
                shutil.copytree(a.source / 'src/cuda_modules', source / 'cuda_modules')
                build_env = dict(env)
                if cap:
                    build_env['NVCC_APPEND_FLAGS'] = f'--maxrregcount={cap}'
                cmd = ['make', '-C', str(source), '-j2', 'models', 'MODELS=mindmap',
                       'METAL=0', 'OPENCL=0', 'CUDA=1', 'X86_ISA=', 'CC=gcc',
                       f'CUDA_ARCH={a.arch}', f'NVCC={a.nvcc}']
                with (directory / 'build.log').open('w') as log:
                    subprocess.run(cmd, env=build_env, stdout=log, stderr=subprocess.STDOUT, check=True)
                binary = source / 'brainchop-mindmap'
                builds[name] = {'binary': str(binary), 'sha256': digest(binary), 'register_cap': cap}
                save()
        baseline = Path(next(iter(builds.values()))['binary'])
        references = [run(baseline, image, f'cpu-{i}', cpu=True)[0] for i, image in enumerate(inputs)]
        for name, build in builds.items():
            binary = Path(build['binary'])
            accepted, family_hashes = {}, {}
            # Interleave candidates to reduce warmup/thermal order bias.
            order = candidates[:]
            for repeat in range(a.repeats + 1):
                random.Random(repeat).shuffle(order)
                for token in order:
                    for i, image in enumerate(inputs):
                        tag = f'{name}-{token}-{repeat}-{i}'
                        try:
                            out, row = run(binary, image, tag, token, profile=True)
                            differences = label_difference(references[i], out)
                            row['label_differences'] = differences
                            if differences > a.max_label_differences or len(row['layers_ms']) != 13:
                                raise ValueError('accuracy gate or 13-layer timing coverage failed')
                            family = (token[0], i)
                            expected_hash = family_hashes.setdefault(family, row['sha256'])
                            if row['sha256'] != expected_hash:
                                raise ValueError('launch variants within an arithmetic family changed output')
                            if repeat:
                                accepted.setdefault(token, []).append(row)
                        except ValueError as error:
                            accepted.pop(token, None)
                            build.setdefault('rejected', {})[token] = str(error)
                        finally:
                            (a.output / (tag + '.nii.gz')).unlink(missing_ok=True)
                            save()
            accepted = {k: v for k, v in accepted.items() if k not in build.get('rejected', {})
                        and len(v) == a.repeats * len(inputs)}
            if not accepted:
                raise ValueError(f'{name}: no validated candidate')
            choices = []
            for layer in range(13):
                eligible = {k: v for k, v in accepted.items() if layer or k.startswith('s')}
                # A wN shorthand uses the existing s3x512 first-layer default.
                if not eligible:
                    choices.append('s3x512'); continue
                choices.append(min(eligible, key=lambda k: statistics.median(r['layers_ms'][layer] for r in eligible[k])))
            schedule = ','.join(choices)
            timings, baseline_times, peaks, baseline_peaks = (
                {i: [] for i in range(len(inputs))}, {i: [] for i in range(len(inputs))}, [], [])
            final_hashes = {}
            for repeat in range(a.repeats + 1):
                for i, image in enumerate(inputs):
                    # Alternate order; compare both with profiling disabled.
                    order = ('default', 'mixed') if repeat % 2 else ('mixed', 'default')
                    for mode in order:
                        out, row = run(binary, image, f'{name}-{mode}-final-{repeat}-{i}',
                                       schedule if mode == 'mixed' else None)
                        row['label_differences'] = label_difference(references[i], out)
                        if row['label_differences'] > a.max_label_differences:
                            raise ValueError('final schedule failed accuracy gate')
                        expected = final_hashes.setdefault((mode, i), row['sha256'])
                        if expected != row['sha256']:
                            raise ValueError('final schedule output is nondeterministic')
                        if repeat:
                            (timings if mode == 'mixed' else baseline_times)[i].append(row['seconds'])
                            if row['peak_gpu_mib'] is not None:
                                (peaks if mode == 'mixed' else baseline_peaks).append(row['peak_gpu_mib'])
                        out.unlink()
            finalist = {'build': name, 'schedule': schedule, 'seconds_by_input': {str(inputs[i]): statistics.median(v) for i, v in timings.items()},
                        'default_seconds_by_input': {str(inputs[i]): statistics.median(v) for i, v in baseline_times.items()},
                        'speedup_by_input': {str(inputs[i]): statistics.median(baseline_times[i]) / statistics.median(timings[i]) for i in timings},
                        'total_median_seconds': sum(statistics.median(v) for v in timings.values()),
                        'peak_gpu_mib': max(peaks) if peaks else None,
                        'default_peak_gpu_mib': max(baseline_peaks) if baseline_peaks else None}
            finalists.append(finalist); save(); print(json.dumps(finalist), flush=True)
        # Keep the measured default in the choice set: individually faster
        # layers need not make a faster complete model, or use less memory.
        alternatives = list(finalists)
        for f in finalists:
            alternatives.append(dict(build=f['build'], schedule=None,
                seconds_by_input=f['default_seconds_by_input'],
                total_median_seconds=sum(f['default_seconds_by_input'].values()),
                peak_gpu_mib=f['default_peak_gpu_mib']))
        eligible = [f for f in alternatives if a.memory_budget_mib is None or
                    (f['peak_gpu_mib'] is not None and f['peak_gpu_mib'] <= a.memory_budget_mib)]
        report['memory_budget_mib'] = a.memory_budget_mib
        report['recommended'] = min(eligible, key=lambda f: f['total_median_seconds']) if eligible else None
        report['pareto_frontier'] = [f for f in eligible if not any(
            g['peak_gpu_mib'] is not None and f['peak_gpu_mib'] is not None and
            g['total_median_seconds'] <= f['total_median_seconds'] and g['peak_gpu_mib'] <= f['peak_gpu_mib'] and
            (g['total_median_seconds'] < f['total_median_seconds'] or g['peak_gpu_mib'] < f['peak_gpu_mib'])
            for g in eligible)]
        report['status'] = 'complete; release artifact and PVE validation still required before promotion'
    finally:
        save()


if __name__ == '__main__':
    main()
