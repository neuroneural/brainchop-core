#!/usr/bin/env python3
"""Sequential native HIP labels, PVE, timing, process memory and fallback gates.

Uses standard library only. Every GPU success requires a nonzero actual HIP
launch count and no fallback. /usr/bin/time host RSS and /proc DRM accounting
are reported separately from successful runtime hipMalloc accounting.
"""
import argparse
import gzip
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

import nii_diff
from cuda_pve_check import compare as compare_pve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from hip_memory import check_memory


def compare(reference, candidate, limit):
    a, b = nii_diff.read(str(reference)), nii_diff.read(str(candidate))
    assert nii_diff.header_matches(a, b), f'headers differ: {reference} {candidate}'
    av, bv = nii_diff.voxels(a, str(reference)), nii_diff.voxels(b, str(candidate))
    assert len(av) == len(bv)
    different = sum(x != y for x, y in zip(av, bv))
    assert different <= limit, (candidate, different, limit)
    return different


def sample_drm(pid):
    """Per-process resident DRM counters, deduplicated by client, in bytes.

    These include driver allocations and residency; they are NOT a second
    measurement of hipMalloc requested bytes. Never use global free memory.
    """
    result = {}
    try:
        children = Path(f'/proc/{pid}/task/{pid}/children').read_text().split()
    except OSError:
        children = []
    for process in [str(pid)] + children:
        clients = {}
        for fd in Path(f'/proc/{process}/fdinfo').glob('*'):
            try:
                text = fd.read_text()
            except OSError:
                continue
            client = re.search(r'^drm-client-id:\s*(\d+)', text, re.M)
            if not client:
                continue
            counters = {}
            for key, number, unit in re.findall(r'^(drm-(?:memory|resident|shared|total|active|purgeable)-[\w-]+):\s*(\d+)\s*(\w*)', text, re.M):
                counters[key] = int(number) * {'KiB': 1024, 'MiB': 1024 ** 2, 'B': 1, '': 1}.get(unit, 1)
            clients[client[1]] = counters
        for counters in clients.values():
            for key, value in counters.items():
                result[key] = result.get(key, 0) + value
    return result


def run_process(command, env, log, timeout=900):
    """Run a wrapper and its inference child in one owned process group.

    Killing only /usr/bin/time leaves its GPU child alive. A separate session
    lets timeout/error cleanup kill the entire group without touching another
    test or the shared host's model server.
    """
    peaks = {}
    started = time.perf_counter()
    proc = subprocess.Popen(command, env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    try:
        while proc.poll() is None:
            for key, value in sample_drm(proc.pid).items():
                peaks[key] = max(value, peaks.get(key, 0))
            if time.perf_counter() - started > timeout:
                raise TimeoutError(f'inference exceeded {timeout}s: {command[0]}')
            time.sleep(.01)
    except BaseException:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
        raise
    return proc.returncode, peaks, time.perf_counter() - started


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binaries', type=Path, default=ROOT / 'src')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--expect-unavailable', action='store_true')
    p.add_argument('--hide-runtime', action='store_true', help='test-only failed dlopen/dlsym runtime shim')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    a.output = a.output.resolve()
    env = dict(os.environ, BC_NUM_THREADS='4', BC_TIME_GPU='1', BC_HIP_MEMORY='1')
    if a.hide_runtime:
        a.expect_unavailable = True
        shim = a.output / 'hidden-runtime'
        shim.mkdir()
        source = shim / 'unavailable.c'
        source.write_text('int bc_unavailable_hip_runtime(void) { return 0; }\n')
        subprocess.run(['gcc', '-shared', '-fPIC', str(source), '-o', str(shim / 'libamdhip64.so')], check=True)
        env['LD_LIBRARY_PATH'] = str(shim)
        env['ROCM_PATH'] = str(shim)
    rows = []
    profiles = {}
    for path in (ROOT / 'src/hip_artifacts').glob('*/*/manifest.json'):
        manifest = json.loads(path.read_text())
        profiles[(manifest['model']['name'], manifest['target']['architecture'])] = manifest
        for mode in (('labels', 'pve') if manifest['model'].get('supports_pve') else ('labels',)):
            check_memory(manifest, mode)

    def run(label, model, fixture='t1_crop', space='conformed', mode='hip', extra=(), success=True):
        output = a.output / (label + '.nii.gz')
        command = [str((a.binaries / ('brainchop-' + model)).resolve()), '-backend', mode,
                   '--no-turbo', '-o', str(output), *extra]
        if space == 'conformed' and model != 'mindgrab':
            command += ['--save-conform']
        command += [str(ROOT / 'test' / (fixture + '.nii.gz'))]
        logpath, timing = a.output / (label + '.log'), a.output / (label + '.time')
        with logpath.open('w') as log:
            returncode, peaks, elapsed = run_process(
                ['/usr/bin/time', '-v', '-o', str(timing), *command], env, log)
        text = logpath.read_text()
        row = dict(case=label, wall_seconds=elapsed,
                   returncode=returncode, drm_process_peak_bytes=peaks)
        rss = re.search(r'Maximum resident set size \(kbytes\): (\d+)', timing.read_text())
        if rss:
            row['host_max_rss_bytes'] = int(rss[1]) * 1024
        if success:
            assert returncode == 0, (label, text)
            expected = 'cpu' if mode == 'cpu' or a.expect_unavailable else 'hip'
            assert f'using {expected} backend' in text, (label, text)
            assert 'falling back' not in text.lower(), (label, text)
            if expected == 'hip':
                launch = re.search(r'HIP (\d+) dispatches in ([0-9.]+) s', text)
                assert launch and int(launch[1]) > 0 and 'gfx1151' in text, (label, text)
                row.update(dispatches=int(launch[1]), gpu_seconds=float(launch[2]))
                memory = re.search(r'high_water=(\d+) bytes', text)
                assert memory, (label, text)
                row['hip_malloc_high_water_bytes'] = int(memory[1])
                identity = re.search(r'model=(\S+) mode=(labels|pve)', text)
                assert identity, (label, 'HIP memory report lacks model/mode', text)
                profile = profiles[(identity[1], 'gfx1151')]
                row['memory_plan'] = check_memory(profile, identity[2], int(memory[1]))
        else:
            assert returncode != 0 and not output.exists(), (label, text)
        rows.append(row)
        print(json.dumps(row), flush=True)
        return output, row

    try:
        models = ('mindgrab', 'mindmap-fast', 'mindmap', 'mindsnap')
        if a.expect_unavailable:
            for model in models:
                listing = subprocess.check_output([str((a.binaries / ('brainchop-' + model)).resolve()),
                    '-a', '-1'], env=env, stderr=subprocess.STDOUT, text=True)
                assert 'hip: no adapters found' in listing, listing
                run(model + '-forced', model, fixture='t1_2mm', space='native', success=False)
                cpu, _ = run(model + '-cpu', model, fixture='t1_2mm', space='native', mode='cpu')
                auto, row = run(model + '-auto', model, fixture='t1_2mm', space='native', mode='auto')
                row['different_voxels'] = compare(cpu, auto, 0)
            return
        listing = subprocess.check_output([str((a.binaries / 'brainchop-mindsnap').resolve()),
            '-a', '-1'], env=env, stderr=subprocess.STDOUT, text=True)
        assert 'gfx1151' in listing and 'no embedded artifact' not in listing, listing
        run('mindsnap-adapter-999', 'mindsnap', fixture='t1_2mm', space='native',
            extra=['-a', '999'], success=False)
        for model, frozen, limits in (
            ('mindgrab', ('bet.nii.gz', None), (64, 64)),
            ('mindmap-fast', ('labels16.nii.gz', 'labels16_native_majority_wasm.nii.gz'), (1024, 256)),
            ('mindmap', ('labels24.nii.gz', 'labels24_native_majority.nii.gz'), (1024, 256)),
            ('mindsnap', ('labels104.nii.gz', 'labels104_native_majority.nii.gz'), (2048, 512))):
            for index, (fixture, space) in enumerate((('t1_crop', 'conformed'), ('t1_2mm', 'native'))):
                extra = ['--legacy-cleanup'] if model == 'mindmap' else []
                label = f'{model}-{fixture}-{space}'
                reference = ROOT / 'test' / frozen[index] if frozen[index] else run(
                    label + '-cpu', model, fixture, space, 'cpu')[0]
                hip, row = run(label + '-hip', model, fixture, space, extra=extra)
                row['different_voxels'] = compare(reference, hip, limits[index])
                auto, row = run(label + '-auto', model, fixture, space, 'auto', extra)
                row['different_from_forced'] = compare(hip, auto, 0)
        # Remaining MindSnap fixture/space cells: CPU reference with same header.
        for fixture, space in (('t1_crop', 'native'), ('t1_2mm', 'conformed')):
            label = f'mindsnap-{fixture}-{space}'
            cpu, _ = run(label + '-cpu', 'mindsnap', fixture, space, 'cpu')
            hip, row = run(label + '-hip', 'mindsnap', fixture, space)
            row['different_voxels'] = compare(cpu, hip, 2048)
            auto, row = run(label + '-auto', 'mindsnap', fixture, space, 'auto')
            row['different_from_forced'] = compare(hip, auto, 0)
        # Default categorical cleanup uses same-run CPU; frozen references above
        # deliberately use --legacy-cleanup. PVE bypasses categorical cleanup.
        for fixture in ('t1_crop', 't1_2mm'):
            space = 'conformed' if fixture == 't1_crop' else 'native'
            label = f'mindmap-default-{fixture}'
            cpu, _ = run(label + '-cpu', 'mindmap', fixture, space, 'cpu')
            hip, row = run(label + '-hip', 'mindmap', fixture, space)
            row['different_voxels'] = compare(cpu, hip, 1024 if space == 'conformed' else 256)
            for space in ('native', 'conformed'):
                label = f'pve-{fixture}-{space}'
                cpu, _ = run(label + '-cpu', 'mindmap', fixture, space, 'cpu', ['--pve'])
                hip, row = run(label + '-hip', 'mindmap', fixture, space, extra=['--pve'])
                row['pve'] = {t: compare_pve(cpu.with_name(cpu.stem.replace('.nii', '') + '_' + t + '.nii.gz'),
                                           hip.with_name(hip.stem.replace('.nii', '') + '_' + t + '.nii.gz'))
                              for t in ('gm', 'wm', 'csf', 'brain')}
        # Warm-up and three separately reported kernel/end-to-end runs.
        for i in range(4):
            _, row = run('timing-warmup' if i == 0 else f'timing-{i}', 'mindsnap')
            row['warmup'] = i == 0
    finally:
        (a.output / 'results.json').write_text(json.dumps(rows, indent=2) + '\n')


if __name__ == '__main__':
    main()
