#!/usr/bin/env python3
"""Test a pip-installed wheel, including real inference against tracked fixtures.

Run with the Python interpreter from the environment under test. No test or ML
packages are required. --smoke-only skips inference for a fast install check.
"""
import argparse
import gzip
import importlib.metadata
import json
from pathlib import Path
import os
import platform
import struct
import subprocess
import sys
import sysconfig
import tempfile

import nii_diff


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-only', action='store_true')
    parser.add_argument('--cpu-fallback-only', action='store_true', help='check all four native CPU kernels, including the baseline x86 contract')
    args = parser.parse_args()
    fixtures = Path(__file__).resolve().parent
    scripts = Path(sysconfig.get_path('scripts'))
    suffix = '.exe' if os.name == 'nt' else ''
    command = scripts / ('brainchop' + suffix)
    import brainchop_cli
    build_record=json.loads((Path(brainchop_cli.__file__).parent/'build.json').read_text())
    baseline_x86=build_record['cpu_target']=='baseline' and platform.machine().lower() in ('x86_64','amd64')
    def parity(actual, reference, model):
        a,b=gzip.decompress(Path(actual).read_bytes()),gzip.decompress(Path(reference).read_bytes())
        assert nii_diff.descrip(a).startswith('brainchop '), ('missing provenance descrip', model, nii_diff.descrip(a))
        if baseline_x86 and model=='mindsnap':
            assert nii_diff.header_matches(a,b) and len(a)==len(b)
            differing=sum(x!=y for x,y in zip(a[352:],b[352:]))
            assert differing<=1, ('baseline MindSnap changed beyond its measured one-voxel rounding difference',differing)
            print(f'PASS: baseline MindSnap differs at {differing} voxels (limit 1); AVX2/FMA reference unchanged',flush=True)
        else: assert nii_diff.equal_ignoring_descrip(a,b),model
    metadata = importlib.metadata.metadata('brainchop')
    assert metadata['Version'] == (fixtures.parent / 'packaging/PYPI_VERSION').read_text().strip()
    assert not metadata.get_all('Requires-Dist'), 'the native package must need no Python dependencies'

    def run(*arguments, check=True, cwd=None):
        result = subprocess.run([str(command), *map(str, arguments)], capture_output=True,
                                text=True, cwd=cwd)
        if check and result.returncode:
            raise AssertionError(f'{arguments}: exit {result.returncode}\n{result.stdout}\n{result.stderr}')
        return result

    for option in ('--help', '--list', '--update', '--version'):
        assert run(option).stdout.strip()
    assert not (scripts / ('brainchop-16chan18cls' + suffix)).exists(), 'duplicate public command remains'
    entry_points = importlib.metadata.distribution('brainchop').entry_points
    assert {e.name for e in entry_points if e.group == 'console_scripts'} == {
        'brainchop', 'brainchop-mindgrab', 'brainchop-mindmap-fast',
        'brainchop-mindmap', 'brainchop-mindsnap'}
    version_header = (Path(__file__).resolve().parents[1] / 'src' / 'version.h').read_text()
    bc_version = next(
        (line.split('"')[1] for line in version_header.splitlines() if '#define BC_VERSION ' in line),
        '1.0.0',
    )
    listing = run('--list').stdout
    for model in ('mindgrab', 'mindmap-fast', 'mindmap', 'mindsnap'):
        assert model in listing
        executable = scripts / ('brainchop-' + model + suffix)
        help_result = subprocess.run([str(executable), '--help'], capture_output=True, text=True)
        assert help_result.returncode == 0 and f'v{bc_version}' in help_result.stdout, help_result.stdout
        assert 'usage: brainchop-'+model in help_result.stdout
    import brainchop as bc_api
    assert set(bc_api.list_models()) == {'mindgrab', 'mindmap-fast', 'mindmap', 'mindsnap'}
    assert run('missing.nii.gz', '-m', 'tissue_fast', check=False).returncode == 2
    assert run('missing.nii.gz', check=False).returncode == 2
    print('PASS: installed console command, four binaries, model listing, error exits, no runtime dependencies', flush=True)
    if args.smoke_only:
        return

    if args.cpu_fallback_only:
        import brainchop_cli
        binaries=Path(brainchop_cli.__file__).parent/'bin'
        assert binaries.is_dir()
        with tempfile.TemporaryDirectory(prefix='brainchop-cpu-fallback-') as temporary:
            for model,reference in (('mindgrab','bet.nii.gz'),('mindmap-fast','labels16.nii.gz'),('mindmap','labels24.nii.gz'),('mindsnap','labels104.nii.gz')):
                output=Path(temporary)/(model+'.nii.gz')
                extra=['--save-conform'] if model!='mindgrab' else []
                if model=='mindmap': extra+=['--legacy-cleanup']
                binary=binaries/('brainchop-'+model+suffix)
                if not binary.exists(): binary=binaries.parent/binary.name
                result=subprocess.run([str(binary),'-cpu','4','--no-turbo',*extra,'-o',str(output),str(fixtures/'t1_crop.nii.gz')],capture_output=True,text=True)
                assert result.returncode==0,result.stderr
                parity(output,fixtures/reference,model)
                print(f'PASS: CPU runtime {model} meets its recorded numerical contract',flush=True)
        return

    def content(path):
        return gzip.decompress(Path(path).read_bytes())

    with tempfile.TemporaryDirectory(prefix='brainchop-pip-inference-') as temporary:
        work = Path(temporary)
        # All CLI models default to input space; pass --save-conform here to
        # verify parity against the conformed 256^3 reference fixtures.
        for model, reference in (('mindmap', 'labels24.nii.gz'),
                                 ('mindmap-fast', 'labels16.nii.gz'),
                                 ('mindsnap', 'labels104.nii.gz')):
            output = work / f'{model} output.nii.gz'
            model_args = [] if model == 'mindmap' else ['-m', model]
            run(fixtures / 't1_crop.nii.gz', *model_args, '--save-conform', '--threads', '4', '--no-turbo', '-o', output)
            expected = fixtures / reference
            if model == 'mindmap':
                # The historical labels24 fixture uses --legacy-cleanup. The
                # adapter must retain the native model's newer default cleanup.
                expected = work / 'mindmap-direct.nii.gz'
                direct = subprocess.run([str(scripts / ('brainchop-mindmap' + suffix)),
                                         '-cpu', '4', '--no-turbo', '--save-conform', '-o', str(expected),
                                         str(fixtures / 't1_crop.nii.gz')],
                                        capture_output=True, text=True)
                assert direct.returncode == 0, direct.stderr
            parity(output,expected,model)
            kind = 'direct native command' if model == 'mindmap' else 'tracked fixture'
            print(f'PASS: installed {model}, conformed output meets its numerical contract against {kind}', flush=True)

        inverse = work / 'inverse.nii.gz'
        result = run(fixtures / 't1_2mm.nii.gz', '-m', 'mindmap-fast', '--inverse-conform',
                     '--threads', '4', '--no-turbo', '-o', inverse)
        # Match src/Makefile: the frozen x86 result differs from ARM at one
        # voxel sampled by majority voting. Both are exact architecture gates.
        inverse_reference = ('labels16_native_majority_wasm.nii.gz'
                             if platform.machine().lower() in ('x86_64', 'amd64')
                             else 'labels16_native_majority.nii.gz')
        inverse_bytes = content(inverse)
        assert nii_diff.descrip(inverse_bytes).startswith('brainchop '), nii_diff.descrip(inverse_bytes)
        assert nii_diff.equal_ignoring_descrip(inverse_bytes, content(fixtures / inverse_reference))
        print('PASS: mindmap-fast and --inverse-conform, input-space output is bit-exact', flush=True)

        stripped = work / 'stripped.nii.gz'
        mask = work / 'mask.nii.gz'
        run(fixtures / 't1_crop.nii.gz', '--skull-strip', '--threads', '4', '--mask', mask,
            '--export-classes', '-o', stripped)
        stripped_bytes = content(stripped)
        assert nii_diff.descrip(stripped_bytes).startswith('brainchop '), nii_diff.descrip(stripped_bytes)
        assert nii_diff.equal_ignoring_descrip(stripped_bytes, content(fixtures / 'bet.nii.gz'))
        mask_bytes = content(mask)
        assert struct.unpack_from('<8h', mask_bytes, 40)[1:4] == (256, 256, 256)
        assert struct.unpack_from('<h', mask_bytes, 70)[0] == 2  # uint8
        assert set(mask_bytes[352:]) <= {0, 1}
        for index in range(2):
            raw = work / f'stripped_c{index}.nii'
            assert raw.is_file(), f'missing legacy export filename: {raw}'
            with raw.open('rb') as handle:
                header = handle.read(352)
            assert struct.unpack_from('<h', header, 70)[0] == 16  # float32
            assert struct.unpack_from('<8h', header, 40)[1:4] == (256, 256, 256)
            assert not raw.with_suffix('.nii.gz').exists()
        print('PASS: --skull-strip, --mask and legacy per-class filenames; stripped output is bit-exact', flush=True)


if __name__ == '__main__':
    main()
