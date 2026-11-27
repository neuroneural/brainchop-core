#!/usr/bin/env python3
"""Audit and archive a Linux/Windows CUDA Driver API build; never bundle drivers."""
import argparse
import hashlib
import json
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'packaging'))
from brainchop_build import audit_linkage  # noqa: E402  one owner of the OS-library allowlist


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--src', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--version', required=True)
    p.add_argument('--dispatcher', default=None, help='optional standalone brainchop CLI dispatcher executable')
    p.add_argument('executables', nargs='+')
    a = p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9._-]+', a.version):
        p.error('version must contain only letters, digits, dot, underscore and hyphen')
    windows = platform.system() == 'Windows'
    if not windows and platform.system() != 'Linux':
        p.error('use the macOS packaging targets on macOS')
    config = (a.src / '.build-config').read_text().strip()
    if 'CUDA=driver' not in config.split() or 'OPENCL=0' not in config.split():
        p.error('requires CUDA=driver OPENCL=0; use make native-package')
    suffix = '.exe' if windows else ''
    modules = json.loads((a.src / 'cuda_modules/manifest.json').read_text())
    # Every model with a checked-in fatbin ships; model16's executable is brainchop-mindmap-fast,
    # and mindmap-omni shares mindmap's 24ch/18cls fatbin.
    model_exes = {'brainchop-' + ('mindmap-fast' if m == 'model16chan18cls' else m) + suffix: m
                  for m in modules['models']}
    if 'mindmap' in modules['models']:
        model_exes['brainchop-mindmap-omni' + suffix] = 'mindmap'
    if set(a.executables) != set(model_exes) or len(a.executables) != len(model_exes):
        p.error('package must contain one executable per CUDA module: ' + ', '.join(sorted(model_exes)))
    records = {}
    for name in a.executables:
        binary = a.src / name
        data = binary.read_bytes()
        model = model_exes[name]
        blob = (a.src / 'cuda_modules' / (model + '.fatbin')).read_bytes()
        if blob not in data:
            raise ValueError(f'{name}: checked-in CUDA module is not embedded')
        records[name] = dict(sha256=hashlib.sha256(data).hexdigest(), imports=audit_linkage(binary))
    staged_exes = list(a.executables)
    if a.dispatcher:
        disp = a.src / a.dispatcher
        if not disp.is_file():
            p.error(f'dispatcher executable not found: {disp}')
        data = disp.read_bytes()
        records[a.dispatcher] = dict(sha256=hashlib.sha256(data).hexdigest(), imports=audit_linkage(disp))
        staged_exes.insert(0, a.dispatcher)
    system = 'windows' if windows else 'linux'
    arch = platform.machine().lower().replace('amd64', 'x86_64')
    name = f'brainchop-{a.version}-{system}-{arch}'
    a.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        stage = Path(temp) / name
        stage.mkdir()
        for exe in staged_exes:
            shutil.copy2(a.src / exe, stage / exe)
        for notice in ('LICENSE', 'THIRD_PARTY_NOTICES.md'):
            shutil.copy2(ROOT / notice, stage / notice)
        shutil.copy2(ROOT / 'packaging/README-native.txt', stage / 'README.txt')
        (stage / 'build.json').write_text(json.dumps(dict(version=a.version, system=system,
            architecture=arch, build_config=config, executables=records,
            minimum_cuda_driver_api=modules['minimum_driver_api'],
            cuda_module_hashes={m: v['sha256'] for m, v in modules['models'].items()}), indent=2) + '\n')
        archive = shutil.make_archive(str((a.output / name).resolve()),
                                      'zip' if windows else 'gztar', temp, name)
    digest = hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    Path(archive + '.sha256').write_text(f'{digest}  {Path(archive).name}\n')
    print(f'PASS: audited {system} archive: {archive}')


if __name__ == '__main__':
    main()
