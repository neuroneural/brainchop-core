"""Dependency-free PEP 517 backend for locally compiled brainchop installations.

Metadata hooks never probe hardware or invoke compilers. build_wheel works on
an isolated copy and records its exact selection; pip may cache that wheel.
"""
import base64
import csv
import hashlib
import gzip
import io
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sysconfig
import tarfile
import tempfile
import zipfile

from brainchop_hardware import MODELS, detect, prune, tool_environment, fatbin_images
from brainchop_cli.adaptive import signature

ROOT=Path(__file__).resolve().parents[1]
VERSION=(ROOT/'packaging/PYPI_VERSION').read_text().strip()
DIST='brainchop-'+VERSION
INFO=DIST+'.dist-info'
EXES=('mindgrab','mindmap-fast','mindmap','mindmap-omni','mindsnap')

def get_requires_for_build_wheel(config_settings=None): return []
def get_requires_for_build_sdist(config_settings=None): return []

def metadata():
    readme = ROOT / 'README.md'
    body = readme.read_text(encoding='utf-8') if readme.is_file() else (
        'Native brain segmentation and skull stripping with five embedded models.\n'
    )
    return (
        'Metadata-Version: 2.2\n'
        'Name: brainchop\n'
        f'Version: {VERSION}\n'
        'Summary: Fast, zero-dependency native brain segmentation and skull stripping with five embedded models\n'
        'Requires-Python: >=3.9\n'
        'License: MIT\n'
        'Keywords: brainchop,neuroimaging,mri,nifti,segmentation,skull-stripping,meshnet\n'
        'Classifier: Development Status :: 5 - Production/Stable\n'
        'Classifier: Intended Audience :: Science/Research\n'
        'Classifier: Intended Audience :: Healthcare Industry\n'
        'Classifier: License :: OSI Approved :: MIT License\n'
        'Classifier: Operating System :: MacOS\n'
        'Classifier: Operating System :: POSIX :: Linux\n'
        'Classifier: Operating System :: Microsoft :: Windows\n'
        'Classifier: Programming Language :: C\n'
        'Classifier: Programming Language :: Python :: 3\n'
        'Classifier: Topic :: Scientific/Engineering :: Medical Science Apps.\n'
        'Project-URL: Homepage, https://brainchop.org\n'
        'Project-URL: Repository, https://github.com/neuroneural/brainchop-core\n'
        'Project-URL: Issues, https://github.com/neuroneural/brainchop-core/issues\n'
        'Description-Content-Type: text/markdown\n\n'
        + body
    )

def entry_points():
    names=['brainchop = brainchop_cli.cli:main']
    names += [f'brainchop-{model} = brainchop_cli.launch:main' for model in EXES]
    return '[console_scripts]\n'+'\n'.join(names)+'\n'

def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    target=Path(metadata_directory)/INFO
    target.mkdir(parents=True,exist_ok=True)
    (target/'METADATA').write_text(metadata(),encoding='utf-8')
    (target/'entry_points.txt').write_text(entry_points(),encoding='utf-8')
    return INFO

def source_files():
    # Explicit source inventory: no Git dependency, training checkpoints,
    # fixtures, caches, binaries, or generation/build toolchains in the sdist.
    yield from (ROOT/name for name in ('pyproject.toml','README.md','LICENSE','THIRD_PARTY_NOTICES.md'))
    yield from (ROOT/'packaging').glob('*.py')
    yield ROOT/'packaging/PYPI_VERSION'
    yield ROOT/'packaging/probe.c'
    yield from (ROOT/'packaging/brainchop').glob('*.py')
    yield from (ROOT/'packaging/brainchop_cli').glob('*.py')
    for path in sorted((ROOT/'src').iterdir()):
        if path.name=='Makefile' or path.suffix in ('.c','.h','.mm','.S'):
            yield path
    for model in MODELS:
        for name in ('model.bin','model_def.h'):
            yield ROOT/'src/models'/model/name
    for directory in ('cuda_modules','hip_artifacts'):
        yield from (p for p in (ROOT/'src'/directory).rglob('*') if p.is_file())
    if (ROOT/'docs/pypi.md').exists(): yield ROOT/'docs/pypi.md'

def source_tarball():
    # Stable across extraction/rebuilds: the source hash identifies contents,
    # never the build host's timestamps, user name or archive creation time.
    buffer=io.BytesIO()
    with gzip.GzipFile(fileobj=buffer,mode='wb',filename='',mtime=0) as compressed:
        with tarfile.open(fileobj=compressed,mode='w',format=tarfile.PAX_FORMAT) as tar:
            entries={path.relative_to(ROOT).as_posix():path.read_bytes() for path in source_files()}
            entries['PKG-INFO']=metadata().encode('utf-8')
            for name,data in sorted(entries.items()):
                info=tarfile.TarInfo(DIST+'/'+name); info.size=len(data); info.mode=0o644
                tar.addfile(info,io.BytesIO(data))
    return buffer.getvalue()


def build_sdist(sdist_directory, config_settings=None):
    name=DIST+'.tar.gz'
    Path(sdist_directory).mkdir(parents=True,exist_ok=True)
    (Path(sdist_directory)/name).write_bytes(source_tarball())
    return name


def build_policy(config_settings):
    settings={k:v for k,v in (config_settings or {}).items() if k.startswith('brainchop.')}
    unknown=set(settings)-{'brainchop.target','brainchop.cpu'}
    if unknown: raise ValueError('unknown brainchop build settings: '+', '.join(sorted(unknown)))
    for key,value in list(settings.items()):
        if isinstance(value,list):
            if len(value)!=1: raise ValueError(key+' accepts one value')
            settings[key]=value[0]
    target=settings.get('brainchop.target',os.environ.get('BRAINCHOP_TARGET_ARCH','auto'))
    cpu=settings.get('brainchop.cpu','auto' if target=='auto' else 'baseline')
    if cpu not in ('auto','baseline'): raise ValueError('brainchop.cpu must be auto or baseline')
    if not isinstance(target,str): raise ValueError('brainchop.target must be a string')
    targets=target.split(',')
    allowed={'auto','cpu','all','gfx1151',*(f'sm{a}' for a in (60,70,75,80,86,89,90))}
    if not targets or any(t not in allowed for t in targets) or len(targets)>1 and any(t in ('auto','cpu','all') for t in targets):
        raise ValueError('brainchop.target must be auto, cpu, all, or comma-separated sm60/70/75/80/86/89/90 and gfx1151')
    settings={'brainchop.target':target,'brainchop.cpu':cpu}
    return dict(target=target,cpu=cpu,settings=settings)


def target_hardware(hardware, policy, inventory):
    if policy['target']=='auto': return hardware
    selected=dict(hardware,devices=[],opencl_build=False)
    targets=policy['target'].split(',')
    if 'all' in targets:
        selected['opencl_build']=True
        if hardware['system'] in ('Linux','Windows'): targets=[f'sm{a}' for a in inventory['cuda_architectures']]
        else: targets=[]
        if hardware['system']=='Linux': targets+=['gfx1151']
        if hardware['system']=='Darwin' and hardware['machine']=='arm64': selected['devices'].append(dict(backend='metal',index=0,arch='apple'))
    for target in targets:
        if target.startswith('sm'):
            if hardware['system'] not in ('Linux','Windows'): raise ValueError('CUDA targets require Linux or Windows')
            selected['devices'].append(dict(backend='cuda',index=len(selected['devices']),arch=int(target[2:]),driver=inventory['cuda_minimum_driver']))
        elif target=='gfx1151':
            if hardware['system']!='Linux': raise ValueError('gfx1151 requires Linux')
            selected['devices'].append(dict(backend='hip',index=0,arch=target))
    return selected


def platform_tag():
    override = os.environ.get('BRAINCHOP_PLATFORM_TAG', '').strip()
    if override:
        if not re.fullmatch(r'[a-z0-9_]+', override):
            raise ValueError(f'invalid BRAINCHOP_PLATFORM_TAG: {override!r}')
        return override
    if platform.system()=='Darwin':
        # Honor an explicit MACOSX_DEPLOYMENT_TARGET (e.g. 13.2 in release
        # builds); otherwise never tag a locally linked host executable for an
        # older OS merely because Python itself used an older deployment target.
        target = os.environ.get('MACOSX_DEPLOYMENT_TARGET', '').strip()
        if target and re.fullmatch(r'\d+\.\d+(?:\.\d+)?', target):
            major, minor = map(int, target.split('.')[:2])
            if major == 10:
                return f'macosx_{major}_{minor}_{platform.machine()}'
            # PEP 425 macOS 11+ tags only use minor=0 (e.g. macosx_14_0_arm64).
            # Round a nonzero minor (like 13.2) up to the next major (14_0) so
            # pip accepts the tag and never installs a 13.2 binary on 13.0/13.1.
            tag_major = major if minor == 0 else major + 1
            return f'macosx_{tag_major}_0_{platform.machine()}'
        major,minor=map(int,platform.mac_ver()[0].split('.')[:2])
        # Before macOS 11, minor versions are distinct deployment targets.
        # A 10.15-linked executable must not advertise compatibility with 10.0.
        return f'macosx_{major}_{minor if major==10 else 0}_{platform.machine()}'
    return sysconfig.get_platform().replace('-','_').replace('.','_')

def audit_linkage(binary, env=None):
    """Reject developer-local libraries before a reusable wheel is produced."""
    system=platform.system()
    if system=='Darwin':
        output=subprocess.check_output(['otool','-L',str(binary)],env=env,text=True)
        deps=[line.strip().split(' (')[0] for line in output.splitlines()[1:]]
        if any(not dep.startswith(('/usr/lib/','/System/Library/')) for dep in deps):
            raise RuntimeError(f'{binary.name}: non-system dependency: {deps}')
    else:
        output=subprocess.check_output(['objdump','-p',str(binary)],env=env,text=True)
        if system=='Windows':
            deps=re.findall(r'DLL Name:\s*(\S+)',output)
            allowed=r'(?:kernel32|msvcrt|ucrtbase|advapi32|user32|shell32|ws2_32|bcrypt)\.dll|api-ms-win-.*\.dll'
        else:
            deps=re.findall(r'^\s*NEEDED\s+(\S+)',output,re.M)
            allowed=r'lib(?:c|m|dl|pthread|rt|z)\.so(?:\.[0-9]+)*|ld-linux[^/]*\.so\.[0-9]+'
            if re.search(r'^\s*(RPATH|RUNPATH)\s+',output,re.M):
                raise RuntimeError(f'{binary.name}: unexpected library search path')
        if not deps or any(not re.fullmatch(allowed,dep.lower()) for dep in deps):
            raise RuntimeError(f'{binary.name}: non-system dependency: {deps}')
    return deps


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    policy=build_policy(config_settings)
    capsule=source_tarball()
    env=tool_environment()
    tag='py3-none-'+platform_tag()
    name=DIST+'-'+tag+'.whl'
    Path(wheel_directory).mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='brainchop-build-') as temp:
        work=Path(temp); source=work/'source'; src=source/'src'
        for path in source_files():
            destination=source/path.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,destination)
        hardware=detect(source,work,env)
        cuda_manifest=json.loads((src/'cuda_modules/manifest.json').read_text())
        inventory=dict(cuda_architectures=sorted({arch for kind,arch,_ in fatbin_images((src/'cuda_modules/mindgrab.fatbin').read_bytes()) if kind==2}),cuda_minimum_driver=cuda_manifest['minimum_driver_api'],hip_architectures=['gfx1151'])
        options,artifacts,diagnostics=prune(src,target_hardware(hardware,policy,inventory))
        print('brainchop hardware: '+json.dumps(hardware),flush=True)
        for message in diagnostics: print('brainchop: '+message,flush=True)
        suffix='.exe' if os.name=='nt' else ''
        package=work/'brainchop_cli'
        shutil.copytree(source/'packaging/brainchop_cli',package)
        api_package=work/'brainchop'
        shutil.copytree(source/'packaging/brainchop',api_package)
        binary_dir=package/'bin'; binary_dir.mkdir()
        common=['make','-C',str(src),'-j'+str(min(os.cpu_count() or 1,4)),'X86_ISA=']
        if os.name=='nt':
            # A globally installed MSVC-targeting clang can pass make's tiny
            # link probe but cannot build this POSIX/pthread runtime. Pin the
            # documented UCRT64 toolchain, already prepended to PATH.
            common+=['CC=gcc','CXX=g++']
        optimized=hardware['cpu']=='avx2-fma' and policy['cpu']=='auto'
        # Preserve FMA on capable hosts; baseline builds support older x86 CPUs.
        cpu_flags='-mavx2 -mfma' if optimized else ''
        command=common+options+['CPUFLAGS='+cpu_flags,'models']
        subprocess.run(command,env=env,check=True)
        shutil.copy2(work/('probe'+suffix),binary_dir/('hardware-probe'+suffix))
        records={}
        for model in EXES:
            path=src/('brainchop-'+model+suffix)
            data=path.read_bytes()
            records[path.name]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data))
            shutil.copy2(path,binary_dir/path.name)
        for path in binary_dir.rglob('*'):
            if path.is_file():
                data=path.read_bytes()
                records[path.relative_to(binary_dir).as_posix()]=dict(sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),linkage=audit_linkage(path,env))
        manifest=dict(version=VERSION,hardware=hardware,options=options,cpu_target='avx2-fma' if optimized else 'baseline',
                      artifacts=artifacts,diagnostics=diagnostics,executables=records,
                      runtime_environment=signature(hardware),build_policy=policy,artifact_policy=inventory,
                      source_capsule=dict(file='source.tar.gz',sha256=hashlib.sha256(capsule).hexdigest()),
                      cache_contract='automatic first-use rebuild for incompatible cached hardware; explicit targets remain fixed')
        (package/'source.tar.gz').write_bytes(capsule)
        (package/'build.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        entries={p.relative_to(work).as_posix():(p.read_bytes(),0o755 if binary_dir in p.parents else 0o644)
                 for root_pkg in (package, api_package) for p in root_pkg.rglob('*') if p.is_file()}
        entries[INFO+'/METADATA']=(metadata().encode('utf-8'),0o644)
        entries[INFO+'/entry_points.txt']=(entry_points().encode('utf-8'),0o644)
        entries[INFO+'/WHEEL']=(f'Wheel-Version: 1.0\nGenerator: brainchop-source\nRoot-Is-Purelib: false\nTag: {tag}\n'.encode('utf-8'),0o644)
        for license_file in ('LICENSE','THIRD_PARTY_NOTICES.md'):
            entries[INFO+'/licenses/'+license_file]=((ROOT/license_file).read_bytes(),0o644)
        record=io.StringIO(newline=''); writer=csv.writer(record,lineterminator='\n')
        with zipfile.ZipFile(Path(wheel_directory)/name,'w',compression=zipfile.ZIP_DEFLATED) as archive:
            for path,(data,mode) in sorted(entries.items()):
                info=zipfile.ZipInfo(path); info.external_attr=(0o100000|mode)<<16; info.compress_type=zipfile.ZIP_DEFLATED
                archive.writestr(info,data)
                digest=base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
                writer.writerow((path,'sha256='+digest,len(data)))
            writer.writerow((INFO+'/RECORD','',''))
            archive.writestr(INFO+'/RECORD',record.getvalue())
    return name
