"""Build-time detection and exact artifact selection, using only the standard library."""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import struct
import subprocess

from brainchop_cli.adaptive import cuda_choice, parse_probe

MODELS = ('mindgrab', 'model16chan18cls', 'mindmap', 'mindmap_omni', 'mindsnap')
# Offline CUDA fatbins on disk (mindmap_omni shares mindmap.fatbin).
CUDA_MODULES = ('mindgrab', 'model16chan18cls', 'mindmap', 'mindsnap')

def fatbin_images(blob):
    """Read the pinned CUDA 12.8 container; preserve each image byte for byte.

    This is not a general NVIDIA file parser. Reject unknown layouts. The
    maintainer inventory tests exercise every checked-in image and ELF target.
    """
    if len(blob) < 16 or struct.unpack_from('<IHH', blob) != (0xba55ed50, 1, 16):
        raise ValueError('unsupported CUDA fatbin header')
    if struct.unpack_from('<Q', blob, 8)[0] != len(blob)-16:
        raise ValueError('invalid CUDA fatbin extent')
    offset, images = 16, []
    while offset < len(blob):
        if offset+64 > len(blob):
            raise ValueError('truncated CUDA image')
        kind, version, header, size = struct.unpack_from('<HHIQ', blob, offset)
        arch = struct.unpack_from('<I', blob, offset+28)[0]
        end = offset+header+size
        if version != 0x101 or (kind, header) not in ((2,64),(1,80)) or end > len(blob) or size == 0:
            raise ValueError('unsupported CUDA image layout')
        image = blob[offset:end]
        if kind == 2:
            elf=image[header:]
            if elf[:5] != b'\x7fELF\x02' or struct.unpack_from('<H',elf,18)[0] != 190 or struct.unpack_from('<I',elf,48)[0] & 255 != arch:
                raise ValueError('CUDA image architecture differs from ELF flags')
        images.append((kind, arch, image))
        offset=end
    return images

def select_cuda(blob, architectures):
    images=fatbin_images(blob)
    selected=[]
    cubins=[a for kind,a,_ in images if kind==2]
    for wanted in sorted(set(architectures)):
        chosen=cuda_choice(cubins,wanted)
        if chosen is not None and chosen not in selected: selected.append(chosen)
    kept=b''.join(data for kind,a,data in images if kind==2 and a in selected)
    if not kept: return None, []
    return struct.pack('<IHHQ',0xba55ed50,1,16,len(kept))+kept, selected

def tool_environment():
    env=dict(os.environ)
    for key in ('MAKEFLAGS','MFLAGS','GNUMAKEFLAGS','MAKELEVEL','MAKEOVERRIDES',
                'CFLAGS','CPPFLAGS','CXXFLAGS','LDFLAGS','LDLIBS','CPATH',
                'LIBRARY_PATH','C_INCLUDE_PATH','CPLUS_INCLUDE_PATH'):
        env.pop(key,None)
    if os.name=='nt':
        # Normal python.org Python from PowerShell works with the conventional
        # MSYS2 installation; no shell activation or user backend flags needed.
        root=Path(env.get('MSYS2_ROOT','C:/msys64'))
        env['PATH']=str(root/'ucrt64/bin')+os.pathsep+str(root/'usr/bin')+os.pathsep+env['PATH']
        env['MSYSTEM']='UCRT64'
    return env

def detect(root, work, env):
    system, machine=platform.system(), platform.machine().lower()
    if struct.calcsize('P')!=8:
        raise RuntimeError('brainchop requires 64-bit Python and a 64-bit native toolchain')
    if system=='Windows' and machine not in ('amd64','x86_64'):
        raise RuntimeError('Windows source builds currently support x64 only')
    if system not in ('Darwin','Linux','Windows') or machine not in ('arm64','aarch64','x86_64','amd64'):
        raise RuntimeError(f'unsupported native platform: {system}/{machine}')
    cc=shutil.which('gcc' if system=='Windows' else 'cc',path=env['PATH']) or shutil.which('gcc',path=env['PATH'])
    if not cc or not shutil.which('make',path=env['PATH']):
        raise RuntimeError('brainchop requires a native C compiler, GNU make and zlib development files; see docs/pypi.md (macOS: Xcode Command Line Tools; Linux: build-essential zlib1g-dev; Windows: MSYS2 UCRT64 GCC, make, zlib)')
    probe=work/('probe.exe' if os.name=='nt' else 'probe')
    command=[cc,'-O2','-I'+str(root/'src'),str(root/'packaging/probe.c'),'-o',str(probe)]
    if system=='Linux': command+=['-ldl']
    if system=='Windows': command+=['-static','-static-libgcc']
    subprocess.run(command,env=env,check=True)
    output=[]; diagnostics=[]
    for backend in ('cuda','hip','opencl'):
        try:
            result=subprocess.run([str(probe),'--'+backend],env=env,text=True,capture_output=True,timeout=30)
        except subprocess.TimeoutExpired:
            result=subprocess.CompletedProcess([],1,'',backend+' probe timed out')
        if result.returncode:
            diagnostics.append(f'{backend} probe failed; omitting this backend: '+result.stderr)
        else:
            output.extend(result.stdout.splitlines())
    cpu_probe=subprocess.run([str(probe),'--cpu-only'],env=env,text=True,capture_output=True,timeout=10,check=True)
    cpu=parse_probe(cpu_probe.stdout)[0] or 'baseline'
    devices=parse_probe('\n'.join(output))[1]
    opencl_build=any(d['backend']=='opencl' for d in devices)
    return dict(system=system,machine=machine,cpu=cpu,devices=devices,opencl_build=opencl_build,diagnostics=diagnostics)

def prune(src, hardware):
    diagnostics=list(hardware.get('diagnostics',[])); selected={}; options=['METAL=0','CUDA=0','HIP=0','OPENCL=0','ZSTD=0','HIP_ARCH=gfx1151']
    cuda=json.loads((src/'cuda_modules/manifest.json').read_text())
    cuda_devices=[d['arch'] for d in hardware['devices'] if d['backend']=='cuda' and d['driver']>=cuda['minimum_driver_api']]
    cuda_targets=[]
    for model in CUDA_MODULES:
        path=src/'cuda_modules'/f'{model}.fatbin'; original=path.read_bytes()
        if hashlib.sha256(original).hexdigest()!=cuda['models'][model]['sha256']:
            raise ValueError('CUDA artifact hash mismatch: '+model)
        blob, targets=select_cuda(original,cuda_devices)
        if blob:
            path.write_bytes(blob); cuda_targets=targets
            selected[path.relative_to(src).as_posix()]=dict(sha256=hashlib.sha256(blob).hexdigest(),architectures=targets,bytes=len(blob))
        else: path.unlink()
    if cuda_targets: options[1]='CUDA=driver'
    else: shutil.rmtree(src/'cuda_modules')
    hip_devices={d['arch'] for d in hardware['devices'] if d['backend']=='hip'}
    for path in (src/'hip_artifacts').glob('*/*'):
        if path.name not in hip_devices or path.name!='gfx1151': shutil.rmtree(path); continue
        manifest=json.loads((path/'manifest.json').read_text())
        blob=(path/'kernels.bin').read_bytes()
        if hashlib.sha256(blob).hexdigest()!=manifest['bundle']['sha256'] or len(blob)!=manifest['bundle']['size']:
            raise ValueError('HIP bundle hash/size mismatch: '+str(path))
        model=path.parent.name
        if hashlib.sha256((src/'models'/model/'model_def.h').read_bytes()).hexdigest()!=manifest['model']['descriptor_sha256']:
            raise ValueError('HIP artifact/model mismatch: '+model)
        selected[(path/'kernels.bin').relative_to(src).as_posix()]=dict(sha256=hashlib.sha256(blob).hexdigest(),architecture=path.name,bytes=len(blob))
        options[2]='HIP=1'
    if any(d['backend']=='metal' for d in hardware['devices']):
        options[0]='METAL=1'
        for path in src.glob('metal_*_generated.h'):
            blob=path.read_bytes()
            selected[path.name]=dict(sha256=hashlib.sha256(blob).hexdigest(),bytes=len(blob),architecture='apple-metal-source')
    else:
        for path in src.glob('metal_*_generated.h'): path.unlink()
    if hardware.get('opencl_build'): options[3]='OPENCL=1'
    elif any(d['backend']=='opencl' for d in hardware['devices']):
        diagnostics.append('OpenCL was disabled by the build selection; CPU fallback remains available.')
    if not selected and options[0]=='METAL=0' and options[3]=='OPENCL=0': diagnostics.append('No validated compatible GPU artifact selected; CPU fallback is available.')
    for d in hardware['devices']:
        if d['backend']=='hip' and d['arch']!='gfx1151': diagnostics.append(f"HIP {d['arch']}: no validated artifact; CPU fallback.")
        if d['backend']=='cuda' and (d['driver']<cuda['minimum_driver_api'] or cuda_choice(cuda_targets,d['arch']) is None):
            diagnostics.append(f"CUDA device {d['index']}: no compatible validated cubin or driver; CPU fallback.")
    return options, selected, diagnostics
