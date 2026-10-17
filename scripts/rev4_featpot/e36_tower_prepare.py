"""Prepare a frozen whole native phase's file closure; never launch or split jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

from e36_pause_budget import resolve
from e36_store import sha
from e36_data import PREFIX,canonical


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def prepare(root,out,image,run_heavy=None):
    import PIL
    out.mkdir();clock_path=root/'bulk-clock-v6-budget2.json';_,clock,budget=resolve(clock_path)
    plan_path=root/'bulk-plan-v6.json';plan=json.loads(plan_path.read_text())
    if sha(plan_path)!=clock['plan_sha256']:raise ValueError('frozen native phase plan changed')
    manifests={};jobs={}
    for role,entry in plan['phases'].items():
        path=root/'runs'/entry['run']/'manifest.json'
        if sha(path)!=entry['manifest_sha256']:raise ValueError('frozen native phase manifest changed')
        items=json.loads(path.read_text());views=[j['kind']['view'] for j in items]
        if len(views)!=len(set(views)):raise ValueError('duplicate native phase view')
        manifests[role]=dict(path=str(path),sha256=sha(path),views=len(views));jobs[role]=set(views)
    if jobs['fallback']&jobs['additions'] or jobs['fallback']|jobs['additions']!=jobs['full']:
        raise ValueError('whole native phases are not the exact disjoint frozen union')
    runtime_path=Path(plan['runtime']);runtime=json.loads(runtime_path.read_text())
    if sha(runtime_path)!=plan['runtime_sha256']:raise ValueError('frozen runtime changed')
    references=json.loads(Path(runtime['references']).read_text())
    if sha(runtime['references'])!=runtime['references_sha256']:raise ValueError('reference index changed')
    refs={r['reference_key']:r for r in references['rows']};files={};executables=set()
    def add(path,expected=None,dest=None,executable=False):
        path=Path(path);dest=str(dest or path)
        if not path.is_absolute() or not Path(dest).is_absolute():raise ValueError('native file closure requires absolute paths')
        value=dict(source=str(path),destination=dest,sha256=expected or sha(path),bytes=path.stat().st_size)
        if dest in files and files[dest]!=value:raise ValueError('conflicting native phase destination')
        files[dest]=value
        if executable:executables.add(str(path))
    add(runtime_path,plan['runtime_sha256']);add(runtime_path.parent/'executor')
    for name,expected in runtime['program_files'].items():add(runtime_path.parent/'program'/name,expected)
    for key in ['references','source_budget','feature_spec','native_pin']:add(runtime[key])
    add(root/'FEATURE_LAYOUT_BINDING.json',runtime['feature_layout_sha256'])
    add(root/'resource-admission.json',plan['resources_sha256']);add(plan_path);add(clock_path)
    for proof in budget:add(proof['path'],proof['sha256'])
    for name,expected in runtime['mode_inventory_sha256'].items():add(Path(runtime['mode_inventory'])/(name+'.jsonl'),expected)
    pin=json.loads(Path(runtime['native_pin']).read_text())
    add(runtime['native_tool'],pin['binary_sha256'],executable=True)
    add(runtime['feature_tool'],pin['pixel_binary_sha256'],executable=True)
    for name in ['zenfleet-worker','zenfleet-ctl']:add(root/'bin'/name,executable=True)
    for name,path in runtime['oracles'].items():add(path,runtime['oracle_sha256'][name],executable=True)
    selected={json.loads(v[len(PREFIX):])['reference_key'] for v in jobs['additions']}
    # The already-frozen first-producer phase is the parity/admission oracle,
    # never a new subset of the bulk matrix.
    first=root/'runs/first-producer-v6/manifest.json'
    if not first.exists():raise ValueError('original first-producer declaration is required')
    first_jobs=json.loads(first.read_text())
    for j in first_jobs:
        v=json.loads(j['kind']['view'][len(PREFIX):]);selected.add(v['reference_key'])
        if v['role']!='first-producer' or v['runtime_sha256']!=plan['runtime_sha256']:raise ValueError('first-producer runtime/role differs')
    add(first)
    for role in ['additions']:add(manifests[role]['path'],manifests[role]['sha256'])
    for key in selected:
        r=refs[key];add(r['destination'],r['sha256'])
    # Inspection-only Pillow is bundled byte-for-byte, including its wheel's
    # native libraries. No third-party encoder produces a training target.
    site=Path(PIL.__file__).parent.parent
    for package in ['PIL','pillow.libs']:
        directory=site/package
        if not directory.is_dir():raise ValueError('frozen Pillow package closure missing: '+package)
        for path in directory.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts:add(path,dest=Path('/opt/e36/python')/path.relative_to(site))
    library_paths=set()
    for path in sorted(executables):
        text=subprocess.check_output(['ldd',path],text=True)
        if 'not found' in text:raise ValueError('native runtime library unresolved')
        for line in text.splitlines():
            match=re.search(r'(?:=>\s+)?(/\S+)\s+\(0x',line)
            if match:library_paths.add(match.group(1))
    for path in sorted(library_paths):add(path)
    transport=Path(__file__).with_name('e36_local_store_transport.py');add(transport,dest='/opt/e36/transport/ssh')
    add(Path(__file__).with_name('e36_phase_worker.py'),dest='/opt/e36/phase-worker.py')
    if run_heavy is not None:add(run_heavy,dest='/opt/e36/run-heavy')
    write(out/'files.json',dict(schema='e36-whole-native-phase-files-v1',files=list(files.values()),library_paths=sorted(library_paths),
        manifests=manifests,first_producer=dict(path=str(first),sha256=sha(first),jobs=len(first_jobs)),
        image=image,root=str(root),python_source_root=str(site),clock=dict(path=str(clock_path),sha256=sha(clock_path)),runtime=dict(path=str(runtime_path),sha256=sha(runtime_path)),
        source_views=len(jobs['additions']),reference_files=len(selected),file_bytes=sum(r['bytes'] for r in files.values()),
        pillow_version=PIL.__version__,scientific_change=False,fits=0,launch_authorized=False,
        staging_only=True,worker_slots=8,packing_ram_bytes=24*1024**3,container_memory_bytes=32*1024**3,cpuset='0-7,16-23',cpu_shares=256))
    sources=sorted({r['source'].lstrip('/') for r in files.values()})
    with (out/'sources.files0').open('xb') as f:f.write(b'\0'.join(p.encode() for p in sources)+b'\0')
    print(json.dumps(dict(phase_views=len(jobs['additions']),references=len(selected),files=len(files),bytes=sum(r['bytes'] for r in files.values()),staging_only=True,fits=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--image',required=True);p.add_argument('--run-heavy',type=Path)
    a=p.parse_args();prepare(a.root,a.out,a.image,a.run_heavy)
