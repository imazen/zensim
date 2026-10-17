"""Freeze E36 native data views; zenfleet alone schedules and records execution."""
import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

from e36_data import Q, PREFIX, CAPABILITY, canonical
from e36_store import sha
from e34_encode import E36_CODECS


def write(p,v):
    with Path(p).open('x') as f:f.write(json.dumps(v,indent=2,allow_nan=False)+'\n')


def expand_modes(initial,out):
    out.mkdir()
    census={}
    for codec,_,_ in E36_CODECS:
        src=initial/'native-plans-main-effects'
        rows=[json.loads(l) for l in (src/(codec+'.jsonl')).read_text().splitlines()]
        physical={json.loads(r['knob_tuple_json'])['cell']+'_q'+str(r['q']):r for r in rows}
        labels=sorted({k.rsplit('_q',1)[0] for k in physical})
        aliases=json.loads((src/(codec+'.plan.json')).read_text())['aliases']
        resolved={a:physical[r['cell']] for r in aliases if r['cell'] in physical for a in r['merged']}
        expanded=[];inactive=[]
        for label in labels:
            emitted=[r for k,r in physical.items() if k.rsplit('_q',1)[0]==label]
            for q in Q:
                key=label+'_q'+str(q)
                row=physical.get(key) or resolved.get(key)
                reason='native plan alias' if key not in physical else None
                if row is None:
                    # These explicit native lossless plans have no lossy-quality
                    # axis. Preserve all requested q values as inactive aliases.
                    if len(emitted)!=1 or not ((codec=='webp' and label.startswith('vp8l-')) or (codec=='jxl' and label.startswith('mod-'))):
                        raise ValueError('unresolved active native quality: '+key)
                    row=emitted[0];reason='native lossless quality axis inactive'
                    if label not in inactive:inactive.append(label)
                item=dict(row,requested_q=q,requested_mode=label,native_alias=reason)
                # Execute the native canonical configuration; retain the requested
                # group separately, since canonical aliases can cross labels.
                item['mode_key']=canonical(dict(cell=label,plan=json.loads(row['knob_tuple_json'])['plan']))
                expanded.append(item)
        with (out/(codec+'.jsonl')).open('x') as f:
            for r in expanded:f.write(canonical(r)+'\n')
        census[codec]=dict(labels=labels,physical=len(rows),requested=len(expanded),
            lossless_quality_inactive=inactive,physical_source_sha256=sha(src/(codec+'.jsonl')),
            aliases_source_sha256=sha(src/(codec+'.plan.json')))
    write(out/'coverage.json',dict(schema='e36-native-mode-expansion-v1',census=census,
        claim='Native aliases only; nonbaseline large-size effects are not tested'))


def freeze(root,initial,version,commit):
    out=root/version;out.mkdir()
    program=out/'program';program.mkdir()
    files=['e36_data.py','e36_store.py','e34_encode.py','e34_augmentation.py']
    for n in files:shutil.copyfile(Path(__file__).with_name(n),program/n)
    modes=out/'mode-inventory';expand_modes(initial,modes)
    oracles={n:str(Path(shutil.which(n)).resolve()) for n in ['djpeg','dwebp','ffmpeg','djxl']}
    spec=dict(schema='e36-native-data-runtime-v1',fits=0,root=str(root),build_commit=commit,
        native_tool=str(root/'bin/zenmetrics'),feature_tool=str(root/'bin/e36-pixels'),
        feature_spec=str(root/'e34-feature-spec.json'),native_pin=str(root/'NATIVE_TOOL_PIN.json'),
        feature_layout_sha256=sha(root/'FEATURE_LAYOUT_BINDING.json'),
        references=str(root/'references.json'),references_sha256=sha(root/'references.json'),
        storage=json.loads((root/'storage-runtime.json').read_text()),
        program_files={n:sha(program/n) for n in files},mode_inventory=str(modes),
        mode_inventory_sha256={n:sha(modes/(n+'.jsonl')) for n,_,_ in E36_CODECS},
        oracles=oracles,oracle_sha256={n:sha(p) for n,p in oracles.items()})
    write(out/'runtime.json',spec)
    launcher=out/'executor'
    with launcher.open('x') as f:f.write('#!/bin/sh\nexport PYTHONDONTWRITEBYTECODE=1\nexec '+shlex.join([sys.executable,str(program/'e36_data.py'),'--runtime',str(out/'runtime.json')])+' "$@"\n')
    launcher.chmod(0o755)
    print(canonical(dict(runtime=str(out/'runtime.json'),sha256=sha(out/'runtime.json'))))


def views(root,runtime,phase):
    spec=json.loads(runtime.read_text());rsha=sha(runtime)
    refs={r['reference_key']:r for r in json.loads((root/'references.json').read_text())['rows']}
    plan=json.loads((root/'source-budget/plan.json').read_text());out=[]
    def add(key,codec,qs,panel='baseline',role=None):
        ref=refs[key];native=next(c for n,c,_ in E36_CODECS if n==codec)
        view=dict(runtime_sha256=rsha,reference_key=key,codec=codec,q_grid=qs,panel=panel,
            role=role or ('bulk' if phase in ['fallback','additions'] else phase),deadline_seconds=14400)
        out.append(dict(view=PREFIX+canonical(view),
            cell=dict(image_path=ref['destination'],codec=native,q=-1,knob_tuple_json='{}'),
            inputs=[rsha,ref['sha256']],hint=dict(peak_mem_bytes=(4 if ref['size']>=1024 else 1)*1024**3,
                threads=1,vram_bytes=None),requires=[CAPABILITY]))
    if phase=='first-producer':
        for codec,_,_ in E36_CODECS:add(plan['panels']['cost'][0],codec,[0])
    elif phase=='cost':
        for key in plan['panels']['cost']:
            for codec,_,_ in E36_CODECS:add(key,codec,[0,1,20,60,100])
    elif phase=='square-witness':
        for key in plan['panels'][phase]:
            for codec,_,_ in E36_CODECS:add(key,codec,[0,60,100])
    elif phase=='geometry':
        for key in plan['panels']['32:fast']['full']:
            for codec,_,_ in E36_CODECS:add(key,codec,[0])
    elif phase=='mode-cost':
        for key in [plan['panels']['mode-panel']['64'][0],plan['panels']['mode-panel']['256'][0]]:
            for codec,_,_ in E36_CODECS:add(key,codec,[0,20,60,100],panel='mode-panel')
    elif phase in ['fallback','additions']:
        for k,p in plan['panels'].items():
            if ':' not in k:continue
            _,group=k.split(':')
            keys=p['fallback'] if phase=='fallback' else [x for x in p['full'] if x not in p['fallback']]
            for key in keys:
                for codec,_,_ in E36_CODECS:
                    if (codec=='rav1e')==(group=='rav1e'):add(key,codec,Q)
        if phase=='fallback':
            for keys in plan['panels']['mode-panel'].values():
                for key in keys:
                    for codec,_,_ in E36_CODECS:add(key,codec,Q,panel='mode-panel')
    else:raise ValueError(phase)
    return out


def declare(root,runtime,phase,name):
    out=root/'runs'/name;out.mkdir(parents=True)
    spec=out/'views.json';items=views(root,runtime,phase);write(spec,dict(schema=1,views=items))
    subprocess.run([str(root/'bin/zenfleet-ctl'),'declare-views','--spec',str(spec),'--out',str(out/'manifest.json')],check=True)
    write(out/'declaration.json',dict(phase=phase,runtime=str(runtime),runtime_sha256=sha(runtime),
        jobs=len(items),views_sha256=sha(spec),manifest_sha256=sha(out/'manifest.json'),fits=0))
    print(canonical(dict(run=str(out),jobs=len(items))))


def run(root,name,slots,hours):
    out=root/'runs'/name;d=json.loads((out/'declaration.json').read_text());runtime=Path(d['runtime'])
    if sha(runtime)!=d['runtime_sha256']:raise ValueError('runtime changed')
    env=dict(os.environ,ZEN_RAM_BYTES=str(24*1024**3),ZEN_CHUNK_WALL_SEC='1',
        ZEN_PERSISTENT_EXEC='1',ZEN_PERSISTENT_KINDS='bake',RAYON_NUM_THREADS='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if hours:
        deadline=time.time()+hours*3600;env['E36_GENERATION_DEADLINE']=str(deadline)
        write(out/'deadline.json',dict(started_unix=time.time(),deadline_unix=deadline,hours=hours))
    argv=['taskset','--cpu-list',f'0-{slots-1}',str(root/'bin/zenfleet-worker'),
        '--manifest',str(out/'manifest.json'),'--ledger-out',str(out/'ledger.parquet'),
        '--blobs',str(out/'blobs'),'--exec',str(runtime.parent/'executor'),'--worker','e36-dev',
        '--provider','local','--max-attempts','1']
    if hours:argv=['/usr/bin/timeout','--signal=TERM','--kill-after=120',str(hours*3600),*argv]
    started=time.monotonic();rc=subprocess.run(argv,env=env).returncode
    write(out/'pass.json',dict(returncode=rc,wall_seconds=time.monotonic()-started,fits=0))
    if rc:raise SystemExit(rc)


def census(root,name):
    out=root/'runs'/name;rows=[];files=[]
    for p in sorted((out/'blobs').rglob('*')):
        if not p.is_file():continue
        try:doc=json.loads(gzip.decompress(p.read_bytes()))
        except (ValueError,OSError):continue
        if doc.get('schema')!='e36-native-ladder-receipt-v1':continue
        if not doc['complete']:raise ValueError('incomplete output')
        rows.extend(doc['rows']);files.append(dict(path=str(p),sha256=sha(p)))
    groups=collections.defaultdict(list)
    for r in rows:groups[f"{r['size']}:{r['codec']}"] .append(r)
    stats={}
    import statistics
    for k,values in groups.items():
        physical=[r for r in values if not r['physical_alias']]
        stats[k]=dict(logical=len(values),physical=len(physical),
            stage_median_seconds={s:statistics.median(r[s]['wall_seconds'] for r in physical) for s in ['encode_execution','score_execution','feature_execution']},
            peak_native_rss_kib=max(r[s]['peak_rss_kib'] for r in physical for s in ['encode_execution','score_execution','feature_execution']),
            bundle_bytes=sum(r['bundle']['bytes'] for r in physical)) if physical else dict(logical=len(values),physical=0)
    result=dict(schema='e36-data-census-v1',fits=0,completed_ladders=len(files),logical_rows=len(rows),
        output_receipts=files,groups=stats)
    write(out/'census.json',result);print(canonical(result))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','declare','run','census'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--initial',type=Path)
    p.add_argument('--version');p.add_argument('--commit');p.add_argument('--runtime',type=Path)
    p.add_argument('--phase');p.add_argument('--name');p.add_argument('--slots',type=int,default=1)
    p.add_argument('--hours',type=float);a=p.parse_args()
    if a.mode=='freeze':freeze(a.root,a.initial,a.version,a.commit)
    elif a.mode=='declare':declare(a.root,a.runtime,a.phase,a.name)
    elif a.mode=='run':run(a.root,a.name,a.slots,a.hours)
    else:census(a.root,a.name)

if __name__=='__main__':main()
