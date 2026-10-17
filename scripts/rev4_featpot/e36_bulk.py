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


def freeze(root,initial,version,commit,references_path=None,budget_path=None):
    out=root/version;out.mkdir()
    program=out/'program';program.mkdir()
    files=['e36_data.py','e36_store.py','e34_encode.py','e34_augmentation.py']
    for n in files:shutil.copyfile(Path(__file__).with_name(n),program/n)
    modes=out/'mode-inventory';expand_modes(initial,modes)
    oracles={n:str(Path(shutil.which(n)).resolve()) for n in ['djpeg','dwebp','ffmpeg','djxl']}
    references_path=references_path or root/'references.json'
    budget_path=budget_path or root/'source-budget/plan.json'
    spec=dict(schema='e36-native-data-runtime-v1',fits=0,root=str(root),build_commit=commit,
        native_tool=str(root/'bin/zenmetrics'),feature_tool=str(root/'bin/e36-pixels'),
        feature_spec=str(root/'e34-feature-spec.json'),native_pin=str(root/'NATIVE_TOOL_PIN.json'),
        feature_layout_sha256=sha(root/'FEATURE_LAYOUT_BINDING.json'),
        references=str(references_path),references_sha256=sha(references_path),
        source_budget=str(budget_path),source_budget_sha256=sha(budget_path),
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
    refs={r['reference_key']:r for r in json.loads(Path(spec['references']).read_text())['rows']}
    plan=json.loads(Path(spec.get('source_budget',root/'source-budget/plan.json')).read_text());out=[]
    resources=json.loads((root/'resource-admission.json').read_text()) if phase in ['fallback','additions','full'] else None
    def add(key,codec,qs,panel='baseline',role=None):
        ref=refs[key];native=next(c for n,c,_ in E36_CODECS if n==codec)
        view=dict(runtime_sha256=rsha,reference_key=key,codec=codec,q_grid=qs,panel=panel,
            role=role or ('bulk' if phase in ['fallback','additions','full'] else phase),deadline_seconds=14400)
        hint_bytes=((4 if ref['size']>=1024 else 1)*1024**3 if resources is None else resources['mode_panel_bytes'] if panel=='mode-panel' else resources['by_size_bytes'][str(ref['size'])])
        out.append(dict(view=PREFIX+canonical(view),
            cell=dict(image_path=ref['destination'],codec=native,q=-1,knob_tuple_json='{}'),
            inputs=[rsha,ref['sha256']],hint=dict(peak_mem_bytes=hint_bytes,
                threads=1,vram_bytes=None),requires=[CAPABILITY]))
    if phase=='first-producer':
        for codec,_,_ in E36_CODECS:add(plan['panels']['cost'][0],codec,[0])
    elif phase=='cost':
        for key in plan['panels']['cost']:
            for codec,_,_ in E36_CODECS:add(key,codec,[0,1,20,60,100])
    elif phase=='square-witness':
        for key in plan['panels'][phase]:
            for codec,_,_ in E36_CODECS:add(key,codec,[0,60,100])
    elif phase=='max-area-witness':
        candidates=[r for r in refs.values() if r['size']==4096 and r['split'] in ['fit','selection']]
        ref=sorted(candidates,key=lambda r:(-r['width']*r['height'],r['origin']))[0]
        for codec,_,_ in E36_CODECS:add(ref['reference_key'],codec,[0,60,100])
    elif phase=='geometry':
        for key in plan['panels']['32:fast']['full']:
            for codec,_,_ in E36_CODECS:add(key,codec,[0])
    elif phase=='mode-cost':
        for key in [plan['panels']['mode-panel']['64'][0],plan['panels']['mode-panel']['256'][0]]:
            for codec,_,_ in E36_CODECS:add(key,codec,[0,20,60,100],panel='mode-panel')
    elif phase in ['fallback','additions','full']:
        for k,p in plan['panels'].items():
            if ':' not in k:continue
            _,group=k.split(':')
            keys=p['fallback'] if phase=='fallback' else p['full'] if phase=='full' else [x for x in p['full'] if x not in p['fallback']]
            for key in keys:
                for codec,_,_ in E36_CODECS:
                    if (codec=='rav1e')==(group=='rav1e'):add(key,codec,Q)
        if phase in ['fallback','full']:
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


def run(root,name,slots,hours=None,deadline=None):
    handoff=root/'TOWER_PHASE_HANDOFF.json'
    if handoff.exists() and json.loads(handoff.read_text())['run']==name:
        from e36_phase_handoff import wait_and_harvest
        return wait_and_harvest(root,name,deadline,handoff)
    out=root/'runs'/name;d=json.loads((out/'declaration.json').read_text());runtime=Path(d['runtime'])
    if sha(runtime)!=d['runtime_sha256']:raise ValueError('runtime changed')
    env=dict(os.environ,ZEN_RAM_BYTES=str(24*1024**3),ZEN_CHUNK_WALL_SEC='300',
        ZEN_CHUNK_REF_CONCURRENCY=str(slots),ZEN_CHUNK_TAIL_SPREAD='1',ZEN_DEBUG_ADMIT='1',
        ZEN_PERSISTENT_EXEC='1',ZEN_PERSISTENT_KINDS='bake',RAYON_NUM_THREADS=str(slots),
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    if hours:
        if deadline is not None:raise ValueError('two wall clocks are forbidden')
        deadline=time.time()+hours*3600
    if deadline is not None:
        if deadline<=time.time():raise TimeoutError('shared bulk wall budget expired')
        env['E36_GENERATION_DEADLINE']=str(deadline)
    argv=['taskset','--cpu-list',f'0-{slots-1}',str(root/'bin/zenfleet-worker'),
        '--manifest',str(out/'manifest.json'),'--ledger-out',str(out/'ledger.parquet'),
        '--blobs',str(out/'blobs'),'--exec',str(runtime.parent/'executor'),'--worker','e36-dev',
        '--provider','local','--max-attempts','1']
    for ledger in sorted(out.glob('ledger*.parquet')):
        argv.extend(['--ledger-in',str(ledger)])
    if deadline is not None:argv=['/usr/bin/timeout','--signal=TERM','--kill-after=120',str(max(0.001,deadline-time.time())),*argv]
    started=time.monotonic();rc=subprocess.run(argv,env=env).returncode
    peak=None
    cg=Path('/proc/self/cgroup')
    if cg.exists():
        line=next((x for x in cg.read_text().splitlines() if x.startswith('0::')),None)
        if line:
            path=Path('/sys/fs/cgroup')/line[3:].lstrip('/')/'memory.peak'
            if path.exists():peak=int(path.read_text())
    write(out/('pass-'+str(time.time_ns())+'.json'),dict(returncode=rc,wall_seconds=time.monotonic()-started,fits=0,
        worker_rayon_threads=slots,native_kernel_threads=1,scope_memory_peak_bytes_so_far=peak,
        resource_scope='cgroup aggregate memory.peak, including earlier stages in the same run-heavy scope; not process RSS'))
    if rc:raise SystemExit(rc)


def census(root,name):
    out=root/'runs'/name;rows=[];files=[];completed=[]
    for p in sorted((out/'blobs').rglob('*')):
        if not p.is_file():continue
        try:doc=json.loads(gzip.decompress(p.read_bytes()))
        except (ValueError,OSError):continue
        if doc.get('schema')!='e36-native-ladder-receipt-v1':continue
        if not doc['complete']:raise ValueError('incomplete output')
        rows.extend(doc['rows']);files.append(dict(path=str(p),sha256=sha(p)))
        completed.append(PREFIX+canonical(doc['view']))
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
        if physical and all('cell_total_seconds' in r for r in physical):
            stats[k].update(total_median_seconds=statistics.median(r['cell_total_seconds'] for r in physical),
                total_sum_seconds=sum(r['cell_total_seconds'] for r in physical),
                persistence_sum_seconds=sum(r['artifact_persistence_seconds'] for r in physical),
                map_sum_seconds=sum(m['execution']['wall_seconds'] for r in physical for m in r['maps']),
                peak_native_rss_kib=max([r[s]['peak_rss_kib'] for r in physical
                    for s in ['encode_execution','score_execution','feature_execution','oracle_execution']]
                    +[m['execution']['peak_rss_kib'] for r in physical for m in r['maps']]))
    expected=[j['kind']['view'] for j in json.loads((out/'manifest.json').read_text())]
    admitted=len(completed)==len(set(completed)) and set(completed)==set(expected)
    result=dict(schema='e36-data-census-v1',fits=0,completed_ladders=len(files),logical_rows=len(rows),
        output_receipts=files,groups=stats,admitted=admitted,declared_ladders=len(expected))
    write(out/'census.json',result);print(canonical({k:result[k] for k in ['admitted','completed_ladders','declared_ladders','logical_rows','fits']}))
    if not admitted:raise ValueError('native ledger outputs do not complete the declared matrix')



def resource_admission(root):
    cost=root/'MEASURED_STAGE_COSTS-v2.json';rows=json.loads(cost.read_text())['rows']
    maxrun=root/'runs/max-area-v4/census.json';maxdoc=json.loads(maxrun.read_text())
    if not maxdoc['admitted']:raise ValueError('actual largest-area witness incomplete')
    # Explicit coarse reservations, never reported as measured upper bounds.
    by_size={str(n):(512*1024**2 if n<=768 else 1024**3 if n<=1536 else 1536*1024**2 if n==2048 else 4*1024**3)
        for n in [32,40,48,64,80,96,128,160,192,256,320,384,512,640,768,1024,1280,1536,2048,4096]}
    maxima={}
    for r in rows:
        k=str(r['size']);maxima[k]=max(maxima.get(k,0),*(s['peak_rss_kib'] for s in r['stages'].values()))
    for k,v in maxdoc['groups'].items():
        size=k.split(':')[0];maxima[size]=max(maxima.get(size,0),v['peak_native_rss_kib'])
    for size,observed in maxima.items():
        if observed*1024+256*1024**2>=by_size[size]:raise ValueError('coarse reservation lacks the declared headroom over observed native peak')
    write(root/'resource-admission.json',dict(schema='e36-data-resource-admission-v1',by_size_bytes=by_size,
        mode_panel_bytes=2*1024**3,hard_scope_bytes=24*1024**3,worker_ram_admission_bytes=18*1024**3,
        cpus=list(range(28)),native_kernel_threads=1,observed_native_peak_rss_kib=maxima,
        measured_cost_sha256=sha(cost),max_area_census_sha256=sha(maxrun),
        qualification='Coarse configured reservations from bounded preflight, not measured bounds for every image. Native GNU time-v peaks are per invocation; hard scope stays24GiB.',fits=0))
    print(canonical(dict(resources=str(root/'resource-admission.json'),fits=0)))


def prepare_bulk(root,runtime,tag,plan_path=None):
    for phase in ['fallback','additions','full']:
        declare(root,runtime,phase,phase+'-'+tag)
    records={};ids={}
    for phase in ['fallback','additions','full']:
        out=root/'runs'/(phase+'-'+tag)
        result=subprocess.run([str(root/'bin/zenfleet-ctl'),'ids','--manifest',str(out/'manifest.json')],capture_output=True,check=True,text=True)
        (out/'native-ids.tsv').write_text(result.stdout)
        ids[phase]={line.split('\t')[1] for line in result.stdout.splitlines() if line and not line.startswith('idx\t')}
        records[phase]=dict(run=phase+'-'+tag,manifest_sha256=sha(out/'manifest.json'),native_ids_sha256=sha(out/'native-ids.tsv'),jobs=len(ids[phase]))
    if ids['fallback']&ids['additions'] or ids['fallback']|ids['additions']!=ids['full']:
        raise ValueError('native augmented manifest does not equal the disjoint mandatory-plus-additions union')
    write(plan_path or root/'bulk-plan.json',dict(schema='e36-six-hour-data-plan-v1',runtime=str(runtime),runtime_sha256=sha(runtime),
        resources_sha256=sha(root/'resource-admission.json'),source_budget_sha256=json.loads(runtime.read_text())['source_budget_sha256'],
        phases=records,deadline_seconds=21600,slots=28,fit_release=False,
        selection='full only if complete inside the same wall budget; otherwise exactly complete fallback; no intermediate matrix'))
    print(canonical(dict(qualifying_manifests=records,fits=0)))


def qualify(root,name,start,deadline):
    out=root/'runs'/name
    try:
        if not (out/'census.json').exists():census(root,name)
        doc=json.loads((out/'census.json').read_text())
        if not doc['admitted']:return False
        for r in doc['output_receipts']:
            p=Path(r['path'])
            if not start<=p.stat().st_mtime<=deadline:return False
            d=json.loads(gzip.decompress(p.read_bytes()))
            if not start<=d['completed_unix']<=deadline:return False
        return True
    except (ValueError,KeyError):return False


def clock_equivalence(root,plan_path,plan,inherit_clock):
    original=json.loads(inherit_clock.read_text());old_plan=root/'bulk-plan.json';old=json.loads(old_plan.read_text())
    if original['plan_sha256']!=sha(old_plan) or original['deadline_unix']!=original['started_unix']+21600:raise ValueError('invalid original deadline chain')
    if any(old[k]!=plan[k] for k in ['resources_sha256','source_budget_sha256','deadline_seconds','slots','fit_release']):raise ValueError('transport repair changed scientific/resource budget')
    if sha(old['runtime'])!=old['runtime_sha256'] or sha(plan['runtime'])!=plan['runtime_sha256']:raise ValueError('original/new runtime pin changed')
    old_runtime=json.loads(Path(old['runtime']).read_text());new_runtime=json.loads(Path(plan['runtime']).read_text())
    if set(old_runtime)!=set(new_runtime) or set(old_runtime['program_files'])!=set(new_runtime['program_files']) or set(old['phases'])!=set(plan['phases']):raise ValueError('transport repair changed runtime/program/phase schema')
    if any(old_runtime[k]!=new_runtime[k] for k in old_runtime if k not in ['build_commit','program_files','mode_inventory']):raise ValueError('transport repair changed native numerical/source inputs')
    if any(old_runtime['program_files'][k]!=v for k,v in new_runtime['program_files'].items() if k not in ['e36_data.py','e36_store.py']):raise ValueError('transport repair changed another producer module')
    bind=lambda p:dict(path=str(p),sha256=sha(p))
    manifests={}
    for phase in old['phases']:
        def science(entry,runtime_sha):
            path=root/'runs'/entry['run']/'manifest.json'
            if sha(path)!=entry['manifest_sha256']:raise ValueError('original/new phase manifest pin changed')
            rows=json.loads(path.read_text());values=[]
            for r in rows:
                v=json.loads(r['kind']['view'][len(PREFIX):])
                if v.pop('runtime_sha256')!=runtime_sha or runtime_sha not in r['inputs']:raise ValueError('view/runtime input binding differs')
                r['kind']['view']=PREFIX+canonical(v);r['inputs']=['RUNTIME' if x==runtime_sha else x for x in r['inputs']]
                values.append(canonical(r))
            return sorted(values),bind(path)
        old_values,old_bind=science(old['phases'][phase],old['runtime_sha256']);new_values,new_bind=science(plan['phases'][phase],plan['runtime_sha256'])
        if old_values!=new_values:raise ValueError('transport repair changed fixed matrix/inputs/hints/capabilities')
        manifests[phase]=dict(original=old_bind,repaired=new_bind,normalized_views_sha256=hashlib.sha256(canonical(old_values).encode()).hexdigest())
    return dict(schema='e36-original-clock-equivalence-v1',admitted=True,original_clock=bind(inherit_clock),
        original_plan=bind(old_plan),repaired_plan=bind(plan_path),original_runtime=bind(old['runtime']),repaired_runtime=bind(plan['runtime']),
        manifests=manifests,started_unix=original['started_unix'],deadline_unix=original['deadline_unix'],
        owner_sha256=sha(Path(__file__)),allowed_changes=['e36_data storage startup/error framing','e36_store client transport','version-location fields'],fits=0)


def bulk(root,plan_path=None,clock_path=None,inherit_clock=None):
    plan_path=plan_path or root/'bulk-plan.json'
    plan=json.loads(plan_path.read_text())
    if plan['fit_release'] is not False or plan['deadline_seconds']!=21600:raise ValueError('wrong pre-fit data plan')
    if sha(plan['runtime'])!=plan['runtime_sha256'] or sha(root/'resource-admission.json')!=plan['resources_sha256']:raise ValueError('frozen bulk admission changed')
    for d in plan['phases'].values():
        if sha(root/'runs'/d['run']/'manifest.json')!=d['manifest_sha256']:raise ValueError('frozen qualifying manifest changed')
    clock=clock_path or root/'bulk-clock.json'
    from e36_pause_budget import resolve
    base_times,times,_=resolve(clock) if clock.exists() else (None,None,[])
    inherited=inherit_clock or (Path(times['inherited_clock']['path']) if times and 'inherited_clock' in times else None)
    if plan_path!=root/'bulk-plan.json' and inherited is None:raise ValueError('an amended plan must inherit the original clock; a fresh window is forbidden')
    equivalence=None
    if inherited is not None:
        equivalence=clock_equivalence(root,plan_path,plan,inherited)
        ep=clock.with_name(clock.stem+'-equivalence.json')
        if ep.exists():
            if json.loads(ep.read_text())!=equivalence:raise ValueError('immutable original-clock equivalence changed')
        else:write(ep,equivalence)
    if times is not None:
        if times['plan_sha256']!=sha(plan_path):raise ValueError('cannot replace a running wall clock')
        if equivalence is not None and (base_times.get('inherited_clock')!=equivalence['original_clock'] or base_times.get('equivalence')!=dict(path=str(ep),sha256=sha(ep)) or any(base_times[k]!=equivalence[k] for k in ['started_unix','deadline_unix'])):raise ValueError('resumed repair changed original clock or equivalence')
    else:
        start=time.time() if equivalence is None else equivalence['started_unix']
        times=dict(started_unix=start,deadline_unix=start+21600,plan_sha256=sha(plan_path),fits=0)
        if equivalence is not None:times.update(inherited_clock=equivalence['original_clock'],equivalence=dict(path=str(ep),sha256=sha(ep)),amendment='storage transport only; original window unchanged')
        write(clock,times)
    start,deadline=times['started_unix'],times['deadline_unix'];qualified=None;errors=[]
    mandatory=plan['phases']['fallback']['run'];extra=plan['phases']['additions']['run']
    try:run(root,mandatory,plan['slots'],deadline=deadline)
    except (SystemExit,TimeoutError) as exc:errors.append(str(exc))
    if qualify(root,mandatory,start,deadline):
        qualified='fallback'
        if (root/'TOWER_PHASE_HANDOFF.json').exists():
            from e36_phase_handoff import wait_and_harvest
            wait_and_harvest(root,extra,deadline,root/'TOWER_PHASE_HANDOFF.json')
        elif time.time()<deadline:
            try:run(root,extra,plan['slots'],deadline=deadline)
            except (SystemExit,TimeoutError) as exc:errors.append(str(exc))
        # Completed native evidence is qualified by its original timestamps,
        # independently of when control or evidence transport returns.
        if qualify(root,extra,start,deadline):qualified='full'
    choice=dict(schema='e36-fixed-matrix-choice-v1',qualified=qualified,clock_sha256=sha(clock),fits=0,
        plan_sha256=sha(plan_path),plan_path=str(plan_path),clock_path=str(clock),errors=errors,closed_unix=time.time(),
        selected_runs=([mandatory,extra] if qualified=='full' else [mandatory] if qualified=='fallback' else []),
        teacher_support_not_used_for_choice=True)
    write(root/'bulk-choice.json',choice);print(canonical(choice))
    if qualified is None:raise ValueError('mandatory matrix incomplete in registered window; no fit pool qualifies')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['freeze','declare','run','census','resources','prepare-bulk','bulk'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--initial',type=Path)
    p.add_argument('--version');p.add_argument('--commit');p.add_argument('--runtime',type=Path)
    p.add_argument('--references',type=Path);p.add_argument('--budget',type=Path)
    p.add_argument('--phase');p.add_argument('--name');p.add_argument('--slots',type=int,default=1)
    p.add_argument('--plan',type=Path);p.add_argument('--clock',type=Path);p.add_argument('--inherit-clock',type=Path)
    p.add_argument('--hours',type=float);a=p.parse_args()
    if a.mode=='freeze':freeze(a.root,a.initial,a.version,a.commit,a.references,a.budget)
    elif a.mode=='declare':declare(a.root,a.runtime,a.phase,a.name)
    elif a.mode=='run':run(a.root,a.name,a.slots,a.hours)
    elif a.mode=='census':census(a.root,a.name)
    elif a.mode=='resources':resource_admission(a.root)
    elif a.mode=='prepare-bulk':prepare_bulk(a.root,a.runtime,a.version,a.plan)
    else:bulk(a.root,a.plan,a.clock,a.inherit_clock)

if __name__=='__main__':main()
