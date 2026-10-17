"""Strict owner-authorized rav1e subtraction from the original native matrix."""
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

import pyarrow.parquet as pq

from e36_data import PREFIX, canonical
from e36_store import sha


def bind(path):
    return dict(path=str(path), sha256=sha(path))


def checked(pin):
    path = Path(pin['path'])
    if sha(path) != pin['sha256']:
        raise ValueError('SVT owner evidence changed: ' + str(path))
    return path


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def view(job):
    return json.loads(job['kind']['view'][len(PREFIX):])


def prepare(root):
    root = Path(root)
    out = root/'owner-svt-ssd-v1'
    original = root/'bulk-plan-v6.json'
    old = json.loads(original.read_text())
    plan = copy.deepcopy(old)
    phases = {}
    for phase, entry in old['phases'].items():
        source = root/'runs'/entry['run']/'manifest.json'
        if sha(source) != entry['manifest_sha256']:
            raise ValueError('original phase changed')
        jobs = json.loads(source.read_text())
        filtered = [job for job in jobs if view(job)['codec'] != 'rav1e']
        name = 'svt-'+phase+'-v1'
        directory = root/'runs'/name
        directory.mkdir()
        manifest = directory/'manifest.json'
        write(manifest, filtered)
        declaration = json.loads((source.parent/'declaration.json').read_text())
        declaration.update(phase=phase, jobs=len(filtered), manifest_sha256=sha(manifest),
                           owner_amendment=bind(out/'amendment.json'))
        # Do not carry a view-spec hash describing a different manifest.
        declaration.pop('views_sha256', None)
        write(directory/'declaration.json', declaration)
        ids = directory/'native.ids.tsv'
        with ids.open('x') as stream:
            subprocess.run([str(root/'bin/zenfleet-ctl'),'ids','--manifest',str(manifest)], stdout=stream, check=True)
        schedule = out/(phase+'-scheduled.json')
        if phase in ['fallback','additions']:
            scheduled = json.loads((root/'scheduling-v7'/(phase+'.json')).read_text())
            selected = [job for job in scheduled if view(job)['codec'] != 'rav1e']
            if len(selected) != len(filtered) or any({k:v for k,v in a.items() if k!='hint'} != {k:v for k,v in b.items() if k!='hint'} for a,b in zip(selected,filtered)):
                raise ValueError('filtered scheduler changed scientific identity')
            write(schedule, selected)
        phase_pin = dict(run=name, original=bind(source), manifest=bind(manifest),
                         native_ids=bind(ids), original_native_ids=bind(source.parent/'native-ids.tsv'),
                         old_run=entry['run'], jobs=len(filtered))
        if schedule.exists():
            phase_pin['scheduled'] = bind(schedule)
        phases[phase] = phase_pin
        plan['phases'][phase] = dict(entry, run=name, manifest_sha256=sha(manifest), native_ids_sha256=sha(ids), jobs=len(filtered))
    plan_path = out/'bulk-plan.json'
    write(plan_path, plan)
    proof = dict(schema='e36-svt-only-native-matrix-v1', original_plan=bind(original),
                 plan=bind(plan_path), phases=phases, owner_amendment=bind(out/'amendment.json'),
                 overlay=bind(out/'e36_owner_amendment.py'), cutoff=bind(out/'live-cutoff.json'),
                 clock=bind(root/'bulk-clock-v7-budget3.json'),
                 scheduling=bind(root/'scheduling-v8/amendment.json'),
                 excluded_codecs=['rav1e'], preference_weights={'jpeg':2,'svt':1,'webp':1,'jxl':1},
                 fits=0, fit_release=False)
    write(out/'matrix-binding.json', proof)
    return proof


def configuration(root):
    root = Path(root)
    proof_path = root/'owner-svt-ssd-v1/matrix-binding.json'
    proof = json.loads(proof_path.read_text())
    if proof['schema']!='e36-svt-only-native-matrix-v1' or proof['fits']!=0 or proof['fit_release'] is not False or proof['excluded_codecs']!=['rav1e'] or proof['preference_weights']!={'jpeg':2,'svt':1,'webp':1,'jxl':1}:
        raise ValueError('exact registered owner amendment required')
    for key in ['original_plan','plan','owner_amendment','overlay','cutoff','clock','scheduling']:
        checked(proof[key])
    for pin in proof.get('resource_bootstrap',{}).values():checked(pin)
    extension_path=root/'owner-svt-ssd-v1/resource-closure-v3.json'
    if extension_path.exists():
        extension=json.loads(extension_path.read_text())
        if extension['schema']!='e36-owner-resource-closure-v3' or extension['scientific_change'] is not False or extension['fits']!=0 or checked(extension['matrix_binding'])!=proof_path:
            raise ValueError('resource closure changed the scientific matrix')
        for key in ['controller_binding','controller_environment','cutoff']:checked(extension[key])
        for pin in extension['files'].values():checked(pin)
    urgent_path=root/'owner-svt-ssd-v1/urgent-mirror-v10-binding.json'
    if urgent_path.exists():
        urgent=json.loads(urgent_path.read_text())
        restart=json.loads(checked(urgent['previous']).read_text())
        first=json.loads(checked(restart['previous']).read_text())
        policy=json.loads(checked(first['policy']).read_text())
        if (policy['authorization']!='coordinator-urgent-disk-2026-10-17'
            or [policy[k] for k in ['pause_below_bytes','resume_above_bytes','stop_below_bytes']]!=[30*1024**3,40*1024**3,20*1024**3]
            or policy['keep_inflight'] is not True or urgent['max_storage_transfers']!=4):
            raise ValueError('exact coordinator disk policy differs')
        for value in [urgent,restart,first]:
            if value['fits']!=0 or value['scientific_change'] is not False:
                raise ValueError('urgent disk operation changed science')
            for field in ['cutoff','controller_environment','monitor','mirror','monitor_source']:
                if field in value:checked(value[field])
            for field in ['files','mirror_files']:
                for pin in value.get(field,{}).values():checked(pin)
        checked(first['previous'])
    supervision=root/'owner-svt-ssd-v1/resource-supervision-v11.json'
    if supervision.exists():
        latest=json.loads(supervision.read_text())
        if latest['schema']!='e36-reviewed-resource-closure-v11' or latest['fits']!=0 or latest['scientific_change'] is not False or checked(latest['previous'])!=urgent_path:
            raise ValueError('resource supervision changed scientific inputs')
        for key in ['cutoff','controller_environment','monitor','mirror']:checked(latest[key])
        for key in ['files','mirror_files']:
            for pin in latest[key].values():checked(pin)
    monitor_restart=root/'owner-svt-ssd-v1/monitor-restart-binding.json'
    if monitor_restart.exists():
        restarted=json.loads(monitor_restart.read_text())
        if (restarted['schema']!='e36-actual-owned-pause-restart-v1'
            or restarted['fits']!=0 or restarted['scientific_change'] is not False
            or checked(restarted['previous'])!=supervision):
            raise ValueError('monitor restart changed resource evidence owner')
        for key in ['before','source','monitor']:checked(restarted[key])
    def schedule_chain(pin):
        a=json.loads(checked(pin).read_text())
        for key in ['plan','effective_clock','worker']:checked(a[key])
        for entry in a['manifests'].values():
            checked(entry);checked(entry['original'])
        for p in a.get('resource_bootstrap',{}).values():checked(p)
        if 'previous_scheduling_amendment' in a:
            previous=json.loads(checked(a['previous_scheduling_amendment']).read_text())
            changes={'effective_clock','deadline_changed','disk_floor_bytes','resource_bootstrap','previous_scheduling_amendment'}
            if {k:v for k,v in a.items() if k not in changes}!={k:v for k,v in previous.items() if k not in changes}:
                raise ValueError('resource scheduling predecessor changed scientific inputs')
            schedule_chain(a['previous_scheduling_amendment'])
    schedule_chain(proof['scheduling'])
    old=json.loads(checked(proof['original_plan']).read_text())
    new=json.loads(checked(proof['plan']).read_text())
    if {k:v for k,v in old.items() if k!='phases'}!={k:v for k,v in new.items() if k!='phases'}:
        raise ValueError('SVT amendment changed another scientific axis')
    all_views={}
    for phase,pin in proof['phases'].items():
        original=checked(pin['original']); manifest=checked(pin['manifest']);checked(pin['native_ids'])
        if original!=root/'runs'/old['phases'][phase]['run']/'manifest.json' or sha(original)!=old['phases'][phase]['manifest_sha256']:
            raise ValueError('owner filter original does not match frozen plan')
        original_jobs=json.loads(original.read_text())
        selected=[j for j in original_jobs if view(j)['codec']!='rav1e']
        jobs=json.loads(manifest.read_text())
        if jobs!=selected or len(jobs)!=pin['jobs'] or sha(manifest)!=new['phases'][phase]['manifest_sha256'] or new['phases'][phase]['run']!=pin['run']:
            raise ValueError('owner matrix is not exact rav1e subtraction')
        all_views[phase]={j['kind']['view'] for j in jobs}
        if len(all_views[phase])!=len(jobs):
            raise ValueError('duplicate filtered view')
        original_ids_path=checked(pin['original_native_ids'])
        if sha(original_ids_path)!=old['phases'][phase]['native_ids_sha256']:
            raise ValueError('original native ID owner differs')
        original_ids=[line.split('\t')[1] for line in original_ids_path.read_text().splitlines()]
        filtered_ids=[line.split('\t')[1] for line in checked(pin['native_ids']).read_text().splitlines()]
        if len(original_ids)!=len(original_jobs) or len(set(original_ids))!=len(original_ids) or filtered_ids!=[i for i,j in zip(original_ids,original_jobs) if view(j)['codec']!='rav1e']:
            raise ValueError('native original-to-filtered ID mapping differs')
        if 'scheduled' in pin:
            scheduled=json.loads(checked(pin['scheduled']).read_text())
            if len(scheduled)!=len(jobs) or any({k:v for k,v in a.items() if k!='hint'}!={k:v for k,v in b.items() if k!='hint'} for a,b in zip(scheduled,jobs)):
                raise ValueError('owner scheduling changed a scientific view')
    if all_views['fallback'] & all_views['additions'] or all_views['fallback'] | all_views['additions'] != all_views['full']:
        raise ValueError('filtered native matrix union changed')
    resolver_path=Path(__file__).with_name('e36_pause_budget.py')
    resolver_spec=importlib.util.spec_from_file_location('e36_svt_clock_resolver',resolver_path)
    resolver=importlib.util.module_from_spec(resolver_spec);resolver_spec.loader.exec_module(resolver)
    base,clock,evidence=resolver.resolve(checked(proof['clock']))
    if clock['plan_sha256']!=proof['original_plan']['sha256']:
        raise ValueError('owner matrix lost original clock-plan binding')
    if 'inherited_clock' in base:
        ep=checked(base['equivalence'])
        frozen=json.loads(ep.read_text())
        owners=[p for p in root.glob('control*/e36_bulk.py') if sha(p)==frozen['owner_sha256']]
        if not owners:
            raise ValueError('original hash-bound transport owner missing')
        spec=importlib.util.spec_from_file_location('e36_original_clock_owner',owners[0])
        owner=importlib.util.module_from_spec(spec);spec.loader.exec_module(owner)
        if frozen!=owner.clock_equivalence(root,checked(proof['original_plan']),old,checked(base['inherited_clock'])):
            raise ValueError('original numerical transport equivalence changed')
    return proof,clock,evidence


def publication(root, name, completed=None, files=None):
    root=Path(root);proof,clock,evidence=configuration(root)
    pin=next(p for p in proof['phases'].values() if p['run']==name)
    jobs=json.loads(checked(pin['manifest']).read_text())
    ids=[line.split('\t')[1] for line in checked(pin['native_ids']).read_text().splitlines()]
    if len(ids)!=len(jobs) or len(set(ids))!=len(ids):
        raise ValueError('filtered native IDs missing/duplicate')
    desired=dict(zip(ids,jobs));published={};ledgers=[]
    old_jobs=json.loads(checked(pin['original']).read_text())
    # Every nonselected row must still be a known frozen rav1e identity. Its
    # native kind is matched; only its completion requirement is withdrawn.
    old_ids=checked(pin['original_native_ids'])
    universe=dict(zip([line.split('\t')[1] for line in old_ids.read_text().splitlines()],old_jobs))
    directories=[root/'runs'/pin['old_run'],root/'runs'/name]
    for path in sorted({p for directory in directories for p in directory.glob('ledger*.parquet')}):
        for batch in pq.ParquetFile(path).iter_batches(columns=['job_id','status','attempts','output_sha','kind_json'],batch_size=256):
            for row in batch.to_pylist():
                job_id=row['job_id']
                if job_id not in universe or json.loads(row['kind_json'])!=universe[job_id]['kind']:
                    raise ValueError('unknown/changed native row in owner matrix')
                if job_id not in desired:
                    if view(universe[job_id])['codec']!='rav1e':
                        raise ValueError('owner matrix omits a required native row')
                    continue
                if row['status']!='done' or row['attempts']!=1 or not row['output_sha']:
                    raise ValueError('required SVT matrix native publication failed')
                prior=published.setdefault(job_id,row['output_sha'])
                if prior!=row['output_sha']:
                    raise ValueError('conflicting native output')
        ledgers.append(bind(path))
    if set(published)!=set(desired):
        raise ValueError('required SVT matrix native publication incomplete')
    for log in [root/'GENERATION-v7.log']:
        if any('ledger write' in line and 'failed' in line for line in log.read_text().splitlines()):
            raise ValueError('publication failure needs explicit reconciliation')
    if completed is None:
        return published,desired,ledgers
    by_view={j['kind']['view']:i for i,j in desired.items()}
    if len(completed)!=len(files) or len(completed)!=len(set(completed)) or set(completed)!=set(by_view):
        raise ValueError('filtered census/native matrix differs')
    if any(published[by_view[v]]!=f['sha256'] for v,f in zip(completed,files)):
        raise ValueError('filtered census output differs from native publication')
    result=dict(schema='e36-owner-filtered-native-publication-v1',admitted=True,jobs=len(desired),
                published=len(published),native_ledger_attempts_per_selected_job=1,ledgers=ledgers,
                matrix_binding=bind(root/'owner-svt-ssd-v1/matrix-binding.json'),native_ids=pin['native_ids'],
                census_output_inventory_sha256=hashlib.sha256(canonical(files).encode()).hexdigest(),fits=0)
    path=root/'runs'/name/('native-publication-'+str(time.time_ns())+'.json')
    write(path,result)
    return bind(path)


def qualify(root,name,start,deadline):
    root=Path(root);proof,clock,_=configuration(root)
    if start!=clock['started_unix'] or deadline!=clock['deadline_unix']:
        raise ValueError('controller changed registered generation clock')
    try:
        published,jobs,_=publication(root,name)
    except ValueError as error:
        print('E36 owner matrix not qualified:',error,flush=True)
        return False
    pin=next(p for p in proof['phases'].values() if p['run']==name)
    directory=root/'runs'/name;blobs=directory/'blobs';blobs.mkdir(exist_ok=True)
    for digest in published.values():
        candidates=[root/'runs'/pin['old_run']/'blobs'/digest,blobs/digest]
        source=next((p for p in candidates if p.exists()),None)
        if source is None or sha(source)!=digest:
            raise ValueError('required native CAS output missing/changed')
        target=blobs/digest
        if not target.exists():
            os.link(source,target)
        doc=json.loads(gzip.decompress(target.read_bytes()))
        if not start<=doc['completed_unix']<=deadline or not start<=target.stat().st_mtime<=deadline:
            raise ValueError('required native output outside original amended clock')
    census=directory/'census.json'
    if census.exists():
        census.rename(directory/('census.retained-'+str(time.time_ns())+'.json'))
    spec=importlib.util.spec_from_file_location('e36_svt_frozen_census',Path(__file__).with_name('e36_census_stream.py'))
    census_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(census_module)
    census_module.census(root,name,publication_admitter=publication)
    return True


def admit(root,names):
    root=Path(root);proof,clock,evidence=configuration(root)
    choice_path=root/'bulk-choice.json';choice=json.loads(choice_path.read_text())
    if choice['schema']!='e36-fixed-matrix-choice-v1' or choice['qualified'] not in ['fallback','full'] or choice['teacher_support_not_used_for_choice'] is not True:
        raise ValueError('qualified owner matrix choice required')
    if choice['plan_sha256']!=proof['plan']['sha256'] or choice['clock_sha256']!=proof['clock']['sha256']:
        raise ValueError('owner choice plan/clock differs')
    expected=[proof['phases']['fallback']['run']]
    if choice['qualified']=='full':
        expected.append(proof['phases']['additions']['run'])
    if names!=expected or choice['selected_runs']!=expected:
        raise ValueError('exact owner-selected native run list required')
    output_hashes=[];declarations={};seen=set();publications=[]
    for name in names:
        directory=root/'runs'/name;pin=next(p for p in proof['phases'].values() if p['run']==name)
        c=json.loads((directory/'census.json').read_text())
        if c['admitted'] is not True:
            raise ValueError('owner matrix incomplete')
        files=c['output_receipts'];views=[]
        for item in files:
            path=checked(item);doc=json.loads(gzip.decompress(path.read_bytes()))
            v=PREFIX+canonical(doc['view'])
            if not doc['complete'] or doc['view']['codec']=='rav1e' or v in seen or not clock['started_unix']<=doc['completed_unix']<=clock['deadline_unix'] or not clock['started_unix']<=path.stat().st_mtime<=clock['deadline_unix']:
                raise ValueError('duplicate/rav1e/incomplete/out-of-window owner output')
            seen.add(v);views.append(v);output_hashes.append(item['sha256'])
        publications.append(publication(root,name,views,files))
        declarations[name]=pin['manifest']
    full_expected={j['kind']['view'] for j in json.loads(checked(proof['phases'][choice['qualified']]['manifest']).read_text())}
    if seen!=full_expected:
        raise ValueError('missing/extra owner native views')
    location_path=root/'ARTIFACT_LOCATION_ADMISSION.json'
    locations=json.loads(location_path.read_text())
    if locations['schema']!='e36-artifact-location-admission-v1' or locations['admitted'] is not True or locations['fits']!=0 or locations['relocation_eligibility_replayed'] is not True:
        raise ValueError('frozen verified artifact location admission required')
    for key in ['journal','resolver','storage']:checked(locations[key])
    if locations['unique_relocated_objects'] and len(locations['actual_random_readbacks'])!=min(3,locations['unique_relocated_objects']):
        raise ValueError('final archive readback sample missing')
    return dict(schema='e36-qualified-matrix-admission-v1',matrix=choice['qualified'],choice=bind(choice_path),
                plan=proof['plan'],clock=proof['clock'],selected_runs=names,manifests=declarations,
                completed_views=len(seen),outputs=output_hashes,complete=True,budget_evidence=evidence,
                remote_phase_evidence=[],scheduling_evidence=[proof['scheduling']],
                owner_amendment=proof['owner_amendment'],owner_matrix=bind(root/'owner-svt-ssd-v1/matrix-binding.json'),
                excluded_codecs=['rav1e'],native_publication=publications,artifact_location=bind(location_path),
                resource_closure=bind(root/'owner-svt-ssd-v1/resource-closure-v3.json'),
                urgent_disk=bind(root/'owner-svt-ssd-v1/urgent-mirror-v10-binding.json') if (root/'owner-svt-ssd-v1/urgent-mirror-v10-binding.json').exists() else None,
                resource_supervision=bind(root/'owner-svt-ssd-v1/resource-supervision-v11.json') if (root/'owner-svt-ssd-v1/resource-supervision-v11.json').exists() else None,
                monitor_restart=bind(root/'owner-svt-ssd-v1/monitor-restart-binding.json') if (root/'owner-svt-ssd-v1/monitor-restart-binding.json').exists() else None)


def install_controller(root,receipt):
    """Replace only the waiting controller's future qualifier and phase mapping."""
    import sys
    root=Path(root);proof,clock,_=configuration(root)
    frames=[]
    for top in sys._current_frames().values():
        f=top
        while f:
            if f.f_code.co_name=='run' and Path(f.f_code.co_filename)==root/'control-resume-v7/e36_resume_schedule.py' and f.f_locals.get('root')==root:
                frames.append(f)
            f=f.f_back
    if len(frames)!=1 or 'rc' in frames[0].f_locals or (root/'bulk-choice.json').exists():
        raise ValueError('unique waiting pre-choice E36 controller required')
    f=frames[0];old_plan=dict(f.f_locals['plan']);new_plan=json.loads(checked(proof['plan']).read_text())
    original_scheduling=dict(f.f_locals['a'])
    updated=dict(original_scheduling,plan=proof['plan'],scientific_change=True,schema='e36-owner-svt-only-scheduling-v1',owner_matrix=bind(root/'owner-svt-ssd-v1/matrix-binding.json'))
    updated['manifests']={phase:dict(**pin['scheduled'],original=pin['manifest'],jobs=pin['jobs']) for phase,pin in proof['phases'].items() if 'scheduled' in pin}
    scheduled_path=root/'owner-svt-ssd-v1/controller-scheduling.json'
    write(scheduled_path,updated)
    # Preserve the short-job windows established by the prior census binding.
    from e36_phase_handoff import supervisor_lock
    paper_lock=Path('/home/lilith/tmp/zensim-paper/rev4/heavy.lock')
    supervisor=2672125
    def window(name,stage):
        stem=name+'-'+stage+'-'+str(time.time_ns())
        for operation,label in [(8,'release'),(2,'reacquire')]:
            p=root/'step-lock-windows'/(stem+'-'+label+'.log')
            supervisor_lock(supervisor,paper_lock,operation,p)
    def qualified(root_arg,name,start,deadline):
        window(name,'owner-before')
        try:return qualify(root_arg,name,start,deadline)
        finally:window(name,'owner-after')
    f.f_locals['owner'].qualify=qualified
    f.f_locals['plan'].clear();f.f_locals['plan'].update(new_plan)
    f.f_locals['a'].clear();f.f_locals['a'].update(updated)
    f.f_locals['amendment_path']=scheduled_path
    f.f_locals['env']['PYTHONPATH']=str(root/'owner-svt-ssd-v1')
    os.environ['PYTHONPATH']=str(root/'owner-svt-ssd-v1')
    write(receipt,dict(schema='e36-live-owner-controller-binding-v1',pid=os.getpid(),utc=time.time(),
          original_plan_sha256=hashlib.sha256(canonical(old_plan).encode()).hexdigest(),
          plan=proof['plan'],clock=proof['clock'],matrix_binding=bind(root/'owner-svt-ssd-v1/matrix-binding.json'),
          controller_scheduling=bind(scheduled_path),qualifier=bind(Path(__file__)),
          paper_lock=str(paper_lock),supervisor=supervisor,short_job_windows=True,fits=0))
