"""Strict frozen deadline/matrix chain; metadata only, no fit or label reads."""
import gzip
import json
from pathlib import Path
from e36_data import PREFIX,canonical
from e36_store import sha


def scheduled_evidence(root,choice,names):
    """Retain the resource chain and checked native publication for every run."""
    if 'scheduling_amendment' not in choice:
        if (root/'scheduling-v7/amendment.json').exists():
            raise ValueError('native scheduling amendment missing from choice')
        return []
    evidence=[]
    def checked(pin):
        p=Path(pin['path'])
        if sha(p)!=pin['sha256']:raise ValueError('scheduling/publication evidence changed')
        evidence.append(pin)
        return p
    amendment=checked(choice['scheduling_amendment']);a=json.loads(amendment.read_text())
    if (a['schema']!='e36-native-scheduling-only-v1' or a['scientific_change'] is not False
            or a['fits']!=0 or a['plan']['sha256']!=choice['plan_sha256']
            or a['effective_clock']['sha256']!=choice['clock_sha256']):
        raise ValueError('scheduling amendment differs from qualified choice')
    checked(a['plan']);checked(a['effective_clock']);checked(a['worker'])
    original_amendment=amendment
    if 'previous_scheduling_amendment' in a:
        original_amendment=checked(a['previous_scheduling_amendment'])
        previous=json.loads(original_amendment.read_text())
        changes={'effective_clock','deadline_changed','disk_floor_bytes','resource_bootstrap','previous_scheduling_amendment'}
        if {k:v for k,v in a.items() if k not in changes}!={k:v for k,v in previous.items() if k not in changes}:
            raise ValueError('resource amendment changed native scheduling inputs')
        for pin in a['resource_bootstrap'].values():checked(pin)
    for name in names:
        census=json.loads((root/'runs'/name/'census.json').read_text())
        proof=json.loads(checked(census['native_publication']).read_text())
        if (proof['schema']!='e36-native-publication-admission-v1' or proof['admitted'] is not True
                or proof['fits']!=0 or proof['producer_attempts_per_job']!=1
                or proof['jobs']!=proof['published'] or proof['jobs']!=len(census['output_receipts'])
                or proof['amendment']['sha256']!=sha(original_amendment)):
            raise ValueError('complete native publication proof differs')
        from e36_data import canonical
        import hashlib
        if proof['census_output_inventory_sha256']!=hashlib.sha256(canonical(census['output_receipts']).encode()).hexdigest():
            raise ValueError('native publication output inventory changed')
        for key in ['native_ids','scheduled_ids','amendment']:checked(proof[key])
        for pin in proof['ledgers']:checked(pin)
    return evidence


def admit(root,names,archive_pending=False):
    if (root/'owner-svt-ssd-v1/matrix-binding.json').exists():
        from e36_svt_matrix import admit as admit_svt
        return admit_svt(root,names,archive_pending=archive_pending)
    if archive_pending:raise ValueError('deferred archive preparation requires qualified SVT owner')
    cp=root/'bulk-choice.json';choice=json.loads(cp.read_text())
    pp=Path(choice.get('plan_path',root/'bulk-plan.json'));tp=Path(choice.get('clock_path',root/'bulk-clock.json'))
    plan=json.loads(pp.read_text())
    from e36_pause_budget import resolve
    base_clock,clock,budget_evidence=resolve(tp)
    if (choice['schema']!='e36-fixed-matrix-choice-v1' or plan['schema']!='e36-six-hour-data-plan-v1'
        or choice['qualified'] not in ['fallback','full'] or choice['plan_sha256']!=sha(pp)
        or choice['clock_sha256']!=sha(tp) or clock['plan_sha256']!=sha(pp)
        or plan['deadline_seconds']!=21600 or base_clock['deadline_unix']!=base_clock['started_unix']+21600
        or choice['teacher_support_not_used_for_choice'] is not True or plan['fit_release'] is not False):
        raise ValueError('deadline-qualified frozen matrix chain required')
    if 'inherited_clock' in base_clock:
        inherited=base_clock['inherited_clock'];previous=json.loads(Path(inherited['path']).read_text())
        if sha(inherited['path'])!=inherited['sha256'] or any(base_clock[k]!=previous[k] for k in ['started_unix','deadline_unix']):raise ValueError('transport repair reset the original deadline')
        from e36_bulk import clock_equivalence
        eq=base_clock['equivalence'];ep=Path(eq['path'])
        if sha(ep)!=eq['sha256'] or json.loads(ep.read_text())!=clock_equivalence(root,pp,plan,Path(inherited['path'])):
            raise ValueError('original/repaired scientific equivalence no longer verifies')
    selected=[plan['phases']['fallback']['run']]
    if choice['qualified']=='full':selected.append(plan['phases']['additions']['run'])
    if names!=selected or choice['selected_runs']!=selected or len(names)!=len(set(names)):
        raise ValueError('exact unique deadline-qualified run list required')
    declarations={};expected={}
    for phase,entry in plan['phases'].items():
        path=root/'runs'/entry['run']/'manifest.json'
        if sha(path)!=entry['manifest_sha256']:raise ValueError('frozen matrix manifest changed')
        items=json.loads(path.read_text());views=[j['kind']['view'] for j in items]
        if len(views)!=len(set(views)):raise ValueError('duplicate declared view')
        expected[phase]=set(views);declarations[entry['run']]=dict(path=str(path),sha256=sha(path))
    if expected['fallback']&expected['additions'] or expected['fallback']|expected['additions']!=expected['full']:
        raise ValueError('frozen native matrix union differs')
    completed=set();output_hashes=[]
    for name in names:
        run=root/'runs'/name;census_path=run/'census.json';census=json.loads(census_path.read_text())
        if census['admitted'] is not True:raise ValueError('native matrix incomplete')
        for item in census['output_receipts']:
            p=Path(item['path'])
            if sha(p)!=item['sha256']:raise ValueError('native matrix receipt changed')
            doc=json.loads(gzip.decompress(p.read_bytes()));view=PREFIX+canonical(doc['view'])
            if (not doc['complete'] or view in completed or not clock['started_unix']<=doc['completed_unix']<=clock['deadline_unix']
                or not clock['started_unix']<=p.stat().st_mtime<=clock['deadline_unix']):
                raise ValueError('duplicate/incomplete/out-of-window native output')
            completed.add(view);output_hashes.append(item['sha256'])
    if completed!=expected[choice['qualified']]:raise ValueError('selected matrix has missing/swapped/extra declared views')
    remote_evidence=[]
    if choice['qualified']=='full' and (root/'TOWER_PHASE_HANDOFF.json').exists():
        from e36_phase_handoff import admit_installed
        remote_evidence=admit_installed(root,tp,plan)
    scheduling_evidence=scheduled_evidence(root,choice,names)
    bind=lambda p:dict(path=str(p),sha256=sha(p))
    return dict(schema='e36-qualified-matrix-admission-v1',matrix=choice['qualified'],choice=bind(cp),plan=bind(pp),clock=bind(tp),
        selected_runs=names,manifests=declarations,completed_views=len(completed),outputs=output_hashes,complete=True,budget_evidence=budget_evidence,remote_phase_evidence=remote_evidence,scheduling_evidence=scheduling_evidence)
