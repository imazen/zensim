"""Strict frozen deadline/matrix chain; metadata only, no fit or label reads."""
import gzip
import json
from pathlib import Path
from e36_data import PREFIX,canonical
from e36_store import sha


def admit(root,names):
    cp=root/'bulk-choice.json';choice=json.loads(cp.read_text())
    pp=Path(choice.get('plan_path',root/'bulk-plan.json'));tp=Path(choice.get('clock_path',root/'bulk-clock.json'))
    plan=json.loads(pp.read_text());clock=json.loads(tp.read_text())
    if (choice['schema']!='e36-fixed-matrix-choice-v1' or plan['schema']!='e36-six-hour-data-plan-v1'
        or choice['qualified'] not in ['fallback','full'] or choice['plan_sha256']!=sha(pp)
        or choice['clock_sha256']!=sha(tp) or clock['plan_sha256']!=sha(pp)
        or plan['deadline_seconds']!=21600 or clock['deadline_unix']!=clock['started_unix']+21600
        or choice['teacher_support_not_used_for_choice'] is not True or plan['fit_release'] is not False):
        raise ValueError('deadline-qualified frozen matrix chain required')
    if 'inherited_clock' in clock:
        inherited=clock['inherited_clock'];previous=json.loads(Path(inherited['path']).read_text())
        if sha(inherited['path'])!=inherited['sha256'] or any(clock[k]!=previous[k] for k in ['started_unix','deadline_unix']):raise ValueError('transport repair reset the original deadline')
        from e36_bulk import clock_equivalence
        eq=clock['equivalence'];ep=Path(eq['path'])
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
    bind=lambda p:dict(path=str(p),sha256=sha(p))
    return dict(schema='e36-qualified-matrix-admission-v1',matrix=choice['qualified'],choice=bind(cp),plan=bind(pp),clock=bind(tp),
        selected_runs=names,manifests=declarations,completed_views=len(completed),outputs=output_hashes,complete=True)
