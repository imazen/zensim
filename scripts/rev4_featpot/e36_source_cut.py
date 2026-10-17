"""Owner-authorized largest-size source truncation; native identities unchanged."""
import copy
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import shutil
from e36_store import sha


def pin(path):return dict(path=str(path),sha256=sha(path))
def checked(value):
    path=Path(value['path'])
    if sha(path)!=value['sha256']:raise ValueError('source-count amendment proof changed')
    return path

def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')

def prepare(root,projection,commit):
    from e36_svt_matrix import configuration
    root=Path(root);out=root/'source-count-amendment-v1'
    if out.exists():raise ValueError('preserve existing source-count amendment')
    base,clock,_=configuration(root);basis=json.loads(Path(projection).read_text())
    if (basis['registered_full_including_optional']['misses22Z'] is not True or basis['array_rate_basis'].startswith('Actual completed disk1 packs') is False or basis['fits']!=0):raise ValueError('measured array/full-matrix deadline miss required')
    if list((root/'runs'/base['phases']['additions']['run']).glob('ledger*.parquet')):raise ValueError('optional source truncation must precede optional generation')
    runtime=json.loads((root/'runtime-v6/runtime.json').read_text());old_budget=Path(runtime['source_budget']);budget=json.loads(old_budget.read_text());old=copy.deepcopy(budget)
    panel=budget['panels']['4096:fast'];before=panel['full'];after=panel['fallback']
    if len(before)!=16 or len(after)!=6 or not set(after)<set(before):raise ValueError('exact frozen16-to6 centroid subset required')
    retained=set(after);removed=set(before)-retained
    panel['full']=list(after)
    for record in budget['census']:
        if record['size']==4096 and record['group']=='fast':
            record['requested']=len(after);record['full']=len(after);record['full_strata']=dict(Counter(budget['references'][k]['content']+':'+budget['references'][k]['split'] for k in after))
            record['full_families']=len({budget['references'][k].get('family',budget['references'][k]['origin']) for k in after})
    for key in list(budget['centroid_receipts']):
        if key.startswith('4096:fast:full:'):
            fallback_key=key.replace(':full:',':mandatory-six:',1)
            budget['centroid_receipts'][key]=copy.deepcopy(old['centroid_receipts'][fallback_key])
    budget['previous_amendment']=old.get('amendment')
    budget['supersedes_sha256']=sha(old_budget);budget['amendment']=dict(kind='authorized-source-count-only',size=4096,group='fast',before=16,after=6,selection='Exact already-frozen stratified centroid-nearest fallback subset; no clustering/score/quality/completion-based selection.')
    out.mkdir();budget_path=out/'source-budget.json';write(budget_path,budget)
    refs={r['reference_key']:r for r in json.loads(Path(runtime['references']).read_text())['rows']}
    def keep(job):
        view=json.loads(job['kind']['view'].split(':',1)[1]);return refs[view['reference_key']]['size']!=4096 or view['reference_key'] in retained
    phases=copy.deepcopy(base['phases']);counts={}
    for phase in ['additions','full']:
        old_phase=base['phases'][phase];jobs=json.loads(checked(old_phase['manifest']).read_text());selected=[j for j in jobs if keep(j)];run='svt-'+phase+'-source-count-v1';directory=root/'runs'/run;directory.mkdir();manifest=directory/'manifest.json';write(manifest,selected)
        declaration=json.loads((checked(old_phase['manifest']).parent/'declaration.json').read_text());declaration.update(jobs=len(selected),manifest_sha256=sha(manifest),source_count_amendment=str(out/'amendment.json'));declaration.pop('views_sha256',None);write(directory/'declaration.json',declaration)
        ids=directory/'native.ids.tsv'
        with ids.open('x') as stream:subprocess.run([str(root/'bin/zenfleet-ctl'),'ids','--manifest',str(manifest)],stdout=stream,check=True)
        entry=dict(old_phase,run=run,manifest=pin(manifest),native_ids=pin(ids),jobs=len(selected))
        if phase=='additions':
            scheduled=json.loads(checked(old_phase['scheduled']).read_text());schedule=out/'additions-scheduled.json';write(schedule,[j for j in scheduled if keep(j)]);entry['scheduled']=pin(schedule)
        phases[phase]=entry;counts[phase]=dict(before=len(jobs),after=len(selected),excluded=len(jobs)-len(selected))
    if counts!={'additions':{'before':424,'after':384,'excluded':40},'full':{'before':12724,'after':12684,'excluded':40}}:raise ValueError('exact largest-size recomputed matrix differs')
    amendment=dict(schema='e36-authorized-source-count-amendment-v1',authorization='Coordinator2026-10-17: if storage-limited projection misses22Z, cut sourcecounts at largest sizes only; all61q/fourcodecs/20sizes/nonzero.',
        utc=datetime.now(timezone.utc).isoformat(),build_commit=commit,scientific_change=True,source_count_change=True,fits=0,labels_read=False,deadline_unix=clock['deadline_unix'],
        projection_basis=pin(Path(projection)),original_budget=pin(old_budget),effective_budget=pin(budget_path),largest_size=4096,old_full_sources_per_codec=16,new_full_sources_per_codec=6,
        retained_reference_keys=after,excluded_reference_keys=sorted(removed),mandatory_views_unchanged=12300,phase_counts=counts,all_other19size_counts_unchanged=True,qualities=61,codecs=['jpeg','svt','webp','jxl'],sizes=20,
        native_runtime_unchanged=pin(root/'runtime-v6/runtime.json'),teachers_and_assessment_unchanged=True,source_selection='Reuses original frozen fallback centroid subset; no score or completion choice.',
        modeled_removed_optional_bytes=basis['optional_bytes_by_size']['4096'],original_full_projection=basis['registered_full_including_optional'])
    write(out/'amendment.json',amendment)
    plan=json.loads(checked(base['plan']).read_text())
    for phase in ['additions','full']:plan['phases'][phase].update(run=phases[phase]['run'],manifest_sha256=phases[phase]['manifest']['sha256'],native_ids_sha256=phases[phase]['native_ids']['sha256'],jobs=phases[phase]['jobs'])
    plan['effective_source_budget']=pin(budget_path);plan['source_count_amendment']=pin(out/'amendment.json');write(out/'bulk-plan.json',plan)
    resources=out/'archive-resources';resources.mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copyfile(source,resources/source.name)
    (resources/'sitecustomize.py').write_text('import sys,runpy\nsys.path.insert(0,'+repr(str(resources))+')\nif "--runtime" in sys.argv:\n runpy.run_path('+repr(str(root/'owner-svt-ssd-v1/closure-v6/sitecustomize.py'))+')\n sys.path.insert(0,'+repr(str(resources))+')\n import e36_budget4\n e36_budget4.startup()\n')
    write(resources/'sources.json',dict(schema='e36-source-cut-archive-resources-v1',files={p.name:pin(p) for p in resources.glob('*.py')},fits=0,scientific_change=False))
    binding=dict(schema='e36-source-count-native-binding-v1',amendment=pin(out/'amendment.json'),base_matrix=pin(root/'owner-svt-ssd-v1/matrix-binding.json'),base_plan=base['plan'],base_clock=base['clock'],plan=pin(out/'bulk-plan.json'),source_budget=pin(budget_path),archive_resources=pin(resources/'sources.json'),phases=phases,scientific_change=True,fits=0,fit_release=False)
    write(out/'binding.json',binding);return binding


def apply(root,base,clock):
    path=Path(root)/'source-count-amendment-v1/binding.json'
    if not path.exists():return base
    binding=json.loads(path.read_text());amendment=json.loads(checked(binding['amendment']).read_text());old=json.loads(checked(amendment['original_budget']).read_text());new=json.loads(checked(binding['source_budget']).read_text())
    if binding['schema']!='e36-source-count-native-binding-v1' or binding['fits']!=0 or binding['fit_release'] is not False or binding['base_plan']!=base['plan'] or binding['base_clock']!=base['clock'] or checked(binding['base_matrix'])!=Path(root)/'owner-svt-ssd-v1/matrix-binding.json':raise ValueError('source-count predecessor differs')
    expected=copy.deepcopy(old);expected['panels']['4096:fast']['full']=expected['panels']['4096:fast']['fallback']
    if new['panels']!=expected['panels'] or new['references']!=old['references'] or amendment['qualities']!=61 or amendment['sizes']!=20 or amendment['deadline_unix']!=clock['deadline_unix'] or amendment['fits']!=0 or amendment['teachers_and_assessment_unchanged'] is not True:raise ValueError('source-count change exceeds authorized largest-size subset')
    refs=old['references'];retained=set(old['panels']['4096:fast']['fallback'])
    def keep(job):
        v=json.loads(job['kind']['view'].split(':',1)[1]);return int(refs[v['reference_key']]['size'])!=4096 or v['reference_key'] in retained
    for phase in ['additions','full']:
        original=json.loads(checked(base['phases'][phase]['manifest']).read_text());entry=binding['phases'][phase];actual=json.loads(checked(entry['manifest']).read_text())
        if actual!=[j for j in original if keep(j)] or entry['jobs']!=len(actual):raise ValueError('source-count native matrix is not exact original subset')
        old_ids=[l.split('\t')[1] for l in checked(base['phases'][phase]['native_ids']).read_text().splitlines()];new_ids=[l.split('\t')[1] for l in checked(entry['native_ids']).read_text().splitlines()]
        if new_ids!=[i for i,j in zip(old_ids,original) if keep(j)]:raise ValueError('source-count native ID identity changed')
        if phase=='additions':
            scheduled=json.loads(checked(entry['scheduled']).read_text());previous=json.loads(checked(base['phases'][phase]['scheduled']).read_text())
            if scheduled!=[j for j in previous if keep(j)]:raise ValueError('source-count scheduler changed retained jobs')
    if binding['phases']['fallback']!=base['phases']['fallback']:raise ValueError('mandatory fallback changed')
    resources=binding['archive_resources'];correction=Path(root)/'source-count-amendment-v1/archive-resources-v2-binding.json'
    if correction.exists():
        repair=json.loads(correction.read_text())
        if repair['schema']!='e36-source-cut-archive-repair-v1' or repair['predecessor']!=pin(path) or repair['previous_resources']!=resources or repair['fits']!=0 or repair['scientific_change'] is not False:raise ValueError('source-count archive repair predecessor differs')
        resources=repair['resources']
    source_manifest=json.loads(checked(resources).read_text())
    for value in source_manifest['files'].values():checked(value)
    checked(binding['plan']);result=copy.deepcopy(base);result.update(plan=binding['plan'],phases=binding['phases'],source_count_amendment=pin(path),effective_source_budget=binding['source_budget'],archive_resources=resources,archive_resource_repair=pin(correction) if correction.exists() else None)
    return result


def repair_archive(root):
    """Version storage helpers without rewriting the scientific amendment."""
    root=Path(root);out=root/'source-count-amendment-v1';binding=out/'binding.json';previous=json.loads(binding.read_text())
    if list((root/'runs'/previous['phases']['additions']['run']).glob('ledger*.parquet')):raise ValueError('freeze corrected archive helpers before optional generation')
    resources=out/'archive-resources-v2';resources.mkdir()
    for source in Path(__file__).parent.glob('*.py'):shutil.copyfile(source,resources/source.name)
    startup=checked(previous['archive_resources']).parent/'sitecustomize.py'
    (resources/'sitecustomize.py').write_text(startup.read_text().replace(str(startup.parent),str(resources)))
    write(resources/'sources.json',dict(schema='e36-source-cut-archive-resources-v1',files={p.name:pin(p) for p in resources.glob('*.py')},fits=0,scientific_change=False))
    correction=dict(schema='e36-source-cut-archive-repair-v1',predecessor=pin(binding),previous_resources=previous['archive_resources'],resources=pin(resources/'sources.json'),utc=datetime.now(timezone.utc).isoformat(),fits=0,scientific_change=False,
        reason='Discover amended optional CAS through the same native-run owner as ledgers; preserve all historical stores and source counts.')
    path=out/'archive-resources-v2-binding.json';write(path,correction);return pin(path)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--projection',type=Path);parser.add_argument('--commit');parser.add_argument('--repair-archive',action='store_true');a=parser.parse_args()
    if not a.repair_archive and (a.projection is None or a.commit is None):parser.error('projection and commit required for scientific amendment')
    print(json.dumps(repair_archive(a.root) if a.repair_archive else prepare(a.root,a.projection,a.commit)))
