"""Reference-only, nested E36 source budgets and native normalization jobs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sklearn

from e34_augmentation import cluster

SIZES=[32,40,48,64,80,96,128,160,192,256,320,384,512,640,768,1024,1280,1536,2048,4096]
CLASSES=['photo','screen','lineart','mixed']
STRATA=[(c,s) for c in CLASSES for s in ['fit','selection']]


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2);f.write('\n')


def quotas(size,group,fallback=False):
    if size<=768:return None
    if size==4096:
        if group=='rav1e':return {k:1 for k in STRATA if k not in [('screen','fit'),('screen','selection'),('lineart','selection')]}
        values=([3,1,0,0,1,0,3,2] if fallback else [6,1,0,0,2,0,5,2])
        return dict(zip(STRATA,values))
    if group=='rav1e':
        fit,selection=((1,1) if fallback or size>=1536 else (8,2) if size==1024 else (4,1))
    else:
        fit,selection=((4,1) if size<=1280 else (2,1)) if fallback else ((16,4) if size==1024 else (8,2) if size==1280 else (4,1))
    return {(c,s):fit if s=='fit' else selection for c,s in STRATA}


def choose(rows,quota,features,receipts,label):
    if quota is None:return list(rows)
    selected=[]
    for c,s in STRATA:
        k=quota.get((c,s),0)
        if not k:continue
        pool=sorted([r for r in rows if r['content']==c and r['split']==s],key=lambda r:r['origin'])
        if len(pool)<k:raise ValueError(f'quota exceeds structural reachability: {label} {c} {s} {k}/{len(pool)}')
        result=cluster([features[r['origin']] for r in pool],k,36)
        receipts[label+':'+c+':'+s]=result
        ids=set(result['origins']);selected.extend(r for r in pool if r['origin'] in ids)
    return selected


def prepare(root,features_path,initial):
    out=root/'source-budget';out.mkdir(exist_ok=False)
    selection=json.loads((root/'cohort-selection.json').read_text())
    assert selection['labels_read'] is False and selection['internal_overlap']['admitted'] is True
    rows=selection['rows'];assert len(rows)==200
    features={r['origin']:r for r in map(json.loads,features_path.read_text().splitlines())}
    assert all(r['origin'] in features for r in rows)
    all_refs={};panels={};receipts={};census=[]
    def add(ref,size):
        key=f"{ref['origin']}-{size}"
        value=dict(ref,size=size,reference_key=key,destination=str(root/'references'/str(size)/(ref['origin']+'.png')))
        if key in all_refs:assert all_refs[key]==value
        else:all_refs[key]=value
        return key
    for size in SIZES:
        eligible=[r for r in rows if max(r['width'],r['height'])>=size]
        for group in ['rav1e','fast']:
            full=choose(eligible,quotas(size,group),features,receipts,f'{size}:{group}:full')
            fallback=choose(full,quotas(size,group,True),features,receipts,f'{size}:{group}:fallback')
            full_keys=[add(r,size) for r in full];fallback_keys=[add(r,size) for r in fallback]
            assert set(fallback_keys)<=set(full_keys)
            panels[f'{size}:{group}']=dict(full=full_keys,fallback=fallback_keys)
            census.append(dict(size=size,group=group,requested=200,reachable=len(eligible),full=len(full),
                fallback=len(fallback),full_families=len({r['family'] for r in full}),
                fallback_families=len({r['family'] for r in fallback}),
                full_strata={':'.join(k):v for k,v in Counter((r['content'],r['split']) for r in full).items()},
                reachable_strata={':'.join(k):v for k,v in Counter((r['content'],r['split']) for r in eligible).items()}))
    aux=choose(rows,{k:1 for k in STRATA},features,receipts,'aux')
    panels['mode-panel']={str(size):[add(r,size) for r in aux] for size in [64,256]}
    # Capacity sources are chosen before scoring. Their targets never enter a
    # TRAIN table, even if the same origin appears in a later training panel.
    old=json.loads((initial/'smoke-references-32.json').read_text())[0]
    native={r['path']:r for r in json.loads(Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5/p4-inventory-r2/audit.hashes.json').read_text())['training']}
    pilot={k:old[k] for k in ['origin','family','source','source_sha256','content']}
    pilot.update(width=native[pilot['source']]['width'],height=native[pilot['source']]['height'],split='diagnostic')
    panels['cost']=[add(pilot,size) for size in SIZES]
    square=sorted([r for r in rows if max(r['width'],r['height'])>=4096 and r['content']=='mixed' and r['split']=='fit'],
        key=lambda r:(-min(r['width'],r['height'])/max(r['width'],r['height']),r['origin']))[0]
    panels['square-witness']=[add(square,4096)]
    # Reuse the exact already-admitted opaque32 pixels, with their native receipt.
    previous={r['source_sha256']:r for r in map(json.loads,(initial/'opaque-references.jsonl').read_text().splitlines())}
    reference_rows=[]
    for ref in all_refs.values():
        dest=Path(ref['destination']);dest.parent.mkdir(parents=True,exist_ok=True)
        if ref['size']==32 and ref['source_sha256'] in previous:
            import os
            p=previous[ref['source_sha256']];assert p['color_policy']=='srgb-opaque-v1'
            os.link(p['destination'],dest)
            reference_rows.append(dict(ref,**{k:p[k] for k in ['width','height','sha256','pixels_sha256','color_policy']},color_admitted=True))
    with (out/'reused-references.jsonl').open('x') as f:
        for row in reference_rows:f.write(json.dumps(row)+'\n')
    reused={r['reference_key'] for r in reference_rows};jobs=[]
    for size in SIZES:
        todo=[r for k,r in all_refs.items() if r['size']==size and k not in reused]
        if not todo:continue
        job=out/f'normalize-{size}.json'
        write(job,dict(mode='normalize',longest_side=size,never_upscale=True,color_policy='srgb-opaque-v1',
            rows=todo,output=str(out/f'normalized-{size}.jsonl')))
        jobs.append(str(job))
    write(out/'plan.json',dict(schema='e36-source-budget-v1',labels_read=False,fits=0,
        selection_sha256=sha(root/'cohort-selection.json'),features_sha256=sha(features_path),
        sklearn_version=sklearn.__version__,seed=36,panels=panels,references=all_refs,census=census,
        centroid_receipts=receipts,normalize_jobs=jobs,
        fallback='complete fallback matrix first; additions qualify only as a wholly completed fixed matrix'))
    print(json.dumps(dict(references=len(all_refs),normalization_jobs=len(jobs),census=census)))


def normalize(root,pixels):
    import subprocess
    plan=json.loads((root/'source-budget/plan.json').read_text())
    for job in plan['normalize_jobs']:
        subprocess.run([str(pixels),job],check=True)
    refs=[json.loads(line) for p in sorted((root/'source-budget').glob('*references.jsonl')) for line in p.read_text().splitlines()]
    base=plan['references']
    for p in sorted((root/'source-budget').glob('normalized-*.jsonl')):
        for value in map(json.loads,p.read_text().splitlines()):
            matches=[r for r in base.values() if r['destination']==value['destination']]
            assert len(matches)==1
            refs.append(dict(matches[0],**{k:value[k] for k in ['width','height','sha256','pixels_sha256','color_policy']},color_admitted=True))
    assert len(refs)==len(base) and len({r['reference_key'] for r in refs})==len(refs)
    write(root/'references.json',dict(schema='e36-native-opaque-references-v1',labels_read=False,fits=0,
        source_budget_sha256=sha(root/'source-budget/plan.json'),rows=refs))
    print(json.dumps(dict(normalized_references=len(refs),fits=0)))


def amend(root,features_path):
    """Cost-only prospective cut, before bulk; retain every earlier receipt."""
    old=root/'source-budget/plan.json';plan=json.loads(old.read_text())
    features={r['origin']:r for r in map(json.loads,features_path.read_text().splitlines())}
    if sha(features_path)!=plan['features_sha256']:raise ValueError('reference embedding owner changed')
    out=root/'source-budget-v2';out.mkdir();panel=plan['panels']['4096:fast']
    pool=[plan['references'][k] for k in panel['fallback']]
    quota=dict(zip(STRATA,[1,1,0,0,1,0,1,2]))
    selected=choose(pool,quota,features,plan['centroid_receipts'],'4096:fast:mandatory-six')
    panel['fallback']=[r['reference_key'] for r in selected]
    assert len(selected)==6 and set(panel['fallback'])<=set(panel['full'])
    for r in plan['census']:
        if r['size']==4096 and r['group']=='fast':
            r['fallback']=6;r['fallback_families']=len({s['family'] for s in selected})
    plan['supersedes_sha256']=sha(old)
    plan['amendment']='Mandatory4096 fast sources10 to6: reference-only nested centroids; retain all3 available selection sources; rav1e5 unchanged. Before bulk; measured cost motivation only.'
    write(out/'plan.json',plan)
    refs=json.loads((root/'references.json').read_text());refs['source_budget_sha256']=sha(out/'plan.json')
    write(root/'references-v2.json',refs)
    print(json.dumps(dict(mandatory_fast_4096=panel['fallback'],source_budget_sha256=sha(out/'plan.json'),labels_read=False,fits=0)))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['prepare','normalize','amend'])
    p.add_argument('--root',required=True,type=Path);p.add_argument('--features',type=Path)
    p.add_argument('--initial',type=Path);p.add_argument('--pixels',type=Path);a=p.parse_args()
    if a.mode=='prepare':prepare(a.root,a.features,a.initial)
    elif a.mode=='normalize':normalize(a.root,a.pixels)
    else:amend(a.root,a.features)
