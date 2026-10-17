"""Preview registered teacher support on completed largest-size checkpoints only."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
from e36_data import Q,canonical
from e36_store import sha
from e36_semantic import required_strata
sys.path.insert(0,str(Path(__file__).parents[1]/'canonical_corpus'))
from build_ladder_tv_pairs import e36_pairs


def preview(root,out):
    root=Path(root);binding=json.loads((root/'source-count-amendment-v1/binding.json').read_text());budget_path=Path(binding['source_budget']['path'])
    if sha(budget_path)!=binding['source_budget']['sha256']:raise ValueError('effective source budget changed')
    budget=json.loads(budget_path.read_text());size=max(r['size'] for r in budget['references'].values())
    census=root/'runs/svt-fallback-v1/census.json'
    if json.loads(census.read_text())['admitted'] is not True:raise ValueError('qualified mandatory data required')
    def largest(jobs):
        return [job for job in jobs if (lambda v:v['panel']=='baseline' and budget['references'][v['reference_key']]['size']==size)(json.loads(job['kind']['view'].split(':',1)[1]))]
    jobs=largest(json.loads((census.parent/'manifest.json').read_text()));full=largest(json.loads(Path(binding['phases']['full']['manifest']['path']).read_text()))
    if {j['kind']['view'] for j in jobs}!={j['kind']['view'] for j in full}:raise ValueError('largest preview must cover the entire final largest-size subset')
    expected={k:v for k,v in required_strata(budget,'full',['rav1e']).items() if json.loads(k)[2]==size};counts=defaultdict(int);inputs=[];total=0
    runtime_sha=sha(root/'runtime-v6/runtime.json')
    for job in jobs:
        text=job['kind']['view'];key=hashlib.sha256(text.encode()).hexdigest();view=json.loads(text.split(':',1)[1]);rows=[]
        if view['q_grid']!=Q:raise ValueError('largest preview requires every registered q')
        for ordinal,q in enumerate(Q):
            path=root/'progress'/key/(str(ordinal)+'.json.gz');row=json.loads(gzip.decompress(path.read_bytes()))
            if row['runtime_sha256']!=runtime_sha or row['view_sha256']!=key or row['q']!=q:raise ValueError('checkpoint is not bound to the exact native view')
            row['knobs']=json.loads(row['mode_key']);rows.append(row);inputs.append(dict(path=str(path),sha256=sha(path)))
        pairs,_=e36_pairs(rows);total+=len(pairs)
        for pair in pairs:
            key=canonical([pair['codec'],pair['content'],pair['size'],pair['band'],pair['split']]);counts[key]+=1
    missing=[key for key in expected if counts[key]==0]
    result=dict(schema='e36-largest-size-consensus-preview-v1',size=size,views=len(jobs),candidate_preferences=total,required_strata=len(expected),zero_candidate_support_strata=missing,
        support={key:dict(selected_references=expected[key],candidate_preferences=counts[key]) for key in expected},checkpoint_inputs=inputs,budget=binding['source_budget'],mandatory_census=dict(path=str(census),sha256=sha(census)),consensus_source_sha256=sha(Path(__file__).parents[1]/'canonical_corpus/build_ladder_tv_pairs.py'),
        qualifies_training_data=False,physical_deduplication_not_replayed=True,source_or_margin_changes=False,labels_read=False,optimizer_steps=0,fits=0)
    with Path(out).open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps({key:value for key,value in result.items() if key not in ['support','checkpoint_inputs']}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args();preview(a.root,a.out)
