"""Diagnose failed support and the predeclared optional sources, without fitting."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
from e36_store import sha
from e36_data import Q,canonical
sys.path.insert(0,str(Path(__file__).parents[1]/'canonical_corpus'))
from build_ladder_tv_pairs import e36_pairs


def diagnose(root,out):
    root=Path(root);metadata=json.loads((root/'pools-final-v1/_MANIFEST.json').read_text());binding=json.loads((root/'source-count-amendment-v1/binding.json').read_text())
    budget_path=Path(binding['source_budget']['path'])
    if sha(budget_path)!=binding['source_budget']['sha256']:raise ValueError('registered source budget changed')
    budget=json.loads(budget_path.read_text());phase=binding['phases']['additions'];manifest=Path(phase['manifest']['path'])
    if sha(manifest)!=phase['manifest']['sha256']:raise ValueError('registered optional matrix changed')
    missing=metadata['zero_support_required_strata'];results=[];inputs=[]
    for job in json.loads(manifest.read_text()):
        text=job['kind']['view'];view=json.loads(text.split(':',1)[1]);ref=budget['references'][view['reference_key']]
        candidates=[k for k in missing if (lambda k:(k[0],k[1],k[2],k[4])==(view['codec'],ref['content'],ref['size'],ref['split']))(json.loads(k))]
        if not candidates:continue
        if view['q_grid']!=Q or view['role']!='bulk' or view['panel']!='baseline':raise ValueError('exact registered baseline ladder required')
        v=hashlib.sha256(text.encode()).hexdigest();rows=[]
        for i,q in enumerate(Q):
            path=root/'progress'/v/(str(i)+'.json.gz');row=json.loads(gzip.decompress(path.read_bytes()))
            if row['view_sha256']!=v or row['q']!=q or row['runtime_sha256']!=view['runtime_sha256']:raise ValueError('exact optional checkpoint identity differs')
            row['knobs']=json.loads(row['mode_key']);rows.append(row);inputs.append(dict(path=str(path),sha256=sha(path)))
        pairs,counts=e36_pairs(rows)
        support={k:sum(canonical([p['codec'],p['content'],p['size'],p['band'],p['split']])==k for p in pairs) for k in candidates}
        results.append(dict(reference_key=view['reference_key'],codec=view['codec'],panel=view['panel'],support=support,by_stratum={k:counts['by_stratum'][k] for k in candidates if k in counts['by_stratum']}))
    value=dict(schema='e36-zero-support-optional-preview-v1',missing=missing,optional_matching_views=len(results),optional_support=results,checkpoint_inputs=inputs,source_budget=binding['source_budget'],optional_manifest=phase['manifest'],qualifies_training_data=False,physical_deduplication_not_replayed=True,teacher_and_margin_changes=False,labels_read=False,fits=0)
    with Path(out).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in value.items() if k!='checkpoint_inputs'}));return value

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();diagnose(a.root,a.out)
