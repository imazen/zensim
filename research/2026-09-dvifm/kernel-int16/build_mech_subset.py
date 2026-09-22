#!/usr/bin/env python3
"""DVIFM screen-1 mechanism-check subset: row subsets of the already-admitted
minimal-top segments (same admissions; subsetting cannot cross the boundary)."""
import json, hashlib
from pathlib import Path
src = Path('/home/lilith/work/zensim-validation-2026-09-13/minimal-top')
out = Path('/mnt/v/output/zensim/dvifm-screen-2026-09-19/mech')
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
train = json.loads((src/'train-segment.json').read_text())['rows']
evalr = json.loads((src/'eval-segment.json').read_text())['rows']
# eval subset: the four spatial-case origins keep the audit's spatial stage runnable
eval_origins = {'kadid:I01', 'kadid:I21', 'kadid:I41', 'kadid:I61'}
eval_sub = [r for r in evalr if r['origin'] in eval_origins]
# train subset: first 4 sorted kadid train origins + first 200 tid rows
kadid_train_origins = sorted({r['origin'] for r in train if r['corpus'] == 'kadid'})[:4]
train_sub = [r for r in train if r['origin'] in set(kadid_train_origins)]
train_sub += [r for r in train if r['corpus'] == 'tid'][:200]
print('train subset:', len(train_sub), 'origins', kadid_train_origins)
print('eval subset:', len(eval_sub), 'origins', sorted(eval_origins))
assert len(train_sub) == 700 and len(eval_sub) == 500
for role, rows in (('train', train_sub), ('eval', eval_sub)):
    seg = {'schema': 'zensim-feature-segment-v1', 'role': role, 'rows': rows}
    p = out / f'{role}-segment.json'
    p.write_text(json.dumps(seg, indent=2, allow_nan=False) + '\n')
    print(role, 'segment sha256:', sha(p))
