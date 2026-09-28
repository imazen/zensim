#!/usr/bin/env python3
# Rev4 E4 exploratory checks (NOT preregistered). Run from the repo root after `rev4_e4_agreement.py analyze`.
# EXPLORATORY (not preregistered): leave-one-lineage-out stability, D1-only gate, descriptives.
import json, sys, collections, statistics as st
sys.path.insert(0, 'scripts')
from lib import zen_stats
rows = json.load(open('/var/tmp/rev4-e4/table_MAIN.json'))
pred = {c['name']: c for c in json.load(open('/var/tmp/rev4-e4/ladder_predictors.json'))['cells']}
out = {}
def partial(rr, p, y):
    x = [-r['P'][p] for r in rr]; Y = [r['Y'][y] for r in rr]; C = [r['C'][y] for r in rr]
    res = zen_stats.panel_batch([('py', x, Y), ('pc', x, C), ('cy', C, Y)], stats='srocc')
    a, c, d = [r.get('srocc_signed', r['srocc']) for r in res]
    return (a - c * d) / ((1 - c * c) * (1 - d * d)) ** 0.5, a
lins = sorted({r['lineage'] for r in rows})
for p, y in (('P_dis', 'D2_floors'), ('P_dis', 'H4_csiq'), ('P_bu', 'D1_contract'), ('P_dis', 'H3_konjnd504')):
    vals = {}
    for L in lins:
        rr = [r for r in rows if r['lineage'] != L]
        vals[L] = round(partial(rr, p, y)[0], 3)
    out[f'LOLO partial {p}->{y}'] = {'min': min(vals.values()), 'max': max(vals.values()),
                                     'argmin': min(vals, key=vals.get), 'argmax': max(vals, key=vals.get)}
tau = 0.009938837920489297
t = collections.Counter()
for r in rows:
    t[('reject' if r['P']['P_dis'] > tau else 'keep') + '_' + ('contractfail' if r['Y']['D1_contract'] == 0 else 'nofail')] += 1
out['gate P_dis>tau vs D1 only'] = dict(t)
tb = collections.Counter()
mb = st.median(r['P']['P_bu'] for r in rows)
for r in rows:
    tb[('reject' if r['P']['P_bu'] > mb else 'keep') + '_' + ('contractfail' if r['Y']['D1_contract'] == 0 else 'nofail')] += 1
out['gate P_bu>median vs D1 only'] = {'median_P_bu': mb, **dict(tb)}
enc = [pred[r['name']]['counts'].get('inv_encoder', 0) for r in rows]
out['encoder_attributed_rungs'] = {'min': min(enc), 'median': st.median(enc), 'max': max(enc)}
out['D1_fail_count'] = sum(1 for r in rows if r['Y']['D1_contract'] == 0)
per = collections.defaultdict(list)
for r in rows: per[r['lineage']].append(r)
out['per_lineage'] = {L: {'n': len(v), 'P_dis_med': round(st.median(x['P']['P_dis'] for x in v), 4),
                          'P_bu_med': round(st.median(x['P']['P_bu'] for x in v), 4),
                          'C_A_med': round(st.median(x['C']['C_A'] for x in v), 4),
                          'D2_med': round(st.median(x['Y']['D2_floors'] for x in v), 3),
                          'D1_fail': sum(1 for x in v if x['Y']['D1_contract'] == 0)} for L, v in sorted(per.items())}
json.dump(out, open('/var/tmp/rev4-e4/explore.json', 'w'), indent=1)
print(json.dumps(out, indent=1))
