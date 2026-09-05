#!/usr/bin/env python3
"""Replication-wave analysis. Re-derives NO correlation: every SROCC and every CI is
READ from a bake_verdict --full-json fulleval (the owner). Arm mean/spread are plain
summaries of those measured values; the CI half-width they are compared against is the
owner's own bootstrap."""
import json, os, sys, statistics as st

W = '/mnt/v/output/zensim/replication-2026-09-05'
FE_WAVE = f'{W}/fulleval'
FE_BOARD = '/mnt/v/output/zensim/reports/fulleval'
AXES = ['cid22', 'konjnd', 'aic3', 'csiq', 'live', 'nonphoto', 'imazen26']

# recipe -> (S0, the board cell holding the legacy diagonal draw at S0)
DIAG = {'LSTAR': (4021, 'LSTAR_s4021_packed'),
        'LSTAR3': (4041, 'LSTAR3_s4041_packed'),
        'W11J': (4013, 'W11J_s4013_packed')}
ERA2 = {'A3b'}   # reported separately, never pooled


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def axis(o, a):
    r = ((o or {}).get('rank') or {}).get(a) or {}
    ci = r.get('srocc_ci')
    lo, hi = (ci if isinstance(ci, (list, tuple)) and len(ci) == 2 else (None, None))
    return r.get('srocc_signed'), lo, hi


def comp(o):
    return (o or {}).get('product_composite') or (o or {}).get('composite')


fits = json.load(open('/home/lilith/tmp/replicate/fits.json'))
members = {}          # recipe -> arm -> [(label, obj)]
for f in fits:
    tag = f['tag']
    if tag.startswith('CTL'):
        continue
    rec, arm = tag.split('__')[0], f['arm']
    o = load(f'{FE_WAVE}/{tag}_packed.fulleval.json')
    if o is None:
        print(f'  MISSING harvest: {tag}', file=sys.stderr); continue
    members.setdefault(rec, {}).setdefault(arm, []).append((f"i{f['init']}/p{f['sample']}", o))

# the legacy diagonal belongs to BOTH arms (CTL-B proved --seed X == split X/X)
for rec, (s0, cell) in DIAG.items():
    o = load(f'{FE_BOARD}/{cell}.fulleval.json')
    if o is None or rec not in members:
        continue
    for arm in ('S', 'I'):
        members[rec].setdefault(arm, []).append((f'i{s0}/p{s0} (diagonal)', o))

print('# REPLICATION WAVE — arm decomposition (ORDER vs INIT)\n')
print('Arm S = sample/ORDER varies, init fixed. Arm I = init varies, order fixed.')
print('Every SROCC and CI is read from the owner (bake_verdict --full-json); nothing recomputed.\n')

rows = []
for rec in sorted(members):
    tag = ' [era-2 root — reported separately, never pooled]' if rec in ERA2 else ''
    print(f'\n## {rec}{tag}')
    for a in AXES + ['composite']:
        line = []
        for arm in ('S', 'I', 'D'):
            ms = members[rec].get(arm) or []
            vals, halfs = [], []
            for lab, o in ms:
                if a == 'composite':
                    v, lo, hi = comp(o), None, None
                else:
                    v, lo, hi = axis(o, a)
                if v is None:
                    continue
                vals.append((lab, v))
                if lo is not None and hi is not None:
                    halfs.append((hi - lo) / 2.0)
            if len(vals) < 2:
                continue
            xs = [v for _, v in vals]
            line.append((arm, len(xs), st.mean(xs), max(xs) - min(xs),
                         st.median(halfs) if halfs else None, vals))
        if not line:
            continue
        print(f'\n  {a}:')
        for arm, k, m, sp, ci, vals in line:
            cis = f'{ci:.4f}' if ci is not None else '—'
            print(f'    arm {arm}  k={k}  mean={m:+.6f}  spread={sp:.6f}  median CI half-width={cis}')
            for lab, v in sorted(vals):
                print(f'        {lab:<26} {v:+.6f}')
        d = {arm: (sp, ci) for arm, k, m, sp, ci, _ in line}
        if 'S' in d and 'I' in d:
            sS, cS = d['S']; sI, cI = d['I']
            ref = max(x for x in (cS, cI) if x is not None) if (cS or cI) else None
            call = 'UNRESOLVED (both spreads inside the per-model CI)' if (
                ref is not None and sS < ref and sI < ref) else (
                'ORDER > INIT' if sS > sI else 'INIT > ORDER')
            print(f'    => spread(order)={sS:.6f} vs spread(init)={sI:.6f}  :: {call}')
            rows.append((rec, a, sS, sI, ref, call))

print('\n\n## SUMMARY (order-spread vs init-spread, per axis per recipe)')
print(f'{"recipe":<8} {"axis":<10} {"spread_order":>13} {"spread_init":>12} {"CI half":>9}  call')
for rec, a, sS, sI, ref, call in rows:
    r = f'{ref:.4f}' if ref is not None else '—'
    print(f'{rec:<8} {a:<10} {sS:>13.6f} {sI:>12.6f} {r:>9}  {call}')
