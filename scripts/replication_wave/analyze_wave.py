#!/usr/bin/env python3
"""Replication-wave analysis. Re-derives NO correlation: every SROCC and every CI is
READ from a bake_verdict --full-json fulleval (the owner). Arm mean/spread are plain
summaries of those measured values; the CI half-width they are compared against is the
owner's own bootstrap."""
import json, os, sys, statistics as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from lib import srocc_ci  # noqa: E402  (each stored interval's kind; never mixed)

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
    """`(srocc_signed, lo, hi, kind)` of one stored axis; `kind` is the interval kind
    (`ref` / `pair`, scripts/lib/srocc_ci.py), None without an interval."""
    r = ((o or {}).get('rank') or {}).get(a) or {}
    iv = srocc_ci.interval(r)
    lo, hi, kind = iv if iv is not None else (None, None, None)
    return r.get('srocc_signed'), lo, hi, kind


def ci_half(intervals):
    """Median 95% CI half-width over `(lo, hi, kind)` intervals of ONE kind, as
    `(median, kind)`. Intervals of different kinds are never pooled: a mixed set returns
    `(None, 'mixed')`, and an empty one `(None, None)`."""
    kind = srocc_ci.common_kind(k for _, _, k in intervals)
    if not intervals:
        return None, None
    if kind is None:
        return None, 'mixed'
    return st.median((hi - lo) / 2.0 for lo, hi, _ in intervals), kind


def fmt_half(half, kind):
    if kind == 'mixed':
        return 'mixed CI kinds'
    return f'{half:.4f} {kind}' if half is not None else '—'


def comp(o):
    return (o or {}).get('product_composite') or (o or {}).get('composite')


def main():
    fits = json.load(open('/home/lilith/tmp/replicate/fits.json'))
    members = {}          # recipe -> arm -> [(label, obj)]
    for f in fits:
        tag = f['tag']
        if tag.startswith('CTL'):
            continue
        rec, arm = tag.split('__')[0], f['arm']
        o = load(f'{FE_WAVE}/{tag}_packed.fulleval.json')
        if o is None:
            # A3b is an era-2 x radius-4 recipe: bake_verdict REFUSES to score it on
            # the canonical folded root (it structurally uses 72 caller lines in
            # f156-371 that the folded root feeds as zeros), so its arm is scored at
            # its NATIVE root and lands here instead. Reported separately, never pooled.
            o = load(f'{W}/verdicts_a3b/{tag}_packed.fulleval.json')
        if o is None:
            print(f'  MISSING harvest: {tag}', file=sys.stderr); continue
        lab = f"i{f['init']}/p{f['sample']}"
        if arm == 'D':
            # A re-run diagonal (init == sample == S0) is a member of BOTH arms, the
            # same way a legacy --seed draw is — CTL-B measured that equivalence.
            lab += ' (diagonal, re-run)'
            for a2 in ('S', 'I'):
                members.setdefault(rec, {}).setdefault(a2, []).append((lab, o))
        else:
            members.setdefault(rec, {}).setdefault(arm, []).append((lab, o))

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
                vals, intervals = [], []
                for lab, o in ms:
                    if a == 'composite':
                        v, lo, hi, kind = comp(o), None, None, None
                    else:
                        v, lo, hi, kind = axis(o, a)
                    if v is None:
                        continue
                    vals.append((lab, v))
                    if lo is not None and hi is not None:
                        intervals.append((lo, hi, kind))
                if len(vals) < 2:
                    continue
                xs = [v for _, v in vals]
                line.append((arm, len(xs), st.mean(xs), max(xs) - min(xs),
                             ci_half(intervals), vals))
            if not line:
                continue
            print(f'\n  {a}:')
            for arm, k, m, sp, ci, vals in line:
                print(f'    arm {arm}  k={k}  mean={m:+.6f}  spread={sp:.6f}  '
                      f'median CI half-width={fmt_half(*ci)}')
                for lab, v in sorted(vals):
                    print(f'        {lab:<26} {v:+.6f}')
            d = {arm: (sp, ci) for arm, k, m, sp, ci, _ in line}
            if 'S' in d and 'I' in d:
                (sS, (cS, kS)), (sI, (cI, kI)) = d['S'], d['I']
                # The CI the spreads are compared with must be ONE interval kind; arms whose
                # intervals differ in kind (or mix within an arm) give no CI-based call.
                kind = srocc_ci.common_kind([kS, kI]) if 'mixed' not in (kS, kI) else None
                halves = [x for x in (cS, cI) if x is not None]
                ref = max(halves) if kind is not None and halves else None
                if ref is None and halves:
                    call = 'NO CI CALL (interval kinds differ: ref vs pair)'
                else:
                    call = 'UNRESOLVED (both spreads inside the per-model CI)' if (
                        ref is not None and sS < ref and sI < ref) else (
                        'ORDER > INIT' if sS > sI else 'INIT > ORDER')
                print(f'    => spread(order)={sS:.6f} vs spread(init)={sI:.6f}  :: {call}')
                rows.append((rec, a, sS, sI, (ref, kind), call))

    print('\n\n## SUMMARY (order-spread vs init-spread, per axis per recipe)')
    print(f'{"recipe":<8} {"axis":<10} {"spread_order":>13} {"spread_init":>12} {"CI half":>14}  call')
    for rec, a, sS, sI, (ref, kind), call in rows:
        r = f'{ref:.4f} {kind}' if ref is not None else '—'
        print(f'{rec:<8} {a:<10} {sS:>13.6f} {sI:>12.6f} {r:>14}  {call}')


if __name__ == '__main__':
    main()
