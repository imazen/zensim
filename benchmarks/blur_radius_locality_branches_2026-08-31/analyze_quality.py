#!/usr/bin/env python3
"""Per-corpus SROCC delta vs the radius-5 baseline, for each model and radius.
The bar (era-2 §21.1, registered before any candidate existed):
  PASS iff no corpus loses more than 0.005 SROCC and the composite does not fall.
SROCC is compared in MAGNITUDE (several corpora carry a distortion-oriented
target, so their canonical sign is negative); a SIGN FLIP is reported as a
failure regardless of magnitude."""
import json, glob, os, sys
BASE=os.environ.get('ZBR_VERDICTS', '/mnt/v/output/zensim/blurradius-2026-08-31')
RADII=[5,4,3,2]
models=['B','C944','WLIN7b_g020','WLIN7b_g025']
def load(R,m):
    p=f'{BASE}/verdicts-r{R}/{m}.fulleval.json'
    return json.load(open(p)) if os.path.exists(p) else None
BAR=0.005
allrows=[]
for m in models:
    b=load(5,m)
    if not b: continue
    corpora=sorted(b['rank'])
    print(f"\n### {m}  (n_inputs={b.get('n_inputs')})\n")
    hdr=f"{'corpus':>10} " + " ".join(f"{'R='+str(R):>17}" for R in RADII)
    print(hdr)
    worst={R:0.0 for R in RADII}
    for c in corpora:
        bv=b['rank'][c].get('srocc_signed')
        if bv is None: continue
        cells=[]
        for R in RADII:
            d=load(R,m)
            v=d['rank'][c].get('srocc_signed') if d and c in d['rank'] else None
            if v is None: cells.append(f"{'--':>17}"); continue
            flip = (v<0) != (bv<0)
            delta = abs(v)-abs(bv)          # positive = gained magnitude
            worst[R]=min(worst[R],delta)
            mark='!FLIP' if flip else ''
            cells.append(f"{abs(v):7.4f} ({delta:+.4f}){mark:>1}")
        print(f"{c:>10} " + " ".join(f"{x:>17}" for x in cells))
    # composite
    cells=[]
    for R in RADII:
        d=load(R,m); cv=d.get('composite') if d else None
        cells.append(f"{cv:7.4f} ({cv-b['composite']:+.4f})" if cv is not None else f"{'--':>17}")
    print(f"{'COMPOSITE':>10} " + " ".join(f"{x:>17}" for x in cells))
    cells=[]
    for R in RADII:
        d=load(R,m)
        if not d: cells.append(f"{'--':>17}"); continue
        ok = worst[R] >= -BAR and d.get('composite',0) >= b['composite']
        cells.append(f"{'PASS' if ok else 'FAIL':>7} (worst {worst[R]:+.4f})")
        allrows.append((m,R,ok,worst[R],d.get('composite')-b['composite']))
    print(f"{'BAR':>10} " + " ".join(f"{x:>17}" for x in cells))
print("\n### Verdict against the registered bar (no corpus loses > 0.005; composite does not fall)\n")
print(f"{'model':>14} {'R':>2} {'worst corpus delta':>19} {'composite delta':>16} {'':>6}")
for m,R,ok,w,cd in allrows:
    print(f"{m:>14} {R:>2} {w:>+19.4f} {cd:>+16.4f} {'PASS' if ok else 'FAIL':>6}")
