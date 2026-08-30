#!/usr/bin/env python3
"""Per-slot DRIFT CHARACTERIZATION between two 372-col feature tables.

Answers "which slots moved, by how much, on how many rows, and does it clear
the golden tolerance" for a (stored, fresh) pair of v1-372 feature tables —
the instrument behind `benchmarks/v1_extractor_drift_2026-08-30.md` and
`docs/DATASET_HISTORY.md` §3.27.

WHY THIS IS NOT A DUPLICATE (per zensim/CLAUDE.md "NO DUPLICATE
IMPLEMENTATIONS"; read this before writing a third one):

  * `gate_backfill944.py` is the BITWISE PASS/FAIL gate for a 944-vs-924
    pair. It requires row-order identity, compares at the stored dtype, and
    exits 0/1. It cannot characterize a drift (no magnitudes, no grouping)
    and cannot align two tables that were built by different tools.
  * `promote_ext944_canonical.py`'s `EXT944_DRIFT_ROOT` reports drift, but
    POSITIONALLY and as four scalars per leg, inline in a promoter.
  * This tool KEY-JOINS (`ref_basename` with the image extension stripped,
    `round(human_score, 9)`; it REFUSES if that key is not unique on either
    side) and reports, per slot and grouped by block and scale: max_abs,
    max_rel, the fraction of rows differing at all, and the count of cells
    outside the repo's golden tolerance.

It computes NO IQA statistic. Every SROCC/PLCC/Z-RMSE in the record above
comes from `bake_verdict` -> `zensim_validate::panel` -> `zenstats`; nothing
here is a second implementation of any of those.

Tolerance is the repo's golden policy (zensim/CLAUDE.md, 2026-08-05 user
ruling): `|d| <= max(1e-6 abs, 1e-5 * scale)`, `scale = max(|a|, |b|)`.

Inputs may be `.parquet` or the extractors' `.csv`; both need
`ref_basename`, `human_score`, `f0..f371` (extra columns are ignored).

Block layout (`zensim/src/metric.rs::combine_scores`, block-major then
scale-major then channel):
  f0..155   basic  (scale*39 + ch*13 + k)
  f156..227 peaks  (156 + scale*18 + ch*6 + k)
  f228..299 masked (228 + scale*18 + ch*6 + k)
  f300..371 IW     (300 + scale*18 + ch*6 + k)

usage: feature_table_drift.py A B label_a label_b [out.json]
"""

import sys, os, json, math
import pyarrow.parquet as pq
import pyarrow.csv as pacsv

EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif', '.jxl')

def norm(r):
    rl = r.lower()
    for e in EXT:
        if rl.endswith(e):
            return r[: -len(e)]
    return r

def load(path):
    if path.endswith('.parquet'):
        t = pq.read_table(path)
    else:
        t = pacsv.read_csv(path)
    d = t.to_pydict()
    n = len(d['ref_basename'])
    keys = [(norm(d['ref_basename'][i]), round(float(d['human_score'][i]), 9)) for i in range(n)]
    feats = [[float(d[f'f{j}'][i]) for j in range(372)] for i in range(n)]
    return keys, feats

def block_of(j):
    if j < 156: return 'basic'
    if j < 228: return 'peaks'
    if j < 300: return 'masked'
    return 'iw'

def scale_of(j):
    if j < 156: return j // 39
    return ((j - 156) % 216) // 18 if False else ((j - (156 if j < 228 else (228 if j < 300 else 300))) // 18)

def chan_of(j):
    if j < 156: return (j % 39) // 13
    base = 156 if j < 228 else (228 if j < 300 else 300)
    return ((j - base) % 18) // 6

def sub_of(j):
    if j < 156: return j % 13
    base = 156 if j < 228 else (228 if j < 300 else 300)
    return (j - base) % 6

PEAKNM = ['ssim_max','art_max','det_max','ssim_p95','art_p95','det_p95']
MSKNM  = ['masked_ssim0','masked_ssim1','masked_ssim2','masked_art_4th','masked_det_4th','masked_mse']
IWNM   = ['iw_ssim0','iw_ssim1','iw_ssim2','iw_art_4th','iw_det_4th','iw_mse']
BASNM  = ['ssim_a','ssim_b','ssim_2nd','edge0','edge1','edge_2nd0','edge2','edge3','edge_2nd1','mse','hf_e_loss','hf_m_loss','hf_e_gain']

def slotname(j):
    b = block_of(j)
    nm = {'basic': BASNM, 'peaks': PEAKNM, 'masked': MSKNM, 'iw': IWNM}[b]
    return f"{b}/s{scale_of(j)}/c{chan_of(j)}/{nm[sub_of(j)]}"

def main(a_path, b_path, label_a, label_b, out_json=None, topn=25):
    ka, fa = load(a_path)
    kb, fb = load(b_path)
    ma, mb = {}, {}
    for i, k in enumerate(ka): ma.setdefault(k, []).append(i)
    for i, k in enumerate(kb): mb.setdefault(k, []).append(i)
    dupa = sum(1 for v in ma.values() if len(v) > 1)
    dupb = sum(1 for v in mb.values() if len(v) > 1)
    if dupa or dupb:
        sys.exit(f"REFUSE: non-unique key ({dupa} dup in A, {dupb} in B)")
    common = sorted(set(ma) & set(mb))
    print(f"# A={label_a} rows={len(ka)}  B={label_b} rows={len(kb)}  common={len(common)}")
    if len(common) != len(ka) or len(common) != len(kb):
        print(f"# WARNING: key sets differ (A-only {len(ma)-len(common)}, B-only {len(mb)-len(common)})")
    rows = []
    for j in range(372):
        mx_abs = 0.0; mx_rel = 0.0; ndiff = 0
        for k in common:
            x = fa[ma[k][0]][j]; y = fb[mb[k][0]][j]
            if x != y:
                ndiff += 1
                d = abs(x - y)
                if d > mx_abs: mx_abs = d
                den = max(abs(x), abs(y))
                r = d / den if den > 0 else float('inf')
                if r > mx_rel: mx_rel = r
        rows.append(dict(slot=j, name=slotname(j), block=block_of(j), scale=scale_of(j),
                         chan=chan_of(j), max_abs=mx_abs, max_rel=mx_rel,
                         frac_diff=ndiff / len(common), n=len(common)))
    # rows/cells over the golden-policy tolerance, per block
    def over(x, y):
        return abs(x - y) > max(1e-6, 1e-5 * max(abs(x), abs(y)))
    print(f"\n## rows/cells OVER golden tolerance |d|<=max(1e-6,1e-5*scale), per block")
    print(f"{'block':8} {'rows_over':>9} {'frac_rows':>9} {'cells_over':>10} {'max_abs_over':>13} {'max_rel_over':>12}")
    blkinfo = {}
    for b in ['basic', 'peaks', 'masked', 'iw']:
        js = [j for j in range(372) if block_of(j) == b]
        rows_over = 0; cells = 0; mxa = 0.0; mxr = 0.0
        for k in common:
            ia_, ib_ = ma[k][0], mb[k][0]
            hit = False
            for j in js:
                x = fa[ia_][j]; y = fb[ib_][j]
                if over(x, y):
                    hit = True; cells += 1
                    d = abs(x - y)
                    if d > mxa: mxa = d
                    den = max(abs(x), abs(y)); r = d / den if den > 0 else float('inf')
                    if r > mxr: mxr = r
            if hit: rows_over += 1
        blkinfo[b] = (rows_over, cells, mxa, mxr)
        print(f"{b:8} {rows_over:9d} {rows_over/len(common):9.4f} {cells:10d} {mxa:13.6g} {mxr:12.4g}")
    # block summary
    print(f"\n## block summary ({label_a} vs {label_b})")
    print(f"{'block':8} {'slots':>5} {'slots_diff':>10} {'max_abs':>12} {'max_rel':>10} {'max_frac_rows':>13}")
    for b in ['basic', 'peaks', 'masked', 'iw']:
        sub = [r for r in rows if r['block'] == b]
        sd = [r for r in sub if r['frac_diff'] > 0]
        print(f"{b:8} {len(sub):5d} {len(sd):10d} {max(r['max_abs'] for r in sub):12.6g} "
              f"{max(r['max_rel'] for r in sub):10.4g} {max(r['frac_diff'] for r in sub):13.4f}")
    print(f"\n## per (block,scale)")
    print(f"{'block':8} {'scale':>5} {'slots_diff':>10} {'max_abs':>12} {'max_rel':>10} {'max_frac':>9}")
    for b in ['basic', 'peaks', 'masked', 'iw']:
        for s in range(4):
            sub = [r for r in rows if r['block'] == b and r['scale'] == s]
            if not sub: continue
            sd = [r for r in sub if r['frac_diff'] > 0]
            print(f"{b:8} {s:5d} {len(sd):10d} {max(r['max_abs'] for r in sub):12.6g} "
                  f"{max(r['max_rel'] for r in sub):10.4g} {max(r['frac_diff'] for r in sub):9.4f}")
    print(f"\n## top {topn} slots by max_rel")
    print(f"{'slot':>4} {'name':34} {'max_abs':>12} {'max_rel':>10} {'frac_rows':>9}")
    for r in sorted(rows, key=lambda r: -r['max_rel'])[:topn]:
        print(f"{r['slot']:4d} {r['name']:34} {r['max_abs']:12.6g} {r['max_rel']:10.4g} {r['frac_diff']:9.4f}")
    # golden policy: |d| <= max(1e-6 abs, 1e-5*scale) where scale = max(|x|,|y|)
    viol = 0; viol_slots = set()
    for j in range(372):
        for k in common:
            x = fa[ma[k][0]][j]; y = fb[mb[k][0]][j]
            tol = max(1e-6, 1e-5 * max(abs(x), abs(y)))
            if abs(x - y) > tol:
                viol += 1; viol_slots.add(j)
    print(f"\n## golden-policy tolerance |d| <= max(1e-6, 1e-5*scale): "
          f"{viol} violating cells over {len(common)*372} ({100.0*viol/(len(common)*372):.3f}%), "
          f"{len(viol_slots)} of 372 slots violate on >=1 row")
    if out_json:
        json.dump(dict(blkinfo={k:list(v) for k,v in blkinfo.items()}, a=label_a, b=label_b, a_path=a_path, b_path=b_path, n_common=len(common),
                       viol_cells=viol, viol_slots=sorted(viol_slots), slots=rows),
                  open(out_json, 'w'), indent=1)
        print(f"# wrote {out_json}")

if __name__ == '__main__':
    main(*sys.argv[1:5], out_json=(sys.argv[5] if len(sys.argv) > 5 else None))
