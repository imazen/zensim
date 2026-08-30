#!/usr/bin/env python3
"""Per-slot drift between a STORED 372-col feature table and a FRESH extraction.

Alignment key: (ref_basename normalized by stripping a trailing image extension,
round(human_score, 9)). Refuses if the key is not unique on either side or the
common set is smaller than either table.
`--positional` compares row i of A with row i of B (guarded: the two tables
must have the same length and elementwise-equal `human_score`). Use it when the
two tables come from the SAME loader over the SAME label file — then row order
is the alignment, and it is the only correct one for a corpus whose
(ref, human_score) key repeats (KADID 64.8 % of rows, AIC-3 100 %, TID 24.2 %).
Key-based pairing collapses such a group onto one member and silently compares
an image with itself; positional does not.

`--ordinal` disambiguates corpora whose (ref, human_score) key legitimately
repeats (KADID/TID/CSIQ/AIC-3 all have distorted variants that share a score):
the Nth occurrence of a key on side A is compared with the Nth on side B, and
the run REFUSES if the two sides disagree on any group's size. Without the flag
a repeated key is still a hard refusal, so the drift lane's recorded CID22 /
kon504 numbers reproduce from this file unchanged.

This is the maintained copy; the as-run instrument of the 2026-08-30 drift study
is frozen (sha-recorded) at /mnt/v/output/zensim/v1-extractor-drift-2026-08-30/.

Block layout (zensim/src/metric.rs combine_scores, block-major):
  f0..155   basic  (scale*39 + ch*13 + k)
  f156..227 peaks  (156 + scale*18 + ch*6 + k)
  f228..299 masked (228 + scale*18 + ch*6 + k)
  f300..371 IW     (300 + scale*18 + ch*6 + k)
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

def _ordinals(keys):
    """0-based occurrence index of each key within its own group, in row order."""
    seen = {}
    out = []
    for k in keys:
        n = seen.get(k, 0)
        out.append(n)
        seen[k] = n + 1
    return out

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

def main(a_path, b_path, label_a, label_b, out_json=None, topn=25, ordinal=False,
         positional=False):
    ka, fa = load(a_path)
    kb, fb = load(b_path)
    if positional:
        if len(ka) != len(kb):
            sys.exit(f"REFUSE (--positional): row counts differ ({len(ka)} vs {len(kb)})")
        bad = [i for i, (x, y) in enumerate(zip(ka, kb)) if abs(x[1] - y[1]) > 1e-12]
        if bad:
            sys.exit(f"REFUSE (--positional): human_score differs on {len(bad)} rows, "
                     f"e.g. row {bad[0]} ({ka[bad[0]]} vs {kb[bad[0]]})")
        ka = kb = list(range(len(ka)))
        print(f"# --positional: row-order alignment over {len(ka)} rows "
              f"(human_score elementwise-equal)")
    ma, mb = {}, {}
    for i, k in enumerate(ka): ma.setdefault(k, []).append(i)
    for i, k in enumerate(kb): mb.setdefault(k, []).append(i)
    dupa = sum(1 for v in ma.values() if len(v) > 1)
    dupb = sum(1 for v in mb.values() if len(v) > 1)
    if dupa or dupb:
        if not ordinal:
            sys.exit(f"REFUSE: non-unique key ({dupa} dup in A, {dupb} in B)")
        # Ordinal mode: the Nth occurrence of a key on A pairs with the Nth on B.
        # Legitimate whenever both tables enumerate the SAME label file in the
        # same order (all of KADID/TID/CSIQ/AIC-3 do). Sizes must agree — a
        # mismatch means the row sets are not the same and positional pairing
        # would silently compare different images.
        bad = [k for k in (set(ma) | set(mb))
               if len(ma.get(k, [])) != len(mb.get(k, []))]
        if bad:
            sys.exit(f"REFUSE (--ordinal): {len(bad)} keys have different group "
                     f"sizes in A vs B, e.g. {bad[:3]}")
        ka = [(k, n) for k, n in zip(ka, _ordinals(ka))]
        kb = [(k, n) for k, n in zip(kb, _ordinals(kb))]
        ma, mb = {}, {}
        for i, k in enumerate(ka): ma.setdefault(k, []).append(i)
        for i, k in enumerate(kb): mb.setdefault(k, []).append(i)
        print(f"# --ordinal: {dupa} repeated keys disambiguated by occurrence index")
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
    flags = {'--ordinal', '--positional'}
    argv = [a for a in sys.argv[1:] if a not in flags]
    main(*argv[:4], out_json=(argv[4] if len(argv) > 4 else None),
         ordinal=('--ordinal' in sys.argv), positional=('--positional' in sys.argv))
