#!/usr/bin/env python3
"""transplant lane: assemble the final benchmark record.

Consumes every fits/logs/*.train.log, then emits
  benchmarks/dvifm_transplant_2026-09-20.json

Layers on top of score_fits.py's fit_results.json:
  * convergence — best val SROCC reached by epoch<=50 vs by epoch<=100
    (plan gate: dev metric must be rising-or-flat, not falling)
  * domain coverage — per dev-leg srocc at the best-val epoch, per arm,
    so a later regression is attributable to a leg
"""
import json, os, re, glob
import numpy as np

OUT = '/mnt/v/output/zensim/dvifm-transplant-2026-09-20'
BENCH = '/home/lilith/work/zen/zensim--transplant/benchmarks'
SEEDS = [17101, 17103, 17107, 17111, 17113]
RE_BEST = re.compile(r'best validation mean SROCC = ([0-9.]+)')
RE_EP = re.compile(
    r'epoch\s+(\d+)\s+\|.*?val\(geomean3\)=([0-9.]+).*?\| '
    r'safesyn_development: srocc=([0-9.]+).*?\| '
    r'cid22_development: srocc=([0-9.]+).*?\| '
    r'human_development: srocc=([0-9.]+).*?\| '
    r'codec_development: srocc=([0-9.]+)')
DEV_LEGS = ['safesyn_development', 'cid22_development',
            'human_development', 'codec_development']


def parse_log(path):
    best = None
    traj = []            # (epoch, val, [4 dev srocc])
    for line in open(path, errors='replace'):
        m = RE_BEST.search(line)
        if m:
            best = float(m.group(1))
        e = RE_EP.search(line)
        if e:
            traj.append((int(e.group(1)), float(e.group(2)),
                         [float(e.group(k)) for k in (3, 4, 5, 6)]))
    rec = {'best': best}
    if traj:
        # convergence: val reading at the epoch-50 and epoch-100 checkpoints
        by_ep = {ep: v for ep, v, _ in traj}
        rec['val_by50'] = max(v for ep, v, _ in traj if ep <= 50) \
            if any(ep <= 50 for ep, _, _ in traj) else None
        rec['val_by100'] = max(v for ep, v, _ in traj if ep <= 100) \
            if any(ep <= 100 for ep, _, _ in traj) else None
        rec['val_at50'] = by_ep.get(50)
        rec['val_at100'] = by_ep.get(100)
        # dev-leg srocc at the epoch that hit the best val
        bep, _,bdev = max(traj, key=lambda t: t[1])
        rec['dev_at_best'] = dict(zip(DEV_LEGS, bdev))
        rec['best_epoch'] = bep
    return rec


def main():
    runs = {}
    for f in sorted(glob.glob(f'{OUT}/fits/logs/*.train.log')):
        rid = os.path.basename(f)[:-10]
        runs[rid] = parse_log(f)

    # per-arm convergence + coverage aggregates
    arms = {}
    for rid, r in runs.items():
        arm = rid.rsplit('_s', 1)[0]
        a = arms.setdefault(arm, {'seeds': {}, 'n': 0})
        a['seeds'][rid.rsplit('_s', 1)[1]] = r
        a['n'] += 1
    for arm, a in arms.items():
        b50 = [r['val_by50'] for r in a['seeds'].values()
               if r.get('val_by50') is not None]
        b100 = [r['val_by100'] for r in a['seeds'].values()
                if r.get('val_by100') is not None]
        a['conv'] = {'val_by50_mean': float(np.mean(b50)) if b50 else None,
                     'val_by100_mean': float(np.mean(b100)) if b100 else None,
                     'rising_or_flat': (float(np.mean(b100)) >=
                                        float(np.mean(b50)) - 1e-4)
                     if b50 and b100 else None}
        cov = {}
        for leg in DEV_LEGS:
            vals = [r['dev_at_best'][leg] for r in a['seeds'].values()
                    if r.get('dev_at_best', {}).get(leg) is not None]
            cov[leg] = {'mean': float(np.mean(vals)) if vals else None,
                        'min': float(np.min(vals)) if vals else None}
        a['dev_coverage'] = cov

    fit_results = {}
    frp = f'{OUT}/reports/fit_results.json'
    if os.path.exists(frp):
        fit_results = json.load(open(frp))
    x7 = {}
    xp = f'{OUT}/reports/x7_agreement.json'
    if os.path.exists(xp):
        x7 = json.load(open(xp))
    tab = {}
    tp = f'{OUT}/manifests/tables_report.json'
    if os.path.exists(tp):
        tab = json.load(open(tp))

    doc = {
        'lane': 'transplant', 'date': '2026-09-20',
        'core': {'name': 'joint-core-v2', 'pairs': 105614,
                 'cohorts': {'v1': 52963, 'v2reused': 8331,
                             'v2fresh': 44320}},
        'seeds': SEEDS,
        'metric': 'best validation mean SROCC (geomean3 over 4 dev legs)',
        'admission': {'step0': 'qualified w944/ceiling_rev3',
                      'x_arms': '--historical-replay (unregistered '
                                'appended cols f944-1039)'},
        'step0_sensitivity': fit_results.get('step0', {}),
        'x4': fit_results.get('x4', {}),
        'x5': fit_results.get('x5', {}),
        'x7_fits': fit_results.get('x7', {}),
        'x7_within_image_agreement': x7,
        'convergence': {arm: a['conv'] for arm, a in arms.items()},
        'dev_coverage': {arm: a['dev_coverage'] for arm, a in arms.items()},
        'runs': runs,
        'tables_report': tab,
    }
    os.makedirs(BENCH, exist_ok=True)
    json.dump(doc, open(f'{BENCH}/dvifm_transplant_2026-09-20.json', 'w'),
              indent=1)
    print('arms:', {k: v['n'] for k, v in sorted(arms.items())})
    print('wrote benchmarks/dvifm_transplant_2026-09-20.json')


def fmt(x, n=5):
    return f'{x:.{n}f}' if isinstance(x, (int, float)) else str(x)


def results_md(fr):
    """Render the Results section markdown from score_fits' fit_results."""
    L = []
    s0 = fr.get('step0', {})
    L.append('### Sensitivity curve (per-seed cost, real−perm30)')
    for size in ('s53k_v1', 's61k', 's83k', 's105k'):
        e = s0.get(size)
        if not e or e.get('base_mean') is None:
            continue
        g = e.get('gate') or {}
        cm = e.get('cost_mean', (e.get('cost') or {}).get('mean'))
        L.append(f'- **{size}**: base {fmt(e.get("base_mean"))} '
                 f'(spread {fmt(e.get("base_spread"))}), '
                 f'cost {fmt(cm)} → '
                 f'ratio {fmt(g.get("cost_over_floor"),2)} '
                 f'**{g.get("verdict","?")}**')
    base105 = (s0.get('s105k') or {}).get('base') or {}
    for exp in ('x4', 'x5', 'x7'):
        d = fr.get(exp)
        if not d:
            continue
        L.append(f'\n### {exp.upper()} — paired diffs vs base@s105k '
                 f'(same seed), sign counts, permuted control')
        for arm, a in d.items():
            if not isinstance(a, dict) or 'vs_base' not in a:
                continue
            vb = a['vs_base']
            if not vb:
                continue
            L.append(f'\n`{arm}` vs base — mean {fmt(vb["mean"],4)} '
                     f'[{fmt(vb["min"],4)}..{fmt(vb["max"],4)}], '
                     f'pos/neg/zero {vb["pos"]}/{vb["neg"]}/{vb["zero"]}:')
            per = vb['per_seed']
            L.append('| seed | arm | base | diff |')
            L.append('|---|---|---|---|')
            for s, diff in per.items():
                L.append(f'| {s} | {fmt(a["scores"].get(int(s)))} | '
                         f'{fmt(base105.get(int(s)))} | '
                         f'{fmt(diff,4)} |')
        for k, a in d.items():
            if '_vs_' not in k or not a:
                continue
            L.append(f'\n`{k}` — mean {fmt(a["mean"],4)} '
                     f'[{fmt(a["min"],4)}..{fmt(a["max"],4)}], '
                     f'pos/neg/zero {a["pos"]}/{a["neg"]}/{a["zero"]}')
    return '\n'.join(L)


if __name__ == '__main__':
    main()
