"""E34 draft: characterize G-STEER failures from STORED engineering-packet outputs only.

No model is run, no pixel or label is read. Inputs are the per-block rows the registered G-STEER
runs already wrote (steerfix_packet::engineering_packet): per block, the exact full-repair score change
(`score_delta`) and its first-order prediction from the base sensitivities (`linearized_gain`).
M2 = Spearman(linearized_gain, score_delta) over a case's blocks; M3f = Spearman(refinement_gain, score_delta).

usage: characterize.py OUT_DIR
Writes per_case.{json,tsv} and summary.json. Every number the E34 draft quotes from stored rows comes from summary.json.
Populations: SERVED = the seven served models (A, C seeds 0-2; production seed 0); dense bakes are excluded from every
summary. "Failing" excludes production's four floor-tied KADID JPEG cases (M2 = 0) unless stated.
"""
import json
import math
import statistics
import sys
from pathlib import Path

E33 = Path('/mnt/v/output/zensim/e33-impl-2026-10-09/gates')
QA = Path('/mnt/v/output/zensim/qual-a-2026-10-10')
RUNS = {  # model label -> stored G-STEER output (registered runs; seed-0 A/C/production from QUAL-A)
    'A-s0': QA / 'steer-s0-a.json', 'A-s1': E33 / 'steer-s1-a.json', 'A-s2': E33 / 'steer-s2-a.json',
    'C-s0': QA / 'steer-s0-c.json', 'C-s1': E33 / 'steer-s1-c.json', 'C-s2': E33 / 'steer-s2-c.json',
    'P-s0': QA / 'steer-s0-seed0.json',
    'A-s0-dense': E33 / 'steer-s0-a-dense.json', 'C-s0-dense': E33 / 'steer-s0-c-dense.json',
}
ROSTER = Path('/mnt/v/output/zensim/prodqual-b-2026-10-07/STEERING_PREREAD.json')


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    ma, mb = statistics.fmean(a), statistics.fmean(b)
    num = sum((x - ma) * (y - mb) for x, y in zip(a, b))
    den = math.sqrt(sum((x - ma) ** 2 for x in a) * sum((y - mb) ** 2 for y in b))
    return num / den if den else float('nan')


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def case_stats(row):
    lin = [b['linearized_gain'] for b in row['blocks']]
    tru = [b['score_delta'] for b in row['blocks']]
    n = len(tru)
    rl, rt = ranks(lin), ranks(tru)
    d2 = [(x - y) ** 2 for x, y in zip(rl, rt)]
    tot = sum(d2) or 1.0
    top = sorted(range(n), key=lambda i: -d2[i])
    # Leave-out: M2 after dropping the k most displaced blocks (concentration of the miss).
    drop = {}
    for k in (1, 2, 3):
        keep = [i for i in range(n) if i not in set(top[:k])]
        drop[k] = spearman([lin[i] for i in keep], [tru[i] for i in keep]) if len(keep) > 2 else float('nan')
    absg = sorted(abs(t) for t in tru)
    pct = lambda v: sum(1 for a in absg if a <= abs(v)) / n
    over = sum(1 for x, y in zip(lin, tru) if x >= y)
    rel_l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(lin, tru)) / (sum(y * y for y in tru) or 1.0))
    ratio = [y / x for x, y in zip(lin, tru) if abs(x) > 1e-12]
    return dict(n_blocks=n, m2=row['m2'], m3f=row['m3f'], passes=row['pass'], base=row['base_score'],
                top3_share_of_d2=sum(d2[i] for i in top[:3]) / tot,
                m2_drop1=drop[1], m2_drop2=drop[2], m2_drop3=drop[3],
                top3_gain_percentiles=[round(pct(tru[i]), 3) for i in top[:3]],
                frac_linear_ge_true=over / n, rel_l2_lin_vs_true=rel_l2,
                median_true_over_linear=statistics.median(ratio) if ratio else float('nan'),
                zero_true_blocks=sum(1 for t in tru if t == 0.0), max_true_gain=max(tru), sum_true_gain=sum(tru))


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    roster = {(r['case'], r['block']): r for r in json.loads(ROSTER.read_text())['cases']}
    table = {}
    for label, path in RUNS.items():
        rows = json.loads(path.read_text())['rows']
        assert len(rows) == 135, (label, len(rows))
        for row in rows:
            block = row['blocks'][0]['bounds'][2] - row['blocks'][0]['bounds'][0]
            key = (row['key'], block)
            meta = roster[key]
            rec = case_stats(row)
            rec.update(model=label, case=row['key'], block=block, panel=meta['panel'], ref=meta['ref_group'],
                       width=meta['width'], height=meta['height'],
                       level=meta['source_distorted'].rsplit('-', 1)[-1].split('.')[0] if meta['panel'] == 'broad' else None)
            table[(label, row['key'], block)] = rec
    recs = list(table.values())
    (out / 'per_case.json').write_text(json.dumps(recs, indent=0) + '\n')
    cols = ['model', 'case', 'block', 'panel', 'ref', 'width', 'height', 'level', 'n_blocks', 'm2', 'm3f', 'passes', 'base',
            'top3_share_of_d2', 'm2_drop1', 'm2_drop2', 'm2_drop3', 'frac_linear_ge_true', 'rel_l2_lin_vs_true',
            'median_true_over_linear', 'zero_true_blocks']
    with (out / 'per_case.tsv').open('w') as f:
        f.write('\t'.join(cols) + '\n')
        for r in recs:
            f.write('\t'.join(str(round(r[c], 6) if isinstance(r[c], float) else r[c]) for c in cols) + '\n')
    (out / 'summary.json').write_text(json.dumps(summary(recs), indent=1) + '\n')
    print('runs', {k: str(v) for k, v in RUNS.items()})


SERVED = ('A-s0', 'A-s1', 'A-s2', 'C-s0', 'C-s1', 'C-s2', 'P-s0')


def summary(recs):
    srv = [r for r in recs if r['model'] in SERVED]
    floor = lambda r: r['panel'] == 'jpeg8' and r['m2'] == 0.0
    fail = [r for r in srv if not r['passes']]
    nonfloor = [r for r in fail if not floor(r)]
    out = {'pass_counts': {m: sum(r['passes'] for r in recs if r['model'] == m) for m in RUNS},
           'failures_served': len(fail), 'failures_nonfloor': len(nonfloor),
           'ac_failures_all_m2': all(r['m2'] < 0.99 for r in fail if r['model'][0] in 'AC'),
           'ac_failures': sum(1 for r in fail if r['model'][0] in 'AC')}
    for key in ('block', 'panel', 'ref', 'level'):
        tot = {}
        bad = {}
        for r in srv:
            tot[str(r[key])] = tot.get(str(r[key]), 0) + 1
        for r in fail:
            bad[str(r[key])] = bad.get(str(r[key]), 0) + 1
        out[f'fail_by_{key}'] = {k: [bad.get(k, 0), tot[k]] for k in sorted(tot)}
    out['drop1_rescues_by_block'] = {str(b): [sum(1 for r in nonfloor if r['block'] == b and r['m2_drop1'] >= 0.99),
                                             sum(1 for r in nonfloor if r['block'] == b)] for b in (8, 16, 32, 64)}
    pairs = {}
    for r in fail:
        pairs.setdefault(f"{r['case']}@b{r['block']}", []).append(r['model'])
    out['failing_pairs'] = len(pairs)
    out['failing_pairs_in_2plus_models'] = sum(1 for v in pairs.values() if len(v) >= 2)
    out['most_frequent_pair'] = max(pairs.items(), key=lambda kv: len(kv[1]))
    rel = {}
    for scope, keep in (('all_panels', lambda r: True), ('broad_only', lambda r: r['panel'] == 'broad')):
        rel[scope] = {}
        for b in (8, 16, 32, 64):
            f_ = [r['rel_l2_lin_vs_true'] for r in nonfloor if r['block'] == b and keep(r)]
            p_ = [r['rel_l2_lin_vs_true'] for r in srv if r['passes'] and r['block'] == b and keep(r)]
            mf, mp = statistics.median(f_), statistics.median(p_)
            rel[scope][str(b)] = dict(failing_median=mf, passing_median=mp, ratio=mf / mp, n_failing=len(f_), n_passing=len(p_))
    out['rel_l2_by_block'] = rel
    broad_pass = [r['base'] for r in srv if r['panel'] == 'broad' and r['passes']]
    broad_fail = [r['base'] for r in fail if r['panel'] == 'broad']
    out['broad_base_median'] = dict(passing=statistics.median(broad_pass), failing=statistics.median(broad_fail))
    out['base_median_by_level_seed0'] = {lv: statistics.median(r['base'] for r in recs if r['model'] in ('A-s0', 'C-s0', 'P-s0') and r['level'] == lv)
                                         for lv in ('10', '13', '17')}
    return out


if __name__ == '__main__':
    main(sys.argv[1])
