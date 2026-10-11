"""Add the small-G Student interval to clustered.json (IQA guidance 2026-10-11 §3):
estimate ± t(G−1) · se · √(G/(G−1)), with se and G from zenmetrics-stats' cluster bootstrap and the t quantile from
scipy (an independent reference for a distribution constant; zenmetrics-stats has no public Student helper).

usage: student.py clustered.json
"""
import json
import math
import sys

from scipy.stats import t

path = sys.argv[1]
d = json.load(open(path))
for r in d['results']:
    g = r['clusters']
    q = t.ppf(0.975, g - 1)
    half = q * r['se'] * math.sqrt(g / (g - 1))
    r['student'] = {'lo': r['estimate'] - half, 'hi': r['estimate'] + half, 't_quantile': q, 'df': g - 1,
                    'formula': 'estimate ± t(G-1)·se·sqrt(G/(G-1))', 't_source': 'scipy.stats.t.ppf(0.975, G-1)'}
    f = math.exp if 'log' in r['statistic'] else (lambda x: x)
    print(f"{r['statistic'][:66]:66} G={g:2} est={f(r['estimate']):.3f} pct=[{f(r['lo']):.3f}, {f(r['hi']):.3f}] "
          f"student=[{f(r['student']['lo']):.3f}, {f(r['student']['hi']):.3f}] p0={r['p_zero']:.4f} finite={r['n_finite']}")
json.dump(d, open(path, 'w'), indent=1)
