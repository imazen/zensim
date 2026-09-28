#!/usr/bin/env python3
"""Independent synthetic parity gate for Rust scatter math and single-owner IO."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lib.zen_stats import scatter

rng = np.random.default_rng(914)
for n in (100, 1001):
    target = rng.normal(size=n)
    pred = target + rng.normal(scale=.2, size=n)
    got = scatter(pred, target)
    order = np.argsort(pred)
    mapped = np.empty(n); mapped[order] = np.sort(target)
    residual = mapped - target
    mad = np.median(np.abs(residual - np.median(residual))) * 1.4826
    span = np.percentile(target, 99) - np.percentile(target, 1)
    np.testing.assert_allclose(got["normalized_pred"], mapped, atol=1e-12)
    np.testing.assert_allclose([got["geo"]["mad"], got["geo"]["p99d"], got["geo"]["maxd"]],
        [mad, np.percentile(np.abs(residual), 99)/span, np.max(np.abs(residual))/span], rtol=1e-10)
    line = np.polyfit(target, pred, 1)
    raw = pred - np.polyval(line, target)
    bins, _ = np.histogram(pred, bins=20)
    np.testing.assert_allclose([got['raw']['cov'], got['raw']['clump']],
        [np.mean(bins >= np.ceil(n/200)), np.max(bins)/n])
    scale = 1.4826*np.median(np.abs(raw-np.median(raw)))
    np.testing.assert_allclose([got["raw"]["p99"], got["raw"]["max"]],
        [np.percentile(np.abs(raw),99)/scale,np.max(np.abs(raw))/scale],rtol=1e-10)
    tied = np.round(pred)
    a = scatter(tied,target); perm = rng.permutation(n); b = scatter(tied[perm],target[perm])
    np.testing.assert_allclose([a["geo"][k] for k in ("out4","p99d","maxd","clump")],
                               [b["geo"][k] for k in ("out4","p99d","maxd","clump")],rtol=1e-10)
constant = scatter([7.3]*100,[7.3]*100)
assert constant['geo']['clampLo']==constant['geo']['clampHi']==1
assert constant['geo']['out4']==0
assert constant['raw']['p99'] is None
target = np.linspace(0, 1, 1000)
compressed = scatter(target**10, target)
assert compressed['geo']['maxd'] == 0
assert compressed['raw']['clump'] > .7
for x,y in [([1]*100,[1]*99),([float('nan')]+[1]*99,[1]*100)]:
    try: scatter(x,y)
    except ValueError: pass
    else: raise AssertionError('bad input admitted')
print('scatter: independent NumPy parity, ties, saturation and input refusal pass')
