#!/usr/bin/env python3
"""gmsd lane: parse fit logs -> reports/fit_results.json + a markdown table.

Metric (identical to the zgeom/block5 scorers): the trainer's
`best validation mean SROCC` (mean policy, geomean3 aggregate over the four
dev legs), and the per-dev-leg SROCC printed on that best epoch. Every
contrast is a per-seed PAIRED difference (same seed, same sample stream).

Contrasts:
  arm − G          does the added block help at all?
  arm − armp       does it beat its size-matched permuted control? (the gate:
                   a gain that does not beat its control is not a gain)
  armp − G         what the extra width alone does (column-count artifact)
Held-out development legs: the same three contrasts on the mean of the four
dev-leg SROCCs at the best epoch, plus per-leg signs.
"""
import json, os, re, glob, sys
import numpy as np

L = '/var/tmp/gmsd-lane'
SEEDS = [17101, 17103, 17107, 17111, 17113]
RE = re.compile(r'best validation mean SROCC = ([0-9.]+)')
LEGS = ('safesyn', 'cid22', 'human', 'codec')
LEG_RE = {leg: re.compile(rf'{leg}_development: srocc=([0-9.]+)') for leg in LEGS}
BEST_RE = re.compile(r'epoch (\d+) .*?best=([0-9.]+)')
VAL_RE = re.compile(r'val\(geomean3\)=([0-9.]+) \(best=([0-9.]+)\)')


def load():
    scores, legs = {}, {}
    for f in glob.glob(f'{L}/fits/logs*/*.log'):
        rid = os.path.basename(f)[:-4]
        txt = open(f).read()
        m = RE.findall(txt)
        if not m:
            continue
        best = float(m[-1])
        scores[rid] = best
        # the per-leg values printed on the epoch whose val equals the best
        for line in txt.splitlines():
            vm = VAL_RE.search(line)
            if vm and float(vm.group(1)) == best:
                legs[rid] = {leg: float(LEG_RE[leg].search(line).group(1))
                             for leg in LEGS if LEG_RE[leg].search(line)}
                break
    return scores, legs


def stats(d):
    if not d:
        return None
    v = np.array(list(d.values()))
    sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    mean = float(np.mean(v))
    t = mean / (sd / np.sqrt(len(v))) if sd > 0 else (float('inf') if mean > 0 else 0.0)
    return dict(n=len(v), mean=mean, min=float(v.min()), max=float(v.max()),
                pos=int((v > 0).sum()), neg=int((v < 0).sum()), paired_t=float(t),
                per_seed={str(s): float(d[s]) for s in sorted(d)})


def paired(get, a, b):
    out = {}
    for s in SEEDS:
        x, y = get(f'{a}_s{s}'), get(f'{b}_s{s}')
        if x is not None and y is not None:
            out[s] = x - y
    return out


def main():
    scores, legs = load()
    arms = sorted({k.rsplit('_s', 1)[0] for k in scores})
    devagg = {k: float(np.mean([v[l] for l in LEGS])) for k, v in legs.items()
              if all(l in v for l in LEGS)}
    res = {'seeds': SEEDS, 'scores': scores, 'dev_legs': legs, 'dev_agg': devagg,
           'arms': arms, 'val': {}, 'dev': {}, 'dev_leg_signs': {}}
    pairs = []
    for a in arms:
        if a == 'G' or a.endswith('p'):
            continue
        pairs += [(f'{a}-G', a, 'G'), (f'{a}-{a}p', a, a + 'p'), (f'{a}p-G', a + 'p', 'G')]
    for name, a, b in pairs:
        res['val'][name] = stats(paired(scores.get, a, b))
        res['dev'][name] = stats(paired(devagg.get, a, b))
        res['dev_leg_signs'][name] = {
            leg: stats(paired(lambda k, leg=leg: legs.get(k, {}).get(leg), a, b))
            for leg in LEGS}
    os.makedirs(f'{L}/reports', exist_ok=True)
    json.dump(res, open(f'{L}/reports/fit_results.json', 'w'), indent=1)
    print('| contrast | val mean Δ | pos/neg | t | dev-agg mean Δ | pos/neg | t | dev legs (saf/cid/hum/cod mean Δ) |')
    print('|---|---|---|---|---|---|---|---|')
    for name, _, _ in pairs:
        v, d = res['val'][name], res['dev'][name]
        if not v:
            continue
        lg = res['dev_leg_signs'][name]
        legs_s = ' / '.join(f"{lg[l]['mean']:+.4f}" if lg[l] else '—' for l in LEGS)
        print(f"| {name} | {v['mean']:+.4f} | {v['pos']}/{v['neg']} | {v['paired_t']:+.2f} | "
              f"{d['mean']:+.4f} | {d['pos']}/{d['neg']} | {d['paired_t']:+.2f} | {legs_s} |")


if __name__ == '__main__':
    main()
