#!/usr/bin/env python3
"""gmsd lane: emit fits/runs.json — the leaders' recipe (R915 / zgeom /
block5: hidden 128, 120 epochs, 50k uniform pairs/epoch, mse 1, mean val
policy, geomean3, f32, no early stop) over five paired seeds.

Arms: G (the zgeom z2_box2 228-col tables, the block5 baseline),
A_gms/A_gmsp (+12 GMS-std, and its permuted control),
A_dev/A_devp (+96 map-std, and its permuted control), plus any
replacement arm named with --extra NAME=WIDTH (tables under TAB/NAME).

Usage: run_fits.py [--seeds 17101,...] [--list a,b] [--extra R_x=240]
"""
import json, os, sys

L = '/var/tmp/gmsd-lane'
OUT = f'{L}/fits'
TAB = f'{L}/tables'
ZG = '/mnt/v/output/zensim/zgeom-2026-09-21'
BIN = f'{L}/bin/zensim_mlp_train'
SEEDS = [17101, 17103, 17107, 17111, 17113]
SSEED = {str(s): (i + 1) * 10**9 for i, s in enumerate(SEEDS)}
G = {
    'safesyn': ('fresh_safesyn', '1.0168526508775275', '0.5', 'withinref,both'),
    'cid22': ('cid22', '1.0115735134169879', '2.0', 'withinref,both'),
    'human': ('human', '0.5041192364219411', '1.0', 'withinref,rank'),
    'codec': ('fresh_imazen26', '0.6060902647942771', '1.0', 'withinref,both'),
}
WIDTH = {'G': 228, 'A_gms': 240, 'A_gmsp': 240, 'A_dev': 324, 'A_devp': 324}


def root(arm):
    return f'{ZG}/z2/box2' if arm == 'G' else f'{TAB}/{arm}'


def cmd(arm, s, w):
    r = root(arm)
    keep = ','.join(str(i) for i in range(w))
    g = []
    for dleg, (tleg, wt, v, m) in G.items():
        g.append(f'{dleg}:{r}/features/{tleg}.parquet:{wt}:0:{m}')
        g.append(f'{dleg}_development:{r}/dev/{dleg}_development.parquet:0:{v}:{m}')
    gg = ' '.join(f'--group "{x}"' for x in g)
    replay = ('' if arm == 'G' else
              f' --historical-replay "gmsd-lane-2026-09-22: composite zgeom box2:glob + '
              f'gms/map-deviation columns under arm {arm} — component sets in _MANIFEST.json"')
    if arm == 'G':
        replay = (' --historical-replay "gmsd-lane-2026-09-22: G baseline on the zgeom '
                  'z2_box2 tables (block5 recipe), refit with this lane\'s trainer build"')
    return (f'{BIN} {gg} --target-column human_score --target-scale 1 '
            f'--hidden 128 --epochs 120 --pairs-per-epoch 50000 '
            f'--seed {s} --init-seed {s} --sample-seed {SSEED[str(s)]} '
            f'--pair-sampling uniform --max-features {w} '
            f'--keep-features {keep} --allow-narrow-features '
            f'--mse-weight 1 --early-stop-patience 0 '
            f'--val-policy mean --val-aggregate geomean3 '
            f'--out-dtype f32 --log-every 1 --no-auto-eval{replay} '
            f'--out {OUT}/{arm}_s{s}')


def main():
    seeds, only = SEEDS, None
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == '--seeds':
            seeds = [int(x) for x in args[i + 1].split(',')]; i += 2
        elif args[i] == '--list':
            only = set(args[i + 1].split(',')); i += 2
        elif args[i] == '--extra':
            n, w = args[i + 1].split('='); WIDTH[n] = int(w); i += 2
        else:
            raise SystemExit(f'unknown arg {args[i]}')
    arms = [a for a in WIDTH if only is None or a in only]
    runs = [dict(id=f'{a}_s{s}', seed=s, arm=a, width=WIDTH[a], cmd=cmd(a, s, WIDTH[a]))
            for a in arms for s in seeds]
    os.makedirs(OUT, exist_ok=True)
    json.dump({'seeds': seeds, 'runs': runs}, open(f'{OUT}/runs.json', 'w'), indent=1)
    print(len(runs), 'runs across', len(arms), 'arms')


if __name__ == '__main__':
    main()
