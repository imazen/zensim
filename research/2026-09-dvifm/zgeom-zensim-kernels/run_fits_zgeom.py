#!/usr/bin/env python3
"""zgeom lane: emit fits/runs.json — the full paired-seed run grid.

Z2 kernel arms (all keep 0..227 on a fresh-rev3 228-wide surface):
  z2box2     box2 pyramid    (baseline — also the Z1 baseline)
  z2bin121   bin121 pyramid
  z2bin1331  bin1331 pyramid

Z1 pooling arms (box2 pyramid, same rows):
  z1gate     block-peak x two-state gate  (the REPLACEMENT arm)
  z1max      ungated block-peak           (decomposition)
  z1gateperm z1gate, all 228 cols row-permuted  (control)
  z1maxperm  z1max,  all 228 cols row-permuted  (control)

Same R915 recipe/seeds/groups as the transplant lane; per-arm dev tables
(feature values differ per arm — that IS the experiment).
"""
import json, os

OUT = '/mnt/v/output/zensim/zgeom-2026-09-21'
BIN = '/home/lilith/work/zen/zensim/target/release/zensim_mlp_train'
SEEDS = [17101, 17103, 17107, 17111, 17113]
SSEED = {str(s): (i + 1) * 10**9 for i, s in enumerate(SEEDS)}
G = {
    'safesyn': ('fresh_safesyn', '1.0168526508775275', '0.5', 'withinref,both'),
    'cid22': ('cid22', '1.0115735134169879', '2.0', 'withinref,both'),
    'human': ('human', '0.5041192364219411', '1.0', 'withinref,rank'),
    'codec': ('fresh_imazen26', '0.6060902647942771', '1.0', 'withinref,both'),
}
K227 = ','.join(str(i) for i in range(228))
ARMS = []
for k in ('box2', 'bin121', 'bin1331'):
    ARMS.append(dict(arm=f'z2{k}', size='s105k', kind='kernel',
                     spec=f'zgeom_{k}_glob',
                     train=f'{OUT}/z2/{k}/features',
                     dev=f'{OUT}/z2/{k}/dev'))
for a, tok, spec in (('z1gate', 'b5gate', 'zgeom_box2_b5gate'),
                     ('z1max', 'b5max', 'zgeom_box2_b5max'),
                     ('z1gateperm', 'b5gate', 'zgeom_box2_b5gate@perm'),
                     ('z1maxperm', 'b5max', 'zgeom_box2_b5max@perm')):
    ARMS.append(dict(arm=a, size='s105k', kind='pool',
                     spec=spec,
                     train=f'{OUT}/z1/{a}/features',
                     dev=f'{OUT}/z1/{a}/dev'))


def cmd(a, s):
    g = []
    for dleg, (tleg, w, v, m) in G.items():
        g.append(f'{dleg}:{a["train"]}/{tleg}.parquet:{w}:0:{m}')
        g.append(f'{dleg}_development:{a["dev"]}/'
                 f'{dleg}_development.parquet:0:{v}:{m}')
    gg = ' '.join(f'--group "{x}"' for x in g)
    replay = (' --historical-replay "zgeom-lane-2026-09-21: basic+peaks 228 '
              f'under spec {a["spec"]} — kernel/pool arms never column-mix"')
    return (f'{BIN} {gg} --target-column human_score --target-scale 1 '
            f'--hidden 128 --epochs 120 --pairs-per-epoch 50000 '
            f'--seed {s} --init-seed {s} --sample-seed {SSEED[str(s)]} '
            f'--pair-sampling uniform --max-features 228 '
            f'--keep-features {K227} '
            f'--mse-weight 1 --early-stop-patience 0 '
            f'--val-policy mean --val-aggregate geomean3 '
            f'--out-dtype f32 --log-every 1 --no-auto-eval{replay} '
            f'--out {OUT}/fits/{a["arm"]}_s{s}')


runs = []
for a in ARMS:
    for s in SEEDS:
        runs.append(dict(id=f"{a['arm']}_s{s}", seed=s,
                         sseed=SSEED[str(s)], cmd=cmd(a, s),
                         arm=a['arm'], kind=a['kind'], spec=a['spec'],
                         size=a['size'], keep='0..227', maxf=228))
os.makedirs(f'{OUT}/fits', exist_ok=True)
json.dump({'seeds': SEEDS, 'arms': ARMS, 'runs': runs},
          open(f'{OUT}/fits/runs.json', 'w'), indent=1)
print(len(runs), 'runs across', len(ARMS), 'arms')
