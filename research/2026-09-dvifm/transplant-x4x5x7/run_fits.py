#!/usr/bin/env python3
"""transplant lane: emit fits/runs.json — the full paired-seed run grid.

Arms (keep-features on the 1040-wide master layout):
  step0   base@{s61k,s83k,s105k}   keep 0..227            train=size subset, dev=frozen
          perm30@{sizes}           keep 0..227            train=size perm30,  dev=v1 perm30
  X4      x4 / x4perm              keep 0..227+944..967   +24 pooled terms
  X5      x5 / x5perm              keep 0..227+980..1039  +60 chroma terms
  X7      x7t1{,p}                 keep 0..227+968..970   raw L0 block peak
          x7t2{,p}                 keep 0..227+971..973   across/within discriminator
          x7t3fix                  keep 0..227+974..976   fixed-phase boundary
          x7t3max{,p}              keep 0..227+977..979   8-phase max boundary

Dev tables: frozen dedup for step-0 base arms; v1 fits/perm30/dev for
step-0 perm30; the 1040-wide dev/ tables for X arms; perm/dev_* for
permuted X controls. Every run carries its full trainer command (`cmd`).
"""
import json, os, sys

OUT = '/mnt/v/output/zensim/dvifm-transplant-2026-09-20'
DEDUP = '/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables'
V1P30 = '/mnt/v/output/zensim/joint-core-v1/fits/perm30/dev'
BIN = '/home/lilith/work/zen/zensim/target/release/zensim_mlp_train'
SEEDS = [17101, 17103, 17107, 17111, 17113]
SSEED = {str(s): (i + 1) * 10**9 for i, s in enumerate(SEEDS)}
# group -> (train table, weight, val weight, modes) — the R915 recipe
G = {
    'safesyn': ('fresh_safesyn', '1.0168526508775275', '0.5', 'withinref,both'),
    'cid22': ('cid22', '1.0115735134169879', '2.0', 'withinref,both'),
    'human': ('human', '0.5041192364219411', '1.0', 'withinref,rank'),
    'codec': ('fresh_imazen26', '0.6060902647942771', '1.0', 'withinref,both'),
}

K227 = ','.join(str(i) for i in range(228))
def keep(extra):
    return K227 + ''.join(f',{i}' for i in extra)

ARMS = []
for size in ('s61k', 's83k', 's105k'):
    ARMS.append(dict(arm=f'base@{size}', size=size,
                     train=f'{OUT}/subsets/{size}', trainpat='{tleg}.parquet',
                     dev=DEDUP, devpat='{dleg}_development.parquet',
                     keep=keep([]), maxf=944))
    ARMS.append(dict(arm=f'perm30@{size}', size=size,
                     train=f'{OUT}/subsets/{size}/perm30',
                     trainpat='{tleg}.parquet',
                     dev=V1P30, devpat='{dleg}_development.parquet',
                     keep=keep([]), maxf=944))
for name, extra, perm in [
        ('x4', range(944, 968), False), ('x4perm', range(944, 968), True),
        ('x5', range(980, 1040), False), ('x5perm', range(980, 1040), True),
        ('x7t1', range(968, 971), False), ('x7t1p', range(968, 971), True),
        ('x7t2', range(971, 974), False), ('x7t2p', range(971, 974), True),
        ('x7t3fix', range(974, 977), False),
        ('x7t3max', range(977, 980), False),
        ('x7t3maxp', range(977, 980), True)]:
    ARMS.append(dict(arm=name, size='s105k',
                     train=f'{OUT}/perm' if perm else f'{OUT}/features',
                     trainpat='{tleg}.parquet',
                     dev=f'{OUT}/perm' if perm else f'{OUT}/dev',
                     devpat=('dev_{dleg}.parquet' if perm
                             else '{dleg}_development.parquet'),
                     keep=keep(extra), maxf=1040))


def cmd(a, s):
    g = []
    for dleg, (tleg, w, v, m) in G.items():
        g.append(f'{dleg}:{a["train"]}/{a["trainpat"].format(tleg=tleg)}'
                 f':{w}:0:{m}')
        g.append(f'{dleg}_development:{a["dev"]}/'
                 f'{a["devpat"].format(dleg=dleg)}:0:{v}:{m}')
    gg = ' '.join(f'--group "{x}"' for x in g)
    # 1040-wide arms read appended research cols (f944..f1039) that are not a
    # registered feature-set, so table admission runs under --historical-replay
    # (the f944+ coverage gap is recorded in the report, not guessed away).
    replay = (' --historical-replay "transplant-lane-2026-09-20: canonical-944 '
              '(ceiling_rev3) + unregistered appended research cols f944-1039 '
              '(X4 pool-24, X7 edge-12, dvifm-Cb30, dvifm-Cr30)"'
              if a['maxf'] == 1040 else '')
    return (f'{BIN} {gg} --target-column human_score --target-scale 1 '
            f'--hidden 128 --epochs 120 --pairs-per-epoch 50000 '
            f'--seed {s} --init-seed {s} --sample-seed {SSEED[str(s)]} '
            f'--pair-sampling uniform --max-features {a["maxf"]} '
            f'--keep-features {a["keep"]} '
            f'--mse-weight 1 --early-stop-patience 0 '
            f'--val-policy mean --val-aggregate geomean3 '
            f'--out-dtype f32 --log-every 1 --no-auto-eval{replay} '
            f'--out {OUT}/fits/{a["arm"]}_s{s}')


runs = []
for a in ARMS:
    for s in SEEDS:
        runs.append(dict(id=f"{a['arm']}_s{s}", seed=s,
                         sseed=SSEED[str(s)], cmd=cmd(a, s),
                         **{k: a[k] for k in
                            ('arm', 'size', 'keep', 'maxf')}))
os.makedirs(f'{OUT}/fits', exist_ok=True)
json.dump({'seeds': SEEDS, 'arms': ARMS, 'runs': runs},
          open(f'{OUT}/fits/runs.json', 'w'), indent=1)
print(len(runs), 'runs across', len(ARMS), 'arms')
