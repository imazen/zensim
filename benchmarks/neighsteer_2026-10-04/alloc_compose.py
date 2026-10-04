#!/usr/bin/env python3
"""Allocation test (neighbour-aware steering study): upgrade a fraction of 8x8 blocks of the q_lo decode to the q_hi decode,
chosen by a ranking, and write the composite PNG. Rankings: map (refinement_gain), oracle (true single-block ΔS), random,
anti (lowest map gain). Lossless PNG IO only (the composites are pixel-exact block swaps of zenjpeg 4:4:4 decodes)."""
import json, sys
import numpy as np
from PIL import Image
J = '/var/tmp/neighsteer/jpeg'; O = '/var/tmp/neighsteer/alloc'
img, qlo, qhi, frac = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
lo = np.array(Image.open(f'{J}/{img}_q{qlo}.png').convert('RGB')); hi = np.array(Image.open(f'{J}/{img}_q{qhi}.png').convert('RGB'))
d = json.load(open(f'/var/tmp/neighsteer/swap/by_v2fy-s5101-{img}-q{qlo}to{qhi}-b8.json'))
B = d['blocks']; n = len(B); k = int(round(frac * n))
keys = {'map': [-b['refinement_gain'] for b in B], 'oracle': [-b['score_delta'] for b in B],
        'random': list(np.random.default_rng(20261004).permutation(n)), 'anti': [b['refinement_gain'] for b in B]}
for name, key in keys.items():
    pick = np.argsort(key, kind='stable')[:k]
    out = lo.copy()
    for j in pick:
        x0, y0, x1, y1 = B[j]['bounds']; out[y0:y1, x0:x1] = hi[y0:y1, x0:x1]
    Image.fromarray(out).save(f'{O}/{img}-q{qlo}to{qhi}-f{frac}-{name}.png')
print(img, qlo, qhi, frac, n, k)
