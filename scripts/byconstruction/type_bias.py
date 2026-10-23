"""Per distortion type: mean (pooled prediction rank - pooled human rank), arm vs E34 base-c, seeds 0-2. Negative = the
model rates that type worse than humans do. Type names: KADID-10k and TID2013 type order; label = type - 1.
Usage: type_bias.py [ARM]   (cell directory name under ~/tmp/byc/cells; default mono-thr)"""
import json, sys, numpy as np, pandas as pd
from scipy.stats import rankdata
from pathlib import Path
panel = json.load(open('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json'))
cells = Path.home()/'tmp/byc/cells'
ARM = sys.argv[1] if len(sys.argv) > 1 else 'mono-thr'
KN = {1:'gblur',2:'lensblur',3:'motionblur',4:'colordiffuse',5:'colorshift',6:'colorquant',7:'sat1',8:'sat2',9:'jp2k',10:'jpeg',11:'whitenoise',12:'colornoise',13:'impulse',14:'multnoise',15:'denoise',16:'brighten',17:'darken',18:'meanshift',19:'jitter',20:'noneccpatch',21:'pixelate',22:'quantize',23:'colorblock',24:'sharpen',25:'contrast'}
TN = {1:'awgn',2:'awgn-color',3:'spatial-noise',4:'masked-noise',5:'hf-noise',6:'impulse',7:'quant',8:'gblur',9:'denoise',10:'jpeg',11:'jp2k',12:'jpeg-tx',13:'jp2k-tx',14:'pattern-noise',15:'blockwise',16:'meanshift',17:'contrast',18:'saturation',19:'mult-noise',20:'comfort-noise',21:'lossy-noisy',22:'cquant-dither',23:'chrom-aberr',24:'sparse-sampling'}
for d in panel['datasets']:
    f = d['name']
    if f not in ('kadid','tid2013'): continue
    names = KN if f=='kadid' else TN
    t = np.array(d['target']); cod = np.array(d['codec'])+1; n=len(t)
    rt = rankdata(t)/n
    out=[]
    for typ in sorted(set(cod)):
        m = cod==typ
        bt, bc = [], []
        for s in (0,1,2):
            pt = pd.read_csv(cells/ARM/f'without_{f}_s{s}'/f'pred_{f}.tsv',sep='\t').pred.to_numpy()
            pc = np.array(d['predictions']['base-c'][s])
            bt.append(np.mean(rankdata(pt)[m]/n - rt[m])); bc.append(np.mean(rankdata(pc)[m]/n - rt[m]))
        out.append((names[typ], np.mean(bt), np.mean(bc)))
    print(f"== {f}: mean (pred rank - human rank), negative = model rates the type WORSE than humans")
    for nm, a, b in sorted(out, key=lambda x: x[1]-x[2])[:8]:
        print(f"   {nm:14s} {ARM}={a:+.3f}  base-c={b:+.3f}  arm-base={a-b:+.3f}")
