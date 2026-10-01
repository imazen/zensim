"""Canon binary set (zensim main, TRAINERMEM landed) vs the fleet v2 set: trainer, predictor, panel byte parity."""
import json, os, re, subprocess, sys
from pathlib import Path
sys.path.insert(0, '/var/tmp/fitv2/tools/scripts/jobsys')
import harvest_fit_cells as h
OLD = Path('/var/tmp/fitv2/bin-v2'); NEW = Path(sys.argv[1]); OUT = Path('/var/tmp/fitv2/canon_parity')
ARGV = json.load(open('/home/lilith/tmp/zensim-paper/rev4/argv.json'))
env = dict(os.environ, ZENSIM_MAX_TIER='v3', RAYON_NUM_THREADS='4')
rows = []

def train(binary, argv, out, log):
    a = [str(binary)] + argv[1:]
    a[a.index('--out') + 1] = str(out); a[a.index('--epochs') + 1] = '2'
    subprocess.run(a, stdout=open(log, 'w'), stderr=subprocess.STDOUT, env=env, check=True)

def curve(log):
    keep = []
    for line in open(log):
        if 'val(geomean3)' in line:
            line = re.sub(r'\| t=[0-9.]+s', '', line)
            keep.append(' | '.join(seg for seg in line.split(' | ')
                                   if not re.match(r'\s*(safesyn|cid22|human):', seg)))  # train-only segments (D1)
    return keep

for tag, fam in (('r0', 'main'), ('oracle_hi', 'aux')):
    argv = [x.replace('/wide/main/real/', f'/wide/{fam}/real/') for x in ARGV]
    if tag == 'oracle_hi':
        keep = OUT / 'keep_oracle_hi.txt'
        keep.write_text('\n'.join(str(i) for i in json.load(open('/var/tmp/rev4-featpot/v2/wide/keep_lists.json'))['specs']['oracle_hi']['keep']) + '\n')
        argv[argv.index('--keep-features') + 1] = str(keep)
    for which, b in (('old', OLD), ('new', NEW)):
        train(b / 'zensim_mlp_train', argv, OUT / f'{tag}_{which}.bin', OUT / f'{tag}_{which}.log')
    rows.append((f'trainer {tag} weights', h.weights_sha(OUT / f'{tag}_old.bin') == h.weights_sha(OUT / f'{tag}_new.bin')))
    rows.append((f'trainer {tag} dev curve', curve(OUT / f'{tag}_old.log') == curve(OUT / f'{tag}_new.log')))

cells = sorted(Path('/var/tmp/rev4-featpot/v2/cells').glob('*__*/without_*_s0'))[:8]
for c in cells:
    res = json.loads((c / 'result.json').read_text())
    fam, var = res['family'], res['variant']
    table = Path(f'/var/tmp/rev4-featpot/v2/wide/{fam}/{var}/{res["heldout"]}.parquet')
    outs = []
    for which, b in (('old', OLD), ('new', NEW)):
        o = OUT / f'pred_{c.parent.name}_{c.name}_{which}.tsv'
        subprocess.run([str(b / 'bake_dial_refit'), 'predict', '--bake', str(c / 'refit/best.bin'), '--corpus', str(table),
                        '--score-units', '--out', str(o)], stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env, check=True)
        outs.append(o.read_bytes())
    rows.append((f'predict {c.parent.name}/{c.name}', outs[0] == outs[1]))

sys.path.insert(0, str(Path.home() / 'work/zen/zensim--featbank-potential/scripts'))
import importlib
for which, b in (('old', OLD), ('new', NEW)):
    os.environ['ZEN_PANEL_BIN'] = str(b / 'panel')
    import lib.zen_stats as zs; importlib.reload(zs)
    import numpy as np
    rng = np.random.default_rng(5); x = rng.normal(size=3000); y = x + rng.normal(size=3000)
    r = zs.panel_batch([('a', x, y), ('b', y, x ** 3)], stats='full')
    (OUT / f'panel_{which}.json').write_text(json.dumps(r, sort_keys=True))
rows.append(('panel full stats', (OUT / 'panel_old.json').read_text() == (OUT / 'panel_new.json').read_text()))
for name, ok in rows:
    print(f'{"PASS" if ok else "FAIL"}  {name}')
print('ALL_IDENTICAL' if all(ok for _, ok in rows) else 'DIFFERENCES')
