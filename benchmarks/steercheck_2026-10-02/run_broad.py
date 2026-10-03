#!/usr/bin/env python3
"""Broad spatial panel for the STEERCHECK candidates: 24 TRAIN pairs x blocks 8/16/32/64 = 96 cases per candidate,
through the current `diffmap_block_coherence` with the uniform ensemble of each candidate's three seeds.
Orchestration only; the scorer, the finite repairs and M2/M3f are the existing Rust owners. Adapted from the
09-13 steerable `reproduce/zensim_steering_broad_dense.py` (same register, same gates M2>=.99, M3f>=.70).

usage: run_broad.py <screen-output-dir> <diffmap_block_coherence binary> <out-dir> [workers] [arm ...]
"""
import hashlib, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

screen, tool, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
workers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
only = sys.argv[5:]
REG = Path('/home/lilith/work/zensim-validation-2026-09-08/max-attribution')
SEEDS = [5101, 5103, 5107]
BLOCKS = [8, 16, 32, 64]
MIN_M2, MIN_M3F = 0.99, 0.70


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


reg = json.load(open(REG / 'PIXEL_REGISTER.json'))
sources = json.load(open(REG / 'INPUTS.json'))['sources']['sources']
assert all(x['split'] == 'train' for x in sources)
train_origins = {s['origin'] for s in sources}
for c in reg['inputs']:
    assert c['origin'] in train_origins
    assert sha(c['reference']) == c['reference_sha256'] and sha(c['decoded']) == c['decoded_file_sha256']
arms = sorted({p.name.split('-')[1] for p in screen.glob('human-*-h128-full-s5101.bin')})
if only:
    arms = [a for a in arms if a in only]
models = {}
for a in arms:
    ps = [screen / f'human-{a}-h128-full-s{s}.bin' for s in SEEDS]
    assert all(p.exists() for p in ps), a
    models[a] = ps
out.mkdir(parents=True, exist_ok=True)
manifest = {'schema': 'steercheck-broad-v1', 'qualified': False, 'binary_sha256': sha(tool),
            'register_sha256': sha(REG / 'PIXEL_REGISTER.json'), 'blocks': BLOCKS, 'gates': {'m2': MIN_M2, 'm3f': MIN_M3F},
            'models': {n: [{'path': str(p), 'sha256': sha(p)} for p in ps] for n, ps in models.items()},
            'selection': 'uniform three-seed ensembles; TRAIN-role pairs; development diagnostic, not held-out evidence',
            'rows': []}
# Default: the cached compute_with_ref_and_attribution path the owner's audit also uses (before STEERAPI,
# prepare_steering refused any model reading IDs >= 228). STEERCHECK_PREPARED=1 serves through prepare_steering.
env = dict(os.environ, RAYON_NUM_THREADS='1', ZENSIM_FORMULA_REV='3')  # the screen's Rev3 pixels
if os.environ.get('STEERCHECK_PREPARED') == '1':
    # Serve through BakeScorer::prepare_steering (STEERAPI); v2 bakes need it to accept IDs >= 228.
    env['ZENSIM_PREPARED_STEERING'] = '1'



def one(job):
    name, c, block = job
    path = out / f"{name}-{c['index']}-b{block}.json"
    if not path.exists():
        ps = models[name]
        cmd = [str(tool), c['reference'], c['decoded'], '--block', str(block), '--json', str(path),
               '--ensemble', ','.join(map(str, ps)), '--ensemble-weights', ','.join([str(1 / len(ps))] * len(ps))]
        with path.with_suffix('.log').open('w') as f:
            subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    d = json.load(path.open())
    unsupported = (not d['refinement_available']) or bool(d['refinement_unsupported_ids'])
    status = 'UNSUPPORTED' if unsupported else ('PASS' if d['m2'] >= MIN_M2 and d['m3f'] >= MIN_M3F else 'FAIL')
    return {'model': name, 'index': c['index'], 'origin': c['origin'], 'content_class': c['content_class'],
            'distance': c['distance'], 'block': block, 'score': d['base_score'], 'm2': d['m2'], 'm3f': d['m3f'],
            'interventions': d['pixel_interventions'], 'status': status,
            'unsupported_ids': d['refinement_unsupported_ids'], 'report': path.name, 'sha256': sha(path)}


jobs = [(n, c, b) for n in models for c in reg['inputs'] for b in BLOCKS]
with ThreadPoolExecutor(max_workers=workers) as pool:
    for i, row in enumerate(pool.map(one, jobs)):
        manifest['rows'].append(row)
        if i % 24 == 23:
            print(i + 1, '/', len(jobs), flush=True)
manifest['status'] = 'COMPLETE_UNQUALIFIED'
(out / 'RESULT.json').write_text(json.dumps(manifest, indent=1) + '\n')
for n in models:
    rows = [r for r in manifest['rows'] if r['model'] == n]
    cnt = {s: sum(r['status'] == s for r in rows) for s in ('PASS', 'FAIL', 'UNSUPPORTED')}
    print(n, len(rows), cnt, 'min m2 %.4f min m3f %.4f' % (min(r['m2'] for r in rows), min(r['m3f'] for r in rows)), flush=True)
