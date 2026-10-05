#!/usr/bin/env python3
"""R5STEER2 cross-arithmetic and owner-size orchestration, unchanged scorer."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[2]
OUT = Path('/var/tmp/r5steer2')
ORIGINAL = Path('/var/tmp/r5steer')
sys.path.insert(0, str(REPO / 'scripts/rev4_featpot'))
from v2_common import dense_bake  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment(rev):
    env = dict(os.environ, RAYON_NUM_THREADS='1', ZENSIM_FORMULA_REV=str(rev), ZENSIM_PREPARED_STEERING='1')
    for key in ['ZENSIM_NEIGHBOUR_EXACT', 'ZENSIM_FINITE_MOMENTS', 'ZENSIM_REPAIR_ALPHA',
                'ZENSIM_REPAIR_SOURCE', 'ZENSIM_STEERING_BIN', 'ZENSIM_V2_DIAG']:
        env.pop(key, None)
    return env


if '--gradient' in sys.argv:
    (OUT / 'bin').mkdir(exist_ok=True)
    tool = OUT / 'bin/diffmap_block_coherence'
    built = OUT / 'target/release/examples/diffmap_block_coherence'
    if not tool.exists():
        shutil.copy2(built, tool)
    assert sha(tool) == sha(built)
    jobs = json.loads((OUT / 'JOBS.json').read_text())
    reports = []

    def gradient_one(name):
        src = Path(name)
        job = json.loads(src.read_text())
        folder = OUT / 'gradient'
        folder.mkdir(exist_ok=True)
        dst = folder / src.name
        if not dst.exists():
            with dst.with_suffix('.log').open('w') as log:
                subprocess.run([str(tool), '--gradient-check', str(src), '--json', str(dst)],
                    env=environment(job['revision']), stdout=log, stderr=subprocess.STDOUT, check=True)
        d = json.loads(dst.read_text())
        assert d['job_sha256'] == sha(src)
        return {'job': str(src), 'report': str(dst), 'sha256': sha(dst)}

    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(gradient_one, name) for name in jobs]):
            reports.append(future.result())
            (OUT / 'GRADIENT_PARTIAL.json').write_text(json.dumps(reports, indent=2) + '\n')
            print(f'gradient {len(reports)}/{len(jobs)}', flush=True)
    (OUT / 'GRADIENT_COMPLETE.json').write_text(json.dumps({'status': 'COMPLETE_UNQUALIFIED',
        'binary_sha256': sha(tool), 'reports': reports}, indent=2) + '\n')
    sys.exit(0)


inputs = json.loads((REPO / 'benchmarks/r5steer_2026-10-04/INPUTS.json').read_text())
models = [r for r in inputs['models'] if r['arm'] == 'fpbyv2fy']
cross = OUT / 'crossscreen'
cross.mkdir(parents=True, exist_ok=True)
for model in models:
    source = Path(model['source'].replace('/v2c5/', '/v2c/'))
    assert sha(source) == model['rev4_source_sha256']
    dense = dense_bake(source, ORIGINAL / 'rev4-cache')
    target = cross / Path(model['rev4']).name
    if not target.exists():
        subprocess.run([str(ORIGINAL / 'bin/bake_stamp_revision'), str(dense), '5', str(target)], check=True)
    model['cross'] = str(target)
    model['cross_sha256'] = sha(target)
    model['rev4_dense'] = str(dense)
(OUT / 'MODELS.json').write_text(json.dumps(models, indent=2) + '\n')
tool = inputs['binary']
assert sha(tool) == inputs['binary_sha256']
jobs = []
for model in models:
    for image in ['I01', 'I21', 'I41', 'I61']:
        jobs.append(('cross-owner', 5, model['cross'], image, 32))
for rev in [4, 5]:
    for block in [8, 16, 32, 64]:
        jobs.append((f'owner-rev{rev}', rev, models[2][f'rev{rev}'], 'I61', block))


def one(job):
    panel, rev, model, image, block = job
    folder = OUT / panel
    folder.mkdir(exist_ok=True)
    path = folder / f'{Path(model).stem}-{image}-b{block}.json'
    if not path.exists():
        with path.with_suffix('.log').open('w') as log:
            subprocess.run([tool, f'/mnt/v/dataset/kadid10k/images/{image}.png',
                f'/mnt/v/dataset/kadid10k/images/{image}_10_03.png', '--block', str(block),
                '--bake', model, '--json', str(path)], env=environment(rev), stdout=log,
                stderr=subprocess.STDOUT, check=True)
    d = json.loads(path.read_text())
    assert d['refinement_available'] and not d['refinement_unsupported_ids']
    return {'panel': panel, 'revision': rev, 'model': model, 'image': image, 'block': block,
            'm2': d['m2'], 'm3f': d['m3f'], 'pass': int(d['m2'] >= .99 and d['m3f'] >= .70),
            'report': str(path), 'sha256': sha(path)}


rows = []
with ThreadPoolExecutor(max_workers=4) as pool:
    for future in as_completed([pool.submit(one, job) for job in jobs]):
        rows.append(future.result())
        print(f'owner/cross {len(rows)}/{len(jobs)}', flush=True)
(OUT / 'OWNER.json').write_text(json.dumps(rows, indent=2) + '\n')
env = environment(5)
env.update(STEERCHECK_PREPARED='1', STEERCHECK_REV='5')
subprocess.run([sys.executable, str(REPO / 'benchmarks/steercheck_2026-10-02/run_broad.py'),
    str(cross), tool, str(OUT / 'cross-broad'), '4', 'fpbyv2fy'], env=env, check=True)
(OUT / 'CROSS_COMPLETE.json').write_text(json.dumps({'status': 'COMPLETE_UNQUALIFIED',
    'owner_cases': 20, 'broad_cases': 96, 'binary_sha256': sha(tool)}, indent=2) + '\n')
