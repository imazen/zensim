#!/usr/bin/env python3
"""R5STEER orchestration; all scoring and gates remain with existing Rust owners.

Run through scripts/run-heavy; scratch is /var/tmp/r5steer, four single-thread
workers. Broad cases use the existing run_broad.py unchanged.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = Path(__file__).resolve().parents[2]
OUT = Path('/var/tmp/r5steer')
OLD = Path('/var/tmp/neighsteer')
IMAGES = Path('/mnt/v/dataset/kadid10k/images')
SEEDS = [5101, 5103, 5107]
ARMS = {'fpbyv2fy': 'sel:59f0bbc2f290', 'fpv2basic': 'set:v2+basic'}
sys.path.insert(0, str(REPO / 'scripts/rev4_featpot'))
from v2_common import dense_bake, FITBIN  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def prepare():
    (OUT / 'bin').mkdir(parents=True, exist_ok=True)
    for name, src in {
        'diffmap_block_coherence': Path('/home/lilith/work/zen/zensim--featbank-potential/target/release/examples/diffmap_block_coherence'),
        'bake_stamp_revision': Path('/var/tmp/r5integ/target-root/release/bake_stamp_revision'),
    }.items():
        dst = OUT / 'bin' / name
        if not dst.exists():
            shutil.copy2(src, dst)
    tool = OUT / 'bin/diffmap_block_coherence'
    assert sha(tool) == '28e22ee289fa859da84a29a12639e01115de354c41095f4f16d37b84c7d7c8ec'
    manifest = {'schema': 'r5steer-inputs-v1', 'qualified': False,
                'binary': str(tool), 'binary_sha256': sha(tool),
                'binary_build_commit': '70a5066e', 'workspace_base': 'c83d22cd',
                'stamp_sha256': sha(OUT / 'bin/bake_stamp_revision'),
                'densify_binary': str(FITBIN), 'densify_sha256': sha(FITBIN),
                'workers': 4, 'rayon_threads': 1, 'gates': {'m2': .99, 'm3f': .70},
                'models': [], 'kadid_inputs': []}
    for arm, spec in ARMS.items():
        for i, label in enumerate(SEEDS):
            cell = Path('/var/tmp/rev4-featpot/v2c5/cells') / f'{spec}@h32:H128:cv16:cf98__N/without_kadid_s{i}'
            receipt = json.loads((cell / 'fleet_receipt.json').read_text())
            result = json.loads((cell / 'result.json').read_text())
            assert result['seed_index'] == i
            assert result['init_seed'] == [1101, 1103, 1107][i]
            assert result['sample_seed'] == [101, 100000101, 200000101][i]
            src = cell / 'refit/last.bin'
            assert sha(src) == receipt['selected_bake_sha'] == receipt['files']['refit/last.bin']
            dense = dense_bake(src, OUT / 'cache')
            screen = OUT / 'rev5screen'
            screen.mkdir(exist_ok=True)
            dst = screen / f'human-{arm}-h128-full-s{label}.bin'
            if not dst.exists():
                subprocess.run([str(OUT / 'bin/bake_stamp_revision'), str(dense), '5', str(dst)], check=True)
            old = OLD / 'fpscreen' / dst.name
            old_cell = Path('/var/tmp/rev4-featpot/v2c/cells') / f'{spec}@h32:H128:cv16:cf98__N/without_kadid_s{i}'
            manifest['models'].append({'arm': arm, 'seed_index': i, 'legacy_label': label,
                'source': str(src), 'source_sha256': sha(src), 'dense_sha256': sha(dense),
                'rev5': str(dst), 'rev5_sha256': sha(dst), 'rev4': str(old), 'rev4_sha256': sha(old),
                'fleet_receipt_sha256': sha(cell / 'fleet_receipt.json'), 'receipt': receipt,
                'init_seed': result['init_seed'], 'sample_seed': result['sample_seed'],
                'rev4_source_sha256': sha(old_cell / 'refit/last.bin')})
    for image in ['I01', 'I21', 'I41', 'I61']:
        for level in ['03', '05']:
            ref, dist = IMAGES / f'{image}.png', IMAGES / f'{image}_10_{level}.png'
            manifest['kadid_inputs'].append({'case': f'{image}_10_{level}', 'reference': str(ref),
                'distorted': str(dist), 'reference_file_sha256': sha(ref), 'distorted_file_sha256': sha(dist)})
    write(OUT / 'INPUTS.json', manifest)
    return manifest


def panels(manifest):
    jobs = []
    for revision, exact, panel in [(5, False, 'kadidjpeg'), (5, False, 'owner'),
                                    (5, True, 'kadid_exact'), (4, True, 'kadid_exact')]:
        for model in manifest['models']:
            for pair in manifest['kadid_inputs']:
                if panel == 'owner' and not pair['case'].endswith('_03'):
                    continue
                jobs.append((revision, exact, panel, model, pair))

    def one(job):
        rev, exact, panel, model, pair = job
        block = 32 if panel == 'owner' else 8
        folder = OUT / f'rev{rev}' / panel
        folder.mkdir(parents=True, exist_ok=True)
        name = f"{model['arm']}-s{model['legacy_label']}-{pair['case']}-b{block}"
        path = folder / f'{name}.json'
        env = dict(os.environ, RAYON_NUM_THREADS='1', ZENSIM_FORMULA_REV=str(rev), ZENSIM_PREPARED_STEERING='1')
        for key in ['ZENSIM_NEIGHBOUR_EXACT', 'ZENSIM_FINITE_MOMENTS', 'ZENSIM_REPAIR_ALPHA', 'ZENSIM_REPAIR_SOURCE', 'ZENSIM_STEERING_BIN', 'ZENSIM_V2_DIAG']:
            env.pop(key, None)
        if exact:
            env['ZENSIM_NEIGHBOUR_EXACT'] = '1'
        if not path.exists():
            with (folder / f'{name}.log').open('w') as log:
                subprocess.run([manifest['binary'], pair['reference'], pair['distorted'], '--block', str(block),
                    '--bake', model[f'rev{rev}'], '--json', str(path)], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        d = json.loads(path.read_text())
        old_panel = 'kadidjpeg' if panel == 'kadid_exact' else panel
        baseline = json.loads((OLD / 'fppanels' / old_panel / f'{name}.json').read_text())
        for key in ['reference_pixels_sha256', 'distorted_pixels_sha256', 'block_size', 'pixel_interventions']:
            assert d[key] == baseline[key], (name, key)
        assert d['models'][0]['sha256'] == model[f'rev{rev}_sha256']
        assert d['refinement_available'] and not d['refinement_unsupported_ids']
        return {'revision': rev, 'neighbour_exact': exact, 'panel': panel, 'arm': model['arm'],
                'seed_index': model['seed_index'], 'seed_label': model['legacy_label'], 'case': pair['case'],
                'level': pair['case'][-2:], 'block': block, 'm2': d['m2'], 'm3f': d['m3f'],
                'pass': d['m2'] >= .99 and d['m3f'] >= .70, 'report': str(path), 'sha256': sha(path)}

    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one, job) for job in jobs]):
            rows.append(future.result())
            write(OUT / 'PANELS_PARTIAL.json', rows)
            if len(rows) % 4 == 0:
                print(f'panels {len(rows)}/{len(jobs)}', flush=True)
    write(OUT / 'PANELS.json', sorted(rows, key=lambda x: (x['revision'], x['panel'], x['arm'], x['seed_index'], x['case'])))


if __name__ == '__main__':
    inputs = prepare()
    panels(inputs)
    env = dict(os.environ, STEERCHECK_PREPARED='1', STEERCHECK_REV='5', RAYON_NUM_THREADS='1')
    for key in ['ZENSIM_NEIGHBOUR_EXACT', 'ZENSIM_FINITE_MOMENTS', 'ZENSIM_REPAIR_ALPHA', 'ZENSIM_REPAIR_SOURCE', 'ZENSIM_STEERING_BIN', 'ZENSIM_V2_DIAG']:
        env.pop(key, None)
    subprocess.run([sys.executable, str(REPO / 'benchmarks/steercheck_2026-10-02/run_broad.py'),
        str(OUT / 'rev5screen'), inputs['binary'], str(OUT / 'rev5/broad'), '4'], env=env, check=True)
    write(OUT / 'RUN_COMPLETE.json', {'status': 'COMPLETE_UNQUALIFIED', 'panel_cases': 168, 'broad_cases': 192})
