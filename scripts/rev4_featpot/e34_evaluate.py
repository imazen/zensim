"""E34 frozen assessment adapters; statistics stay in the Rust panel owner.

The preread inventory and committed exposure ledger must precede target access.
Original fitted cells and frozen packets are never modified.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
from datetime import datetime, timezone

R = Path('/mnt/v/output/zensim/e34-launch-2026-10-15')
OUT = R / 'evaluation'
ROOT = Path('/mnt/v/output/zensim/v40r4-2026-10-08/v2e29')
FOLDS = ['kadid', 'tid2013', 'konfig', 'cid22_a25']
COUNTS = [7869, 3000, 756, 2192]
GROUPS = [65, 25, 7, 25]
LABELS = {'controla': 'base-a', 'controlc': 'base-c', 'wa': 'w-a',
          'wc': 'w-c', 'k1a': 'k-a', 'k10c': 'k-c', 'la': 'l-a'}
MODEL_ORDER = [f'{arm}-s{s}' for arm in ['w-a', 'k-a', 'l-a', 'w-c', 'k-c', 'base-a', 'base-c']
               for s in range(3)]
os.environ.setdefault('REV4_V2_BIN_DIR', str(R / 'bin'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def capture():
    import pandas as pd
    import pyarrow.parquet as pq
    from v2_lodo_mlp import predict
    ledger = Path(__file__).resolve().parents[2] / 'docs/DATA_SPLITS.md'
    assert 'E34 registered evaluation preread' in ledger.read_text()
    inventory = OUT / 'preread.json'
    assert sha(inventory) == 'e3aae632f8cc4537bb91c48c3de88ce039b4c691c8bd0e77c7025715590d1909'
    cells = json.loads(inventory.read_text())['cells']
    grids = {label: {} for label in LABELS.values()}
    for c in cells:
        name = c['jobset'].split('-')[1]
        if name in LABELS and c['fold'] != 'full':
            assert sha(c['selected_model']) == c['selected_sha256']
            key = (c['fold'], c['seed'])
            assert key not in grids[LABELS[name]]
            grids[LABELS[name]][key] = c
    expected = {(f, s) for f in FOLDS for s in range(10)}
    assert all(set(grid) == expected for grid in grids.values())
    receipt = json.loads((ROOT / 'wide/main/real/receipt.json').read_text())
    admitted = []
    for fold, count, groups in zip(FOLDS, COUNTS, GROUPS, strict=True):
        rec = receipt['legs'][fold]['full']
        table = ROOT / rec['rel']
        keys = table.with_suffix('.keys.parquet')
        assert sha(table) == rec['sha256']
        assert sha(str(table) + '.manifest.json') == rec['manifest_sha256']
        assert sha(keys) == rec['keys_sha256']
        meta = pq.read_table(keys).to_pandas()
        assert len(meta) == count
        names = meta.ref_basename.tolist()
        if fold == 'konfig':
            names = [n.removesuffix('_parta').removesuffix('_partb') for n in names]
        ids = {v: i for i, v in enumerate(sorted(set(names)))}
        assert len(ids) == groups
        members = sorted(set(meta.member_set))
        joins = []
        for member in members:
            path = Path('/var/tmp/rev4-featbank/bank') / member / 'keys.parquet'
            assert sha(path) == receipt['bank'][member]['keys_sha256']
            data = pq.read_table(path, columns=['pair_key', 'codec', 'dist_path']).to_pandas()
            if fold in ('kadid', 'tid2013'):
                data['codec'] = [Path(p).stem.split('_')[1] for p in data.dist_path]
            joins.append(data[['pair_key', 'codec']])
        joined = pd.concat(joins).drop_duplicates()
        assert not joined.pair_key.duplicated().any()
        codec = joined.set_index('pair_key').codec.reindex(meta.pair_key)
        assert codec.notna().all()
        codec_ids = {v: i for i, v in enumerate(sorted(set(codec)))}
        admitted.append((fold, table, [ids[v] for v in names], [codec_ids[v] for v in codec]))
    write(OUT / 'label-read-started.json', dict(ledger_sha256=sha(ledger),
          preread_sha256=sha(inventory), script_sha256=sha(__file__),
          scope='four frozen D1 target columns; seven paired ten-seed grids'))
    datasets = []
    for fold, table, reference, codec in admitted:
        target = pq.read_table(table, columns=['human_score'])['human_score'].to_pylist()
        predictions = {}
        for label, grid in grids.items():
            predictions[label] = []
            for seed in range(10):
                dest = OUT / 'predictions' / label / f'{fold}_s{seed}'
                dest.mkdir(parents=True)
                pred = predict(Path(grid[fold, seed]['selected_model']), table, dest / 'pred.tsv')
                predictions[label].append(pred.tolist())
                print(f'captured {label} {fold} seed {seed}', flush=True)
        datasets.append(dict(name=fold, target=target, reference=reference, codec=codec,
                             orientation=1., predictions=predictions))
    tests = [('w-a', 'base-a', True), ('k-a', 'base-a', True), ('l-a', 'base-a', True),
             ('w-c', 'base-c', True), ('k-c', 'base-c', True), ('k-a', 'w-a', False),
             ('l-a', 'w-a', False), ('l-a', 'k-a', False), ('k-c', 'w-c', False)]
    write(OUT / 'panel-input.json', dict(schema='e34-paired-seed-v1', datasets=datasets,
          tests=[dict(index=i + 1, arm=a, control=c, eligibility=e)
                 for i, (a, c, e) in enumerate(tests)]))
    write(OUT / 'capture-complete.json', dict(input_sha256=sha(OUT / 'panel-input.json'),
          prediction_panels=280, observations=sum(COUNTS), references=sum(GROUPS)))


def steering():
    """One fixed post-fit batch, with served qualification and raw diagnostic.

    Both objectives consume the same immutable native repair captures. The
    raw objective is report-only and cannot replace the served verdict.
    """
    models = json.loads((OUT / 'full-models.json').read_text())
    binary = R / 'bin/dev96-instrument-83b13003'
    assert sha(binary) == '9478ca6fc7a94b937cdcde57764668aec33ff04adfdcf770b3ae0918dce36d9f'
    env = dict(os.environ, ZENSIM_FORMULA_REV='5', ZENSIM_NEIGHBOUR_EXACT='1',
               ZENSIM_MAX_TIER='v3', RAYON_NUM_THREADS='1')
    base = json.loads(Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part6/cached-gate-a/job.json').read_text())
    for name, record in models.items():
        assert sha(record['path']) == record['sha256']
        for mode in ['steer135', 'raw135']:
            dest = OUT / mode / name
            dest.mkdir(parents=True, exist_ok=True)
            model, digest = record['path'], record['sha256']
            if mode == 'raw135':
                stripped = dest / 'nospline.bin'
                assert not stripped.exists()
                subprocess.run([str(R / 'bin/bake_dial_refit'), 'strip', '--in', model,
                                '--out', str(stripped)], check=True)
                model, digest = str(stripped), sha(stripped)
            job = json.loads(json.dumps(base))
            for case in job['cases']:
                case.update(model=model, model_sha256=digest)
            job['linear_probe'] = name.startswith('l-a')
            job['output'] = str(dest / 'result.json')
            path = dest / 'job.json'
            if path.exists():
                assert json.loads(path.read_text()) == job
            else:
                write(path, job)
            assert not Path(job['output']).exists(), 'never repeat a completed assessment'
            with (dest / 'run.log').open('x') as log:
                subprocess.run([str(binary), 'metric::bake::steerfix_packet::engineering_packet',
                                '--exact', '--nocapture'], check=True, stdout=log,
                               stderr=subprocess.STDOUT, env=dict(env, STEERFIX_PACKET=str(path)))
            result = json.loads(Path(job['output']).read_text())['rows']
            assert len(result) == 135
            assert all(r['model'] == digest and r['sensitivity_method'] == 'exact-gradient-d12'
                       and r['sensitivity_shared_owner_exact'] and r['engine_feature_disagreements'] == 0
                       for r in result)
            print(f"{mode} {name}: {sum(r['pass'] for r in result)}/135", flush=True)


def development():
    """Report the fixed dev96 panel on all 21 models; reuse the two pilot reads."""
    binding = json.loads((R / 'packet/DEVELOPMENT_BINDING.json').read_text())
    path = Path(binding['assessment_inputs'])
    assert sha(path) == binding['assessment_inputs_sha256']
    panel = json.loads(path.read_text())
    assert len(panel['cases']) == 96 and len({c['origin'] for c in panel['cases']}) == 8
    models = json.loads((OUT / 'full-models.json').read_text())
    selection = json.loads((R / 'launch/LAMBDA_SELECTION.json').read_text())
    pilot = {r['model_sha256']: r for r in selection['reports']}
    binary = R / 'bin/dev96-instrument-83b13003'
    assert sha(binary) == binding['instrument_sha256']
    reports = {}
    for name, model in models.items():
        assert sha(model['path']) == model['sha256']
        if model['sha256'] in pilot:
            rec = pilot[model['sha256']]
            assert sha(rec['assessment']) == rec['assessment_sha256']
            reports[name] = dict(path=rec['assessment'], sha256=rec['assessment_sha256'],
                                 reuse='immutable completed lambda pilot')
            continue
        dest = OUT / 'development' / name
        dest.mkdir(parents=True)
        output = dest / 'result.json'
        cases = [{**c, 'model': model['path'], 'model_sha256': model['sha256']} for c in panel['cases']]
        job = dict(cases=cases, output=str(output), engine=False, floor_recovery=True,
                   linear_probe=name.startswith('l-a'))
        write(dest / 'job.json', job)
        env = dict(os.environ, ZENSIM_FORMULA_REV='5', ZENSIM_MAX_TIER='v3',
                   RAYON_NUM_THREADS='1', ZENSIM_NEIGHBOUR_EXACT='0',
                   STEERFIX_PACKET=str(dest / 'job.json'))
        with (dest / 'run.log').open('x') as log:
            subprocess.run([str(binary), 'metric::bake::steerfix_packet::engineering_packet',
                            '--exact', '--nocapture'], check=True, stdout=log, stderr=subprocess.STDOUT, env=env)
        rows = json.loads(output.read_text())['rows']
        assert len(rows) == 96 and all(r['model'] == model['sha256']
            and r['sensitivity_method'] == 'exact-gradient-d12' for r in rows)
        reports[name] = dict(path=str(output), sha256=sha(output), reuse=False)
        print(f"dev96 {name}: {sum(r['pass'] for r in rows)}/96", flush=True)
    write(OUT / 'development-results.json', reports)


def gates():
    """Use the existing E33 C/N/K gate dispatchers with explicit E34 bindings."""
    import e33_gates as owner
    owner.GATES = OUT / 'gates'
    owner.BIN = OUT / 'bin'
    owner.GATES.mkdir()
    models = json.loads((OUT / 'full-models.json').read_text())
    found = {name.removesuffix('-s0'): (Path(r['path']), r['sha256'], Path(r['cell']))
             for name, r in models.items() if name.endswith('-s0')}
    assert len(found) == 7
    owner.MODELS_OVERRIDE = found
    labels = {name: name.replace('-', '') for name in found}
    assert len(set(labels.values())) == len(found)
    write(owner.GATES / 'nearid-labels.json', dict(
        scope='artifact labels only; native owner accepts lowercase alphanumeric candidate labels',
        labels={name: dict(native_label=labels[name], model_sha256=found[name][1]) for name in found}))
    for name in found:
        # Preserve the canonical owner's label grammar and all score/gate code.
        label = labels[name]
        owner.MODELS_OVERRIDE = {label: found[name]}
        owner.nearid(label, 0)
        owner.MODELS_OVERRIDE = found
        near = owner.GATES / 'nearid-s0'
        for suffix in ['jsonl', 'metadata.json', 'GATES.json']:
            (near / f'candidate-{name}.{suffix}').symlink_to(f'candidate-{label}.{suffix}')
        for grid in ['standard', 'ladder']:
            owner.verdict(name, 0, grid)
        print(f'C/N grids complete {name}', flush=True)
    # E33 section7 E8 names every full-data seed. Prove the entire frozen
    # model set, while the engineering-grid verdicts remain seed0 gates.
    owner.MODELS_OVERRIDE = {name: (Path(r['path']),r['sha256'],Path(r['cell']))
                            for name,r in models.items()}
    for name,r in models.items():
        raw = OUT / 'raw135' / name / 'nospline.bin'
        owner.MODELS_OVERRIDE['raw-'+name] = (raw,sha(raw),Path(r['cell']))
    owner.identity(0)
    owner.served_identity()
    owner.MODELS_OVERRIDE = found
    # Reuse the exact raw forwards already measured in the fixed batch.
    k4 = owner.GATES / 'k4'
    k4.mkdir()
    bounds = json.loads((OUT / 'pack-checks.json').read_text())['models']
    bound = {}
    for name, (model, digest, _) in found.items():
        raw = OUT / 'raw135' / f'{name}-s0'
        (k4 / f'{name}-nospline.bin').symlink_to(raw / 'nospline.bin')
        (k4 / f'{name}-steer-raw.json').symlink_to(raw / 'result.json')
        bound[name] = bounds[f'{name}-s0']
        assert bound[name]['model_sha256'] == digest
    owner.k4_full(fitbin=R / 'bin/bake_dial_refit', bounds=bound,
                  calibration_input=ROOT / 'wide/main/real/cid22_fit.parquet')
    write(OUT / 'gate-dispatch-complete.json', dict(models=sorted(found),
          scope='registered C1-C6, G-DIAL, N1-N3 and K1-K5; no E33 summary/verdict reused'))


def prepare_p3():
    root = Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part4')
    models = json.loads((root / 'full-job.json').read_text())['models']
    captured = [r for p in sorted((root / 'full').glob('case-*.json')) for r in json.loads(p.read_text())]
    cases = []
    for model, expected in zip(models[:2], [6, 8], strict=True):
        assert model['name'] in ['a', 'c'] and sha(model['path']) == model['sha256']
        assert sha(model['oracle']) == model['oracle_sha256']
        failed = [r['key'] for r in json.loads(Path(model['oracle']).read_text())['rows'] if not r['pass']]
        assert len(failed) == expected
        rows = [r for r in captured if r['model'] == model['name'] and r['key'] in failed]
        assert len(rows) == expected
        for r in rows:
            assert r['model_sha256'] == model['sha256'] and sha(r['feature_file']) == r['feature_sha256']
            cases.append(dict(name=model['name'], model=model['path'], model_sha256=model['sha256'],
                **{k: r[k] for k in ['key', 'block', 'feature_file', 'feature_sha256', 'feature_width', 'blocks']}))
    write(OUT / 'p3-job.json', dict(cases=cases, output=str(OUT / 'p3-switches.json'),
          scope='registered old E33 seed0 failures only; immutable feature captures; no new pixels or labels'))


def prepare_reports():
    import numpy as np
    source = Path(__file__).resolve().parents[2] / 'benchmarks/e34_draft_2026-10-10/characterize.py'
    spec = importlib.util.spec_from_file_location('e34_registered_characterization', source)
    characterization = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(characterization)
    roster = json.loads(Path('/mnt/v/output/zensim/prodqual-b-2026-10-07/STEERING_PREREAD.json').read_text())['cases']
    def picture(name):
        stem = name.split('.png')[0]
        return stem if stem.startswith('o_') else stem.split('_')[0]
    names = [picture(r['ref_group']) for r in roster]
    ids = {name: i for i, name in enumerate(sorted(set(names)))}
    assert len(ids) == 15 and len(roster) == 135
    refs = [ids[name] for name in names]
    panels, per_case, series = {}, [], []
    models = json.loads((OUT / 'full-models.json').read_text())
    assert set(models) == set(MODEL_ORDER)
    for name in MODEL_ORDER:
        path = OUT / 'steer135' / name / 'result.json'
        doc = json.loads(path.read_text())
        rows = doc['rows']
        raw_path = OUT / 'raw135' / name / 'result.json'
        raw = json.loads(raw_path.read_text())['rows']
        assert len(rows) == len(raw) == 135
        assert doc['packet_sha256'] == sha(path.parent / 'job.json')
        for r, p, meta in zip(rows, raw, roster, strict=True):
            assert r['key'] == p['key'] == meta['case']
            assert r['blocks'][0]['bounds'][2] - r['blocks'][0]['bounds'][0] == meta['block']
            assert r['model'] == models[name]['sha256']
            assert r['m2'] is not None and r['m3f'] is not None, 'registered numerical metric undefined'
            assert sum(b['score_delta']**2 for b in r['blocks']) > 0, 'registered relative-L2 denominator is zero'
            point = characterization.case_stats(r)['rel_l2_lin_vs_true']
            per_case.append(dict(model=name, key=r['key'], block=meta['block'],
                reference=picture(meta['ref_group']), panel=meta['panel'],
                m2=r['m2'], m3f=r['m3f'], passed=r['pass'], rel_l2=point,
                raw_true_gt_linear=sum(b['score_delta'] > b['linearized_gain'] for b in p['blocks']),
                blocks=len(r['blocks']), linear_probe=r['linear_probe'],raw_linear_probe=p['linear_probe']))
        panels[name] = rows
        series.append(dict(name=f'gsteer-pass-rate/{name}', reference=refs,
                           values=[float(r['pass']) for r in rows], exp=False))
    development = json.loads((OUT / 'development-results.json').read_text())
    binding = json.loads((R / 'packet/DEVELOPMENT_BINDING.json').read_text())
    dev_cases = json.loads(Path(binding['assessment_inputs']).read_text())['cases']
    dev_ids = {v: i for i, v in enumerate(sorted({r['origin'] for r in dev_cases}))}
    assert len(dev_ids) == 8
    for name in MODEL_ORDER:
        rec = development[name]
        assert sha(rec['path']) == rec['sha256']
        rows = json.loads(Path(rec['path']).read_text())['rows']
        assert len(rows) == 96 and all(r['model'] == models[name]['sha256'] for r in rows)
        assert all(r['key']==c['key'] and r['blocks'][0]['bounds'][2]-r['blocks'][0]['bounds'][0]==c['block']
                   for r,c in zip(rows,dev_cases,strict=True))
        series.append(dict(name=f'dev-pass-rate/{name}', reference=[dev_ids[c['origin']] for c in dev_cases],
                           values=[float(r['pass']) for r in rows], exp=False))
    point_index = {(r['model'], r['key'], r['block']): r['rel_l2'] for r in per_case}
    for arm in ['w-a', 'k-a', 'w-c', 'k-c']:
        for seed in range(3):
            values = []
            for r in panels[f'{arm}-s{seed}']:
                block = r['blocks'][0]['bounds'][2] - r['blocks'][0]['bounds'][0]
                a = point_index[f'{arm}-s{seed}', r['key'], block]
                b = point_index[f'base-{arm[-1]}-s{seed}', r['key'], block]
                assert a > 0 and b > 0, 'registered log Rel.L2 ratio undefined; no epsilon substitution'
                values.append(math.log(a / b))
            series.append(dict(name=f'rel-l2-ratio/{arm}-s{seed}', reference=refs, values=values, exp=True))
    paired = json.loads((OUT / 'panel-input.json').read_text())
    thresholds = [float(np.quantile(d['target'], .8)) for d in paired['datasets']]
    order = json.loads((OUT / 'report-order.json').read_text())
    assert order['offset_start'] == 2000
    assert [s['name'] for s in series] == [s['name'] for s in order['series']]
    assert [s['seed'] for s in order['series']] == list(range(20263011, 20263011+len(series)))
    write(OUT / 'per-case.json', per_case)
    write(OUT / 'report-input.json', dict(schema='e34-report-input-v1',
          paired_input=str(OUT / 'panel-input.json'), paired_sha256=sha(OUT / 'panel-input.json'),
          quality_thresholds=thresholds, series=series))


def progress():
    log = Path.home() / 'tmp/e34-eval/steering.log'
    lines = [s for s in log.read_text().splitlines() if s.startswith(('steer135 ', 'raw135 '))]
    dev_log = Path.home() / 'tmp/e34-eval/development.log'
    dev_lines = ([s for s in dev_log.read_text().splitlines() if s.startswith('dev96 ')]
                 if dev_log.exists() else [])
    path = OUT / 'PROGRESS.json'
    previous = json.loads(path.read_text()) if path.exists() else {'completed_panels': []}
    now = datetime.now(timezone.utc).isoformat()
    record = dict(time=now, completed_panels=lines,
                  served_panels=sum(s.startswith('steer135 ') for s in lines),
                  raw_panels=sum(s.startswith('raw135 ') for s in lines),
                  expected_each=21, development_measured=dev_lines,
                  development_complete=(OUT / 'development-results.json').exists())
    path.write_text(json.dumps(record, indent=2)+'\n')
    new = [s for s in lines if s not in previous['completed_panels']]
    new += [s for s in dev_lines if s not in previous.get('development_measured', [])]
    if new:
        status = Path.home() / 'tmp/zensim-paper/rev4/E34_LAUNCH_STATUS.md'
        with status.open('a') as f:
            f.write('\nEvaluation native/development progress '+now+'\n\n'+'\n'.join('- '+s for s in new)+'\n')
    lock = Path(__file__).resolve().parents[2] / '.workongoing'
    assert ' codex-e34-eval ' in lock.read_text()
    lock.write_text(now+' codex-e34-eval registered assessment in progress\n')
    print(json.dumps(dict(time=now,served_panels=record['served_panels'],raw_panels=record['raw_panels'],
                          expected_each=21,development_new_panels=len(dev_lines),
                          development_complete=record['development_complete'],new=new),indent=2))


def integrity():
    """Verify the installed artifacts still match the pre-exposure inventory."""
    inventory = OUT / 'preread.json'
    assert sha(inventory) == 'e3aae632f8cc4537bb91c48c3de88ce039b4c691c8bd0e77c7025715590d1909'
    cells = json.loads(inventory.read_text())['cells']
    assert len(cells) == 305 and len({c['jobset'] for c in cells}) == 18
    files = {}
    for c in cells:
        cell = Path(c['cell'])
        receipt = cell / 'fleet_receipt.json'
        assert sha(receipt) == c['fleet_receipt_sha256']
        assert sha(cell / 'result.json') == c['result_sha256']
        for rel, digest in json.loads(receipt.read_text())['files'].items():
            path = cell / rel
            assert path.resolve().is_relative_to(cell.resolve())
            assert sha(path) == digest, f'installed artifact changed: {path}'
            files[str(path)] = digest
        assert sha(c['selected_model']) == c['selected_sha256']
        if c['packed_model'] is not None:
            assert sha(c['packed_model']) == c['packed_sha256']
    write(OUT / 'installed-integrity-final.json', dict(time=datetime.now(timezone.utc).isoformat(),
          preread_sha256=sha(inventory), cells=len(cells), jobsets=18, files=len(files),
          unchanged=True, file_sha256=files))
    print(json.dumps(dict(cells=len(cells),files=len(files),unchanged=True)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture', 'steering', 'development', 'gates', 'prepare-p3', 'prepare-reports', 'progress','integrity'])
    args = parser.parse_args()
    {'capture': capture, 'steering': steering, 'development': development, 'gates': gates,
     'prepare-p3': prepare_p3, 'prepare-reports': prepare_reports, 'progress': progress,'integrity':integrity}[args.mode]()
