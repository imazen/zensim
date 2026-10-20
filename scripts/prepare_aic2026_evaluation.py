#!/usr/bin/env python3
"""Prepare an explicit evaluation-only suite from the AIC2026 objective tables.

No image decoding, training or subjective-label inference. Default output has
no target. --objective-target opts into a named objective teacher comparison.
"""
import argparse
import csv
import hashlib
import json
import math
import struct
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare_jnd(root, labels, output, bake, build_commit):
    """Admit one frozen-model read of an explicit JND reconstruction table.

    This only prepares inputs for score_pairs_tuner and panel evaluate. Missing
    pointwise observations form an explicitly counted supplementary population;
    no statistic, decoder, model fit or JND calibration is implemented here.
    """
    images = root / 'extracted/AIC2026-dataset-cropped/PTC-crops-all'
    label_sha = digest(labels)
    with labels.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        original = list(reader.fieldnames or [])
        required = {'source', 'distorted', 'bpp', 'JND_curve', 'JND_pointwise'}
        if not required <= set(original):
            raise ValueError(f'missing reconstruction columns: {required - set(original)}')
        rows = list(reader)
    if not rows or len({r['distorted'] for r in rows}) != len(rows):
        raise ValueError('nonempty unique distorted IDs required')
    audited = {}
    prepared = []
    for index, row in enumerate(rows):
        for column in ('source', 'distorted'):
            name = row[column]
            if not name.startswith('PTC_') or Path(name).name != name:
                raise ValueError(f'unsafe/non-PTC image ID: {name!r}')
            path = images / f'{name}.png'
            if name not in audited:
                with path.open('rb') as f:
                    header = f.read(29)
                if (len(header) != 29 or header[:8] != b'\x89PNG\r\n\x1a\n'
                        or header[12:16] != b'IHDR' or header[24:26] != b'\x08\x02'):
                    raise ValueError(f'expected native RGB8 PNG, no narrowing: {path}')
                width, height = struct.unpack('>II', header[16:24])
                audited[name] = {'path': str(path), 'sha256': digest(path),
                                 'width': width, 'height': height}
        a, b = audited[row['source']], audited[row['distorted']]
        if (a['width'], a['height']) != (b['width'], b['height']):
            raise ValueError(f'pair dimensions differ: {row["distorted"]}')
        for target in ('JND_curve', 'JND_pointwise'):
            if target == 'JND_pointwise' and not row[target].strip():
                continue
            if not math.isfinite(float(row[target])):
                raise ValueError(f'nonfinite {target}: {row["distorted"]}')
        if not math.isfinite(float(row['bpp'])):
            raise ValueError(f'nonfinite bpp: {row["distorted"]}')
        codec, rung = row['distorted'].rsplit('_', 2)[-2:]
        prepared.append(dict(row, row_index=str(index), codec=codec,
                             quality=rung, ref_path=a['path'], dist_path=b['path']))
        if (index + 1) % 250 == 0 or index + 1 == len(rows):
            print(f'JND admission: {index + 1}/{len(rows)} rows, {len(audited)} image hashes',
                  flush=True)
    if digest(labels) != label_sha:
        raise ValueError('reconstruction CSV changed during admission')
    # Refuse reuse/overwrite, including a previous scoring attempt.
    output.mkdir(parents=True, exist_ok=False)
    columns = original + ['row_index', 'codec', 'quality', 'ref_path', 'dist_path']
    def table(path, population, delimiter=','):
        with path.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(population)
    table(output / 'pairs.tsv', prepared, '\t')
    datasets = []
    populations = {}
    for target in ('JND_curve', 'JND_pointwise'):
        population = [r for r in prepared if r[target].strip()]
        populations[target] = len(population)
        if not population:
            continue
        path = output / f'{target}.csv'
        table(path, population)
        datasets.append({'id': target, 'path': str(path), 'sha256': digest(path),
                         'role': 'eval', 'target_kind': 'human',
                         'target_direction': 'lower', 'quality_direction': 'lower',
                         'columns': {'id': 'distorted', 'reference_id': 'source',
                                     'codec': 'codec', 'quality': 'quality', 'target': target}})
    bake_sha = digest(bake)
    suite = {'schema': 'metric-evaluation-v1', 'datasets': datasets,
             'metrics': [{'id': 'zensim-b', 'direction': 'higher',
                          'implementation': f'diff+frag-820; bake SHA256 {bake_sha}; source {build_commit}',
                          'input_contract': 'Native supplied PTC RGB8 crops; explicit reconstructed JND labels, lower is better; no model fit or JND calibration',
                          'ladder_epsilon': 0,
                          'source': {'kind': 'table', 'path': str(output / 'scores.parquet'),
                                     'sha256': '', 'id_column': 'distorted', 'score_column': 'score_b'}}],
             'max_panel_rows': 10000}
    # The score-table digest is bound only after the single scoring run finishes.
    (output / 'suite.pending.json').write_text(json.dumps(suite, indent=2) + '\n')
    admission = {'build_commit': build_commit, 'labels_sha256': label_sha,
                 'bake_sha256': bake_sha, 'primary_target': 'JND_curve',
                 'supplementary_target': 'JND_pointwise', 'populations': populations,
                 'rows': len(rows), 'references': len({r['source'] for r in rows}),
                 'images': audited, 'evaluation_only': True,
                 'scoring_runs_authorized': 1, 'fit_or_calibration': False,
                 'uncertainty': 'Reconstruction confidence limits retained; not treated as observer standard deviations.'}
    (output / 'ADMISSION.json').write_text(json.dumps(admission, indent=2) + '\n')
    return admission


def bind_jnd_scores(output, scores_tsv):
    """Join the native full-population score export onto each declared subset.

    The panel's table adapter deliberately refuses extra IDs, so the sparse
    pointwise labels need their own exact-population table. No pixel is rescored.
    """
    suite = json.loads((output / 'suite.pending.json').read_text())
    scores = {}
    with scores_tsv.open(newline='') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row['id'] in scores or row['error'] or not math.isfinite(float(row['score'])):
                raise ValueError('unique finite native scores without errors required')
            scores[row['id']] = row['score']
    primary = next(d for d in suite['datasets'] if d['id'] == 'JND_curve')
    with Path(primary['path']).open(newline='') as f:
        expected = {r['distorted'] for r in csv.DictReader(f)}
    if set(scores) != expected:
        raise ValueError('native scores must cover exactly the declared curve population')
    for dataset in suite['datasets']:
        path = Path(dataset['path'])
        if digest(path) != dataset['sha256']:
            raise ValueError('declared label table changed before score join')
        with path.open(newline='') as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames)
            rows = [dict(r, score_b=scores[r['distorted']]) for r in reader]
        joined = path.with_suffix('.scored.csv')
        with joined.open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=columns + ['score_b'])
            writer.writeheader()
            writer.writerows(rows)
        dataset.update(path=str(joined), sha256=digest(joined))
        dataset['columns']['score_b'] = 'score_b'
    metric = suite['metrics'][0]
    metric['source'] = {'kind': 'column', 'column': 'score_b'}
    metric['artifacts'] = [{'path': str(scores_tsv), 'sha256': digest(scores_tsv)}]
    with (output / 'suite.scored.json').open('x') as f:
        json.dump(suite, f, indent=2)
        f.write('\n')
    return suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--objective-target', choices=['CVVDP', 'SSIMULACRA2'])
    parser.add_argument('--jnd-reconstruction', type=Path,
                        help='Explicit owner-authorized JND CSV; output becomes a new directory')
    parser.add_argument('--bake', type=Path, help='Frozen model identity for JND mode')
    parser.add_argument('--build-commit', help='Scorer source identity for JND mode')
    parser.add_argument('--bind-scores-tsv', type=Path,
                        help='Bind a complete native id/score/error export to admitted JND tables')
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    if args.bind_scores_tsv:
        bind_jnd_scores(args.output.resolve(strict=True), args.bind_scores_tsv.resolve(strict=True))
        print(args.output / 'suite.scored.json')
        return
    if args.jnd_reconstruction:
        if not args.bake or not args.build_commit or args.objective_target:
            parser.error('JND mode requires --bake and --build-commit, without --objective-target')
        result = prepare_jnd(root, args.jnd_reconstruction.resolve(strict=True),
                             args.output.resolve(), args.bake.resolve(strict=True), args.build_commit)
        print(json.dumps({k: result[k] for k in ('rows', 'references', 'populations', 'bake_sha256')}))
        return
    datasets = []
    for resolution in ('cropped', 'fullres'):
        path = root / f'metrics_{resolution}.csv'
        columns = {'id': 'distorted', 'reference_id': 'source',
                   'codec': 'codec_acronym', 'quality': 'distortion_level',
                   'cvvdp': 'CVVDP', 'ssim2': 'SSIMULACRA2',
                   'butteraugli': 'proposal-Butteraugli'}
        dataset = {'id': f'aic2026-{resolution}', 'path': str(path),
                   'sha256': digest(path), 'role': 'public_test',
                   'quality_direction': 'lower', 'columns': columns}
        if args.objective_target:
            columns['target'] = args.objective_target
            dataset.update(target_kind='objective', target_direction='higher')
        datasets.append(dataset)
    metrics = []
    for name, column, direction in [('cvvdp', 'CVVDP', 'higher'),
                                    ('ssim2', 'SSIMULACRA2', 'higher'),
                                    ('butteraugli', 'proposal-Butteraugli', 'lower')]:
        if column == args.objective_target:
            continue  # A teacher's self-correlation is not a metric comparison.
        metrics.append({'id': name, 'direction': direction,
                        'implementation': f'AIC2026 supplied {column} column; see encoding_recipes.md',
                        'input_contract': 'Original supplied objective scores; cropped and fullres kept separate; no human judgments',
                        'ladder_epsilon': 0,
                        'source': {'kind': 'column', 'column': name}})
    suite = {'schema': 'metric-evaluation-v1', 'datasets': datasets,
             'metrics': metrics, 'max_panel_rows': 10000}
    with args.output.open('x') as f:
        json.dump(suite, f, indent=2)
        f.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
