#!/usr/bin/env python3
"""Prepare an explicit evaluation-only suite from the AIC2026 objective tables.

No image decoding, training or subjective-label inference. Default output has
no target. --objective-target opts into a named objective teacher comparison.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--objective-target', choices=['CVVDP', 'SSIMULACRA2'])
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
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
