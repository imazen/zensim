#!/usr/bin/env python3
"""transplant lane: joint-core-v2 root _MANIFEST.json — the versioned
core identity (rows, cohorts, provenance, gates), matching v1's shape.
"""
import csv, json, hashlib, os, collections, subprocess

V2 = '/mnt/v/output/zensim/joint-core-v2'
V1 = '/mnt/v/output/zensim/joint-core-v1'


def sha(path, head=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(head):
            h.update(b)
    return h.hexdigest()


def main():
    cov = json.load(open(f'{V2}/plan/coverage_report.json'))
    rows = list(csv.DictReader(open(f'{V2}/pairs/pairs_core.tsv'),
                               delimiter='\t'))
    kernels = collections.defaultdict(collections.Counter)
    for r in rows:
        kernels[r['leg']][r['kernel']] += 1
    commit = subprocess.run(
        ['jj', 'log', '-r', '@', '--no-graph', '-T', 'commit_id'],
        capture_output=True, text=True,
        cwd='/home/lilith/work/zen/zensim--transplant').stdout.strip()
    manifest = {
        'name': 'joint-core-v2',
        'built': '2026-09-20',
        'parent_core': 'joint-core-v1 (unmodified)',
        'build_workspace': '/home/lilith/work/zen/zensim--transplant',
        'build_commit': commit,
        'total_pairs': len(rows),
        'cohorts': dict(collections.Counter(r['cohort'] for r in rows)),
        'legs': dict(collections.Counter(r['leg'] for r in rows)),
        'coverage': cov,
        'kernels': {k: dict(v) for k, v in kernels.items()},
        'selection': {
            'v2reused': 'complement of v1 k-means selection inside each '
                        'reused table (same positional machinery)',
            'v2fresh': 'clustered rendition-level selection, seed per '
                       'select_core_v2.py; kernel/band/group quotas',
            'seed': 17},
        'gates': {
            'mid_share>=0.55': cov.get('mid_share_gate_ok'),
            'photo_share>=0.75': cov.get('photo_share_gate_ok')},
        'inputs_sha256': {
            'pairs_core.tsv': sha(f'{V2}/pairs/pairs_core.tsv'),
            'pairs_provenance.tsv': sha(f'{V2}/pairs/pairs_provenance.tsv'),
            'renditions_v2.tsv': sha(f'{V2}/plan/renditions_v2.tsv'),
            'cells_v2.tsv': sha(f'{V2}/plan/cells_v2.tsv'),
            'reused_pairs_v2.tsv': sha(f'{V2}/plan/reused_pairs_v2.tsv'),
            'reused_rows_v2.json': sha(f'{V2}/plan/reused_rows_v2.json')},
        'v1_sha256': {
            'pairs_core.tsv': sha(f'{V1}/pairs/pairs_core.tsv'),
            'reused_rows.json': sha(f'{V1}/plan/reused_rows.json')},
    }
    with open(f'{V2}/_MANIFEST.json', 'w') as f:
        json.dump(manifest, f, indent=1)
    print(json.dumps({k: manifest[k] for k in
                      ('total_pairs', 'cohorts', 'legs', 'gates')},
                     indent=1))


if __name__ == '__main__':
    main()
