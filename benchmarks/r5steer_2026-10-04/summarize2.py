#!/usr/bin/env python3
"""Summarize production-forward probes; retain raw outputs on tower."""
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

REPO = Path(__file__).resolve().parents[2]
OUT = Path('/var/tmp/r5steer2')
DEST = REPO / 'benchmarks/r5steer2_2026-10-04'
DEST.mkdir(exist_ok=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(name, data):
    with (DEST / name).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(data)


def aggregate(rows):
    return {'n': len(rows), 'pass': sum(r['pass'] for r in rows),
        'm2_min': min(r['m2'] for r in rows), 'm3f_median': statistics.median(r['m3f'] for r in rows),
        'm3f_min': min(r['m3f'] for r in rows)}


if '--archive' in sys.argv:
    tower = Path('/mnt/tower/output/zensim/r5steer2-2026-10-04')
    tower.mkdir(parents=True, exist_ok=True)
    inventory = []

    def copy(src, relative):
        dst = tower / relative
        dst.parent.mkdir(parents=True, exist_ok=True)
        digest = sha(src)
        if dst.exists():
            assert sha(dst) == digest, f'refusing to replace different evidence: {dst}'
        else:
            shutil.copy2(src, dst)
        assert sha(dst) == digest
        inventory.append({'path': str(relative), 'sha256': digest, 'bytes': dst.stat().st_size})

    for entry in sorted(OUT.iterdir()):
        if entry.name in ['target', 'tmp', 'pycache', 'archive.log']:
            continue
        files = sorted(entry.rglob('*')) if entry.is_dir() else [entry]
        for src in files:
            if src.is_file():
                copy(src, src.relative_to(OUT))
    for src in sorted(DEST.iterdir()):
        if src.is_file() and src.name not in ['pointer.json', 'archive_inventory.tsv']:
            copy(src, Path('results') / src.name)
    copy(Path('/var/tmp/r5steer/bin/diffmap_block_coherence'), Path('bin/original_diffmap_block_coherence'))
    copy(Path('/var/tmp/r5steer/bin/bake_stamp_revision'), Path('bin/bake_stamp_revision'))
    save('archive_inventory.tsv', inventory)
    copy(DEST / 'archive_inventory.tsv', Path('archive_inventory.tsv'))
    pointer = {'schema': 'r5steer2-pointer-v1', 'root': str(tower), 'verified_files': len(inventory),
        'bytes': sum(r['bytes'] for r in inventory), 'inventory_sha256': sha(DEST / 'archive_inventory.tsv'),
        'raw_final_gradient_reports': 20, 'cross_reports': 108, 'owner_size_reports': 8,
        'all_sha256_verified': True, 'prior_reference_archive': '/mnt/tower/output/zensim/r5steer-2026-10-04'}
    (DEST / 'pointer.json').write_text(json.dumps(pointer, indent=2) + '\n')
    print(json.dumps(pointer, indent=2))
    sys.exit(0)

complete = json.loads((OUT / 'GRADIENT_COMPLETE.json').read_text())
assert complete['status'] == 'COMPLETE_UNQUALIFIED' and len(complete['reports']) == 20
assert (OUT / 'CROSS_COMPLETE.json').exists()
gradient, directional, pixels, curvature, details, paired = [], [], [], [], [], {}
for receipt in complete['reports']:
    assert sha(receipt['report']) == receipt['sha256']
    d = json.loads(Path(receipt['report']).read_text())
    rev = int(d['revision'])
    paired.setdefault(d['name'], {})[rev] = d
    for r in d['gradient_steps']:
        gradient.append({'case': d['name'], 'revision': rev, 'factor': r['factor'],
            **r['error_vs_m2'], 'm2_full_repairs': r['m2_full_repairs']})
    same = next(r for r in d['gradient_steps'] if r['factor'] == .001)
    assert same['error_vs_m2']['relative_l2'] == 0.0
    for r in d['directional_feature_probes']:
        directional.append({'case': d['name'], 'revision': rev, 'eps': r['eps'], **r['error']})
        for p in r['probes']:
            details.append({'case': d['name'], 'revision': rev, 'kind': 'feature_direction',
                'eps': r['eps'], 'probe': p['probe'], 'predicted': p['predicted'], 'fd': p['central_fd'],
                'relative_error': p['relative_error'], 'score_span': p['score_span']})
    for r in d['pixel16_probes']:
        pixels.append({'case': d['name'], 'revision': rev, 'code_step': r['code_step'],
            'normalized_eps': r['normalized_eps'], **r['error'], 'rgb16_minus_rgb8_score': d['rgb16_minus_rgb8_score']})
        for p in r['probes']:
            details.append({'case': d['name'], 'revision': rev, 'kind': 'single_pixel16' if p['single_pixel'] else 'block16',
                'eps': r['normalized_eps'], 'probe': str(p['block']), 'predicted': p['predicted'],
                'fd': p['pixel_central_fd'], 'relative_error': p['relative_error'], 'score_span': p['score_span']})
    for r in d['finite_repair_curvature']:
        curvature.append({'case': d['name'], 'revision': rev, 'fraction': r['fraction'],
            'm2': r['m2'], **r['error']})
assert len(paired) == 10 and all(set(p) == {4, 5} for p in paired.values())
save('gradient.tsv', sorted(gradient, key=lambda r: (r['case'], r['revision'], r['factor'])))
save('directional.tsv', sorted(directional, key=lambda r: (r['case'], r['revision'], r['eps'])))
save('pixel16.tsv', sorted(pixels, key=lambda r: (r['case'], r['revision'], r['code_step'])))
save('probe_details.tsv', details)
save('curvature.tsv', sorted(curvature, key=lambda r: (r['case'], r['revision'], r['fraction'])))
case_summary = []
for name, versions in sorted(paired.items()):
    row = {'case': name}
    for rev in [4, 5]:
        d = versions[rev]
        fine = next(r for r in d['gradient_steps'] if r['factor'] == .0003)
        coarse = next(r for r in d['gradient_steps'] if r['factor'] == .01)
        small = next(r for r in d['finite_repair_curvature'] if r['fraction'] == .01)
        px = next(r for r in d['pixel16_probes'] if r['code_step'] == 16)
        row.update({f'r{rev}_m2': d['m2'], f'r{rev}_fine_gradient_rel_l2': fine['error_vs_m2']['relative_l2'],
            f'r{rev}_fine_gradient_cosine': fine['error_vs_m2']['cosine'],
            f'r{rev}_m2_fine': fine['m2_full_repairs'], f'r{rev}_m2_coarse': coarse['m2_full_repairs'],
            f'r{rev}_m2_one_percent': small['m2'], f'r{rev}_pixel16_rel_l2': px['error']['relative_l2'],
            f'r{rev}_pixel16_cosine': px['error']['cosine']})
    case_summary.append(row)
save('cases.tsv', case_summary)

owners = json.loads((OUT / 'OWNER.json').read_text())
sizes = [r for r in owners if r['panel'].startswith('owner-rev')]
assert len(sizes) == 8
save('owner_sizes.tsv', sorted(sizes, key=lambda r: (r['revision'], r['block'])))
cross = [r for r in owners if r['panel'] == 'cross-owner']
assert len(cross) == 12
cross_rows = [{'panel': 'owner', 'index': r['image'], 'block': r['block'], 'm2': r['m2'],
    'm3f': r['m3f'], 'pass': r['pass'], 'report': r['report'], 'sha256': r['sha256']} for r in cross]
broad = json.loads((OUT / 'cross-broad/RESULT.json').read_text())
assert len(broad['rows']) == 96
for r in broad['rows']:
    p = OUT / 'cross-broad' / r['report']
    assert sha(p) == r['sha256']
    d = json.loads(p.read_text())
    reference = json.loads((Path('/var/tmp/neighsteer/fpbroad') / r['report']).read_text())
    for field in ['reference_pixels_sha256', 'distorted_pixels_sha256', 'block_size', 'pixel_interventions']:
        assert d[field] == reference[field]
    cross_rows.append({'panel': 'broad', 'index': r['index'], 'block': r['block'], 'm2': r['m2'],
        'm3f': r['m3f'], 'pass': int(r['status'] == 'PASS'), 'report': str(p), 'sha256': sha(p)})
save('cross.tsv', cross_rows)
cross_summary = [{'panel': 'owner', 'block': 32, **aggregate(cross)}]
for block in [None, 8, 16, 32, 64]:
    rows = [r for r in cross_rows if r['panel'] == 'broad' and (block is None or r['block'] == block)]
    cross_summary.append({'panel': 'broad', 'block': block, **aggregate(rows)})
save('cross_summary.tsv', cross_summary)
summary = {'schema': 'r5steer2-summary-v1', 'cases': case_summary, 'cross': cross_summary,
    'same_step_independent_gradient_relative_error': 0.0, 'reports': 20,
    'diagnostic_binary_sha256': complete['binary_sha256'],
    'classification': 'b dominates; shared finite-difference precision affects near-threshold cases',
    'rev5_specific_derivative_implementation_defect_observed': False,
    'analytic_pixel_gradient_exists': False,
    'extraction_or_score_parity_failures': 0}
(DEST / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
for name in ['MODELS.json', 'cases.json', 'JOBS.json', 'CROSS_COMPLETE.json', 'GRADIENT_COMPLETE.json']:
    shutil.copy2(OUT / name, DEST / name)
print('PASS: 20 paired gradient reports, 108 cross cases, 8 owner-size cases')
print('cross', cross_summary)
for rev in [4, 5]:
    rows = [r for r in gradient if r['revision'] == rev and r['factor'] == .0003]
    print('rev', rev, 'fine gradient rel L2 median/max', statistics.median(r['relative_l2'] for r in rows),
        max(r['relative_l2'] for r in rows), 'cosine min', min(r['cosine'] for r in rows))
