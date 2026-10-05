#!/usr/bin/env python3
"""Join measured Rust reports by panel, arm, seed and pair; never recompute metrics."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

REPO = Path(__file__).resolve().parents[2]
OUT = Path('/var/tmp/r5steer')
OLD = Path('/var/tmp/neighsteer')
DEST = REPO / 'benchmarks/r5steer_2026-10-04'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def status(d):
    assert d['refinement_available'] and not d['refinement_unsupported_ids']
    return int(d['m2'] >= .99 and d['m3f'] >= .70)


def key(r):
    return r['panel'], r['arm'], r['seed_index'], r['case'], r['block']


def stats(rows):
    return {'n': len(rows), 'pass': sum(int(r['pass']) for r in rows),
            'm3f_median': statistics.median(r['m3f'] for r in rows),
            'm3f_min': min(r['m3f'] for r in rows), 'm2_min': min(r['m2'] for r in rows)}


assert (OUT / 'RUN_COMPLETE.json').exists(), 'Full measurement run must finish first'
rows = json.loads((OUT / 'PANELS.json').read_text())
assert len(rows) == 168
for r in rows:
    assert sha(r['report']) == r['sha256']
for folder in ['kadidjpeg', 'owner']:
    files = sorted((OLD / 'fppanels' / folder).glob('*.json'))
    assert len(files) == (48 if folder == 'kadidjpeg' else 24)
    for p in files:
        arm, seed, image, block = p.stem.split('-')
        d = json.loads(p.read_text())
        rows.append({'revision': 4, 'neighbour_exact': False, 'panel': folder, 'arm': arm,
            'seed_index': [5101, 5103, 5107].index(int(seed[1:])), 'seed_label': int(seed[1:]),
            'case': image, 'level': image[-2:], 'block': int(block[1:]), 'm2': d['m2'],
            'm3f': d['m3f'], 'pass': status(d), 'report': str(p), 'sha256': sha(p)})
for rev, folder in [(4, OLD / 'fpbroad'), (5, OUT / 'rev5/broad')]:
    result = json.loads((folder / 'RESULT.json').read_text())
    assert len(result['rows']) == 192
    for r in result['rows']:
        p = folder / r['report']
        assert sha(p) == r['sha256']
        d = json.loads(p.read_text())
        assert d['m2'] == r['m2'] and d['m3f'] == r['m3f']
        assert status(d) == int(r['status'] == 'PASS')
        rows.append({'revision': rev, 'neighbour_exact': False, 'panel': 'broad', 'arm': r['model'],
            'seed_index': 'ensemble', 'seed_label': '5101,5103,5107', 'case': str(r['index']),
            'level': r['distance'], 'block': r['block'], 'm2': r['m2'], 'm3f': r['m3f'],
            'pass': status(d), 'report': str(p), 'sha256': sha(p)})
assert len(rows) == 624
for r in rows:
    r['pass'] = int(r['pass'])
paired = {}
for r in rows:
    k = key(r)
    assert r['revision'] not in paired.setdefault(k, {})
    paired[k][r['revision']] = r
assert len(paired) == 312 and all(set(p) == {4, 5} for p in paired.values())
for revision in [4, 5]:
    plain = {key(r)[1:]: r for r in rows if r['revision'] == revision and r['panel'] == 'kadidjpeg'}
    exact = [r for r in rows if r['revision'] == revision and r['panel'] == 'kadid_exact']
    assert len(exact) == len(plain) == 48
    for r in exact:
        baseline = plain[key(r)[1:]]
        assert r['m2'] == baseline['m2'], 'Exact steering must not change the score-change ground truth'
        assert r['m3f'] != baseline['m3f'], 'Exact mode must change the measured map ranking on these pairs'
delta = []
for k, versions in sorted(paired.items()):
    a, b = versions[4], versions[5]
    da, db = (json.loads(Path(r['report']).read_text()) for r in [a, b])
    for field in ['reference_pixels_sha256', 'distorted_pixels_sha256', 'block_size', 'pixel_interventions']:
        assert da[field] == db[field], (k, field)
    delta.append({'panel': k[0], 'arm': k[1], 'seed_index': k[2], 'case': k[3], 'block': k[4],
        'rev4_m2': a['m2'], 'rev5_m2': b['m2'], 'delta_m2': b['m2'] - a['m2'],
        'rev4_m3f': a['m3f'], 'rev5_m3f': b['m3f'], 'delta_m3f': b['m3f'] - a['m3f'],
        'rev4_pass': int(a['pass']), 'rev5_pass': int(b['pass'])})


def tsv(path, data):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(data[0]), delimiter='\t')
        writer.writeheader()
        writer.writerows(data)


tsv(REPO / 'benchmarks/r5steer_2026-10-04.tsv', sorted(rows, key=lambda r: (r['revision'], key(r))))
tsv(DEST / 'paired.tsv', delta)
summary = []
for arm in ['fpbyv2fy', 'fpv2basic']:
    for panel, level, block in [('kadidjpeg', '03', 8), ('kadidjpeg', '05', 8), ('owner', '03', 32),
                                 ('kadid_exact', '03', 8), ('kadid_exact', '05', 8),
                                 ('broad', None, None)] + [('broad', None, b) for b in [8, 16, 32, 64]]:
        group = [r for r in rows if r['arm'] == arm and r['panel'] == panel
                 and (level is None or r['level'] == level) and (block is None or r['block'] == block)]
        a, b = [[r for r in group if r['revision'] == rev] for rev in [4, 5]]
        assert len(a) == len(b)
        by_key = {key(r): r for r in a}
        changes = [r['m3f'] - by_key[key(r)]['m3f'] for r in b]
        summary.append({'arm': arm, 'panel': panel, 'level': level, 'block': block,
            'rev4': stats(a), 'rev5': stats(b), 'paired_m3f_delta_median': statistics.median(changes),
            'paired_m3f_delta_min': min(changes), 'paired_m3f_delta_max': max(changes),
            'fail_to_pass': sum(not by_key[key(r)]['pass'] and r['pass'] for r in b),
            'pass_to_fail': sum(by_key[key(r)]['pass'] and not r['pass'] for r in b)})
(DEST / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
lines = ['| Arm | Panel | Rev4 pass | Rev4 M3f median (min) | Rev4 M2 min | Rev5 pass | Rev5 M3f median (min) | Rev5 M2 min | Paired ΔM3f median |',
         '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for s in summary:
    a, b = s['rev4'], s['rev5']
    arm = {'fpbyv2fy': 'by_v2fy', 'fpv2basic': 'v2 + basic'}[s['arm']]
    panel = {'kadidjpeg': 'KADID JPEG', 'owner': 'Owner', 'kadid_exact': 'KADID exact', 'broad': 'Broad'}[s['panel']]
    label = panel + (f" L{s['level']}" if s['level'] else '') + (f" b{s['block']}" if s['block'] else ' all')
    lines.append(f"| {arm} | {label} | {a['pass']}/{a['n']} | {a['m3f_median']:.6f} ({a['m3f_min']:.6f}) | {a['m2_min']:.6f} | {b['pass']}/{b['n']} | {b['m3f_median']:.6f} ({b['m3f_min']:.6f}) | {b['m2_min']:.6f} | {s['paired_m3f_delta_median']:+.6f} |")
(DEST / 'table.md').write_text('\n'.join(lines) + '\n')
for filename in ['INPUTS.json', 'REV4_MODEL_VERIFY.json', 'RUN_COMPLETE.json']:
    (DEST / filename).write_bytes((OUT / filename).read_bytes())
inputs = json.loads((DEST / 'INPUTS.json').read_text())
for model in inputs['models']:
    source = Path(model['source'])
    a = json.loads((source.parents[1] / 'result.json').read_text())
    b = json.loads((Path(str(source).replace('/v2c5/', '/v2c/')).parents[1] / 'result.json').read_text())
    for field in ['seed_index', 'init_seed', 'sample_seed', 'epochs']:
        assert a[field] == b[field]
        model[field] = a[field]
(DEST / 'INPUTS.json').write_text(json.dumps(inputs, indent=2) + '\n')
print('\n'.join(lines))
print('PASS: 624 measured rows, 312 matched revision pairs, every pixel hash/block count matches')
