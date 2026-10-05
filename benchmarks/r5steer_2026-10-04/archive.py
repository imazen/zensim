#!/usr/bin/env python3
"""Persist every measured JSON and its replay evidence to the mounted tower."""
import csv
import hashlib
import json
from pathlib import Path
import shutil

REPO = Path(__file__).resolve().parents[2]
OUT = Path('/var/tmp/r5steer')
DEST = REPO / 'benchmarks/r5steer_2026-10-04'
TOWER = Path('/mnt/tower/output/zensim/r5steer-2026-10-04')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


assert (OUT / 'RUN_COMPLETE.json').exists()
assert (DEST / 'summary.json').exists()
TOWER.mkdir(parents=True, exist_ok=True)
inventory = []


def copy(src, relative):
    dst = TOWER / relative
    dst.parent.mkdir(parents=True, exist_ok=True)
    digest = sha(src)
    if dst.exists():
        assert sha(dst) == digest, f'Refusing to overwrite differing evidence: {dst}'
    else:
        shutil.copy2(src, dst)
    assert sha(dst) == digest
    inventory.append({'source': str(src), 'tower_relative_path': str(relative),
                      'sha256': digest, 'bytes': dst.stat().st_size})


with (REPO / 'benchmarks/r5steer_2026-10-04.tsv').open() as f:
    rows = list(csv.DictReader(f, delimiter='\t'))
assert len(rows) == 624
for r in rows:
    src = Path(r['report'])
    assert sha(src) == r['sha256']
    target = Path(f"rev{r['revision']}") / r['panel'] / src.name
    copy(src, target)
    log = src.with_suffix('.log')
    if log.exists():
        copy(log, target.with_suffix('.log'))
for rev, src in [(4, Path('/var/tmp/neighsteer/fpbroad/RESULT.json')), (5, OUT / 'rev5/broad/RESULT.json')]:
    copy(src, Path(f'rev{rev}/broad/RESULT.json'))
inputs = json.loads((DEST / 'INPUTS.json').read_text())
for model in inputs['models']:
    for rev in [4, 5]:
        src = Path(model[f'rev{rev}'])
        copy(src, Path(f'models/rev{rev}') / src.name)
for name in ['summary.json', 'INPUTS.json', 'REV4_MODEL_VERIFY.json', 'RUN_COMPLETE.json']:
    src = DEST / name
    copy(src, Path('provenance') / src.name)
for name in ['EXECUTION.json', 'run.log', 'clippy.log', 'lint-scripts.log', 'rev4-verify.log',
             'owner-regression-repeat.json', 'owner-regression-repeat.log']:
    copy(OUT / name, Path('provenance') / name)
(DEST / 'archive_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
copy(DEST / 'archive_inventory.json', Path('archive_inventory.json'))
total = sum(r['bytes'] for r in inventory)
pointer = {
    'schema': 'r5steer-tower-pointer-v1', 'tower_root': str(TOWER), 'verified_files': len(inventory),
    'bytes': total, 'raw_case_reports': 624, 'repeated_owner_case_reports': 1,
    'inventory_sha256': sha(DEST / 'archive_inventory.json'), 'all_copies_sha256_verified': True,
}
(DEST / 'pointer.json').write_text(json.dumps(pointer, indent=2) + '\n')
print(json.dumps(pointer, indent=2))
