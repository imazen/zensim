"""Remove only this lane's superseded local stages after its tower audit.

The frozen full mirror receipt remains on tower; final prepared data, archives,
producer binaries, manifests and evidence stay local. No broad cache cleanup.
"""
import argparse
import hashlib
import random
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path('/mnt/v/output/zensim/e29-2026-10-07')
TOWER = Path('/mnt/tower/output/zensim/e29-2026-10-07')
SCRATCH = Path('/home/lilith/tmp/e29')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def mirror_final():
    """Verify the kept local subset without overwriting the full tower archive."""
    old = json.loads((TOWER / 'ARCHIVE_MIRROR_RECEIPT.json').read_text())
    records = {}
    for path in sorted(ROOT.rglob('*')):
        if not path.is_file():
            continue
        rel = str(path.relative_to(ROOT))
        if rel in ('FINAL_LOCAL_MIRROR_RECEIPT.json', 'logs/final-mirror.log'):
            continue
        digest = sha(path)
        existing = TOWER / rel
        if existing.is_file() and sha(existing) == digest:
            dest = existing
        else:
            # Earlier evidence remains immutable at its original tower path.
            dest = TOWER / 'final-code-state' / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and sha(dest) != digest:
                raise ValueError(f'final evidence already differs: {dest}')
            if not dest.exists():
                shutil.copy2(path, dest)
        assert sha(dest) == digest, rel
        records[rel] = {'sha256': digest, 'bytes': path.stat().st_size,
                        'tower_relative_path': str(dest.relative_to(TOWER))}
    selected = random.SystemRandom().sample(list(records), 3)
    for rel in selected:
        assert sha(ROOT / rel) == sha(TOWER / records[rel]['tower_relative_path'])
    receipt = dict(schema='e29-final-local-subset-mirror-v1', status='PASS',
                   build_commit=json.loads((ROOT / 'CODE_STATE.json').read_text())['build_commit'],
                   file_count=len(records), files=records, three_random_files=selected,
                   full_archive_receipt_sha256=sha(TOWER / 'ARCHIVE_MIRROR_RECEIPT.json'),
                   full_original_archive_files=old['file_count'],
                   scope='retained local subset; earlier full tower archive preserved unchanged')
    output = ROOT / 'FINAL_LOCAL_MIRROR_RECEIPT.json'
    output.write_text(json.dumps(receipt, indent=2)+'\n')
    shutil.copy2(output, TOWER / output.name)
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'}), flush=True)


def main():
    receipt_path = ROOT / 'MIRROR_RECEIPT.json'
    receipt = json.loads(receipt_path.read_text())
    assert receipt['status'] == 'PASS' and receipt['file_count'] == 11020
    assert sha(receipt_path) == sha(TOWER / receipt_path.name)
    frozen = TOWER / 'ARCHIVE_MIRROR_RECEIPT.json'
    if frozen.exists():
        assert sha(frozen) == sha(receipt_path)
    else:
        shutil.copy2(receipt_path, frozen)
    for name, item in receipt['three_random_files'].items():
        assert sha(TOWER / name) == item['sha256']
    assert not subprocess.check_output(['docker', 'ps', '-q'], text=True).strip()
    active = subprocess.check_output(['ps', '-eo', 'comm='], text=True).splitlines()
    assert not any(c.strip() in ('cargo', 'rustc', 'zensim_mlp_trai') for c in active)
    candidates = [ROOT / 'container-scratch', ROOT / 'superseded-pre-import-guard']
    candidates += list(ROOT.glob('*.bak-*'))
    candidates += [ROOT / 'PINNED_ARTIFACTS.before-counter-clarification.json']
    for arm in ('base', 'hb4', 'hc4'):
        candidates += [ROOT / f'{arm}-harvest-smoke', ROOT / f'{arm}-smoke-install-refusal']
        candidates += list(ROOT.glob(f'{arm}-negative-*'))
    candidates = [p for p in candidates if p.exists()]
    deleted = []
    # Confirm every removed artifact has the already byte-verified tower copy.
    for path in candidates:
        files = [p for p in path.rglob('*') if p.is_file()] if path.is_dir() else [path]
        for p in files:
            key = str(p.relative_to(ROOT))
            expected = receipt['files'][key]
            assert (TOWER / key).stat().st_size == expected['bytes']
        size = int(subprocess.check_output(['du', '-sB1', str(path)], text=True).split()[0])
        deleted.append({'path': str(path), 'allocated_bytes': size, 'verified_tower_files': len(files)})
    # Cargo output and task-specific source clones: all source pins, lock and CLI
    # were separately retained in CTL_PRODUCER before removing these own caches.
    producer = json.loads((ROOT / 'CTL_PRODUCER.json').read_text())
    for p in (ROOT / 'bin/zenfleet-ctl', ROOT / 'source-snapshots/zenfleet-ctl.Cargo.lock',
              ROOT / 'source-snapshots/zenfleet-ctl.sibling-pins.tsv', ROOT / 'CTL_PRODUCER.json'):
        assert sha(p) == sha(TOWER / p.relative_to(ROOT))
    assert producer
    for name in ('target', 'zenmetrics-target'):
        assert (SCRATCH / name / '.rustc_info.json').is_file()
    size = int(subprocess.check_output(['du', '-sB1', str(SCRATCH)], text=True).split()[0])
    deleted.append({'path': str(SCRATCH), 'allocated_bytes': size,
                    'scope': 'own task caches; pinned CLI, source lock and sibling pins preserved'})
    out = dict(schema='e29-storage-cleanup-v1', status='VERIFIED_BEFORE_REMOVAL',
               full_archive_receipt_sha256=sha(frozen), full_archive_receipt=str(frozen),
               removed=deleted, allocated_bytes_removed=sum(x['allocated_bytes'] for x in deleted),
               preserved_local='final prepared root, one data/program/image archive, binaries, manifests, evidence')
    (ROOT / 'CLEANUP_RECEIPT.json').write_text(json.dumps(out, indent=2)+'\n')
    for path in candidates:
        if path.is_dir():
            subprocess.run(['sudo', 'rm', '-rf', '--', str(path)], check=True)
        else:
            path.unlink()
    for p in SCRATCH.iterdir():
        if p.name in ("target", "zenmetrics-target", "ctl-snapshot", "pycache", "fmt-initial.log") or p.name.startswith("tmp") and "." not in p.name:
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()
    out['status'] = 'PASS'
    (ROOT / 'CLEANUP_RECEIPT.json').write_text(json.dumps(out, indent=2)+'\n')
    shutil.copy2(ROOT / 'CLEANUP_RECEIPT.json', TOWER / 'CLEANUP_RECEIPT.json')
    print(json.dumps({k:v for k,v in out.items() if k!='removed'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mirror-final', action='store_true')
    args = parser.parse_args()
    mirror_final() if args.mirror_final else main()
