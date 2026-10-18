#!/usr/bin/env python3
"""Copy a qualified MAPPERF or PERF2 evidence tree and verify three sampled files."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--lane', choices=['MAPPERF', 'PERF2'], default='MAPPERF')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    destination = args.destination
    if args.lane == 'MAPPERF':
        assert root.name == 'mapperf-2026-10-17'
        assert destination.name == 'zensim-mapperf-archive-2026-10-17'
        assert json.loads((root / 'src0-final/FULL_NATIVE_CACHE_PARITY.json').read_text())['status'] == 'PASS'
        seed = 20261017
    else:
        assert root.name == 'perf2-2026-10-18'
        assert destination.name == 'zensim-perf2-archive-2026-10-18'
        remote = root / 'remote'
        assert json.loads((remote / 'matched-grain-final-QUALIFIED.json').read_text())['status'] == 'PASS'
        proof = json.loads((remote / 'native-grain-final/original-packets/FULL_NATIVE_CACHE_PARITY.json').read_text())
        assert proof['status'] == 'PASS'
        assert sum(model['rows'] for model in proof['models'].values()) == 270
        assert all(model['every_output_row_field_equal'] for model in proof['models'].values())
        assert len(list((remote / 'profiles-grain-final').glob('*-PASS.json'))) == 6
        assert len(list((remote / 'profiles-grain-final-stats').glob('*-PASS.json'))) == 6
        seed = 20261018
    owner = destination / f'{args.lane}_ARCHIVE_OWNER.json'
    binding = dict(source=str(root), lane=args.lane)
    if destination.exists():
        assert json.loads(owner.read_text()) == binding, 'unrecognized existing archive'
    else:
        destination.mkdir()
        owner.write_text(json.dumps(binding, indent=2) + '\n')
    # The archive share controls ownership and modes; preserve file contents.
    subprocess.run(['rsync', '-r', '--links', '--checksum', '--no-owner', '--no-group',
                    '--no-perms', '--no-times', '--bwlimit=60M', '--stats',
                    str(root) + '/', str(destination) + '/'], check=True)
    files = sorted(p.relative_to(root) for p in root.rglob('*') if p.is_file())
    checked = []
    for relative in random.Random(seed).sample(files, 3):
        source_sha = sha(root / relative)
        assert sha(destination / relative) == source_sha, str(relative)
        checked.append(dict(file=str(relative), sha256=source_sha))
    receipt = root / 'provenance/TOWER_MIRROR.json'
    receipt.write_text(json.dumps(dict(time_utc=datetime.now(timezone.utc).isoformat(),
                                      source=str(root), destination=str(destination),
                                      seed=seed, checked=checked), indent=2) + '\n')
    shutil.copyfile(receipt, destination / receipt.relative_to(root))
    assert sha(receipt) == sha(destination / receipt.relative_to(root))
    print(receipt.read_text())


if __name__ == '__main__':
    main()
