"""Mirror a completed raw evidence root and verify three random files."""
import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('mirror', type=Path)
    args = parser.parse_args()
    assert args.root.is_dir() and not args.mirror.exists()
    args.mirror.mkdir(parents=True, exist_ok=False)
    # Shared NFS exports can reject ownership metadata; evidence needs bytes,
    # timestamps, symlinks and hardlinks, rather than source uid/gid values.
    subprocess.run(['rsync', '-rltH', '--', str(args.root) + '/', str(args.mirror) + '/'], check=True)
    files = [path for path in args.root.rglob('*') if path.is_file()]
    verified = []
    for path in random.SystemRandom().sample(files, 3):
        relative = path.relative_to(args.root)
        digest = sha(path)
        assert digest == sha(args.mirror / relative), relative
        verified.append(dict(path=str(relative), sha256=digest))
    receipt = dict(status='PASS', source=str(args.root), mirror=str(args.mirror), verified=verified)
    for root in (args.root, args.mirror):
        with (root / 'provenance/archive-verification.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
