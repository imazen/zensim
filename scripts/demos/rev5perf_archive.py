"""Mirror completed raw evidence; verify three random files or every file."""
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
    parser.add_argument('--verify-all', action='store_true')
    parser.add_argument('--dereference', action='store_true', help='copy file links as their target bytes')
    parser.add_argument('--exclude-operator-locks', action='store_true', help='omit .workongoing control files')
    args = parser.parse_args()
    assert args.root.is_dir() and not args.mirror.exists()
    paths = list(args.root.rglob('*'))
    if args.verify_all and args.dereference:
        assert not any(p.is_symlink() and p.is_dir() for p in paths), 'directory links require separate admission'
    args.mirror.mkdir(parents=True, exist_ok=False)
    # Shared NFS exports can reject ownership metadata; evidence needs bytes,
    # timestamps, symlinks and hardlinks, rather than source uid/gid values.
    options = (['--copy-links'] if args.dereference else [])
    if args.verify_all:
        options.append('--stats')
    if args.exclude_operator_locks:
        options.append('--exclude=.workongoing')
    subprocess.run(['rsync', '-rltH', *options, '--', str(args.root) + '/', str(args.mirror) + '/'], check=True)
    files = [p for p in paths if p.is_file() and not (args.exclude_operator_locks and p.name=='.workongoing')]
    assert files, 'completed evidence root must contain files'
    verified = []
    for path in files if args.verify_all else random.SystemRandom().sample(files, 3):
        relative = path.relative_to(args.root)
        digest = sha(path)
        if args.dereference:
            assert not (args.mirror / relative).is_symlink(), relative
        assert digest == sha(args.mirror / relative), relative
        verified.append(dict(path=str(relative), sha256=digest))
    receipt = dict(status='PASS', source=str(args.root), mirror=str(args.mirror), verified=verified,
                   verified_all=args.verify_all,dereferenced_file_links=args.dereference,
                   excluded_operator_locks=args.exclude_operator_locks)
    for root in (args.root, args.mirror):
        (root / 'provenance').mkdir(exist_ok=True)
        with (root / 'provenance/archive-verification.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
            stream.write('\n')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
