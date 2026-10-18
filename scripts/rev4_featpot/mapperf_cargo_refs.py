#!/usr/bin/env python3
"""Preserve this lane's reference artifacts before cleaning its Cargo output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def check_idle(workspace, targets, lane='MAPPERF'):
    if lane == 'PERF2':
        assert workspace.name.endswith('--perf2') and workspace.name.removesuffix('--perf2') == 'zensim'
        assert ' codex-perf2 ' in (workspace / '.workongoing').read_text()
    else:
        assert workspace.name.endswith('--mapperf') or workspace == Path.home() / 'tmp/mapperf-20261017'
        assert ' codex-MAPPERF ' in (workspace / '.workongoing').read_text()
    for target in targets:
        assert target.is_dir() and not target.is_symlink()
        assert (target / 'CACHEDIR.TAG').read_text().splitlines()[0] == 'Signature: 8a477f597d28d172789f06886806bc55'
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():
            continue
        try:
            exe = (process / 'exe').resolve(strict=True)
            cwd = (process / 'cwd').resolve(strict=True)
            args = (process / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (OSError, RuntimeError):
            continue
        assert not any(exe.is_relative_to(t) or cwd.is_relative_to(t) for t in targets), process.name
        if exe.name in ['cargo', 'rustc']:
            assert not cwd.is_relative_to(workspace), (process.name, 'build in this workspace')
            assert not any(str(t) in arg for t in targets for arg in args), process.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--lane', choices=['MAPPERF', 'PERF2'], default='MAPPERF')
    parser.add_argument('--artifacts', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--remote-scratch', action='store_true')
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    if args.lane == 'PERF2':
        assert not args.remote_scratch
        assert workspace.name.endswith('--perf2') and workspace.name.removesuffix('--perf2') == 'zensim'
        names = ['target', 'target-native']
    elif args.remote_scratch:
        assert workspace == Path.home() / 'tmp/mapperf-20261017'
        names = ['priority-test-target', 'priority-release-target']
    else:
        assert workspace.name.endswith('--mapperf')
        names = ['target', 'target-before', 'zensim-bench/addrfix-runtime/target']
    targets = [workspace / p for p in names]
    check_idle(workspace, targets, args.lane)
    if args.check_only:
        if args.lane == 'PERF2':
            manifest = json.loads((args.artifacts / 'preserved-cargo/MANIFEST.json').read_text())
            assert manifest['targets'] == [str(target) for target in targets]
            assert manifest['files']
            for record in manifest['files']:
                assert sha(Path(record['preserved'])) == record['sha256'], record['preserved']
        print('Owned Cargo targets are inactive:', *targets)
        return
    dest = args.artifacts / 'preserved-cargo'
    dest.mkdir(exist_ok=False)
    records = []
    data_suffixes = {'.bin', '.csv', '.tsv', '.json', '.jsonl', '.png', '.jpg', '.webp', '.parquet', '.npy', '.npz'}
    for target in targets:
        for path in sorted(target.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(target)
            if any(part in ['build', '.fingerprint', 'incremental'] for part in relative.parts):
                continue
            if path.name == '.rustc_info.json':
                continue
            with path.open('rb') as source:
                magic = source.read(4)
            if magic != b'\x7fELF' and magic != b'\0asm' and magic[:2] != b'MZ' and path.suffix not in data_suffixes:
                continue
            digest = sha(path)
            saved = dest / (digest + '-' + path.name[:96])
            if not saved.exists():
                shutil.copy2(path, saved)
            assert sha(saved) == digest
            records.append(dict(source=str(path), preserved=str(saved), sha256=digest))
    check_idle(workspace, targets, args.lane)
    (dest / 'MANIFEST.json').write_text(json.dumps(dict(targets=[str(t) for t in targets], files=records), indent=2)+'\n')
    print('Preserved and hash-verified reference artifacts:', len(records))


if __name__ == '__main__':
    main()
