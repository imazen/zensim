#!/usr/bin/env python3
"""Inventory a supplied metric dataset and optionally extract every ZIP safely.

Original files remain unchanged. Each archive has its own directory; existing
members are accepted only after exact size/CRC validation. The JSON report keeps
complete Markdown and CSV headers, counts, missingness and small row previews.
This discovers evidence; it does not assign training roles or infer human labels.
"""
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import zipfile
import zlib


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def member_path(destination, info):
    member = PurePosixPath(info.filename)
    if (member.is_absolute() or '..' in member.parts or '\\' in info.filename
            or ':' in info.filename or stat.S_ISLNK(info.external_attr >> 16)):
        raise ValueError(f'unsafe ZIP member: {info.filename!r}')
    target = destination.joinpath(*member.parts)
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f'escaping ZIP member: {info.filename!r}')
    return target


def extract(path, destination):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        targets = [member_path(destination, info) for info in infos]
        if len(set(targets)) != len(targets):
            raise ValueError(f'duplicate ZIP names: {path}')
        pending = sum(i.file_size for i, p in zip(infos, targets) if not p.exists())
        if shutil.disk_usage(destination).free < pending + 2 * 1024**3:
            raise ValueError(f'insufficient disk for {pending} bytes plus 2 GiB reserve')
        for index, (info, target) in enumerate(zip(infos, targets)):
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            if target.exists():
                if target.stat().st_size != info.file_size:
                    raise ValueError(f'existing file has different size: {target}')
                crc = 0
                with target.open('rb') as f:
                    for block in iter(lambda: f.read(1024 * 1024), b''):
                        crc = zlib.crc32(block, crc)
                if crc != info.CRC:
                    raise ValueError(f'existing file has different CRC: {target}')
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                # Archive reads validate CRC; keep incomplete output identifiable.
                partial = target.with_name(target.name + '.extracting')
                with archive.open(info) as src, partial.open('xb') as dst:
                    shutil.copyfileobj(src, dst, 1024 * 1024)
                os.rename(partial, target)
            if index % 500 == 0:
                print(f'{path.name}: {index}/{len(infos)}', flush=True)
        return {'archive': str(path), 'sha256': digest(path),
                'members': len(infos), 'bytes': sum(i.file_size for i in infos),
                'destination': str(destination), 'crc_verified': True}


def inspect(root):
    result = {'schema': 'metric-dataset-inventory-v1', 'root': str(root),
              'markdown': [], 'csv': [], 'files': [], 'links': []}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        result['files'].append({'path': relative, 'bytes': path.stat().st_size})
        if path.suffix.lower() == '.md':
            text = path.read_text(encoding='utf-8-sig')
            result['markdown'].append({'path': relative, 'sha256': digest(path), 'text': text})
            for anchor, url in re.findall(r'\[([^\]]*)\]\((https?://[^)]+)\)', text):
                result['links'].append({'source': relative, 'text': anchor, 'href': url})
            for url in re.findall(r'https?://[^\s<>`]+', text):
                result['links'].append({'source': relative, 'text': url, 'href': url})
        if path.suffix.lower() == '.csv':
            with path.open(newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                missing = collections.Counter()
                first = []
                n = 0
                for row in reader:
                    n += 1
                    if n <= 2:
                        first.append(row)
                    missing.update(k for k, v in row.items() if not v or not v.strip())
                result['csv'].append({'path': relative, 'sha256': digest(path),
                    'rows': n, 'columns': reader.fieldnames, 'missing': dict(missing),
                    'preview': first})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--extract', action='store_true')
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('report exists; choose a new path')
    root = args.root.resolve(strict=True)
    receipts = []
    if args.extract:
        done = set()
        while True:
            pending = sorted(p for p in root.rglob('*') if p.suffix.lower() == '.zip' and p not in done)
            if not pending:
                break
            for path in pending:
                dest = path.parent / 'extracted' / path.stem
                receipts.append(extract(path, dest))
                done.add(path)
    result = inspect(root)
    result['extractions'] = receipts
    with args.report.open('x') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
        f.write('\n')
    with args.report.with_suffix('.links.jsonl').open('x') as f:
        for link in result['links']:
            f.write(json.dumps(link, ensure_ascii=False) + '\n')
    print(json.dumps({'report': str(args.report), 'archives': len(receipts),
                      'markdown': len(result['markdown']), 'csv': len(result['csv']),
                      'files': len(result['files'])}), flush=True)


if __name__ == '__main__':
    main()
