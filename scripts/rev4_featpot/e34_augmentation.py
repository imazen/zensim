"""Label-free E34 source inventory; explicit inputs, fresh outputs, no label reads.

The canonical split owner is imported from its pinned capture. Pixel operations
and dHash belong to the native e34_pixels and check_holdout_overlap executables.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import pyarrow.parquet as pq

BROAD = {'1054', '2010', '6068', '6610', '7066', '8206', '8384', '9380'}
D11 = {'1444', '1458'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')


def admitted(path):
    p = Path(path).resolve(strict=True)
    # Population names are path components, not words appearing in an image's
    # subject (the admitted corpus includes terminal-window screenshots).
    parts = p.parts[:-1] if p.suffix.lower() in {'.png', '.jpg', '.jpeg', '.bmp'} else p.parts
    if '_sealed' in str(p).lower() or any(
            token in {'terminal', 't0'} for part in parts
            for token in re.split(r'[_\-.]', part.lower())):
        raise ValueError(f'protected source: {p}')
    return p


def inventory(a):
    out = Path(a.out)
    out.mkdir(exist_ok=False)
    spec = importlib.util.spec_from_file_location('e34_origin_split', admitted(a.split_owner))
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    split_rows = list(csv.DictReader(admitted(a.family_map).open(), delimiter='\t'))
    families = {r['id']: r['family'] or 'origin:' + r['id'] for r in split_rows}
    excluded_families = {families[x] for x in BROAD}
    paths = {}
    for p in Path(a.pixels).rglob('*.png'):
        if p.name.endswith('.hdr.png'):
            continue
        oid = owner.origin_id(p.name)
        if oid is not None:
            if oid in paths:
                if sha(paths[oid]) != sha(p):
                    raise ValueError(f'ambiguous SDR origin {oid}: {paths[oid]}, {p}')
                paths[oid] = min((paths[oid], p), key=lambda x: (len(x.parts), str(x)))
            else:
                paths[oid] = p
    dual_train = [r for r in split_rows if owner.split_of(r['id']) == 'train' and r['split'] == 'train']
    assert len(dual_train) == 1018
    assert D11 <= {r['id'] for r in dual_train}
    eligible = [r for r in dual_train if r['id'] not in D11]
    assert len(eligible) == 1016
    missing = [r for r in eligible if r['id'] not in paths]
    named_exclusions = dict(owner='owner', date='2026-10-15', decision='D11',
        origins=sorted(D11), reason='DNG sources without official SDR renditions; no compatibility thumbnails')
    write(out / 'source-preflight.json', dict(labels_read=False, dual_train=len(dual_train),
          after_named_exclusions=len(eligible), named_exclusions=named_exclusions,
          missing_sdr=missing, complete=not missing))
    if missing:
        raise ValueError('missing registered SDR origins: ' + ', '.join(r['id'] for r in missing))
    origins = []
    for r in split_rows:
        oid = r['id']
        if oid in D11:
            continue
        if owner.split_of(oid) != 'train' or r['split'] != 'train':
            continue
        p = admitted(paths[oid])
        origins.append(dict(origin=oid, family=families[oid], source=str(p),
                            source_sha256=sha(p), excluded_family=families[oid] in excluded_families))
    refs = {}
    def add(path, role, expected=None):
        p = admitted(path)
        digest = sha(p)
        if expected and digest != expected:
            raise ValueError(f'reference hash changed: {p}')
        row = refs.setdefault(digest, dict(source=str(p), source_sha256=digest, roles=[]))
        if role not in row['roles']:
            row['roles'].append(role)
    for oid in sorted(BROAD):
        add(paths[oid], 'broad:' + oid)
    for row in json.loads(admitted(a.identity).read_text()):
        add(row['ref_path'], row['ref_group'], row['file_sha256'])
    for row in json.loads(admitted(a.packet).read_text())['cases']:
        add(row['reference'], 'steering:' + row['key'], row['reference_sha256'])
    keys_bindings = []
    for name in ('kadid_train', 'kadid_select', 'tid2013', 'konfig_train', 'konfig_val', 'cid22_a25'):
        p = admitted(Path(a.bank) / name / 'keys.parquet')
        keys_bindings.append(dict(path=str(p), sha256=sha(p)))
        for row in pq.read_table(p, columns=['ref_path', 'ref_group']).to_pylist():
            if not any(row['ref_group'] in v['roles'] for v in refs.values()):
                add(row['ref_path'], row['ref_group'])
    for key in a.instrument_keys:
        p = admitted(key)
        keys_bindings.append(dict(path=str(p), sha256=sha(p)))
        seen = set()
        for row in pq.read_table(p, columns=['ref_path', 'ref_group']).to_pylist():
            if row['ref_path'] not in seen:
                add(row['ref_path'], 'instrument:' + row['ref_group'])
                seen.add(row['ref_path'])
    safe_keys = admitted(Path(a.bank) / 'safesyn/keys.parquet')
    safe = sorted(set(pq.read_table(safe_keys, columns=['ref_path'])['ref_path'].to_pylist()))
    for p in safe:
        admitted(p)
    keys_bindings.append(dict(path=str(safe_keys), sha256=sha(safe_keys)))
    refs_dir = out / 'excluded-png'
    refs_dir.mkdir()
    for digest, row in refs.items():
        row['destination'] = str(refs_dir / (digest + '.png'))
    write(out / 'normalize.json', dict(mode='normalize', longest_side=0,
          output=str(out / 'normalization.jsonl'), rows=list(refs.values())))
    sources = sorted({r['source'] for r in origins if not r['excluded_family']} | set(safe))
    with (out / 'audit.csv').open('x') as f:
        writer = csv.writer(f)
        writer.writerow(['source'])
        writer.writerows([p] for p in sources)
    write(out / 'inventory.json', dict(schema='e34-pixel-inventory-v1', labels_read=False,
          corpus_revision='187fbf338ce08e8e6654db7f04ddae58d5263da2',
          family_map=dict(path=a.family_map, sha256=sha(a.family_map)),
          split_owner=dict(path=a.split_owner, sha256=sha(a.split_owner)),
          keys_bindings=keys_bindings, named_exclusions=named_exclusions,
          origins=origins, exclusions=list(refs.values()),
          safesyn_sources=safe, audit_sources=len(sources), audit_references=len(refs)))
    print(json.dumps(dict(dual_train_origins=len(origins), audit_sources=len(sources),
                         audit_references=len(refs), safesyn_references=len(safe))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('inventory')
    for key in ('out', 'split-owner', 'family-map', 'pixels', 'identity', 'packet', 'bank'):
        i.add_argument('--' + key, required=True)
    i.add_argument('--instrument-keys', action='append', default=[])
    a = p.parse_args()
    inventory(a)


if __name__ == '__main__':
    main()
