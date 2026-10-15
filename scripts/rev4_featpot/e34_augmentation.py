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


def audit(a):
    root=Path(a.root)
    inv=json.loads((root/'inventory.json').read_text())
    hashes=json.loads((root/'audit.hashes.json').read_text())
    normalization=[json.loads(line) for line in (root/'normalization.jsonl').read_text().splitlines()]
    assert len(normalization)==inv['audit_references']==len(hashes['holdout'])
    assert len(hashes['training'])==inv['audit_sources']
    assert hashes['threshold']==10 and hashes['hash_era']=='bt709-encoded-luma-zenresize-lanczos-gray8-v1'
    train={r['path']:r for r in hashes['training']}
    hold={r['path']:r for r in hashes['holdout']}
    for row in normalization:
        assert row['sha256']==hold[row['destination']]['sha256']
    for row in inv['origins']:
        if not row['excluded_family']:assert train[row['source']]['sha256']==row['source_sha256']
    flags={r['training'] for r in hashes['close_pairs'] if r['strict_flag']}
    families={r['family'] for r in inv['origins'] if r['excluded_family'] or r['source'] in flags}
    eligible=[r for r in inv['origins'] if r['family'] not in families]
    rejected=[dict(r,reason='registered broad family' if r['excluded_family'] else
        'dHash <= 10' if r['source'] in flags else 'family closure of a dHash rejection')
        for r in inv['origins'] if r['family'] in families]
    assert len(eligible)+len(rejected)==1016
    safe=set(inv['safesyn_sources']);safe_rows=[r for r in hashes['training'] if r['path'] in safe]
    intersection=[]
    for reference in inv['exclusions']:
        for role in reference['roles']:
            if not role.startswith('broad:'):continue
            h=hold[reference['destination']]['dhash']
            matched=[dict(r,hamming=(r['dhash']^h).bit_count()) for r in safe_rows if (r['dhash']^h).bit_count()<=10]
            intersection.append(dict(origin=role.split(':')[1],exact_file_matches=sum(r['sha256']==reference['source_sha256'] for r in safe_rows),
                minimum_hamming=min((r['dhash']^h).bit_count() for r in safe_rows),screened_matches=matched))
    review=json.loads(Path(a.context_review).read_text())
    expected={r['sha256'] for x in intersection for r in x['screened_matches']}
    assert expected=={r['source_sha256'] for r in review['rows']}
    assert all(r['same_picture'] is False for r in review['rows'])
    receipt=dict(schema='e34-eligibility-v1',labels_read=False,named_exclusions=inv['named_exclusions'],
        dual_train=1018,after_named_exclusions=1016,remaining_origins=len(eligible),rejected_origins=len(rejected),
        broad_family_rejections=sum(r['excluded_family'] for r in inv['origins']),
        direct_dhash_rejections=sum(r['source'] in flags for r in inv['origins']),
        excluded_families=sorted(families),eligible=eligible,rejected=rejected,
        safesyn_intersection=intersection,context_review=review,
        safesyn_conclusion='No exact-file matches to the eight broad references; all four dHash<=10 candidates are visually different pictures. This pixel screen does not recover unavailable parent/crop lineage.',
        bindings={name:sha(root/name) for name in ['inventory.json','audit.hashes.json','normalization.jsonl']})
    write(root/'eligibility.json',receipt)
    refs=root/'references';refs.mkdir()
    rows=[dict(r,destination=str(refs/(r['origin']+'.png')),max_decode_pixels=120000000) for r in eligible]
    write(root/'resize.json',dict(mode='normalize',longest_side=256,output=str(root/'references.jsonl'),rows=rows))
    print(json.dumps({k:receipt[k] for k in ['remaining_origins','rejected_origins','broad_family_rejections','direct_dhash_rejections']}))


def reference_job(a):
    root=Path(a.root)
    eligible={r['origin']:r for r in json.loads((root/'eligibility.json').read_text())['eligible']}
    rows=[]
    for line in (root/'references.jsonl').read_text().splitlines():
        r=json.loads(line);oid=Path(r['destination']).stem
        assert r['source_sha256']==eligible[oid]['source_sha256']
        rows.append(dict(origin=oid,source=r['destination'],source_sha256=r['sha256']))
    assert len(rows)==len(eligible) and len({r['origin'] for r in rows})==len(rows)
    write(root/'reference-features-job.json',dict(output=str(root/'reference-features.jsonl'),rows=rows))


def select(a):
    import numpy as np
    import sklearn
    from sklearn.cluster import KMeans
    from threadpoolctl import threadpool_limits
    root=Path(a.root)
    admission=json.loads((root/'eligibility.json').read_text())
    eligible={r['origin']:r for r in admission['eligible']}
    rows=[json.loads(line) for line in (root/'reference-features.jsonl').read_text().splitlines()]
    rows.sort(key=lambda r:r['origin'])
    assert {r['origin'] for r in rows}==eligible.keys()
    feature_ids=[int(k) for k,v in rows[0]['features']]
    assert all([int(k) for k,v in r['features']]==feature_ids for r in rows)
    def cluster(panel,k,seed):
        assert len(panel)>=k
        x=np.asarray([[v for _,v in r['features']] for r in panel],dtype=np.float64)
        assert np.isfinite(x).all()
        mu=x.mean(axis=0);sd=x.std(axis=0);sd[sd==0]=1
        z=(x-mu)/sd
        with threadpool_limits(limits=1):
            fit=KMeans(n_clusters=k,random_state=seed,n_init=4).fit(z)
        assert len(set(fit.labels_))==k
        picks=[];sizes=[]
        for c in range(k):
            ix=np.flatnonzero(fit.labels_==c)
            assert len(ix)>0
            chosen=ix[np.square(z[ix]-fit.cluster_centers_[c]).sum(axis=1).argmin()]
            picks.append(panel[int(chosen)]['origin']);sizes.append(len(ix))
        return dict(origins=picks,cluster_sizes=sizes,singletons=sum(n==1 for n in sizes),
            means=mu.tolist(),scales=sd.tolist(),inertia=float(fit.inertia_),iterations=int(fit.n_iter_),seed=seed,n_init=4)
    augmentation=cluster(rows,128,17)
    families={eligible[o]['family'] for o in augmentation['origins']}
    dev=cluster([r for r in rows if eligible[r['origin']]['family'] not in families],8,18)
    dev_families={eligible[o]['family'] for o in dev['origins']}
    assert families.isdisjoint(dev_families)
    result=dict(schema='e34-reference-selection-v1',labels_read=False,reference_only=True,
        feature_ids=feature_ids,reference_features_sha256=sha(root/'reference-features.jsonl'),
        eligibility_receipt_sha256=sha(root/'eligibility.json'),sklearn_version=sklearn.__version__,
        method='standardize reference features, Lloyd k-means, centroid-nearest per nonempty cluster; keep singletons',
        augmentation=augmentation,development=dev,family_overlap=[],origin_count=len(eligible))
    write(root/'selection.json',result)
    print(json.dumps(dict(augmentation=128,dev=8,features=len(feature_ids),
        augmentation_singletons=augmentation['singletons'],dev_singletons=dev['singletons'],family_overlap=0)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    i = sub.add_parser('inventory')
    for key in ('out', 'split-owner', 'family-map', 'pixels', 'identity', 'packet', 'bank'):
        i.add_argument('--' + key, required=True)
    i.add_argument('--instrument-keys', action='append', default=[])
    for command in ['audit','reference-job','select']:
        c=sub.add_parser(command)
        c.add_argument('--root',required=True)
        if command=='audit':c.add_argument('--context-review',required=True)
    a = p.parse_args()
    {'inventory':inventory,'audit':audit,'reference-job':reference_job,'select':select}[a.command](a)


if __name__ == '__main__':
    main()
