#!/usr/bin/env python3
"""Prepare the preregistered TRAIN-only native and box-derived pixel inputs."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path('/var/tmp/gmsd-chroma/c8')


def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def select():
    import pyarrow.parquet as pq
    source = Path('/var/tmp/gmsbank/calibration')
    expected = {
        'selection.json': 'e01e1f09b4688a7a77e92b28a367549ddb5975191d0ae2656d700402f9d7a84f',
        'planes.tsv': 'aa49fbe114acb464c9d88316c28b7bd1e0b1ee8a17b6def9a742b6d4d034a6d3',
    }
    for name,digest in expected.items():
        assert sha(source/name) == digest
    selected = {r['pair_key']:r for r in json.loads((source/'selection.json').read_text())['pairs']}
    groups = {}
    keys = dict(cid22_train='c99a0887705b17bff05bdbfc55b89f9b4f060bc94a009302543f15d9a0127219',
                safesyn='12d48d7fc02afd5026067a348ea20a3896ea6de9a94bd99a25ec2db071c2b28f')
    for name,digest in keys.items():
        path = Path('/var/tmp/rev4-featbank/bank')/name/'keys.parquet'
        assert sha(path) == digest
        for row in pq.read_table(path,columns=['pair_key','ref_group']).to_pylist():
            if row['pair_key'] in selected and selected[row['pair_key']]['set'] == name:
                groups[row['pair_key']] = row['ref_group']
    assert groups.keys() == selected.keys()
    rows = []
    files = set()
    for line in (source/'planes.tsv').read_text().splitlines():
        key,w,h,ref,dist = line.split('\t')
        assert key in selected and '/' not in ref and '/' not in dist
        rows.append(dict(**selected[key], ref_group=groups[key], width=int(w),height=int(h),
                         ref_rgb=ref,dist_rgb=dist))
        files.update([ref,dist])
    assert len(rows) == len(selected) == 652
    ROOT.mkdir(parents=True,exist_ok=True)
    (ROOT/'source_rows.json').write_text(json.dumps(dict(source_sha256=expected,keys_sha256=keys,rows=rows),indent=2)+'\n')
    (ROOT/'native_files.txt').write_text('\n'.join(sorted(files))+'\n')
    print('frozen_native_pairs',len(rows))
    print('source_rows_sha256',sha(ROOT/'source_rows.json'))


def derive():
    import numpy as np
    from collections import defaultdict,Counter
    records = json.loads((ROOT/'source_rows.json').read_text())['rows']
    out = ROOT/'calibration'
    out.mkdir(exist_ok=False)
    (out/'rgb').mkdir()
    chosen = [dict(r,geometry='native',parent_pair_key=r['pair_key']) for r in records]
    rejected = []
    for limit,kind in [(128,'tiny'),(384,'small')]:
        strata = defaultdict(lambda: defaultdict(list))
        for r in records:
            w,h = r['width'],r['height']
            longest = max(w,h)
            nw,nh = ((2*v*limit+longest)//(2*longest) for v in (w,h))
            if min(nw,nh)<64:
                rejected.append(dict(parent_pair_key=r['pair_key'],size_class=kind,width=nw,height=nh))
                continue
            assert limit<longest
            strata[(r['set'],r['content_class'])][r['ref_group']].append((r,nw,nh))
        for _,groups in sorted(strata.items()):
            for group in sorted(groups,key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:32]:
                for r,nw,nh in sorted(groups[group],key=lambda t:t[0]['pair_key'])[:2]:
                    row=dict(r,geometry=f'box{limit}',parent_pair_key=r['pair_key'],
                             source_width=r['width'],source_height=r['height'],width=nw,height=nh,size_class=kind)
                    row['pair_key']=hashlib.sha256((r['pair_key']+f':box{limit}:{nw}x{nh}').encode()).hexdigest()
                    chosen.append(row)
    cache = {}
    verified = set()
    def pixels(name,w,h):
        path=ROOT/'native'/name
        if name not in verified:
            assert sha(path)==path.stem
            verified.add(name)
        image=np.fromfile(path,dtype=np.uint8)
        assert image.size==w*h*3
        return image.reshape(h,w,3)
    with (out/'planes.tsv').open('x') as f:
        for number,row in enumerate(chosen):
            names=[]
            for key in ['ref_rgb','dist_rgb']:
                original=row[key]
                if row['geometry']=='native':
                    pixels(original,row['width'],row['height'])
                    names.append('../native/'+original)
                    continue
                signature=(original,row['width'],row['height'])
                if signature not in cache:
                    src=pixels(original,row['source_width'],row['source_height'])
                    h,w,_=src.shape; nw,nh=row['width'],row['height']
                    xs=np.arange(nw+1,dtype=np.int64)*w//nw
                    ys=np.arange(nh+1,dtype=np.int64)*h//nh
                    reduced=np.empty((nh,nw,3),dtype=np.uint8)
                    for y in range(nh):
                        summed=src[ys[y]:ys[y+1]].sum(axis=0,dtype=np.uint64)
                        sums=np.add.reduceat(summed,xs[:-1],axis=0)
                        count=((ys[y+1]-ys[y])*np.diff(xs)).astype(np.uint64)[:,None]
                        reduced[y]=((2*sums+count)//(2*count)).astype(np.uint8)
                    raw=reduced.tobytes();digest=hashlib.sha256(raw).hexdigest()
                    path=out/'rgb'/(digest+'.rgb')
                    if not path.exists():path.write_bytes(raw)
                    cache[signature]='rgb/'+path.name
                names.append(cache[signature])
            row['prepared_rgb']=names
            f.write('\t'.join(map(str,[row['pair_key'],row['width'],row['height'],*names]))+'\n')
            if number%100==0:print('prepared_pair',number,flush=True)
    assert len({r['pair_key'] for r in chosen})==len(chosen)
    counts=Counter('/'.join(r[k] for k in ['set','content_class','size_class']) for r in chosen)
    result=dict(source_rows_sha256=sha(ROOT/'source_rows.json'),rows=chosen,
                counts=dict(sorted(counts.items())),rejected_geometries=rejected,
                planes_sha256=sha(out/'planes.tsv'),human_labels_read=False)
    (out/'selection.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(pairs=len(chosen),counts=result['counts'],rejected=len(rejected)),sort_keys=True))


if __name__=='__main__':
    {'select':select,'derive':derive}[sys.argv[1]]()
