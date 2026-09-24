#!/usr/bin/env python3
"""Freeze the existing 144-pair gate from key metadata, without label reads."""
import csv
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

root = Path('/var/tmp/gmsd-chroma/c8/identity_v3')
root.mkdir(exist_ok=False)
sets = [('cid22', 'cid22_train', 'cid22-64.tsv', 64),
        ('safesyn', 'safesyn', 'safesyn-64.tsv', 64),
        ('kadid', 'kadid_train', 'kadid-train16-pairs.tsv', 16)]
records, files = [], set()
with (root/'decode_pairs.tsv').open('x') as out:
    for name, bank, gate, count in sets:
        source = Path('/var/tmp/featbank-impl/evidence/gate')/gate
        wanted = list(csv.DictReader(source.open(), delimiter='\t'))
        keys = pq.read_table(Path('/var/tmp/rev4-featbank/bank')/bank/'keys.parquet',
            columns=['pair_key','ref_path','dist_path','ref_pixels_sha256',
                     'dist_pixels_sha256','width','height']).to_pylist()
        by_path = {(r['ref_path'],r['dist_path']):r for r in keys}
        if name == 'kadid':
            peer = Path('/var/tmp/gmsbank/peer_gmsd/kadid_train.keys.tsv')
            assert hashlib.sha256(peer.read_bytes()).hexdigest() == '0d573a4452d908c79c7128f62d3e2d3ddc02c9ef4a57c0ba732507b9dd00d398'
            for line in peer.read_text().splitlines():
                k, a, b, w, h, rp, dp = line.split('\t')
                by_path.setdefault((rp,dp),dict(pair_key=k,ref_pixels_sha256=a,
                    dist_pixels_sha256=b,width=int(w),height=int(h),ref_path=rp,dist_path=dp))
        assert len(wanted) == count
        for index, pair in enumerate(wanted):
            r = by_path.get((pair['ref_path'],pair['dist_path']))
            if r is None:
                assert name == 'kadid' and all(Path(v).suffix == '.png' for v in pair.values())
                r = dict(pair, pair_key='path-'+hashlib.sha256((pair['ref_path']+'\0'+pair['dist_path']).encode()).hexdigest(),
                         ref_pixels_sha256=None,dist_pixels_sha256=None,width=None,height=None)
                records.append(dict(r,set=name,row_index=index,role='TRAIN',
                    gate_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
                files.update(r[k].lstrip('/') for k in ['ref_path','dist_path'])
                continue
            a, b = r['ref_pixels_sha256'], r['dist_pixels_sha256']
            assert hashlib.sha256((a+b+'legacy-rgb8').encode()).hexdigest() == r['pair_key']
            paths = [str(root/'encoded')+r[k] for k in ['ref_path','dist_path']]
            out.write('\t'.join([r['pair_key'],a,b,*paths])+'\n')
            records.append(dict(r, set=name, row_index=index, role='TRAIN',
                gate_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            files.update(r[k].lstrip('/') for k in ['ref_path','dist_path'])
(root/'source_files.txt').write_text(''.join(x+'\n' for x in sorted(files)))
(root/'population.json').write_text(json.dumps(records,indent=2,sort_keys=True)+'\n')
print('pairs',len(records),'encoded_files',len(files))
print('population_sha256',hashlib.sha256((root/'population.json').read_bytes()).hexdigest())
