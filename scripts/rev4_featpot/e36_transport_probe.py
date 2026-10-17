"""Concurrent storage readback admission only; no job scheduling or data generation."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
from e36_store import Client,sha

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    spec=json.loads((a.root/'storage-runtime.json').read_text());os.environ['E36_STORE_CONNECT_GATE']=str(a.root/'storage-connect-gate')
    receipts=json.loads((a.root/'runs/first-producer-v5/census.json').read_text())['output_receipts'];objects=[]
    for r in receipts:
        d=json.loads(gzip.decompress(Path(r['path']).read_bytes()));objects.append(d['rows'][0]['bundle']['sha256'])
    def check(i):
        client=Client(spec['command'],spec['program_sha256'])
        try:return client.verify(objects[i%len(objects)])
        finally:client.close()
    with ThreadPoolExecutor(max_workers=28) as workers:results=list(workers.map(check,range(28)))
    if len(results)!=28 or not all(r['ok'] for r in results):raise ValueError('concurrent storage readback admission failed')
    with a.out.open('x') as f:json.dump(dict(schema='e36-concurrent-storage-admission-v1',connections=28,verified=28,results=results,client_sha256=sha(Path(__file__).with_name('e36_store.py')),labels_read=False,fits=0),f,indent=2);f.write('\n')
