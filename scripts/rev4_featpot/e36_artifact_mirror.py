"""Mirror immutable artifacts of native-published views; no job claims or fits."""
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time

from e36_store import Client, sha
from e36_artifact_location import access, sync_directory
import pyarrow.parquet as pq
import pyarrow as pa

ROOT=Path('/mnt/v/output/zensim/e36-bulk-2026-10-17')
LOCAL=ROOT/'artifacts-local-v1'


def main():
    import fcntl
    lock=(ROOT/'owner-svt-ssd-v1/mirror.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    storage=json.loads((ROOT/'storage-runtime.json').read_text())
    client=Client(storage['command'],storage['program_sha256'])
    journal=ROOT/'owner-svt-ssd-v1/artifact-relocations.jsonl'
    seen=set();queue={}
    cutoff=json.loads((ROOT/'owner-svt-ssd-v1/live-cutoff.json').read_text())['closed_unix_ns']/1e9-120
    while time.time()<1792256400 and not (ROOT/'owner-svt-ssd-v1/mirror-stop.json').exists():
        free=shutil.disk_usage(LOCAL).free
        if free<48*1024**3:
            published=set()
            for name in ['fallback-v6','svt-additions-v1']:
                for ledger in (ROOT/'runs'/name).glob('ledger*.parquet'):
                    try:
                        rows=pq.read_table(ledger,columns=['status','attempts','output_sha']).to_pylist()
                    except (OSError,pa.ArrowInvalid):
                        # A sidecar still being written grants no eligibility.
                        continue
                    published.update(row['output_sha'] for row in rows if row['status']=='done' and row['attempts']==1 and row['output_sha'])
            for directory in [ROOT/'runs/fallback-v6/blobs',ROOT/'runs/svt-additions-v1/blobs']:
                if not directory.exists():continue
                for path in directory.iterdir():
                    if not path.is_file() or path.name not in published or path.name in seen or path.stat().st_mtime<cutoff:continue
                    doc=json.loads(gzip.decompress(path.read_bytes()))
                    if doc.get('schema')!='e36-native-ladder-receipt-v1' or doc.get('complete') is not True:
                        raise ValueError('mirror requires complete native view')
                    if sha(path)!=path.name:
                        raise ValueError('native view CAS changed')
                    for row in doc['rows']:
                        bundle=row['bundle']
                        if bundle['host']=='dev' and bundle['root']==str(LOCAL):
                            digest=bundle['sha256'];artifact=LOCAL/bundle['relative_path']
                            if artifact.exists():queue[digest]=(artifact,path.name)
                    seen.add(path.name)
            # Reclaim space with fewer remote metadata operations. Tiny-first
            # directory ordering can spend minutes copying sub-MB objects.
            ordered=sorted(queue.items(),key=lambda item:item[1][0].stat().st_size if item[1][0].exists() else 0,reverse=True)
            for digest,(artifact,view_sha) in ordered:
                if (ROOT/'owner-svt-ssd-v1/mirror-stop.json').exists():return
                if shutil.disk_usage(LOCAL).free>=56*1024**3:break
                if not artifact.exists():
                    del queue[digest];continue
                with access(ROOT,digest,exclusive=True):
                    if sha(artifact)!=digest:raise ValueError('local artifact hash changed')
                    ack=client.put(artifact)
                    verified=client.verify(digest)
                    if verified['bytes']!=artifact.stat().st_size or ack['bytes']!=verified['bytes']:
                        raise ValueError('mirror size/hash verification failed')
                    record=dict(utc=time.time(),sha256=digest,bytes=verified['bytes'],completed_view_sha256=view_sha,
                                local_path=str(artifact),canonical_host=storage['host'],canonical_root=storage['root'],
                                relative_path=verified['relative_path'],program_sha256=verified['program_sha256'],
                                sha_verified=True,action='canonical-mirror-verified-before-local-unlink',fits=0)
                    with journal.open('a') as stream:
                        stream.write(json.dumps(record,separators=(',',':'))+'\n');stream.flush();os.fsync(stream.fileno())
                    sync_directory(journal.parent)
                    artifact.unlink()
                    sync_directory(artifact.parent)
                del queue[digest]
                print(json.dumps(dict(mirrored_sha256=digest,bytes=record['bytes'],free_bytes=shutil.disk_usage(LOCAL).free)),flush=True)
        print(json.dumps(dict(utc=time.time(),free_bytes=shutil.disk_usage(LOCAL).free,queued_bundles=len(queue),seen_complete_views=len(seen))),flush=True)
        time.sleep(10)


if __name__=='__main__':main()
