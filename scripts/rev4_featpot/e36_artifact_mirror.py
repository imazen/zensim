"""Mirror immutable artifacts of native-published views; no job claims or fits."""
import gzip
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor,as_completed

from e36_store import Client, sha
from e36_artifact_location import access, sync_directory
from e36_artifact_move import Mover
import pyarrow.parquet as pq
import pyarrow as pa

ROOT=Path('/mnt/v/output/zensim/e36-bulk-2026-10-17')
LOCAL=ROOT/'artifacts-local-v1'


def mirror_view(path):
    """Back up a complete native receipt while retaining the trainer's copy."""
    if not os.path.ismount('/mnt/tower'):
        raise ValueError('tower NFS must be mounted before metadata mirroring')
    destination=Path('/mnt/tower/output/zensim/e36-bulk-2026-10-17')/path.relative_to(ROOT)
    if destination.exists():
        if sha(destination)!=path.name:raise ValueError('conflicting archived native receipt')
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.with_name(destination.name+'.'+uuid.uuid4().hex+'.partial')
    with path.open('rb') as source,temporary.open('xb') as target:
        shutil.copyfileobj(source,target,length=1024**2);target.flush();os.fsync(target.fileno())
    if sha(temporary)!=path.name:raise ValueError('native receipt mirror SHA differs')
    stat=path.stat();os.utime(temporary,ns=(stat.st_atime_ns,stat.st_mtime_ns))
    try:os.link(temporary,destination)
    except FileExistsError:
        if sha(destination)!=path.name:raise ValueError('native receipt mirror collision')
    sync_directory(destination.parent)
    temporary.unlink()
    record=dict(utc=time.time(),sha256=path.name,bytes=stat.st_size,source=str(path),destination=str(destination),sha_verified=True,
                local_retained_reason='feature-only trainer pool not yet materialized',fits=0)
    with (ROOT/'owner-svt-ssd-v1/view-mirrors.jsonl').open('a') as stream:
        stream.write(json.dumps(record)+'\n');stream.flush();os.fsync(stream.fileno())


def run(pool):
    import fcntl
    lock=(ROOT/'owner-svt-ssd-v1/mirror.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    storage=json.loads((ROOT/'storage-runtime.json').read_text())
    os.environ['E36_STORE_CONNECT_GATE']=str(ROOT/'storage-connect-gate')
    mover=Mover(ROOT,storage)
    seen=set();queue={};ledger_cache={};metadata_pending={}
    cutoff=json.loads((ROOT/'owner-svt-ssd-v1/live-cutoff.json').read_text())['closed_unix_ns']/1e9-120
    while time.time()<1792256400 and not (ROOT/'owner-svt-ssd-v1/mirror-stop.json').exists():
        free=shutil.disk_usage(LOCAL).free
        eager=(ROOT/'owner-svt-ssd-v1/urgent-disk-policy.json').exists()
        if eager or free<48*1024**3:
            published=set()
            for name in ['fallback-v6','svt-additions-v1']:
                for ledger in (ROOT/'runs'/name).glob('ledger*.parquet'):
                    stat=ledger.stat();identity=(stat.st_mtime_ns,stat.st_size)
                    previous=ledger_cache.get(ledger)
                    if previous is not None and previous[0]==identity:
                        published.update(previous[1]);continue
                    try:
                        rows=pq.read_table(ledger,columns=['status','attempts','output_sha']).to_pylist()
                    except (OSError,pa.ArrowInvalid):
                        # A sidecar still being written grants no eligibility.
                        continue
                    values={row['output_sha'] for row in rows if row['status']=='done' and row['attempts']==1 and row['output_sha']}
                    ledger_cache[ledger]=(identity,values);published.update(values)
            for directory in [ROOT/'runs/fallback-v6/blobs',ROOT/'runs/svt-additions-v1/blobs']:
                if not directory.exists():continue
                for path in directory.iterdir():
                    if not path.is_file() or path.name not in published or path.name in seen or path.stat().st_mtime<cutoff:continue
                    doc=json.loads(gzip.decompress(path.read_bytes()))
                    if doc.get('schema')!='e36-native-ladder-receipt-v1' or doc.get('complete') is not True:
                        raise ValueError('mirror requires complete native view')
                    if sha(path)!=path.name:
                        raise ValueError('native view CAS changed')
                    if eager:metadata_pending[path.name]=path
                    for row in doc['rows']:
                        bundle=row['bundle']
                        if bundle['host']=='dev' and bundle['root']==str(LOCAL):
                            digest=bundle['sha256'];artifact=LOCAL/bundle['relative_path']
                            if artifact.exists():queue[digest]=(artifact,path.name)
                    seen.add(path.name)
            # Reclaim space with fewer remote metadata operations. Tiny-first
            # directory ordering can spend minutes copying sub-MB objects.
            ordered=sorted(queue.items(),key=lambda item:item[1][0].stat().st_size if item[1][0].exists() else 0,reverse=True)
            batch=[]
            for digest,(artifact,view_sha) in ordered[:4]:
                if (ROOT/'owner-svt-ssd-v1/mirror-stop.json').exists():return
                if not eager and shutil.disk_usage(LOCAL).free>=56*1024**3:break
                if not artifact.exists():
                    del queue[digest];continue
                batch.append(pool.submit(mover.move,digest,artifact,view_sha))
            for future in as_completed(batch):
                record=future.result()
                if record is None:continue
                del queue[record['sha256']]
                print(json.dumps(dict(mirrored_sha256=record['sha256'],bytes=record['bytes'],free_bytes=shutil.disk_usage(LOCAL).free)),flush=True)
            # Artifact eviction owns the urgent disk relief. Back up a bounded
            # number of trainer-needed receipts between reclamation batches.
            if eager and shutil.disk_usage(LOCAL).free>40*1024**3:
                for key,path in list(metadata_pending.items())[:4]:
                    try:mirror_view(path)
                    except (OSError,ValueError) as error:
                        failure=dict(event='native-receipt-backup-refused',path=str(path),error=str(error),
                                     retained_local=True,artifact_eviction_continues=True,sha256=key,utc=time.time())
                        with (ROOT/'owner-svt-ssd-v1/view-backup-failures.jsonl').open('a') as stream:
                            stream.write(json.dumps(failure)+'\n');stream.flush();os.fsync(stream.fileno())
                        del metadata_pending[key]
                        print(json.dumps(failure),flush=True)
                    else:del metadata_pending[key]
        print(json.dumps(dict(utc=time.time(),free_bytes=shutil.disk_usage(LOCAL).free,queued_bundles=len(queue),seen_complete_views=len(seen))),flush=True)
        time.sleep(.1 if queue and (eager or shutil.disk_usage(LOCAL).free<48*1024**3) else 10)


def main():
    with ThreadPoolExecutor(max_workers=4,thread_name_prefix='e36-storage') as pool:
        run(pool)


if __name__=='__main__':main()
