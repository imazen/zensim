"""Mirror immutable artifacts of native-published views; no job claims or fits."""
import gzip
import heapq
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
METADATA_CLIENT=None
METADATA_ARCHIVED=None
MIRROR_LOCKS=[]

from e36_archive_runs import native_runs


def mirror_view(path):
    """Back up a complete native receipt while retaining the trainer's copy."""
    if (ROOT/'storage-archive-v2.json').exists():
        global METADATA_CLIENT,METADATA_ARCHIVED
        # New receipts use the distinct disk archive. Existing NFS backups stay
        # where recorded; no file is opened through both namespace paths.
        from e36_archive_store import ArchiveClient
        storage=json.loads((ROOT/'storage-archive-v2.json').read_text())
        if METADATA_ARCHIVED is None:
            METADATA_ARCHIVED=set()
            for journal in ['view-mirrors.jsonl','view-mirrors-v2.jsonl']:
                source=ROOT/'owner-svt-ssd-v1'/journal
                if source.exists():
                    with source.open() as stream:METADATA_ARCHIVED.update(json.loads(line)['sha256'] for line in stream)
        if path.name in METADATA_ARCHIVED:return
        if METADATA_CLIENT is None:METADATA_CLIENT=ArchiveClient(storage['command'],storage['program_sha256'])
        with (ROOT/'owner-svt-ssd-v1/view-mirrors-v2.jsonl').open('a') as stream:
            answer=METADATA_CLIENT.put(path);actual=METADATA_CLIENT.verify(path.name)
            if actual!=answer or actual['bytes']!=path.stat().st_size:raise ValueError('native receipt archive differs')
            stream.write(json.dumps(dict(utc=time.time(),source=str(path),canonical_root=storage['root'],
                **actual,local_retained_reason='feature-only trainer pool not yet materialized',fits=0))+'\n')
            stream.flush();os.fsync(stream.fileno());METADATA_ARCHIVED.add(path.name)
        return
    if not os.path.ismount('/mnt/tower'):raise ValueError('tower NFS must be mounted before metadata mirroring')
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
    lock=(ROOT/'owner-svt-ssd-v1/mirror.lock').open('a');MIRROR_LOCKS.append(lock)
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    storage_path=ROOT/'storage-archive-v2.json'
    storage=json.loads((storage_path if storage_path.exists() else ROOT/'storage-runtime.json').read_text())
    if storage.get('encoding')=='zstd':
        proof=json.loads((ROOT/'owner-svt-ssd-v1/zstd-map-probe-v1/receipt.json').read_text())
        if proof.get('admitted') is not True or proof['storage']['sha256']!=sha(storage_path) or len(proof['maps'])!=2:
            raise ValueError('actual map-owner zstd readback admission required')
        os.environ['E36_ARCHIVE_STAGE']=str(ROOT/'archive-staging-v2')
        os.environ['E36_ARCHIVE_ROOT']=str(ROOT)
    os.environ['E36_STORE_CONNECT_GATE']=str(ROOT/'storage-connect-gate')
    mover=Mover(ROOT,storage)
    if (ROOT/'owner-svt-ssd-v1/pack-storage-admission-v1.json').exists():
        from e36_pack_archive import Dispatcher
    else:
        from e36_mirror_dispatch import Dispatcher
    dispatcher=Dispatcher(ROOT,pool,storage)
    artifacts_only=os.environ.get('E36_ARTIFACTS_ONLY')=='1'
    interleave=os.environ.get('E36_ARCHIVE_INTERLEAVE_LOCK')
    if artifacts_only and not (ROOT/'owner-svt-ssd-v1/native-continuation-final-v1.json').exists():raise ValueError('native control must be closed before archive-only drain')
    if interleave:
        if not artifacts_only:raise ValueError('lock windows require closed native generation')
        from e36_archive_windows import Windows
        dispatcher=Windows(ROOT,dispatcher,interleave)
    ready=os.environ.get('E36_RESOURCE_READY_PATH')
    if ready:
        temporary=Path(ready).with_name(Path(ready).name+'.'+uuid.uuid4().hex+'.partial')
        with temporary.open('x') as stream:
            json.dump(dict(role='mirror',pid=os.getpid(),scope='/sys/fs/cgroup'+Path('/proc/self/cgroup').read_text().strip().split('::')[1],utc=time.time(),fits=0),stream)
            stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,ready)
        sync_directory(Path(ready).parent)
    seen=set();queue={};ledger_cache={};metadata_pending={};priority=[];next_scan=0
    cutoff=0 if artifacts_only else json.loads((ROOT/'owner-svt-ssd-v1/live-cutoff.json').read_text())['closed_unix_ns']/1e9-120
    inventory_done=False;closed_proof=None;drain_complete=False
    try:
        if artifacts_only:
            from e36_closed_archive import inventory
            queue,closed=inventory(ROOT)
            priority=[(-path.stat().st_size,h) for h,(path,_) in queue.items()];heapq.heapify(priority)
            seen={p['sha256'] for p in closed['views']};next_scan=float('inf')
            inventory_path=Path(os.environ.get('E36_ARCHIVE_INVENTORY_PATH',str(ROOT/'owner-svt-ssd-v1'/('closed-archive-inventory-'+uuid.uuid4().hex+'.json'))))
            with inventory_path.open('x') as stream:json.dump(closed,stream,separators=(',',':'));stream.flush();os.fsync(stream.fileno())
            sync_directory(inventory_path.parent);closed_proof=dict(path=str(inventory_path),sha256=sha(inventory_path));inventory_done=True
        while time.time()<1792274400 and not (ROOT/'owner-svt-ssd-v1/mirror-stop.json').exists():
            free=shutil.disk_usage(LOCAL).free
            eager=(ROOT/'owner-svt-ssd-v1/urgent-disk-policy.json').exists()
            if artifacts_only or eager or free<48*1024**3:
                scan=time.monotonic()>=next_scan
                published=set()
                for name in (native_runs(ROOT) if scan else []):
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
                from e36_archive_runs import native_cas_directories
                directories=native_cas_directories(ROOT) if scan else []
                for directory in directories:
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
                                if artifact.exists() and digest not in queue:
                                    queue[digest]=(artifact,path.name)
                                    heapq.heappush(priority,(-artifact.stat().st_size,digest))
                        seen.add(path.name)
                if scan:next_scan=float("inf") if artifacts_only else time.monotonic()+10
                # Reclaim space with fewer remote metadata operations. Tiny-first
                # directory ordering can spend minutes copying sub-MB objects.
                for record in dispatcher.cycle(queue,priority):
                    print(json.dumps(dict(mirrored_sha256=record['sha256'],bytes=record['bytes'],free_bytes=shutil.disk_usage(LOCAL).free)),flush=True)
                # Artifact eviction owns the urgent disk relief. Back up a bounded
                # number of trainer-needed receipts between reclamation batches.
                if not artifacts_only and eager and not priority and not dispatcher.pending and shutil.disk_usage(LOCAL).free>40*1024**3:
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
            pending_recovery=getattr(getattr(dispatcher,"dispatcher",dispatcher),"recovery",[])
            if artifacts_only and inventory_done and not queue and not priority and not dispatcher.pending and not pending_recovery:
                stop=ROOT/'owner-svt-ssd-v1/mirror-stop.json'
                with stop.open('x') as stream:json.dump(dict(reason='all native-published local bundles drained; metadata retained for separate canonical archive',inventory=closed_proof,utc=time.time(),fits=0),stream);stream.flush();os.fsync(stream.fileno())
                sync_directory(stop.parent);drain_complete=True
                break
            print(json.dumps(dict(utc=time.time(),free_bytes=shutil.disk_usage(LOCAL).free,queued_bundles=len(queue),seen_complete_views=len(seen),
                pending_transfers=len(dispatcher.pending),deferred_native_receipt_backups=len(metadata_pending))),flush=True)
            time.sleep(.1 if queue and (eager or shutil.disk_usage(LOCAL).free<48*1024**3) else 10)
    finally:
        try:dispatcher.drain(queue)
        finally:
            if interleave:dispatcher.close()
            if not dispatcher.pending:lock.close();MIRROR_LOCKS.remove(lock)
    if artifacts_only and not drain_complete:raise TimeoutError('archive-only run ended before its complete published-bundle drain; retained local data is not admitted')


def main():
    try:
        with ThreadPoolExecutor(max_workers=8,thread_name_prefix='e36-storage') as pool:run(pool)
    finally:
        try:
            if METADATA_CLIENT is not None:METADATA_CLIENT.close()
        finally:
            for lock in MIRROR_LOCKS:lock.close()
            MIRROR_LOCKS.clear()


if __name__=='__main__':main()
