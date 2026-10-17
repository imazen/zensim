"""Cache-first sequential storage batches, preserving native bundle identities."""
import gzip
import hashlib
import heapq
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import wait,FIRST_COMPLETED

from e36_archive_assignments import Assignments
from e36_artifact_location import sync_directory,durable_mkdir
from e36_cache_archive import CachePolicy
from e36_pack_store import build,sha


def append_batch(path,rows):
    with Path(path).open('a') as stream:
        for row in rows:stream.write(json.dumps(row,separators=(',',':'))+'\n')
        stream.flush();os.fsync(stream.fileno())
    sync_directory(Path(path).parent)


def atomic_json(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    with temporary.open('x') as stream:json.dump(value,stream,separators=(',',':'));stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    except FileExistsError:
        if json.loads(path.read_text())!=value:raise ValueError('immutable archive intent differs')
    sync_directory(path.parent);temporary.unlink()


def publish_segment(owner,rows):
    directory=Path(owner)/'artifact-relocation-segments-v1';durable_mkdir(directory)
    data=''.join(json.dumps(row,separators=(',',':'))+'\n' for row in rows).encode()
    path=directory/(hashlib.sha256(data).hexdigest()+'.jsonl');temporary=directory/(uuid.uuid4().hex+'.partial')
    with temporary.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    try:os.link(temporary,path)
    except FileExistsError:
        if path.read_bytes()!=data:raise ValueError('relocation segment collision')
    sync_directory(directory);temporary.unlink();return path


def remote(storage,verifier,*args):
    return ['ssh','-T','-oBatchMode=yes','-oConnectTimeout=10','tower',
            'docker exec -i '+shlex.quote(storage['container_id'])+' python3 '+shlex.quote('/data/program/'+verifier+'.py')+' '+shlex.join(args)]


def readback(storage,record):
    verifier=record['pack_verifier_sha256'];relative=record['relative_path']
    if relative!='packs/'+record['pack_id']+'.tar':raise ValueError('pack relative path differs')
    actual=json.loads(subprocess.check_output(remote(storage,verifier,'verify','/data/'+relative)))
    if (actual['verifier_sha256']!=verifier or actual['stored_sha256']!=record['stored_sha256']
        or actual['stored_bytes']!=record['stored_bytes'] or actual['members'].get(record['sha256'])!=record['bytes']):raise ValueError('actual packed native bytes differ')
    return dict(ok=True,sha256=record['sha256'],bytes=record['bytes'],relative_path=relative,encoding='tar-pack-v1',
                stored_sha256=actual['stored_sha256'],stored_bytes=actual['stored_bytes'],program_sha256=storage['program_sha256'])


class Dispatcher:
    def __init__(self,root,pool,disk):
        self.root=Path(root);self.owner=self.root/'owner-svt-ssd-v1';self.pool=pool;self.pending={}
        self.assignments=Assignments(root);self.policy=CachePolicy(root,json.loads((self.root/'storage-cache-v1.json').read_text()))
        census=json.loads((self.owner/'archive-assignment-census-v1/receipt.json').read_text())
        if not census['cache_array_duplicate_hashes']==[]:raise ValueError('archive duplicate audit required')
        self.cache=self.policy.storage;self.disk=disk
        self.verifier=sha(Path(__file__).with_name('e36_pack_store.py'))
        config=json.loads((self.owner/'pack-storage-admission-v1.json').read_text())
        directory_proof=json.loads((self.owner/'pack-directory-durability-v1.json').read_text())
        if directory_proof.get('admitted') is not True:raise ValueError('actual durable remote pack directories required')
        if config['verifier_sha256']!=self.verifier or config.get('admitted') is not True:raise ValueError('actual map pack readback admission required')
        self.intents=self.owner/'pack-intents-v1';durable_mkdir(self.intents)
        self.journal_lock=threading.Lock();self.cache_active=True
        self.used=self.policy.total+config['probe_reserved_bytes'];self.recovery=[]
        self.locations=self.policy.locations
        for path in sorted(self.intents.glob('*.json')):
            doc=json.loads(path.read_text())
            if doc['storage']['root']==self.cache['root']:self.used+=doc['reserved_bytes']
            if not (self.intents/(doc['pack_id']+'.complete')).exists():self.recovery.append(doc)
        if self.used>250000000000:raise ValueError('cache total allowance exceeded')
        self.recovery_hashes={i['sha256'] for doc in self.recovery for i in doc['items']}

    def fresh_cache(self,size,recovery=False):
        # The old pause protected an incomplete audit. Its completed receipt is
        # required above; this replacement does not silently reopen the old writer.
        control=Path.home()/'tmp/e36-cache-capacity-%C.sock'
        command=['ssh','-T','-oBatchMode=yes','-oConnectTimeout=10','-oControlMaster=auto','-oControlPersist=30','-oControlPath='+str(control),'tower']
        human=subprocess.check_output(command+['df -h /mnt/cache'],text=True)
        exact=subprocess.check_output(command+['df -B1 --output=avail /mnt/cache'],text=True)
        free=int(exact.splitlines()[-1]);append_batch(self.owner/'pack-cache-space-v1.jsonl',[dict(utc=time.time(),command='ssh tower df -h /mnt/cache',df_h=human,free_bytes=free,reserved_before=self.used,next_pack_bound=size,fits=0)])
        if (self.owner/'CACHE180_STOP.json').exists() or free-size-sum(doc['reserved_bytes'] for doc in self.pending.values() if doc['storage']['root']==self.cache['root'])<180*1024**3 or self.used+(0 if recovery else size)>250000000000:
            if free<180*1024**3 and not (self.owner/'CACHE180_STOP.json').exists():atomic_json(self.owner/'CACHE180_STOP.json',dict(utc=time.time(),free_bytes=free,fits=0,reason='actual below-floor observation; sticky across restart'))
            self.cache_active=False;return False
        return True


    def validate_intent(self,doc):
        import pyarrow.parquet as pq
        storage=doc['storage']
        expected=self.cache if storage['root']==self.cache['root'] else self.disk
        if any(storage[k]!=expected[k] for k in ['root','host','program_sha256','container_id']):raise ValueError('recovered pack owner changed')
        items=doc['items']
        if not items or len({i['sha256'] for i in items})!=len(items):raise ValueError('recovered pack has duplicate/empty members')
        key=hashlib.sha256(json.dumps([(i['sha256'],i['bytes']) for i in items],separators=(',',':')).encode()).hexdigest()
        size=sum(512+((i['bytes']+511)//512)*512 for i in items);reserve=((size+1024+10239)//10240)*10240
        if key!=doc['pack_id'] or reserve!=doc['reserved_bytes']:raise ValueError('recovered pack identity/size differs')
        published=set()
        for name in ['fallback-v6','svt-additions-v1']:
            for path in (self.root/'runs'/name).glob('ledger*.parquet'):
                for batch in pq.ParquetFile(path).iter_batches(columns=['status','attempts','output_sha']):published.update(r['output_sha'] for r in batch.to_pylist() if r['status']=='done' and r['attempts']==1)
        views={}
        for item in items:
            h=item['sha256'];view=item['view_sha256'];local=self.root/'artifacts-local-v1/blobs'/h[:2]/(h+'.tar')
            if str(local)!=item['local_path'] or view not in published:raise ValueError('recovered pack lost native eligibility')
            if view not in views:
                path=next((self.root/'runs'/name/'blobs'/view for name in ['fallback-v6','svt-additions-v1'] if (self.root/'runs'/name/'blobs'/view).exists()),None)
                if path is None or sha(path)!=view:raise ValueError('recovered native CAS differs')
                document=json.loads(gzip.decompress(path.read_bytes()))
                if document.get('complete') is not True or document.get('schema')!='e36-native-ladder-receipt-v1':raise ValueError('recovered native view incomplete')
                views[view]={r['bundle']['sha256']:r['bundle'] for r in document['rows']}
            bundle=views[view].get(h)
            if bundle is None or bundle['host']!='dev' or bundle['root']!=str(self.root/'artifacts-local-v1') or bundle['bytes']!=item['bytes']:raise ValueError('recovered native bundle binding differs')

    def assigned_batch(self,doc):
        self.assignments.seed_pack([dict(sha256=i['sha256'],canonical_host=doc['storage']['host'],canonical_root=doc['storage']['root'],program_sha256=doc['storage']['program_sha256'],pack_id=doc['pack_id'],fits=0) for i in doc['items']])

    def exists(self,doc):
        if doc['storage']['root']==self.cache['root']:return (Path('/mnt/tower/output/zensim/e36-bulk-2026-10-17-cache')/'packs'/(doc['pack_id']+'.tar')).exists()
        code='from pathlib import Path; import sys,json; print(json.dumps(Path(sys.argv[1]).is_file()))'
        command=['ssh','-T','-oBatchMode=yes','tower',shlex.join(['docker','exec',doc['storage']['container_id'],'python3','-c',code,'/data/packs/'+doc['pack_id']+'.tar'])]
        return json.loads(subprocess.check_output(command))

    def pack(self,doc):
        start=time.monotonic();storage=doc['storage'];pack_id=doc['pack_id'];items=doc['items'];relative='packs/'+pack_id+'.tar'
        if self.exists(doc):
            actual=json.loads(subprocess.check_output(remote(storage,self.verifier,'verify','/data/'+relative)))
        elif storage['root']==self.cache['root']:
            base=Path('/mnt/tower/output/zensim/e36-bulk-2026-10-17-cache')
            final=base/relative;temporary=base/'.pack-incoming'/(pack_id+'-'+uuid.uuid4().hex+'.partial')
            if final.exists():
                actual=json.loads(subprocess.check_output(remote(storage,self.verifier,'verify','/data/'+relative)))
            else:
                with temporary.open('xb',buffering=1024**2) as target:
                    built=build(target,items);target.flush();os.fsync(target.fileno())
                if built['stored_bytes']>doc['reserved_bytes']:raise ValueError('pack exceeded reserved cache bytes')
                try:os.link(temporary,final)
                except FileExistsError:
                    if sha(final)!=built['stored_sha256']:raise ValueError('cache pack collision')
                sync_directory(final.parent);temporary.unlink()
                actual=json.loads(subprocess.check_output(remote(storage,self.verifier,'verify','/data/'+relative)))
                if any(actual[k]!=built[k] for k in built):raise ValueError('cache pack actual readback differs from written stream')
        else:
            # Array is used only once the cache allowance or floor is exhausted.
            command=remote(storage,self.verifier,'receive','/data','--pack-id',pack_id,'--size',str(doc['reserved_bytes']),'--sha256','-')
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None)
            error=None
            try:built=build(process.stdin,items)
            except BaseException as failure:error=failure
            finally:
                try:process.stdin.close()
                except OSError:pass
                process.stdin=None
                while True:
                    try:answer,_=process.communicate(timeout=30);break
                    except subprocess.TimeoutExpired:
                        try:
                            from e36_generation_resume import hold_admission
                            scope=Path('/sys/fs/cgroup'+Path('/proc/self/cgroup').read_text().strip().split('::')[1])
                            hold_admission(self.root,scope,self.root/'bin/zenfleet-worker-fair-v7','array pack transfer still draining; owned handle retained')
                        except BaseException as hold_error:
                            try:print('E36 pack admission hold failed; retaining and joining the same receive process: '+repr(hold_error),file=sys.stderr,flush=True)
                            except BaseException:pass
            if error is not None:raise error
            if process.returncode!=0:raise RuntimeError('array pack transfer failed; originals retained')
            actual=json.loads(answer)
            if any(actual[k]!=built[k] for k in built):raise ValueError('array pack actual readback differs')
        if actual['verifier_sha256']!=self.verifier or actual['members']!={i['sha256']:i['bytes'] for i in items}:raise ValueError('packed native membership differs')
        rows=[]
        for item in items:
            rows.append(dict(utc=time.time(),sha256=item['sha256'],bytes=item['bytes'],completed_view_sha256=item['view_sha256'],
                local_allocated_bytes=(Path(item['local_path']).stat().st_blocks*512 if Path(item['local_path']).exists() else 0),local_path=item['local_path'],canonical_host=storage['host'],canonical_root=storage['root'],program_sha256=storage['program_sha256'],
                relative_path=relative,pack_id=pack_id,pack_verifier_sha256=self.verifier,stored_sha256=actual['stored_sha256'],stored_bytes=actual['stored_bytes'],
                encoding='tar-pack-v1',sha_verified=True,action='pack-verified-journal-batch-before-local-unlink',fits=0))
        # One durable journal batch before any local object is removed. No
        # per-object flock, fsync, or destination upload exists on this path.
        with self.journal_lock:
            publish_segment(self.owner,rows)
            for item in items:
                try:Path(item['local_path']).unlink()
                except FileNotFoundError:pass
            marker=self.intents/(pack_id+'.complete')
            if not marker.exists():
                atomic_json(marker,dict(pack_id=pack_id,utc=time.time(),stored_sha256=actual['stored_sha256'],fits=0))
            append_batch(self.owner/'pack-rates-v1.jsonl',[dict(utc=time.time(),pack_id=pack_id,backend='cache' if storage['root']==self.cache['root'] else 'disk1',native_bundles=len(items),raw_bytes=sum(i['bytes'] for i in items),stored_bytes=actual['stored_bytes'],elapsed_seconds=time.monotonic()-start,fits=0)])
        return rows

    def cycle(self,queue,priority):
        completed=[]
        for future,doc in list(self.pending.items()):
            if not future.done():continue
            rows=future.result();completed.extend(rows);del self.pending[future]
            for item in doc['items']:queue.pop(item['sha256'],None)
            self.locations.update({row['sha256']:row for row in rows})
        if (self.owner/'PACK_REPAIR_HOLD.json').exists() or len(self.pending)>=2:return completed
        if self.recovery:
            doc=self.recovery[0];self.validate_intent(doc);self.assigned_batch(doc)
            if doc['storage']['root']==self.cache['root'] and not self.exists(doc) and not self.fresh_cache(doc['reserved_bytes'],recovery=True):return completed
            self.recovery.pop(0);self.pending[self.pool.submit(self.pack,doc)]=doc;return completed
        selected=[];deferred=[];size=0
        # Read the assignment index once per group, never a flock per object.
        with self.assignments.lock,self.assignments.path.open('a+') as stream:self.assignments.refresh(stream)
        while priority and size<256*1024**2 and len(selected)<32768:
            _,h=heapq.heappop(priority)
            if h not in queue:continue
            artifact,view=queue[h]
            if not artifact.exists():queue.pop(h,None);continue
            assigned=self.assignments.rows.get(h)
            if assigned is not None or h in self.recovery_hashes:
                # Historical objects stay at their first physical owner. They
                # are reconciled separately after fresh packs, never copied.
                deferred.append((-artifact.stat().st_size,h));continue
            n=artifact.stat().st_size;selected.append(dict(sha256=h,bytes=n,local_path=str(artifact),view_sha256=view));size+=512+((n+511)//512)*512
        for value in deferred:heapq.heappush(priority,value)
        if not selected:
            if self.pending:wait(self.pending,timeout=.1,return_when=FIRST_COMPLETED)
            return completed
        reserve=((size+1024+10239)//10240)*10240
        if self.cache_active:self.fresh_cache(reserve)
        storage=self.cache if self.cache_active else self.disk
        pack_id=hashlib.sha256(json.dumps([(i['sha256'],i['bytes']) for i in selected],separators=(',',':')).encode()).hexdigest()
        doc=dict(pack_id=pack_id,storage=storage,items=selected,reserved_bytes=reserve,utc=time.time(),fits=0)
        path=self.intents/(pack_id+'.json')
        atomic_json(path,doc)
        self.assigned_batch(doc)
        if self.cache_active:self.used+=reserve
        self.pending[self.pool.submit(self.pack,doc)]=doc
        return completed

    def drain(self,queue):
        for future,doc in list(self.pending.items()):
            future.result();del self.pending[future]
            for item in doc['items']:queue.pop(item['sha256'],None)
