"""Independent bounded cache/array transport batches, sharing native eligibility."""
import heapq
import json
from pathlib import Path
import time
from concurrent.futures import wait,FIRST_COMPLETED

from e36_artifact_move import Mover
from e36_cache_archive import CachePolicy
from e36_artifact_location import access,record_owner,sync_directory
from e36_archive_store import ArchiveClient
from e36_store import Client,sha
from e36_archive_assignments import Assignments


def reconcile(root,row,artifact):
    """A prior fsynced relocation stays its owner; no duplicate upload."""
    legacy=json.loads((root/'storage-runtime.json').read_text());storage=record_owner(root,row,legacy)
    cls=ArchiveClient if storage.get('encoding')=='zstd' else Client
    client=cls(storage['command'],storage['program_sha256'])
    try:
        with access(root,row['sha256'],exclusive=True):
            if not artifact.exists():return None
            result=client.verify(row['sha256'])
            if result['bytes']!=row['bytes'] or sha(artifact)!=row['sha256']:raise ValueError('existing archival owner differs')
            if storage.get('encoding')=='zstd' and any(result[k]!=row[k] for k in ['stored_sha256','stored_bytes']):raise ValueError('existing archive stored bytes differ')
            artifact.unlink();sync_directory(artifact.parent)
            return row
    finally:client.close()


class Dispatcher:
    def __init__(self,root,pool,disk):
        self.root=Path(root);self.pool=pool;self.pending={};self.disk=Mover(root,disk)
        self.assignments=Assignments(root);self.disk.assignments=self.assignments
        self.movers={disk['root']:self.disk}
        path=self.root/'storage-cache-v1.json'
        self.cache=None;self.policy=None;self.cache_active=False
        if path.exists():
            storage=json.loads(path.read_text());self.cache=Mover(root,storage)
            self.cache.journal_lock=self.disk.journal_lock
            self.cache.assignments=self.assignments;self.movers[storage['root']]=self.cache
            self.policy=CachePolicy(root,storage);self.cache_active=True

    def get_mover(self,assignment):
        key=assignment['canonical_root']
        if key not in self.movers:
            candidates=[self.root/name for name in ['storage-runtime.json','storage-archive-v2.json','storage-cache-v1.json']]
            storage=next(json.loads(path.read_text()) for path in candidates if path.exists() and json.loads(path.read_text())['root']==key)
            value=Mover(self.root,storage);value.assignments=self.assignments;value.journal_lock=self.disk.journal_lock;self.movers[key]=value
        return self.movers[key]

    def cycle(self,queue,priority):
        completed=[]
        for future,(digest,backend) in list(self.pending.items()):
            if not future.done():continue
            result=future.result();del self.pending[future];del queue[digest]
            if result is not None and self.policy is not None:self.policy.locations[digest]=result
            if result is not None:completed.append(result)
        for backend in ['cache','disk']:
            if backend=='cache' and not self.cache_active:continue
            active=sum(value[1]==backend for value in self.pending.values())
            if backend=='cache' and active:continue
            if backend=='cache':
                if not self.policy.before_batch():self.cache_active=False;continue
                limit=4;mover=self.cache
            else:limit=(4 if self.cache_active else 8)-active;mover=self.disk
            selected=[];deferred=[]
            while priority and len(selected)<limit and len(self.pending)+len(selected)<8:
                _,digest=heapq.heappop(priority);artifact,view=queue[digest]
                if not artifact.exists():del queue[digest];continue
                previous=self.policy.locations.get(digest) if self.policy is not None else None
                if previous is not None:
                    selected.append((digest,artifact,view,previous));continue
                if backend=='cache' and digest in self.policy.excluded:
                    deferred.append((-artifact.stat().st_size,digest));continue
                assigned=self.assignments.get(digest)
                if backend=='cache' and assigned is not None and assigned['canonical_root']!=self.cache.storage['root']:
                    deferred.append((-artifact.stat().st_size,digest));continue
                if backend=='cache' and not self.policy.claim(digest,artifact.stat().st_size):
                    heapq.heappush(priority,(-artifact.stat().st_size,digest))
                    self.cache_active=False;break
                if backend=='disk' and self.policy is not None and digest in self.policy.reserved:
                    raise ValueError('unfinished cache assignment must not be uploaded to a second root')
                selected.append((digest,artifact,view,None))
            for item in deferred:heapq.heappush(priority,item)
            for digest,artifact,view,previous in selected:
                if previous is not None:future=self.pool.submit(reconcile,self.root,previous,artifact)
                else:
                    assigned=self.assignments.get(digest)
                    target=self.get_mover(assigned) if assigned is not None else mover
                    self.assignments.ensure(digest,target.storage)
                    future=self.pool.submit(target.move,digest,artifact,view)
                self.pending[future]=(digest,backend)
        if self.pending:wait(self.pending,timeout=.1,return_when=FIRST_COMPLETED)
        return completed

    def drain(self,queue):
        for future,(digest,_) in list(self.pending.items()):
            future.result();del self.pending[future];del queue[digest]
