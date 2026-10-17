"""Bound the separate, explicitly authorized cache archive; no data-job claims."""
import ctypes
import ctypes.util
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from e36_artifact_location import sync_directory

LIBRARY=ctypes.CDLL(ctypes.util.find_library('zstd'))
LIBRARY.ZSTD_compressBound.argtypes=[ctypes.c_size_t];LIBRARY.ZSTD_compressBound.restype=ctypes.c_size_t


def compression_bound(size):
    # Include additional frame-profile slack beyond the library bound.
    return int(LIBRARY.ZSTD_compressBound(size))+65536


class CachePolicy:
    def __init__(self,root,storage):
        self.root=Path(root);self.storage=storage;self.owner=self.root/'owner-svt-ssd-v1'
        self.locations={};self.excluded=set()
        journal=self.owner/'artifact-relocations.jsonl'
        from e36_artifact_location import relocation_records
        for row in relocation_records(self.root):
            previous=self.locations.setdefault(row['sha256'],row)
            if previous['canonical_root']!=row['canonical_root']:raise ValueError('same bundle assigned to two archive roots')
        probe=self.owner/'zstd-map-probe-v1/receipt.json'
        if probe.exists():self.excluded.add(json.loads(probe.read_text())['bundle_sha256'])
        if storage.get('authorization')!='coordinator-2026-10-17-cache250GB' or storage['cache_budget_bytes']!=250000000000 or storage['cache_reserve_bytes']!=180*1024**3:
            raise ValueError('exact cache grant required')
        self.claims=self.owner/'cache-reservations-v1.jsonl';self.reserved={};self.epoch=None
        if self.claims.exists():
            for line in self.claims.read_text().splitlines():
                row=json.loads(line)
                if row['sha256'] in self.reserved and self.reserved[row['sha256']]!=row['reserved_bytes']:raise ValueError('conflicting cache reservation')
                self.reserved[row['sha256']]=row['reserved_bytes']
        self.total=sum(self.reserved.values())
        if self.total>storage['cache_budget_bytes']:raise ValueError('cache reservations exceed grant')
        self.budget_left=storage['cache_budget_bytes']-self.total

    def append(self,path,value):
        with path.open('a') as stream:stream.write(json.dumps(value,separators=(',',':'))+'\n');stream.flush();os.fsync(stream.fileno())
        sync_directory(path.parent)

    def before_batch(self):
        if (self.owner/'CACHE180_STOP.json').exists() or (self.owner/'CACHE_ASSIGNMENT_AUDIT_PAUSE.json').exists() or self.budget_left<=0:return False
        control=Path.home()/'tmp/e36-cache-capacity-%C.sock'
        command=['ssh','-T','-oBatchMode=yes','-oConnectTimeout=10',
            '-oControlMaster=auto','-oControlPersist=30','-oControlPath='+str(control),'tower']
        # Keep the requested human-readable observation plus exact byte count.
        human=subprocess.check_output(command+['df -h /mnt/cache'],text=True)
        exact=subprocess.check_output(command+['df -B1 --output=avail /mnt/cache'],text=True)
        free=int(exact.splitlines()[-1]);self.epoch=dict(id=uuid.uuid4().hex,utc=time.time(),free_bytes=free,
            command='ssh tower df -h /mnt/cache',argv=command+['df -h /mnt/cache'],
            df_h=human,df_bytes=exact,batch_reserved_bytes=0,fits=0)
        self.append(self.owner/'cache-space-checks-v1.jsonl',self.epoch)
        if free<self.storage['cache_reserve_bytes']:
            self.append(self.owner/'CACHE180_STOP.json',dict(utc=time.time(),free_bytes=free,reason='cache grant floor reached; no silent resume',fits=0))
            return False
        return True

    def claim(self,digest,raw_bytes):
        if (self.owner/'CACHE180_STOP.json').exists():return False
        if self.epoch is None:raise ValueError('cache batch lacks its mandatory fresh space observation')
        reserve=self.reserved.get(digest,compression_bound(raw_bytes))
        if ((digest not in self.reserved and self.total+reserve>self.storage['cache_budget_bytes']) or self.epoch['free_bytes']-self.epoch['batch_reserved_bytes']-reserve<self.storage['cache_reserve_bytes']):
            return False
        record=dict(sha256=digest,raw_bytes=raw_bytes,reserved_bytes=reserve,batch_id=self.epoch['id'],
            canonical_root=self.storage['root'],utc=time.time(),fits=0)
        if digest not in self.reserved:
            self.append(self.claims,record)
            self.reserved[digest]=reserve;self.total+=reserve;self.budget_left=self.storage['cache_budget_bytes']-self.total
        self.epoch['batch_reserved_bytes']+=reserve
        return True
