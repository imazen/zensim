"""Durable archive destination intent, shared by every storage transfer."""
import hashlib
import uuid
import fcntl
import json
import os
from pathlib import Path
import threading
import time

from e36_artifact_location import sync_directory,durable_mkdir


class Assignments:
    def __init__(self,root):
        self.path=Path(root)/'owner-svt-ssd-v1/archive-assignments-v1.jsonl'
        self.lock=threading.RLock();self.rows={};self.offset=0;self.segments=set()
        with self.lock,self.path.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);self.refresh(stream)

    def merge(self,row):
        previous=self.rows.get(row['sha256'])
        if previous is not None and any(previous[k]!=row[k] for k in ['canonical_host','canonical_root','program_sha256']):raise ValueError('immutable archive destination changed')
        self.rows[row['sha256']]={**(previous or {}),**row}

    def refresh(self,stream):
        if os.fstat(stream.fileno()).st_size<self.offset:raise ValueError('archive assignment history shrank')
        stream.seek(self.offset)
        for line in stream:
            if not line.endswith('\n'):raise ValueError('incomplete archive assignment record')
            self.merge(json.loads(line))
        self.offset=stream.tell()
        directory=self.path.parent/'archive-assignment-segments-v1'
        for path in sorted(directory.glob('*.jsonl')):
            if path.name in self.segments:continue
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=path.stem or not data.endswith(b'\n'):raise ValueError('committed assignment segment changed')
            for line in data.splitlines():self.merge(json.loads(line))
            self.segments.add(path.name)

    def seed_pack(self,rows):
        """One atomic committed segment for an entire pack's owner intent."""
        with self.lock,self.path.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);self.refresh(stream)
            for row in rows:
                prior=self.rows.get(row['sha256'])
                if prior is not None and any(prior[k]!=row[k] for k in ['canonical_host','canonical_root','program_sha256']):raise ValueError('immutable archive destination changed')
                if prior is not None and prior.get('pack_id') not in [None,row['pack_id']]:raise ValueError('bundle belongs to a different pack')
            data=''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows).encode()
            directory=self.path.parent/'archive-assignment-segments-v1';durable_mkdir(directory)
            target=directory/(hashlib.sha256(data).hexdigest()+'.jsonl');temporary=directory/(uuid.uuid4().hex+'.partial')
            with temporary.open('xb') as output:output.write(data);output.flush();os.fsync(output.fileno())
            try:os.link(temporary,target)
            except FileExistsError:
                if target.read_bytes()!=data:raise ValueError('assignment segment collision')
            sync_directory(directory);temporary.unlink()
            for row in rows:self.merge(row)
            self.segments.add(target.name)

    def seed(self,rows):
        with self.lock,self.path.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);self.refresh(stream);pending=[]
            for row in rows:
                previous=self.rows.get(row['sha256'])
                if previous is not None and any(previous[k]!=row[k] for k in ['canonical_host','canonical_root','program_sha256']):raise ValueError('immutable archive destination changed')
                merged={**(previous or {}),**row}
                if previous!=merged:pending.append(merged)
            stream.seek(0,2)
            for row in pending:stream.write(json.dumps(row,separators=(',',':'))+'\n')
            stream.flush();os.fsync(stream.fileno());self.offset=stream.tell();sync_directory(self.path.parent)
            for row in pending:self.merge(row)

    def get(self,digest):
        with self.lock,self.path.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_SH);self.refresh(stream);return self.rows.get(digest)

    def ensure(self,digest,storage):
        row=dict(sha256=digest,canonical_host=storage['host'],canonical_root=storage['root'],
                 program_sha256=storage['program_sha256'],utc=time.time(),fits=0)
        with self.lock,self.path.open('a+') as stream:
            fcntl.flock(stream,fcntl.LOCK_EX);self.refresh(stream)
            prior=self.rows.get(digest)
            if prior is not None:
                if any(prior[k]!=row[k] for k in ['canonical_host','canonical_root','program_sha256']):raise ValueError('refusing a second archive destination')
                os.fsync(stream.fileno())
                return prior
            stream.seek(0,2);stream.write(json.dumps(row,separators=(',',':'))+'\n');stream.flush();os.fsync(stream.fileno())
            self.offset=stream.tell();sync_directory(self.path.parent);self.merge(row);return row
