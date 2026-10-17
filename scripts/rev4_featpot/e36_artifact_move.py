"""Bounded storage transport only; native zenfleet still owns every data job."""
import json
import os
from pathlib import Path
import threading
import time

from e36_store import Client,sha
from e36_artifact_location import access,sync_directory


class Mover:
    def __init__(self,root,storage):
        self.root=Path(root);self.storage=storage;self.clients=threading.local()
        self.journal_lock=threading.Lock()

    def move(self,digest,artifact,view_sha):
        artifact=Path(artifact)
        with access(self.root,digest,exclusive=True):
            if not artifact.exists():return None
            if sha(artifact)!=digest:raise ValueError('local artifact hash changed')
            if not hasattr(self.clients,'client'):
                self.clients.client=Client(self.storage['command'],self.storage['program_sha256'])
            client=self.clients.client;ack=client.put(artifact);verified=client.verify(digest)
            if verified['bytes']!=artifact.stat().st_size or ack['bytes']!=verified['bytes']:
                raise ValueError('mirror size/hash verification failed')
            record=dict(utc=time.time(),sha256=digest,bytes=verified['bytes'],completed_view_sha256=view_sha,
                        local_allocated_bytes=artifact.stat().st_blocks*512,local_path=str(artifact),
                        canonical_host=self.storage['host'],canonical_root=self.storage['root'],
                        relative_path=verified['relative_path'],program_sha256=verified['program_sha256'],
                        sha_verified=True,action='canonical-mirror-verified-before-local-unlink',fits=0)
            journal=self.root/'owner-svt-ssd-v1/artifact-relocations.jsonl'
            with self.journal_lock:
                with journal.open('a') as stream:
                    stream.write(json.dumps(record,separators=(',',':'))+'\n');stream.flush();os.fsync(stream.fileno())
                sync_directory(journal.parent)
                artifact.unlink();sync_directory(artifact.parent)
            return record
