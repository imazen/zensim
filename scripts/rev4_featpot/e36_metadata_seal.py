"""Prospective terminal interlock: sealed metadata forbids producer restarts."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time
from e36_artifact_location import durable_mkdir,sync_directory
from e36_store import sha


@contextmanager
def lifecycle(root,seal=False):
    root=Path(root);owner=root/'owner-svt-ssd-v1';durable_mkdir(owner)
    with (owner/'metadata-lifecycle.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX if seal else fcntl.LOCK_SH)
        marker=owner/'metadata-sealed-v1.json'
        if marker.exists() and not seal:raise ValueError('metadata sealed; producer/archive writer restart refused')
        yield marker


def freeze(root,drain):
    with lifecycle(root,seal=True) as marker:
        if marker.exists():
            old=json.loads(marker.read_text())
            if old['schema']!='e36-metadata-terminal-seal-v1' or old['raw_drain']!={'path':str(drain),'sha256':sha(drain)} or old['fits']!=0 or old['producer_restarts'] is not False:raise ValueError('metadata seal predecessor differs')
            return marker
        value=dict(schema='e36-metadata-terminal-seal-v1',raw_drain=dict(path=str(drain),sha256=sha(drain)),utc=time.time(),fits=0,producer_restarts=False)
        with marker.open('x') as stream:json.dump(value,stream,indent=2);stream.flush();__import__('os').fsync(stream.fileno())
        sync_directory(marker.parent);return marker
