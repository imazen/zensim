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


def snapshot(root):
    from e36_closed_archive import closed_generation
    from e36_archive_terminal import validate,checked
    root=Path(root);terminal,ledgers=closed_generation(root);drain,value=validate(root)
    stop=json.loads(checked(value['stop']).read_text());inventory=json.loads(checked(stop['inventory']).read_text())
    current={str(path):sha(path) for path in ledgers};recorded={row['path']:row['sha256'] for row in inventory['ledgers']}
    terminal_pin=dict(path=str(terminal),sha256=sha(terminal))
    if current!=recorded or inventory['terminal']!=terminal_pin:raise ValueError('raw drain is stale against current closed native ledgers')
    optional=root/'owner-svt-ssd-v1/native-additions-final-v1.json'
    return dict(raw_drain=dict(path=str(drain),sha256=sha(drain)),native_terminal=terminal_pin,native_ledgers=[dict(path=path,sha256=value) for path,value in sorted(current.items())],inventory=stop['inventory'],optional_terminal=dict(path=str(optional),sha256=sha(optional)) if optional.exists() else None)


def freeze(root,drain=None):
    with lifecycle(root,seal=True) as marker:
        evidence=snapshot(root)
        if drain is not None and evidence['raw_drain']!={'path':str(drain),'sha256':sha(drain)}:raise ValueError('metadata seal raw drain differs')
        if marker.exists():
            old=json.loads(marker.read_text())
            if old['schema']!='e36-metadata-terminal-seal-v1' or any(old[key]!=value for key,value in evidence.items()) or old['fits']!=0 or old['producer_restarts'] is not False:raise ValueError('metadata seal predecessor differs')
            return marker
        value=dict(schema='e36-metadata-terminal-seal-v1',**evidence,utc=time.time(),fits=0,producer_restarts=False)
        with marker.open('x') as stream:json.dump(value,stream,indent=2);stream.flush();__import__('os').fsync(stream.fileno())
        sync_directory(marker.parent);return marker


def validate(root,pin):
    with lifecycle(root,seal=True) as marker:
        if str(marker)!=pin['path'] or sha(marker)!=pin['sha256']:raise ValueError('metadata lifecycle seal changed')
        document=json.loads(marker.read_text());current=snapshot(root)
        if document['producer_restarts'] is not False or any(document[key]!=value for key,value in current.items()):raise ValueError('metadata sealed native/inventory evidence changed')
        return document
