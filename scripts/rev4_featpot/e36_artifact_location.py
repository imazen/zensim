"""Resolve immutable E36 bundle receipts through verified archival relocations."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path

from e36_store import digest, sha


def sync_directory(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def durable_mkdir(path):
    path=Path(path)
    if path.is_dir():return
    durable_mkdir(path.parent)
    try:path.mkdir()
    except FileExistsError:
        if not path.is_dir():raise
    sync_directory(path)
    sync_directory(path.parent)


@contextmanager
def access(root,hash_value,exclusive=False):
    digest(hash_value)
    directory=Path(root)/'artifact-access-locks'
    durable_mkdir(directory)
    with (directory/(hash_value[:2]+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
        yield


def resolve(bundle,root,remote_verify,storage,journal_pin=None):
    """Verify actual bytes at the recorded owner or a hash-bound relocation."""
    root=Path(root);hash_value=digest(bundle['sha256'])
    expected_relative='blobs/'+hash_value[:2]+'/'+hash_value+'.tar'
    if bundle['relative_path']!=expected_relative:
        raise ValueError('artifact path is not its canonical content address')
    if bundle['host']==storage['host'] and bundle['root']==storage['root']:
        answer=remote_verify(hash_value)
        if answer['bytes']!=bundle['bytes']:raise ValueError('legacy bundle size changed')
        return dict(host=storage['host'],root=storage['root'],**answer)
    local=root/'artifacts-local-v1'
    if bundle['host']!='dev' or bundle['root']!=str(local):
        raise ValueError('unknown artifact owner')
    with access(root,hash_value):
        path=local/expected_relative
        if path.exists():
            if path.stat().st_size!=bundle['bytes'] or sha(path)!=hash_value:
                raise ValueError('local artifact bytes changed')
            return dict(ok=True,host='dev',root=str(local),sha256=hash_value,
                        bytes=bundle['bytes'],relative_path=expected_relative)
        journal=root/'owner-svt-ssd-v1/artifact-relocations.jsonl'
        if journal_pin is not None:
            if Path(journal_pin['path'])!=journal or sha(journal)!=journal_pin['sha256']:
                raise ValueError('frozen artifact relocation journal changed')
        found=None
        with journal.open() as stream:
            for line in stream:
                record=json.loads(line)
                if record['sha256']!=hash_value:continue
                if (record['sha_verified'] is not True or record['canonical_host']!=storage['host']
                    or record['canonical_root']!=storage['root'] or record['relative_path']!=expected_relative
                    or record['program_sha256']!=storage['program_sha256'] or record['bytes']!=bundle['bytes']):
                    raise ValueError('artifact relocation owner/hash/size differs')
                found=record
        if found is None:raise ValueError('missing local artifact has no verified relocation')
        actual=remote_verify(hash_value)
        if actual['bytes']!=bundle['bytes']:raise ValueError('relocated artifact byte count changed')
        return dict(host=storage['host'],root=storage['root'],**actual)


def freeze(root,out):
    """Bind final relocation history and replay native eligibility plus readback."""
    import gzip
    import random
    import shutil
    import time
    import pyarrow.parquet as pq
    from e36_store import Client
    root=Path(root);out=Path(out)
    stop=root/'owner-svt-ssd-v1/mirror-stop.json'
    if not stop.exists():raise ValueError('stop archival writer before freezing provenance')
    for pin in (root/'owner-svt-ssd-v1').glob('mirror*pid.json'):
        pid=json.loads(pin.read_text())['pid'];p=Path('/proc')/str(pid)
        if p.exists() and b'e36_artifact_mirror' in (p/'cmdline').read_bytes():
            raise ValueError('artifact mirror still active')
    journal=root/'owner-svt-ssd-v1/artifact-relocations.jsonl'
    if not journal.exists():
        with journal.open('x') as stream:stream.flush();os.fsync(stream.fileno())
        sync_directory(journal.parent)
    journal_sha=sha(journal)
    rows=[json.loads(line) for line in journal.read_text().splitlines()]
    published=set();ledger_pins=[]
    for name in ['fallback-v6','svt-additions-v1']:
        for p in (root/'runs'/name).glob('ledger*.parquet'):
            for batch in pq.ParquetFile(p).iter_batches(columns=['status','attempts','output_sha']):
                published.update(r['output_sha'] for r in batch.to_pylist() if r['status']=='done' and r['attempts']==1 and r['output_sha'])
            ledger_pins.append(dict(path=str(p),sha256=sha(p)))
    storage_path=root/'storage-runtime.json';storage=json.loads(storage_path.read_text())
    unique={};by_view={}
    for row in rows:
        h=digest(row['sha256']);relative='blobs/'+h[:2]+'/'+h+'.tar'
        if (row['sha_verified'] is not True or row['canonical_host']!=storage['host'] or row['canonical_root']!=storage['root'] or row['relative_path']!=relative or row['program_sha256']!=storage['program_sha256'] or row['completed_view_sha256'] not in published):
            raise ValueError('archived artifact lacks exact store/native DONE eligibility')
        by_view.setdefault(row['completed_view_sha256'],[]).append(row)
        prior=unique.setdefault(h,row)
        if any(prior[k]!=row[k] for k in ['bytes','canonical_host','canonical_root','relative_path','program_sha256']):
            raise ValueError('conflicting artifact relocations')
    for v,records in by_view.items():
        candidates=[root/'runs/fallback-v6/blobs'/v,root/'runs/svt-additions-v1/blobs'/v]
        p=next((p for p in candidates if p.exists()),None)
        if p is None or sha(p)!=v:raise ValueError('published archive eligibility CAS missing')
        doc=json.loads(gzip.decompress(p.read_bytes()))
        if doc.get('complete') is not True:raise ValueError('archive source ladder incomplete')
        native={r['bundle']['sha256']:r['bundle'] for r in doc['rows']}
        for row in records:
            b=native.get(row['sha256'])
            if b is None or b['host']!='dev' or b['root']!=str(root/'artifacts-local-v1') or b['bytes']!=row['bytes']:
                raise ValueError('archived object absent from its exact native view')
    sampled=random.Random(36).sample(sorted(unique),min(3,len(unique)))
    client=Client(storage['command'],storage['program_sha256']);checks=[]
    try:
        for h in sampled:
            answer=client.verify(h)
            if answer['bytes']!=unique[h]['bytes']:raise ValueError('final archive readback size changed')
            checks.append(answer)
    finally:
        # EOF closes the idle verifier naturally; no process is terminated.
        if client.process is not None:
            client.process.stdin.close();client.process.wait(timeout=30);client.process.stdout.close()
    if sha(journal)!=journal_sha:raise ValueError('relocation journal changed during freeze')
    resolver_path=root/'owner-svt-ssd-v1/artifact-location-final.py'
    if resolver_path.exists():
        if sha(resolver_path)!=sha(__file__):raise ValueError('preserve previous artifact resolver before changing it')
    else:shutil.copyfile(__file__,resolver_path)
    result=dict(schema='e36-artifact-location-admission-v1',admitted=True,
                journal=dict(path=str(journal),sha256=journal_sha),
                resolver=dict(path=str(resolver_path),sha256=sha(resolver_path)),
                storage=dict(path=str(storage_path),sha256=sha(storage_path)),
                native_ledgers=ledger_pins,unique_relocated_objects=len(unique),native_complete_views=len(by_view),
                actual_random_readbacks=checks,relocation_eligibility_replayed=True,utc=time.time(),fits=0)
    with out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
    return result
