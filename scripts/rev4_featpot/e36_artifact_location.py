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
