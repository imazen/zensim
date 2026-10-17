"""Prospective owner SVT-only and SSD-persistence overlay; never fits."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import types
from e36_artifact_location import access, durable_mkdir, resolve, sync_directory
from e36_disk_reserve import install as install_disk_reserve

ROOT = Path('/mnt/v/output/zensim/e36-bulk-2026-10-17')
LOCAL = ROOT / 'artifacts-local-v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def install(runtime):
    if getattr(runtime, '_owner_svt_ssd', False) == 2:
        return
    if runtime.root != ROOT:
        raise ValueError('owner overlay refuses another experiment')
    original_execute = runtime.execute
    original_cell = runtime.execute_cell
    original_verify = runtime.store.verify
    while hasattr(original_verify,'__func__') and 'original_verify' in original_verify.__func__.__code__.co_freevars:
        index=original_verify.__func__.__code__.co_freevars.index('original_verify')
        original_verify=original_verify.__func__.__closure__[index].cell_contents
    durable_mkdir(LOCAL)
    storage = json.loads((ROOT/'storage-runtime.json').read_text())

    def verify(store, digest):
        # The frozen checkpoint caller supplies its immutable bundle in the
        # suspended frame. Honor that recorded owner, even if another local
        # object happens to have the same hash.
        frame=sys._getframe(1)
        recorded=frame.f_locals.get('result',{}).get('bundle')
        if recorded and recorded['sha256']==digest:
            return resolve(recorded,ROOT,original_verify,storage)
        path = LOCAL / 'blobs' / digest[:2] / (digest + '.tar')
        with access(ROOT,digest):
            if not path.exists():
                return original_verify(digest)
            if sha(path)!=digest:raise ValueError('local canonical hash mismatch')
            return dict(ok=True,sha256=digest,bytes=path.stat().st_size,
                        relative_path=str(path.relative_to(LOCAL)),program_sha256=sha(__file__))

    def put(store, bundle):
        bundle = Path(bundle)
        if shutil.disk_usage(LOCAL).free < 40 * 1024**3:
            raise OSError('owner SSD 40GiB free-space floor')
        digest = sha(bundle)
        path = LOCAL / 'blobs' / digest[:2] / (digest + '.tar')
        durable_mkdir(path.parent)
        with access(ROOT,digest,exclusive=True):
            with bundle.open('rb') as stream:
                os.fsync(stream.fileno())
            try:
                os.link(bundle, path)
            except FileExistsError:
                if sha(path) != digest:
                    raise ValueError('refusing conflicting local object')
            sync_directory(path.parent)
        # Set the receipt location only after this local put succeeds. A remote
        # put already in flight at installation must keep its tower location.
        runtime.spec['storage'] = dict(runtime.spec['storage'], root=str(LOCAL), host='dev')
        return dict(ok=True, sha256=digest, bytes=path.stat().st_size,
                    relative_path=str(path.relative_to(LOCAL)), program_sha256=sha(__file__))

    def execute(owner, job):
        view = json.loads(job['kind']['view'].split(':', 1)[1])
        if view['codec'] == 'rav1e':
            raise ValueError('E36_OWNER_EXCLUDED_RAV1E: prospective SVT-only amendment; preserved diagnostics')
        return original_execute(job)

    def cell(owner, ref, codec, *args, **kwargs):
        if codec == 'rav1e':
            raise ValueError('E36_OWNER_EXCLUDED_RAV1E: no new physical cell after owner cutoff')
        return original_cell(ref, codec, *args, **kwargs)

    runtime.store.put = types.MethodType(put, runtime.store)
    runtime.store.verify = types.MethodType(verify, runtime.store)
    runtime.execute = types.MethodType(execute, runtime)
    runtime.execute_cell = types.MethodType(cell, runtime)
    # Historical checkpoints retain their original tower owner. Only future
    # bundle receipts use this explicitly registered storage overlay.
    runtime._owner_svt_ssd = 2
    install_disk_reserve(runtime)


def live(receipt):
    instances = {}
    for top in sys._current_frames().values():
        frame = top
        while frame:
            candidate = frame.f_locals.get('runtime')
            if candidate is not None and getattr(candidate, 'root', None) == ROOT:
                instances[id(candidate)] = candidate
            candidate = frame.f_locals.get('self')
            if candidate is not None and getattr(candidate, 'root', None) == ROOT and hasattr(candidate, 'execute_cell'):
                instances[id(candidate)] = candidate
            frame = frame.f_back
    if len(instances) != 1:
        raise ValueError('unique owned native Runtime required')
    install(next(iter(instances.values())))
    with Path(receipt).open('x') as out:
        json.dump(dict(pid=os.getpid(), applied_unix_ns=time.time_ns(),
                       source_sha256=sha(__file__), new_rav1e_cells=False,
                       future_put_root=str(LOCAL), fits=0), out)


def startup():
    if '--runtime' not in sys.argv or str(ROOT/'runtime-v6/runtime.json') not in sys.argv:
        return
    def ready(frame, event, arg):
        if event == 'return' and frame.f_code.co_name == '__init__':
            candidate = frame.f_locals.get('self')
            if type(candidate).__name__ == 'Runtime' and getattr(candidate, 'root', None) == ROOT:
                install(candidate)
                sys.setprofile(None)
    sys.setprofile(ready)
