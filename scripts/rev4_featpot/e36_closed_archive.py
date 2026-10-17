"""Complete, target-free inventory after the native generation control closes."""
import gzip
import json
import os
from pathlib import Path
import pyarrow.parquet as pq
from e36_store import sha,digest

from e36_archive_runs import native_runs


def checked(pin):
    path=Path(pin['path'])
    if sha(path)!=pin['sha256']:raise ValueError('closed native proof changed')
    return path


def closed_generation(root):
    root=Path(root);fresh=(root/'owner-svt-ssd-v1/native-continuation-v3-binding.json').exists();terminal_path=root/'owner-svt-ssd-v1'/('native-continuation-final-v3.json' if fresh else 'native-continuation-final-v1.json');terminal=json.loads(terminal_path.read_text())
    if terminal['schema']!=('e36-fresh-native-continuation-final-v1' if fresh else 'e36-native-continuation-final-v1') or terminal['closed'] is not True or terminal['fits']!=0 or terminal['scientific_change'] is not False:raise ValueError('closed native terminal proof required')
    binding=json.loads(checked(terminal['fresh_binding'] if fresh else terminal['continuation']).read_text());launch=json.loads(checked(binding['launch']).read_text())
    if binding['schema']!=('e36-fresh-native-continuation-v1' if fresh else 'e36-native-continuation-binding-v1') or binding['fits']!=0 or binding['scientific_change'] is not False or launch['schema']!='e36-existing-native-ledger-continuation-v1' or launch['fits']!=0 or launch['scientific_change'] is not False:raise ValueError('closed native launch/source binding differs')
    exit_record=json.loads(checked(terminal['exit']).read_text());helpers=json.loads(checked(terminal['helper_exits']).read_text());checked(terminal['log'])
    if exit_record['returncode']!=terminal['returncode'] or len(helpers['exits'])!=2 or any(r['returncode']!=0 for r in helpers['exits']):raise ValueError('native terminal exit/helper evidence differs')
    worker=root/'bin/zenfleet-worker-fair-v7'
    for process in Path('/proc').iterdir():
        if not process.name.isdigit():continue
        try:
            if process.stat().st_uid!=os.getuid():continue
            if (process/'exe').resolve(strict=True)==worker:raise ValueError('native generation is still active')
        except FileNotFoundError:continue
        except PermissionError:
            # An unrelated nondumpable process grants no authority; owned E36
            # worker identity is also bound by its controller terminal record.
            continue
    controller=Path('/proc')/str(launch['controller_pid'])
    if controller.exists() and b'e36_generation_resume.py' in (controller/'cmdline').read_bytes():raise ValueError('native controller remains active')
    retained=[checked(p) for p in launch['ledger_inputs']]
    paths=set(retained)
    for name in native_runs(root):paths.update((root/'runs'/name).glob('ledger*.parquet'))
    if not paths:raise ValueError('closed native ledger inventory is empty')
    return terminal_path,sorted(paths)


def inventory(root):
    root=Path(root);terminal,ledgers=closed_generation(root);published=set();ledger_pins=[]
    for path in ledgers:
        # Closed inputs cannot be treated as an in-progress graceful skip.
        for batch in pq.ParquetFile(path).iter_batches(columns=['status','attempts','output_sha']):
            published.update(r['output_sha'] for r in batch.to_pylist() if r['status']=='done' and r['attempts']==1 and r['output_sha'])
        ledger_pins.append(dict(path=str(path),sha256=sha(path)))
    queue={};views=[];local=root/'artifacts-local-v1'
    for value in sorted(published):
        h=digest(value);paths=[root/'runs'/n/'blobs'/h for n in native_runs(root)];path=next((p for p in paths if p.exists()),None)
        if path is None or sha(path)!=h:raise ValueError('closed native published CAS missing or changed')
        document=json.loads(gzip.decompress(path.read_bytes()))
        if document.get('schema')!='e36-native-ladder-receipt-v1' or document.get('complete') is not True:raise ValueError('closed native CAS incomplete')
        views.append(dict(path=str(path),sha256=h))
        for row in document['rows']:
            bundle=row['bundle'];b=digest(bundle['sha256'])
            if bundle['host']=='dev' and bundle['root']==str(local):
                relative='blobs/'+b[:2]+'/'+b+'.tar'
                if bundle['relative_path']!=relative:raise ValueError('closed bundle path differs from content address')
                artifact=local/relative
                if artifact.exists():
                    if artifact.stat().st_size!=bundle['bytes']:raise ValueError('closed local bundle size differs')
                    queue.setdefault(b,(artifact,h))
    proof=dict(schema='e36-closed-native-archive-inventory-v1',terminal=dict(path=str(terminal),sha256=sha(terminal)),ledgers=ledger_pins,views=views,
               published_views=len(views),local_eligible_bundles=len(queue),local_eligible_bytes=sum(p.stat().st_size for p,_ in queue.values()),fits=0,qualifies_data=False)
    return queue,proof
