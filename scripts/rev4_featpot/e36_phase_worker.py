"""Execute one entire frozen manifest through the native worker, never a fit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def run(files_path,phase):
    files_path=Path(files_path);spec=json.loads(files_path.read_text());root=Path(spec['root'])
    if spec['schema']!='e36-whole-native-phase-files-v1' or spec['scientific_change'] is not False or spec['fits']!=0:
        raise ValueError('whole frozen data-only phase required')
    for row in spec['files']:
        if sha(row['destination'])!=row['sha256']:raise ValueError('native phase input changed: '+row['destination'])
    clock_path=Path(spec['clock']['path'])
    if sha(clock_path)!=spec['clock']['sha256']:raise ValueError('effective clock changed')
    deadline=json.loads(clock_path.read_text())['deadline_unix'];remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('registered generation deadline expired before native phase start')
    entry=spec['first_producer'] if phase=='first-producer' else spec['manifests']['additions']
    manifest=Path(entry['path'])
    if sha(manifest)!=entry['sha256']:raise ValueError('native phase manifest changed')
    jobs=json.loads(manifest.read_text())
    if len(jobs)!=(entry['jobs'] if phase=='first-producer' else entry['views']):raise ValueError('native phase job count differs')
    for job in jobs:
        if job['kind']['kind']!='bake' or not job['kind']['view'].startswith('e36-native-ladder-v1:'):
            raise ValueError('native phase refuses every non-data job, including fits')
    directory=manifest.parent;execution=directory/'phase-execution.json'
    if execution.exists():raise ValueError('retain prior native execution before declaring a new pass')
    argv=['/usr/bin/timeout','--signal=TERM','--kill-after=120',str(remaining),str(root/'bin/zenfleet-worker'),
        '--manifest',str(manifest),'--ledger-out',str(directory/'ledger.parquet'),'--blobs',str(directory/'blobs'),
        '--exec',str(root/'runtime-v6/executor'),'--worker','e36-tower-'+phase,'--provider','local','--max-attempts','1']
    for ledger in sorted(directory.glob('ledger*.parquet')):argv.extend(['--ledger-in',str(ledger)])
    env={**os.environ,'ZEN_RAM_BYTES':str(spec['packing_ram_bytes']),'ZEN_CHUNK_WALL_SEC':'300',
        'ZEN_CHUNK_REF_CONCURRENCY':str(spec['worker_slots']),'ZEN_CHUNK_TAIL_SPREAD':'1','ZEN_DEBUG_ADMIT':'1',
        'ZEN_PERSISTENT_EXEC':'1','ZEN_PERSISTENT_KINDS':'bake','RAYON_NUM_THREADS':str(spec['worker_slots']),
        'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','PYTHONDONTWRITEBYTECODE':'1',
        'PYTHONPATH':'/opt/e36/python','PATH':'/opt/e36/transport:'+os.environ['PATH'],
        'E36_STORAGE_SPEC':spec['runtime']['path'],'E36_GENERATION_DEADLINE':str(deadline)}
    started=time.time_ns();result=subprocess.run(argv,env=env)
    # The filesystem evidence is measured after native execution. Harvest must
    # preserve these source timestamps exactly, never use transfer timestamps.
    inventory=[]
    for path in sorted(directory.rglob('*')):
        if path.is_file():
            st=path.stat();inventory.append(dict(relative=str(path.relative_to(directory)),sha256=sha(path),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
    for row in spec['files']:
        if sha(row['destination'])!=row['sha256']:raise ValueError('native phase input changed during execution')
    receipt=dict(schema='e36-whole-native-phase-execution-v1',phase=phase,manifest_sha256=entry['sha256'],files_manifest_sha256=sha(files_path),
        argv=argv,argv_sha256=hashlib.sha256(json.dumps(argv,separators=(',',':')).encode()).hexdigest(),
        effective_clock_sha256=spec['clock']['sha256'],native_worker_sha256=sha(root/'bin/zenfleet-worker'),
        started_unix_ns=started,closed_unix_ns=time.time_ns(),returncode=result.returncode,native_files=inventory,
        source_timestamps_preserved=True,scientific_change=False,fits=0,optimizer_steps=0)
    with execution.open('x') as f:json.dump(receipt,f,indent=2)
    return result.returncode


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--files',type=Path,required=True)
    p.add_argument('--phase',choices=['first-producer','additions'],required=True)
    a=p.parse_args();sys.exit(run(a.files,a.phase))
