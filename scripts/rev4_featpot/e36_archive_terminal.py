"""Own and close a storage-only raw drain, with a terminal inventory proof."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from e36_store import sha
from e36_artifact_location import durable_mkdir,sync_directory


def pin(path):return dict(path=str(path),sha256=sha(path))

def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    sync_directory(Path(path).parent)


def checked(value):
    path=Path(value['path'])
    if sha(path)!=value['sha256']:raise ValueError('archive terminal evidence changed')
    return path


def validate(root):
    root=Path(root);path=root/'owner-svt-ssd-v1/archive-drain-final-v1.json';value=json.loads(path.read_text())
    if value['schema']!='e36-raw-drain-terminal-v1' or value['closed'] is not True or value['fits']!=0 or value['returncode']!=0:raise ValueError('successful raw drain terminal required')
    launch=json.loads(checked(value['launch']).read_text());exit_record=json.loads(checked(value['exit']).read_text());stop=json.loads(checked(value['stop']).read_text());inventory=json.loads(checked(stop['inventory']).read_text());log=checked(value['log']);sources=json.loads(checked(value['sources']).read_text())
    if exit_record['returncode']!=0 or stop['reason']!='all native-published local bundles drained; metadata retained for separate canonical archive' or stop['fits']!=0 or inventory['schema']!='e36-closed-native-archive-inventory-v1' or inventory['fits']!=0 or inventory['qualifies_data'] is not False:raise ValueError('raw drain completion semantics differ')
    if not any(line.startswith('run-heavy: done rc=0 ') for line in log.read_text().splitlines()):raise ValueError('raw drain terminal resource line missing')
    checked(launch['source']);checked(launch['mirror'])
    for source in sources['files'].values():checked(source)
    for pid in [launch['controller_pid'],launch['mirror_pid']]:
        process=Path('/proc')/str(pid)
        if process.exists():
            args=(process/'cmdline').read_bytes()
            if b'e36_artifact_mirror.py' in args or b'e36_archive_terminal.py' in args:raise ValueError('owned raw archive child still active')
    if checked(value['stop'])!=root/'owner-svt-ssd-v1/mirror-stop.json':raise ValueError('raw drain stop owner differs')
    return path,value


def run(root,out,resources):
    from e36_closed_archive import closed_generation
    root=Path(root);out=Path(out);resources=Path(resources);closed_generation(root)
    if (root/'owner-svt-ssd-v1/metadata-sealed-v1.json').exists():raise ValueError('metadata already sealed')
    if (root/'owner-svt-ssd-v1/mirror-stop.json').exists():raise ValueError('preserve prior stop marker before raw drain')
    durable_mkdir(out);mirror=resources/'e36_artifact_mirror.py';process=None
    env={**os.environ,'PYTHONPATH':str(resources),'E36_ARTIFACTS_ONLY':'1','E36_ARCHIVE_INTERLEAVE_LOCK':str(Path.home()/'tmp/zensim-paper/rev4/heavy.lock'),'E36_ARCHIVE_INVENTORY_PATH':str(out/'inventory.json'),'E36_RESOURCE_READY_PATH':str(out/'ready.json'),'PYTHONDONTWRITEBYTECODE':'1'}
    from e36_process_owner import DeferredCancellation
    cancellation=DeferredCancellation();cancellation.__enter__()
    try:
        process=cancellation.spawn(['python3','-u',str(mirror)],env=env)
        cancellation.installed()
        write(out/'launch.json',dict(schema='e36-raw-drain-launch-v1',controller_pid=os.getpid(),mirror_pid=process.pid,source=pin(Path(__file__)),mirror=pin(mirror),sources=pin(resources/'sources.json'),fits=0,utc=time.time()))
        write(root/'owner-svt-ssd-v1/mirror-drain-pid.json',dict(pid=process.pid,controller_pid=os.getpid(),utc=time.time(),fits=0))
        code=process.wait()
    finally:
        if process is not None:
            while True:
                try:code=process.wait();break
                except BaseException:continue
            write(out/'exit.json',dict(returncode=code,closed_unix=time.time(),fits=0))
        cancellation.__exit__(None,None,None)
    if code:raise RuntimeError('raw archive drain did not complete')


def close(root,out,log):
    root=Path(root);out=Path(out);launch=json.loads((out/'launch.json').read_text());exit_record=json.loads((out/'exit.json').read_text());stop=root/'owner-svt-ssd-v1/mirror-stop.json'
    if exit_record['returncode']!=0 or not any(line.startswith('run-heavy: done rc=0 ') for line in Path(log).read_text().splitlines()):raise ValueError('successful raw drain run-heavy terminal required')
    value=dict(schema='e36-raw-drain-terminal-v1',launch=pin(out/'launch.json'),exit=pin(out/'exit.json'),log=pin(log),sources=launch['sources'],stop=pin(stop),returncode=0,closed=True,fits=0,utc=time.time())
    path=root/'owner-svt-ssd-v1/archive-drain-final-v1.json';write(path,value);validate(root);return pin(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['run','close']);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--resources',type=Path);parser.add_argument('--log',type=Path);a=parser.parse_args()
    if a.operation=='run':
        from e36_metadata_seal import lifecycle
        with lifecycle(a.root):run(a.root,a.out,a.resources)
    else:print(json.dumps(close(a.root,a.out,a.log)))
