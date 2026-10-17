"""Create a bounded storage-only container from an already installed image."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys


def remote(host, argv, payload=None):
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        host,shlex.join(list(map(str,argv)))],input=payload,capture_output=True,check=True)
    if result.stderr:
        sys.stderr.buffer.write(result.stderr)
    return result.stdout


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host',required=True)
    p.add_argument('--root',required=True)
    p.add_argument('--image',required=True)
    p.add_argument('--container',required=True)
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args()
    root=Path(a.root)
    if not root.is_absolute() or len(root.parts)<4 or root.parts[1]!='mnt' or not root.parts[2].startswith('disk'):
        raise ValueError('bulk storage requires an explicit direct array-disk path')
    if a.output.exists():raise FileExistsError(a.output)
    image=remote(a.host,['docker','image','inspect',a.image,'--format','{{.Id}}']).decode().strip()
    source=Path(__file__).with_name('e36_store.py').read_bytes()
    digest=hashlib.sha256(source).hexdigest()
    bootstrap='''import hashlib,pathlib,sys
data=sys.stdin.buffer.read()
digest=hashlib.sha256(data).hexdigest()
p=pathlib.Path('/data/program')/(digest+'.py')
p.parent.mkdir(exist_ok=True)
if p.exists():
 assert p.read_bytes()==data
else:
 with p.open('xb') as f:f.write(data)
print(digest)
'''
    common=['--network=none','--cpuset-cpus=0-3','--cpu-shares=256',
        '--memory=2g','--memory-swap=2g','--pids-limit=128',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','TMPDIR=/data/scratch',
        '--mount',f'type=bind,source={a.root},target=/data',
        '--entrypoint','python3']
    # Docker creates a missing -v source; the explicit bind form below is used
    # only after this one capped initializer has made the data directory.
    initialized=remote(a.host,['docker','run','--rm','-i','--network=none',
        '--cpuset-cpus=0-3','--cpu-shares=256','--memory=256m','--memory-swap=256m',
        '--env','PYTHONDONTWRITEBYTECODE=1','--volume',f'{a.root}:/data',
        '--entrypoint','python3',image,'-c',bootstrap],source).decode().strip()
    if initialized!=digest:raise ValueError('storage program changed during bootstrap')
    container=remote(a.host,['docker','run','-d','--name',a.container,
        '--label','e36.role=storage-only','--read-only',*common,image,'-u','-c',
        'import time;time.sleep(86400)']).decode().strip()
    info=json.loads(remote(a.host,['docker','inspect',container]))[0]
    assert info['Image']==image and info['HostConfig']['Memory']==2*1024**3
    command=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',a.host,
        shlex.join(['docker','exec','-i',container,'python3','-u',
            '/data/program/'+digest+'.py','serve','--root','/data','--min-free-gib','128'])]
    result=dict(host=a.host,root=a.root,image_id=image,container_id=container,
        container_name=a.container,program_sha256=digest,command=command,
        memory_cap_bytes=2*1024**3,cpuset='0-3',cpu_shares=256,
        min_free_gib=128,role='storage-only; no generation, scoring, fitting or job claims')
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(container=container,program_sha256=digest,role=result['role'])))


if __name__=='__main__':main()
