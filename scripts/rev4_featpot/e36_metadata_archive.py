"""Retain closed E36 metadata as sequential opaque, independently verified archives."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from e36_store import sha
from e36_pack_store import HashReader,HashWriter


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())


def plan(root,out,commit):
    from e36_closed_archive import closed_generation
    root=Path(root);out=Path(out);terminal,_=closed_generation(root)
    stop=root/'owner-svt-ssd-v1/mirror-stop.json'
    if not stop.exists():raise ValueError('stop archive writers before metadata inventory')
    for receipt in (root/'owner-svt-ssd-v1').glob('mirror*pid.json'):
        pid=json.loads(receipt.read_text()).get('pid');process=Path('/proc')/str(pid)
        if process.exists() and b'e36_artifact_mirror' in (process/'cmdline').read_bytes():raise ValueError('metadata inventory requires closed mirror')
    out.mkdir();files=[];unique={};links=[]
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if relative.parts[0] in ['artifacts-local-v1','artifact-access-locks'] or path.is_relative_to(out):continue
        if path.is_symlink():links.append(dict(path=str(relative),target=os.readlink(path)));continue
        if not path.is_file():continue
        stat=path.stat();h=sha(path)
        if path.stat().st_size!=stat.st_size or path.stat().st_mtime_ns!=stat.st_mtime_ns:raise ValueError('metadata changed during inventory')
        row=dict(path=str(relative),sha256=h,bytes=stat.st_size);files.append(row);unique.setdefault(h,dict(row,local_path=str(path)))
    groups=[];current=[];size=0
    for row in unique.values():
        if current and size+row['bytes']>512*1024**2:groups.append(current);current=[];size=0
        current.append(row);size+=row['bytes']
    if current:groups.append(current)
    jobs=[]
    for group in groups:
        value=hashlib.sha256(json.dumps(group,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        raw=sum(512+((r['bytes']+511)//512)*512 for r in group);stored=((raw+1024+10239)//10240)*10240
        jobs.append(dict(id=value,items=group,tar_bytes=stored))
    result=dict(schema='e36-closed-metadata-archive-plan-v1',root=str(root),build_commit=commit,terminal=dict(path=str(terminal),sha256=sha(terminal)),stop=dict(path=str(stop),sha256=sha(stop)),files=files,symlinks=links,jobs=jobs,raw_bytes=sum(r['bytes'] for r in unique.values()),fits=0,labels_read=False,local_copies_retained=True)
    write(out/'plan.json',result);return result


def transmit(job,command,program,destination='/data/metadata-v1'):
    process=subprocess.Popen(command+['receive',str(destination),'--id',job['id'],'--size',str(job['tar_bytes'])],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    error=None;writer=HashWriter(process.stdin)
    try:
        with tarfile.open(fileobj=writer,mode='w|',format=tarfile.USTAR_FORMAT,bufsize=1024**2) as archive:
            for row in job['items']:
                info=tarfile.TarInfo(row['sha256']+'.blob');info.size=row['bytes'];info.mode=0o444
                with Path(row['local_path']).open('rb') as source:
                    reader=HashReader(source);archive.addfile(info,reader)
                    if reader.hash.hexdigest()!=row['sha256'] or reader.size!=info.size or source.read(1):raise ValueError('opaque metadata changed during transfer')
    except BaseException as exc:error=exc
    finally:
        try:process.stdin.close()
        except OSError:pass
        # EOF then join this exact process; never terminate or launch over it.
        blocks=[]
        while True:
            try:
                block=os.read(process.stdout.fileno(),1024**2)
                if not block:break
                blocks.append(block)
            except (KeyboardInterrupt,InterruptedError) as exc:
                if error is None:error=exc
        while True:
            try:code=process.wait();break
            except (KeyboardInterrupt,InterruptedError) as exc:
                if error is None:error=exc
        answer=b''.join(blocks);process.stdout.close()
    if error is not None:raise error
    if code:raise RuntimeError('metadata receiver failed: '+str(code))
    actual=json.loads(answer)
    expected={row['sha256']:row['bytes'] for row in job['items']}
    if actual['members']!=expected or actual['sha256']!=writer.hash.hexdigest() or actual['bytes']!=job['tar_bytes'] or writer.size!=job['tar_bytes'] or actual['program_sha256']!=program:raise ValueError('metadata remote whole/member proof differs')
    return actual


def archive(plan_path,storage,out,lock):
    out=Path(out);out.mkdir();plan_path=Path(plan_path);p=json.loads(plan_path.read_text());config=json.loads(Path(storage).read_text())
    if p['schema']!='e36-closed-metadata-archive-plan-v1' or p['fits']!=0 or p['labels_read'] is not False:raise ValueError('closed metadata plan required')
    command=config['command'];receipts=[]
    with Path(lock).open('a') as guard:
        for job in p['jobs']:
            fcntl.flock(guard,fcntl.LOCK_EX)
            try:
                started=time.time();answer=transmit(job,command,config['program_sha256']);row=dict(job_id=job['id'],canonical_root=config['root'],relative_path=job['id']+'.tar',elapsed_seconds=time.time()-started,**answer)
                write(out/(job['id']+'.json'),row);receipts.append(row)
            finally:fcntl.flock(guard,fcntl.LOCK_UN)
            time.sleep(.2)
            print(json.dumps(dict(completed=len(receipts),total=len(p['jobs']),bytes=sum(r['bytes'] for r in receipts),fits=0)),flush=True)
    import random
    sample=random.Random(36).sample(p['files'],min(3,len(p['files'])))
    for item in sample:
        if not any(receipt['members'].get(item['sha256'])==item['bytes'] for receipt in receipts):raise ValueError('sampled metadata file lacks remote rehash')
    result=dict(schema='e36-metadata-archive-admission-v1',plan=dict(path=str(plan_path),sha256=sha(plan_path)),storage=dict(path=str(storage),sha256=sha(storage)),archives=receipts,all_members_remote_rehashed=True,three_random_files_remote_rehashed=sample,local_copies_retained=True,fits=0,labels_read=False)
    write(out/'receipt.json',result);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['plan','archive']);parser.add_argument('--root',type=Path);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--commit');parser.add_argument('--plan',type=Path);parser.add_argument('--storage',type=Path);parser.add_argument('--lock',type=Path);a=parser.parse_args()
    plan(a.root,a.out,a.commit) if a.operation=='plan' else archive(a.plan,a.storage,a.out,a.lock)
