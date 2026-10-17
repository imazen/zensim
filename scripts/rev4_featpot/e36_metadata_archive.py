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
from e36_artifact_location import durable_mkdir,sync_directory
from e36_store import sha,digest
from e36_pack_store import HashReader,HashWriter


class PipeWriter(HashWriter):
    def write(self,data):
        view=memoryview(data);written=0
        while written<len(view):
            count=self.stream.write(view[written:])
            if not count:raise BrokenPipeError('metadata stream made no progress')
            self.hash.update(view[written:written+count]);self.size+=count;written+=count
        return written


def write(path,value):
    with Path(path).open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    sync_directory(Path(path).parent)



def tar_format(job):
    value=job.get('tar_format','ustar')
    if value not in ['ustar','pax']:raise ValueError('explicit supported metadata tar format required')
    return tarfile.PAX_FORMAT if value=='pax' else tarfile.USTAR_FORMAT


def member_info(row):
    info=tarfile.TarInfo(row.get('member_name',row['sha256']+'.blob'));info.size=row['bytes'];info.mode=0o444
    return info


def tar_bytes(items,format_name='ustar'):
    format_code=tar_format(dict(tar_format=format_name))
    raw=sum(len(member_info(row).tobuf(format=format_code))+((row['bytes']+511)//512)*512 for row in items)
    return ((raw+1024+10239)//10240)*10240

def plan(root,out,commit):
    root=Path(root);out=Path(out)
    from e36_metadata_seal import freeze as seal_metadata
    seal=seal_metadata(root);sealed=json.loads(seal.read_text())
    terminal=Path(sealed['native_terminal']['path']);drain_path=Path(sealed['raw_drain']['path'])
    stop=root/'owner-svt-ssd-v1/mirror-stop.json'
    if not stop.exists():raise ValueError('stop archive writers before metadata inventory')
    for receipt in (root/'owner-svt-ssd-v1').glob('mirror*pid.json'):
        pid=json.loads(receipt.read_text()).get('pid');process=Path('/proc')/str(pid)
        if process.exists() and b'e36_artifact_mirror' in (process/'cmdline').read_bytes():raise ValueError('metadata inventory requires closed mirror')
    durable_mkdir(out);files=[];unique={};links=[]
    for path in sorted(root.rglob('*')):
        relative=path.relative_to(root)
        if relative.parts[0]=='artifact-access-locks' or path.is_relative_to(out):continue
        if path.is_symlink():links.append(dict(path=str(relative),target=os.readlink(path)));continue
        if not path.is_file():continue
        stat=path.stat();h=sha(path)
        if path.stat().st_size!=stat.st_size or path.stat().st_mtime_ns!=stat.st_mtime_ns:raise ValueError('metadata changed during inventory')
        row=dict(path=str(relative),sha256=h,bytes=stat.st_size);files.append(row);unique.setdefault(h,dict(row,local_path=str(path)))
    groups=[];current=[];size=0
    format_name='pax' if any(row['bytes']>=8*1024**3 for row in unique.values()) else 'ustar'
    for row in unique.values():
        if current and size+row['bytes']>512*1024**2:groups.append(current);current=[];size=0
        current.append(row);size+=row['bytes']
    if current:groups.append(current)
    jobs=[]
    for group in groups:
        value=hashlib.sha256(json.dumps(group,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        stored=tar_bytes(group,format_name)
        if stored>64*1024**3:raise ValueError('metadata job exceeds receiver cap')
        jobs.append(dict(id=value,items=group,tar_bytes=stored,tar_format=format_name))
    result=dict(schema='e36-closed-metadata-archive-plan-v2' if format_name=='pax' else 'e36-closed-metadata-archive-plan-v1',archive_format=format_name,root=str(root),build_commit=commit,raw_drain=dict(path=str(drain_path),sha256=sha(drain_path)),seal=dict(path=str(seal),sha256=sha(seal)),ustar_members_fit=all(row['bytes']<8*1024**3 for row in files),terminal=dict(path=str(terminal),sha256=sha(terminal)),stop=dict(path=str(stop),sha256=sha(stop)),files=files,symlinks=links,jobs=jobs,raw_bytes=sum(r['bytes'] for r in unique.values()),fits=0,labels_read=False,local_copies_retained=True)
    result['client_source']=pin(Path(__file__))
    result['retained_local_raw_artifacts']=dict(files=[row for row in files if Path(row['path']).parts[0]=='artifacts-local-v1'],training_admission=False,local_copies_retained=True)
    sources=Path(__file__).with_name('sources.json')
    if sources.exists():result['control_sources']=pin(sources)
    write(out/'plan.json',result);return result


def validate_closed_plan(plan):
    from e36_metadata_seal import validate as validate_seal
    sealed=validate_seal(plan['root'],plan['seal'])
    if sealed['raw_drain']!=plan['raw_drain'] or sealed['native_terminal']!=plan['terminal']:raise ValueError('metadata plan differs from sealed closure')
    large=plan.get('schema')=='e36-closed-metadata-archive-plan-v2'
    if large and plan.get('archive_format')!='pax':raise ValueError('v2 metadata plan requires explicit PAX format')
    expected={r['sha256']:r['bytes'] for r in plan['files']};actual={}
    for job in plan['jobs']:
        if job.get('tar_format','ustar')!=('pax' if large else 'ustar'):raise ValueError('metadata schema/format differs')
        limit=64*1024**3 if large else 8*1024**3
        if any(not 0<=row['bytes']<limit for row in job['items']) or not 0<job['tar_bytes']<=64*1024**3:raise ValueError('metadata USTAR/receiver preflight failed')
        if job['tar_bytes']!=tar_bytes(job['items'],job.get('tar_format','ustar')):raise ValueError('metadata declared tar length differs')
        for row in job['items']:
            if row['sha256'] in actual:raise ValueError('metadata payload hash scheduled twice')
            actual[row['sha256']]=row['bytes']
    if expected!=actual:raise ValueError('metadata archive member union differs from file map')


def validate_storage(config):
    binding=Path(config['base_storage']['path'])
    if sha(binding)!=config['base_storage']['sha256']:raise ValueError('metadata storage predecessor changed')
    base=json.loads(binding.read_text());identity=digest(base['container_id']);program=digest(config['program_sha256'])
    expected=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','tower','docker exec -i '+identity+' python3 -u /data/program/'+program+'.py']
    if base['host']!='tower' or base['root']!='/mnt/disk1/coefficient/output/zensim/e36-bulk-2026-10-17-disk1' or config['root']!=base['root']+'/metadata-v1' or config['destination']!='/data/metadata-v1' or config['command']!=expected:raise ValueError('metadata destination must be the admitted direct disk1 container')
    captured=Path(config['container_inspect']['path'])
    if sha(captured)!=config['container_inspect']['sha256']:raise ValueError('metadata storage inspect proof changed')
    observed=json.loads(subprocess.check_output(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','tower','docker inspect '+identity]))[0]
    frozen=json.loads(captured.read_text())[0]
    def identity_fields(document):
        return {k:document[k] for k in ['Id','Image','Mounts']}|dict(Hostname=document['Config']['Hostname'],Memory=document['HostConfig']['Memory'],CpusetCpus=document['HostConfig']['CpusetCpus'],CpuShares=document['HostConfig']['CpuShares'])
    if identity_fields(observed)!=identity_fields(frozen) or observed['Id']!=identity or observed['Image']!=base['image_id'] or observed['HostConfig']['Memory']!=2147483648 or observed['Config']['Hostname']!=config['hostname'] or not observed['State']['Running']:raise ValueError('metadata storage container identity or cap differs')
    mounts=[m for m in observed['Mounts'] if m['Destination']=='/data']
    if len(mounts)!=1 or mounts[0]['Source']!=base['root'] or mounts[0]['RW'] is not True:raise ValueError('metadata disk1 bind mapping differs')


def owned_exchange(command,send):
    from e36_process_owner import DeferredCancellation
    process=None;error=None;writer=None;answer=b'';code=None
    cancellation=DeferredCancellation();cancellation.__enter__()
    try:
        process=cancellation.spawn(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,bufsize=0)
        cancellation.installed()
        writer=PipeWriter(process.stdin);send(writer)
    except BaseException as exc:error=exc
    finally:
        if process is not None:
            try:
                while not process.stdin.closed:
                    try:process.stdin.close()
                    except BaseException as exc:
                        if error is None:error=exc
                        # Retry the same unbuffered owner until it records
                        # closure. Closing only its numeric FD would leave a
                        # stale file object able to close a later reused FD.
                        continue
                blocks=[]
                while True:
                    try:
                        block=os.read(process.stdout.fileno(),1024**2)
                        if not block:break
                        blocks.append(block)
                    except (KeyboardInterrupt,InterruptedError) as exc:
                        if error is None:error=exc
                    except OSError as exc:
                        if error is None:error=exc
                        # Closing the reader also prevents a failed stdout drain
                        # from leaving a child blocked on a full reply pipe.
                        process.stdout.close();break
                answer=b''.join(blocks)
            except BaseException as exc:
                if error is None:error=exc
            finally:
                while True:
                    try:code=process.wait();break
                    except BaseException as exc:
                        if error is None:error=exc
                try:process.stdout.close()
                except BaseException as exc:
                    if error is None:error=exc
        try:cancellation.__exit__(None,None,None)
        except BaseException as exc:
            if error is None:error=exc
    if error is not None:raise error
    if code:raise RuntimeError('metadata receiver failed: '+str(code))
    return answer,writer


def transmit(job,command,program,destination='/data/metadata-v1',hostname=None):
    def send(writer):
        with tarfile.open(fileobj=writer,mode='w|',format=tar_format(job),bufsize=1024**2) as archive:
            for row in job['items']:
                info=member_info(row)
                with Path(row['local_path']).open('rb') as source:
                    reader=HashReader(source);archive.addfile(info,reader)
                    if reader.hash.hexdigest()!=row['sha256'] or reader.size!=info.size or source.read(1):raise ValueError('opaque metadata changed during transfer')
    answer,writer=owned_exchange(command+['receive',str(destination),'--id',job['id'],'--size',str(job['tar_bytes'])],send)
    actual=json.loads(answer);expected={row['sha256']:row['bytes'] for row in job['items']}
    index=next((row['sha256'] for row in job['items'] if row.get('member_name')=='INDEX.json'),None)
    if actual['members']!=expected or actual['sha256']!=writer.hash.hexdigest() or actual['bytes']!=job['tar_bytes'] or writer.size!=job['tar_bytes'] or actual['program_sha256']!=program or actual['destination']!=str(destination) or (hostname is not None and actual['hostname']!=hostname) or actual.get('index_sha256')!=index:raise ValueError('metadata remote whole/member/owner proof differs')
    return actual


def archive(plan_path,storage,out,lock):
    out=Path(out);durable_mkdir(out);plan_path=Path(plan_path);p=json.loads(plan_path.read_text());config=json.loads(Path(storage).read_text())
    if p['schema'] not in ['e36-closed-metadata-archive-plan-v1','e36-closed-metadata-archive-plan-v2'] or p['fits']!=0 or p['labels_read'] is not False:raise ValueError('closed metadata plan required')
    validate_storage(config)
    validate_closed_plan(p)
    command=config['command'];receipts=[]
    with Path(lock).open('a') as guard:
        for job in p['jobs']:
            fcntl.flock(guard,fcntl.LOCK_EX)
            try:
                validate_closed_plan(p);validate_storage(config)
                started=time.time();answer=transmit(job,command,config['program_sha256'],config['destination'],config['hostname']);row=dict(job_id=job['id'],canonical_root=config['root'],relative_path=job['id']+'.tar',elapsed_seconds=time.time()-started,**answer)
                write(out/(job['id']+'.json'),row);receipts.append(row)
            finally:fcntl.flock(guard,fcntl.LOCK_UN)
            time.sleep(.2)
            print(json.dumps(dict(completed=len(receipts),total=len(p['jobs']),bytes=sum(r['bytes'] for r in receipts),fits=0)),flush=True)
    import random
    sample=random.Random(36).sample(p['files'],min(3,len(p['files'])))
    for item in sample:
        if not any(receipt['members'].get(item['sha256'])==item['bytes'] for receipt in receipts):raise ValueError('sampled metadata file lacks remote rehash')
    result=dict(schema='e36-metadata-payload-completion-v1',restorable_backup_complete=False,plan=dict(path=str(plan_path),sha256=sha(plan_path)),storage=dict(path=str(storage),sha256=sha(storage)),archives=receipts,all_members_remote_rehashed=True,three_random_files_remote_rehashed=sample,local_copies_retained=True,fits=0,labels_read=False)
    write(out/'payload-receipt.json',result)
    return finalize_index(plan_path,storage,out,lock,result,config)


def final_index_job(out,paths):
    files=[]
    for path in paths:
        path=Path(path);files.append(dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size))
    descriptor=Path(out)/'INDEX.json';write(descriptor,dict(schema='e36-restorable-metadata-index-v1',files=files,restore='Read INDEX.json first; recover each file from its SHA.blob member. Payload plan maps every original relative path and symlink target. No label decoding required.',fits=0))
    items=[dict(local_path=str(descriptor),sha256=sha(descriptor),bytes=descriptor.stat().st_size,member_name='INDEX.json')]
    unique={}
    for row in files:unique.setdefault(row['sha256'],dict(local_path=row['path'],sha256=row['sha256'],bytes=row['bytes']))
    items.extend(unique.values());raw=sum(512+((r['bytes']+511)//512)*512 for r in items);size=((raw+1024+10239)//10240)*10240
    if size>64*1024**3 or any(r['bytes']>=8*1024**3 for r in items):raise ValueError('metadata final index exceeds USTAR/receiver limits')
    value=hashlib.sha256(json.dumps(items,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(id=value,items=items,tar_bytes=size)


def finalize_index(plan_path,storage,out,lock,payload,config):
    paths=[Path(plan_path),Path(storage),Path(config['base_storage']['path']),Path(config['container_inspect']['path']),out/'payload-receipt.json']+[out/(row['job_id']+'.json') for row in payload['archives']]
    job=final_index_job(out,paths);write(out/'index-job.json',job)
    with Path(lock).open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        try:
            validate_closed_plan(json.loads(Path(plan_path).read_text()));validate_storage(config);answer=transmit(job,config['command'],config['program_sha256'],config['destination'],config['hostname'])
            locator=dict(schema='e36-restorable-metadata-locator-v1',canonical_root=config['root'],index_id=job['id'],index_sha256=answer['sha256'],descriptor_sha256=answer['index_sha256'],plan_sha256=sha(plan_path),fits=0)
            body=(json.dumps(locator,sort_keys=True)+'\n').encode();value=hashlib.sha256(body).hexdigest()
            # A bounded control-only locator is published in the same admitted
            # container; the receiver re-verifies its referenced index first.
            output,_=owned_exchange(config['command']+['locate',config['destination'],'--id',value,'--size',str(len(body))],lambda writer:writer.write(body))
            located=json.loads(output)
            if located['sha256']!=value or located['destination']!=config['destination'] or located['hostname']!=config['hostname'] or located['program_sha256']!=config['program_sha256']:raise ValueError('metadata locator owner/hash differs')
        finally:fcntl.flock(guard,fcntl.LOCK_UN)
    result=dict(schema='e36-metadata-archive-admission-v1',restorable_backup_complete=True,payload=pin(out/'payload-receipt.json'),index=answer,index_job=pin(out/'index-job.json'),remote_locator=located,locator=locator,storage=pin(Path(storage)),fits=0,labels_read=False,local_copies_retained=True)
    write(out/'receipt.json',result);return result


def pin(path):return dict(path=str(path),sha256=sha(path))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['plan','archive']);parser.add_argument('--root',type=Path);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--commit');parser.add_argument('--plan',type=Path);parser.add_argument('--storage',type=Path);parser.add_argument('--lock',type=Path);a=parser.parse_args()
    plan(a.root,a.out,a.commit) if a.operation=='plan' else archive(a.plan,a.storage,a.out,a.lock)
