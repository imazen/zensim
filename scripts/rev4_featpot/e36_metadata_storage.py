"""Bind a metadata receiver to the existing capped direct-disk storage container."""
import argparse
import json
from pathlib import Path
import shlex
import shutil
import subprocess
from e36_store import sha,digest
from e36_artifact_location import durable_mkdir
from e36_metadata_archive import validate_storage,write,pin


def prepare(root,out,commit):
    root=Path(root);out=Path(out);durable_mkdir(out);base_path=root/'storage-archive-v2.json';base=json.loads(base_path.read_text());identity=digest(base['container_id'])
    if base['host']!='tower' or base['root']!='/mnt/disk1/coefficient/output/zensim/e36-bulk-2026-10-17-disk1':raise ValueError('existing direct disk1 storage binding required')
    ssh=['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10','tower']
    inspect=json.loads(subprocess.check_output(ssh+['docker inspect '+identity]));write(out/'container-inspect.json',inspect)
    source=Path(__file__).with_name('e36_metadata_store.py');retained=out/'e36_metadata_store.py'
    if retained.exists():raise ValueError('preserve earlier metadata receiver source')
    shutil.copyfile(source,retained);program=sha(retained)
    code='''import os,hashlib,json,socket,sys
from pathlib import Path
p=Path('/data/program/'''+program+'''.py');body=sys.stdin.buffer.read()
if hashlib.sha256(body).hexdigest()!='''+repr(program)+''':raise ValueError('program upload hash differs')
if p.exists():
 if hashlib.sha256(p.read_bytes()).hexdigest()!='''+repr(program)+''':raise ValueError('program collision')
else:
 with p.open('xb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
 fd=os.open(p.parent,os.O_RDONLY|os.O_DIRECTORY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(),hostname=socket.gethostname(),destination=str(p))),flush=True)
'''
    reply=json.loads(subprocess.check_output(ssh+['docker exec -i '+identity+' python3 -c '+shlex.quote(code)],input=retained.read_bytes()))
    if reply['sha256']!=program or reply['hostname']!=inspect[0]['Config']['Hostname']:raise ValueError('actual metadata receiver readback differs')
    write(out/'program-readback.json',reply)
    config=dict(schema='e36-bound-metadata-storage-v1',base_storage=pin(base_path),container_inspect=pin(out/'container-inspect.json'),server_source=pin(retained),program_readback=pin(out/'program-readback.json'),host='tower',root=base['root']+'/metadata-v1',destination='/data/metadata-v1',hostname=reply['hostname'],program_sha256=program,
        command=ssh+['docker exec -i '+identity+' python3 -u /data/program/'+program+'.py'],fits=0,build_commit=commit)
    validate_storage(config);write(out/'storage.json',config);write(out/'_MANIFEST.json',dict(build_commit=commit,storage=pin(out/'storage.json'),fits=0,labels_read=False))
    return pin(out/'storage.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','out']:parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--commit',required=True);a=parser.parse_args();print(json.dumps(prepare(a.root,a.out,a.commit)))
