"""Actual map-owner readback through a zstd archive; never labels or fitting."""
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import tarfile
import time

from e36_archive_store import ArchiveClient,raw_identity
from e36_data import validate_map
from e36_store import sha


def probe(root,path,out):
    root=Path(root);out=Path(out);out.mkdir()
    storage_path=root/'storage-archive-v2.json';storage=json.loads(storage_path.read_text())
    os.environ['E36_ARCHIVE_STAGE']=str(out/'upload-transient')
    client=ArchiveClient(storage['command'],storage['program_sha256'])
    try:ack=client.put(path);verified=client.verify(sha(path))
    finally:client.close()
    if ack!=verified:raise ValueError('probe archive changed')
    compressed=out/'readback.tar.zst';decoded=out/'readback.tar'
    command=storage['command'][:-1]+[shlex.join(['docker','exec','-i',storage['container_id'],'python3','-c',
        'import sys,shutil;shutil.copyfileobj(open(sys.argv[1],"rb"),sys.stdout.buffer,1024**2)','/data/'+ack['relative_path']])]
    with compressed.open('xb') as stream:subprocess.run(command,stdout=stream,check=True)
    if sha(compressed)!=ack['stored_sha256'] or raw_identity(compressed)!=(sha(path),path.stat().st_size):raise ValueError('actual archive readback hash differs')
    with decoded.open('xb') as stream:subprocess.run(['zstd','-q','-d','-c','--',str(compressed)],stdout=stream,check=True)
    if sha(decoded)!=sha(path):raise ValueError('decompressed tar bytes differ')
    maps=[]
    with tarfile.open(path) as original,tarfile.open(decoded) as replay:
        meta=json.load(original.extractfile('metadata.json'))
        for item in meta['maps']:
            a=original.extractfile(item['member']).read();b=replay.extractfile(item['member']).read()
            if a!=b or hashlib.sha256(b).hexdigest()!=item['sha256'] or gzip.decompress(a)!=gzip.decompress(b):raise ValueError('map bytes changed')
            target=out/(item['metric'].replace('@','_')+'.pfm.gz');target.write_bytes(b)
            validate_map(target,meta['width'],meta['height'])
            maps.append(dict(metric=item['metric'],compressed_map_sha256=item['sha256'],uncompressed_pfm_sha256=hashlib.sha256(gzip.decompress(b)).hexdigest(),map_owner_readback_passed=True))
    result=dict(schema='e36-zstd-actual-map-readback-v1',admitted=True,bundle_sha256=sha(path),bundle_bytes=path.stat().st_size,
        archive=ack,storage=dict(path=str(storage_path),sha256=sha(storage_path)),maps=maps,
        map_owner=dict(path=str(Path(validate_map.__code__.co_filename)),sha256=sha(validate_map.__code__.co_filename)),
        retained_raw_byte_identity=True,utc=time.time(),fits=0)
    with (out/'receipt.json').open('x') as stream:json.dump(result,stream,indent=2)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','bundle','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(json.dumps(probe(args.root,args.bundle,args.out)))
