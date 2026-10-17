"""Recover pre-intent canonical archive ownership without copying or deleting."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time

from e36_archive_assignments import Assignments
from e36_store import digest,sha


def census(root,out):
    root=Path(root);out=Path(out);out.mkdir();index=Assignments(root);stores={};physical={};pins=[]
    code='''import os,re,sys
root='/data/blobs'
if os.path.isdir(root):
 for prefix in os.scandir(root):
  if not prefix.is_dir():continue
  for item in os.scandir(prefix.path):
   name=item.name
   if re.fullmatch('[0-9a-f]{64}'+re.escape(sys.argv[1]),name) and item.is_file():print(name.split('.')[0])
'''
    for name in ['storage-runtime.json','storage-archive-v2.json','storage-cache-v1.json']:
        path=root/name;storage=json.loads(path.read_text());key=storage['root'];stores[key]=storage
        command=['ssh','-T','-oBatchMode=yes','-oConnectTimeout=10',storage['host']]
        info=json.loads(subprocess.check_output(command+[shlex.join(['docker','inspect',storage['container_id']])]))[0]
        if info['Image']!=storage['image_id'] or not info['State']['Running'] or not any(m['Source']==key and m['Destination']=='/data' for m in info['Mounts']):raise ValueError('actual archive container/image/mount differs')
        capture=out/(name+'.hashes.txt')
        with capture.open('xb') as stream:subprocess.run(command+[shlex.join(['docker','exec',storage['container_id'],'python3','-c',code,'.tar.zst' if storage.get('encoding')=='zstd' else '.tar'])],stdout=stream,check=True)
        values=[digest(line) for line in capture.read_text().splitlines()]
        if len(values)!=len(set(values)):raise ValueError('duplicate physical canonical hash in one store')
        physical[key]=set(values);pins.append(dict(path=str(capture),sha256=sha(capture),objects=len(values)))
    cache_root=json.loads((root/'storage-cache-v1.json').read_text())['root'];array=set().union(*(v for k,v in physical.items() if k!=cache_root))
    overlap=sorted(array & physical[cache_root])
    if overlap:
        (out/'DUPLICATE_CACHE_ARRAY.json').write_text(json.dumps(dict(hashes=overlap,fits=0),indent=2)+'\n')
        raise ValueError('finished bundles exist in both cache and array; preserve all copies and report')
    prior={}
    with (root/'owner-svt-ssd-v1/artifact-relocations.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line);prior[row['sha256']]=row['canonical_root']
    rows=[]
    for h in sorted(set().union(*physical.values())):
        current=index.get(h);keys=[key for key,values in physical.items() if h in values]
        key=current['canonical_root'] if current is not None else prior.get(h,keys[0])
        if key not in keys:raise ValueError('prior immutable assignment conflicts with physical canonical owner')
        storage=stores[key];rows.append(dict(sha256=h,canonical_host=storage['host'],canonical_root=key,
            program_sha256=storage['program_sha256'],physical_seen=True,source='pre-cache-recovery-census',utc=time.time(),fits=0))
    claims=root/'owner-svt-ssd-v1/cache-reservations-v1.jsonl'
    if claims.exists():
        for line in claims.read_text().splitlines():
            item=json.loads(line);h=item['sha256']
            if h in array:raise ValueError('cache assignment conflicts with earlier array publication')
            if h not in physical[cache_root]:
                storage=stores[cache_root];rows.append(dict(sha256=h,canonical_host=storage['host'],canonical_root=cache_root,program_sha256=storage['program_sha256'],source='prior-cache-reservation',fits=0))
    index.seed(rows)
    result=dict(schema='e36-prior-archive-ownership-census-v1',actual_stores=pins,cache_array_duplicate_hashes=overlap,
        recovered_assignments=len(rows),array_hashes=len(array),cache_hashes=len(physical[cache_root]),
        source=dict(path=str(Path(__file__)),sha256=sha(__file__)),fits=0,utc=time.time())
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();census(args.root,args.out)
