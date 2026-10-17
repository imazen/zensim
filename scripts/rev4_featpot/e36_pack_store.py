"""Sequential lossless E36 bundle packs; standalone storage-only verifier."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import uuid

HEX=re.compile(r'^[0-9a-f]{64}$')
CHUNK=1024**2


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(CHUNK):h.update(chunk)
    return h.hexdigest()


class HashReader:
    def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256();self.size=0
    def read(self,size=-1):
        if size<0:size=CHUNK
        data=self.stream.read(size);self.hash.update(data);self.size+=len(data);return data


class HashWriter:
    def __init__(self,stream):self.stream=stream;self.hash=hashlib.sha256();self.size=0
    def write(self,data):
        self.stream.write(data);self.hash.update(data);self.size+=len(data);return len(data)


def build(stream,items):
    target=HashWriter(stream)
    with tarfile.open(fileobj=target,mode='w|',format=tarfile.USTAR_FORMAT,bufsize=CHUNK) as archive:
        for item in items:
            h=item['sha256'];path=Path(item['local_path'])
            if not HEX.fullmatch(h):raise ValueError('invalid bundle hash')
            info=tarfile.TarInfo(h+'.tar');info.size=item['bytes'];info.mode=0o444
            with path.open('rb') as source:
                reader=HashReader(source);archive.addfile(info,reader)
                if reader.size!=info.size or reader.hash.hexdigest()!=h or source.read(1):raise ValueError('bundle changed during pack write')
    return dict(stored_sha256=target.hash.hexdigest(),stored_bytes=target.size)


def verify(path):
    members={}
    with Path(path).open('rb') as source:
        reader=HashReader(source)
        with tarfile.open(fileobj=reader,mode='r|',bufsize=CHUNK) as archive:
            for member in archive:
                h=member.name.removesuffix('.tar')
                if not member.isfile() or member.name!=h+'.tar' or not HEX.fullmatch(h) or h in members:raise ValueError('invalid pack member')
                value=hashlib.sha256();size=0
                content=archive.extractfile(member)
                while chunk:=content.read(CHUNK):value.update(chunk);size+=len(chunk)
                if size!=member.size or value.hexdigest()!=h:raise ValueError('raw bundle hash differs after actual pack readback')
                members[h]=size
        while reader.read(CHUNK):pass
    if not members:raise ValueError('empty pack')
    return dict(members=members,stored_sha256=reader.hash.hexdigest(),stored_bytes=reader.size,
                verifier_sha256=sha(__file__),encoding='tar-pack-v1')


def receive(root,pack_id,size,expected):
    if not HEX.fullmatch(pack_id) or (expected!='-' and not HEX.fullmatch(expected)) or not 0<size<=2*1024**3:raise ValueError('invalid pack frame')
    root=Path(root);temporary=root/'.pack-incoming'/(pack_id+'-'+uuid.uuid4().hex+'.partial')
    temporary.parent.mkdir(parents=True,exist_ok=True)
    with temporary.open('xb') as target:
        remaining=size;h=hashlib.sha256()
        while remaining:
            data=sys.stdin.buffer.read(min(CHUNK,remaining))
            if not data:raise EOFError('incomplete pack retained')
            target.write(data);h.update(data);remaining-=len(data)
        target.flush();os.fsync(target.fileno())
    if expected!='-' and h.hexdigest()!=expected:raise ValueError('received pack hash differs')
    final=root/'packs'/(pack_id+'.tar');final.parent.mkdir(parents=True,exist_ok=True)
    try:os.link(temporary,final)
    except FileExistsError:
        if sha(final)!=h.hexdigest():raise ValueError('pack collision')
    fd=os.open(final.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    temporary.unlink();return verify(final)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['verify','receive']);parser.add_argument('path');parser.add_argument('--pack-id');parser.add_argument('--size',type=int);parser.add_argument('--sha256')
    args=parser.parse_args()
    result=verify(args.path) if args.operation=='verify' else receive(args.path,args.pack_id,args.size,args.sha256)
    print(json.dumps(result,separators=(',',':')),flush=True)


if __name__=='__main__':main()
