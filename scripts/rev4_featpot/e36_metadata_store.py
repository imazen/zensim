"""Opaque metadata archive receiver; hashes every member without interpreting data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tarfile
import uuid

CHUNK=1024**2
HEX=re.compile('[0-9a-f]{64}')


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def verify(path):
    members={}
    with tarfile.open(path,mode='r|',bufsize=CHUNK) as archive:
        for member in archive:
            value=member.name.removesuffix('.blob')
            if not member.isfile() or member.name!=value+'.blob' or HEX.fullmatch(value) is None or value in members:raise ValueError('invalid metadata member')
            source=archive.extractfile(member);h=hashlib.sha256();size=0
            while block:=source.read(CHUNK):h.update(block);size+=len(block)
            if size!=member.size or h.hexdigest()!=value:raise ValueError('metadata member hash differs')
            members[value]=size
    return dict(sha256=sha(path),bytes=Path(path).stat().st_size,members=members,program_sha256=sha(__file__))


def receive(root,value,size):
    if HEX.fullmatch(value) is None or not 0<size<=64*1024**3:raise ValueError('invalid metadata archive declaration')
    root=Path(root);root.mkdir(exist_ok=True);incoming=root/'.incoming';incoming.mkdir(exist_ok=True);sync(root.parent);sync(root)
    if shutil.disk_usage(root).free<size+128*1024**3:raise OSError('array free floor refuses metadata archive')
    temporary=incoming/(value+'-'+uuid.uuid4().hex+'.partial');remaining=size
    with temporary.open('xb') as target:
        while remaining:
            block=sys.stdin.buffer.read(min(CHUNK,remaining))
            if not block:raise EOFError('incomplete metadata archive retained')
            target.write(block);remaining-=len(block)
        target.flush();os.fsync(target.fileno())
    result=verify(temporary);final=root/(value+'.tar')
    try:os.link(temporary,final)
    except FileExistsError:
        if sha(final)!=result['sha256']:raise ValueError('metadata archive collision')
    sync(root);temporary.unlink();sync(incoming);return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['receive','verify']);parser.add_argument('path');parser.add_argument('--id');parser.add_argument('--size',type=int);a=parser.parse_args()
    print(json.dumps(receive(a.path,a.id,a.size) if a.operation=='receive' else verify(a.path)),flush=True)
