"""Opaque metadata archive receiver; hashes every member without interpreting data."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
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
    members={};index=None
    with tarfile.open(path,mode='r|',bufsize=CHUNK) as archive:
        for member in archive:
            value=member.name.removesuffix('.blob')
            is_index=member.name=='INDEX.json'
            if not member.isfile() or (not is_index and (member.name!=value+'.blob' or HEX.fullmatch(value) is None)) or (is_index and index is not None) or value in members:raise ValueError('invalid metadata member')
            source=archive.extractfile(member);h=hashlib.sha256();size=0
            while block:=source.read(CHUNK):h.update(block);size+=len(block)
            if size!=member.size or (not is_index and h.hexdigest()!=value):raise ValueError('metadata member hash differs')
            if is_index:value=h.hexdigest();index=value
            if value in members:raise ValueError('duplicate metadata content member')
            members[value]=size
    return dict(sha256=sha(path),bytes=Path(path).stat().st_size,members=members,program_sha256=sha(__file__),index_sha256=index)


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
    sync(root);temporary.unlink();sync(incoming);result.update(destination=str(root.resolve()),hostname=socket.gethostname());return result


def locate(root,value,size):
    if HEX.fullmatch(value) is None or not 0<size<=1024**2:raise ValueError('invalid metadata locator')
    root=Path(root);body=sys.stdin.buffer.read(size)
    if len(body)!=size or hashlib.sha256(body).hexdigest()!=value:raise ValueError('metadata locator hash differs')
    document=json.loads(body);index_id=document['index_id']
    if HEX.fullmatch(index_id) is None:raise ValueError('invalid final index archive identity')
    index=root/(index_id+'.tar');actual=verify(index)
    if actual['sha256']!=document['index_sha256'] or actual['index_sha256']!=document['descriptor_sha256']:raise ValueError('locator does not identify verified restorable index')
    destination=root/('LOCATOR-'+value+'.json')
    with destination.open('xb') as stream:stream.write(body);stream.flush();os.fsync(stream.fileno())
    sync(root)
    return dict(sha256=sha(destination),bytes=len(body),relative_path=destination.name,destination=str(root.resolve()),hostname=socket.gethostname(),program_sha256=sha(__file__))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('operation',choices=['receive','verify','locate']);parser.add_argument('path');parser.add_argument('--id');parser.add_argument('--size',type=int);a=parser.parse_args()
    print(json.dumps(receive(a.path,a.id,a.size) if a.operation=='receive' else locate(a.path,a.id,a.size) if a.operation=='locate' else verify(a.path)),flush=True)
