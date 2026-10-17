"""Lossless zstd archive transport; raw-byte hashes remain the object identity."""
import argparse
import ctypes as c
import ctypes.util
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid

from e36_store import Client, digest, response, sha
from e36_artifact_location import durable_mkdir, sync_directory


class Buffer(c.Structure):
    _fields_=[('data',c.c_void_p),('size',c.c_size_t),('pos',c.c_size_t)]


def raw_identity(path):
    """Bounded streaming libzstd decode, rejecting truncation/trailing junk."""
    lib=c.CDLL(ctypes.util.find_library('zstd'))
    lib.ZSTD_createDStream.restype=c.c_void_p
    for name,args in [('ZSTD_initDStream',[c.c_void_p]),('ZSTD_freeDStream',[c.c_void_p]),
                      ('ZSTD_decompressStream',[c.c_void_p,c.POINTER(Buffer),c.POINTER(Buffer)])]:
        getattr(lib,name).argtypes=args;getattr(lib,name).restype=c.c_size_t
    lib.ZSTD_isError.argtypes=[c.c_size_t];lib.ZSTD_isError.restype=c.c_uint
    stream=lib.ZSTD_createDStream()
    if not stream:raise MemoryError('zstd decoder allocation failed')
    h=hashlib.sha256();size=0;remaining=1
    try:
        if lib.ZSTD_isError(lib.ZSTD_initDStream(stream)):raise ValueError('zstd initialization failed')
        with Path(path).open('rb') as source:
            while chunk:=source.read(1024**2):
                data=c.create_string_buffer(chunk);incoming=Buffer(c.addressof(data),len(chunk),0)
                while incoming.pos<incoming.size:
                    target=c.create_string_buffer(1024**2);out=Buffer(c.addressof(target),1024**2,0)
                    before=incoming.pos
                    remaining=lib.ZSTD_decompressStream(stream,c.byref(out),c.byref(incoming))
                    if lib.ZSTD_isError(remaining):raise ValueError('invalid zstd frame')
                    h.update(target.raw[:out.pos]);size+=out.pos
                    if before==incoming.pos and not out.pos:raise ValueError('stalled zstd frame')
        if remaining!=0:raise ValueError('truncated zstd archive')
        return h.hexdigest(),size
    finally:lib.ZSTD_freeDStream(stream)


def verify(root,expected,program):
    path=root/'blobs'/expected[:2]/(expected+'.tar.zst')
    actual,size=raw_identity(path)
    if actual!=expected:raise ValueError('archive decompressed byte hash differs')
    return dict(ok=True,sha256=actual,bytes=size,relative_path=str(path.relative_to(root)),
                encoding='zstd',stored_sha256=sha(path),stored_bytes=path.stat().st_size,program_sha256=program)


def serve(root,source,target,min_free,reserve_path=None):
    root=Path(root).resolve(strict=True);incoming=root/'.incoming';durable_mkdir(incoming)
    program=sha(__file__)
    while line:=source.readline(4097):
        if len(line)>4096 or not line.endswith(b'\n'):raise ValueError('invalid archive frame')
        header=json.loads(line);expected=digest(header['sha256'])
        if header['operation']=='verify':response(target,verify(root,expected,program));continue
        if header['operation']!='put':raise ValueError('unsupported archive operation')
        size=header['stored_bytes'];raw_size=header['bytes'];stored_sha=digest(header['stored_sha256'])
        if type(size) is not int or not 0<size<=2*1024**3 or type(raw_size) is not int or not 0<raw_size<=1024**3:
            raise ValueError('invalid archive size')
        if shutil.disk_usage(reserve_path or root).free<min_free+size:raise OSError('archive reserve refuses write')
        path=incoming/(expected+'-'+uuid.uuid4().hex+'.partial');remaining=size;h=hashlib.sha256()
        with path.open('xb') as stream:
            while remaining:
                data=source.read(min(1024**2,remaining))
                if not data:raise EOFError('incomplete archive retained')
                stream.write(data);h.update(data);remaining-=len(data)
            stream.flush();os.fsync(stream.fileno())
        if h.hexdigest()!=stored_sha or raw_identity(path)!=(expected,raw_size):
            raise ValueError('archive stored/decompressed hash differs; partial retained')
        final=root/'blobs'/expected[:2]/(expected+'.tar.zst');durable_mkdir(final.parent)
        try:os.link(path,final)
        except FileExistsError:
            if verify(root,expected,program)['bytes']!=raw_size:raise ValueError('archive collision')
        sync_directory(final.parent);path.unlink();sync_directory(incoming)
        response(target,verify(root,expected,program))


class ArchiveClient(Client):
    def hold(self,reason):
        root=Path(os.environ['E36_ARCHIVE_ROOT'])
        scope=Path('/sys/fs/cgroup'+Path('/proc/self/cgroup').read_text().strip().split('::')[1])
        from e36_generation_resume import hold_admission
        hold_admission(root,scope,root/'bin/zenfleet-worker-fair-v7',reason)

    def disconnect(self):
        """EOF and join the owned transfer; never terminate it or replace it live."""
        process=self.process
        if process is None:return None
        try:process.stdin.close()
        except OSError:pass
        while True:
            try:code=process.wait(timeout=30);break
            except subprocess.TimeoutExpired:
                # Retain this exact handle and raw original. The existing
                # monitor reconciles the admission hold; no new stream starts.
                try:self.hold('archive transport failed to drain cooperatively')
                except Exception as error:
                    print('E36 archive drain hold persistence failed; retaining owned transfer: '+repr(error),file=sys.stderr,flush=True)
        process.stdout.close();self.process=None
        return code

    def close(self):
        code=self.disconnect()
        if code not in [None,0]:raise RuntimeError('archive process exited '+str(code))

    def put(self,path):
        path=Path(path);expected=sha(path);size=path.stat().st_size
        stage=Path(os.environ['E36_ARCHIVE_STAGE']);durable_mkdir(stage)
        compressed=stage/(expected+'-'+uuid.uuid4().hex+'.zst')
        with compressed.open('xb') as output:
            subprocess.run(['zstd','-q','-1','-T1','-c','--',str(path)],stdout=output,check=True)
        try:
            if compressed.stat().st_size>getattr(self,'max_stored_bytes',2*1024**3):
                raise ValueError('compressed archive exceeds its pre-upload reservation')
            header=dict(operation='put',sha256=expected,bytes=size,stored_sha256=sha(compressed),stored_bytes=compressed.stat().st_size)
            result=self.exchange(header,compressed)
            if result['bytes']!=size or result['encoding']!='zstd' or result['stored_sha256']!=header['stored_sha256']:
                raise ValueError('archive acknowledgement differs')
            return result
        finally:
            # This transient is redundant with the protected raw original.
            compressed.unlink();sync_directory(stage)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--min-free-gib',type=int,required=True)
    parser.add_argument('--reserve-path',type=Path)
    args=parser.parse_args()
    if args.min_free_gib<=0:raise ValueError('positive archive reserve required')
    serve(args.root,sys.stdin.buffer,sys.stdout.buffer,args.min_free_gib*1024**3,args.reserve_path)
