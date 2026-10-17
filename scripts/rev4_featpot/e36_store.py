"""Streaming content-addressed bundle storage; no job scheduling or claims.

The server runs in a capped storage-only container. Its caller owns transport.
Incomplete and corrupt uploads remain named diagnostics. Canonical objects are
never overwritten; a successful acknowledgement follows byte-hash verification.
"""
import argparse
import fcntl
import time
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    if not isinstance(value, str) or re.fullmatch('[0-9a-f]{64}', value) is None:
        raise ValueError('expected a lowercase SHA256, not a path')
    return value


def response(stream, value):
    stream.write(json.dumps(value, separators=(',', ':')).encode() + b'\n')
    stream.flush()


def serve(root, source, target, min_free_bytes=0):
    root = Path(root).resolve(strict=True)
    incoming = root / '.incoming'
    incoming.mkdir(exist_ok=True)
    program_sha = sha(__file__)
    while True:
        line = source.readline(4097)
        if not line:
            return
        if len(line) > 4096 or not line.endswith(b'\n'):
            raise ValueError('invalid storage frame header')
        header = json.loads(line)
        expected = digest(header['sha256'])
        final = root / 'blobs' / expected[:2] / (expected + '.tar')
        if header['operation'] == 'verify':
            actual = sha(final)
            if actual != expected:
                raise ValueError('canonical bundle bytes do not match their content address')
            response(target, dict(ok=True, sha256=actual, bytes=final.stat().st_size,
                relative_path=str(final.relative_to(root)), program_sha256=program_sha))
            continue
        if header['operation'] != 'put':
            raise ValueError('unknown storage operation')
        size = header['bytes']
        if type(size) is not int or not 0 < size <= 1024**3:
            raise ValueError('bundle must be nonempty and at most 1GiB')
        if shutil.disk_usage(root).free < min_free_bytes + size:
            raise OSError('storage free-space floor refuses this complete bundle')
        temporary = incoming / (expected + '-' + uuid.uuid4().hex + '.partial')
        remaining, h = size, hashlib.sha256()
        with temporary.open('xb') as output:
            while remaining:
                data = source.read(min(1024**2, remaining))
                if not data:
                    raise EOFError(f'truncated upload retained at {temporary}')
                output.write(data)
                h.update(data)
                remaining -= len(data)
        if h.hexdigest() != expected:
            failed = temporary.with_suffix('.failed-sha256')
            temporary.rename(failed)
            raise ValueError(f'corrupt upload retained at {failed}')
        final.parent.mkdir(parents=True, exist_ok=True)
        existed = False
        try:
            os.link(temporary, final)
        except FileExistsError:
            existed = True
            if final.stat().st_size != size or sha(final) != expected:
                raise ValueError('refusing to replace a conflicting canonical object')
        # The canonical hardlink owns the same verified bytes before this
        # redundant incoming link disappears. This is not artifact eviction.
        temporary.unlink()
        response(target, dict(ok=True, sha256=expected, bytes=size, existed=existed,
            relative_path=str(final.relative_to(root)), program_sha256=program_sha))


class StorageTransportError(ConnectionError):
    pass


class Client:
    def __init__(self, command, program_sha):
        self.command, self.program_sha = command, digest(program_sha)
        self.process = None

    def connect(self):
        if self.process is None:
            gate=os.environ.get('E36_STORE_CONNECT_GATE')
            if gate and Path(self.command[0]).name=='ssh':
                # Pace only connection startup, never storage operations or job
                # claims. Persistent streams remain fully concurrent.
                with Path(gate).open('a+') as lock:
                    fcntl.flock(lock,fcntl.LOCK_EX);lock.seek(0);last=lock.read()
                    delay=.25-(time.monotonic()-float(last or '0'))
                    if delay>0:time.sleep(delay)
                    self.process = subprocess.Popen(self.command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None)
                    lock.seek(0);lock.truncate();lock.write(str(time.monotonic()));lock.flush()
            else:
                self.process = subprocess.Popen(self.command, stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=None)
        return self.process

    def reply(self, expected):
        line = self.process.stdout.readline(4097)
        if not line or len(line) > 4096:
            raise StorageTransportError('storage process returned no bounded acknowledgement')
        value = json.loads(line)
        if value.get('ok') is not True or value['sha256'] != expected or value['program_sha256'] != self.program_sha:
            raise ValueError('storage acknowledgement is not bound to the object/program')
        return value

    def disconnect(self):
        p,self.process=self.process,None
        if p is None:return
        try:p.stdin.close()
        except OSError:pass
        if p.poll() is None:p.terminate()
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:p.kill();p.wait()
        p.stdout.close()

    def exchange(self,header,path=None):
        for attempt in range(3):
            try:
                p=self.connect();p.stdin.write(json.dumps(header).encode()+b'\n')
                if path is not None:
                    with Path(path).open('rb') as f:
                        while data:=f.read(1024**2):p.stdin.write(data)
                p.stdin.flush();return self.reply(header['sha256'])
            except (OSError,StorageTransportError) as exc:
                self.disconnect()
                if attempt==2:raise StorageTransportError('storage transport exhausted3 sends of the same hash-bound bundle') from exc
                sys.stderr.write('E36 storage transport reconnect; same immutable object, prior partial upload retained\n')
                time.sleep(.25*(attempt+1))
        raise AssertionError('unreachable storage exchange')

    def put(self, path):
        path=Path(path);expected,size=sha(path),path.stat().st_size
        answer=self.exchange(dict(operation='put',sha256=expected,bytes=size),path)
        if answer['bytes']!=size:raise ValueError('storage byte count changed')
        return answer

    def verify(self, expected):
        return self.exchange(dict(operation='verify',sha256=digest(expected)))

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            code = self.process.wait(timeout=30)
            if code:
                raise RuntimeError(f'storage process exited {code}')
            self.process = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['serve'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--min-free-gib', required=True, type=int)
    args = parser.parse_args()
    if args.min_free_gib <= 0:
        raise ValueError('a positive storage free-space reserve is required')
    serve(args.root, sys.stdin.buffer, sys.stdout.buffer, args.min_free_gib*1024**3)
