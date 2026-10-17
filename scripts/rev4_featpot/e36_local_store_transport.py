"""Strict local transport for the unchanged frozen storage protocol on its host."""
import hashlib
import json
import os
from pathlib import Path
import sys


def main():
    spec=json.loads(Path(os.environ['E36_STORAGE_SPEC']).read_text())['storage']
    if sys.argv[1:]!=spec['command'][1:]:raise ValueError('local transport refuses nonregistered SSH argv')
    server=Path('/data/program')/(spec['program_sha256']+'.py')
    if hashlib.sha256(server.read_bytes()).hexdigest()!=spec['program_sha256']:raise ValueError('local canonical storage owner changed')
    os.execv(sys.executable,[sys.executable,'-u',str(server),'serve','--root','/data','--min-free-gib','128'])


if __name__=='__main__':main()
