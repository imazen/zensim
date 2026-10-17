"""Lossless archive diagnostic on immutable local bundles; no data admission."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time

from e36_artifact_location import access


def probe(root,out):
    out.mkdir()
    candidates={}
    # A bounded diagnostic sample of existing files, not a population sample.
    for prefix in sorted((root/'artifacts-local-v1/blobs').iterdir()):
        for path in prefix.iterdir():
            size=path.stat().st_size
            group='small' if size<2*1024**2 else 'mid' if size<32*1024**2 else 'large'
            candidates.setdefault(group,path)
            if len(candidates)==3:break
        if len(candidates)==3:break
    rows=[]
    for group,path in sorted(candidates.items()):
        digest=path.stem
        with access(root,digest):
            if not path.exists():continue
            begin=time.monotonic();raw=hashlib.sha256();size=0
            target=out/(digest+'.tar.gz')
            with path.open('rb') as source,target.open('xb') as output:
                with gzip.GzipFile(filename='',fileobj=output,mode='wb',compresslevel=1,mtime=0) as zipped:
                    while data:=source.read(1024**2):
                        raw.update(data);size+=len(data);zipped.write(data)
            elapsed=time.monotonic()-begin
        if raw.hexdigest()!=digest:raise ValueError('source bundle changed')
        verify=hashlib.sha256();count=0;begin=time.monotonic()
        with gzip.open(target,'rb') as stream:
            while data:=stream.read(1024**2):verify.update(data);count+=len(data)
        if count!=size or verify.hexdigest()!=digest:raise ValueError('lossless archive readback differs')
        with target.open('rb') as stream:stored=hashlib.file_digest(stream,'sha256').hexdigest()
        rows.append(dict(source=str(path),source_sha256=digest,bytes=size,sample_group=group,
            compressed_path=str(target),compressed_sha256=stored,compressed_bytes=target.stat().st_size,
            compress_seconds=elapsed,verify_seconds=time.monotonic()-begin,raw_sha_readback_matches=True))
    result=dict(schema='e36-lossless-archive-probe-v1',rows=rows,fits=0,labels_read=False,
        qualifies_data=False,basis='At most one first-encountered immutable local bundle per file-size bucket. Diagnostic only; no transfer to unmeasured content or sizes, no production format change.')
    with (out/'receipt.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['root','out']:parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();probe(args.root,args.out)
