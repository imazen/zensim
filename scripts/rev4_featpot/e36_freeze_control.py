"""Freeze control source bytes independently of immutable numerical executors."""
import argparse
import json
from pathlib import Path
import shutil
from e36_store import sha
from e36_artifact_location import durable_mkdir,sync_directory


def freeze(out,commit):
    out=Path(out)
    if out.exists():raise ValueError('preserve existing E36 control source snapshot')
    durable_mkdir(out)
    for source in Path(__file__).parent.glob('*.py'):shutil.copyfile(source,out/source.name)
    files={path.name:dict(path=str(path),sha256=sha(path)) for path in sorted(out.glob('*.py'))}
    with (out/'sources.json').open('x') as stream:json.dump(dict(schema='e36-frozen-control-sources-v1',build_commit=commit,files=files,fits=0),stream,indent=2);stream.write('\n');stream.flush();__import__('os').fsync(stream.fileno())
    sync_directory(out);return dict(path=str(out/'sources.json'),sha256=sha(out/'sources.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--out',type=Path,required=True);parser.add_argument('--commit',required=True);a=parser.parse_args();print(json.dumps(freeze(a.out,a.commit)))
