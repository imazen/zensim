"""Admit the real feature-only pool with the final CLI, without an optimizer."""
import argparse
import json
from pathlib import Path
import subprocess
from e36_store import sha


def run(binary,pool,out):
    out.mkdir();manifest=pool/'manifest.json';contract=pool/'contract.json'
    m=json.loads(manifest.read_text());c=json.loads(contract.read_text())
    if c['manifest_sha256']!=sha(manifest) or c['fits_released'] is not False:
        raise ValueError('exact unreleased native pool contract required')
    semantic=json.loads(Path(c['semantic_export']['path']).read_text())
    owner=Path(__file__).parents[1]/'canonical_corpus/build_ladder_tv_pairs.py'
    if semantic['consensus_source_sha256']!=sha(owner):raise ValueError('registered consensus source changed')
    argv=[str(binary),'--e36-tv-admit','--e36-tv-manifest',str(manifest),
          '--e36-tv-contract',str(contract),'--e36-tv-contract-sha256',sha(contract),
          '--derived-inputs',m['derived']['path'],'--max-features','1853','--tv-weight','1']
    with (out/'stdout.jsonl').open('x') as stdout,(out/'stderr.log').open('x') as stderr:
        subprocess.run(['/usr/bin/time','-v','-o',str(out/'RESOURCE_USAGE.txt'),*argv],
                       stdout=stdout,stderr=stderr,check=True)
    rows=[json.loads(line) for line in (out/'stdout.jsonl').read_text().splitlines() if line.startswith('{')]
    if len(rows)!=1:raise ValueError('missing/ambiguous native admission result')
    r=rows[0]
    if (r.get('schema')!='e36-tv-admission-v1' or r.get('feature_rows')!=m['feature_rows']
            or r.get('pairs')!=m['pair_rows'] or r.get('dropped_pairs')!=0
            or r.get('optimizer_steps')!=0 or r.get('labels_read') is not False):
        raise ValueError('actual native pool admission incomplete')
    result=dict(schema='e36-actual-pool-admission-v1',binary_sha256=sha(binary),manifest_sha256=sha(manifest),
        contract_sha256=sha(contract),consensus_source_sha256=sha(owner),argv=argv,native=r,
        resource_receipt_sha256=sha(out/'RESOURCE_USAGE.txt'),labels_read=False,optimizer_steps=0)
    with (out/'receipt.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['binary','pool','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();run(a.binary,a.pool,a.out)
