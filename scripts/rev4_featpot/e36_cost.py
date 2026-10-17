"""Measured E36 stage census. Reused targets never duplicate stage costs."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

from e36_store import sha


def census(root,out):
    groups=defaultdict(list);inputs=[]
    for name in ['cost-v4','square-v4','modes-v4','max-area-v4']:
        run=root/'runs'/name;receipt=json.loads((run/'census.json').read_text())
        if not receipt['admitted']:raise ValueError('incomplete preflight')
        inputs.append(dict(name=name,census_sha256=sha(run/'census.json')))
        for item in receipt['output_receipts']:
            p=Path(item['path'])
            if sha(p)!=item['sha256']:raise ValueError('preflight output changed')
            d=json.loads(gzip.decompress(p.read_bytes()))
            for r in d['rows']:groups[(name,r['size'],r['codec'])].append(r)
    data=[]
    for (run,size,codec),rows in sorted(groups.items()):
        physical=[r for r in rows if not r['physical_alias']]
        # V4 baselines really execute every requested q. Mode aliases instead
        # reuse exact same encode command receipt (argv+timing); count it once.
        encodes={json.dumps(r['encode_execution'],sort_keys=True):r['encode_execution'] for r in rows}
        stages={"encode":list(encodes.values()),"scalars":[r['score_execution'] for r in physical],
            "features":[r['feature_execution'] for r in physical],"maps":[m['execution'] for r in physical for m in r['maps']],
            "independent_inspection":[r['oracle_execution'] for r in physical]}
        measured={k:dict(n=len(v),median_seconds=statistics.median(x['wall_seconds'] for x in v),
            sum_seconds=sum(x['wall_seconds'] for x in v),peak_rss_kib=max(x['peak_rss_kib'] for x in v)) for k,v in stages.items()}
        data.append(dict(run=run,size=size,codec=codec,logical=len(rows),unique_variants=len(physical),stages=measured,
            durable_put_seconds=sum(r['artifact_persistence_seconds'] for r in physical),
            unique_complete_cell_median_seconds=statistics.median(r['cell_total_seconds'] for r in physical),
            unique_complete_cell_n=len(physical),
            total_bytes=sum(r['bundle']['bytes'] for r in physical),
            scope='V4 alias cell_total_seconds is inherited and INVALID: excluded. Native command timings are genuine; mode PNG-convert timing remains in archived diagnostics. Unique complete cells include Python/storage work. No full-grid forecast.'))
    with out.open('x') as f:json.dump(dict(schema='e36-measured-stage-cost-v1',inputs=inputs,rows=data,fits=0),f,indent=2);f.write('\n')
    print(json.dumps(dict(groups=len(data),fits=0,output=str(out))))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();census(a.root,a.out)
