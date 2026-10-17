"""Private/normal CLI boundaries and invented-row raw forward; no optimizer."""
import argparse
import json
from pathlib import Path
import subprocess
import math
from e36_store import sha

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--private',type=Path,required=True);p.add_argument('--normal',type=Path,required=True);p.add_argument('--raw',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir()
    fixture=a.root/'ingress-fixtures-v4';contract=fixture/'valid-contract.json'
    argv=['--e36-tv-admit','--e36-tv-manifest',str(fixture/'valid.json'),'--e36-tv-contract',str(contract),'--e36-tv-contract-sha256',sha(contract),'--derived-inputs',str(fixture/'fx1.json'),'--max-features','1853','--tv-weight','1']
    normal=subprocess.run([str(a.normal),*argv],text=True,capture_output=True)
    (a.out/'normal.stderr').write_text(normal.stderr)
    if normal.returncode==0 or 'explicit private build' not in normal.stderr:raise ValueError('normal build failed to refuse E36 at its protocol boundary')
    private=subprocess.run([str(a.private),'--group','invented:never-open:1:0:rank','--out',str(a.out/'never-created.bin'),'--tv-pairs-file',str(a.out/'never-open.tsv')],text=True,capture_output=True)
    (a.out/'private-legacy.stderr').write_text(private.stderr)
    if private.returncode==0 or 'private build refuses legacy TV pair files' not in private.stderr:raise ValueError('private build failed to refuse legacy TV inputs')
    spec=json.loads((a.root/'e34-feature-spec.json').read_text());layout=json.loads((a.root/'FEATURE_LAYOUT_BINDING.json').read_text())
    rows=[dict(row_idx=i,width=32,height=32,features=[i*.1]*420) for i in range(3)]
    header=dict(bake=spec['model'],bake_sha256=spec['model_sha256'],feature_ids=layout['consumed_ids'])
    raw=subprocess.run([str(a.raw)],input='\n'.join(json.dumps(v) for v in [header,*rows])+'\n',text=True,capture_output=True)
    (a.out/'raw.stdout').write_text(raw.stdout);(a.out/'raw.stderr').write_text(raw.stderr)
    if raw.returncode:raise ValueError('invented-row raw forward failed')
    values=[json.loads(l) for l in raw.stdout.splitlines()]
    if [v['row_idx'] for v in values]!=list(range(3)) or not all(math.isfinite(v['raw_distance']) for v in values):raise ValueError('invented-row raw forward is unordered/nonfinite')
    with (a.out/'receipt.json').open('x') as f:json.dump(dict(schema='e36-nontraining-mode-admission-v1',normal_refuses_e36=True,private_refuses_legacy=True,invented_feature_rows=3,optimizer_steps=0,labels_read=False,
        private_binary_sha256=sha(a.private),normal_binary_sha256=sha(a.normal),raw_binary_sha256=sha(a.raw),invented_only=True),f,indent=2);f.write('\n')
