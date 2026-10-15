"""Bind final local E34 preparation receipts and preserve command resource logs."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from e34_augmentation import sha,write

ROOT=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5')
P4=ROOT/'p4-inventory-r2'

def main():
    logs=ROOT/'run-heavy-logs';logs.mkdir()
    resources=[]
    for p in sorted(Path('/home/lilith/tmp/e34prep').glob('p5-*.log')):
        content=p.read_text(errors='replace')
        lines=re.findall(r'run-heavy: done[^\n]*',content)
        assert lines,p
        dst=logs/p.name
        assert not dst.exists()
        shutil.copyfile(p,dst)
        resources.append(dict(log=p.name,sha256=sha(dst),resource=lines[-1]))
    write(ROOT/'resources.json',resources)
    bindings={}
    for rel in ['python-seed-parity.json','rust-seed-parity.json','rust-seed-parity-final.json','bin/archive.json','seed-parity-check.json','noise-capture.json','noise-result.json',
                'p4-inventory-r2/inventory.json','p4-inventory-r2/eligibility.json','p4-inventory-r2/selection.json',
                'p4-inventory-r2/encodes/tools.json','p4-inventory-r2/encodes/augmentation/bases.json',
                'p4-inventory-r2/encodes/scalar-canary/comparison.json','p4-inventory-r2/development/panel.json','p4-inventory-r2/development/assessment-inputs.json',
                'p4-inventory-r2/repairs/augmentation.parquet.manifest.json','p4-inventory-r2/parity/comparison.json',
                'p4-inventory-r2/repairs/rank-inputs/augmentation.parquet.manifest.json','smokes/complete/k-a.smoke.json','smokes/complete/k-c.smoke.json','smokes/k-a.smoke.json','resources.json']:
        p=ROOT/rel;bindings[rel]=dict(bytes=p.stat().st_size,sha256=sha(p))
    write(ROOT/'part5-receipt.json',dict(source_revision=subprocess.check_output(['jj','log','-r','@-','--no-graph','-T','commit_id'],text=True).strip(),schema='e34-preparation-part5-v1',labels_read='one authorized original E33 E21 capture; P4 pixels/features only; K smokes retain strict TRAIN',bindings=bindings,
        source_limit='existing byte-pinned zenmetrics executable source closure was not recovered',
        not_qualified=['D12 sensitivity decision pending','fleet smokes/caps and image/jobset pending','no full fit, lambda selection, new E21 assessment or model adoption']))
    print(json.dumps(dict(receipts=len(bindings),resource_logs=len(resources))),flush=True)
if __name__=='__main__':main()
