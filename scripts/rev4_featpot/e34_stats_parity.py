"""Compare independent Python and Rust E33 seed-unit receipts, no new data."""
import json
from pathlib import Path

root=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part5')
a=json.loads((root/'python-seed-parity.json').read_text())
b=json.loads((root/'rust-seed-parity.json').read_text())
errors=[]
def check(a,b,key=''):
    if isinstance(a,dict):
        assert a.keys()==b.keys(),key
        for k in a:check(a[k],b[k],key+'.'+k)
    elif isinstance(a,list):
        assert len(a)==len(b),key
        for i,(v,w) in enumerate(zip(a,b,strict=True)):check(v,w,key+f'[{i}]')
    elif isinstance(a,(float,int)):
        error=abs(a-b);errors.append((error,key))
        assert error<1e-12,(key,a,b)
    else:assert a==b,key
check(a,b)
receipt=dict(state='PASS',maximum_absolute_error=max(errors)[0],worst_coordinate=max(errors)[1],numeric_comparisons=len(errors))
with (root/'seed-parity-check.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
