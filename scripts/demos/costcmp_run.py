#!/usr/bin/env python3
"""COSTCMP configuration for SPEEDQ's existing collectors and admission rules."""
import argparse
import hashlib
import json
import math
import struct
from pathlib import Path
import speedq_run as owner

ARMS = ['by_v2fy_r5', 'zensim_A', 'zensim_B', 'fast_ssim2_main', 'fast_ssim2']
TIERS = ['v4x', 'v3']
THREADS = [1, 4, 16, 32]
FAST_MAIN = '09ec3e7c78bd230b1545572a188e162313c59fda'

def cells():
    return [(g, t, n) for t in TIERS for n in THREADS for g in owner.GEOMETRIES]

def signature(rec):
    return (rec['input_sha256'], rec['score_bits'], rec['model'])

def validate_ready(rows, name, geometry, tier, threads, rec):
    old = rows[(geometry, tier, threads, name)]
    assert signature(rec) == signature(old), 'STOP: worker input/score/model differs from COSTCMP preflight'
    assert rec['arm'] == ('by_v2fy' if name == ARMS[0] else name)
    if name == ARMS[0]:
        assert rec['model']['source_sha256'] == owner.SOURCE_SHA
        assert rec['model']['revision'] == '5'
    elif name in ('zensim_A', 'zensim_B'):
        assert rec['model']['revision'] == '1'
        assert len(rec['model']['source_sha256']) == 64
    elif name == 'fast_ssim2_main':
        assert rec['model']['commit'] == FAST_MAIN
    else:
        assert rec['model']['version'] == '0.8.2'

def receipt(path):
    value = json.loads(Path(path).read_text())
    assert value['schema'] == 'costcmp-preflight-v1' and value['status'] == 'PASS'
    rows = {(f"{r['width']}x{r['height']}", r['tier'], int(r['threads']), r['name']): r for r in value['records']}
    assert len(rows) == len(value['records']) == 320
    assert set(rows) == {(g,t,n,a) for g,t,n in cells() for a in ARMS}
    for g in owner.GEOMETRIES:
        production = [rows[(g,t,n,ARMS[0])] for t in TIERS for n in THREADS]
        assert len({r['input_sha256'] for r in production}) == 1
        assert len({r['score_bits'] for r in production}) == 1, 'strict Rev5 score parity differs'
        bits = []
        for rec in production:
            f = rec['feature_values']
            assert len(f) == 420 and all(math.isfinite(v) for v in f)
            bits.append(b''.join(struct.pack('>d',v) for v in f))
        assert len(set(bits)) == 1, 'strict Rev5 consumed-feature parity differs'
    for (g,t,n,a),rec in rows.items():
        assert math.isfinite(rec['score'])
        assert rec['input_sha256'] == rows[(g,t,n,ARMS[0])]['input_sha256']
        validate_ready(rows,a,g,t,n,rec)
    return rows

def production_rows(rows):
    return {(g,t,n,5):rec for (g,t,n,a),rec in rows.items() if a == ARMS[0]}

def preflight(binary, dest):
    root=Path(dest);root.mkdir(parents=True,exist_ok=False);records=[]
    for g,t,n in cells():
        for name in ARMS:
            arm='by_v2fy' if name==ARMS[0] else name
            rec=owner.worker_run(binary,arm,5 if name==ARMS[0] else 1,g,t,n,root/f'{t}-t{n}-{g}-{name}.log')
            rec['name']=name;records.append(rec)
        print(f'preflight {t}-t{n}-{g}',flush=True)
    path=root/'PREFLIGHT_PASS.json'
    owner.write(path,dict(schema='costcmp-preflight-v1',status='PASS',records=records,strict_rev5_cells=64,consumed_features=420))
    try:
        receipt(path)
    except Exception:
        path.rename(root/'PREFLIGHT_FAILED.json')
        raise

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['parity','timing','rss','status'])
    ap.add_argument('--binary',type=Path);ap.add_argument('--dest',type=Path,required=True)
    ap.add_argument('--parity',type=Path);ap.add_argument('--analyzer',type=Path)
    ap.add_argument('--lock',type=Path);ap.add_argument('--only')
    args=ap.parse_args()
    if args.mode=='status':
        owner.collection_status(args.dest,expected_timing=64,expected_rss=80);return
    if args.binary is None:ap.error('--binary is required')
    if args.mode=='parity':preflight(args.binary,args.dest);return
    if args.parity is None:ap.error('--parity is required')
    rows=receipt(args.parity)
    check=lambda name,g,t,n,rec:validate_ready(rows,name,g,t,n,rec)
    loader=lambda path:production_rows(receipt(path))
    if args.mode=='timing':
        if args.analyzer is None or args.lock is None:ap.error('--analyzer and --lock are required')
        chosen=[c for c in cells() if args.only is None or f'{c[1]}-t{c[2]}-{c[0]}'==args.only]
        assert chosen,'unknown COSTCMP grid tag'
        owner.timing(args.binary,args.dest,32,args.parity,args.analyzer,arms=ARMS,lock=args.lock,
                     cells=chosen,parity_loader=loader,baseline=ARMS[0],ready_check=check)
    else:
        if args.lock is None:ap.error('--lock is required')
        with owner.segment_lock(args.lock):
            owner.rss(args.binary,args.dest,args.parity,arms=ARMS,parity_loader=loader,ready_check=check)

if __name__=='__main__':main()
