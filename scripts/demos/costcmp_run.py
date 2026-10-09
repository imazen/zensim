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

SCALING_ARMS = ['by_v2fy_r5', 'by_v2fy_r4', 'zensim_A', 'zensim_B', 'by_v2fy_r5_before']
SCALING_THREADS = [1, 2, 4, 8, 16, 32]
SCALING_GEOMETRIES = ['1024x1024', '2048x2048', '4096x4096', '1920x1080']


def cells(scaling=False):
    return [(g, t, n) for t in TIERS
            for n in (SCALING_THREADS if scaling else THREADS)
            for g in (SCALING_GEOMETRIES if scaling else owner.GEOMETRIES)]

def signature(rec):
    return (rec['input_sha256'], rec['score_bits'], rec['model'])

def validate_ready(rows, name, geometry, tier, threads, rec):
    old = rows[(geometry, tier, threads, name)]
    assert (f"{rec['width']}x{rec['height']}",rec['tier'],int(rec['threads'])) == (geometry,tier,threads), 'STOP: worker coordinates differ from COSTCMP grid cell'
    assert signature(rec) == signature(old), 'STOP: worker input/score/model differs from COSTCMP preflight'
    assert rec['arm'] == ('by_v2fy' if name.startswith('by_v2fy_r') else name)
    if name.startswith('by_v2fy_r'):
        assert rec['model']['source_sha256'] == owner.SOURCE_SHA
        assert rec['model']['revision'] == ('4' if name == 'by_v2fy_r4' else '5')
    elif name in ('zensim_A', 'zensim_B'):
        assert rec['model']['revision'] == '1'
        assert len(rec['model']['source_sha256']) == 64
    elif name == 'fast_ssim2_main':
        assert rec['model']['commit'] == FAST_MAIN
    else:
        assert rec['model']['version'] == '0.8.2'

def receipt(path, scaling=False):
    value = json.loads(Path(path).read_text())
    assert value['schema'] == ('costcmp-scaling-preflight-v1' if scaling else 'costcmp-preflight-v1') and value['status'] == 'PASS'
    rows = {(f"{r['width']}x{r['height']}", r['tier'], int(r['threads']), r['name']): r for r in value['records']}
    arms = SCALING_ARMS if scaling else ARMS
    assert len(rows) == len(value['records']) == len(cells(scaling))*len(arms)
    assert set(rows) == {(g,t,n,a) for g,t,n in cells(scaling) for a in arms}
    groups = [['by_v2fy_r5','by_v2fy_r5_before'], ['by_v2fy_r4']] if scaling else [[ARMS[0]]]
    for g in (SCALING_GEOMETRIES if scaling else owner.GEOMETRIES):
        for group in groups:
            production = [rows[(g,t,n,a)] for t in TIERS
                          for n in (SCALING_THREADS if scaling else THREADS) for a in group]
            assert len({r['input_sha256'] for r in production}) == 1
            assert len({r['score_bits'] for r in production}) == 1, f'strict {group} score parity differs'
            bits = []
            for rec in production:
                f = rec['feature_values']
                assert len(f) == 420 and all(math.isfinite(v) for v in f)
                bits.append(b''.join(struct.pack('>d',v) for v in f))
            assert len(set(bits)) == 1, f'strict {group} consumed-feature parity differs'
    for (g,t,n,a),rec in rows.items():
        assert math.isfinite(rec['score'])
        assert rec['input_sha256'] == rows[(g,t,n,ARMS[0])]['input_sha256']
        validate_ready(rows,a,g,t,n,rec)
    return rows

def production_rows(rows):
    return {(g,t,n,int(a[-1])):rec for (g,t,n,a),rec in rows.items() if a in ('by_v2fy_r4','by_v2fy_r5')}

def preflight(binary, dest, scaling=False, before_binary=None):
    root=Path(dest);root.mkdir(parents=True,exist_ok=False);records=[]
    for g,t,n in cells(scaling):
        for name in (SCALING_ARMS if scaling else ARMS):
            arm='by_v2fy' if name.startswith('by_v2fy_r') else name
            revision = 4 if name=='by_v2fy_r4' else (5 if arm=='by_v2fy' else 1)
            executable = before_binary if name=='by_v2fy_r5_before' else binary
            rec=owner.worker_run(executable,arm,revision,g,t,n,root/f'{t}-t{n}-{g}-{name}.log')
            rec['name']=name;records.append(rec)
        print(f'preflight {t}-t{n}-{g}',flush=True)
    path=root/'PREFLIGHT_PASS.json'
    owner.write(path,dict(schema='costcmp-scaling-preflight-v1' if scaling else 'costcmp-preflight-v1',status='PASS',records=records,strict_rev5_cells=len(cells(scaling)) if scaling else 64,consumed_features=420))
    try:
        receipt(path,scaling)
    except Exception:
        path.rename(root/'PREFLIGHT_FAILED.json')
        raise

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=['parity','timing','rss','status'])
    ap.add_argument('--binary',type=Path);ap.add_argument('--dest',type=Path,required=True)
    ap.add_argument('--parity',type=Path);ap.add_argument('--analyzer',type=Path)
    ap.add_argument('--lock',type=Path);ap.add_argument('--only')
    ap.add_argument('--scaling',action='store_true',help='Rev5/Rev4/A/B plus frozen before Rev5; 24 scaling cells and 24 low-thread controls')
    ap.add_argument('--before-binary',type=Path)
    args=ap.parse_args()
    if args.scaling and args.before_binary is None:ap.error('--scaling requires --before-binary')
    if args.mode=='status':
        owner.collection_status(args.dest,expected_timing=48 if args.scaling else 64,expected_rss=48 if args.scaling else 80);return
    if args.binary is None:ap.error('--binary is required')
    if args.mode=='parity':preflight(args.binary,args.dest,args.scaling,args.before_binary);return
    if args.parity is None:ap.error('--parity is required')
    rows=receipt(args.parity,args.scaling)
    check=lambda name,g,t,n,rec:validate_ready(rows,name,g,t,n,rec)
    loader=lambda path:production_rows(receipt(path,args.scaling))
    if args.mode=='timing':
        if args.analyzer is None or args.lock is None:ap.error('--analyzer and --lock are required')
        chosen=[c for c in cells(args.scaling) if args.only is None or f'{c[1]}-t{c[2]}-{c[0]}'==args.only]
        assert chosen,'unknown COSTCMP grid tag'
        owner.timing(args.binary,args.dest,32,args.parity,args.analyzer,arms=SCALING_ARMS if args.scaling else ARMS,lock=args.lock,
                     cells=chosen,parity_loader=loader,baseline=ARMS[0],ready_check=check,
                     worker_binaries={'by_v2fy_r5_before':args.before_binary} if args.scaling else None)
    else:
        if args.lock is None:ap.error('--lock is required')
        with owner.segment_lock(args.lock):
            owner.rss(args.binary,args.dest,args.parity,
                      arms=['by_v2fy_r5','by_v2fy_r5_before'] if args.scaling else ARMS,
                      parity_loader=loader,ready_check=check,
                      geometries=SCALING_GEOMETRIES if args.scaling else None,
                      thread_counts=SCALING_THREADS if args.scaling else None,
                      worker_binaries={'by_v2fy_r5_before':args.before_binary} if args.scaling else None)

if __name__=='__main__':main()
