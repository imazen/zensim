#!/usr/bin/env python3
"""COSTCMP configuration for SPEEDQ's existing collectors and admission rules."""
import argparse
import hashlib
import json
import math
import struct
import re
from pathlib import Path
import speedq_run as owner

ARMS = ['by_v2fy_r5', 'zensim_A', 'zensim_B', 'fast_ssim2_main', 'fast_ssim2']
TIERS = ['v4x', 'v3']
THREADS = [1, 4, 16, 32]
FAST_MAIN = '09ec3e7c78bd230b1545572a188e162313c59fda'

SCALING_ARMS = ['by_v2fy_r5', 'by_v2fy_r4', 'zensim_A', 'zensim_B', 'by_v2fy_r5_before']
SCALING_THREADS = [1, 2, 4, 8, 16, 32]
SCALING_GEOMETRIES = ['1024x1024', '2048x2048', '4096x4096', '1920x1080']
BUDGET_ARMS = ['by_v2fy_r5_before', 'by_v2fy_r5_b64', 'by_v2fy_r5_b128', 'by_v2fy_r5_b256']
BUDGET_THREADS = [8, 16, 32]
BUDGET_GEOMETRIES = ['1024x1024', '4096x4096', '8192x4096']


def configuration(scaling=False, budget=False):
    assert not (scaling and budget), 'select one COSTCMP grid'
    if budget:
        return ['v4x'], BUDGET_THREADS, BUDGET_GEOMETRIES, BUDGET_ARMS, 'costcmp-budget-preflight-v1'
    return (TIERS, SCALING_THREADS if scaling else THREADS,
            SCALING_GEOMETRIES if scaling else owner.GEOMETRIES,
            SCALING_ARMS if scaling else ARMS,
            'costcmp-scaling-preflight-v1' if scaling else 'costcmp-preflight-v1')


def cells(scaling=False, budget=False):
    tiers, threads, geometries, _, _ = configuration(scaling, budget)
    return [(g,t,n) for t in tiers for n in threads for g in geometries]


def budget_inventory(path):
    manifest = json.loads(Path(path).read_text())
    assert manifest['schema'] == 'rev5perf5-binaries-v1'
    assert set(manifest['arms']) == set(BUDGET_ARMS)
    dependencies = []
    binaries = {}
    for arm, rec in manifest['arms'].items():
        binary = Path(rec['binary'])
        assert hashlib.sha256(binary.read_bytes()).hexdigest() == rec['binary_sha256']
        artifact = json.loads(Path(rec['artifact']).read_text())
        assert artifact['binary_sha256'] == rec['binary_sha256']
        dependencies.append(artifact['dependencies'])
        source = Path(rec['runtime_source']).read_bytes()
        assert hashlib.sha256(source).hexdigest() == rec['runtime_source_sha256']
        caps = re.findall(rb'const REV5_JOB_BUDGET_BYTES: usize = (\d+) \* 1024 \* 1024;', source)
        expected = [] if arm.endswith('_before') else [arm.rsplit('b',1)[1].encode()]
        assert caps == expected, 'source byte cap does not match candidate identity'
        binaries[arm] = binary
    assert all(value == dependencies[0] for value in dependencies), 'candidate dependencies differ'
    return binaries

def signature(rec):
    return (rec['input_sha256'], rec['score_bits'], rec['model'])

def validate_ready(rows, name, geometry, tier, threads, rec):
    old = rows[(geometry, tier, threads, name)]
    assert (f"{rec['width']}x{rec['height']}",rec['tier'],int(rec['threads'])) == (geometry,tier,threads), 'STOP: worker coordinates differ from COSTCMP grid cell'
    assert signature(rec) == signature(old), 'STOP: worker input/score/model differs from COSTCMP preflight'
    assert rec['arm'] == ('by_v2fy' if name.startswith('by_v2fy_r') else name)
    if name.startswith('by_v2fy_r'):
        assert rec['model']['source_sha256'] == owner.SOURCE_SHA
        assert rec['model']['revision'] == str(owner.worker_revision(name))
    elif name in ('zensim_A', 'zensim_B'):
        assert rec['model']['revision'] == '1'
        assert len(rec['model']['source_sha256']) == 64
    elif name == 'fast_ssim2_main':
        assert rec['model']['commit'] == FAST_MAIN
    else:
        assert rec['model']['version'] == '0.8.2'

def receipt(path, scaling=False, budget=False):
    value = json.loads(Path(path).read_text())
    tiers, threads, geometries, arms, schema = configuration(scaling, budget)
    assert value['schema'] == schema and value['status'] == 'PASS'
    rows = {(f"{r['width']}x{r['height']}", r['tier'], int(r['threads']), r['name']): r for r in value['records']}
    assert len(rows) == len(value['records']) == len(cells(scaling,budget))*len(arms)
    assert set(rows) == {(g,t,n,a) for g,t,n in cells(scaling,budget) for a in arms}
    groups = [arms] if budget else ([['by_v2fy_r5','by_v2fy_r5_before'], ['by_v2fy_r4']] if scaling else [[ARMS[0]]])
    for g in geometries:
        for group in groups:
            production = [rows[(g,t,n,a)] for t in tiers for n in threads for a in group]
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
        assert rec['input_sha256'] == rows[(g,t,n,arms[0])]['input_sha256']
        validate_ready(rows,a,g,t,n,rec)
    return rows

def production_rows(rows):
    return {(g,t,n,owner.worker_revision(a)):rec for (g,t,n,a),rec in rows.items() if a.startswith('by_v2fy_r')}

def preflight(binary, dest, scaling=False, before_binary=None, budget=False, worker_binaries=None):
    root=Path(dest);root.mkdir(parents=True,exist_ok=False);records=[]
    _, _, _, arms, schema = configuration(scaling,budget)
    for g,t,n in cells(scaling,budget):
        for name in arms:
            arm='by_v2fy' if name.startswith('by_v2fy_r') else name
            revision = owner.worker_revision(name)
            executable = (worker_binaries or {}).get(name, before_binary if name=='by_v2fy_r5_before' else binary)
            rec=owner.worker_run(executable,arm,revision,g,t,n,root/f'{t}-t{n}-{g}-{name}.log')
            rec['name']=name;records.append(rec)
        print(f'preflight {t}-t{n}-{g}',flush=True)
    path=root/'PREFLIGHT_PASS.json'
    owner.write(path,dict(schema=schema,status='PASS',records=records,strict_rev5_cells=len(cells(scaling,budget)),consumed_features=420))
    try:
        receipt(path,scaling,budget)
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
    ap.add_argument('--budget-grid',action='store_true',help='64/128/256 MiB plus uncapped Rev5 at 8/16/32 threads and three geometries')
    ap.add_argument('--budget-binaries',type=Path,help='frozen executable/source inventory for the byte-budget grid')
    args=ap.parse_args()
    if args.scaling and args.budget_grid:ap.error('select one COSTCMP grid')
    if args.budget_grid and args.budget_binaries is None:ap.error('--budget-grid requires --budget-binaries')
    if args.budget_binaries is not None and not args.budget_grid:ap.error('--budget-binaries requires --budget-grid')
    if args.scaling and args.before_binary is None:ap.error('--scaling requires --before-binary')
    if args.mode=='status':
        owner.collection_status(args.dest,expected_timing=9 if args.budget_grid else (48 if args.scaling else 64),expected_rss=36 if args.budget_grid else (48 if args.scaling else 80));return
    if args.binary is None:ap.error('--binary is required')
    binaries = budget_inventory(args.budget_binaries) if args.budget_grid else ({'by_v2fy_r5_before':args.before_binary} if args.scaling else None)
    if args.budget_grid:assert args.binary.resolve() in {p.resolve() for p in binaries.values()}
    if args.mode=='parity':preflight(args.binary,args.dest,args.scaling,args.before_binary,args.budget_grid,binaries);return
    if args.parity is None:ap.error('--parity is required')
    rows=receipt(args.parity,args.scaling,args.budget_grid)
    check=lambda name,g,t,n,rec:validate_ready(rows,name,g,t,n,rec)
    loader=lambda path:production_rows(receipt(path,args.scaling,args.budget_grid))
    _, _, _, arms, _ = configuration(args.scaling,args.budget_grid)
    if args.mode=='timing':
        if args.analyzer is None or args.lock is None:ap.error('--analyzer and --lock are required')
        chosen=[c for c in cells(args.scaling,args.budget_grid) if args.only is None or f'{c[1]}-t{c[2]}-{c[0]}'==args.only]
        assert chosen,'unknown COSTCMP grid tag'
        owner.timing(args.binary,args.dest,32,args.parity,args.analyzer,arms=arms,lock=args.lock,
                     cells=chosen,parity_loader=loader,baseline=arms[0],ready_check=check,
                     worker_binaries=binaries)
    else:
        if args.lock is None:ap.error('--lock is required')
        with owner.segment_lock(args.lock):
            owner.rss(args.binary,args.dest,args.parity,
                      arms=arms if args.budget_grid else (['by_v2fy_r5','by_v2fy_r5_before'] if args.scaling else ARMS),
                      parity_loader=loader,ready_check=check,
                      geometries=BUDGET_GEOMETRIES if args.budget_grid else (SCALING_GEOMETRIES if args.scaling else None),
                      thread_counts=BUDGET_THREADS if args.budget_grid else (SCALING_THREADS if args.scaling else None),
                      worker_binaries=binaries)

if __name__=='__main__':main()
