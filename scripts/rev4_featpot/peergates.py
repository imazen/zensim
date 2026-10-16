#!/usr/bin/env python3
"""Pinned label-free peer packet transport; the Rust owners compute gates/stats."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path('/mnt/v/output/zensim/peergates-2026-10-16')
PRIVATE = Path.home() / 'tmp/zensim-paper/rev4'
REPO = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.file_digest(Path(path).open('rb'), 'sha256').hexdigest()


def write(path, obj):
    with Path(path).open('x') as f:
        json.dump(obj, f, indent=1, allow_nan=False)
        f.write('\n')


def progress(message):
    now = datetime.now(timezone.utc).isoformat()
    with (PRIVATE / 'PEERGATES_PROGRESS.md').open('a') as f:
        f.write(f'\n- {now}: {message}\n')
    print(message, flush=True)


def inputs():
    import pyarrow.parquet as pq
    p6 = Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part6')
    packet = read(p6 / 'cached-gate-a/job.json')
    receipts = read(REPO / 'benchmarks/e34_d12_2026-10-15.json')
    rows = {}
    pins = {}
    for m in ['a', 'c']:
        p = p6 / f'cached-gate-{m}/result.json'
        assert sha(p) == receipts['models'][m]['result_sha256']
        rows[m] = {r['key']: r for r in read(p)['rows']}
        pins[m] = dict(path=str(p), sha256=sha(p), model=receipts['models'][m]['model_sha256'])
    assert len(packet['cases']) == 135
    for c in packet['cases']:
        ar, cr = rows['a'][c['key']], rows['c'][c['key']]
        assert len(ar['blocks']) == len(cr['blocks'])
        c['candidate_blocks'] = []
        for a, b in zip(ar['blocks'], cr['blocks'], strict=True):
            assert a['bounds'] == b['bounds']
            c['candidate_blocks'].append(dict(bounds=a['bounds'], a_map=a['refinement_gain'],
                c_map=b['refinement_gain'], a_gain=a['score_delta']))
    write(ROOT / 'steering-input.json', packet)
    write(ROOT / 'steering-smoke.json', dict(cases=packet['cases'][:1]))
    manifest = dict(candidate_captures=pins, tables={})
    base = Path('/var/tmp/shippath7/verified/instruments')
    for grid, n in [('standard',4424),('ladder',9593),('negtail',2000),('identity',38)]:
        p = base / f'{grid}.parquet'
        schema = pq.ParquetFile(p).schema_arrow.names
        # Explicit projection: neither feature nor subjective/metric targets read.
        cols = ['ref_path','dist_path','reference_file_sha256','distorted_file_sha256',
                'reference_pixels_sha256','distorted_pixels_sha256','width','height','pair_key']
        cols += ['image_id','codec','q'] if grid in ['standard','ladder'] else ['entry']
        assert set(cols) <= set(schema)
        meta = pq.read_table(p,columns=cols).to_pylist()
        assert len(meta) == n
        cases=[]
        for i,r in enumerate(meta):
            cases.append(dict(key=str(i), reference=r['ref_path'],distorted=r['dist_path'],
                reference_sha256=r['reference_file_sha256'],distorted_sha256=r['distorted_file_sha256'],
                reference_pixels_sha256=r['reference_pixels_sha256'],
                distorted_pixels_sha256=r['distorted_pixels_sha256'],metadata=r))
        write(ROOT / f'{grid}-input.json',dict(cases=cases))
        manifest['tables'][grid] = dict(path=str(p),sha256=sha(p),columns_read=cols,rows=n,
            ordered_metadata_sha256=sha(ROOT / f'{grid}-input.json'))
    write(ROOT/'inputs.json',manifest)
    progress('Inputs pinned:135steering cases/55,515forwards;4424standard+9593ladder+2000negative+38identity rows, explicit metadata-only projection. No label columns loaded.')


def steering_report():
    path = ROOT/'steering/results.jsonl'
    rows=[json.loads(l) for l in path.read_text().splitlines()]
    assert len(rows)==135 and len({r['key'] for r in rows})==135
    result=dict(cases=135,forwards=sum(len(r['blocks'])+1 for r in rows),matrix={},own={})
    assert result['forwards']==55515
    for m in ['diff410','diff_frag820','butter','ssim2']:
        result['matrix'][m]={}
        for j in ['fixed_zensim','butter','ssim2']:
            v=[r['cross_judge'][m][j] for r in rows]
            finite=[x for x in v if x is not None]
            result['matrix'][m][j]=dict(passed=sum(x>=.70 for x in finite),total=135,
                undefined=135-len(finite),mean=statistics.mean(finite),median=statistics.median(finite),minimum=min(finite))
    for m in ['butter','ssim2']:
        result['own'][m]=dict(result['matrix'][m][m],failures=[dict(key=r['key'],rho=r[f'{m}_rho'])
            for r in rows if r[f'{m}_rho'] is None or r[f'{m}_rho']<.70],m2='N/A: exact peer gradient unavailable')
    write(ROOT/'steering-summary.json',result)
    progress('Steering complete: '+ '; '.join(f"{m} own-maprho>=.70 {result['own'][m]['passed']}/135" for m in ['butter','ssim2'])+'. Cross-judge3×4 matrix saved; candidates exact M2 remains a separate requirement.')
    print(json.dumps(result,indent=1))


def peer_tables(grid):
    cases=read(ROOT/f'{grid}-input.json')['cases']
    rows=[json.loads(l) for l in (ROOT/grid/'results.jsonl').read_text().splitlines()]
    assert len(cases)==len(rows)
    for label,a in [('ssim2',None),('butter5',5),('butter10',10),('butter20',20)]:
        p=ROOT/f'{grid}-{label}.tsv'
        with p.open('x') as f:
            f.write('image_id\tcodec\tq\tpred\n' if grid in ['standard','ladder'] else 'entry\tpred\n')
            for i,(c,r) in enumerate(zip(cases,rows,strict=True)):
                assert r['index']==i and r['key']==c['key']
                score=r['ssim2'] if a is None else 100-a*r['butter_pnorm3']
                meta=c['metadata']
                keys=[meta['image_id'],meta['codec'],str(meta['q'])] if grid in ['standard','ladder'] else [meta['entry']]
                f.write('\t'.join(keys+[repr(score)])+'\n')
    progress(f'{grid} peer scores and map artifacts complete: {len(rows)} rows; scalar/map parity checked for every SSIM2 pair. Native SSIM2 and all three fixed butter mappings exported for canonical owner.')


def quiet():
    sys.path.insert(0,str(REPO/'scripts/demos'))
    import speedq_run
    state=speedq_run.quiet_state()
    with (ROOT/'quiet-probes.jsonl').open('a') as f:f.write(json.dumps(state)+'\n')
    print(json.dumps(state))


def landed():
    """Append each completed case once; a partial last line is not a result."""
    path=ROOT/'steering/results.jsonl'
    state=ROOT/'progress-cases.json'
    seen=read(state)['cases'] if state.exists() else 0
    rows=[json.loads(l) for l in path.read_text().splitlines() if l.endswith('}')]
    for r in rows[seen:]:
        progress(f"G-STEER case {r['key']}: butter rho={r['butter_rho']}, SSIM2 rho={r['ssim2_rho']}; bar .70; {len(r['blocks'])} independent block repairs. Partial panel, not qualification.")
    state.write_text(json.dumps(dict(cases=len(rows)))+'\n')
    return len(rows)


def watch():
    """Visible, caller-owned read-only progress relay; never refreshes repo locks."""
    while True:
        if landed()==135:
            return
        log=(ROOT/'steering.log').read_text()
        if 'run-heavy: done rc=' in log:
            raise RuntimeError('steering ended without135results; inspect retained log')
        time.sleep(2)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inputs','steering-report','peer-tables','quiet','landed','watch']);p.add_argument('--grid',choices=['standard','ladder','negtail','identity']);a=p.parse_args()
    if a.mode=='inputs':inputs()
    elif a.mode=='steering-report':steering_report()
    elif a.mode=='peer-tables':peer_tables(a.grid)
    elif a.mode=='quiet':quiet()
    elif a.mode=='watch':watch()
    else:landed()


if __name__=='__main__':main()
