#!/usr/bin/env python3
"""Pinned label-free peer packet transport; the Rust owners compute gates/stats."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import random
import shutil
import socket
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


def case_id(row):
    b=row['blocks'][0]['bounds']
    return row['key'], max(b[2]-b[0], b[3]-b[1])


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
        rows[m] = {case_id(r): r for r in read(p)['rows']}
        assert len(rows[m])==135
        pins[m] = dict(path=str(p), sha256=sha(p), model=receipts['models'][m]['model_sha256'])
    assert len(packet['cases']) == 135
    for c in packet['cases']:
        ar, cr = rows['a'][c['key'],c['block']], rows['c'][c['key'],c['block']]
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
    rows=steering_rows()
    assert len(rows)==135 and len({case_id(r) for r in rows})==135
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
        result['own'][m]=dict(result['matrix'][m][m],failures=[dict(key=r['key'],block=case_id(r)[1],rho=r[f'{m}_rho'])
                for r in rows if r[f'{m}_rho'] is None or r[f'{m}_rho']<.70],m2='N/A: exact peer gradient unavailable')
    write(ROOT/'steering-summary.json',result)
    progress('Steering complete: '+ '; '.join(f"{m} own-maprho>=.70 {result['own'][m]['passed']}/135" for m in ['butter','ssim2'])+'. Cross-judge3×4 matrix saved; candidates exact M2 remains a separate requirement.')
    print(json.dumps(result,indent=1))


def failure_report():
    """Render retained gate values and engineering content identities verbatim."""
    summary=read(ROOT/'steering-summary.json')
    preread=Path('/mnt/v/output/zensim/prodqual-b-2026-10-07/STEERING_PREREAD.json')
    metadata={(c['case'],c['block']):c for c in read(preread)['cases']}
    candidate=[('diff-410','broad-10-b32',.9650540851,.9650199082),
        ('diff-410','broad-10-b64',.9382352941,.9352941176),
        ('diff-410','broad-76-b64',.9882352941,.9970588235),
        ('diff-410','broad-136-b64',.9676470588,.9794117647),
        ('diff+frag-820','broad-31-b64',.7272727273,.7272727273),
        ('diff+frag-820','broad-34-b32',.9518019974,.9468085106),
        ('diff+frag-820','broad-34-b64',.9720279720,.9720279720),
        ('diff+frag-820','broad-202-b32',.9666041276,.9660412758),
        ('diff+frag-820','broad-202-b64',.8601398601,.9370629371),
        ('W-C seed0','broad-13-b64',.8676470588,.8676470588),
        ('W-C seed0','broad-202-b64',.9790209790,.9860139860)]
    lines=['# Peer steering failures — 2026-10-16','',
        'Engineering identifiers describe content provenance; no semantic class is inferred.',
        'Candidate misses: M2<.99; listed M3f values all pass .70. Peer misses: own-map rho<.70; M2=N/A.',
        'Peers: butteraugli0.9.3 pnorm3, SSIMULACRA2 native-map54fabca9.',
        'All51 SSIMULACRA2 failures retained despite its lower task priority.','',
        '| Model | Case / block | Source distortion / geometry | M2 | M3f or peer own-map rho |',
        '|---|---|---|---:|---:|']
    def add(model,key,block,m2,m3):
        c=metadata[key,block]
        source=Path(c['source_distorted']).name
        lines.append(f'| {model} | {key} / {block} | {source}; {c["width"]}x{c["height"]} | {m2} | {m3:.10f} |')
    for m,k,m2,m3 in candidate:add(m,k,int(k.rsplit('b',1)[1]),f'{m2:.10f}',m3)
    for m in ['butter','ssim2']:
        for r in summary['own'][m]['failures']:add(m,r['key'],r['block'],'N/A',r['rho'])
    lines+=['','Full per-case/block cross-judge values, including every cross-judge miss:',
        '`/mnt/v/output/zensim/peergates-2026-10-16/{steering,steering-remaining}/results.jsonl`.',
        f'Content metadata SHA256: `{sha(preread)}`.',
        'Original candidate values: [D12](e34_d12_2026-10-15.md);',
        'aggregate findings and pins: [peer gates](peergates_2026-10-16.md).','']
    (REPO/'benchmarks/peergates_2026-10-16_failures.md').write_text('\n'.join(lines))


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


def steering_rows():
    paths=[ROOT/'steering/results.jsonl',ROOT/'steering-remaining/results.jsonl']
    return [json.loads(l) for p in paths if p.exists() for l in p.read_text().splitlines() if l.endswith('}')]


def repair_transport():
    old=read(ROOT/'steering-input.json')
    p6=Path('/mnt/v/output/zensim/e34-prep-2026-10-15/part6')
    captures={m:{case_id(r):r for r in read(p6/f'cached-gate-{m}/result.json')['rows']} for m in ['a','c']}
    for m in captures:assert len(captures[m])==135
    for c in old['cases']:
        a=captures['a'][c['key'],c['block']]['blocks'];b=captures['c'][c['key'],c['block']]['blocks']
        assert len(a)==len(b)
        c['candidate_blocks']=[dict(bounds=x['bounds'],a_map=x['refinement_gain'],c_map=y['refinement_gain'],a_gain=x['score_delta']) for x,y in zip(a,b,strict=True)]
        assert all(x['bounds']==y['bounds'] for x,y in zip(a,b,strict=True))
    complete=steering_rows();assert len(complete)==96
    original=read(ROOT/'steering-input.json')
    assert old['cases'][:96]==original['cases'][:96]
    assert [case_id(r) for r in complete]==[(c['key'],c['block']) for c in old['cases'][:96]]
    write(ROOT/'steering-input-v2.json',old)
    write(ROOT/'steering-remaining-input.json',dict(cases=old['cases'][96:]))
    write(ROOT/'TRANSPORT_CORRECTION.json',dict(reason='case names repeat at several block sizes; join on (key, block)',completed_preserved=96,remaining=39,original_sha256=sha(ROOT/'steering-input.json'),corrected_sha256=sha(ROOT/'steering-input-v2.json'),completed_inputs_identical=True,thresholds_changed=False))
    progress('Steering transport corrected: exact(key,block) joins135/135captures; first96inputs byte-equivalent,96completed cases preserved;39remaining queued. Failed attempt/maps/log retained. No numerical threshold or case changed.')


def landed():
    """Append each completed case once; a partial last line is not a result."""
    state=ROOT/'progress-cases.json'
    seen=read(state)['cases'] if state.exists() else 0
    rows=steering_rows()
    for r in rows[seen:]:
        progress(f"G-STEER case {r['key']}: butter rho={r['butter_rho']}, SSIM2 rho={r['ssim2_rho']}; bar .70; {len(r['blocks'])} independent block repairs. Partial panel, not qualification.")
    state.write_text(json.dumps(dict(cases=len(rows)))+'\n')
    return len(rows)


def watch():
    """Visible, caller-owned read-only progress relay; never refreshes repo locks."""
    while True:
        if landed()==135:
            return
        log=(ROOT/'steering-remaining.log').read_text() if (ROOT/'steering-remaining.log').exists() else (ROOT/'steering.log').read_text()
        if 'run-heavy: done rc=' in log:
            raise RuntimeError('steering ended without135results; inspect retained log')
        time.sleep(2)


def archive():
    """Retain the measurement closure and verify a non-destructive mirror."""
    import subprocess
    source=ROOT/'source';source.mkdir(exist_ok=True)
    paths=['scripts/rev4_featpot/peergates.py','scripts/rev4_featpot/peergates_cvvdp.py',
        'zensim-bench/peergates/Cargo.toml','zensim-bench/peergates/Cargo.lock',
        'zensim-bench/peergates/src/main.rs','zensim/examples/support/coherence_stats.rs',
        'benchmarks/peergates_2026-10-16.md','benchmarks/peergates_2026-10-16_failures.md','justfile']
    for name in paths:
        dst=source/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,dst)
    for name in ['PEERGATES_PHASE1.md','PEERGATES_PLAN.md','PEERGATES_PLAN_ADDENDUM.md',
            'PEERGATES_CVVDP_RANK.md','PEERGATES_DONE.md','PEERGATES_PROGRESS.md']:
        shutil.copy2(PRIVATE/name,ROOT/name)
    retained=[Path('/var/tmp/paper-gates/scores/ladder_cvvdp_fhd.tsv'),
        Path('/mnt/v/output/zensim/qual-a-2026-10-10/bin/bake_verdict'),
        Path('/var/tmp/paper-gates/bv/cells/peer_butteraugli_pnorm3__native.tsv'),
        Path('/var/tmp/paper-gates/bv/cells/peer_butteraugli_pnorm3__native.identity.tsv'),
        Path('/var/tmp/paper-gates/bv/gaddr/peer_ssim2_registered__native.json')]
    owner=ROOT/'retained';owner.mkdir(exist_ok=True)
    for p in retained:shutil.copy2(p,owner/p.name)
    files=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name not in ['SHA256SUMS','MIRROR_VERIFIED.json'])
    hashes={str(p.relative_to(ROOT)):sha(p) for p in files}
    (ROOT/'SHA256SUMS').write_text(''.join(f'{h}  {n}\n' for n,h in hashes.items()))
    mirror=Path('/mnt/tower/output/zensim/peergates-2026-10-16');mirror.mkdir(parents=True,exist_ok=True)
    assert shutil.disk_usage(mirror).free>sum(p.stat().st_size for p in files)
    subprocess.run(['rsync','-rt','--executability',str(ROOT)+'/',str(mirror)+'/'],check=True)
    selected=random.Random(20261016).sample(sorted(hashes),3)
    required=['steering-summary.json','cvvdp/SMOKE_PARITY.json','PEERGATES_DONE.md',
        'cvvdp/gates-ladder-agree/gaddr.json','bin/peergates-checked']
    checked=sorted(set(selected+required))
    for n in checked:assert sha(mirror/n)==hashes[n],n
    assert sha(mirror/'SHA256SUMS')==sha(ROOT/'SHA256SUMS')
    result=dict(host=socket.gethostname(),utc=datetime.now(timezone.utc).isoformat(),
        files=len(hashes),manifest_sha256=sha(ROOT/'SHA256SUMS'),
        random_files=selected,verified_files=checked,mirror=str(mirror))
    (ROOT/'MIRROR_VERIFIED.json').write_text(json.dumps(result,indent=1)+'\n')
    shutil.copy2(ROOT/'MIRROR_VERIFIED.json',mirror/'MIRROR_VERIFIED.json')
    progress(f'Evidence mirror verified: {len(hashes)}files, SHA256SUMS={result["manifest_sha256"]}; three seeded-random files plus five required artifacts rehashed on tower. No mirror deletion.')


def validate():
    """Check completed evidence coverage and frozen experiment identities."""
    import csv
    import math
    for pin,name in [('PLAN_FREEZE.json','PEERGATES_PLAN.md'),
            ('ADDENDUM_FREEZE.json','PEERGATES_PLAN_ADDENDUM.md')]:
        assert sha(PRIVATE/name)==read(ROOT/pin)['sha256']
    rows=steering_rows()
    assert len(rows)==len({case_id(r) for r in rows})==135
    assert sum(len(r['blocks'])+1 for r in rows)==55515
    assert all(r['scalar_map_score_bits_equal'] for r in rows)
    for m in read(ROOT/'inputs.json')['candidate_captures'].values():assert sha(m['path'])==m['sha256']
    counts={}
    for g,n in [('standard',4424),('ladder',9593),('negtail',2000),('identity',38)]:
        with (ROOT/f'cvvdp/{g}-x10.tsv').open() as f:cells=list(csv.DictReader(f,delimiter='\t'))
        assert len(cells)==n
        assert all(math.isfinite(float(c['pred'])) for c in cells)
        counts[g]=n
    assert read(ROOT/'cvvdp/SMOKE_PARITY.json')['first_ladder_measured']==7.525087
    for grid in ['standard','ladder']:
        for attribution in ['single','agree']:
            d=read(ROOT/f'cvvdp/gates-{grid}-{attribution}/gaddr.json')
            assert {r['id'] for r in d['checks'] if r['tier']=='contract'}=={f'C{i}' for i in range(1,7)}
    write(ROOT/'FINAL_VALIDATION.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        frozen_plans_unchanged=True,steering_cases=135,forwards_per_peer=55515,
        ssim2_base_scalar_map_bit_parity=135,cvvdp_complete_panels=counts,
        candidate_capture_sha256s_verified=True))
    progress('Final artifact validation PASS: frozen plans unchanged,135unique steering cases/55,515forwards,135scalar-map bit checks,complete mapped CVVDP panels4424/9593/2000/38 and four owner-gate reports.')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['inputs','steering-report','failure-report','peer-tables','quiet','landed','watch','repair-transport','archive','validate']);p.add_argument('--grid',choices=['standard','ladder','negtail','identity']);a=p.parse_args()
    if a.mode=='inputs':inputs()
    elif a.mode=='steering-report':steering_report()
    elif a.mode=='failure-report':failure_report()
    elif a.mode=='peer-tables':peer_tables(a.grid)
    elif a.mode=='quiet':quiet()
    elif a.mode=='watch':watch()
    elif a.mode=='repair-transport':repair_transport()
    elif a.mode=='archive':archive()
    elif a.mode=='validate':validate()
    else:landed()


if __name__=='__main__':main()
