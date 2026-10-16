#!/usr/bin/env python3
"""CVVDP-FHD transport for the owner-authorized PEERGATES addendum.

This consumes only explicit engineering pixels, their metadata, and existing
metric outputs. All gate definitions stay in bake_verdict.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from peergates import ROOT, REPO, read, sha, write, progress

CV = ROOT/'cvvdp'
SCORES = Path('/var/tmp/paper-gates/scores/ladder_cvvdp_fhd.tsv')
OLD = Path('/mnt/v/output/zensim/ladder-2026-09-05/instruments')
PROBES = Path('/mnt/v/output/zensim/dialgate-2026-09-04')
BV = Path('/mnt/v/output/zensim/qual-a-2026-10-10/bin/bake_verdict')


def restore_pixels():
    """Restore absent engineering bytes from their existing canonical mirror."""
    restored=[]
    for grid in ['standard','ladder','negtail','identity']:
        for c in read(ROOT/f'{grid}-input.json')['cases']:
            for side in ['reference','distorted']:
                dst=Path(c[side])
                if dst.exists():continue
                assert str(dst).startswith('/mnt/v/output/zensim/')
                src=Path(str(dst).replace('/mnt/v/','/mnt/tower/',1))
                assert src.is_file(),src
                assert shutil.disk_usage(dst.parents[3]).free-src.stat().st_size>20*1024**3
                expected=c[f'{side}_sha256']
                assert sha(src)==expected,src
                dst.parent.mkdir(parents=True,exist_ok=True)
                with src.open('rb') as a,dst.open('xb') as b:shutil.copyfileobj(a,b,1024**2)
                assert sha(dst)==expected,dst
                restored.append(dict(source=str(src),destination=str(dst),sha256=expected,bytes=dst.stat().st_size))
        progress(f'CVVDP pixel restoration: {grid} ready; {len(restored)} missing files recovered byte-identically from existing tower mirror; no pixels regenerated.')
    write(CV/'RESTORED_PIXELS.json',dict(files=restored))


def tsv(path):
    with Path(path).open() as f:
        return list(csv.DictReader(f,delimiter='\t'))


def verify_pixels():
    assert sha('/var/tmp/paper-cvvdp/bin/zenmetrics')=='ca154a6a39f910c9cba21422eb9b238463661bfa2d4a8d8f10407904c83e4cd1'
    seen={}
    for grid in ['standard','ladder','negtail','identity']:
        for c in read(ROOT/f'{grid}-input.json')['cases']:
            for side in ['reference','distorted']:
                p=c[side];h=c[f'{side}_sha256']
                if p in seen:assert seen[p]==h
                else:assert sha(p)==h,p
                seen[p]=h
        print(f'pixel hashes verified: {grid}',flush=True)
    write(CV/'PIXEL_BYTES_VERIFIED.json',dict(unique_files=len(seen),files=seen))
    progress('CVVDP-FHD pixel admission: every unique standard/ladder/negative/identity file SHA verified against frozen E34 metadata; retained ca154a6a binary SHA verified.')


def smoke():
    rows={int(r['row_index']):r for r in tsv(CV/'smoke-scores.tsv')}
    assert set(rows)=={0,1}
    field='cvvdp_cpu_imazen_v0_1_0_standard_fhd'
    expected=float(tsv(SCORES)[0][field])
    got=float(rows[0][field]);identity=float(rows[1][field])
    assert abs(expected-got)<=.5e-6,(expected,got)
    assert identity==10.0,identity
    write(CV/'SMOKE_PARITY.json',dict(first_ladder_expected=expected,first_ladder_measured=got,
        identity_measured=identity,printed_precision_tolerance=.5e-6,
        binary_sha256=sha('/var/tmp/paper-cvvdp/bin/zenmetrics'),smoke_scores_sha256=sha(CV/'smoke-scores.tsv')))
    progress(f'CVVDP-FHD retained binary reproduces stored first ladder cell: {got} versus {expected}; fresh identity={identity}JOD. Stored9593-rung FHD table admitted after exact path/key/native-score and parity checks.')


def mapped(grid):
    cases=read(ROOT/f'{grid}-input.json')['cases']
    field='cvvdp_cpu_imazen_v0_1_0_standard_fhd'
    if grid=='ladder':
        lut={(r['ref_path'],r['dist_path']):r for r in tsv(SCORES)}
        rows=[lut[c['reference'],c['distorted']] for c in cases]
        source=SCORES
    else:
        source=CV/f'{grid}-scores.tsv';raw=tsv(source)
        lut={int(r['row_index']):r for r in raw}
        assert len(lut)==len(raw)==len(cases)
        rows=[lut[i] for i in range(len(cases))]
    path=CV/f'{grid}-x10.tsv'
    with path.open('x') as f:
        writer=csv.writer(f,delimiter='\t',lineterminator='\n')
        writer.writerow(['image_id','codec','q','pred'] if grid in ['standard','ladder'] else ['entry','pred'])
        for c,r in zip(cases,rows,strict=True):
            assert (c['reference'],c['distorted'])==(r['ref_path'],r['dist_path'])
            meta=c['metadata'];score=10*float(r[field])
            assert math.isfinite(score), (grid,c['key'],score)
            keys=[meta['image_id'],meta['codec'],meta['q']] if grid in ['standard','ladder'] else [meta['entry']]
            writer.writerow(keys+[repr(score)])
    progress(f'CVVDP-FHD {grid}: complete{len(cases)}rows mapped by frozenQ=10×JOD; source SHA{sha(source)}; canonical owner input saved.')


def gates(grid, directory=CV, label='peer_cvvdp_fhd_x10', negative=True):
    assert (CV/'SMOKE_PARITY.json').exists() and (CV/'PIXEL_BYTES_VERIFIED.json').exists()
    missing=CV/'empty';missing.mkdir(exist_ok=True)
    panel=OLD/'dial_grid_372col_ladder.parquet' if grid=='ladder' else Path('/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet')
    for attribution in ['single','agree']:
        out=directory/f'gates-{grid}-{attribution}';out.mkdir()
        command=[str(BV),'--bake',str(REPO/'zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin'),
            '--regime','372','--features-root',str(missing),'--corpora','none',
            '--corruption-grid',str(missing/'excluded-corruption.parquet'),
            '--perpair-metrics',str(missing/'excluded-perpair.parquet'),
            '--dial-grid',str(panel),'--dial-peer-scores',f'{label}={directory}/{grid}-x10.tsv',
            '--negtail-probe',str(PROBES/'negtail_probe_372_2026-09-04.parquet' if negative else missing/'unmeasured-negtail.parquet'),
            '--identity-probe',str(PROBES/'identity_probe_372_2026-09-04.parquet'),
            '--identity-peer-scores',f'{label}={directory}/identity-x10.tsv',
            '--gaddr-value-pins','report','--inversion-truth',attribution,
            '--gaddr-json',str(out/'gaddr.json'),'--output',str(out/'verdict.md')]
        if negative:command+=['--negtail-peer-scores',f'{label}={directory}/negtail-x10.tsv']
        if grid=='ladder':
            command += ['--gaddr-grid-truth',str(OLD/'dialcells_ssim2_ladder.tsv'),
                '--reference-truth',str(OLD/'reference_truth_ladder_pnorm3.tsv')+':pnorm3',
                '--floor-rule','resolvable','--floor-margin','0.5']
        write(out/'COMMAND.json',dict(argv=command,binary_sha256=sha(BV),grid_sha256=sha(panel)))
        with (out/'owner.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
        d=read(out/'gaddr.json');assert len([r for r in d['checks'] if r['id'].startswith('C')])==6
        failures=[(r['id'],r['measured'],r['bar']) for r in d['checks'] if r['state']=='fail' and r['tier']!='report-only']
        progress(f'{label} {grid}/{attribution} canonical gates: CONTRACT{d["contract"]},REGRESSION{d["regression"]}; failures{failures}. Full row values/bars retained.')


def butter_history():
    """Cheap fixed-mapping replays of the historical stored pnorm3 table."""
    source=Path('/var/tmp/paper-gates/bv/cells/peer_butteraugli_pnorm3__native.tsv')
    identities=source.with_name(source.stem+'.identity.tsv')
    rows=tsv(source);ids=tsv(identities)
    cases=read(ROOT/'ladder-input.json')['cases']
    key=lambda r:(str(r['image_id']),str(r['codec']),float(r['q']))
    assert len(rows)==len(cases)==9593
    assert {key(r) for r in rows}=={key(c['metadata']) for c in cases}
    assert {r['entry'] for r in ids}=={c['metadata']['entry'] for c in read(ROOT/'identity-input.json')['cases']}
    for scale in [5,10,20]:
        out=ROOT/f'butter-history-{scale}';out.mkdir()
        write(out/'SOURCE.json',dict(path=str(source),sha256=sha(source),
            identity_path=str(identities),identity_sha256=sha(identities),
            mapping=f'Q=100+{scale}*stored_negative_pnorm3',
            provenance='historical stored peer, not fresh0.9.3 scoring; negative probes unavailable'))
        for data,filename in [(rows,'ladder-x10.tsv'),(ids,'identity-x10.tsv')]:
            with (out/filename).open('x') as f:
                w=csv.DictWriter(f,fieldnames=list(data[0]),delimiter='\t',lineterminator='\n');w.writeheader()
                for r in data:w.writerow({**r,'pred':repr(100+scale*float(r['pred']))})
        gates('ladder',out,f'peer_butter_history_a{scale}',False)


def pipeline():
    """Sequential stages, invoked inside one caller-owned heavy scope/lock."""
    if not (CV/'SMOKE_PARITY.json').exists():smoke()
    restore_pixels()
    verify_pixels()
    mapped('ladder')
    for grid in ['identity', 'negtail', 'standard']:
        output=CV/f'{grid}-scores.tsv'
        assert not output.exists(), output
        command=['/var/tmp/paper-cvvdp/bin/zenmetrics','batch','--metric','cvvdp',
            '--display-model','standard_fhd','--bench-on-start','no',
            '--use-legacy-scheduler','--pairs',str(CV/f'{grid}-pairs.tsv'),
            '--output',str(output),'--jobs','4','--group-by-ref']
        write(CV/f'{grid}-COMMAND.json',dict(argv=command,binary_sha256=sha(command[0]),
            input_sha256=sha(CV/f'{grid}-pairs.tsv'),rayon_threads=1))
        with (CV/f'{grid}.log').open('x') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                env={**os.environ,'RAYON_NUM_THREADS':'1'})
        mapped(grid)
    gates('ladder')
    gates('standard')


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['verify-pixels','smoke','mapped','gates','pipeline','butter-history','all-gates']);p.add_argument('--grid',choices=['standard','ladder','negtail','identity']);a=p.parse_args()
    if a.mode=='verify-pixels':verify_pixels()
    elif a.mode=='smoke':smoke()
    elif a.mode=='mapped':mapped(a.grid)
    elif a.mode=='pipeline':pipeline()
    elif a.mode=='butter-history':butter_history()
    elif a.mode=='all-gates':
        gates('ladder')
        gates('standard')
    else:gates(a.grid)


if __name__=='__main__':main()
