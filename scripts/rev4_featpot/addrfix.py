#!/usr/bin/env python3
"""ADDRFIX metadata transport; the existing Rust owners compute all gates."""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess

from peergates import read, sha, write

ROOT = Path('/mnt/v/output/zensim/addrfix-2026-10-16')
PRIV = Path.home()/'tmp/zensim-paper/rev4'
REPO = Path(__file__).resolve().parents[2]
SHIP = Path('/mnt/v/output/zensim/shipgates-2026-10-16/gaddr-attempt3')
PEER = Path('/mnt/v/output/zensim/peergates-2026-10-16')


def progress(message):
    with (PRIV/'ADDRFIX_PROGRESS.md').open('a') as f:
        f.write('\n- '+datetime.now(timezone.utc).isoformat()+': '+message+'\n')
    print(message, flush=True)


def owner(binary, trace, peers, curve_models=None, control_name="curve-control"):
    """Replay exact admitted commands, replacing only outputs/scorer tables."""
    bindings=read(SHIP/'BINDINGS.json')
    destination=ROOT/(control_name if curve_models else 'peer-replay' if peers else 'candidate-trace' if trace else 'candidate-baseline')
    destination.mkdir()
    models=['cvvdp_fhd','ssim2'] if peers else ['diff-410','diff-frag-820','w-c']
    for model in models:
        for grid in ['standard','ladder']:
            record=bindings['instruments'][grid]
            assert sha(record['grid'])==record['grid_sha256']
            assert sha(SHIP/f'{grid}-mentor.tsv')==record['bound_mentor_sha256']
            src=SHIP/('diff-410' if peers else model)/grid
            command=read(src/'command.json')
            command[0]=binary
            if curve_models:command[command.index('--bake')+1]=str(curve_models[model])
            out=destination/model/grid;out.mkdir(parents=True)
            for flag in ['--gaddr-json','--output','--fulleval']:
                i=command.index(flag)
                if peers and flag=='--fulleval':del command[i:i+2]
                else:command[i+1]=str(out/Path(command[i+1]).name)
            if peers:
                scores=PEER/f'cvvdp/{grid}-x10.tsv' if model=='cvvdp_fhd' else SHIP/f'{grid}-mentor.tsv'
                command+=['--dial-peer-scores',f'addrfix_{model}={scores}']
                command+=['--identity-peer-scores',f'addrfix_{model}={PEER}/cvvdp/identity-x10.tsv'] if model=='cvvdp_fhd' else []
                command+=['--negtail-peer-scores',f'addrfix_{model}={PEER}/cvvdp/negtail-x10.tsv'] if model=='cvvdp_fhd' else []
                if model=='ssim2':
                    # This replay requests A7r only; do not let the carrier
                    # bake supply an unrelated peer's probe scores.
                    for flag in ['--identity-probe','--negtail-probe']:
                        command[command.index(flag)+1]=str(out/'not-requested.parquet')
            env={**os.environ,'TMPDIR':str(Path.home()/'tmp'),'ZENSIM_FORMULA_REV':'5',
                'ZENSIM_DIAL_PRED_OUT':str(out/'scores.tsv')}
            if trace:env['ZENSIM_GADDR_LADDER_TRACE']=str(out/'ladder-trace.jsonl')
            write(out/'COMMAND.json',dict(argv=command,binary_sha256=sha(binary),
                diagnostic_trace=trace,grid_sha256=sha(record['grid']),
                mentor_sha256=sha(SHIP/f'{grid}-mentor.tsv')))
            with (out/'owner.log').open('x') as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            d=read(out/'gaddr.json')
            measured=d['measured']['codec_floor']
            if not peers and not curve_models:
                old=read(src/'gaddr.json')['measured']['codec_floor']
                assert measured==old,(model,grid,'owner output changed')
            rows=[dict(codec=r['codec'],fraction=r['represented_frac'],bar=r['represented_frac_reference'],
                state=r['state'],order_failures=r['n_fail_order'],clamp_failures=r['n_fail_clamp']) for r in measured]
            progress(f'{model}/{grid}: A7r canonical rows {rows}; trace={trace}; original candidate aggregate parity verified={not peers and not curve_models}; exploratory model control={control_name if curve_models else None}.')


def curve_control(binary):
    """Explore whether K creates the floor-order defect; no gate changes."""
    refit=Path('/mnt/v/output/zensim/e34-launch-2026-10-15/bin/bake_dial_refit')
    directory=ROOT/'curve-models';directory.mkdir()
    models={}
    for name in ['diff-410','diff-frag-820','w-c']:
        command=read(SHIP/name/'standard/command.json')
        source=Path(command[command.index('--bake')+1]);pin=sha(source)
        dest=directory/f'{name}-no-output-curve.bin'
        args=[str(refit),'strip','--in',str(source),'--out',str(dest)]
        with (directory/f'{name}.log').open('x') as log:
            subprocess.run(args,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert sha(source)==pin
        write(directory/f'{name}-PROVENANCE.json',dict(argv=args,binary_sha256=sha(refit),
            source_sha256=pin,output_sha256=sha(dest),exploratory=True,
            meaning='diagnostic removal of output curve; not a qualified score dial'))
        models[name]=dest
    owner(binary,False,False,models)


def unpacked_control(binary, checkpoint='best'):
    """Check retained f32 checkpoints, with canonical dense-contract admission."""
    refit=Path('/mnt/v/output/zensim/e34-launch-2026-10-15/bin/bake_dial_refit')
    directory=ROOT/('unpacked-models' if checkpoint=='best' else 'last-f32-models');directory.mkdir()
    control_name='unpacked-control' if checkpoint=='best' else 'last-f32-control'
    models={};pins={}
    for name in ['diff-410','diff-frag-820','w-c']:
        command=read(SHIP/name/'standard/command.json')
        packed=Path(command[command.index('--bake')+1])
        source=packed.parent/(checkpoint+'.bin');pin=sha(source)
        if checkpoint=='last':
            receipt=read(packed.parent.parent/'result.json')
            assert receipt['selection']['epoch_rule']=='last'
            assert receipt['selected_bake_sha256']==pin and receipt['packed_model_sha256']==sha(packed)
        output=directory/f'{name}-dense-f32.bin'
        args=[str(refit),'densify','--in',str(source),'--out',str(output)]
        with (directory/f'{name}-densify.log').open('x') as log:
            result=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        assert sha(source)==pin and sha(packed)==sha(ROOT/'cost-payload'/f'models/{name}.bin')
        pins[name]=dict(source=str(source),source_sha256=pin,packed_sha256=sha(packed),argv=args,
            canonical_tool_sha256=sha(refit),returncode=result.returncode,checkpoint=checkpoint)
        if result.returncode:
            pins[name]['state']='admission-refused'
            progress(f'Exploratory unpacked {name}: canonical dense-contract owner refuses rc{result.returncode}; log retained, no gate relaxation.')
            continue
        assert output.is_file()
        pins[name].update(state='canonical-densify-admitted',output=str(output),output_sha256=sha(output))
        models[name]=output
    write(directory/'PROVENANCE.json',pins)
    # Best changes the training epoch as well; last is the production epoch.
    # Neither comparison attributes an effect to quantization alone.
    result=dict(exploratory=True,qualification_allowed=False,
        checkpoint=checkpoint,
        meaning='f32 checkpoint vs production; best changes epoch, last matches production epoch; canonical identity admission',
        admitted_models=list(models),refused_models=[name for name in pins if name not in models])
    if len(models)!=3:
        result['gate_replay']='not-run: allthree exact contracts required'
        write(directory/'RESULT.json',result)
        return
    try:
        owner(binary,False,False,models,control_name)
    except subprocess.CalledProcessError as exc:
        result['gate_replay']=f'canonical scorer admission refused rc{exc.returncode}'
        progress('Unpacked checkpoint gate replay refused; exact scorer error retained; no labels or thresholds changed.')
    else:
        return unpacked_report(checkpoint)
    write(directory/'RESULT.json',result)


def unpacked_report(checkpoint='best'):
    """Render retained exact owner outputs; no numerical replay or label access."""
    directory=ROOT/('unpacked-models' if checkpoint=='best' else 'last-f32-models');pins=read(directory/'PROVENANCE.json')
    control_name='unpacked-control' if checkpoint=='best' else 'last-f32-control'
    assert set(pins)=={'diff-410','diff-frag-820','w-c'}
    rows={}
    for name,pin in pins.items():
        assert pin['state']=='canonical-densify-admitted'
        assert sha(pin['source'])==pin['source_sha256'] and sha(pin['output'])==pin['output_sha256']
        rows[name]={}
        for grid in ['standard','ladder']:
            actual=read(ROOT/control_name/name/grid/'gaddr.json')['measured']['codec_floor']
            baseline=read(ROOT/'candidate-baseline'/name/grid/'gaddr.json')['measured']['codec_floor']
            assert [r['codec'] for r in actual]==[r['codec'] for r in baseline]
            assert all(r['n_ladders']==b['n_ladders'] and r['represented_frac_reference']==b['represented_frac_reference']
                for r,b in zip(actual,baseline,strict=True))
            rows[name][grid]=[dict(codec=r['codec'],fraction=r['represented_frac'],denominator=r['n_ladders'],
                reference=r['represented_frac_reference'],state=r['state'],
                delta_fraction=r['represented_frac']-b['represented_frac']) for r,b in zip(actual,baseline,strict=True)]
    with (ROOT/'diagnosis/FAILING_RUNGS.tsv').open() as f:edges=list(csv.DictReader(f,delimiter='\t'))
    comparisons=[];orientations={}
    for name in pins:
        counts=dict(still_inverted=0,exact_tie=0,now_increasing=0)
        for grid in ['standard','ladder']:
            with (ROOT/control_name/name/grid/'scores.tsv').open() as f:
                scores={(r['image_id'],r['codec'],float(r['q'])):float(r['pred']) for r in csv.DictReader(f,delimiter='\t')}
            for edge in edges:
                if edge['model']!=name or edge['grid']!=grid:continue
                left=scores[edge['image_id'],edge['codec'],float(edge['q0'])]
                right=scores[edge['image_id'],edge['codec'],float(edge['q1'])]
                state='still_inverted' if right<left else 'exact_tie' if right==left else 'now_increasing'
                counts[state]+=1
                comparisons.append(dict(model=name,grid=grid,codec=edge['codec'],image_id=edge['image_id'],
                    q0=edge['q0'],q1=edge['q1'],f32_left=left,f32_right=right,orientation=state))
        orientations[name]=counts
    assert len(comparisons)==len(edges)==97
    with (directory/'EDGE_COMPARISON.tsv').open('x') as f:
        w=csv.DictWriter(f,fieldnames=list(comparisons[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(comparisons)
    write(directory/'RESULT.json',dict(exploratory=True,qualification_allowed=False,gate_replay='complete',
        checkpoint=checkpoint,
        meaning='retained f32 checkpoint vs production bake; best differs in epoch, last is selected production epoch; canonical densify identity admission',
        rows=rows,original_failed_edge_orientations=orientations))
    progress(f'Exploratory {checkpoint} f32 checkpoint control complete: '+json.dumps(rows,separators=(',',':')))


def trace_report():
    """Render the owner's rung choices and failure causes, never re-grade."""
    from collections import Counter
    contexts=[];edges=[];counts={}
    for grid in ['standard','ladder']:
        cases=read(PEER/f'{grid}-input.json')['cases']
        pixels={(str(c['metadata']['image_id']),c['metadata']['codec'],float(c['metadata']['q'])):c for c in cases}
        for model in ['diff-410','diff-frag-820','w-c']:
            path=ROOT/'candidate-trace'/model/grid
            gate=read(path/'gaddr.json')
            floors=gate['measured']['codec_floor']
            n=sum(r['n_ladders'] for r in floors)
            rows=[json.loads(line) for line in (path/'ladder-trace.jsonl').read_text().splitlines()]
            assert len(rows)==2*n,(model,grid,len(rows),n)
            candidate,mentor=rows[:n],rows[n:]
            with (ROOT/'curve-control'/model/grid/'scores.tsv').open() as f:
                raw={(r['image_id'],r['codec'],float(r['q'])):float(r['pred']) for r in csv.DictReader(f,delimiter='\t')}
            failures={r['codec'] for r in floors if r['state']=='fail'}
            for r,m in zip(candidate,mentor,strict=True):
                assert (r['image_id'],r['codec'])==(m['image_id'],m['codec'])
                assert [(x['position'],x['q'],x['mentor']) for x in r['selected']]==[(x['position'],x['q'],x['candidate']) for x in m['selected']]
                if r['codec'] not in failures or r['represented']:continue
                contexts.append(dict(model=model,grid=grid,**r))
                by_position={v['position']:v for v in r['selected']}
                for edge in r['edges']:
                    if edge['cause']=='increasing' and not r['clamped']:continue
                    left=by_position[edge['from_position']];right=by_position[edge['to_position']]
                    lo=pixels[r['image_id'],r['codec'],left['q']];hi=pixels[r['image_id'],r['codec'],right['q']]
                    cause=edge['cause'] if edge['cause']!='increasing' else 'shared_clamp'
                    raw0=raw[r['image_id'],r['codec'],left['q']];raw1=raw[r['image_id'],r['codec'],right['q']]
                    edges.append(dict(model=model,grid=grid,codec=r['codec'],image_id=r['image_id'],
                        q0=left['q'],q1=right['q'],candidate0=left['candidate'],candidate1=right['candidate'],
                        mentor0=left['mentor'],mentor1=right['mentor'],
                        mentor_delta=right['mentor']-left['mentor'],delta_candidate=edge['delta'],
                        raw0=raw0,raw1=raw1,raw_order='inversion' if raw1<raw0 else 'exact_tie' if raw1==raw0 else 'increasing',
                        curve_induced=raw1>raw0 and edge['delta']<=0,
                        cause=cause,sub_material=edge['sub_material'],clamped=r['clamped'],
                        reference_path=lo['reference'],distorted0_path=lo['distorted'],distorted1_path=hi['distorted']))
            for floor in floors:
                traced=[r for r in candidate if r['codec']==floor['codec']]
                assert len(traced)==floor['n_ladders']
                assert sum(not r['ordered'] for r in traced)==floor['n_fail_order']
                assert sum(r['clamped'] for r in traced)==floor['n_fail_clamp']
            counts[f'{model}/{grid}']=dict(gate_failed_codecs=sorted(failures),
                failed_ladders=sum(not r['represented'] and r['codec'] in failures for r in candidate),
                trace_sha256=sha(path/'ladder-trace.jsonl'),gate_sha256=sha(path/'gaddr.json'))
    output=ROOT/'diagnosis';output.mkdir()
    with (output/'FAILING_RUNGS.tsv').open('x') as f:
        writer=csv.DictWriter(f,fieldnames=list(edges[0]),delimiter='\t',lineterminator='\n');writer.writeheader();writer.writerows(edges)
    write(output/'FAILING_LADDERS.json',contexts)
    write(output/'SUMMARY.json',dict(panels=counts,failed_edges=len(edges),
        causes=dict(Counter(r['cause'] for r in edges)),
        raw_causes=dict(Counter(r['raw_order'] for r in edges)),
        curve_induced_failed_edges=sum(r['curve_induced'] for r in edges),
        mentor_increasing_failed_edges=sum(r['mentor_delta']>0 for r in edges),
        sub_material_failed_edges=sum(r['sub_material'] for r in edges),
        original_scores_sha256s={str(p.relative_to(ROOT)):sha(p) for p in (ROOT/'candidate-baseline').glob('*/*/scores.tsv')}))
    progress(f'Exact A7r diagnosis complete: {len(contexts)}failed ladders in failing codec rows; {len(edges)}failing rung edges; causes={dict(Counter(r["cause"] for r in edges))}. Every q, candidate/mentor/raw score, physical image/path and selected window retained; trace aggregates exactly match original owner counts.')


def rank_pairs():
    """Only label-free physical keys and pixel metadata are projected here."""
    import pyarrow.parquet as pq
    base=Path('/mnt/v/output/zensim/v40r4-2026-10-08/v2e29')
    receipt=read(base/'wide/main/real/receipt.json')
    output=ROOT/'ranking';output.mkdir()
    allowed={'kadid_train','kadid_select','tid2013','konfig_train','konfig_val','cid22_a25'}
    panels={}
    for fold,n,groups in zip(['kadid','tid2013','konfig','cid22_a25'],[7869,3000,756,2192],[65,25,7,25],strict=True):
        rec=receipt['legs'][fold]['full']
        keys=(base/rec['rel']).with_suffix('.keys.parquet')
        assert sha(keys)==rec['keys_sha256']
        meta=pq.read_table(keys,columns=['pair_key','source_row_id','ref_basename','member_set']).to_pylist()
        assert len(meta)==n and len({(r['member_set'],r['source_row_id']) for r in meta})==n
        names={r['ref_basename'].removesuffix('_parta').removesuffix('_partb') for r in meta}
        assert len(names)==groups
        banks={}
        bank_pins={}
        for member in {r['member_set'] for r in meta}:
            assert member in allowed
            path=Path('/var/tmp/rev4-featbank/bank')/member/'keys.parquet'
            assert sha(path)==receipt['bank'][member]['keys_sha256']
            rows=pq.read_table(path,columns=['pair_key','row_id','ref_path','dist_path',
                'ref_pixels_sha256','dist_pixels_sha256','width','height']).to_pylist()
            banks[member]={}
            for row in rows:
                old=banks[member].setdefault(row['pair_key'],row)
                assert {k:v for k,v in old.items() if k!='row_id'}=={k:v for k,v in row.items() if k!='row_id'}
            bank_pins[member]=dict(path=str(path),sha256=sha(path))
        cells=[]
        with (output/f'{fold}-pairs.tsv').open('x') as f:
            writer=csv.writer(f,delimiter='\t',lineterminator='\n');writer.writerow(['ref_path','dist_path'])
            for i,r in enumerate(meta):
                # source_row_id is the admitted target's physical identity,
                # not the pixel bank's row_id. Keep it, join pixels by pair.
                b=banks[r['member_set']][r['pair_key']]
                assert b['pair_key']==r['pair_key']
                writer.writerow([b['ref_path'],b['dist_path']])
                cells.append(dict(row_index=i,**{**r,**{k:v for k,v in b.items() if k!='row_id'}}))
        write(output/f'{fold}-PIXEL_KEYS.json',cells)
        missing=sorted({c[s] for c in cells for s in ['ref_path','dist_path'] if not Path(c[s]).is_file()})
        panels[fold]=dict(rows=n,references=groups,missing_pixels=missing,
            metadata_keys_sha256=sha(keys),metadata_pixels_sha256=sha(output/f'{fold}-PIXEL_KEYS.json'),
            pairs_sha256=sha(output/f'{fold}-pairs.tsv'),bank_pins=bank_pins)
        progress(f'Matched ranking {fold}: exact{n}physical observations/{groups}references bound through admitted label-free keys; missing pixel files={len(missing)}. No target payload opened.')
    write(output/'PAIR_ADMISSION.json',panels)


def rank_score(resume=False):
    """Score explicit pixels only; subjective targets remain owner-private."""
    import shutil
    binary=Path('/var/tmp/paper-cvvdp/bin/zenmetrics')
    assert sha(binary)=='ca154a6a39f910c9cba21422eb9b238463661bfa2d4a8d8f10407904c83e4cd1'
    directory=ROOT/'ranking'
    pins={}
    for fold in ['kadid','tid2013','konfig','cid22_a25']:
        cells=read(directory/f'{fold}-PIXEL_KEYS.json')
        for c in cells:
            for key in ['ref_path','dist_path']:
                path=c[key]
                if path not in pins:pins[path]=sha(path)
    if resume:
        assert read(directory/'FILE_BYTES.json')['files']==pins
    else:
        write(directory/'FILE_BYTES.json',dict(files=pins,unique_files=len(pins),
            meaning='Exact admitted file paths and bytes; no claim of cross-decoder pixel parity'))
    # Parallel scheduling must reproduce the retained one-thread scalar smoke.
    smoke=directory/'smoke-pairs.tsv'
    if resume:assert smoke.read_bytes()==(PEER/'cvvdp/smoke-pairs.tsv').read_bytes()
    else:
        with smoke.open('x') as f:f.write((PEER/'cvvdp/smoke-pairs.tsv').read_text())
    jobs=[('smoke',smoke)]+[(fold,directory/f'{fold}-pairs.tsv') for fold in ['kadid','tid2013','konfig','cid22_a25']]
    peers={}
    field='cvvdp_cpu_imazen_v0_1_0_standard_fhd'
    for fold,pairs in jobs:
        assert shutil.disk_usage(Path.home()).free>20*1024**3
        reuse=resume and fold in ['smoke','kadid']
        output=directory/f'{fold}-scores.tsv'
        if not reuse:
            assert not output.exists()
            if fold!='smoke':
                indexed=directory/f'{fold}-indexed-pairs.tsv'
                cells=read(directory/f'{fold}-PIXEL_KEYS.json')
                admission=read(directory/'PAIR_ADMISSION.json')[fold]
                assert sha(pairs)==admission['pairs_sha256']
                assert sha(directory/f'{fold}-PIXEL_KEYS.json')==admission['metadata_pixels_sha256']
                with indexed.open('x') as f:
                    w=csv.writer(f,delimiter='\t',lineterminator='\n');w.writerow(['ref_path','dist_path','row_index'])
                    for i,c in enumerate(cells):w.writerow([c['ref_path'],c['dist_path'],i])
                pairs=indexed
        elif fold=='kadid':
            assert sha(output)=='f1b83c617aa104be23cc88c3c95c43ba82f794c91c6419953ac2d83424dfdc4f'
        else:
            with output.open() as f:prior=list(csv.DictReader(f,delimiter='\t'))
            with (PEER/'cvvdp/smoke-scores.tsv').open() as f:expected=list(csv.DictReader(f,delimiter='\t'))
            assert {r['row_index']:r[field] for r in prior}=={r['row_index']:r[field] for r in expected}
        command=[str(binary),'batch','--metric','cvvdp','--display-model','standard_fhd',
            '--bench-on-start','no','--use-legacy-scheduler','--pairs',str(pairs),
            '--output',str(output),'--jobs','4','--group-by-ref']
        if not reuse:write(directory/f'{fold}-COMMAND.json',dict(argv=command,binary_sha256=sha(binary),
            pairs_sha256=sha(pairs),rayon_threads=4))
        if not reuse:
            with (directory/f'{fold}.log').open('x') as log:
                subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,
                    env={**os.environ,'RAYON_NUM_THREADS':'4','TMPDIR':str(Path.home()/'tmp')})
        with output.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
        if fold=='smoke':
            with (PEER/'cvvdp/smoke-scores.tsv').open() as f:old=list(csv.DictReader(f,delimiter='\t'))
            assert {r['row_index']:r[field] for r in rows}=={r['row_index']:r[field] for r in old}
            progress('CVVDP-FHD four-thread scheduling reproduces retained scalar smoke exactly at output precision; binary/display unchanged.')
        else:
            keys=read(directory/f'{fold}-PIXEL_KEYS.json');assert len(rows)==len(keys)
            if 'row_index' not in rows[0]:
                assert reuse and fold=='kadid'
                by_pair={}
                for row in rows:
                    key=row['ref_path'],row['dist_path']
                    assert by_pair.setdefault(key,row)==row, 'duplicate pair scores differ'
                assert set(by_pair)=={(k['ref_path'],k['dist_path']) for k in keys}
                indexed=directory/'kadid-indexed-scores.tsv'
                with indexed.open('x') as f:
                    w=csv.DictWriter(f,fieldnames=['ref_path','dist_path','row_index',field],delimiter='\t',lineterminator='\n');w.writeheader()
                    for i,k in enumerate(keys):w.writerow(dict(by_pair[k['ref_path'],k['dist_path']],row_index=i))
                write(directory/'KADID_INDEX_REPAIR.json',dict(source_sha256=sha(output),output_sha256=sha(indexed),
                    pixel_keys_sha256=sha(directory/'kadid-PIXEL_KEYS.json'),physical_rows=len(keys),unique_pairs=len(by_pair),
                    duplicate_scores_identical=True,method='exact ref/dist pair join; preserve every physical observation'))
                output=indexed
                with output.open() as f:rows=list(csv.DictReader(f,delimiter='\t'))
            lut={int(r['row_index']):r for r in rows};assert len(lut)==len(keys)
            for i,k in enumerate(keys):
                assert (k['ref_path'],k['dist_path'])==(lut[i]['ref_path'],lut[i]['dist_path'])
            peers[fold]=dict(tsv=str(output),sha256=sha(output),pixel_keys=str(directory/f'{fold}-PIXEL_KEYS.json'),
                pixel_keys_sha256=sha(directory/f'{fold}-PIXEL_KEYS.json'))
            progress(f'CVVDP-FHD matched {fold}: complete{len(rows)}physical pairs, row/path coverage verified; score TSV SHA{sha(output)}. No subjective targets opened.')
    assert all(sha(path)==pin for path,pin in pins.items()), 'pixel bytes changed during scoring'
    write(directory/'OWNER_JOB.json',dict(schema='addrfix-matched-cvvdp-v1',
        paired_input='/mnt/v/output/zensim/e34-prep-2026-10-15/part5/noise-input.json',
        paired_sha256='45e5f49dd329eee4bfb7751fe43b5dd70e548094a89b0af5dfdfd31828dd2eda',peer_scores=peers,
        current_input='/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/panel-input.json',
        current_sha256='a1511bab58833e202fe89d3c90ac663cd1e9dbbb88cc71057b5a7d99385caa9c',
        progress_markdown=str(PRIV/'ADDRFIX_PROGRESS.md')))


def rank_report():
    """Render only aggregate results emitted by the registered statistics owner."""
    import shutil
    source=ROOT/'ranking/MATCHED_RESULT.json';data=read(source)
    assert data['schema']=='addrfix-matched-cvvdp-result-v1' and not data['qualification_allowed']
    assert len(data['comparisons'])==15
    models=['diff410','diff_frag820','w_c']
    scopes=['kadid','tid2013','konfig','cid22_a25','overall']
    titles=['KADID','TID2013','KonFiG','CID22-A25','Four-source mean']
    dest=REPO/'benchmarks/addrfix_matched_rank_2026-10-16.json'
    assert not dest.exists();shutil.copyfile(source,dest)
    assert sha(source)==sha(dest) and dest.stat().st_size<30000
    lines=['','## Matched CVVDP-FHD ranking','',
        'Exploratory only. The registered Rust owner checks exact E33/E34 target,',
        'reference and orientation equality; physical pair indices, pixel-key SHA',
        'and all scalar sidecar hashes are bound before assessment. All 13,817',
        'physical observations and 122 references are retained, including three',
        'repeated KADID pixel pairs. No protected holdout is used.', '',
        'CVVDP is the pinned imazen CPU implementation (`ca154a6a`, `standard_fhd`):',
        '1920×1080, 24-inch display, 0.6 m, 200 nit, contrast 1000, 250 lux.',
        'Native JOD is consumed at the CLI’s six-decimal output precision.', '',
        '| Population | Pairs / references | CVVDP-FHD | diff-410 | diff+frag-820 | W-C |',
        '|---|---:|---:|---:|---:|---:|']
    for scope,title in zip(scopes,titles,strict=True):
        if scope=='overall':
            point=data['four_source_mean'];coverage='4 sources'
        else:
            row=data['point_scores'][scope];point={k:v['mean_signed_srocc'] for k,v in row['models'].items()}
            coverage=f"{row['rows']} / {row['references']}"
        lines.append('| '+title+' | '+coverage+' | '+' | '.join(f'{point[k]:.6f}' for k in ['cvvdp_fhd',*models])+' |')
    lines+=['','Candidate minus CVVDP-FHD SROCC, with paired 95% intervals:', '',
        '| Population | diff-410 | diff+frag-820 | W-C |', '|---|---:|---:|---:|']
    for scope,title in zip(scopes,titles,strict=True):
        cells=[]
        for model in models:
            row=next(r for r in data['comparisons'] if r['scope']==scope and r['candidate']==model)
            iv=row['delta'];cells.append(f"{iv['estimate']:+.6f} [{iv['lo']:+.6f}, {iv['hi']:+.6f}]")
        lines.append('| '+title+' | '+' | '.join(cells)+' |')
    lines+=['','Each candidate point averages its ten withheld-fold training seeds; the',
        'overall point weights the four sources equally. The reference is one fixed',
        'score vector, repeated across those ten slots for paired resampling, not',
        'ten independent fits. The existing source-reference × shared-seed bootstrap',
        'uses 10,000 draws, seed 20261016 and type-7 percentile intervals. Every',
        'source-specific interval has 10,000 finite draws; each overall interval',
        'has 9,739 finite draws and 261 dropped draws. Full counts and auxiliary',
        'Student intervals remain in the [owner result](addrfix_matched_rank_2026-10-16.json).',
        'All fifteen descriptive lower bounds exceed zero. These intervals are',
        'unadjusted across comparisons and cannot qualify the adopted confirmatory',
        'gate or select a repair on these previously exposed populations.','',
        'The ancillary E33 A/C four-source means reproduce the prior retained',
        f"points: {data['four_source_mean']['a']:.16f} / {data['four_source_mean']['c']:.16f}.",
        'Resource receipt: `run-heavy rc=0,266s,peak-RSS0.11GiB,',
        'min-avail34817MiB,peak-load22.96` (statistics, not quiet runtime).','']
    with (REPO/'benchmarks/addrfix_2026-10-16.md').open('a') as f:f.write('\n'.join(lines))
    progress('Matched ranking aggregate JSON/table rendered from full canonical owner output; all15 intervals retained; source SHA '+sha(source)+'.')


def cost_summary_report():
    """Render completed canonical timing/RSS captures without recalculating stats."""
    import shutil
    source=ROOT/'cost-analysis/SUMMARY.json';data=read(source)
    assert data['qualification_allowed'] is False and len(data['cells'])==3
    destination=REPO/'benchmarks/addrfix_matched_cost_2026-10-16.json'
    assert not destination.exists();shutil.copyfile(source,destination)
    assert sha(source)==sha(destination) and destination.stat().st_size<30000
    parity=[]
    for model in data['cells']:
        receipt=read(ROOT/'cost-r7900x/results'/model/'PREFLIGHT_PASS.json')
        assert receipt['status']=='PASS'
        parity.extend(r['model'] for r in receipt['cache_parity'])
    assert len(parity)==6
    lines=['','## Matched complete-map cost and peak RSS','',
        'r7900x; v4x, CPU 2, one thread; identical deterministic RGB8 pairs at',
        '1024² and 2048². Both arms prepare their reference cache outside latency',
        'and inside process RSS, then retain the complete native score+map result',
        'through each call. Candidate uses `BakeScorer`; butteraugli 0.9.3 FIR',
        'uses `ButteraugliReference`, full-resolution diffmap and pnorm3. Dependency',
        'features include AVX-512/rayon and exclude IIR. The frozen worker SHA is',
        '`'+data['worker_sha256']+'`.', '',
        'All three preflights pass: candidate scalar bits equal the cached-map path',
        'and prior frozen worker; cached/direct butter scores and native maps',
        f"have maximum observed absolute differences {max(r['score_abs_delta'] for r in parity):g} / {max(r['map_max_abs_delta'] for r in parity):g}.",
        'The pre-existing FIR tolerances remain 1e-10 score and 1e-6 map.',
        'Every cell retains the first 32 quiet paired rounds of at most 64;',
        'all six use exactly 32 rounds, with zero excluded rounds or retries.', '',
        '| Model / size | Candidate map p95 ms | Butter map p95 ms | Difference ms [paired 95% CI] |',
        '|---|---:|---:|---:|']
    for model,grids in data['cells'].items():
        assert len(grids)==2
        for geometry,r in grids.items():
            assert r['accepted_rounds']==32 and r['excluded_rounds']==0
            iv=r['paired_p95_delta_ns']
            lines.append(f"| {model} / {geometry} | {r['map_p95_ns']/1e6:.3f} | {r['butter_map_p95_ns']/1e6:.3f} | {iv['estimate']/1e6:+.3f} [{iv['lo']/1e6:+.3f}, {iv['hi']/1e6:+.3f}] |")
    lines+=['','Nearest-rank p95 and paired-round bootstrap: existing Rust owner, 10,000',
        'finite draws per interval, seed 20261016, 95% type-7 percentiles.',
        'All six candidate p95s meet the absolute 150/600 ms budgets. All three',
        'candidates are slower than matched butter at 1024² and faster at 2048²;',
        'the corresponding paired intervals exclude zero. The matched-peer',
        'latency requirement therefore fails at 1024² for every candidate.', '',
        '| Model / size | Candidate peak KiB | Butter peak KiB | Absolute cap KiB | Cap met |',
        '|---|---:|---:|---:|---|']
    for model,grids in data['cells'].items():
        for geometry,r in grids.items():
            lines.append(f"| {model} / {geometry} | {r['absolute_map_peak_kib']} | {r['butter_absolute_map_peak_kib']} | {r['absolute_rss_cap_kib']} | {'yes' if r['absolute_rss_cap_met'] else 'no'} |")
    lines+=['','Peak is the maximum of three independent fresh-process `/usr/bin/time -v`',
        'runs per arm, 108 records total including map, scalar and baseline arms.',
        'Every candidate peak is below matched butter. All 1024² cells meet the',
        'absolute 128 B/pixel + 64 MiB cap; every 2048² cell exceeds it. Baseline',
        'subtraction would make all six pass the older incremental convention,',
        'which does not satisfy adopted R3. The [full summary](addrfix_matched_cost_2026-10-16.json)',
        'retains every repetition, baselines, increments and scalar-ratio diagnostics.', '',
        'Acquisition resource receipt: `run-heavy rc=0,279s,peak-RSS1.03GiB,',
        'min-avail21497MiB,peak-load1.72`, hard 16 GiB cap. Statistical analysis:',
        '`rc=0,1s,peak-RSS0.02GiB,min-avail38704MiB,peak-load20.04`.',
        'These are measured engineering comparisons, not a new model qualification.','']
    with (REPO/'benchmarks/addrfix_2026-10-16.md').open('a') as f:f.write('\n'.join(lines))
    progress('Matched cost full6cell table and RSS repetitions rendered; all1024peer-p95checks fail, all2048absoluteRSScapsfail; absolute latency budgets and relativeRSS pass. Source SHA '+sha(source)+'.')


def cost_payload():
    """Assemble a reviewable single-host job, without launching remote work."""
    import shutil
    import sys
    sys.path.insert(0,str(REPO/'scripts/demos'))
    import speedq_run
    dest=ROOT/'cost-payload';(dest/'bin').mkdir(parents=True)
    speedq_run.freeze_binary(ROOT/'bench-build.log',dest/'bin/speedq')
    analyzer=Path('/mnt/v/output/zensim/costcmp-2026-10-09/provenance/paired-rounds-analyzer')
    assert sha(analyzer)=='798d938626e31c54e524da56687c468f7178d1629b4ee4f268692a290ccbba82'
    shutil.copy2(analyzer,dest/'bin/paired-rounds-analyzer')
    models={}
    for name,arm in [('diff-410','k-a'),('diff-frag-820','k-c'),('w-c','w-c')]:
        command=read(SHIP/name/'standard/command.json');source=Path(command[command.index('--bake')+1])
        model=dest/'models'/f'{name}.bin';model.parent.mkdir(exist_ok=True);shutil.copy2(source,model)
        prior=Path('/mnt/v/output/zensim/shipgates-2026-10-16/runtime/runtime')/arm/'PREFLIGHT_PASS.json'
        if name=='w-c':prior=Path('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/runtime/w-c/PREFLIGHT_PASS.json')
        golden=dest/'models'/f'{name}-golden.json';shutil.copy2(prior,golden)
        models[name]=dict(path=str(model.relative_to(dest)),sha256=sha(model),hidden=32 if name=='w-c' else 128,
            golden=str(golden.relative_to(dest)),golden_sha256=sha(golden))
    for relative in ['scripts/rev4_featpot/addrfix_cost.py','scripts/rev4_featpot/peergates.py','scripts/demos/speedq_run.py']:
        target=dest/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/relative,target)
    write(dest/'JOB.json',dict(binary_sha256=sha(dest/'bin/speedq'),models=models,
        host='r7900x',geometries=['1024x1024','2048x2048'],tier='v4x',threads=1,cpuset='2',
        quiet_gate='unchanged SPEEDQ load1<2/no build or namedtraining',rounds=32,rss_repetitions=3))
    progress('Matched r7900x cost job assembled locally with exact binary/model/analyzer/source hashes; no remote job launched.')


def cost_report(binary):
    """Render admitted SPEEDQ captures; Rust owns paired p95 statistics."""
    import sys
    sys.path.insert(0,str(REPO/'scripts/demos'))
    import speedq_run
    root=ROOT/'cost-r7900x'
    job=read(root/'JOB.json');assert job==read(ROOT/'cost-payload/JOB.json')
    assert sha(root/'bin/speedq')==job['binary_sha256']
    analysis=ROOT/'cost-analysis';analysis.mkdir()
    summary={}
    for model,binding in job['models'].items():
        summary[model]={}
        preflight=read(root/'results'/model/'PREFLIGHT_PASS.json')
        assert preflight['status']=='PASS' and preflight['binary_sha256']==job['binary_sha256']
        for size in [1024,2048]:
            geometry=f'{size}x{size}';tag=f'v4x-t1-{geometry}'
            cell=root/'results'/model/'timing'/tag
            header=read(cell/'header.json');complete=read(cell/'COMPLETE.json')
            assert header['binary_sha256']==job['binary_sha256'] and header['cpuset']=='2'
            assert header['tier']=='v4x' and header['threads']==1 and header['quiet_gate']['admitted']
            assert complete['status']=='PASS' and complete['rounds']==32 and read(cell/'interference.json')['admitted']
            values,selection=speedq_run.select_clean_rounds(read(cell/'zenbench.inner.json'),32)
            assert all(complete[k]==v for k,v in selection.items())
            dest=analysis/model/geometry;dest.mkdir(parents=True)
            packet=dest/'packet.json'
            write(packet,dict(schema='addrfix-map-cost-v1',baseline_ns=values['e34_butter_map'],
                candidates_ns={k:values[k] for k in ['e34_candidate_map_retained','e34_candidate','e34_butter_scalar']}))
            output=dest/'paired-p95.json'
            command=[binary,'e34','map-cost','--input',str(packet),'--output',str(output)]
            with (dest/'owner.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
            result=read(output);own=result['comparisons']['e34_candidate_map_retained']
            rss={arm:[] for arm in ['e34_candidate_map_retained','e34_candidate_baseline','e34_candidate',
                'e34_butter_map','e34_butter_baseline','e34_butter_scalar']}
            for repeat in range(3):
                for arm in rss:
                    rec=read(root/'results'/model/f'rss-{repeat}'/f'{tag}-{arm}.json')
                    assert rec['quiet_required'] and rec['quiet_gate']['admitted']
                    assert rec['binary_sha256']==job['binary_sha256'] and rec['threads']==1
                    if arm.startswith('e34_candidate'):assert rec['worker']['model']['source_sha256']==binding['sha256']
                    rss[arm].append(rec['max_rss_kib'])
            peak=max(rss['e34_candidate_map_retained']);peer=max(rss['e34_butter_map'])
            increments=[m-b for m,b in zip(rss['e34_candidate_map_retained'],rss['e34_candidate_baseline'],strict=True)]
            cap=(128*size*size+64*1024**2)//1024;budget_ns={1024:150000000,2048:600000000}[size]
            record=dict(map_p95_ns=own['p95_ns'],butter_map_p95_ns=result['baseline_p95_ns'],
                paired_p95_delta_ns=own['delta_ns'],p95_upper_ci_le_butter=own['upper_ci_le_zero'],
                absolute_latency_budget_ns=budget_ns,absolute_latency_budget_met=own['p95_ns']<=budget_ns,
                old_map_over_uncached_scalar=own['p95_ns']/result['comparisons']['e34_candidate']['p95_ns'],
                rss_peaks_kib=rss,absolute_map_peak_kib=peak,butter_absolute_map_peak_kib=peer,
                map_incremental_kib=increments,absolute_rss_cap_kib=cap,
                absolute_rss_cap_met=peak<=cap,absolute_rss_le_butter=peak<=peer,
                incremental_rss_cap_met=max(increments)<=cap,
                qualification_allowed=False,exploratory=True,accepted_rounds=32,
                excluded_rounds=complete['rounds_excluded'])
            summary[model][geometry]=record
            write(dest/'summary.json',record)
            progress(f'Matched map cost {model}/{geometry}: '+json.dumps(record,separators=(',',':')))
    write(analysis/'SUMMARY.json',dict(cells=summary,qualification_allowed=False,
        p95_owner_sha256=sha(binary),worker_sha256=job['binary_sha256']))


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['candidate-trace','peer-replay','rank-pairs','rank-report','curve-control','trace-report','full-trace','rank-score','rank-score-resume','cost-payload','cost-report','cost-summary-report','unpacked-control','unpacked-report'])
    p.add_argument('--binary',default='/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation/bin/bake_verdict')
    p.add_argument('--trace',action='store_true');p.add_argument('--checkpoint',choices=['best','last'],default='best');a=p.parse_args()
    if a.mode=='unpacked-report':unpacked_report(a.checkpoint)
    elif a.mode=='unpacked-control':unpacked_control(a.binary,a.checkpoint)
    elif a.mode=='rank-report':rank_report()
    elif a.mode=='cost-summary-report':cost_summary_report()
    elif a.mode=='cost-report':cost_report(a.binary)
    elif a.mode=='cost-payload':cost_payload()
    elif a.mode=='rank-score':rank_score()
    elif a.mode=='rank-score-resume':rank_score(True)
    elif a.mode=='rank-pairs':rank_pairs()
    elif a.mode=='curve-control':curve_control(a.binary)
    elif a.mode=='trace-report':trace_report()
    elif a.mode=='full-trace':
        owner(a.binary,True,False)
        trace_report()
    else:owner(a.binary,a.trace,a.mode=='peer-replay')


if __name__=='__main__':main()
