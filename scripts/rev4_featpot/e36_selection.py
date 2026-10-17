"""Future label-free raw-margin selection; never fits or decodes human labels."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq
from e36_store import sha
from e36_data import canonical


def write(path,value):
    with path.open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def score(binary,manifest,bake,out):
    m=json.loads(manifest.read_text())
    if m['schema']!='e36-tv-feature-only-v1' or m['split']!='selection' or not m['complete'] or m['labels_read'] is not False:
        raise ValueError('strict feature-only TRAIN selection pool required')
    for key in ['features','references','required_support','pairs','edge_proof','runtime','source_budget','derived','cohort']:
        if sha(m[key]['path'])!=m[key]['sha256']:raise ValueError('selection input binding changed: '+key)
    support=json.loads(Path(m['required_support']['path']).read_text())
    if not support['admitted'] or support['zero_support_required_strata']!=0:raise ValueError('required support incomplete')
    refs={r['sha256']:r for r in json.loads(Path(m['references']['path']).read_text())['rows'] if r['split']=='selection'}
    schema=pq.ParquetFile(m['features']['path']).schema_arrow
    if schema.names!=['row_idx','reference_sha256','encoded_sha256','features']:raise ValueError('feature-only schema required')
    out.mkdir();pred=out/'raw.jsonl';count=0
    with pred.open('x') as output,(out/'stderr.log').open('x') as log:
        process=subprocess.Popen([str(binary)],stdin=subprocess.PIPE,stdout=output,stderr=log,text=True)
        try:
            process.stdin.write(canonical(dict(bake=str(bake),bake_sha256=sha(bake),feature_ids=m['feature_ids']))+'\n')
            for batch in pq.ParquetFile(m['features']['path']).iter_batches(batch_size=256):
                for row in batch.to_pylist():
                    ref=refs.get(row['reference_sha256'])
                    if ref is None or row['row_idx']!=count:raise ValueError('unknown/fit/noncontiguous selection feature')
                    process.stdin.write(canonical(dict(row_idx=count,width=ref['width'],height=ref['height'],features=row['features']))+'\n');count+=1
            process.stdin.close()
            if process.wait():raise ValueError('native raw selection scorer failed; see stderr.log')
        except BaseException:
            if process.poll() is None:process.terminate();process.wait()
            raise
    write(out/'receipt.json',dict(schema='e36-raw-selection-forward-v1',manifest=str(manifest),manifest_sha256=sha(manifest),
        bake_sha256=sha(bake),binary_sha256=sha(binary),output=str(pred),output_sha256=sha(pred),rows=count,
        owner='BakeScorer.without_output_calibration; production fx1/scaler/head',labels_read=False,fits=0))


def select(request,out):
    """Freeze selection before any paired human assessment can be admitted."""
    import math
    req=json.loads(request.read_text())
    if req['schema']!='e36-global-train-selection-job-v1' or len(req['models'])!=9:raise ValueError('three arms times three full-data seeds required')
    manifest=Path(req['manifest']);m=json.loads(manifest.read_text())
    if sha(manifest)!=req['manifest_sha256'] or m['split']!='selection':raise ValueError('selection pool changed')
    if sha(m['pairs']['path'])!=m['pairs']['sha256']:raise ValueError('selection pairs changed')
    plan_path=Path(req['fit_plan']);plan=json.loads(plan_path.read_text())
    if sha(plan_path)!=req['fit_plan_sha256'] or plan['schema']!='e36-future-fit-plan-v1' or plan['fit_release'] is not False or len(plan['cells'])!=129:raise ValueError('frozen129-cell E36 plan required')
    if any(plan[k]!=req[k] for k in ['fit_manifest_sha256','contract_sha256','base_filter_sha256','trainer_sha256','raw_binary_sha256']) or plan['selection_manifest_sha256']!=req['manifest_sha256']:raise ValueError('selection pool/build/mask differs from frozen129-cell plan')
    if plan['selection_owner_sha256']!=sha(Path(__file__)) or plan['inventory_owner_sha256']!=sha(Path(__file__).with_name('e36_inventory.py')):raise ValueError('frozen selector/inventory owner changed')
    from e36_inventory import admit_inventory
    admit_inventory(plan['cells'])
    expected={r['name']:r for r in plan['cells']}
    models={};keys=set()
    for model in req['models']:
        arm,seed=model['arm'],model['seed_index'];key=(arm,seed)
        if arm not in ['tv0-c','tv1-c','tv10-c'] or seed not in [0,1,2] or key in keys:raise ValueError('model inventory differs')
        keys.add(key)
        if model.get('state','complete')!='complete':
            if model['state'] not in ['failed','unavailable'] or 'bake_sha256' not in model or model['bake_sha256'] is not None:raise ValueError('explicit unavailable model state required')
            evidence=Path(model['ledger']);phase=plan['manifests']['e36-'+arm+'-full'];mp=Path(phase['manifest_path'])
            if sha(evidence)!=model['ledger_sha256'] or sha(mp)!=phase['manifest_sha256'] or sha(req['ctl'])!=plan['ctl_sha256']:raise ValueError('unavailable native ledger/manifest/owner changed')
            desired=[r for r in json.loads(mp.read_text()) if r['cell']['image_path']=='e36-'+arm+'__N/full_s'+str(seed)]
            if len(desired)!=1:raise ValueError('unavailable slot absent from frozen native manifest')
            subset=out.parent/(out.name+'-'+arm+'-'+str(seed)+'-unavailable-manifest.json');write(subset,desired)
            native=subprocess.run([str(req['ctl']),'catalog','--manifest',str(subset),'--ledger',str(evidence)],text=True,capture_output=True,check=True)
            catalog=json.loads(native.stdout)
            if sum(v['total'] for v in catalog)!=1 or sum(v['done'] for v in catalog)!=0 or sum(v['poison'] for v in catalog)!=1:raise ValueError('unavailable model requires terminal native failure, not a pending/gap slot')
            models[key]=None;continue
        fleet_path=Path(model['fleet_receipt']);fleet=json.loads(fleet_path.read_text());binding_path=Path(model['model_binding']);binding=json.loads(binding_path.read_text())
        cell='e36-'+arm+'__N/full_s'+str(seed);declared=expected.get(cell)
        if (sha(fleet_path)!=model['fleet_receipt_sha256'] or sha(binding_path)!=model['model_binding_sha256']
            or declared is None or declared['kind']!='full' or declared['arm']!=arm
            or fleet['schema']!='zenfleet-fit-cell-receipt-v1' or fleet['program_sha']!=plan['program_sha256'] or fleet['data_sha']!=plan['data_sha256']
            or fleet['argv_sha']!=declared['argv_sha256'] or fleet['cell']!=cell or fleet['files'].get('e36-model-binding.json')!=sha(binding_path)
            or binding['schema']!='e36-fit-model-binding-v1' or binding['contains_labels'] is not False or binding['label_payload_read_by_projection'] is not False
            or binding['role']!='full-data' or binding['heldout'] is not None or binding['arm']!=arm or binding['seed_index']!=seed or binding['selected_epoch']!=119
            or binding['program_sha256']!=plan['program_sha256'] or binding['packed_model_sha256']!=model['bake_sha256']
            or binding['manifest_sha256']!=req['fit_manifest_sha256'] or binding['contract_sha256']!=req['contract_sha256'] or binding['base_filter_sha256']!=req['base_filter_sha256']
            or binding['trainer_sha256']!=req['trainer_sha256'] or binding['tv_weight']!={'tv0-c':0.,'tv1-c':1.,'tv10-c':10.}[arm]):
            raise ValueError('fresh matched full-data model/native receipt/TV binding differs')
        qualification=Path(model['qualification']);qualified=json.loads(qualification.read_text())
        if (sha(qualification)!=model['qualification_sha256'] or fleet['files'].get('e36-qualified-checkpoint.json')!=sha(qualification)
            or qualified.get('schema')!='e36-qualified-checkpoint-v1' or qualified.get('qualified') is not True
            or qualified.get('execution_contract')!='registered-fit' or qualified.get('cell')!=cell
            or qualified.get('program_sha256')!=plan['program_sha256'] or qualified.get('data_sha256')!=plan['data_sha256']
            or qualified.get('argv_sha256')!=declared['argv_sha256'] or qualified.get('result_sha256')!=fleet['result_sha']
            or qualified.get('raw_checkpoint_sha256')!=binding['raw_checkpoint_sha256']
            or qualified.get('raw_checkpoint_sha256')!=fleet['selected_bake_sha'] or qualified.get('packed_model_sha256')!=model['bake_sha256']
            or qualified.get('model_binding_sha256')!=sha(binding_path)
            or any(qualified.get(k)!=v for k,v in plan['qualification_owners'].items())):
            raise ValueError('successful native decoded-checkpoint qualification required before selection')
        rp=Path(model['raw_receipt']);r=json.loads(rp.read_text())
        if sha(rp)!=model['raw_receipt_sha256'] or r['manifest_sha256']!=req['manifest_sha256'] or r['bake_sha256']!=model['bake_sha256'] or r['binary_sha256']!=req['raw_binary_sha256'] or r['labels_read'] is not False or r['fits']!=0 or sha(r['output'])!=r['output_sha256']:
            raise ValueError('raw model/selection binding changed')
        values=[]
        with Path(r['output']).open() as f:
            for i,line in enumerate(f):
                v=json.loads(line)
                if v['row_idx']!=i or not math.isfinite(v['raw_distance']):raise ValueError('unordered/nonfinite prediction')
                values.append(v['raw_distance'])
        if len(values)!=m['feature_rows']:raise ValueError('missing raw predictions')
        models[key]=values
    available={arm:all(models[arm,s] is not None for s in range(3)) for arm in ['tv0-c','tv1-c','tv10-c']}
    stats={arm:dict(state='complete' if available[arm] else 'unavailable',weight=0.,margin_mass=0.,order_mass=0.) for arm in available}
    for batch in pq.ParquetFile(m['pairs']['path']).iter_batches(batch_size=4096):
        for edge in batch.to_pylist():
            a,b=edge['worse_idx'],edge['better_idx'];cl=edge['pair_weight_class']
            if a==b or not 0<=a<m['feature_rows'] or not 0<=b<m['feature_rows'] or cl not in [0,1]:raise ValueError('invalid selection edge')
            weight=2. if cl==0 else 1.
            for arm,totals in stats.items():
                if not available[arm]:continue
                for seed in range(3):
                    values=models[arm,seed];delta=values[a]-values[b]
                    totals['weight']+=weight;totals['margin_mass']+=weight*(delta>=.05);totals['order_mass']+=weight*(delta>0.)
    for totals in stats.values():
        if totals['state']!='complete':continue
        if totals['weight']<=0:raise ValueError('empty selection support')
        totals['margin_fraction']=totals['margin_mass']/totals['weight'];totals['strict_order_fraction']=totals['order_mass']/totals['weight']
    # Guard outputs must come from the unchanged engineering owners; no new
    # thresholds or assessment result enters the predeclared tie-break.
    guards=Path(req['engineering_guards']);g=json.loads(guards.read_text())
    if sha(guards)!=req['engineering_guards_sha256'] or g['schema']!='e36-frozen-selection-guards-v1' or g['manifest_sha256']!=req['manifest_sha256'] or g['model_bindings']!=[{k:v[k] for k in ['arm','seed_index','bake_sha256']} for v in req['models']]:raise ValueError('selection guard model binding')
    if any(g['passed'][arm] is True and not available[arm] for arm in ['tv1-c','tv10-c']):raise ValueError('unavailable full-data arm cannot pass selection guards')
    eligible=[arm for arm in ['tv1-c','tv10-c'] if available['tv0-c'] and available[arm] and g['passed'][arm] is True]
    selected=min(eligible,key=lambda arm:(-stats[arm]['margin_fraction'],-stats[arm]['strict_order_fraction'],1 if arm=='tv1-c' else 10)) if eligible else None
    write(out,dict(schema='e36-global-train-selection-v1',selected_arm=selected,selection_locked_before_assessment=True,
        request_sha256=sha(request),manifest_sha256=req['manifest_sha256'],fit_plan_sha256=req['fit_plan_sha256'],program_sha256=plan['program_sha256'],base_filter_sha256=req['base_filter_sha256'],trainer_sha256=req['trainer_sha256'],raw_binary_sha256=req['raw_binary_sha256'],engineering_guards_sha256=req['engineering_guards_sha256'],
        model_bindings=[{k:v[k] for k in ['arm','seed_index','bake_sha256']} for v in req['models']],
        stats=stats,raw_margin=.05,aggregation='registered codec-weighted physical edges pooled over strata; three equally weighted full-data seeds',
        tie_break='margin fraction, strict-order fraction, smaller lambda',labels_read=False,fits=0))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='mode',required=True)
    s=sub.add_parser('score');s.add_argument('--binary',type=Path,required=True);s.add_argument('--manifest',type=Path,required=True);s.add_argument('--bake',type=Path,required=True);s.add_argument('--out',type=Path,required=True)
    s=sub.add_parser('select');s.add_argument('--request',type=Path,required=True);s.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    if a.mode=='score':score(a.binary,a.manifest,a.bake,a.out)
    else:select(a.request,a.out)
