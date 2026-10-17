"""Run E36 ingress refusals using invented feature-only Parquet rows; no fit."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import pyarrow as pa
import pyarrow.parquet as pq


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')


def parquet(table,path):
    pq.write_table(table,path,compression='zstd')


def run(binary,layout,out):
    out.mkdir()
    binding=json.loads(layout.read_text());fx=out/'fx1.json'
    shutil.copyfile(Path(__file__).with_name('e33_fx1_declaration.json'),fx)
    root=layout.parent
    refdoc=json.loads((root/'references.json').read_text())
    fit_ref=next(r['sha256'] for r in refdoc['rows'] if r['split']=='fit')
    selection_ref=next(r['sha256'] for r in refdoc['rows'] if r['split']=='selection')
    support=out/'support.json'
    ids=binding['consumed_ids'];assert len(ids)==420
    features=pa.table({'row_idx':pa.array([0,1,2],type=pa.uint64()),
        'reference_sha256':[fit_ref]*3,'encoded_sha256':['b'*64,'c'*64,'d'*64],
        'features':pa.array([[float(i+1)/10]*len(ids) for i in range(3)],type=pa.list_(pa.float32()))})
    pairs=pa.table({'worse_idx':pa.array([0,0],type=pa.uint64()),'better_idx':pa.array([1,2],type=pa.uint64()),
        'pair_weight_class':pa.array([0,1],type=pa.uint8()),'q_band':pa.array([0,2],type=pa.uint8())})
    fp=out/'features.parquet';pp=out/'pairs.parquet';parquet(features,fp);parquet(pairs,pp)
    proof=pa.table({'worse_idx':pa.array([0,0],type=pa.uint64()),'better_idx':pa.array([1,2],type=pa.uint64()),
        'reference_sha256':[fit_ref]*2,'codec':['jpeg','webp'],'size':pa.array([32,32],type=pa.uint32()),
        'mode_key':['{}']*2,'q_low':pa.array([0,62],type=pa.uint8()),'q_high':pa.array([16,64],type=pa.uint8()),
        'worse_encoded_sha256':['b'*64]*2,'better_encoded_sha256':['c'*64,'d'*64],
        'consensus_receipt_sha256':['e'*64]*2})
    ep=out/'edge-proof.parquet';parquet(proof,ep)
    bind=lambda p:dict(path=str(p),sha256=sha(p))
    provenance=dict(references=bind(root/'references.json'),source_budget=bind(root/'source-budget/plan.json'),
        cohort=bind(root/'cohort-selection.json'),runtime=bind(root/'runtime-v4/runtime.json'))
    ma=out/'matrix-admission.json';write(ma,dict(schema='e36-qualified-matrix-admission-v1',complete=True,matrix='fallback',outputs=['e'*64]))
    matrix=out/'whole-matrix.json';write(matrix,dict(admission=bind(ma),schema='e36-whole-matrix-completion-v1',matrix='fallback',
        complete=True,runtime_sha256=provenance['runtime']['sha256'],source_budget_sha256=provenance['source_budget']['sha256'],outputs=['e'*64]))
    report=dict(schema='e36-required-support-v1',admitted=True,zero_support_required_strata=0,
        features_sha256=sha(fp),pairs_sha256=sha(pp),edge_proof_sha256=sha(ep),source_budget_sha256=provenance['source_budget']['sha256'],
        references_sha256=provenance['references']['sha256'],cohort_sha256=provenance['cohort']['sha256'],
        runtime_sha256=provenance['runtime']['sha256'],derived_sha256=sha(fx),feature_rows=3,pair_rows=2,
        teachers=['butteraugli_pnorm3','cvvdp@standard_fhd'],teacher_margins=[.01,.01],whole_matrix=bind(matrix),
        required_strata={'invented':dict(selected_references=1,unique_preferences=2)})
    write(support,report)
    base=dict(schema='e36-tv-feature-only-v1',protocol_build='e36_tv_v1',labels_read=False,fits=0,
        complete=True,split='fit',zero_dropped_pairs=True,mandatory_matrix_complete=True,required_support=bind(support),
        features=bind(fp),pairs=bind(pp),edge_proof=bind(ep),derived=bind(fx),feature_ids=ids,raw_width=1853,feature_rows=3,pair_rows=2,
        **provenance,source_budget_sha256=provenance['source_budget']['sha256'],generation_runtime_sha256=provenance['runtime']['sha256'])
    cases=[('valid',True,base)]
    for key,value in [('split','selection'),('labels_read',True),('complete',False),('zero_dropped_pairs',False),
        ('mandatory_matrix_complete',False),('feature_rows',4),('pair_rows',3),('schema','legacy')]:
        m=copy.deepcopy(base);m[key]=value;cases.append((key,False,m))
    m=copy.deepcopy(base);m['feature_ids']=ids[::-1];cases.append(('layout-order',False,m))
    m=copy.deepcopy(base);m['features']['sha256']='0'*64;cases.append(('digest',False,m))
    m=copy.deepcopy(base);m['unexpected_label_path']='never-opened';cases.append(('unknown-field',False,m))
    for key,values in [('better_idx',[1,99]),('pair_weight_class',[0,2]),('q_band',[0,4]),('better_idx',[1,1])]:
        name=key+'-'+str(values[-1]);p=out/(name+'.parquet');col=pairs.schema.get_field_index(key)
        parquet(pairs.set_column(col,pairs.schema.field(col),pa.array(values,type=pairs.schema.field(col).type)),p)
        m=copy.deepcopy(base);m['pairs']=bind(p);cases.append((name,False,m))
    p=out/'extra-column.parquet';parquet(features.append_column('human_score',pa.array([.1,.2,.3])),p)
    m=copy.deepcopy(base);m['features']=bind(p);cases.append(('label-column',False,m))
    p=out/'nonfinite.parquet';vals=[[.1]*420 for _ in range(3)];vals[1][0]=float('nan')
    parquet(features.set_column(3,features.schema.field(3),pa.array(vals,type=pa.list_(pa.float32()))),p)
    m=copy.deepcopy(base);m['features']=bind(p);cases.append(('nonfinite',False,m))
    # Rebind report to each candidate's exact files, so semantic refusals are
    # exercised beyond the top-level digest-identity gate.
    for name,expected,m in cases:
        r=copy.deepcopy(report);r['features_sha256']=m['features']['sha256'];r['pairs_sha256']=m['pairs']['sha256']
        sp=out/(name+'-support.json');write(sp,r);m['required_support']=bind(sp)
    for name,ref in [('selection-family',selection_ref),('unknown-reference','0'*64)]:
        p=out/(name+'.parquet');parquet(features.set_column(1,features.schema.field(1),pa.array([ref]*3)),p)
        m=copy.deepcopy(base);m['features']=bind(p);r=copy.deepcopy(report);r['features_sha256']=sha(p)
        sp=out/(name+'-support.json');write(sp,r);m['required_support']=bind(sp);cases.append((name,False,m))
    results=[]
    for name,expected,m in cases:
        path=out/(name+'.json');write(path,m)
        semantic=out/(name+'-semantic.json');write(semantic,dict(schema='e36-semantic-export-admission-v1',admitted=True,split='fit',labels_read=False,fits=0,features=m['features'],pairs=m['pairs'],edge_proof=m['edge_proof'],support=m['required_support'],feature_rows=m['feature_rows'],pair_rows=m['pair_rows'],whole_matrix=bind(matrix)))
        contract=out/(name+'-contract.json');write(contract,dict(semantic_export=bind(semantic),matrix_admission=bind(ma),schema='e36-tv-locked-contract-v1',manifest_sha256=sha(path),protocol_build='e36_tv_v1',fits_released=False))
        argv=[str(binary),'--e36-tv-admit','--e36-tv-manifest',str(path),'--e36-tv-contract',str(contract),
            '--e36-tv-contract-sha256',sha(contract),'--derived-inputs',str(fx),
            '--max-features','1853','--tv-weight','1']
        r=subprocess.run(argv,capture_output=True,text=True)
        (out/(name+'.stdout')).write_text(r.stdout);(out/(name+'.stderr')).write_text(r.stderr)
        if (r.returncode==0)!=expected:raise ValueError(f'{name}: incorrect admission result rc{r.returncode}: {r.stderr}')
        if expected:
            receipt=json.loads(r.stdout.strip().splitlines()[-1]);assert receipt['optimizer_steps']==0 and receipt['labels_read'] is False and receipt['dropped_pairs']==0
            from e36_training import protocol_source_sha
            assert receipt['binary_sha256']==sha(binary) and receipt['cfg_e36_tv'] is True
            assert receipt['embedded_protocol_source_sha256']==protocol_source_sha(Path(__file__).resolve().parents[2])
            assert receipt['output_polarity']=='score' and receipt['raw_definition']=='pin-g'
            assert receipt['improvement']=='raw(better)-raw(worse)'
        results.append(dict(case=name,expected_accept=expected,returncode=r.returncode,manifest_sha256=sha(path)))
    write(out/'result.json',dict(schema='e36-invented-ingress-fixtures-v1',fits=0,optimizer_steps=0,
        real_labels_read=False,binary_sha256=sha(binary),cases=results,passed=True))
    print(json.dumps(dict(passed=len(results),optimizer_steps=0,real_labels_read=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--layout',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    run(a.binary,a.layout,a.out)
