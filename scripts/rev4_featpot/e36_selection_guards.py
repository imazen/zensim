"""Invoke the existing identity owner; selection has no human-rank guard."""
import json
import os
from pathlib import Path
import struct
import subprocess

from e36_store import sha


def evaluate(plan, request, values, directory):
    binding=plan['selection_guard_policy'];path=Path(binding['path'])
    if sha(path)!=binding['sha256']:raise ValueError('frozen selection guard policy changed')
    policy=json.loads(path.read_text())
    if (policy['schema']!='e36-train-selection-guard-policy-v1'
            or policy['selection_has_human_ranks'] is not False
            or policy['required_guards']!=['native_raw_forward_finite','existing_e33_exact_identity']
            or policy['identity_expected_rows']!=620 or policy['identity_expected_sources']!=62
            or policy['identity_pin_bits']!=struct.unpack('<Q',struct.pack('<d',100.))[0]
            or policy['admission_owner_sha256']!=sha(Path(__file__))):
        raise ValueError('unregistered E36 TRAIN guard computation/threshold')
    for key in ['binary','sources','fx1','production']:
        item=policy[key]
        if sha(item['path'])!=item['sha256']:raise ValueError('frozen identity owner/input changed: '+key)
    directory.mkdir();passed={};receipts=[]
    for model in request['models']:
        key=(model['arm'],model['seed_index']);slots=values[key]
        if slots is None:
            passed[key]=False;continue
        bake=Path(model['bake'])
        if sha(bake)!=model['bake_sha256']:raise ValueError('identity model differs from qualified packed model')
        out=directory/(model['arm']+'-'+str(model['seed_index'])+'.jsonl')
        argv=[policy['binary']['path'],'--e33-identity',policy['sources']['path'],policy['fx1']['path'],
              policy['production']['path'],str(out),str(bake)]
        with out.with_suffix('.log').open('x') as log:
            rc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,env={**os.environ,'ZENSIM_FORMULA_REV':'5'}).returncode
        rows=[];tail=[]
        if out.exists():
            for line in out.read_text().splitlines():
                r=json.loads(line)
                if 'reference' in r:rows.append(r)
                else:tail.append(r)
        wrong_identity=any(r.get('candidate_score_bits')!=[policy['identity_pin_bits']] for r in rows)
        if rc:
            # The unchanged native owner prints the wrong candidate bits before
            # its exact-identity assertion. Other process failures are incomplete
            # computation, never invented guard outcomes.
            if not wrong_identity:raise ValueError('identity owner incomplete; inspect retained log')
            ok=False
        else:
            if (len(rows)!=620 or len(tail)!=1 or tail[0].get('rows')!=620 or tail[0].get('sources')!=62
                    or tail[0].get('sources_sha256')!=policy['sources']['sha256']
                    or tail[0].get('fx1_sha256')!=policy['fx1']['sha256']
                    or tail[0].get('production_model_sha256')!=policy['production']['sha256']
                    or [v['sha256'] for v in tail[0].get('candidates',[])]!=[model['bake_sha256']]):
                raise ValueError('identity owner roster/model/policy evidence incomplete')
            ok=not wrong_identity
        passed[key]=ok
        receipts.append(dict(arm=key[0],seed_index=key[1],model_sha256=model['bake_sha256'],
            argv=argv,returncode=rc,output_sha256=sha(out),identity_passed=ok,finite_raw_rows=len(slots)))
    result={arm:all(passed[arm,seed] for seed in range(3)) for arm in ['tv0-c','tv1-c','tv10-c']}
    record=dict(schema='e36-computed-train-selection-guards-v1',policy_sha256=binding['sha256'],
        passed=result,slots=receipts,contains_human_ranks=False,
        cross_reference_ranking='assessment: registered noninferiority, dataset point and W2 guards',labels_read=False)
    output=directory/'receipt.json'
    with output.open('x') as f:json.dump(record,f,indent=2);f.write('\n')
    return result,dict(path=str(output),sha256=sha(output))
