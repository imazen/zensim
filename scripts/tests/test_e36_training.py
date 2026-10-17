"""Matched source filtering and immutable argv, with invented targets only."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_training as owner
from e36_store import sha
from v2_teacher import key_path,row_keys_sha


def write(p,x):p.write_text(json.dumps(x))

class TrainingAdapter(unittest.TestCase):
    def temporary(self):return tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-base-invented-')
    def test_one_mask_preserves_targets_features_order_and_all_group_weights(self):
        with self.temporary() as d:
            root=Path(d);inputs=root/'inputs';inputs.mkdir();scratch=root/'scratch';scratch.mkdir();source=inputs/'source.parquet'
            refs=['safesyn:1fb72c0d6d4352c8_512sq','safesyn:ffffffffffffffff_512sq','safesyn:1fb72c0d6d4352c8_769x513']
            table=pa.table({'ref_basename':refs,'f0':pa.array([.25,-.5,2.],type=pa.float32()),'human_score':pa.array([-17.125,.4375,92.],type=pa.float64())})
            pq.write_table(table,source,compression='zstd')
            keys=pa.table(dict(pair_key=['a','b','c'],ref_basename=refs,member_set=['safesyn']*3,row_id=pa.array([0,1,2],type=pa.uint64())))
            pq.write_table(keys,key_path(source),compression='zstd')
            declaration=dict(feature_set_id='basic+peaks+v2@w1825/rev5_localwin#36c3f3af',formula_revision=5,decoder_era='invented-fixture',rows=3,
                keys_sha256=sha(key_path(source)),row_keys_sha256=row_keys_sha(keys),table_sha256=sha(source),row_selection_sha256='invented-original-selection',data_role='TRAIN oracle teacher',immutable_input_roots=[str(inputs)])
            write(Path(str(source)+'.manifest.json'),declaration)
            plan=root/'plan.json';write(plan,dict(schema='e36-matched-base-filter-v2',identity_normalization='member-prefix+rendition-v2',complete=True,labels_read=False,excluded_base_ids=['1fb72c0d6d4352c8'],groups=[dict(path=str(source),rows=3,table_sha256=sha(source),manifest_sha256=sha(Path(str(source)+'.manifest.json')),kept_indices_sha256=hashlib.sha256(np.array([1],dtype='<u8').tobytes()).hexdigest(),keys_sha256=sha(key_path(source)),row_keys_sha256=row_keys_sha(keys),removed=2)]))
            recipe=dict(e36=dict(base_filter=str(plan),base_filter_sha256=sha(plan)))
            original=[('safesyn',source,1.25,.625,'both')];groups=copy.deepcopy(original)
            with patch.dict(os.environ,{},clear=True):
                with self.assertRaisesRegex(ValueError,'only inside'):owner.filter_groups(groups,recipe,scratch)
            # Synthetic fixture authority, never a real fit or label payload.
            with patch.dict(os.environ,{'ZEN_FIT_PROGRAM_SHA':'a'*64}):
                receipt=owner.filter_groups(groups,recipe,scratch)
                self.assertEqual(owner.calibration_anchor(source,recipe),groups[0][1])
                with self.assertRaisesRegex(ValueError,'exact matched'):owner.calibration_anchor(inputs/'unknown.parquet',recipe)
                again=copy.deepcopy(original);owner.filter_groups(again,recipe,scratch)
            # Freeze the child manifest from keys only; it must exactly match
            # the existing unchanged-target writer's actual metadata.
            reader=pq.read_table
            def keys_only(path,*a,**k):
                if Path(path)==source:raise AssertionError('projection decoded target table')
                return reader(path,*a,**k)
            with patch.object(pq,'read_table',side_effect=keys_only):
                projected=owner.curated_projection(plan,root/'projection')
            actual=copy.deepcopy(receipt['groups'][0]['stored_manifest']);actual.pop('table_sha256')
            self.assertEqual(projected[sha(source)]['manifest'],actual)
            self.assertEqual(groups,again);self.assertEqual(groups[0][0],original[0][0]);self.assertEqual(groups[0][2:],original[0][2:])
            self.assertEqual(pq.read_table(groups[0][1]).to_pydict(),table.slice(1,1).to_pydict())
            self.assertEqual(receipt['groups'][0]['removed'],2);self.assertEqual(receipt['targets_changed'],0)
            from v2_lodo_mlp import strict_training_groups
            admitted=strict_training_groups(groups)
            self.assertEqual(len(admitted),1)
            old=json.loads(plan.read_text());old['schema']='e36-matched-base-filter-v1';write(plan,old);recipe['e36']['base_filter_sha256']=sha(plan)
            with patch.dict(os.environ,{'ZEN_FIT_PROGRAM_SHA':'a'*64}):
                with self.assertRaisesRegex(ValueError,'changed'):owner.filter_groups(copy.deepcopy(original),recipe,scratch)
    def test_assessment_inputs_freeze_order_without_targets(self):
        import e34_evaluate
        with self.temporary() as d:
            root=Path(d);real=root/'wide/main/real';real.mkdir(parents=True)
            source=real/'invented.parquet';source.write_bytes(b'opaque never decoded target fixture')
            keys=pa.table(dict(pair_key=['a','b','c'],ref_basename=['ref0','ref0','ref1'],member_set=['invented']*3,row_id=pa.array([0,1,2],type=pa.uint64())))
            pq.write_table(keys,key_path(source),compression='zstd')
            side=Path(str(source)+'.manifest.json');write(side,{'invented':'original metadata'})
            write(real/'receipt.json',dict(legs=dict(kadid=dict(full=dict(rel='wide/main/real/invented.parquet',
                sha256=sha(source),manifest_sha256=sha(side),keys_sha256=sha(key_path(source)),rows=3)))))
            reader=pq.read_table
            def keys_only(path,*a,**k):
                if Path(path)==source:raise AssertionError('assessment plan decoded target table')
                return reader(path,*a,**k)
            out=root/'assessment.json'
            with patch.object(e34_evaluate,'admit_tables',return_value=[('kadid',source,[0,0,1],[0,1,0])]),patch.object(pq,'read_table',side_effect=keys_only):
                owner.assessment_plan(root,out)
            frozen=json.loads(out.read_text());r=frozen['tables']['kadid']
            self.assertFalse(frozen['labels_read']);self.assertEqual(r['reference'],[0,0,1]);self.assertEqual(r['codec'],[0,1,0])
            self.assertEqual(r['row_keys_sha256'],row_keys_sha(keys));self.assertEqual(r['table_sha256'],sha(source))
            self.assertEqual(r['orientation'],1.);self.assertEqual(r['rows'],3)
    def test_three_fixed_arms_and_trainer_digest(self):
        with self.temporary() as d:
            root=Path(d);manifest=root/'manifest.json';write(manifest,{'invented':'feature-only'})
            mask=root/'mask.json';write(mask,dict(schema='e36-matched-base-filter-v2',identity_normalization='member-prefix+rendition-v2',labels_read=False,complete=True))
            trainer=root/'trainer';trainer.write_bytes(b'invented nonexecutable digest fixture')
            contract=root/'contract.json';write(contract,dict(manifest_sha256=sha(manifest),base_filter_sha256=sha(mask),trainer_sha256=sha(trainer)))
            args=argparse.Namespace(e36_arm='tv0-c',spec=owner.SPEC,head='N',strict_admission=True,train_only=True,e36_tv_manifest=manifest,e36_tv_contract=contract,e36_tv_contract_sha256=sha(contract),e36_base_filter_plan=mask,e36_base_filter_sha256=sha(mask))
            for arm,weight in [('tv0-c',0.),('tv1-c',1.),('tv10-c',10.)]:
                args.e36_arm=arm;recipe=owner.options(args,dict(hidden=128,derived_inputs='fx1'));flags=owner.command_flags(recipe,trainer)
                self.assertEqual(float(flags[flags.index('--tv-weight')+1]),weight)
                self.assertEqual(flags[-6:],['--tv-apply-every','50','--tv-batch','32','--tv-margin','0.05'])
            args.seed_index=0;args.heldout=None
            bake=root/'invented-checkpoint.bin';bake.write_bytes(b'invented-not-a-model')
            dest=root/'projection';dest.mkdir()
            with patch.dict(os.environ,{'ZEN_FIT_PROGRAM_SHA':'a'*64}):owner.model_binding(dest,recipe,args,1101,101,bake,dict(packed_model_sha256='f'*64,deployment_order='final epoch; canonical densify; f16 quantization; TRAIN calibration'),selection=dict(epoch_rule='last',selected_epoch=119))
            projection=json.loads((dest/'e36-model-binding.json').read_text())
            self.assertEqual(projection['raw_checkpoint_sha256'],sha(bake));self.assertEqual(projection['packed_model_sha256'],'f'*64)
            self.assertFalse(projection['contains_labels']);self.assertFalse(projection['label_payload_read_by_projection'])
            self.assertEqual(projection['selected_epoch'],119)
            args.local_smoke_budget='2:128';smoke_dest=root/'smoke-projection';smoke_dest.mkdir()
            with patch.dict(os.environ,{'ZEN_FIT_PROGRAM_SHA':'a'*64}):
                owner.model_binding(smoke_dest,recipe,args,1101,101,bake,selection=dict(epoch_rule='last',selected_epoch=1))
            self.assertEqual(json.loads((smoke_dest/'e36-model-binding.json').read_text())['selected_epoch'],1)
            args.local_smoke_budget=None
            with self.assertRaisesRegex(ValueError,'actual final registered'):
                owner.model_binding(root/'refused',recipe,args,1101,101,bake,selection=dict(epoch_rule='last',selected_epoch=1))
            trainer.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'actual trainer'):owner.command_flags(recipe,trainer)
            args.e36_arm=None;normal=dict(hidden=128);self.assertIs(owner.options(args,normal),normal)

if __name__=='__main__':unittest.main()
