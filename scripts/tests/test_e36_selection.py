"""Global selection with invented forward values; no models or fits."""
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_selection as owner
import e36_selection_guards as guard_owner
from e36_store import sha


def write(p,x):p.write_text(json.dumps(x))


class Selection(unittest.TestCase):
    def fixture(self,root,reverse_arm=None):
        pairs=root/'pairs.parquet';pq.write_table(pa.table(dict(worse_idx=[0],better_idx=[1],pair_weight_class=[0])),pairs)
        manifest=root/'manifest.json';write(manifest,dict(split='selection',feature_rows=2,pairs=dict(path=str(pairs),sha256=sha(pairs))))
        ctl=root/'invented-native-owner';ctl.write_bytes(b'invented-not-executable')
        fixed=dict(fit_manifest_sha256='f'*64,contract_sha256='c'*64,base_filter_sha256='b'*64,
                   trainer_sha256='a'*64,raw_binary_sha256='d'*64)
        models=[];cells=[];phases={}
        for arm,weight in [('tv0-c',0.),('tv1-c',1.),('tv10-c',10.)]:
            desired=[]
            for seed in range(3):
                cell='e36-'+arm+'__N/full_s'+str(seed);argv_sha=(str(seed)+str(int(weight))).ljust(64,'0')
                cells.append(dict(name=cell,arm=arm,kind='full',seed_index=seed,heldout=None,argv_sha256=argv_sha))
                desired.append(dict(cell=dict(image_path=cell)))
                folder=root/(arm+'-'+str(seed));folder.mkdir()
                raw=folder/'raw.jsonl';scores=[.25,.125] if arm==reverse_arm else [.125,.25]
                raw.write_text(json.dumps(dict(row_idx=0,raw_score=scores[0]))+'\n'+json.dumps(dict(row_idx=1,raw_score=scores[1]))+'\n')
                raw_receipt=folder/'raw-receipt.json';bake_sha=sha(raw)
                write(raw_receipt,dict(manifest_sha256=sha(manifest),bake_sha256=bake_sha,binary_sha256='d'*64,
                    labels_read=False,fits=0,score_convention='higher_is_better; plain N raw=pin-g',output=str(raw),output_sha256=sha(raw)))
                binding=folder/'binding.json';write(binding,dict(schema='e36-fit-model-binding-v1',contains_labels=False,
                    label_payload_read_by_projection=False,role='full-data',heldout=None,arm=arm,seed_index=seed,selected_epoch=119,
                    program_sha256='e'*64,packed_model_sha256=bake_sha,manifest_sha256='f'*64,contract_sha256='c'*64,
                    base_filter_sha256='b'*64,trainer_sha256='a'*64,tv_weight=weight,raw_checkpoint_sha256='2'*64))
                qualified=folder/'qualification.json';write(qualified,dict(schema='e36-qualified-checkpoint-v1',qualified=True,
                    execution_contract='registered-fit',cell=cell,program_sha256='e'*64,data_sha256='1'*64,argv_sha256=argv_sha,
                    result_sha256='3'*64,raw_checkpoint_sha256='2'*64,packed_model_sha256=bake_sha,model_binding_sha256=sha(binding),
                    qualification_owner_sha256='4'*64,e36_owner_sha256='5'*64,inspector_sha256='6'*64))
                fleet=folder/'fleet.json';write(fleet,dict(schema='zenfleet-fit-cell-receipt-v1',program_sha='e'*64,data_sha='1'*64,
                    result_sha='3'*64,selected_bake_sha='2'*64,argv_sha=argv_sha,cell=cell,
                    files={'e36-model-binding.json':sha(binding),'e36-qualified-checkpoint.json':sha(qualified)}))
                models.append(dict(arm=arm,seed_index=seed,bake_sha256=bake_sha,raw_receipt=str(raw_receipt),raw_receipt_sha256=sha(raw_receipt),
                    qualification=str(qualified),qualification_sha256=sha(qualified),fleet_receipt=str(fleet),fleet_receipt_sha256=sha(fleet),model_binding=str(binding),model_binding_sha256=sha(binding)))
            phase=root/(arm+'-manifest.json');write(phase,desired)
            phases['e36-'+arm+'-full']=dict(manifest_path=str(phase),manifest_sha256=sha(phase))
        cells.extend(dict(name='e36-'+arm+'__N/without_'+source+'_s'+str(seed),arm=arm,kind='lodo',heldout=source,seed_index=seed) for arm in ['tv0-c','tv1-c','tv10-c'] for source in ['kadid','tid2013','konfig','cid22_a25'] for seed in range(10))
        plan=root/'plan.json';write(plan,dict(schema='e36-future-fit-plan-v1',fit_release=False,cells=cells,**fixed,
            selection_manifest_sha256=sha(manifest),ctl_sha256=sha(ctl),program_sha256='e'*64,data_sha256='1'*64,manifests=phases,selection_owner_sha256=sha(Path(owner.__file__)),inventory_owner_sha256=sha(Path(owner.__file__).with_name('e36_inventory.py')),qualification_owners=dict(qualification_owner_sha256='4'*64,e36_owner_sha256='5'*64,inspector_sha256='6'*64)))
        guards=root/'guards.json';write(guards,dict(schema='e36-frozen-selection-guards-v1',manifest_sha256=sha(manifest),
            model_bindings=[{k:m[k] for k in ['arm','seed_index','bake_sha256']} for m in models],passed={'tv1-c':True,'tv10-c':True}))
        request=dict(schema='e36-global-train-selection-job-v1',models=models,manifest=str(manifest),manifest_sha256=sha(manifest),
            fit_plan=str(plan),fit_plan_sha256=sha(plan),**fixed,engineering_guards=str(guards),engineering_guards_sha256=sha(guards),ctl=str(ctl))
        return request

    def run_request(self,root,request,name):
        path=root/(name+'-request.json');write(path,request);out=root/(name+'-selected.json')
        with patch.object(guard_owner,'evaluate',return_value=({'tv0-c':True,'tv1-c':name!='no-eligible','tv10-c':True},{'invented_only':True})):owner.select(path,out)
        return json.loads(out.read_text())

    def test_equal_scores_choose_smaller_lambda_and_bind_fresh_models(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);request=self.fixture(root)
            selected=self.run_request(root,request,'tie')
            self.assertEqual(selected['selected_arm'],'tv1-c')
            for field in ['trainer_sha256','contract_sha256','base_filter_sha256','raw_binary_sha256']:
                bad=copy.deepcopy(request);bad[field]='0'*64
                with self.assertRaisesRegex(ValueError,'frozen129-cell'):self.run_request(root,bad,field)
            missing=copy.deepcopy(request);missing['models'][0]['qualification_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'qualification required'):self.run_request(root,missing,'unqualified')
            plan=Path(request['fit_plan']);original=json.loads(plan.read_text());badplan=copy.deepcopy(original)
            badplan['cells'][-1]=badplan['cells'][-2];write(plan,badplan)
            duplicate=copy.deepcopy(request);duplicate['fit_plan_sha256']=sha(plan)
            with self.assertRaisesRegex(ValueError,'duplicate/unregistered'):self.run_request(root,duplicate,'duplicate-slot')
            write(plan,original)
            # Even a rehashed model projection cannot replace the fresh control
            # with another executable or an older checkpoint epoch.
            m=request['models'][0];path=Path(m['model_binding']);binding=json.loads(path.read_text());binding['selected_epoch']=118
            write(path,binding);m['model_binding_sha256']=sha(path)
            with self.assertRaisesRegex(ValueError,'fresh matched'):self.run_request(root,request,'old-epoch')

    def test_raw_score_polarity_rejects_reversed_arm(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);request=self.fixture(root,reverse_arm='tv1-c')
            result=self.run_request(root,request,'polarity')
            self.assertEqual(result['selected_arm'],'tv10-c')
            self.assertEqual(result['stats']['tv1-c']['margin_fraction'],0.)
            self.assertEqual(result['stats']['tv10-c']['margin_fraction'],1.)

    def test_terminal_failed_arm_never_shrinks_inventory_or_substitutes(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);request=self.fixture(root)
            model=request['models'][-1];ledger=root/'terminal-ledger';ledger.write_bytes(b'invented terminal native receipt')
            model.update(state='failed',bake_sha256=None,ledger=str(ledger),ledger_sha256=sha(ledger))
            guards=Path(request['engineering_guards']);g=json.loads(guards.read_text());g['passed']['tv10-c']=False
            g['model_bindings']=[{k:m[k] for k in ['arm','seed_index','bake_sha256']} for m in request['models']]
            write(guards,g);request['engineering_guards_sha256']=sha(guards)
            with patch.object(owner.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps([dict(total=1,done=0,poison=1)]))):
                chosen=self.run_request(root,request,'failed-other')
                self.assertEqual(chosen['selected_arm'],'tv1-c');self.assertEqual(chosen['stats']['tv10-c']['state'],'unavailable')
                g['passed']['tv1-c']=False;write(guards,g);request['engineering_guards_sha256']=sha(guards)
                self.assertIsNone(self.run_request(root,request,'no-eligible')['selected_arm'])
            with patch.object(owner.subprocess,'run',return_value=SimpleNamespace(stdout=json.dumps([dict(total=1,done=0,poison=0)]))):
                with self.assertRaisesRegex(ValueError,'terminal native'):self.run_request(root,request,'pending')


if __name__=='__main__':unittest.main()
