"""Frozen existing-owner guard computation, with invented identity outputs."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
import e36_selection_guards as owner
from e36_store import sha


def write(p,x):p.write_text(json.dumps(x))


class Guards(unittest.TestCase):
    def fixture(self,root):
        bind=lambda p:dict(path=str(p),sha256=sha(p))
        inputs={}
        for key in ['binary','sources','fx1','production']:
            p=root/key;p.write_bytes(b'invented owner/input');inputs[key]=bind(p)
        policy=dict(schema='e36-train-selection-guard-policy-v1',selection_has_human_ranks=False,
            required_guards=['native_raw_forward_finite','existing_e33_exact_identity'],
            identity_expected_rows=620,identity_expected_sources=62,
            identity_pin_bits=struct.unpack('<Q',struct.pack('<d',100.))[0],
            admission_owner_sha256=sha(Path(owner.__file__)),**inputs)
        path=root/'policy.json';write(path,policy)
        models=[];values={}
        for arm in ['tv0-c','tv1-c','tv10-c']:
            for seed in range(3):
                bake=root/(arm+str(seed)+'.bin');bake.write_bytes(b'invented not a model')
                models.append(dict(arm=arm,seed_index=seed,bake=str(bake),bake_sha256=sha(bake)))
                values[arm,seed]=[.25,.125]
        plan=dict(selection_guard_policy=bind(path))
        return plan,dict(models=models),values,policy

    def test_computed_bits_not_supplied_pass_and_exact_required_policy(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);plan,request,values,policy=self.fixture(root)
            def native(argv,**kwargs):
                out=Path(argv[5]);bake=Path(argv[6])
                bad=bake.name=='tv10-c2.bin'
                bits=policy['identity_pin_bits']-(1 if bad else 0)
                with out.open('x') as f:
                    for i in range(1 if bad else 620):f.write(json.dumps(dict(reference=str(i),candidate_score_bits=[bits]))+'\n')
                    if not bad:f.write(json.dumps(dict(rows=620,sources=62,sources_sha256=policy['sources']['sha256'],
                        fx1_sha256=policy['fx1']['sha256'],production_model_sha256=policy['production']['sha256'],
                        candidates=[dict(sha256=sha(bake))]))+'\n')
                return SimpleNamespace(returncode=101 if bad else 0)
            request['engineering_passed']={'tv1-c':False,'tv10-c':True} # must have no effect
            with patch.object(owner.subprocess,'run',side_effect=native):
                passed,receipt=owner.evaluate(plan,request,values,root/'guards')
            self.assertTrue(passed['tv1-c']);self.assertFalse(passed['tv10-c'])
            self.assertFalse(json.loads(Path(receipt['path']).read_text())['contains_human_ranks'])
            path=Path(plan['selection_guard_policy']['path']);bad=copy.deepcopy(policy);bad['required_guards'].append('new-cross-reference-guard')
            write(path,bad);plan['selection_guard_policy']['sha256']=sha(path)
            with self.assertRaisesRegex(ValueError,'unregistered'):owner.evaluate(plan,request,values,root/'altered-policy')

    def test_missing_rows_are_incomplete_not_a_passing_boolean(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as d:
            root=Path(d);plan,request,values,policy=self.fixture(root)
            def incomplete(argv,**kwargs):
                Path(argv[5]).write_text(json.dumps(dict(reference='one',candidate_score_bits=[policy['identity_pin_bits']]))+'\n')
                return SimpleNamespace(returncode=0)
            with patch.object(owner.subprocess,'run',side_effect=incomplete):
                with self.assertRaisesRegex(ValueError,'evidence incomplete'):owner.evaluate(plan,request,values,root/'guards')


if __name__=='__main__':unittest.main()
