"""Exact diagnostic identity uses metadata only, without a fit."""
import copy
from pathlib import Path
import sys
import unittest
import argparse
import tempfile
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_memory_smoke import seed_zero,diagnostic_result

class Diagnostic(unittest.TestCase):
    def test_six_source_binaries_match_four_archived_runtime_members(self):
        from e36_memory_smoke import runtime_binaries
        from e36_store import sha
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as directory:
            root=Path(directory);names=['zensim_mlp_train','bake_dial_refit','panel','inspect_qualified_checkpoint','check_holdout_overlap','serve_custom_bake']
            for name in names:(root/name).write_bytes(('invented '+name).encode())
            metadata=dict(files={'bin/'+name:sha(root/name) for name in names[:4]})
            self.assertEqual(runtime_binaries(metadata,root),metadata['files'])
            extra=dict(files=metadata['files']|{'bin/serve_custom_bake':sha(root/'serve_custom_bake')})
            with self.assertRaises(ValueError):runtime_binaries(extra,root)
            original=(root/'panel').read_bytes();(root/'panel').write_bytes(b'changed')
            with self.assertRaises(ValueError):runtime_binaries(metadata,root)
            (root/'panel').write_bytes(original);(root/'panel').unlink()
            with self.assertRaises(ValueError):runtime_binaries(metadata,root)

    def test_adapter_refuses_unbound_or_wrong_diagnostic_before_bootstrap(self):
        import e36_training as training
        args=argparse.Namespace(e36_arm='tv10-c',seed_index=0,heldout=None,strict_admission=True,
            train_only=True,head='N',spec=training.SPEC,local_smoke_budget='2:128')
        authorized=dict(ZEN_FIT_PROGRAM_SHA='a'*64,ZEN_E36_ARGV_SHA='b'*64,ZEN_E36_EXECUTION_CONTRACT='local-smoke',ZEN_E36_MEMORY_SMOKE='2:128')
        with patch.dict(training.os.environ,authorized,clear=True):
            self.assertTrue(training.diagnostic(args))
            for key,value in [('seed_index',1),('heldout','kadid'),('local_smoke_budget','2:129'),('e36_arm','tv1-c'),('head','F'),('spec','other')]:
                changed=copy.copy(args);setattr(changed,key,value)
                with self.assertRaises(ValueError):training.bootstrap(changed)
        for key in authorized:
            reduced=dict(authorized);reduced.pop(key)
            with patch.dict(training.os.environ,reduced,clear=True):
                with self.assertRaises(ValueError):training.bootstrap(args)
        args.local_smoke_budget=None
        with patch.dict(training.os.environ,authorized,clear=True):
            with self.assertRaises(ValueError):training.diagnostic(args)

    def test_raw_production_result_uses_nested_final_epoch(self):
        result=dict(execution_contract='local-smoke',selection=dict(epoch_rule='last',selected_epoch=1))
        qualified=dict(execution_contract='local-smoke')
        diagnostic_result(result,qualified)
        self.assertNotIn('selected_epoch',result)
        for key,value in [('epoch_rule','best'),('selected_epoch',119)]:
            bad=copy.deepcopy(result);bad['selection'][key]=value
            with self.assertRaises(ValueError):diagnostic_result(bad,qualified)
        with self.assertRaises(ValueError):diagnostic_result(result,dict(execution_contract='registered-fit'))

    def test_reordered_parent_selects_actual_seed_zero(self):
        cells=[dict(name='e36-tv10-c__N/full_s'+str(s),argv=['v2_confirm_fit.py','--e36-arm','tv10-c','--seed-index',str(s)]) for s in [1,0,2]]
        blueprint=dict(cells=cells);selected=seed_zero(blueprint)
        self.assertEqual(selected['name'],'e36-tv10-c__N/full_s0')
        self.assertEqual(selected['argv'][-1],'0')
        selected['argv'][-1]='changed';self.assertEqual(cells[1]['argv'][-1],'0')
        for mutation in ['duplicate','route','arm','name']:
            bad=copy.deepcopy(blueprint)
            if mutation=='duplicate':bad['cells'][2]=copy.deepcopy(bad['cells'][1])
            if mutation=='route':bad['cells'][0]['argv'][0]='v2_lodo_mlp.py'
            if mutation=='arm':bad['cells'][0]['argv'][2]='tv1-c'
            if mutation=='name':bad['cells'][0]['name']='e36-tv10-c__N/full_s0'
            with self.assertRaises(ValueError):seed_zero(bad)

if __name__=='__main__':unittest.main()
