"""Admission negative controls for COSTCMP using the existing SPEEDQ owner."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import costcmp_run as cmp
import speedq_run as owner

class CostcmpTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path.home()/'tmp');self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'receipt.json'
        records=[]
        for g,t,n in cmp.cells():
            w,h=map(int,g.split('x'))
            for a in cmp.ARMS:
                model={'revision':'5','source_sha256':owner.SOURCE_SHA} if a==cmp.ARMS[0] else (
                    {'revision':'1','source_sha256':a[-1]*64} if a.startswith('zensim_') else
                    {'commit':cmp.FAST_MAIN} if a=='fast_ssim2_main' else {'version':'0.8.2'})
                records.append(dict(name=a,arm='by_v2fy' if a==cmp.ARMS[0] else a,width=w,height=h,tier=t,threads=str(n),
                                    model=model,input_sha256=g,score=50.,score_bits='4049000000000000',feature_values=[0.25]*420 if a==cmp.ARMS[0] else []))
        self.value=dict(schema='costcmp-preflight-v1',status='PASS',records=records)
    def save(self,value):self.path.write_text(json.dumps(value))
    def test_full_grid_admits_and_missing_or_duplicate_cells_refuse(self):
        self.save(self.value);self.assertEqual(len(cmp.receipt(self.path)),320)
        for records in (self.value['records'][:-1],self.value['records']+[self.value['records'][0]]):
            self.save({**self.value,'records':records})
            with self.assertRaises(AssertionError):cmp.receipt(self.path)
    def test_revision_model_input_and_strict_feature_changes_refuse(self):
        mutations=[lambda r:r['model'].update(revision='4'),lambda r:r['model'].update(source_sha256='0'*64),
                   lambda r:r.update(input_sha256='wrong'),lambda r:r['feature_values'].__setitem__(0,.25000000000000006)]
        for mutate in mutations:
            bad=copy.deepcopy(self.value);mutate(bad['records'][5]);self.save(bad)
            with self.assertRaises(AssertionError):cmp.receipt(self.path)
    def test_peer_ready_change_refuses_and_selects_shared_first_clean_indices(self):
        self.save(self.value);rows=cmp.receipt(self.path);rec=copy.deepcopy(rows[(owner.GEOMETRIES[0],'v4x',1,'fast_ssim2_main')]);rec['model']['commit']='wrong'
        with self.assertRaises(AssertionError):cmp.validate_ready(rows,'fast_ssim2_main',owner.GEOMETRIES[0],'v4x',1,rec)
        raw=dict(gate_clean=[False]+[True]*32,paired_rounds={a:list(range(33)) for a in cmp.ARMS},zenbench_unreliable=False)
        selected,selection=owner.select_clean_rounds(raw)
        self.assertEqual(selection['retained_indices'],list(range(1,33)))
        self.assertTrue(all(v==list(range(1,33)) for v in selected.values()))
    def test_custom_grid_keeps_shared_lock_and_canonical_collector(self):
        with patch.object(owner,'segment_lock') as lock,patch.object(owner,'run_segment') as segment:
            owner.timing('binary','root',32,'receipt','analyzer',arms=cmp.ARMS,lock='shared',cells=cmp.cells()[:2],parity_loader=lambda p:{},baseline=cmp.ARMS[0],ready_check='validator')
            self.assertEqual(lock.call_count,2);self.assertEqual(segment.call_count,2)
            self.assertEqual(segment.call_args.kwargs['baseline'],cmp.ARMS[0])
            self.assertEqual(segment.call_args.kwargs['ready_check'],'validator')

if __name__=='__main__':unittest.main()
