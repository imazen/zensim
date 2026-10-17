"""Exact diagnostic identity uses metadata only, without a fit."""
import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_memory_smoke import seed_zero

class Diagnostic(unittest.TestCase):
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
