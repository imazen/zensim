"""E28 uses actual owners, fixed pins, isolated legs and signed statistics."""
import json
import os
from pathlib import Path
import sys
import unittest

import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'scripts/rev4_featpot'))
sys.path.insert(0,str(REPO/'scripts'))
import e28_recipe as e28
import e28_simplex as nm
import v2_common as common
import v2_lodo_mlp as mlp
from lib.zen_stats import recipe_correlations,panel_batch


class E28Recipe(unittest.TestCase):
    def test_grouping_covers_exact_read_set_without_overlap(self):
        grouping=json.loads(e28.GROUPING.read_text())
        groups=grouping['groups']
        cols=[c for ids in groups.values() for c in ids]
        self.assertEqual(sorted(cols),grouping['columns'])
        self.assertEqual(len(set(cols)),420)
        self.assertLessEqual(len(groups),128)
        self.assertEqual(common.selection_id(cols),'59f0bbc2f290')

    def test_monotone_cubic_has_positive_derivative(self):
        q=np.linspace(-10,10,1001)
        for theta in [[-10,0,0,0],[1,2,-3,4],[-7,-2,1,-8]]:
            values=nm.remap(1-q,theta)
            self.assertTrue(np.all(np.diff(values)>0))

    def test_s2m_uses_four_populations_and_no_coverage(self):
        pin=e28.read_pin()
        x=pin['arms']['s2m']
        self.assertEqual(x['teachers'],['cid22'])
        self.assertEqual(x['human_members'],{'kadid':['kadid_train'],'tid2013':['tid2013'],'konfig':['konfig_train']})
        self.assertFalse(x['coverage'])
        self.assertEqual(x['pooled_legs'],['cid22','kadid','tid2013','konfig'])

    def test_only_new_recipe_emits_pooled_flags(self):
        groups=[('cid22',Path('/var/tmp/fixture.parquet'),1,0,'withinref,both'),('coverage',Path('/var/tmp/ordinal.parquet'),16,0,'withinref,rank')]
        for token in ['',':hd4',':hp4',':ha4']:
            recipe=common.recipe_of('sel:59f0bbc2f290@h32:H128:cv16:cf98'+token)
            argv=mlp.train_command(groups,1101,101,1853,Path('/var/tmp/keep.txt'),'N',Path('/var/tmp/out.bin'),recipe)
            self.assertNotIn('--pooled-rank-share',argv)
        for arm in ['s2o','s2m']:
            recipe=common.recipe_of(e28.spec(arm))
            argv=mlp.train_command(groups,1101,101,1853,Path('/var/tmp/keep.txt'),'N',Path('/var/tmp/out.bin'),recipe)
            self.assertEqual(argv[argv.index('--pooled-leg')+1],'cid22')
            self.assertEqual(argv.count('--pooled-leg'),1)
            self.assertEqual(argv[argv.index('--pooled-rank-share')+1],'0.5')
            self.assertEqual(argv[argv.index('--pooled-pearson-weight')+1],'0.5')

    def test_raw_signed_correlations_match_rust_panel(self):
        rng=np.random.default_rng(28)
        for ties in [False,True]:
            x=rng.normal(size=93);y=x*0.7+rng.normal(size=93)*0.4
            if ties:y=np.round(y,1)
            for sign in [1,-1]:
                tau,rho=recipe_correlations(sign*x,y)
                actual=panel_batch([('fixture',sign*x,y)],stats='full')[0]
                self.assertLess(abs(abs(tau)-actual['krocc']),1e-9)
                self.assertLess(abs(rho-actual['plcc_raw']),1e-9)
        self.assertEqual(recipe_correlations([2,2,2],[1,2,3]),(0,0))
        with self.assertRaises(ValueError):recipe_correlations([1,float('nan')],[2,3])


if __name__=='__main__':unittest.main()
