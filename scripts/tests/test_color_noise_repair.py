"""The added inputs are exactly fine X/B; recipe drift is refused before training."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import v2_common as common
import v2_lodo_mlp as fit
import color_noise_repair as owner


class ColorNoise(unittest.TestCase):
    def test_only_missing_fine_chroma_is_added(self):
        old=common.derived_declaration('fx1')[2]
        new=common.derived_declaration('fx2')[2]
        extra=set(new['direct'])-set(old['direct'])
        expected=set(range(13))|set(range(26,39))
        expected|={372+ch*29+i for ch in (0,2) for i in range(29) if i!=21}
        self.assertEqual(extra,expected)
        self.assertTrue(set(old['direct'])<=set(new['direct']))
        self.assertEqual(len(new['direct']),492)
        self.assertEqual({a for a,b in new['products']},set(new['direct']))
        for a,b in new['products']:
            cell=(a//13 if a<156 else (a-372)//29)
            self.assertEqual(b,372+cell*29+21)

    def test_changed_declaration_is_refused(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as td:
            p=Path(td)/'changed.json';p.write_text('{}')
            with patch.object(common,'FX2_DECLARATION',p):
                with self.assertRaisesRegex(ValueError,'changed after registration'):
                    common.derived_declaration('fx2')

    def test_native_argv_and_four_arms(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as td:
            p=Path(td)/'keep.txt';a=owner.arm_record('both')
            p.write_text('\n'.join(map(str,a['columns']))+'\n')
            recipe=common.recipe_of(a['spec'])
            cmd=fit.train_command([],1101,101,1853,p,'N',Path(td)/'out.bin',recipe)
            self.assertEqual(cmd[cmd.index('--derived-inputs')+1],str(common.FX2_DECLARATION))
            self.assertEqual(recipe['coverage_mask'],0x9e)
            self.assertEqual(recipe['coverage_weight'],16)
            p.write_text('13\n')
            with self.assertRaises(ValueError):
                fit.train_command([],1101,101,1853,p,'N',Path(td)/'out.bin',recipe)
        arms=[owner.arm_record(n) for n in owner.ARMS]
        self.assertEqual([(a['coverage_mask'],a['input_count']) for a in arms],[(152,820),(158,820),(152,984),(158,984)])

class Decision(unittest.TestCase):
    def records(self):
        import color_noise_assessment as assessment
        rows=[]
        for arm in owner.ARMS:
            for seed in range(3):
                for population in assessment.SOURCES+['kadid_noise','tid2013_noise']:
                    rows.append(dict(arm=arm,seed=seed,population=population,statistics=dict(srocc_signed=.8)))
        return rows

    def test_noise_improvement_cannot_override_source_regression(self):
        from color_noise_assessment import choose
        rows=self.records()
        for r in rows:
            if r['arm']=='both':
                r['statistics']['srocc_signed']+=.1 if r['population'].endswith('_noise') else 0
                if r['population']=='konfig':r['statistics']['srocc_signed']-=.006
        self.assertEqual(choose(rows)[0],'current')
        for r in rows:
            if r['arm']=='coverage' and r['population'].endswith('_noise'):
                r['statistics']['srocc_signed']+=.01
        self.assertEqual(choose(rows)[0],'coverage')

    def test_seed_indices_never_choose_a_seed_and_ties_follow_registration(self):
        from color_noise_assessment import choose
        rows=self.records()
        for r in rows:
            if r['arm']!='current' and r['population'].endswith('_noise'):
                r['statistics']['srocc_signed']+=.01
        self.assertEqual(choose(rows)[0],'coverage')
        with self.assertRaises(KeyError):choose(rows[:-1])

if __name__=='__main__':unittest.main()
