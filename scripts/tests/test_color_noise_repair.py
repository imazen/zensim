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

class Report(unittest.TestCase):
    def test_report_keeps_every_inspection_pair_and_native_score(self):
        import color_noise_assessment as assessment
        target=[i*.01 for i in range(10)]
        cases=[dict(primary=True,index=i,target=target[i],reference=f'ref{i}',filename=f'{i}.png',ref_image=f'images/ref{i}.png',dist_image=f'images/dist{i}.png') for i in range(10)]
        live=dict(target=target,reference=[f'ref{i}' for i in range(10)],pixels=[dict(dist_path=f'/corpus/wn/{i}.png') for i in range(10)])
        doc=dict(rank={n:dict(srocc_signed=.8,plcc=.9) for n in assessment.EXTERNAL},per_pair=dict(live=dict(pred=list(range(10)))))
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as td:
            root=Path(td)
            (root/'TRAIN_SELECTION.json').write_text('{}')
            (root/'EXTERNAL_EXPOSURE.json').write_text('{}')
            def read(path):
                if path.name=='TRAIN_SELECTION.json':return dict(selected='coverage',decisions=[])
                if path.name=='EXTERNAL_INPUTS.json':return dict(datasets=dict(live=live))
                if path.name=='CASES.json':return dict(cases=cases)
                return doc
            with patch.object(owner,'read',side_effect=read):assessment.report(root)
            text=(root/'index.html').read_text()
            self.assertEqual(text.count('<figure>'),20)
            self.assertIn('ref9 / 9.png',text)
            result=json.loads((root/'REPORT.json').read_text())
            self.assertEqual(len(result['severe_cases']),10)
            self.assertFalse(result['qualified'])
            self.assertEqual(result['severe_cases'][9]['scores']['repair-both'],9)

class DevelopmentBinding(unittest.TestCase):
    def test_cached_predictions_cannot_move_between_models_or_populations(self):
        import color_noise_assessment as assessment
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as td:
            root=Path(td);reg=dict(frozen_sha256='frozen',role_decision_sha256='role');(root/'REGISTRATION.json').write_text(json.dumps(reg))
            table=root/'table.parquet';table.write_bytes(b'fixture')
            p=dict(registration_sha256=common.sha(root/'REGISTRATION.json'),table=str(table),table_sha256=common.sha(table))
            (root/'DEVELOPMENT_POPULATIONS.json').write_text(json.dumps(p))
            cell=root/'cells/current/s0';cell.mkdir(parents=True)
            model=cell/'packed.bin';model.write_bytes(b'model')
            a=owner.arm_record('current')
            r=dict(packed_model=str(model),packed_model_sha256=common.sha(model),execution_contract='registered-fit',selection=dict(selected_epoch=119),epochs=120,pairs_per_epoch=50000,
                spec=a['spec'],head='N',seed_index=0,init_seed=1101,sample_seed=101,kept_features=len(a['columns']),derived_inputs=dict(token=a['token'],declaration_sha256=a['declaration_sha256']),
                frozen_sha256=reg['frozen_sha256'],data_role_decision_sha256=reg['role_decision_sha256'],output_stage='e33-identity-knot-tail2')
            (cell/'result.json').write_text(json.dumps(r))
            dest=root/'development/current/s0';dest.mkdir(parents=True)
            (dest/'pred.tsv').write_text('row_idx\tpred\n0\t1\n')
            cached=dict(model=dict(path=str(model),sha256=common.sha(model),result_sha256=common.sha(cell/'result.json')),
                population_sha256=common.sha(root/'DEVELOPMENT_POPULATIONS.json'),predictions_sha256=common.sha(dest/'pred.tsv'),records=[])
            (dest/'CELL.json').write_text(json.dumps(cached))
            self.assertEqual(assessment.development_cell(root,'current',0),cached)
            (dest/'pred.tsv').write_text('row_idx\tpred\n0\t2\n')
            with self.assertRaisesRegex(ValueError,'cached TRAIN-development identity differs'):
                assessment.development_cell(root,'current',0)
            r['selection']['selected_epoch']=0
            (cell/'result.json').write_text(json.dumps(r))
            with self.assertRaisesRegex(ValueError,'registered full fit/packing missing'):
                assessment.development_cell(root,'current',0)

class RetainedStatistics(unittest.TestCase):
    def test_jnd_population_and_statistics_are_retained_verbatim(self):
        import color_noise_assessment as assessment
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp') as td:
            root=Path(td);base=root/'base';base.mkdir();out=root/'output';out.mkdir()
            scores={'old':dict(pred=[3.,2.,1.])}
            dataset=dict(axis='jnd',target=[30.,20.,10.],reference=['a','b','c'],scores=scores)
            model=dict(name='old')
            spec=dict(schema='zensim-frozen-gauntlet-v1',models=[model]+[dict(name='repair-'+a) for a in owner.ARMS],datasets=dict(konjnd_jpeg_select=dataset),notes=['current study'])
            (out/'EXTERNAL_INPUTS.json').write_text(json.dumps(spec))
            original=dict(name='old',rank=dict(konjnd_jpeg_select=dict(srocc_signed=.75),other=dict(srocc_signed=.2)),
                scatter={},scatter_assessment={},training_overlap={},per_pair=dict(konjnd_jpeg_select=dict(pred=[3.,2.,1.],jnd=[30.,20.,10.])))
            (base/'old.fulleval.json').write_text(json.dumps(original))
            with patch.object(assessment,'BASE',base),patch('build_peer_fullevals.build_comparison') as native,patch.object(assessment.subprocess,'run'):
                assessment.dashboard(out)
            native.assert_called_once()
            fresh=json.loads((out/'NEW_MODEL_INPUTS.json').read_text())
            self.assertEqual([m['name'] for m in fresh['models']],['repair-'+a for a in owner.ARMS])
            retained=json.loads((out/'old.fulleval.json').read_text())
            self.assertEqual(retained['rank'],dict(konjnd_jpeg_select=dict(srocc_signed=.75)))
            self.assertEqual(retained['per_pair'],original['per_pair'])
            self.assertEqual(retained['comparison_notes'],['current study'])

if __name__=='__main__':unittest.main()
