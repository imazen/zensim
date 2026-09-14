#!/usr/bin/env python3
"""Synthetic boundary checks: forbidden roles must fail before feature reads."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v_next'))
import train_corruption_head as trainer
import corruption_gate_eval as evaluator


class Admission(unittest.TestCase):
    @staticmethod
    def native_pair(index=1):
        import hashlib
        import numpy as np
        meta, audit = Admission.honest_pair(index, 5., False)
        identity = dict(pixel_format='Srgb16Rgba', primaries='Srgb', alpha='Unknown',
                        gamut='Clip', width=16, height=16, endianness='little')
        audit.update(schema='canonical-feature-audit-v2', input_contract='sdr-native-clip-v1',
                     width=16, height=16, feature_audit_scope='complete-structural-read-set-v1',
                     consumed_feature_ids=list(range(228)), max_consumed_feature_abs_delta=0.,
                     canonical_feature_count=372,
                     formula_revision='Rev3', candidate_formula_revisions=['Rev3'],
                     root_form_override='sqrt',
                     canonical_features_f32_le_sha256=hashlib.sha256(np.zeros(372,dtype='<f4').tobytes()).hexdigest())
        meta['input_contract'] = 'sdr-native-clip-v1'
        for side, digest in [('reference','a'*64),('distorted','b'*64)]:
            audit[side+'_pixels_sha256'] = digest
            audit[side+'_color'] = dict(contract='sdr-native-clip-v1', endianness='little', scoring_identity=copy.deepcopy(identity))
            meta['expected_'+side+'_pixels_sha256'] = digest
            meta['expected_'+side+'_scoring_identity'] = copy.deepcopy(identity)
        return meta,audit

    def test_native_equal_codes_different_primaries_are_not_identity_or_duplicates(self):
        meta,a = self.native_pair()
        a['distorted_pixels_sha256'] = a['reference_pixels_sha256']
        meta['expected_distorted_pixels_sha256'] = a['distorted_pixels_sha256']
        a['distorted_color']['scoring_identity']['primaries'] = 'DisplayP3'
        meta['expected_distorted_scoring_identity']['primaries'] = 'DisplayP3'
        evaluator.integrity_summary([meta],{1:a})
        other,b = copy.deepcopy(meta),copy.deepcopy(a)
        other['index']=2;b['human_score']=2
        b['distorted_color']['scoring_identity']['primaries']='Bt2020'
        other['expected_distorted_scoring_identity']['primaries']='Bt2020'
        self.assertEqual(evaluator.integrity_summary([meta,other],{1:a,2:b})['unique_rows'],2)
        a['pixels_identical']=True
        with self.assertRaisesRegex(ValueError,'identity flag mismatch'):
            evaluator.integrity_summary([meta],{1:a})

    def test_native_admission_refuses_retagged_or_incomplete_receipts(self):
        meta,a = self.native_pair()
        evaluator.integrity_summary([meta],{1:a})
        for change in ('schema','primaries','scope','coverage','hash','format','geometry','revision','root'):
            b=copy.deepcopy(a)
            if change=='schema': b['schema']='canonical-feature-audit-v1'
            if change=='primaries': b['distorted_color']['scoring_identity']['primaries']='DisplayP3'
            if change=='scope': del b['feature_audit_scope']
            if change=='coverage': b['consumed_feature_ids'].remove(227)
            if change=='hash': del b['canonical_features_f32_le_sha256']
            if change=='format': b['reference_color']['scoring_identity']['pixel_format']='Rgb8'
            if change=='geometry': b['reference_color']['scoring_identity']['width']=8
            if change=='revision': b['candidate_formula_revisions']=['Rev1']
            if change=='root': b['root_form_override']='libm'
            with self.subTest(change=change),self.assertRaises(ValueError):
                evaluator.integrity_summary([meta],{1:b})

    @staticmethod
    def honest_pair(index, knob, active):
        meta = dict(index=index, origin='2010', source_family='origin:2010', role='train',
                    reference='ref', distorted=f'dist{index}', expected_distorted_file_sha256=f'dfile{index}',
                    reference_sha256='rfile', expected_distorted_pixels_sha256=f'dpixels{index}',
                    fit_role='development', disposition='valid', family='honest_jpeg',
                    kind='honest_codec', content_class='fixture', codec='jpeg', knob=knob,
                    knob_axis='quality', knob_direction='higher_is_better', knob_context='reference/config')
        audit = dict(human_score=index, reference='ref', distorted=f'dist{index}',
                     distorted_file_sha256=f'dfile{index}', reference_file_sha256='rfile',
                     distorted_pixels_sha256=f'dpixels{index}', reference_pixels_sha256='rpixels',
                     head_probability=float(active), stored_f32_head_probability=float(active), head_threshold=.9,
                     pixel_composed_score=0. if active else 40., cached_composed_score=0. if active else 40.,
                     stored_f32_composed_score=0. if active else 40., pixels_identical=False, base_score=40., model_inputs={})
        return meta, audit

    def test_explicit_quality_orientation_and_spatial_controls(self):
        low, a = self.honest_pair(1, 5., True)
        high, b = self.honest_pair(2, 95., False)
        rows, audits = [low, high], {1:a, 2:b}
        result = evaluator.integrity_summary(rows, audits)
        self.assertEqual(result['worst_stored_native']['honest_activation']['count'], 1)
        for row in rows:
            row['knob_direction'] = 'higher_is_worse'
        self.assertEqual(evaluator.integrity_summary(rows, audits)['worst_stored_native']['honest_activation']['count'], 0)
        low['knob_direction'] = 'higher_is_better'
        with self.assertRaisesRegex(ValueError, 'conflicting knob'):
            evaluator.integrity_summary(rows, audits)
        high['knob_context'] = 'another configuration'
        self.assertEqual(evaluator.integrity_summary(rows, audits)['worst_stored_native']['n'], 2)
        for row in rows:
            del row['knob_direction']
        with self.assertRaisesRegex(ValueError, 'declare knob orientation'):
            evaluator.integrity_summary(rows, audits)
        for row in rows:
            row['codec'] = 'jxl'
        self.assertEqual(evaluator.integrity_summary(rows, audits)['worst_stored_native']['honest_activation']['count'], 0)
        for row in rows:
            row['kind'] = 'honest_spatial'
            row['knob'] = None
        result = evaluator.integrity_summary(rows, audits)
        self.assertEqual(result['worst_stored_native']['n'], 0)
        self.assertEqual(result['by_codec']['jxl']['native_activation']['count'], 1)

    def test_reference_pixels_and_identity_shortcut_are_bound(self):
        meta, audit = self.honest_pair(1, 5., True)
        meta['expected_reference_pixels_sha256'] = 'rpixels'
        meta['pixels_identical'] = False
        evaluator.integrity_summary([meta], {1:audit})
        audit['reference_pixels_sha256'] = 'different-reference'
        with self.assertRaisesRegex(ValueError, 'reference pixel mismatch'):
            evaluator.integrity_summary([meta], {1:audit})
        audit['reference_pixels_sha256'] = 'rpixels'
        audit['pixels_identical'] = True
        with self.assertRaisesRegex(ValueError, 'identity flag mismatch'):
            evaluator.integrity_summary([meta], {1:audit})
        audit['pixels_identical'] = False
        meta['pixels_identical'] = True
        with self.assertRaisesRegex(ValueError, 'identity flag mismatch'):
            evaluator.integrity_summary([meta], {1:audit})

    def test_train_diagnostic_keeps_development_and_refuses_test(self):
        meta, audit = self.honest_pair(1, 5., False)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'admission.json'; log = root/'audit.jsonl'; out = root/'result.json'
            data = dict(schema='integrity-train-diagnostic-v1',
                        origins={'fit':[], 'development':['2010'], 'calibration':[]}, records=[meta])
            manifest.write_text(json.dumps(data)); log.write_text(json.dumps(audit)+'\n')
            args = ['--integrity-admission',str(manifest),'--audit-jsonl',str(log),'--out-json',str(out)]
            evaluator.integrity_report(args)
            self.assertEqual(json.loads(out.read_text())['by_role']['development']['n'], 1)
            out.unlink(); log.unlink()
            data['records'][0]['role'] = 'test'
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, 'forbidden integrity source role'):
                evaluator.integrity_report(args)
            self.assertFalse(out.exists())

    def test_pixel_duplicates_preserve_known_labels_and_refuse_conflicts(self):
        def pair(index, disposition, family):
            meta = dict(index=index, origin='2010', reference='ref', distorted='dist',
                        expected_distorted_file_sha256='dfile', reference_sha256='rfile',
                        expected_distorted_pixels_sha256='dpixels', fit_role='evaluate',
                        disposition=disposition, family=family, kind='corruption',
                        content_class='fixture')
            audit = dict(reference='ref', distorted='dist', distorted_file_sha256='dfile',
                         reference_file_sha256='rfile', distorted_pixels_sha256='dpixels',
                         reference_pixels_sha256='rpixels', head_probability=1.,
                         stored_f32_head_probability=1., head_threshold=.9,
                         pixel_composed_score=0., cached_composed_score=0.,
                         stored_f32_composed_score=0., pixels_identical=False, base_score=40.)
            return meta, audit

        unknown, a = pair(1, 'ambiguous_or_recoverable', 'aliasing')
        positive, b = pair(2, 'catastrophic_proxy', 'real_bug')
        for records in ([unknown, positive], [positive, unknown]):
            report = evaluator.integrity_summary(records, {1: a, 2: b})
            self.assertEqual(report['unique_rows'], 1)
            row = report['rows'][0]
            self.assertEqual(row['disposition'], 'catastrophic_proxy')
            self.assertEqual(row['catalog_dispositions'], ['ambiguous_or_recoverable', 'catastrophic_proxy'])
            self.assertEqual(row['catalog_families'], ['aliasing', 'real_bug'])
            self.assertEqual(report['by_role']['evaluate']['real_bug_detection']['count'], 1)
        negative, c = pair(3, 'valid', 'honest_anchor')
        with self.assertRaisesRegex(ValueError, 'conflicting positive/negative'):
            evaluator.integrity_summary([unknown, positive, negative], {1: a, 2: b, 3: c})
        b['head_probability'] = b['stored_f32_head_probability'] = 0.
        with self.assertRaisesRegex(ValueError, 'identical pixels have different'):
            evaluator.integrity_summary([unknown, positive], {1: a, 2: b})
        b['stored_f32_head_probability'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite integrity audit'):
            evaluator.integrity_summary([positive], {2: b})

    def test_forbidden_roles_never_open_feature_payloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            admission = dict(schema='integrity-train-admission-v1',origins={'fit':['2010'],'calibrate':['1214']},records=[])
            variants=[]
            bad=copy.deepcopy(admission);bad['origins']['test']=['8462'];variants.append(bad)
            bad=copy.deepcopy(admission);bad['origins']['calibrate']=['8462'];variants.append(bad)
            bad=copy.deepcopy(admission);bad['origins']['calibrate']=['2010'];variants.append(bad)
            bad=copy.deepcopy(admission);bad['records']=[dict(index=0,role='test',fit_role='fit',origin='2010',source_family='fixture')];variants.append(bad)
            for i, value in enumerate(variants):
                ap=root/f'admission{i}.json';ap.write_text(json.dumps(value))
                m=dict(schema='integrity-head-train-v1',formula_revision=1,head_feature_ids=list(range(228)),seed=4101,deadband=.9,
                       admission=dict(path=str(ap),sha256=trainer._sha256(ap)))
                mp=root/f'm{i}.json';mp.write_text(json.dumps(m))
                with patch('pandas.read_csv',side_effect=AssertionError('payload was opened')):
                    with self.assertRaises(ValueError):
                        trainer.strict_train_main(['--strict-train-manifest',str(mp),'--out-dir',str(root/'output')])
                self.assertFalse((root/'output').exists())

    def test_rev3_development_is_not_probability_calibration(self):
        self.training_masks(native_input=False)

    def test_native_training_masks_and_feature_payload_binding(self):
        self.training_masks(native_input=True)

    def training_masks(self, native_input):
        import numpy as np
        import pandas as pd
        class ObservedMasks(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roles = {'fit':['2010'], 'calibration':['1214'], 'development':['1054']}
            tool=root/'tool';tool.write_bytes(b'synthetic tool identity')
            spec=dict(path=str(tool),sha256=trainer._sha256(tool))
            records, features, audits, pairs = [], {}, {}, {}
            native_rows = {}
            for pos, (role, origins) in enumerate(roles.items()):
                ids = [pos*2, pos*2+1]
                for i in ids:
                    if native_input:
                        import hashlib
                        row,a = self.native_pair(i)
                        row.update(role='train',fit_role=role,origin=origins[0],source_family=origins[0],
                                   disposition='valid' if i%2==0 else 'catastrophic_proxy')
                        a['distorted_pixels_sha256'] = row['expected_distorted_pixels_sha256'] = hashlib.sha256(str(i).encode()).hexdigest()
                        values=np.full(372,float(i),dtype='<f4')
                        if i == 0:
                            # Default pandas parsing rounds this across an f32
                            # midpoint; the Rust audit must catch that change.
                            values[227]=0.008768686559051275
                        a['canonical_features_f32_le_sha256'] = hashlib.sha256(values.tobytes()).hexdigest()
                        a['model_inputs'] = [[spec['path'],spec['sha256']]]
                        native_rows[i]=a;records.append(row)
                    else:
                        records.append(dict(index=i, role='train', fit_role=role, origin=origins[0],
                                            source_family=origins[0], disposition='valid' if i%2==0 else 'catastrophic_proxy',
                                            expected_distorted_file_sha256=str(i), expected_distorted_pixels_sha256=str(i)))
                frame = pd.DataFrame({f'f{k}':[float(i) for i in ids] for k in range(372)})
                if native_input and ids[0] == 0:
                    frame.loc[0,'f227']=0.008768686559051275
                frame['human_score'] = ids
                path=root/(role+'.csv');frame.to_csv(path,index=False)
                features[role]=dict(path=str(path),sha256=trainer._sha256(path))
                path=root/(role+'.jsonl')
                path.write_text(''.join(json.dumps(native_rows[i] if native_input else dict(human_score=i,distorted_file_sha256=str(i),distorted_pixels_sha256=str(i),reference_pixels_sha256='ref'))+'\n' for i in ids))
                audits[role]=dict(path=str(path),sha256=trainer._sha256(path))
                if native_input:
                    path=root/(role+'.tsv')
                    path.write_text('ref_path\tdist_path\thuman_score\n'+''.join(f"ref\tdist{i}\t{i}\n" for i in ids))
                    pairs[role]=dict(path=str(path),sha256=trainer._sha256(path))
            admission=root/'admission.json'
            admission.write_text(json.dumps(dict(schema='integrity-train-diagnostic-v2' if native_input else 'integrity-train-diagnostic-v1',input_contract='sdr-native-clip-v1' if native_input else 'legacy-rgb8',origins=roles,records=records)))
            manifest=root/'fit.json'
            manifest.write_text(json.dumps(dict(schema='integrity-head-train-v3' if native_input else 'integrity-head-train-v2',root_form='sqrt',input_contract='sdr-native-clip-v1' if native_input else 'legacy-rgb8',formula_revision=3,
                head_feature_ids=list(range(228)),seed=4101,deadband=.9,
                admission=dict(path=str(admission),sha256=trainer._sha256(admission)),
                features=features,audit=audits,pairs=pairs,base_bake=spec,extractor=spec,parity_binary=spec)))
            def observe(z, y, fit, cal, weights, seed, parameters):
                np.testing.assert_array_equal(np.flatnonzero(fit),[0,1])
                np.testing.assert_array_equal(np.flatnonzero(cal),[2,3])
                np.testing.assert_allclose(z[:2],np.broadcast_to([[-1.],[1.]],(2,228)))
                raise ObservedMasks()
            with patch.object(trainer,'fit_canonical_hgb',side_effect=observe):
                with self.assertRaises(ObservedMasks):
                    trainer.strict_train_main(['--strict-train-manifest',str(manifest),'--out-dir',str(root/'out')])
            self.assertFalse((root/'out').exists())
            if native_input:
                # A feature file can have an updated manifest hash and still
                # disagree with the canonical extraction bound by its audit.
                fpath=Path(features['fit']['path']);frame=pd.read_csv(fpath)
                frame.loc[0,'f227'] += 1.0;frame.to_csv(fpath,index=False)
                m=json.loads(manifest.read_text());m['features']['fit']['sha256']=trainer._sha256(fpath)
                manifest.write_text(json.dumps(m))
                with patch.object(trainer,'fit_canonical_hgb',side_effect=AssertionError('fit before binding')):
                    with self.assertRaisesRegex(ValueError,'canonical feature payload mismatch'):
                        trainer.strict_train_main(['--strict-train-manifest',str(manifest),'--out-dir',str(root/'out')])
                # A legacy receipt cannot pass native admission, even before
                # payload hashing. Make the feature path nonexistent to prove it.
                apath=Path(audits['fit']['path']);entries=[json.loads(x) for x in apath.read_text().splitlines()]
                entries[0]['schema']='canonical-feature-audit-v1'
                apath.write_text(''.join(json.dumps(x)+'\n' for x in entries))
                m['audit']['fit']['sha256']=trainer._sha256(apath)
                m['features']['fit']['path']=str(root/'must-not-open.csv');manifest.write_text(json.dumps(m))
                with self.assertRaisesRegex(ValueError,'audit input era mismatch'):
                    trainer.strict_train_main(['--strict-train-manifest',str(manifest),'--out-dir',str(root/'out')])


if __name__ == '__main__':
    unittest.main()
