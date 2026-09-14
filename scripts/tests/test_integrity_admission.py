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
        import numpy as np
        import pandas as pd
        class ObservedMasks(Exception):
            pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roles = {'fit':['2010'], 'calibration':['1214'], 'development':['1054']}
            records, features, audits = [], {}, {}
            for pos, (role, origins) in enumerate(roles.items()):
                ids = [pos*2, pos*2+1]
                for i in ids:
                    records.append(dict(index=i, role='train', fit_role=role, origin=origins[0],
                                        source_family=origins[0], disposition='valid' if i%2==0 else 'catastrophic_proxy',
                                        expected_distorted_file_sha256=str(i), expected_distorted_pixels_sha256=str(i)))
                frame = pd.DataFrame({f'f{k}':[float(i) for i in ids] for k in range(372)})
                frame['human_score'] = ids
                path=root/(role+'.csv');frame.to_csv(path,index=False)
                features[role]=dict(path=str(path),sha256=trainer._sha256(path))
                path=root/(role+'.jsonl')
                path.write_text(''.join(json.dumps(dict(human_score=i,distorted_file_sha256=str(i),distorted_pixels_sha256=str(i),reference_pixels_sha256='ref'))+'\n' for i in ids))
                audits[role]=dict(path=str(path),sha256=trainer._sha256(path))
            admission=root/'admission.json'
            admission.write_text(json.dumps(dict(schema='integrity-train-diagnostic-v1',origins=roles,records=records)))
            tool=root/'tool';tool.write_bytes(b'synthetic tool identity')
            spec=dict(path=str(tool),sha256=trainer._sha256(tool))
            manifest=root/'fit.json'
            manifest.write_text(json.dumps(dict(schema='integrity-head-train-v2',formula_revision=3,
                head_feature_ids=list(range(228)),seed=4101,deadband=.9,
                admission=dict(path=str(admission),sha256=trainer._sha256(admission)),
                features=features,audit=audits,base_bake=spec,extractor=spec,parity_binary=spec)))
            def observe(z, y, fit, cal, weights, seed, parameters):
                np.testing.assert_array_equal(np.flatnonzero(fit),[0,1])
                np.testing.assert_array_equal(np.flatnonzero(cal),[2,3])
                np.testing.assert_allclose(z[:2],np.broadcast_to([[-1.],[1.]],(2,228)))
                raise ObservedMasks()
            with patch.object(trainer,'fit_canonical_hgb',side_effect=observe):
                with self.assertRaises(ObservedMasks):
                    trainer.strict_train_main(['--strict-train-manifest',str(manifest),'--out-dir',str(root/'out')])
            self.assertFalse((root/'out').exists())


if __name__ == '__main__':
    unittest.main()
