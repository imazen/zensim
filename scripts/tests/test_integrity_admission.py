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


if __name__ == '__main__':
    unittest.main()
