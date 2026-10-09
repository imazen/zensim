"""Negative controls for cross-build COSTCMP scaling admission."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import costcmp_run as cmp
import speedq_run as owner


class ScalingAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'parity.json'
        records = []
        for g, t, n in cmp.cells(scaling=True):
            w, h = map(int, g.split('x'))
            for arm in cmp.SCALING_ARMS:
                rev = '4' if arm == 'by_v2fy_r4' else ('5' if arm.startswith('by_v2fy_r') else '1')
                records.append(dict(name=arm, arm='by_v2fy' if rev != '1' else arm,
                                    width=w, height=h, tier=t, threads=str(n),
                                    model=dict(revision=rev, source_sha256=owner.SOURCE_SHA),
                                    input_sha256=g, score=50., score_bits='4049000000000000',
                                    feature_values=[.25]*420 if rev != '1' else []))
        self.value = dict(schema='costcmp-scaling-preflight-v1', status='PASS', records=records)

    def test_cross_build_bit_drift_or_incomplete_grid_refuses(self):
        self.path.write_text(json.dumps(self.value))
        self.assertEqual(len(cmp.receipt(self.path, scaling=True)), 240)
        before_index = next(i for i,r in enumerate(self.value['records']) if r['name'] == 'by_v2fy_r5_before')
        for field in ('score_bits', 'feature_values'):
            bad = copy.deepcopy(self.value)
            rec = bad['records'][before_index]
            if field == 'score_bits':
                rec[field] = '4049000000000001'
            else:
                rec[field][0] = .25000000000000006
            self.path.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):
                cmp.receipt(self.path, scaling=True)
        for records in (self.value['records'][:-1], self.value['records'] + self.value['records'][:1]):
            self.path.write_text(json.dumps({**self.value, 'records': records}))
            with self.assertRaises(AssertionError):
                cmp.receipt(self.path, scaling=True)

    @patch.object(owner, 'run_segment')
    @patch.object(owner, 'refresh_activity')
    def test_frozen_binary_mapping_reaches_segment_under_lock(self, refresh, segment):
        self.path.write_text(json.dumps(self.value))
        mapping = {'by_v2fy_r5_before': Path('before')}
        owner.timing('after', Path(self.temp.name)/'timing', 32, self.path, 'analyzer',
                     arms=cmp.SCALING_ARMS, lock=Path(self.temp.name)/'heavy.lock',
                     cells=[('1024x1024','v4x',8)],
                     parity_loader=lambda p: cmp.production_rows(cmp.receipt(p, scaling=True)),
                     baseline=cmp.ARMS[0], worker_binaries=mapping)
        self.assertEqual(segment.call_args.kwargs['worker_binaries'], mapping)
        self.assertEqual(segment.call_args.kwargs['baseline'], 'by_v2fy_r5')
        self.assertEqual(segment.call_args.kwargs['arms'], cmp.SCALING_ARMS)


if __name__ == '__main__':
    unittest.main()
