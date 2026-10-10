"""Refuse budget identity drift before launching the existing collectors."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import costcmp_run as cmp
import speedq_run as owner


class BudgetAdmissionTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / 'parity.json'
        records = []
        for g, t, n in cmp.cells(budget=True):
            w, h = map(int, g.split('x'))
            for arm in cmp.BUDGET_ARMS:
                records.append(dict(name=arm, arm='by_v2fy', width=w, height=h,
                                    tier=t, threads=str(n), input_sha256=g,
                                    model=dict(revision='5', source_sha256=owner.SOURCE_SHA),
                                    score=50., score_bits='4049000000000000', feature_values=[.25]*420))
        self.value = dict(schema='costcmp-budget-preflight-v1', status='PASS', records=records)

    def test_every_candidate_is_bit_bound_and_complete(self):
        self.path.write_text(json.dumps(self.value))
        self.assertEqual(len(cmp.receipt(self.path, budget=True)), 36)
        for mutate in [lambda r: r.update(score_bits='4049000000000001'),
                       lambda r: r['feature_values'].__setitem__(0, .25000000000000006),
                       lambda r: r.update(width=8191), lambda r: r.update(threads='4'),
                       lambda r: r['model'].update(revision='4')]:
            bad = copy.deepcopy(self.value)
            mutate(bad['records'][-1])
            self.path.write_text(json.dumps(bad))
            with self.assertRaises(AssertionError):
                cmp.receipt(self.path, budget=True)
        for records in [self.value['records'][:-1], self.value['records']+self.value['records'][:1]]:
            self.path.write_text(json.dumps({**self.value, 'records': records}))
            with self.assertRaises(AssertionError):
                cmp.receipt(self.path, budget=True)

    def test_manifest_binds_binary_and_source_cap(self):
        arms = {}
        for arm in cmp.BUDGET_ARMS:
            binary = self.root / arm
            binary.write_bytes(arm.encode())
            artifact = self.root / (arm+'.json')
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            artifact.write_text(json.dumps(dict(binary_sha256=digest, dependencies={})))
            source = self.root / (arm+'.rs')
            source.write_text('' if arm.endswith('_before') else
                              'const REV5_JOB_BUDGET_BYTES: usize = '+arm.rsplit('b',1)[1]+' * 1024 * 1024;')
            arms[arm] = dict(binary=str(binary), artifact=str(artifact), binary_sha256=digest,
                             runtime_source=str(source), runtime_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        manifest = self.root / 'binaries.json'
        manifest.write_text(json.dumps(dict(schema='rev5perf5-binaries-v1', arms=arms)))
        self.assertEqual(set(cmp.budget_inventory(manifest)), set(cmp.BUDGET_ARMS))
        arm = cmp.BUDGET_ARMS[1]
        original = Path(arms[arm]['runtime_source']).read_text()
        Path(arms[arm]['runtime_source']).write_text(original+'\nfn changed_kernel() {}')
        arms[arm]['runtime_source_sha256'] = hashlib.sha256(Path(arms[arm]['runtime_source']).read_bytes()).hexdigest()
        manifest.write_text(json.dumps(dict(schema='rev5perf5-binaries-v1', arms=arms)))
        with self.assertRaisesRegex(AssertionError, 'beyond'):
            cmp.budget_inventory(manifest)
        Path(arms[arm]['runtime_source']).write_text('const REV5_JOB_BUDGET_BYTES: usize = 128 * 1024 * 1024;')
        arms[arm]['runtime_source_sha256'] = hashlib.sha256(Path(arms[arm]['runtime_source']).read_bytes()).hexdigest()
        manifest.write_text(json.dumps(dict(schema='rev5perf5-binaries-v1', arms=arms)))
        with self.assertRaisesRegex(AssertionError, 'cap'):
            cmp.budget_inventory(manifest)

    @patch.object(owner, 'run_segment')
    @patch.object(owner, 'refresh_activity')
    def test_frozen_candidate_mapping_and_revision_reach_owner(self, refresh, segment):
        self.path.write_text(json.dumps(self.value))
        mapping = {a: self.root/a for a in cmp.BUDGET_ARMS}
        owner.timing(mapping[cmp.BUDGET_ARMS[1]], self.root/'timing', 32, self.path, 'analyzer',
                     arms=cmp.BUDGET_ARMS, lock=self.root/'heavy.lock', cells=[cmp.cells(budget=True)[0]],
                     parity_loader=lambda p: cmp.production_rows(cmp.receipt(p,budget=True)),
                     baseline=cmp.BUDGET_ARMS[0], worker_binaries=mapping)
        self.assertEqual(segment.call_args.kwargs['worker_binaries'], mapping)
        self.assertEqual(segment.call_args.kwargs['baseline'], cmp.BUDGET_ARMS[0])
        for arm in cmp.BUDGET_ARMS:
            self.assertEqual(owner.worker_revision(arm), 5)
        with self.assertRaises(AssertionError):
            owner.worker_revision('by_v2fy_r9_b64')


if __name__ == '__main__':
    unittest.main()
