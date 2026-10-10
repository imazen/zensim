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

    def test_shared_load_rss_records_busy_observation_without_quiet_admission(self):
        arm = cmp.BUDGET_ARMS[0]
        rec = self.value['records'][0]
        binary = self.root/'binary'
        binary.write_bytes(b'frozen')
        def measured(command, **kwargs):
            kwargs['stderr'].write('Maximum resident set size (kbytes): 12345\n')
            kwargs['stderr'].flush()
            return json.dumps(rec)
        busy = dict(admitted=False, load1=12., foreign=['cargo'])
        with patch.object(owner, 'refresh_activity'), patch.object(owner, 'quiet_gate') as quiet, \
                patch.object(owner, 'quiet_state', return_value=busy), \
                patch.object(owner.subprocess, 'check_output', side_effect=measured):
            owner.rss(binary, self.root/'rss', self.path, arms=[arm],
                      parity_loader=lambda p: {('1024x1024','v4x',8,5):rec},
                      geometries=['1024x1024'], thread_counts=[8], require_quiet=False)
        quiet.assert_not_called()
        result = json.loads(next((self.root/'rss').glob('*.json')).read_text())
        self.assertEqual(result['quiet_gate'], busy)
        self.assertFalse(result['quiet_required'])
        self.assertEqual(result['max_rss_kib'], 12345)
        self.assertEqual(result['rss_policy'], owner.RSS_UNDER_LOAD_POLICY)

    def test_rss_authorization_cannot_disable_timing_gate(self):
        with patch('sys.argv', ['costcmp_run.py','timing','--dest',str(self.root),
                               '--budget-grid','--rss-under-load']):
            with self.assertRaises(SystemExit) as result:
                cmp.main()
        self.assertEqual(result.exception.code, 2)

    def test_budget_parity_binds_every_arm_and_refuses_one_flipped_bit(self):
        import rev5perf5_candidate as candidate
        (self.root/'budget/parity').mkdir(parents=True)
        (self.root/'budget/parity/PREFLIGHT_PASS.json').write_text(json.dumps(self.value))
        binary = self.root/'binary'
        binary.write_bytes(b'frozen')
        rows = {(f"{r['width']}x{r['height']}", r['tier'], int(r['threads'])): r for r in self.value['records']}
        def run(flip):
            def worker(_binary, arm, revision, g, t, n, log):
                rec = copy.deepcopy(rows[(g, t, n)])
                rec['threads'] = n
                if flip and g == '8192x4096':
                    rec['feature_values'][419] = .25000000000000006
                return rec
            return worker
        with patch.object(owner, 'worker_run', side_effect=run(False)):
            candidate.budget_parity(self.root, binary, 'ok')
        result = json.loads((self.root/'budget/parity-ok/BUDGET_PARITY_PASS.json').read_text())
        self.assertEqual(result['cells'], 9)
        with patch.object(owner, 'worker_run', side_effect=run(True)):
            with self.assertRaisesRegex(AssertionError, 'consumed feature bits differ'):
                candidate.budget_parity(self.root, binary, 'flipped')
        self.assertFalse((self.root/'budget/parity-flipped/BUDGET_PARITY_PASS.json').exists())

    def floor_manifest(self, floor_source=None, schema='rev5perf5-binaries-v2'):
        """Five-arm v2 inventory: candidates differ only in the cap; the floor arm is cap128 plus the hunk."""
        body = 'fn rev5_job_limit() -> usize {\n' + cmp.CAP_LIMIT + '}\n'
        cap128 = 'const REV5_JOB_BUDGET_BYTES: usize = 128 * 1024 * 1024;\n' + body
        if floor_source is None:
            floor_source = cmp.FLOOR_BLOCK + cap128.replace(cmp.CAP_LIMIT, cmp.FLOOR_LIMIT)
        arms = {}
        for arm in cmp.BUDGET_FLOOR_ARMS:
            binary = self.root / arm
            binary.write_bytes(arm.encode())
            artifact = self.root / (arm+'.json')
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            artifact.write_text(json.dumps(dict(binary_sha256=digest, dependencies={})))
            source = self.root / (arm+'.rs')
            source.write_text(floor_source if arm == cmp.BUDGET_FLOOR_ARM else '' if arm.endswith('_before') else
                              'const REV5_JOB_BUDGET_BYTES: usize = '+arm.rsplit('b',1)[1]+' * 1024 * 1024;\n' + body)
            arms[arm] = dict(binary=str(binary), artifact=str(artifact), binary_sha256=digest,
                             runtime_source=str(source), runtime_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        manifest = self.root / 'binaries-v2.json'
        manifest.write_text(json.dumps(dict(schema=schema, arms=arms)))
        return manifest, cap128

    def test_floor_inventory_binds_cap128_plus_exactly_the_floor(self):
        manifest, cap128 = self.floor_manifest()
        self.assertEqual(set(cmp.budget_inventory(manifest, floor=True)), set(cmp.BUDGET_FLOOR_ARMS))
        with self.assertRaises(AssertionError):
            cmp.budget_inventory(manifest)  # a v2 inventory is not the four-arm v1 grid
        floored = cmp.FLOOR_BLOCK + cap128.replace(cmp.CAP_LIMIT, cmp.FLOOR_LIMIT)
        for label, source in [('extra change', floored + 'fn changed_kernel() {}\n'),
                              ('floor of two', floored.replace('REV5_MIN_JOB_SLOTS: usize = 3', 'REV5_MIN_JOB_SLOTS: usize = 2')),
                              ('256 MiB base', floored.replace('128 * 1024', '256 * 1024')),
                              ('no floor', cap128)]:
            with self.subTest(label):
                manifest, _ = self.floor_manifest(source)
                with self.assertRaises(AssertionError):
                    cmp.budget_inventory(manifest, floor=True)
        manifest, _ = self.floor_manifest(schema='rev5perf5-binaries-v1')
        with self.assertRaises(AssertionError):
            cmp.budget_inventory(manifest, floor=True)

    def test_floor_receipt_requires_all_five_arms_bit_identical(self):
        records = copy.deepcopy(self.value['records'])
        for rec in self.value['records']:
            if rec['name'] == 'by_v2fy_r5_b128':
                records.append({**copy.deepcopy(rec), 'name': cmp.BUDGET_FLOOR_ARM})
        v2 = dict(schema='costcmp-budget-preflight-v2', status='PASS', records=records)
        self.path.write_text(json.dumps(v2))
        self.assertEqual(len(cmp.receipt(self.path, budget=True, floor=True)), 45)
        with self.assertRaises(AssertionError):
            cmp.receipt(self.path, budget=True)
        self.path.write_text(json.dumps(self.value))
        with self.assertRaises(AssertionError):
            cmp.receipt(self.path, budget=True, floor=True)
        bad = copy.deepcopy(v2)
        next(r for r in bad['records'] if r['name'] == cmp.BUDGET_FLOOR_ARM)['feature_values'][0] = .25000000000000006
        self.path.write_text(json.dumps(bad))
        with self.assertRaisesRegex(AssertionError, 'consumed-feature parity'):
            cmp.receipt(self.path, budget=True, floor=True)

    @patch.object(owner, 'timing')
    def test_floor_cli_times_five_arms_against_uncapped_under_the_lock(self, timing):
        manifest, _ = self.floor_manifest()
        records = copy.deepcopy(self.value['records'])
        for rec in self.value['records']:
            if rec['name'] == 'by_v2fy_r5_b128':
                records.append({**copy.deepcopy(rec), 'name': cmp.BUDGET_FLOOR_ARM})
        self.path.write_text(json.dumps(dict(schema='costcmp-budget-preflight-v2', status='PASS', records=records)))
        floor_binary = self.root / cmp.BUDGET_FLOOR_ARM
        with patch('sys.argv', ['costcmp_run.py', 'timing', '--budget-grid', '--floor-arm',
                                '--budget-binaries', str(manifest), '--binary', str(floor_binary),
                                '--dest', str(self.root/'timing'), '--parity', str(self.path),
                                '--analyzer', 'analyzer', '--lock', str(self.root/'heavy.lock')]):
            cmp.main()
        kwargs = timing.call_args.kwargs
        self.assertEqual(kwargs['arms'], cmp.BUDGET_FLOOR_ARMS)
        self.assertEqual(kwargs['baseline'], 'by_v2fy_r5_before')
        self.assertEqual(kwargs['lock'], self.root/'heavy.lock')
        self.assertEqual(kwargs['worker_binaries'][cmp.BUDGET_FLOOR_ARM], floor_binary)
        self.assertEqual(len(kwargs['cells']), 9)
        self.assertEqual(owner.worker_revision(cmp.BUDGET_FLOOR_ARM), 5)

    def test_floor_arm_is_parity_and_timing_only(self):
        for argv in [['rss', '--budget-grid', '--floor-arm'], ['timing', '--floor-arm']]:
            with self.subTest(argv[0]):
                with patch('sys.argv', ['costcmp_run.py', *argv, '--dest', str(self.root)]):
                    with self.assertRaises(SystemExit) as result:
                        cmp.main()
                self.assertEqual(result.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
