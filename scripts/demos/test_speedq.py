"""Negative controls for the pre-timing parity refusal and report."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import speedq_run as runner
import speed_matrix_report as report


class SpeedqTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def row(self, tier='v4x', bits='4052bb6ca0000000', score=74.92850494384766):
        return dict(revision=3, tier=tier, threads='1', width=64, height=64,
                    score_bits=bits, score=score, input_sha256='same', model={}, feature_values=[1.0]*420)

    def test_parity_refuses_and_preserves_first_mismatch(self):
        dest = self.root / 'parity'
        rows = [self.row(), self.row('scalar', '4052bb6d60000000', 74.92855072021484)]
        with patch.object(runner, 'GEOMETRIES', ['64x64']), patch.object(runner, 'THREADS', [1]), patch.object(runner, 'TIERS', ['v4x', 'scalar']), patch.object(runner, 'worker_run', side_effect=[dict(rows[0]) for _ in range(3)] + [dict(rows[1])]) as worker:
            with self.assertRaisesRegex(RuntimeError, 'strict Rev4/Rev5 score parity bug'):
                runner.preflight('binary', dest)
        receipt = json.loads((dest / 'PARITY_FAILED.json').read_text())
        self.assertEqual(worker.call_count, 4)
        self.assertFalse(receipt['timing_started'])
        self.assertFalse((dest / 'PARITY_PASS.json').exists())
        self.assertEqual(receipt['different']['score_bits'], rows[1]['score_bits'])

    def test_parity_passes_identical_bits(self):
        dest = self.root / 'parity'
        with patch.object(runner, 'GEOMETRIES', ['64x64']), patch.object(runner, 'THREADS', [1]), patch.object(runner, 'TIERS', ['v4x', 'scalar']), patch.object(runner, 'worker_run', side_effect=lambda *args: self.row(args[4])):
            runner.preflight('binary', dest)
        self.assertEqual(json.loads((dest / 'PARITY_PASS.json').read_text())['status'], 'PASS')
        self.assertFalse((dest / 'PARITY_FAILED.json').exists())

    def test_rev3_uses_documented_feature_tolerance(self):
        old, rec = self.row(), self.row('scalar', 'different', 75.0)
        rec['feature_values'][13] += 1e-7
        self.assertLess(runner.compare_parity(old, rec, 3)['max_tolerance_fraction'], 1)
        rec['feature_values'][13] += 1e-3
        with self.assertRaisesRegex(RuntimeError, 'exceeds documented golden tolerance'):
            runner.compare_parity(old, rec, 3)

    def test_rev4_remains_bit_strict_despite_small_score_difference(self):
        with self.assertRaisesRegex(RuntimeError, 'strict Rev4/Rev5'):
            runner.compare_parity(self.row(), self.row('scalar', 'different'), 4)

    def test_rev5_cannot_hide_feature_bit_change_behind_same_score(self):
        old,rec=self.row(),self.row('scalar')
        rec['feature_values'][0]+=1e-15
        with self.assertRaisesRegex(RuntimeError,'feature-bit parity bug'):
            runner.compare_parity(old,rec,5)

    def test_quiet_gate_logs_busy_then_admitted(self):
        states = [{'admitted': False, 'load1': 2.1, 'foreign': {}, 'training': []}, {'admitted': True, 'load1': 1.0, 'foreign': {}, 'training': []}]
        with patch.object(runner, 'quiet_state', side_effect=states), patch.object(runner, 'refresh_activity'), patch.object(runner.time, 'sleep') as sleep:
            runner.quiet_gate(self.root / 'quiet.jsonl')
        self.assertEqual(sleep.call_count, 1)
        self.assertEqual(len((self.root / 'quiet.jsonl').read_text().splitlines()), 2)

    def test_incomplete_parity_receipt_cannot_admit_timing(self):
        path=self.root/'parity.json'
        path.write_text(json.dumps({'records':[self.row()], 'strict_revisions':[4,5], 'status':'PASS'}))
        with self.assertRaises(AssertionError):
            runner.parity_receipt(path)

    def test_training_detection_excludes_unrelated_python(self):
        proc = self.root / 'proc'; proc.mkdir()
        for pid, command in [('42', b'python3\0train_hybrid.py\0--run\0'), ('43', b'python3\0review.py\0')]:
            p = proc / pid; p.mkdir(); (p / 'cmdline').write_bytes(command)
        self.assertEqual(runner.training_processes(proc), [{'pid': 42, 'program': ['train_hybrid.py']}])

    def test_report_refuses_false_bug_and_keeps_timings_missing(self):
        (self.root / 'parity').mkdir()
        receipt = dict(status='STOP_SCORE_PARITY_BUG', timing_started=False,
                       baseline=self.row(), different=self.row('scalar', '4052bb6d60000000', 74.92855072021484), checked=[self.row()])
        path = self.root / 'parity/PARITY_FAILED.json'
        path.write_text(json.dumps(receipt))
        args = SimpleNamespace(raw_dir=self.root, out_json=self.root / 'out.json', out_md=self.root / 'out.md')
        report.speedq_stop_report(args)
        out = json.loads(args.out_json.read_text())
        self.assertEqual(out['status'],'HISTORICAL_REV3_BIT_REFUSAL')
        self.assertEqual(out['timing'], [])
        self.assertEqual(out['checked_model_parity_cells'], 1)
        self.assertIn('MISSING', args.out_md.read_text())
        receipt['different']['score_bits'] = receipt['baseline']['score_bits']
        path.write_text(json.dumps(receipt))
        with self.assertRaises(AssertionError):
            report.speedq_stop_report(args)


    def test_contaminated_rounds_are_archived_before_retry(self):
        calls=[]
        def segment(binary,root,geometry,tier,threads,*args):
            path=Path(root)/f'{tier}-t{threads}-{geometry}';path.mkdir()
            calls.append(path)
            if len(calls)==1:
                (path/'raw.json').write_text('contaminated raw evidence')
                raise runner.TimingNoise('foreign build')
            (path/'COMPLETE.json').write_text('{}')
        with patch.object(runner,'parity_receipt',return_value={}), patch.object(runner,'run_segment',side_effect=segment), patch.object(runner,'refresh_activity'), patch.object(runner.time,'sleep'):
            runner.timing('binary',self.root,32,'receipt','analyzer',['v4x-t1-64x64'])
        attempts=list(self.root.glob('*.bak'))
        self.assertEqual(len(attempts),1)
        self.assertEqual((attempts[0]/'raw.json').read_text(),'contaminated raw evidence')
        self.assertTrue((self.root/'v4x-t1-64x64/COMPLETE.json').exists())
        self.assertEqual(len(calls),2)

    def test_freeze_uses_cargo_receipt_and_refuses_ambiguous_artifacts(self):
        stale=self.root/'old';stale.write_bytes(b'old')
        current=self.root/'current';current.write_bytes(b'current');current.chmod(0o755)
        log=self.root/'build.log'
        artifact=dict(reason='compiler-artifact',executable=str(current),target=dict(name='ssim2_speed_bar',kind=['bench']))
        finish=dict(reason='build-finished',success=True)
        dependency=dict(reason='compiler-artifact',package_id='registry#fast-ssim2@0.8.2',features=['rayon'],target=dict(name='fast_ssim2',kind=['lib']))
        log.write_text('run-heavy chatter\n'+json.dumps(artifact)+'\n'+json.dumps(dependency)+'\n'+json.dumps(finish)+'\n')
        runner.freeze_binary(log,self.root/'frozen')
        self.assertEqual((self.root/'frozen').read_bytes(),b'current')
        self.assertTrue((self.root/'frozen').stat().st_mode & 0o111)
        receipt=json.loads((self.root/'frozen.artifact.json').read_text())
        self.assertEqual(receipt['dependencies']['fast_ssim2']['features'],['rayon'])
        other={**artifact,'executable':str(stale)}
        log.write_text(json.dumps(artifact)+'\n'+json.dumps(other)+'\n'+json.dumps(finish)+'\n')
        with self.assertRaisesRegex(AssertionError,'unambiguous'):runner.freeze_binary(log,self.root/'rejected')
        self.assertFalse((self.root/'rejected').exists())

    def test_full_report_fits_measured_axes_and_retains_missing_coverage(self):
        # A mathematical fixture tests the reporter's units and CI decisions;
        # these synthetic values never enter qualification output.
        parity=self.root/'full-parity';parity.mkdir()
        rows=[]
        for g in runner.GEOMETRIES:
            w,h=map(int,g.split('x'))
            for tier in runner.TIERS:
                for n in runner.THREADS:
                    for rev in [3,4,5]:
                        row=self.row(tier);row.update(width=w,height=h,threads=str(n),revision=rev)
                        rows.append(row)
        (parity/'PARITY_STRICT_PASS.json').write_text(json.dumps(dict(
            status='STRICT_PASS_LEGACY_TOLERANCE_FAIL',strict_revisions=[4,5],records=rows,
            rev3_differences=[dict(tolerance_violations=1,max_abs_feature_difference=2e-6,max_tolerance_fraction=2,score_difference=0)])))
        for g in runner.GEOMETRIES:
            pixels=__import__('math').prod(map(int,g.split('x')))
            dest=self.root/'timing'/f'v4x-t1-{g}';dest.mkdir(parents=True)
            (dest/'COMPLETE.json').write_text('{}')
            (dest/'zenbench.inner.json').write_text(json.dumps(dict(
                zenbench_unreliable=False,gate_clean=[True]*32,
                paired_rounds={a:[100+2*pixels]*32 for a in runner.ARMS})))
            (dest/'header.json').write_text(json.dumps(dict(rounds=32,quiet_gate=dict(admitted=True,load1=1))))
            (dest/'paired_analysis.json').write_text(json.dumps(dict(ci_lower=0.1,ci_median=1,ci_upper=1.9,resolution_limited=False,pct_change=0.5)))
        args=SimpleNamespace(raw_dir=self.root,out_json=self.root/'out.json',out_md=self.root/'out.md')
        report.speedq_report(args)
        out=json.loads(args.out_json.read_text())
        self.assertEqual(out['timing_coverage'],[8,192])
        self.assertEqual(out['status'],'INCOMPLETE')
        self.assertEqual(out['alpha_ns_beta_ns_per_pixel_r2'][0][0],[100,2,1])
        self.assertEqual(out['r5_minus_r4_ci_ns'][0][0],[0,1,2])
        self.assertEqual(out['r5_vs_r4_pct_change'][0][0],0.5)
        self.assertEqual(out['r5_vs_r4_resolution_limited'][0][0],'0')
        self.assertEqual(out['r5_vs_r4_resolution_limited'][1][0],'-')
        self.assertEqual(out['verdict']['slower_cells'],8)
        self.assertFalse(out['verdict']['rev5_at_least_as_fast_everywhere'])
        self.assertIn('MISSING',args.out_md.read_text())
        # Corrupt strict score evidence must refuse a plausible looking report.
        rows[0]['revision']=4;rows[0]['score_bits']='different'
        (parity/'PARITY_STRICT_PASS.json').write_text(json.dumps(dict(status='PASS',strict_revisions=[4,5],records=rows)))
        with self.assertRaises(AssertionError): report.speedq_report(args)


if __name__ == '__main__':
    unittest.main()
