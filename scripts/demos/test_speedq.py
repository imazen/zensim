"""Negative controls for the pre-timing parity refusal and report."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import speedq_run as runner
import speed_matrix_report as report
import rev5perf_report as perf_report


class SpeedqTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.home() / 'tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def row(self, tier='v4x', bits='4052bb6ca0000000', score=74.92850494384766):
        return dict(revision=3, tier=tier, threads='1', width=64, height=64,
                    score_bits=bits, score=score, input_sha256='same', model={'source_sha256':runner.SOURCE_SHA}, feature_values=[1.0]*420)

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


    def test_contaminated_segment_does_not_starve_remaining_grid(self):
        calls=[]
        def segment(binary,root,geometry,tier,threads,*args,**kwargs):
            self.assertEqual(kwargs['arms'],runner.ARMS)
            calls.append(geometry)
            if geometry=='64x64' and calls.count(geometry)==1:
                path=Path(root)/f'{tier}-t{threads}-{geometry}'
                path.mkdir()
                (path/'raw.json').write_text('contaminated raw evidence')
                raise runner.TimingNoise('unclean round gate')
        with patch.object(runner,'GEOMETRIES',['64x64','128x128']), patch.object(runner,'THREADS',[1]), patch.object(runner,'TIERS',['v4x']), patch.object(runner,'parity_receipt',return_value={}), patch.object(runner,'run_segment',side_effect=segment), patch.object(runner,'refresh_activity'), patch.object(runner.time,'sleep') as sleep:
            runner.timing('binary',self.root,32,'parity','analyzer')
        self.assertEqual(calls,['64x64','128x128','64x64'])
        self.assertEqual(sleep.call_count,1)
        archived=list(self.root.glob('*.noise-*.bak'))
        self.assertEqual(len(archived),1)
        self.assertEqual((archived[0]/'raw.json').read_text(),'contaminated raw evidence')

    def test_contaminated_rounds_are_archived_before_retry(self):
        calls=[]
        def segment(binary,root,geometry,tier,threads,*args,**kwargs):
            self.assertEqual(kwargs['arms'],runner.ARMS)
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

    def test_clean_round_selection_excludes_flagged_rounds_from_both_arms(self):
        flags=[True]*34;flags[1]=False;flags[4]=None
        inner=dict(zenbench_unreliable=False,gate_clean=flags,
                   paired_rounds={'by_v2fy_r4':list(range(34)),
                                  'by_v2fy_r5':[1000+i for i in range(34)]})
        original=json.dumps(inner)
        values,selection=runner.select_clean_rounds(inner)
        expected=[i for i in range(34) if i not in (1,4)]
        self.assertEqual(values['by_v2fy_r4'],expected)
        self.assertEqual(values['by_v2fy_r5'],[1000+i for i in expected])
        self.assertEqual([b-a for a,b in zip(values['by_v2fy_r4'],values['by_v2fy_r5'])],[1000]*32)
        self.assertEqual(selection['retained_indices'],expected)
        self.assertEqual(selection['excluded_indices'],[1,4])
        self.assertEqual(selection['rounds_total'],34)
        self.assertEqual(selection['rounds_excluded'],2)
        self.assertEqual(selection['rule'],runner.CLEAN_ROUND_RULE)
        self.assertEqual(json.dumps(inner),original)

    def test_clean_round_cap_refuses_31_and_accepts_32_without_salvage(self):
        inner=dict(zenbench_unreliable=False,gate_clean=[False]*33+[True]*31,
                   paired_rounds={'by_v2fy_r4':list(range(64)),
                                  'by_v2fy_r5':list(range(1000,1064))})
        with self.assertRaisesRegex(runner.TimingNoise,'fewer than 32'):
            runner.select_clean_rounds(inner)
        inner['gate_clean'][32]=True
        values,selection=runner.select_clean_rounds(inner)
        self.assertEqual(selection['rounds_total'],64)
        self.assertEqual(selection['rounds_excluded'],32)
        self.assertEqual(values['by_v2fy_r4'],list(range(32,64)))
        inner['zenbench_unreliable']=True
        with self.assertRaisesRegex(runner.TimingNoise,'whole attempt unreliable'):
            runner.select_clean_rounds(inner)

    def test_clean_round_selection_refuses_misalignment_and_overcollection(self):
        inner=dict(zenbench_unreliable=False,gate_clean=[True]*32,
                   paired_rounds={'a':list(range(32)),'b':list(range(31))})
        with self.assertRaisesRegex(AssertionError,'unaligned'):
            runner.select_clean_rounds(inner)
        inner['paired_rounds']['b'].append(31)
        inner['gate_clean'].append(True)
        for series in inner['paired_rounds'].values():series.append(32)
        with self.assertRaisesRegex(AssertionError,'continued past'):
            runner.select_clean_rounds(inner)
        inner['gate_clean']=[False]*65
        inner['paired_rounds']={a:list(range(65)) for a in ['a','b']}
        with self.assertRaisesRegex(AssertionError,'at most 64'):
            runner.select_clean_rounds(inner)

    def test_timing_passes_explicit_six_arm_selection(self):
        selected=runner.ARMS[:-1]
        with patch.object(runner,'parity_receipt',return_value={}), patch.object(runner,'run_segment') as segment:
            runner.timing('binary',self.root,32,'receipt','analyzer',['v4x-t1-64x64'],selected)
        segment.assert_called_once_with('binary',self.root,'64x64','v4x',1,32,{},'analyzer',arms=selected)
        complete=self.root/'v4x-t1-64x64';complete.mkdir();(complete/'COMPLETE.json').write_text('{}')
        with patch.object(runner,'quiet_gate') as gate:
            runner.run_segment('binary',self.root,'64x64','v4x',1,32,{},'analyzer',arms=selected)
        gate.assert_not_called()

    def test_timing_cli_preserves_default_and_refuses_missing_core_arm(self):
        base=['speedq_run.py','timing','--dest',str(self.root),'--binary','binary','--parity','receipt','--analyzer','analyzer']
        for selected in [None,runner.ARMS[:-1]]:
            argv=base+(['--arms',*selected] if selected is not None else [])
            with patch('sys.argv',argv),patch.object(runner,'timing') as timing:
                runner.main()
            self.assertEqual(timing.call_args.args[-1],selected if selected is not None else runner.ARMS)
        with patch('sys.argv',base+['--arms',*runner.ARMS[1:]]),patch.object(runner,'timing') as timing:
            with self.assertRaises(SystemExit):runner.main()
        timing.assert_not_called()

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

    def full_report_parity_fixture(self):
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
        return parity,rows

    def test_research_weight_receipt_cannot_admit_production_timing(self):
        parity,rows=self.full_report_parity_fixture()
        # Identical score/feature bits cannot substitute another source model.
        rows[0]['model']={'source_sha256':'802c6369aa8e68c5458b32cbffa728f882779209d7822a9d1db0f78e4475f4a1'}
        path=parity/'PARITY_STRICT_PASS.json'
        rec=json.loads(path.read_text());rec['records']=rows;path.write_text(json.dumps(rec))
        with self.assertRaisesRegex(AssertionError,'frozen production model'):
            runner.parity_receipt(path)

    def test_frozen_comparison_rejects_common_mode_feature_and_score_changes(self):
        import math
        import shutil
        parity, rows = self.full_report_parity_fixture()
        before = self.root / 'before'
        after = self.root / 'after'
        shutil.copytree(parity, before / 'parity')
        shutil.copytree(parity, after / 'parity')
        self.assertEqual(perf_report.compare_bits(before, after)['strict_score_and_420_feature_checks'], 384)
        path = after / 'parity' / 'PARITY_STRICT_PASS.json'
        rec = json.loads(path.read_text())
        for row in rec['records']:
            row['feature_values'][0] = math.nextafter(1.0, 2.0)
        path.write_text(json.dumps(rec))
        runner.parity_receipt(path)  # Tier parity alone still passes.
        with self.assertRaisesRegex(AssertionError, 'frozen consumed feature bits changed'):
            perf_report.compare_bits(before, after)
        rec['records'] = rows
        for row in rec['records']:
            row['score_bits'] = '4052bb6ca0000001'
        path.write_text(json.dumps(rec))
        runner.parity_receipt(path)
        with self.assertRaisesRegex(AssertionError, 'frozen score bits changed'):
            perf_report.compare_bits(before, after)

    def test_full_report_fits_measured_axes_and_retains_missing_coverage(self):
        # A mathematical fixture tests the reporter's units and CI decisions;
        # these synthetic values never enter qualification output.
        parity,rows=self.full_report_parity_fixture()
        for g in runner.GEOMETRIES:
            pixels=__import__('math').prod(map(int,g.split('x')))
            dest=self.root/'timing'/f'v4x-t1-{g}';dest.mkdir(parents=True)
            (dest/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',rounds=32,paired_alignment_verified=True,zenbench_gate_clean=True)))
            (dest/'interference.json').write_text(json.dumps(dict(admitted=True,foreign=[])))
            (dest/'zenbench.inner.json').write_text(json.dumps(dict(
                zenbench_unreliable=False,gate_clean=[True]*32,
                paired_rounds={a:[100+2*pixels]*32 for a in runner.ARMS})))
            (dest/'header.json').write_text(json.dumps(dict(rounds=32,model_source_sha256=runner.SOURCE_SHA,quiet_gate=dict(admitted=True,load1=1))))
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
        self.assertIn('0.100 / 2 / 1.0000',args.out_md.read_text())
        # Corrupt strict score evidence must refuse a plausible looking report.
        rows[0]['revision']=4;rows[0]['score_bits']='different'
        (parity/'PARITY_STRICT_PASS.json').write_text(json.dumps(dict(status='PASS',strict_revisions=[4,5],records=rows)))
        with self.assertRaises(AssertionError): report.speedq_report(args)

    def test_report_uses_clean_rounds_and_verifies_completion_indices(self):
        self.full_report_parity_fixture()
        dest=self.root/'timing/v4x-t1-64x64';dest.mkdir(parents=True)
        flags=[True]*34;flags[1]=False;flags[4]=False
        values={a:[100+i for i in range(34)] for a in runner.ARMS[:-1]}
        for series in values.values():series[1]=series[4]=999999
        inner=dict(zenbench_unreliable=False,gate_clean=flags,paired_rounds=values)
        selected,selection=runner.select_clean_rounds(inner)
        complete=dict(status='PASS',rounds=32,paired_alignment_verified=True,zenbench_gate_clean=True,**selection)
        (dest/'COMPLETE.json').write_text(json.dumps(complete))
        (dest/'interference.json').write_text(json.dumps(dict(admitted=True,foreign=[])))
        (dest/'zenbench.inner.json').write_text(json.dumps(inner))
        (dest/'header.json').write_text(json.dumps(dict(rounds=32,round_cap=64,round_rule=runner.CLEAN_ROUND_RULE,arms=runner.ARMS[:-1],model_source_sha256=runner.SOURCE_SHA,quiet_gate=dict(admitted=True,load1=1))))
        (dest/'paired_analysis.json').write_text(json.dumps(dict(ci_lower=-1,ci_median=0,ci_upper=1,resolution_limited=False,pct_change=0)))
        args=SimpleNamespace(raw_dir=self.root,out_json=self.root/'out.json',out_md=self.root/'out.md')
        report.speedq_report(args);out=json.loads(args.out_json.read_text())
        expected=__import__('statistics').median(selected['by_v2fy_r4'])
        self.assertEqual(out['medians_ns'][0][0][1],expected)
        self.assertNotEqual(expected,__import__('statistics').median(values['by_v2fy_r4']))
        self.assertEqual(out['timing_round_selection'][0][0],selection)
        self.assertEqual(out['timing_round_rule_counts'],{runner.LEGACY_ROUND_RULE:0,runner.CLEAN_ROUND_RULE:1})
        self.assertIn('Owner-approved clean-round amendment (2026-10-08)',args.out_md.read_text())
        complete['excluded_indices']=[0,4];(dest/'COMPLETE.json').write_text(json.dumps(complete))
        with self.assertRaisesRegex(AssertionError,'selection differs'):
            report.speedq_report(args)

    def test_complete_report_maps_all_axes_and_does_not_call_uncertainty_equivalence(self):
        self.full_report_parity_fixture()
        provenance=self.root/'provenance';provenance.mkdir()
        (provenance/'test.artifact.json').write_text(json.dumps(dict(binary_sha256='test',dependencies={'test':dict(package_id='synthetic fixture',features=[])})))
        configs=[(t,n) for t in runner.TIERS for n in runner.THREADS]
        for k,(tier,n) in enumerate(configs):
            for g in runner.GEOMETRIES:
                pixels=__import__('math').prod(map(int,g.split('x')))
                dest=self.root/'timing'/f'{tier}-t{n}-{g}';dest.mkdir(parents=True)
                (dest/'COMPLETE.json').write_text(json.dumps(dict(status='PASS',rounds=32,paired_alignment_verified=True,zenbench_gate_clean=True)))
                (dest/'interference.json').write_text(json.dumps(dict(admitted=True,foreign=[])))
                slopes=[1,3,2,4,5,6,7]
                values={a:[1000*(k+1)+100*i+slopes[i]*pixels]*32 for i,a in enumerate(runner.ARMS)}
                (dest/'zenbench.inner.json').write_text(json.dumps(dict(zenbench_unreliable=False,gate_clean=[True]*32,paired_rounds=values)))
                (dest/'header.json').write_text(json.dumps(dict(rounds=32,binary_sha256='test',model_source_sha256=runner.SOURCE_SHA,quiet_gate=dict(admitted=True,load1=1))))
                delta=values['by_v2fy_r5'][0]-values['by_v2fy_r4'][0]
                (dest/'paired_analysis.json').write_text(json.dumps(dict(ci_lower=delta-1,ci_median=delta,ci_upper=delta+1,resolution_limited=False,pct_change=100*delta/values['by_v2fy_r4'][0])))
        rss=self.root/'rss';rss.mkdir()
        for g in runner.GEOMETRIES:
            for n in [1,32]:
                for a in runner.ARMS:
                    (rss/f'v4x-t{n}-{g}-{a}.json').write_text(json.dumps(dict(max_rss_kib=12345,quiet_gate=dict(admitted=True),worker={'model':{'source_sha256':runner.SOURCE_SHA}})))
        args=SimpleNamespace(raw_dir=self.root,out_json=self.root/'out.json',out_md=self.root/'out.md')
        report.speedq_report(args)
        out=json.loads(args.out_json.read_text())
        self.assertEqual(out['status'],'MEASURED')
        self.assertEqual(out['missing'],[])
        self.assertEqual(out['alpha_ns_beta_ns_per_pixel_r2'][23][6],[24600,7,1])
        self.assertEqual(out['medians_ns'][23][0][6],24600+7*64*64)
        self.assertEqual(out['rss_coverage'],[112,112])
        self.assertTrue(out['verdict']['rev5_at_least_as_fast_everywhere'])
        self.assertIn('faster in all 192 cells',args.out_md.read_text())
        # Explicitly omitted optional peer values must remain missing while
        # required six-arm timing coverage, gates and round counts stay intact.
        amended=self.root/'timing/scalar-t32-64x64'
        inner=amended/'zenbench.inner.json';rec=json.loads(inner.read_text())
        rec['paired_rounds'].pop('ssimulacra2_rs');inner.write_text(json.dumps(rec))
        header=amended/'header.json';rec=json.loads(header.read_text());rec['arms']=runner.ARMS[:-1];header.write_text(json.dumps(rec))
        report.speedq_report(args);out=json.loads(args.out_json.read_text())
        self.assertEqual(out['status'],'MEASURED')
        self.assertEqual(out['timing_coverage'],[192,192])
        self.assertEqual(out['timing_arm_coverage']['ssimulacra2_rs'],191)
        self.assertEqual(out['timing_arm_coverage']['by_v2fy_r5'],192)
        self.assertEqual(out['timing_segment_arm_counts'],{'6':1,'7':191})
        self.assertIsNone(out['medians_ns'][23][0][6])
        self.assertIsNone(out['alpha_ns_beta_ns_per_pixel_r2'][23][6])
        self.assertEqual(out['alpha_ns_beta_ns_per_pixel_r2'][23][2],[24200,2,1])
        self.assertIn('not measured',args.out_md.read_text())
        # Losing a core arm cannot be described as the coordinator amendment.
        rec=json.loads(inner.read_text());rec['paired_rounds'].pop('by_v2fy_r3');inner.write_text(json.dumps(rec))
        with self.assertRaises(AssertionError):report.speedq_report(args)
        rec['paired_rounds']['by_v2fy_r3']=[24000+64*64]*32;inner.write_text(json.dumps(rec))
        # One interval crossing zero makes the everywhere conclusion unproven.
        path=self.root/'timing/scalar-t32-64x64/paired_analysis.json'
        rec=json.loads(path.read_text());rec.update(ci_upper=1);path.write_text(json.dumps(rec))
        report.speedq_report(args)
        out=json.loads(args.out_json.read_text())
        self.assertIsNone(out['verdict']['rev5_at_least_as_fast_everywhere'])
        self.assertEqual(out['verdict']['inconclusive_cells'],1)
        # A completion filename cannot certify failed alignment or interference.
        complete=self.root/'timing/scalar-t32-64x64/COMPLETE.json'
        rec=json.loads(complete.read_text());rec['paired_alignment_verified']=False;complete.write_text(json.dumps(rec))
        with self.assertRaises(AssertionError): report.speedq_report(args)

        # Traced rounds are diagnostics even if every gate happened to pass.
        rec['paired_alignment_verified']=True;complete.write_text(json.dumps(rec))
        header=complete.parent/'header.json'
        rec=json.loads(header.read_text());rec['gate_trace']=True;header.write_text(json.dumps(rec))
        with self.assertRaisesRegex(AssertionError,'diagnostic tracing'):
            report.speedq_report(args)



if __name__ == '__main__':
    unittest.main()
