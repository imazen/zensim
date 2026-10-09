"""Full-grid comparison and parent-evidence regression tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import speedq_run as runner
import speed_matrix_report as report
import test_speedq as base


class SpeedqExtensionsTest(unittest.TestCase):
    setUp = base.SpeedqTest.setUp
    row = base.SpeedqTest.row
    full_report_parity_fixture = base.SpeedqTest.full_report_parity_fixture

    def test_speedq_comparison_preserves_cell_cis_and_refuses_mismatched_evidence(self):
        before=dict(timing_coverage=[192,192],model_source_sha256=runner.SOURCE_SHA,
                    statistics_owner='pinned engine',
                    axes=dict(tiers=runner.TIERS,threads=runner.THREADS,geometries=runner.GEOMETRIES,arms=runner.ARMS),
                    medians_ns=[[[100]*7 for _ in range(8)] for _ in range(24)],
                    r5_minus_r4_ci_ns=[[[10,20,30] for _ in range(8)] for _ in range(24)],
                    r5_vs_r4_resolution_limited=[['0']*8 for _ in range(24)],
                    verdict=dict(slower_cells=192,faster_cells=0,inconclusive_cells=0))
        after=copy.deepcopy(before)
        after['medians_ns'][0][0][2]=50
        after['r5_minus_r4_ci_ns'][0][0]=[-30,-20,-10]
        after['r5_vs_r4_resolution_limited'][0][1]='1'
        after['r5_vs_r4_cell_verdict']=[['slower']*8 for _ in range(24)]
        after['r5_vs_r4_cell_verdict'][0][0]='faster'
        after['r5_vs_r4_cell_verdict'][0][1]='inconclusive'
        after['r5_minus_r4_ci_ns'][0][2]=[0,0,1]
        after['verdict']=dict(slower_cells=190,faster_cells=1,inconclusive_cells=1)
        result=report.speedq_comparison(before,after)
        self.assertEqual(len(result['cells']),192)
        first=result['cells'][0]
        self.assertEqual((first['tier'],first['threads'],first['geometry']),('v4x',1,'64x64'))
        self.assertEqual((first['before_ci_ns'],first['after_ci_ns']),([10,20,30],[-30,-20,-10]))
        self.assertEqual((first['before_verdict'],first['after_verdict']),('slower','faster'))
        self.assertEqual(first['rev5_median_change_pct'],-50)
        self.assertEqual(result['cells'][1]['after_verdict'],'inconclusive')
        self.assertEqual(result['cells'][2]['after_verdict'],'slower')
        self.assertEqual((result['cells'][-1]['tier'],result['cells'][-1]['threads'],result['cells'][-1]['geometry']),('scalar',32,'1920x1080'))
        self.assertIn('not a paired between-build',report.speedq_comparison_markdown(result,'baseline.json'))
        for field,value in [('timing_coverage',[191,192]),('model_source_sha256','research'),('statistics_owner','different'),('axes',dict(before['axes'],threads=[32,16,8,4,2,1]))]:
            bad=copy.deepcopy(after);bad[field]=value
            with self.assertRaises(AssertionError):report.speedq_comparison(before,bad)


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
        batches=[]
        for i,(offset,end) in enumerate([(0,32),(32,34)]):
            name=f'zenbench.batch-{i}.json'
            batches.append(dict(path=name,round_offset=offset,rounds_requested=end-offset,rounds_total=end-offset,warmup_ms=20 if i==0 else 0))
            (dest/name).write_text(json.dumps(dict(unreliable=False,comparisons=[dict(completed_rounds=end-offset,benchmarks=[dict(name=a) for a in values],samples=[dict(gate_clean=f,iterations=1) for f in flags[offset:end]])])))
        (dest/'zenbench.json').write_text(json.dumps(dict(schema='speedq-parent-batches-v1',round_cap=64,rounds_total=34,batches=batches)))
        args=SimpleNamespace(raw_dir=self.root,out_json=self.root/'out.json',out_md=self.root/'out.md')
        report.speedq_report(args);out=json.loads(args.out_json.read_text())
        expected=__import__('statistics').median(selected['by_v2fy_r4'])
        self.assertEqual(out['medians_ns'][0][0][1],expected)
        self.assertNotEqual(expected,__import__('statistics').median(values['by_v2fy_r4']))
        self.assertEqual(out['timing_round_selection'][0][0],selection)
        self.assertEqual(out['timing_round_rule_counts'],{runner.LEGACY_ROUND_RULE:0,runner.CLEAN_ROUND_RULE:1})
        self.assertIn('Owner-approved clean-round amendment (2026-10-08)',args.out_md.read_text())
        self.assertEqual(out['parent_batch_validation']['segments'],1)
        parent_path=dest/'zenbench.batch-1.json';parent=json.loads(parent_path.read_text())
        parent['comparisons'][0]['samples'][0]['gate_clean']=False
        parent_path.write_text(json.dumps(parent))
        with self.assertRaisesRegex(AssertionError,'parent and worker gate flags differ'):
            report.speedq_report(args)
        parent['comparisons'][0]['samples'][0]['gate_clean']=True
        parent_path.write_text(json.dumps(parent))
        batches[1]['round_offset']=31
        (dest/'zenbench.json').write_text(json.dumps(dict(schema='speedq-parent-batches-v1',round_cap=64,rounds_total=34,batches=batches)))
        with self.assertRaisesRegex(AssertionError,'parent batch offsets differ'):
            report.speedq_report(args)
        batches[1]['round_offset']=32
        (dest/'zenbench.json').write_text(json.dumps(dict(schema='speedq-parent-batches-v1',round_cap=64,rounds_total=34,batches=batches)))
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
        baseline=self.root/'baseline.json';baseline.write_bytes(args.out_json.read_bytes())
        args.speedq_baseline=baseline;args.comparison_md=self.root/'comparison.md'
        report.speedq_report(args)
        comparison=json.loads(args.out_json.read_text())['before_after']
        self.assertEqual(len(comparison['cells']),192)
        self.assertEqual(comparison['cells'][0]['before_ci_ns'],comparison['cells'][0]['after_ci_ns'])
        self.assertEqual(comparison['cells'][0]['rev5_median_change_pct'],0)
        self.assertEqual(comparison['baseline_sha256'],__import__('hashlib').sha256(baseline.read_bytes()).hexdigest())
        self.assertIn('| scalar | 32 | 1920x1080 |',args.comparison_md.read_text())
        del args.speedq_baseline;del args.comparison_md
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


if __name__ == "__main__":
    unittest.main()
