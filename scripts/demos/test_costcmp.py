"""Admission negative controls for COSTCMP using the existing SPEEDQ owner."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import costcmp_report
import costcmp_run as cmp
import speedq_run as owner

class CostcmpTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path.home()/'tmp');self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'receipt.json'
        records=[]
        for g,t,n in cmp.cells():
            w,h=map(int,g.split('x'))
            for a in cmp.ARMS:
                model={'revision':'5','source_sha256':owner.SOURCE_SHA} if a==cmp.ARMS[0] else (
                    {'revision':'1','source_sha256':a[-1]*64} if a.startswith('zensim_') else
                    {'commit':cmp.FAST_MAIN} if a=='fast_ssim2_main' else {'version':'0.8.2'})
                records.append(dict(name=a,arm='by_v2fy' if a==cmp.ARMS[0] else a,width=w,height=h,tier=t,threads=str(n),
                                    model=model,input_sha256=g,score=50.,score_bits='4049000000000000',feature_values=[0.25]*420 if a==cmp.ARMS[0] else []))
        self.value=dict(schema='costcmp-preflight-v1',status='PASS',records=records)
    def save(self,value):self.path.write_text(json.dumps(value))
    def test_full_grid_admits_and_missing_or_duplicate_cells_refuse(self):
        self.save(self.value);self.assertEqual(len(cmp.receipt(self.path)),320)
        for records in (self.value['records'][:-1],self.value['records']+[self.value['records'][0]]):
            self.save({**self.value,'records':records})
            with self.assertRaises(AssertionError):cmp.receipt(self.path)
    def test_revision_model_input_and_strict_feature_changes_refuse(self):
        mutations=[lambda r:r['model'].update(revision='4'),lambda r:r['model'].update(source_sha256='0'*64),
                   lambda r:r.update(input_sha256='wrong'),lambda r:r['feature_values'].__setitem__(0,.25000000000000006)]
        for mutate in mutations:
            bad=copy.deepcopy(self.value);mutate(bad['records'][5]);self.save(bad)
            with self.assertRaises(AssertionError):cmp.receipt(self.path)
    def test_peer_ready_change_refuses_and_selects_shared_first_clean_indices(self):
        self.save(self.value);rows=cmp.receipt(self.path);rec=copy.deepcopy(rows[(owner.GEOMETRIES[0],'v4x',1,'fast_ssim2_main')]);rec['model']['commit']='wrong'
        with self.assertRaises(AssertionError):cmp.validate_ready(rows,'fast_ssim2_main',owner.GEOMETRIES[0],'v4x',1,rec)
        raw=dict(gate_clean=[False]+[True]*32,paired_rounds={a:list(range(33)) for a in cmp.ARMS},zenbench_unreliable=False)
        selected,selection=owner.select_clean_rounds(raw)
        self.assertEqual(selection['retained_indices'],list(range(1,33)))
        self.assertTrue(all(v==list(range(1,33)) for v in selected.values()))
    def test_freeze_keeps_both_fast_ssim2_sources(self):
        root=Path(self.temp.name);binary=root/'binary';binary.write_bytes(b'instrument');binary.chmod(0o755)
        messages=[dict(reason='build-finished',success=True),dict(reason='compiler-artifact',executable=str(binary),target=dict(name='ssim2_speed_bar',kind=['bench']))]
        for source in ('registry+index#fast-ssim2@0.8.2','git+https://github.com/imazen/fast-ssim2#0.9.0'):
            messages.append(dict(reason='compiler-artifact',package_id=source,features=['imgref','rayon'],target=dict(name='fast_ssim2',kind=['lib'])))
        log=root/'build.log';log.write_text('\n'.join(map(json.dumps,messages)))
        owner.freeze_binary(log,root/'frozen');receipt=json.loads((root/'frozen.artifact.json').read_text())
        self.assertEqual(set(receipt['dependencies']),{'fast_ssim2','fast_ssim2_main'})
        self.assertIn('registry+',receipt['dependencies']['fast_ssim2']['package_id'])
        self.assertIn('git+',receipt['dependencies']['fast_ssim2_main']['package_id'])
    def report_fixture(self):
        root=Path(self.temp.name)/'raw';(root/'parity').mkdir(parents=True);(root/'provenance').mkdir();(root/'rss').mkdir()
        (root/'parity/PREFLIGHT_PASS.json').write_text(json.dumps(self.value))
        (root/'provenance/instrument.artifact.json').write_text(json.dumps(dict(binary_sha256='a'*64,dependencies={})))
        (root/'provenance/paired-rounds-analyzer').write_bytes(b'fixture analyzer')
        (root/'provenance/source.json').write_text(json.dumps(dict(paired_analyzer_sha256=hashlib.sha256(b'fixture analyzer').hexdigest())))
        rows=cmp.receipt(root/'parity/PREFLIGHT_PASS.json')
        for g,t,n in cmp.cells():
            w,h=map(int,g.split('x'));dest=root/'timing'/f'{t}-t{n}-{g}';dest.mkdir(parents=True)
            values={a:[1000.+(i+1)*w*h]*32 for i,a in enumerate(cmp.ARMS)}
            inner=dict(gate_clean=[True]*32,paired_rounds=values,zenbench_unreliable=False,
                       workers={a:rows[(g,t,n,a)] for a in cmp.ARMS},timer_resolution_ns=1.)
            _,selection=owner.select_clean_rounds(inner)
            files={
                'COMPLETE.json':dict(status='PASS',rounds=32,paired_alignment_verified=True,zenbench_gate_clean=True,**selection),
                'header.json':dict(quiet_gate=dict(admitted=True,load1=1.),gate_trace=False,binary_sha256='a'*64,arms=cmp.ARMS,round_cap=64,round_rule=owner.CLEAN_ROUND_RULE),
                'interference.json':dict(admitted=True,foreign={}),
                'zenbench.inner.json':inner,
                'paired_analysis.json':dict(baseline_arm=cmp.ARMS[0],comparisons={a:dict(n_samples=32,ci_lower=1.,ci_median=2.,ci_upper=3.,resolution_limited=False) for a in cmp.ARMS[1:]}),
                'zenbench.json':dict(schema='speedq-parent-batches-v1',round_cap=64,rounds_total=32,
                                     batches=[dict(round_offset=0,warmup_ms=20,path='batch.json',rounds_total=32,rounds_requested=32)]),
                'batch.json':dict(unreliable=False,comparisons=[dict(completed_rounds=32,benchmarks=[dict(name=a) for a in cmp.ARMS],samples=[dict(iterations=1,gate_clean=True)]*32)])}
            for name,value in files.items():(dest/name).write_text(json.dumps(value))
        for g in owner.GEOMETRIES:
            for n in (1,32):
                for a in cmp.ARMS:
                    (root/'rss'/f'v4x-t{n}-{g}-{a}.json').write_text(json.dumps(dict(quiet_gate=dict(admitted=True,load1=1.),worker=rows[(g,'v4x',n,a)],max_rss_kib=1234)))
        return SimpleNamespace(raw_dir=root,out_json=Path(self.temp.name)/'out.json',out_md=Path(self.temp.name)/'out.md')
    @patch.object(costcmp_report.subprocess,'check_output')
    def test_report_validates_all_evidence_and_exact_measured_units(self,analyzer):
        analyzer.return_value=json.dumps([dict(n_samples=32,ci_lower=1.,ci_median=2.,ci_upper=3.,resolution_limited=False)]*256)
        args=self.report_fixture();costcmp_report.report(args);result=json.loads(args.out_json.read_text())
        self.assertEqual((result['timing_configurations'],result['arm_timings'],result['rss_observations']),(64,320,80))
        first=result['fits'][0];self.assertAlmostEqual(first['alpha_ns'],1000.);self.assertAlmostEqual(first['beta_ns_per_pixel'],1.)
        self.assertAlmostEqual(first['ms_per_mp_1024sq'],(1000.+1024**2)/1024**2)
        bad=args.raw_dir/'timing/v4x-t1-64x64/header.json';original=bad.read_text();value=json.loads(original);value['quiet_gate']['load1']=2.;bad.write_text(json.dumps(value))
        with self.assertRaises(AssertionError):costcmp_report.report(args)
        bad.write_text(original)
        bad=args.raw_dir/'rss/v4x-t1-64x64-fast_ssim2_main.json';original=bad.read_text();value=json.loads(original);value['worker']['model']['commit']='wrong';bad.write_text(json.dumps(value))
        with self.assertRaises(AssertionError):costcmp_report.report(args)
        bad.write_text(original)
        bad=args.raw_dir/'timing/v4x-t1-64x64/paired_analysis.json';original=bad.read_text();value=json.loads(original);value['comparisons']['zensim_A']['ci_lower']=0.;bad.write_text(json.dumps(value))
        with self.assertRaisesRegex(AssertionError,'replay'):costcmp_report.report(args)
        bad.write_text(original);(args.raw_dir/'timing/v4x-t1-64x64/COMPLETE.json').unlink()
        with self.assertRaises(FileNotFoundError):costcmp_report.report(args)
    def test_custom_grid_keeps_shared_lock_and_canonical_collector(self):
        with patch.object(owner,'segment_lock') as lock,patch.object(owner,'run_segment') as segment:
            owner.timing('binary','root',32,'receipt','analyzer',arms=cmp.ARMS,lock='shared',cells=cmp.cells()[:2],parity_loader=lambda p:{},baseline=cmp.ARMS[0],ready_check='validator')
            self.assertEqual(lock.call_count,2);self.assertEqual(segment.call_count,2)
            self.assertEqual(segment.call_args.kwargs['baseline'],cmp.ARMS[0])
            self.assertEqual(segment.call_args.kwargs['ready_check'],'validator')

if __name__=='__main__':unittest.main()
