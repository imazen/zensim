"""Invented two-reference native-receipt fixtures; never generate pixels or fit."""
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_store import sha
from e36_data import Q,PREFIX,canonical
from e36_materialize import build


def h(s):return hashlib.sha256(s.encode()).hexdigest()
def write(p,v):p.write_text(json.dumps(v))

class Materialize(unittest.TestCase):
    def fixture(self,root,alias):
        fx=json.loads((Path(__file__).parents[1]/'rev4_featpot/e33_fx1_declaration.json').read_text());ids=sorted(set(fx['direct'])|{b for _,b in fx['products']});self.assertEqual(len(ids),420)
        write(root/'FEATURE_LAYOUT_BINDING.json',dict(consumed_ids=ids))
        refs=[dict(origin=split,reference_key=split+'-32',size=32,width=32,height=32,content='photo',split=split,family='family-'+split,sha256=h('png-'+split),pixels_sha256=h('pixels-'+split),source_sha256=h('source-'+split)) for split in ['fit','selection']]
        refpath=root/'references.json';write(refpath,dict(rows=refs));write(root/'cohort-selection.json',dict(invented=True))
        budget=root/'budget.json';write(budget,dict(panels={key:dict(fallback=[r['reference_key'] for r in refs],full=[r['reference_key'] for r in refs]) for key in ['32:fast','32:rav1e']},references={r['reference_key']:r for r in refs}))
        runtime=root/'runtime.json';write(runtime,dict(references=str(refpath),source_budget=str(budget),mode_inventory_sha256={},mode_inventory='unused-fixture'))
        start=time.time()-5;completed=[];manifest=[]
        run=root/'runs/fallback';run.mkdir(parents=True)
        for ref in refs:
            for codec in ['jpeg','rav1e','svt','webp','jxl']:
                view=dict(runtime_sha256=sha(runtime),reference_key=ref['reference_key'],codec=codec,q_grid=Q,panel='baseline',role='bulk')
                rows=[]
                for q in Q:
                    enc=h(ref['origin']+('jpeg' if alias and codec=='rav1e' else codec)+str(q))
                    rows.append(dict(origin=ref['origin'],reference_sha256=ref['sha256'],reference_pixels_sha256=ref['pixels_sha256'],codec=codec,size=32,panel='baseline',split=ref['split'],content='photo',family=ref['family'],width=32,height=32,features=[float(np.float32(q/1000))]*420,feature_ids=ids,encoded_sha256=enc,distorted_pixels_sha256=h(enc),q=q,mode_key='{}',physical_alias=False,butteraugli_pnorm3=100.-q,cvvdp_standard_fhd=float(q)))
                blob=run/(ref['origin']+'-'+codec+'.gz');blob.write_bytes(gzip.compress(json.dumps(dict(schema='e36-native-ladder-receipt-v1',view=view,rows=rows,complete=True,completed_unix=time.time(),runtime_sha256=sha(runtime))).encode()))
                completed.append(dict(path=str(blob),sha256=sha(blob)));manifest.append(dict(kind=dict(view=PREFIX+canonical(view))))
        mp=run/'manifest.json';write(mp,manifest);write(run/'census.json',dict(admitted=True,output_receipts=completed));write(run/'declaration.json',dict(runtime=str(runtime),runtime_sha256=sha(runtime)))
        extra=root/'runs/additions';extra.mkdir();write(extra/'manifest.json',[])
        full=root/'runs/full';full.mkdir();write(full/'manifest.json',manifest)
        plan=root/'bulk-plan.json';write(plan,dict(schema='e36-six-hour-data-plan-v1',phases={name:dict(run=name,manifest_sha256=sha(root/'runs'/name/'manifest.json')) for name in ['fallback','additions','full']},deadline_seconds=21600,fit_release=False))
        clock=root/'bulk-clock.json';write(clock,dict(started_unix=start,deadline_unix=start+21600,plan_sha256=sha(plan)))
        write(root/'bulk-choice.json',dict(schema='e36-fixed-matrix-choice-v1',qualified='fallback',selected_runs=['fallback'],plan_sha256=sha(plan),clock_sha256=sha(clock),teacher_support_not_used_for_choice=True))
    def test_complete_export_and_real_post_dedup_support_blocker(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-invented-export-') as d:
            root=Path(d);self.fixture(root,False);out=root/'pools';build(root,['fallback'],out,'invented-only')
            report=json.loads((out/'_MANIFEST.json').read_text());self.assertTrue(report['required_support_admitted'])
            self.assertTrue(json.loads((out/'fit/semantic-admission.json').read_text())['admitted'])
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-invented-alias-') as d:
            root=Path(d);self.fixture(root,True);out=root/'pools'
            with self.assertRaisesRegex(ValueError,'zero post-dedup'):build(root,['fallback'],out,'invented-only')
            report=json.loads((out/'_MANIFEST.json').read_text());self.assertFalse(report['required_support_admitted'])
            self.assertTrue(any(json.loads(k)[0]=='rav1e' for k in report['zero_support_required_strata']))

if __name__=='__main__':unittest.main()
