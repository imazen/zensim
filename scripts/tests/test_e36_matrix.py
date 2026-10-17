"""Invented metadata/native-consensus-index fixtures. No training or labels."""
import copy
import gzip
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_store import sha
from e36_matrix import admit
from e36_semantic import edge,exact_support


def write(p,v):p.write_text(json.dumps(v))

class MatrixAdmission(unittest.TestCase):
    def test_coordinator_mandatory_choice_never_waits_for_optional(self):
        from unittest.mock import patch
        import e36_svt_matrix as owner
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-mandatory-choice-') as d:
            root=Path(d);folder=root/'owner-svt-ssd-v1';folder.mkdir()
            run=root/'runs/fallback';run.mkdir(parents=True)
            manifest=run/'manifest.json';write(manifest,[{'invented':'native-view'}])
            publication=run/'native-publication.json';write(publication,{'admitted':True})
            census=run/'census.json';write(census,dict(admitted=True,output_receipts=[{'invented':'output'}],native_publication=owner.bind(publication)))
            phase=dict(run='fallback',manifest=owner.bind(manifest),jobs=1)
            proof=dict(phases=dict(fallback=phase,additions=dict(run='optional')),plan=dict(path='plan',sha256='plan-sha'),clock=dict(path='clock',sha256='clock-sha'),scheduling={})
            write(folder/'matrix-binding.json',{})
            path=folder/'mandatory-choice-amendment-v1.json'
            amendment=dict(schema='e36-mandatory-only-choice-amendment-v1',authorization='coordinator-qualified-mandatory-optional-nongating',selected_phase='fallback',selected_runs=['fallback'],matrix=owner.bind(manifest),census=owner.bind(census),optional_fit_pool=False,teacher_support_used=False,fits=0)
            write(path,amendment)
            with patch.object(owner,'configuration',return_value=(proof,{},[])),patch.object(owner,'qualify',side_effect=AssertionError('no optional qualification or recensus')):
                choice=owner.finalize_choice(root)
            self.assertEqual(choice['selected_runs'],['fallback'])
            self.assertEqual(choice['mandatory_choice_amendment'],owner.bind(path))
            for field in ['optional_fit_pool','teacher_support_used']:
                write(path,dict(amendment,**{field:True}))
                with self.assertRaisesRegex(ValueError,'coordinator amendment'):owner.mandatory_choice(root,proof)
            write(path,amendment);census.write_text(census.read_text()+' ')
            with self.assertRaisesRegex(ValueError,'evidence changed'):owner.mandatory_choice(root,proof)

    def test_timely_full_evidence_is_qualified_after_late_control_return(self):
        from unittest.mock import patch
        import e36_bulk
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-late-control-') as d:
            root=Path(d);start=self.fixture(root)
            plan=json.loads((root/'bulk-plan.json').read_text())
            plan.update(runtime=str(root/'runtime.json'),runtime_sha256='pinned',resources_sha256='resources',slots=1)
            write(root/'runtime.json',{});write(root/'resource-admission.json',{})
            plan['runtime_sha256']=sha(root/'runtime.json');plan['resources_sha256']=sha(root/'resource-admission.json')
            write(root/'bulk-plan.json',plan)
            clock=json.loads((root/'bulk-clock.json').read_text());clock['plan_sha256']=sha(root/'bulk-plan.json');write(root/'bulk-clock.json',clock)
            # No replacement of a frozen outcome: this fixture's choice was
            # only setup metadata, removed before running the controller once.
            (root/'bulk-choice.json').unlink()
            with patch.object(e36_bulk,'run') as run,patch.object(e36_bulk.time,'time',return_value=start+21601):
                e36_bulk.bulk(root)
            self.assertEqual(run.call_count,1)
            choice=json.loads((root/'bulk-choice.json').read_text())
            self.assertEqual(choice['qualified'],'full')
            self.assertEqual(choice['selected_runs'],['fallback','additions'])
            self.assertEqual(admit(root,['fallback','additions'])['matrix'],'full')

    def fixture(self,root):
        start=time.time()-10;views={'fallback':['a'],'additions':['b'],'full':['a','b']};phases={}
        for phase,items in views.items():
            run=root/'runs'/phase;run.mkdir(parents=True)
            # Matrix owner treats the view as an opaque native identity.
            manifest=run/'manifest.json';write(manifest,[dict(kind=dict(view='e36-native-ladder-v1:'+json.dumps(dict(id=v),sort_keys=True,separators=(',',':')))) for v in items])
            phases[phase]=dict(run=phase,manifest_sha256=sha(manifest))
            receipts=[]
            if phase!='full':
                for v in items:
                    blob=run/(v+'.gz');blob.write_bytes(gzip.compress(json.dumps(dict(complete=True,completed_unix=start+1,view=dict(id=v))).encode()));os.utime(blob,(start+1,start+1))
                    receipts.append(dict(path=str(blob),sha256=sha(blob)))
                write(run/'census.json',dict(admitted=True,output_receipts=receipts))
        plan=root/'bulk-plan.json';write(plan,dict(schema='e36-six-hour-data-plan-v1',phases=phases,deadline_seconds=21600,fit_release=False))
        clock=root/'bulk-clock.json';write(clock,dict(started_unix=start,deadline_unix=start+21600,plan_sha256=sha(plan)))
        write(root/'bulk-choice.json',dict(schema='e36-fixed-matrix-choice-v1',qualified='full',selected_runs=['fallback','additions'],plan_sha256=sha(plan),clock_sha256=sha(clock),teacher_support_not_used_for_choice=True))
        return start
    def test_exact_matrix_and_original_deadline(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-matrix-') as d:
            root=Path(d);start=self.fixture(root)
            self.assertEqual(admit(root,['fallback','additions'])['matrix'],'full')
            for names in [['fallback','fallback'],['additions','fallback'],['fallback'],['additions']]:
                with self.assertRaises(ValueError):admit(root,names)
            blob=root/'runs/additions/b.gz';doc=json.loads(gzip.decompress(blob.read_bytes()));doc['completed_unix']=start+21601
            blob.write_bytes(gzip.compress(json.dumps(doc).encode()));os.utime(blob,(start+2,start+2))
            write(root/'runs/additions/census.json',dict(admitted=True,output_receipts=[dict(path=str(blob),sha256=sha(blob))]))
            with self.assertRaisesRegex(ValueError,'out-of-window'):admit(root,['fallback','additions'])
    def test_repair_hash_locks_schemas_and_exact_original_clock(self):
        from unittest.mock import patch
        import e36_bulk
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-clock-') as d:
            root=Path(d);write(root/'resource-admission.json',{'invented':True})
            old_runtime=root/'old-runtime.json';new_runtime=root/'new-runtime.json'
            runtime=dict(program_files={'e36_data.py':'a','e36_store.py':'b','e34_encode.py':'c'},build_commit='old',mode_inventory='old-path',native_tool='fixed')
            write(old_runtime,runtime);changed=dict(runtime,build_commit='new',mode_inventory='new-path',program_files=dict(runtime['program_files'],**{'e36_store.py':'new-client'}));write(new_runtime,changed)
            plans=[]
            for version,rp in [('old',old_runtime),('new',new_runtime)]:
                phases={}
                for phase,ids in [('fallback',['a']),('additions',['b']),('full',['a','b'])]:
                    run=version+'-'+phase;directory=root/'runs'/run;directory.mkdir(parents=True)
                    items=[dict(kind=dict(view='e36-native-ladder-v1:'+json.dumps(dict(id=i,runtime_sha256=sha(rp)),sort_keys=True,separators=(',',':'))),inputs=[sha(rp),'fixed-source'],cell={'q':-1},hint={'threads':1},requires=['fixed']) for i in ids]
                    mp=directory/'manifest.json';write(mp,items);phases[phase]=dict(run=run,manifest_sha256=sha(mp))
                plan=dict(schema='e36-six-hour-data-plan-v1',runtime=str(rp),runtime_sha256=sha(rp),resources_sha256=sha(root/'resource-admission.json'),source_budget_sha256='budget',deadline_seconds=21600,slots=28,fit_release=False,phases=phases)
                pp=root/('bulk-plan.json' if version=='old' else 'bulk-plan-v6.json');write(pp,plan);plans.append((pp,plan))
            original=root/'bulk-clock.json';start=time.time()-10;write(original,dict(started_unix=start,deadline_unix=start+21600,plan_sha256=sha(plans[0][0])))
            original_bytes=original.read_bytes();pp,plan=plans[1]
            self.assertTrue(e36_bulk.clock_equivalence(root,pp,plan,original)['admitted'])
            old_bytes=old_runtime.read_bytes();old_runtime.write_bytes(old_bytes+b' ')
            with self.assertRaisesRegex(ValueError,'runtime pin'):e36_bulk.clock_equivalence(root,pp,plan,original)
            old_runtime.write_bytes(old_bytes)
            with self.assertRaisesRegex(ValueError,'fresh window'):e36_bulk.bulk(root,pp,root/'bulk-clock-v6.json')
            with patch.object(e36_bulk,'run'),patch.object(e36_bulk,'qualify',return_value=False):
                with self.assertRaisesRegex(ValueError,'mandatory matrix'):e36_bulk.bulk(root,pp,root/'bulk-clock-v6.json',original)
            self.assertEqual(original.read_bytes(),original_bytes)
            amended=root/'bulk-clock-v6.json';clock=json.loads(amended.read_text());self.assertEqual(clock['started_unix'],start);self.assertEqual(clock['deadline_unix'],start+21600)
            clock['deadline_unix']+=1;write(amended,clock)
            with self.assertRaisesRegex(ValueError,'resumed repair'):e36_bulk.bulk(root,pp,amended,original)

    def test_semantic_native_direction_mode_and_required_set(self):
        db=sqlite3.connect(':memory:');db.executescript('CREATE TABLE features(id INTEGER,ref TEXT,enc TEXT);CREATE TABLE pairs(a INTEGER,b INTEGER,class INTEGER,band INTEGER,codec TEXT,size INTEGER,split TEXT,mode TEXT,qlo INTEGER,qhi INTEGER,source TEXT);')
        db.executemany('INSERT INTO features VALUES(?,?,?)',[(1,'r','worse'),(2,'r','better')]);db.execute("INSERT INTO pairs VALUES(1,2,0,0,'jpeg',32,'fit','{}',0,1,'native')")
        proof=dict(reference_sha256='r',worse_encoded_sha256='worse',better_encoded_sha256='better',codec='jpeg',size=32,mode_key='{}',q_low=0,q_high=1,consensus_receipt_sha256='native')
        edge(db,proof,0,0,'fit')
        for changes in [dict(mode_key='{"other":true}'),dict(worse_encoded_sha256='better',better_encoded_sha256='worse'),dict(consensus_receipt_sha256='different')]:
            changed=dict(proof,**changes)
            with self.assertRaisesRegex(ValueError,'absent'):edge(db,changed,0,0,'fit')
        report=dict(required_strata={'a':dict(selected_references=1,unique_preferences=3),'b':dict(selected_references=2,unique_preferences=4)})
        exact_support(report,{'a':1,'b':2},{'a':3,'b':4})
        missing=copy.deepcopy(report);del missing['required_strata']['b']
        with self.assertRaisesRegex(ValueError,'omitted'):exact_support(missing,{'a':1,'b':2},{'a':3,'b':4})
        db.close()

if __name__=='__main__':unittest.main()
