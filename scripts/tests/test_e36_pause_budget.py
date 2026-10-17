"""Invented budget/clock/frame fixtures; no producers, fits or human targets."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_pause_budget import live_python,resolve,sha,unix_ns


def write(path,value):path.write_text(json.dumps(value))


def fixture(root):
    base=root/'original.json';write(base,dict(started_unix=1000.125,deadline_unix=22600.125,plan_sha256='fixed',fits=0))
    events=root/'events.json';write(events,dict(freeze_unix_ns=2000_000_000_000,resume_unix_ns=3000_123_456_789))
    a=dict(schema='e36-authorized-frozen-budget-v1',authorization='coordinator-2026-10-17-frozen-time-exclusion',scientific_change=False,fits=0,
        original_clock=dict(path=str(base),sha256=sha(base)),freeze_evidence=dict(path=str(events),sha256=sha(events)),
        freeze_unix_ns=2000_000_000_000,resume_unix_ns=3000_123_456_789,excluded_frozen_ns=1000_123_456_789,
        original_deadline_unix_ns=unix_ns(22600.125),effective_deadline_unix_ns=unix_ns(22600.125)+1000_123_456_789)
    ap=root/'amendment.json';write(ap,a)
    clock={**json.loads(base.read_text()),'deadline_unix':a['effective_deadline_unix_ns']/1e9,'budget_only_amendment':dict(path=str(ap),sha256=sha(ap))}
    cp=root/'effective.json';write(cp,clock)
    return base,ap,cp,a,clock


class PauseBudget(unittest.TestCase):
    def test_exact_interval_preserves_original_and_refuses_changed_budget(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-clock-') as d:
            base,ap,cp,a,clock=fixture(Path(d));original=base.read_bytes()
            self.assertEqual(resolve(cp)[1],clock)
            self.assertEqual(base.read_bytes(),original)
            for change in [dict(deadline_unix=clock['deadline_unix']+1e-6),dict(plan_sha256='changed'),dict(started_unix=1001.125)]:
                write(cp,{**clock,**change})
                with self.assertRaisesRegex(ValueError,'budget chain'):resolve(cp)
            write(cp,clock)
            for change in [dict(scientific_change=True),dict(excluded_frozen_ns=a['excluded_frozen_ns']+1),dict(resume_unix_ns=a['resume_unix_ns']+1),dict(authorization='unregistered')]:
                changed={**a,**change};write(ap,changed);write(cp,{**clock,'budget_only_amendment':dict(path=str(ap),sha256=sha(ap))})
                with self.assertRaisesRegex(ValueError,'budget chain'):resolve(cp)
            write(ap,a);write(cp,clock);base.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'original clock'):resolve(cp)

    def test_live_frame_changes_only_deadline_and_keeps_numerical_locals(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-live-frame-') as d:
            root=Path(d);_,_,cp,_,clock=fixture(root);out=root/'receipt.json'
            code=compile('def execute():\n deadline=22600.125\n q=[0,1,20,60,100]\n weight=2.0\n live_python(cp,out)\n return deadline,q,weight\nresult=execute()','e36_data.py','exec')
            env=dict(live_python=live_python,cp=cp,out=out)
            with patch.dict(os.environ,{},clear=False):exec(code,env)
            self.assertEqual(env['result'],(clock['deadline_unix'],[0,1,20,60,100],2.0))
            receipt=json.loads(out.read_text());self.assertEqual(len(receipt['records']),1)
            self.assertEqual(receipt['records'][0]['old_deadline'],22600.125)


if __name__=='__main__':unittest.main()
