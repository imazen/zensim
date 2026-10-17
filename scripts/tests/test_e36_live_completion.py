"""Invented live-tail control-flow oracle; no native work, targets or fits."""
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_live_completion import install_current

SOURCE='''def bulk(root):
    start=1.0;deadline=10.0;qualified=None;errors=[];plan={'slots':1}
    mandatory='fallback';extra='additions'
    run(root,mandatory,plan['slots'],deadline=deadline)
    if qualify(root,mandatory,start,deadline):
        qualified='fallback'
        if time.time()<deadline:
            run(root,extra,plan['slots'],deadline=deadline)
            if qualify(root,extra,start,deadline):qualified='full'
    choice={'qualified':qualified,'deadline':deadline,'closed':time.time()}
    return choice
'''


class LiveCompletion(unittest.TestCase):
    def test_original_clock_and_predicate_survive_late_control_return(self):
        for timely,expected in [(True,'full'),(False,'fallback')]:
            with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-completion-') as d:
                root=Path(d);source=root/'e36_bulk.py';source.write_text(SOURCE);receipt=root/'hook.json';calls=[]
                def run(r,name,slots,deadline):
                    calls.append(name)
                    if name=='fallback':install_current(root,receipt)
                def qualify(r,name,start,deadline):
                    self.assertEqual((start,deadline),(1.,10.))
                    return name=='fallback' or timely
                env=dict(run=run,qualify=qualify,time=types.SimpleNamespace(time=lambda:11.))
                try:
                    exec(compile(SOURCE,str(source),'exec'),env)
                    result=env['bulk'](root)
                finally:sys.settrace(None)
                self.assertEqual(result,dict(qualified=expected,deadline=10.,closed=11.))
                self.assertEqual(calls,['fallback'])
                self.assertIsNone(sys.gettrace())
                self.assertEqual(json.loads((root/'hook-applied.json').read_text())['qualified'],expected)


if __name__=='__main__':unittest.main()
