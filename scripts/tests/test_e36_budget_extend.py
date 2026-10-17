"""Budget-chain and runtime-bound fixtures; no native producers or fits."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).parents[1]/'rev4_featpot'))
from e36_pause_budget import resolve,sha,unix_ns
import e36_budget_extend as owner


class Extension(unittest.TestCase):
    def test_projection_extension_preserves_parent_and_rejects_science_change(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-extension-') as d:
            root=Path(d);parent=root/'parent.json';previous=dict(started_unix=1792216099.,deadline_unix=1792237699.,plan_sha256='frozen',fits=0)
            parent.write_text(json.dumps(previous));projection=root/'projection.json';projection.write_text('{}')
            a=dict(schema='e36-authorized-projection-budget-v1',authorization='coordinator-2026-10-17-deadline-1700Z',previous_clock=dict(path=str(parent),sha256=sha(parent)),projection=dict(path=str(projection),sha256=sha(projection)),previous_deadline_unix_ns=unix_ns(previous['deadline_unix']),effective_deadline_unix_ns=1792256400000000000,scientific_change=False,fits=0)
            ap=root/'amendment.json';ap.write_text(json.dumps(a))
            new={**previous,'deadline_unix':owner.DEADLINE,'budget_only_extension':dict(path=str(ap),sha256=sha(ap))};cp=root/'clock.json';cp.write_text(json.dumps(new))
            self.assertEqual(resolve(cp)[:2],(previous,new))
            for key,value in [('plan_sha256','changed'),('started_unix',0),('deadline_unix',owner.DEADLINE+1)]:
                cp.write_text(json.dumps({**new,key:value}))
                with self.assertRaisesRegex(ValueError,'budget chain'):resolve(cp)
            cp.write_text(json.dumps(new));projection.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError,'projection'):resolve(cp)

    def test_authorized_22z_preserves_entire_prior_extension(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-22z-') as directory:
            root=Path(directory);base=root/'base.json';base.write_text(json.dumps(dict(deadline_unix=100.,plan_sha256='frozen',fits=0)))
            projection=root/'projection.json';projection.write_text('{}');previous=base
            for index,(auth,deadline) in enumerate([('1700',1792256400.),('2200',1792274400.)]):
                old=json.loads(previous.read_text());ap=root/(str(index)+'-amendment.json')
                amendment=dict(schema='e36-authorized-projection-budget-v1',authorization='coordinator-2026-10-17-deadline-'+auth+'Z',previous_clock=dict(path=str(previous),sha256=sha(previous)),projection=dict(path=str(projection),sha256=sha(projection)),previous_deadline_unix_ns=unix_ns(old['deadline_unix']),effective_deadline_unix_ns=unix_ns(deadline),scientific_change=False,fits=0)
                ap.write_text(json.dumps(amendment));cp=root/(str(index)+'-clock.json');cp.write_text(json.dumps({**old,'deadline_unix':deadline,'budget_only_extension':dict(path=str(ap),sha256=sha(ap))}));previous=cp
            self.assertEqual(resolve(previous)[1]['deadline_unix'],1792274400.)
            altered=json.loads(previous.read_text());altered['plan_sha256']='changed';previous.write_text(json.dumps(altered))
            with self.assertRaisesRegex(ValueError,'budget chain'):resolve(previous)

    def test_live_view_retains_four_hour_bound(self):
        with tempfile.TemporaryDirectory(dir=Path.home()/'tmp',prefix='e36-bound-') as d:
            root=Path(d);cp=root/'clock.json';cp.write_text('{}');out=root/'receipt.json'
            source=root/'runtime-v6/program/e36_data.py'
            code=compile('def execute():\n ladder_started=90.0\n view={"deadline_seconds":14400}\n deadline=123.0\n owner.live(out)\n return deadline\nresult=execute()\n',str(source),'exec')
            with patch.object(owner,'ROOT',root),patch.object(owner,'policy',return_value=(cp,{})),patch.object(owner.time,'time',return_value=owner.DEADLINE-100),patch.object(owner.time,'monotonic',return_value=100):
                env=dict(owner=owner,out=out);exec(code,env)
            self.assertEqual(env['result'],owner.DEADLINE)
            out2=root/'receipt2.json'
            with patch.object(owner,'ROOT',root),patch.object(owner,'policy',return_value=(cp,{})),patch.object(owner.time,'time',return_value=owner.DEADLINE-20000),patch.object(owner.time,'monotonic',return_value=100):
                env=dict(owner=owner,out=out2);exec(code,env)
            self.assertEqual(env['result'],owner.DEADLINE-20000+14400-10)


if __name__=='__main__':unittest.main()
