#!/usr/bin/env python3
"""Exercise stage reuse, invalidation and interrupted-run recovery end to end.

Fake instruments count invocations; the actual orchestration, fixture-grid
identity owner and stage files execute unchanged. No metric is approximated.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[2]

class Stages(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='zensim-stages-')
        self.root = Path(self.tmp.name)
        self.out = self.root / 'output with spaces'
        self.fixtures = self.root / 'fixtures'
        self.fixtures.mkdir()
        for content in ('city', 'dog', 'girl'):
            for size in (576, 384, 256):
                (self.fixtures / (f'{content}.png' if size == 576 else f'{content}_{size}.png')).write_text('reference')
                for q in (20, 50, 75):
                    (self.fixtures / f'{content}_{size}_q{q}.jpg').write_text('distortion')
        self.bake = self.root / 'candidate.bin'
        self.bake.write_text('model-one')
        self.table = self.root / 'table.parquet'
        self.table.write_text('data-one')
        self.heavy = self.program('heavy', '#!/bin/bash\nshift 4\nexec "$@"\n')
        self.verdict = self.program('verdict', '''#!/usr/bin/env python3
import hashlib,json,os,pathlib,sys
args=sys.argv[1:]
def value(flag): return args[args.index(flag)+1]
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
if '--print-features-root' in args: print(os.environ['TEST_ROOT']);sys.exit(0)
identity={'model':sha(value('--bake')), 'table':sha(os.environ['TEST_ROOT']+'/table.parquet'), 'evaluator':sha(__file__)}
if '--print-inputs' in args: print(json.dumps(identity));sys.exit(0)
with open(os.environ['TEST_ROOT']+'/verdict.calls','a') as f: f.write('score\\n')
pathlib.Path(value('--fulleval')).write_text(json.dumps({'rank':{'control':1},'scoring':{'surface':'zensim::BakeScorer'},'input_identity':identity}))
pathlib.Path(value('--output')).write_text('report')
''')
        self.diffmap = self.program('diffmap', '''#!/bin/bash
echo score >> "$TEST_ROOT/diffmap.calls"
[[ "${TEST_FAIL:-0}" == 1 ]] && exit 1
printf '  M3 =0.8\n  M3a =0.9\nmass: 0.2\n'
''')
        self.env = dict(os.environ, TEST_ROOT=str(self.root), ZENSIM_RUN_HEAVY=str(self.heavy),
            ZENSIM_BAKE_VERDICT=str(self.verdict), ZENSIM_DIFFMAP_BIN=str(self.diffmap),
            ZENSIM_M3_FIXTURES=str(self.fixtures), ZENSIM_FULLEVAL_OUT=str(self.out))
        for key in ('ZENSIM_M3_ONLY','ZENSIM_M3_REUSE','ZENSIM_EVAL_STAGE'):
            self.env.pop(key, None)

    def tearDown(self):
        self.tmp.cleanup()

    def program(self, name, text):
        p = self.root / name
        p.write_text(text)
        p.chmod(0o755)
        return p

    def run_stage(self, stage='all', success=True, **env):
        result = subprocess.run([str(REPO/'scripts/run_full_eval.sh'), '--stage', stage,
            str(self.bake),'control','372',str(self.root)], env=dict(self.env, **env), capture_output=True, text=True)
        self.assertEqual(result.returncode == 0, success, result.stderr)
        return result

    def counts(self):
        return tuple(len((self.root/f'{name}.calls').read_text().splitlines()) if (self.root/f'{name}.calls').exists() else 0 for name in ('verdict','diffmap'))

    def test_identity_reuse_and_recovery(self):
        self.run_stage('verdict')
        self.assertEqual(self.counts(), (1,0))
        self.run_stage()
        self.assertEqual(self.counts(), (1,27))
        aggregate = self.out/'control.fulleval.json'
        report = json.loads(aggregate.read_text())
        report['dial_ladder'] = {'contract':'FAIL'}
        report['qualification'] = {'status':'failed'}
        aggregate.write_text(json.dumps(report))
        self.run_stage()
        self.assertEqual(self.counts(), (1,27))
        reused = json.loads(aggregate.read_text())
        self.assertEqual(reused['dial_ladder'], {'contract':'FAIL'})
        self.assertNotIn('qualification', reused)
        self.table.write_text('data-two')
        self.run_stage()
        self.assertEqual(self.counts(), (2,27))
        self.assertNotIn('dial_ladder', json.loads(aggregate.read_text()))
        (self.fixtures/'city.png').write_text('new-reference-era')
        self.run_stage(success=False, TEST_FAIL='1')
        self.assertEqual(self.counts(), (2,54))
        self.assertTrue((self.out/'control.verdict-stage.json').exists())
        self.run_stage('coherence')
        self.assertEqual(self.counts(), (2,81))
        self.bake.write_text('model-two')
        self.run_stage('coherence', success=False)
        self.assertEqual(self.counts(), (2,81))
        self.run_stage()
        self.assertEqual(self.counts(), (3,108))
        report=json.loads((self.out/'control.fulleval.json').read_text())
        self.assertEqual(report['evaluation_stages'], {'verdict':'complete','coherence':'complete'})

    def test_harvest_invokes_owner_once_and_validates_existing_results(self):
        env = dict(self.env, ZENSIM_VERDICT_DIR=str(self.root/'verdicts'))
        cmd = [str(REPO/'scripts/harvest_bakes.sh'), '--bake', str(self.bake),
            '--stem', 'control', '--regime', '372', '--features-root', str(self.root),
            '--no-fulleval', '--heartbeat', str(self.root/'harvest')]
        for expected in (1, 1, 2):
            if expected == 2:
                self.bake.write_text('replacement-model')
            result = subprocess.run(cmd, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(self.counts(), (expected, 0))
        self.assertTrue((self.root/'verdicts/control.full.json').exists())

    def test_missing_fixture_never_shrinks_grid(self):
        (self.fixtures/'dog_384_q50.jpg').unlink()
        self.run_stage(success=False)
        self.assertEqual(self.counts(), (1,0))
        self.assertTrue((self.out/'control.verdict-stage.json').exists())

if __name__ == '__main__':
    unittest.main()
