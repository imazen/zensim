"""Registered seed floor and sequential-incumbent counterexamples."""
import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rev4_featpot'))
from e34_verdict import GATE_KEYS, steer_rule, verdict


class E34VerdictTest(unittest.TestCase):
    def fixture(self):
        pairs = [('w-a', 'base-a'), ('k-a', 'base-a'), ('l-a', 'base-a'),
                 ('w-c', 'base-c'), ('k-c', 'base-c'), ('k-a', 'w-a'),
                 ('l-a', 'w-a'), ('l-a', 'k-a'), ('k-c', 'w-c')]
        tests = [dict(test=i+1, arm=a, control=c, state='COMPLETE',
                      dataset_guards=[dict(dataset=d, point_guard=True)
                                      for d in ['kadid','tid2013','konfig','cid22_a25']],
                      w2=dict(point_guard=True), e33_mean_point_guard=True,
                      improvement_point_guard=True) for i, (a, c) in enumerate(pairs)]
        stats = dict(eligibility_margin=.0065, resamples=10000, level=.95,
                     percentile='Hyndman-Fan type 7', tests=tests, holm_reject=[True]*9)
        gates = {a: {k: dict(pass_=True) for k in GATE_KEYS} for a, _ in pairs[:5]}
        return stats, gates

    def test_seed_floor_is_not_the_old_paired_control_rule(self):
        self.assertFalse(steer_rule([135,132,135], [128,128,130])['pass_'])
        self.assertFalse(steer_rule([135,133,133], [128,128,134])['pass_'])
        self.assertFalse(steer_rule([134,135,135], [128,128,130])['pass_'])
        self.assertTrue(steer_rule([135,133,134], [128,128,134])['pass_'])

    def test_l_compares_with_the_actual_incumbent(self):
        stats, gates = self.fixture()
        stats['holm_reject'][5] = False  # K cannot replace W.
        stats['holm_reject'][7] = False  # L-vs-K is irrelevant.
        self.assertEqual(verdict(stats,gates)['families']['a']['incumbent'],'l-a')
        gates['w-a']['G-STEER']['pass_'] = False
        self.assertEqual(verdict(stats,gates)['families']['a']['incumbent'],'k-a')

    def test_ineligible_arm_never_enters_and_practical_bar_is_strict(self):
        stats, gates = self.fixture()
        stats['holm_reject'][2] = False
        stats['tests'][5]['improvement_point_guard'] = False
        self.assertEqual(verdict(stats,gates)['families']['a']['incumbent'],'w-a')
        for gate in gates.values():
            gate['G-STEER']['pass_'] = False
        result=verdict(stats,gates)
        self.assertEqual(result['families']['a']['incumbent'],'base-a')
        self.assertEqual(result['families']['c']['incumbent'],'base-c')

    def test_incomplete_hypothesis_and_wrong_margin_never_qualify(self):
        stats, gates = self.fixture()
        incomplete=copy.deepcopy(stats);incomplete['tests'][8]['state']='INCOMPLETE'
        self.assertEqual(verdict(incomplete,gates)['state'],'INCOMPLETE')
        stats['eligibility_margin']=.002
        with self.assertRaises(AssertionError):verdict(stats,gates)

    def test_missing_gate_is_incomplete_even_when_every_present_gate_passes(self):
        stats, gates=self.fixture()
        del gates['k-a']['K4']
        self.assertEqual(verdict(stats,gates)['state'],'INCOMPLETE')


if __name__ == '__main__':
    unittest.main()
