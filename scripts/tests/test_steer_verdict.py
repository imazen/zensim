"""Invented gate packets and (when explicitly supplied) canonical Rust statistics."""
import copy
import json
import os
import unittest
from pathlib import Path
from scripts.rev4_featpot.e34_verdict import GATE_KEYS, STEER_PAIRS, STEER_IDENTICAL, steer_verdict


def fixture():
    tests = []
    for i, (a, c) in enumerate(STEER_PAIRS, 1):
        tests.append(dict(test=i, arm=a, control=c,
                          state='IDENTICAL_BY_CONSTRUCTION' if i in STEER_IDENTICAL else 'COMPLETE',
                          p=dict(decision=1 if i in STEER_IDENTICAL else .000001),
                          dataset_guards=[dict(dataset=d, point_guard=True) for d in ['kadid','tid2013','konfig','cid22_a25']],
                          w2=dict(point_guard=i not in (2,5))))
    stats=dict(schema='steer-statistics-v1',state='COMPLETE_FOR_S',alias_policy='q12-a-v1',
               eligibility_margin=.0065,resamples=10000,level=.95,percentile='Hyndman-Fan type 7',
               tests=tests,holm_reject=[i not in STEER_IDENTICAL for i in range(1,13)])
    gates={a:{k:dict(pass_=True) for k in GATE_KEYS}
           for a in ['base-a','base-c','s-a','s-c','r-a','r-c','sr-a','sr-c']}
    return stats,gates


class SteerVerdict(unittest.TestCase):
    def test_valid_aliases_allow_only_s(self):
        stats,gates=fixture();v=steer_verdict(stats,gates)
        self.assertEqual(v['state'],'COMPLETE_FOR_S')
        self.assertTrue(all(f['eligible'] for f in v['families'].values()))
        self.assertEqual(v['diagnostic_only'],['r-a','sr-a','r-c','sr-c'])

    def test_missing_or_invalid_required_evidence_is_packet_fatal(self):
        for which in ['slot','state','alias-p','pair','diagnostic-gate','guard','r-w2']:
            with self.subTest(which=which):
                s,g=fixture()
                if which=='slot':s['tests'].pop()
                elif which=='state':s['tests'][0]['state']='INCOMPLETE'
                elif which=='alias-p':s['tests'][1]['p']['decision']=0
                elif which=='pair':s['tests'][0]['arm']='r-a'
                elif which=='diagnostic-gate':g['r-a'].pop('spatial')
                elif which=='guard':s['tests'][0]['w2']['point_guard']=None
                elif which=='r-w2':s['tests'][1]['w2']['point_guard']=True
                self.assertEqual(steer_verdict(s,g)['state'],'INCOMPLETE')

    def test_measured_diagnostic_failure_is_not_missing(self):
        s,g=fixture();g['r-a']['spatial']['pass_']=False
        self.assertTrue(steer_verdict(s,g)['families']['a']['eligible'])

    def test_no_diagnostic_fallback(self):
        s,g=fixture();g['s-a']['G-STEER']['pass_']=False
        self.assertEqual(steer_verdict(s,g)['families']['a']['incumbent'],'base-a')

    def test_nonrejecting_s_and_strict_w2_false_do_not_qualify(self):
        s,g=fixture();s['holm_reject'][0]=False
        self.assertFalse(steer_verdict(s,g)['families']['a']['eligible'])
        s,g=fixture();s['tests'][0]['w2']['point_guard']=False
        self.assertFalse(steer_verdict(s,g)['families']['a']['eligible'])

    def test_pending_and_b_have_no_s_verdict(self):
        s,g=fixture();s['state']='PENDING_OWNER_Q12'
        self.assertEqual(steer_verdict(s,g)['state'],'PENDING_OWNER_Q12')
        s,g=fixture();s['alias_policy']='e34-strict-b-v1'
        v=steer_verdict(s,g);self.assertEqual(v['state'],'INCOMPLETE');self.assertEqual(v['scope'],'DIAGNOSTIC_ONLY')


def canonical_receipt_test():
    """Caller explicitly selects this integration; never silently skip it."""
    path=Path(os.environ['STEER_INVENTED_RESULT'])
    s=json.loads(path.read_text());_,g=fixture()
    v=steer_verdict(s,g)
    assert v['state']=='COMPLETE_FOR_S',v
    assert all(f['eligible'] for f in v['families'].values()),v
    print('PASS canonical Rust bootstrap/Student/Holm invented predictions -> S-only verdict; exact aliases retained')


if __name__=='__main__':
    unittest.main()
