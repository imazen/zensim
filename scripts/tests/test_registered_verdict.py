import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'rev4_featpot'))
from registered_verdict import verdict, PAIRS, MODELS, GATE_KEYS


def fixture():
    stats=dict(schema='registered-factorial-statistics-v1',state='COMPLETE',decision_allowed=True,
        base_seed=20261016,resamples=10000,level=.95,percentile='Hyndman-Fan type 7',holm_reject=[True]*12,
        tests=[dict(test=i+1,arm=a,control=c,margin=.0065 if e else 0.,state='COMPLETE',
            dataset_guards=[dict(dataset=d,point_guard=True) for d in ('kadid','tid2013','konfig','cid22_a25')],
            w2=dict(point_guard=True),e33_mean_point_guard=False,improvement_point_guard=True)
            for i,(a,c,e) in enumerate(PAIRS)])
    gates={m:{k:dict(pass_=True) for k in GATE_KEYS} for m in MODELS}
    for m in MODELS:gates[m]['G-STEER']['counts']=[135,133,134]
    return stats,gates


class RegisteredVerdict(unittest.TestCase):
    def test_sequential_replacement_and_fixed_family_priority(self):
        s,g=fixture();v=verdict(s,g)
        self.assertEqual(v['acceptance_candidate'],'ph-a')
        self.assertEqual(v['acceptance_seed'],0)
        self.assertFalse(v['acceptance_read_authorized'])
        s['holm_reject'][3]=False # H cannot replace P; PH must compare to P, not H.
        s['tests'][4]['improvement_point_guard']=False
        self.assertEqual(verdict(s,g)['families']['a']['incumbent'],'p-a')
        s['tests'][4]['improvement_point_guard']=True
        self.assertEqual(verdict(s,g)['families']['a']['incumbent'],'ph-a')

    def test_no_eligible_new_arm_no_acceptance_and_report_only_mean(self):
        s,g=fixture()
        for i in (0,1,2,6,7,8):s['holm_reject'][i]=False
        v=verdict(s,g)
        self.assertIsNone(v['acceptance_candidate'])
        self.assertEqual(v['families']['a']['incumbent'],'base-a')
        s['holm_reject'][6]=True
        self.assertEqual(verdict(s,g)['acceptance_candidate'],'p-c')
        self.assertTrue(verdict(s,g)['families']['c']['arms']['p-c']['eligible'])

    def test_every_inherited_gate_and_guard_remains_decisive(self):
        s,g=fixture()
        for key in GATE_KEYS:
            x=copy.deepcopy(g);x['p-a'][key]['pass_']=False
            if key=='G-STEER':x['p-a'][key]['counts']=[134,133,134]
            self.assertFalse(verdict(s,x)['families']['a']['arms']['p-a']['eligible'],key)
        for i in range(4):
            x=copy.deepcopy(s);x['tests'][0]['dataset_guards'][i]['point_guard']=False
            self.assertFalse(verdict(x,g)['families']['a']['arms']['p-a']['eligible'])
        s['tests'][0]['w2']['point_guard']=False
        self.assertFalse(verdict(s,g)['families']['a']['arms']['p-a']['eligible'])

    def test_incomplete_inventory_and_paired_steering_refuse(self):
        s,g=fixture()
        for index in range(12):
            x=copy.deepcopy(s);x['tests'][index]['state']='INCOMPLETE'
            self.assertEqual(verdict(x,g)['state'],'INCOMPLETE')
        x=copy.deepcopy(g);x['base-a']['G-STEER']['counts']=[135,135,134]
        self.assertEqual(verdict(s,x)['state'],'INCOMPLETE') # Candidate's pass claim is false.
        del g['ph-c']['K5']
        self.assertEqual(verdict(s,g)['state'],'INCOMPLETE')


if __name__=='__main__':unittest.main()
