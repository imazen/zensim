"""Fixed factorial decision adapter; consumes numerical and measured owners.

No fitting, selection by descriptive outcomes, label access or production change.
"""
from e34_verdict import GATE_KEYS, steer_rule

PAIRS = tuple((a+'-'+f, c+'-'+f, eligibility) for f in ('a','c') for a,c,eligibility in (
    ('p','base',True),('h','base',True),('ph','base',True),
    ('h','p',False),('ph','p',False),('ph','h',False)))
MODELS = {arm+'-'+family for family in ('a','c') for arm in ('base','p','h','ph')}


def verdict(statistics, gates):
    """Retain every incomplete slot and every inherited engineering gate."""
    incomplete = dict(state='INCOMPLETE', production_changed=False,
                      reason='complete registered statistics and measured gates required')
    if (statistics.get('schema') != 'registered-factorial-statistics-v1'
            or statistics.get('state') != 'COMPLETE' or statistics.get('decision_allowed') is not True
            or statistics.get('base_seed') != 20261016 or statistics.get('resamples') != 10000
            or statistics.get('level') != .95 or statistics.get('percentile') != 'Hyndman-Fan type 7'):
        return incomplete
    tests, reject = statistics.get('tests', []), statistics.get('holm_reject', [])
    if (len(tests) != 12 or len(reject) != 12 or any(type(x) is not bool for x in reject)
            or set(gates) != MODELS):
        return incomplete
    for i,(test,(arm,control,eligibility)) in enumerate(zip(tests,PAIRS,strict=True),1):
        if ((test.get('test'),test.get('arm'),test.get('control'),test.get('margin'))
                != (i,arm,control,.0065 if eligibility else 0.) or test.get('state') != 'COMPLETE'):
            return incomplete
        if eligibility:
            if ([d.get('dataset') for d in test.get('dataset_guards',[])]
                    != ['kadid','tid2013','konfig','cid22_a25']
                    or any(type(d.get('point_guard')) is not bool for d in test['dataset_guards'])
                    or type(test.get('w2',{}).get('point_guard')) is not bool
                    or type(test.get('e33_mean_point_guard')) is not bool):
                return incomplete
        elif type(test.get('improvement_point_guard')) is not bool:
            return incomplete
    for measured in gates.values():
        if (set(measured) != set(GATE_KEYS)
                or any(type(entry.get('pass_')) is not bool for entry in measured.values())):
            return incomplete
    families = {}
    for family,offset in (('a',0),('c',6)):
        base='base-'+family
        try:
            control_counts=gates[base]['G-STEER']['counts']
            steer_rule(control_counts,control_counts)  # Validate actual seed-0/1/2 counts.
        except (KeyError,AssertionError,TypeError):
            return incomplete
        incumbent=None; arms={}; steps=[]
        for local,kind in enumerate(('p','h','ph')):
            arm=kind+'-'+family; t=tests[offset+local]
            try:
                steering=steer_rule(gates[arm]['G-STEER']['counts'],control_counts)
            except (KeyError,AssertionError,TypeError):
                return incomplete
            if steering['pass_'] != gates[arm]['G-STEER']['pass_']:
                return incomplete
            failed=[]
            if not reject[offset+local]:failed.append('eligibility Holm')
            failed.extend('dataset '+d['dataset'] for d in t['dataset_guards'] if not d['point_guard'])
            if not t['w2']['point_guard']:failed.append('two-way W2 point guard')
            failed.extend(name for name,entry in gates[arm].items() if not entry['pass_'])
            eligible=not failed
            arms[arm]=dict(eligible=eligible,failed=failed,
                e33_mean_point_guard_report_only=t['e33_mean_point_guard'])
            previous=incumbent; rule='not eligible'
            if eligible and incumbent is None:
                incumbent=arm; rule='first eligible in P, H, PH order'
            elif eligible:
                j=next(i for i,pair in enumerate(PAIRS) if pair[:2]==(arm,incumbent))
                if reject[j] and tests[j]['improvement_point_guard']:
                    incumbent=arm; rule='Holm improvement and point delta > +0.002'
                else:rule='retain incumbent: improvement Holm or practical point bar fails'
            steps.append(dict(arm=arm,previous=previous,incumbent=incumbent,rule=rule))
        families[family]=dict(arms=arms,incumbent=incumbent or base,steps=steps)
    candidate=next((families[f]['incumbent'] for f in ('a','c')
                    if families[f]['incumbent'] != 'base-'+f),None)
    return dict(state='COMPLETE',families=families,production_changed=False,
                acceptance_candidate=candidate,acceptance_seed=0 if candidate else None,
                acceptance_read_authorized=False)
