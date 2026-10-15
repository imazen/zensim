"""E34 revision-5 decision rule; consumes Rust statistics and measured gates.

No p-values, confidence intervals, Holm decisions or models are fitted here.
The E33 point form remains report-only as specified in registration 4.2.
"""

ARMS = {'a': ['w-a', 'k-a', 'l-a'], 'c': ['w-c', 'k-c']}
ELIGIBILITY = {'w-a': 0, 'k-a': 1, 'l-a': 2, 'w-c': 3, 'k-c': 4}
IMPROVEMENT = {('k-a', 'w-a'): 5, ('l-a', 'w-a'): 6,
               ('l-a', 'k-a'): 7, ('k-c', 'w-c'): 8}
GATE_KEYS = tuple([f'C{i}' for i in range(1,7)] + ['G-DIAL']
                  + [f'N{i}' for i in range(1,4)] + [f'K{i}' for i in range(1,6)]
                  + ['G-STEER', 'runtime', 'spatial', 'memory'])


def steer_rule(candidate, control):
    assert len(candidate) == len(control) == 3
    assert all(type(n) is int and 0 <= n <= 135 for n in candidate + control)
    required = [135, max(control[1], 133), max(control[2], 133)]
    return dict(counts=candidate, paired_control=control, required=required,
                passed=[a >= b for a, b in zip(candidate, required, strict=True)],
                pass_=all(a >= b for a, b in zip(candidate, required, strict=True)))


def verdict(statistics, gates):
    assert statistics['eligibility_margin'] == .0065
    assert statistics['resamples'] == 10000 and statistics['level'] == .95
    assert statistics['percentile'] == 'Hyndman-Fan type 7'
    tests = statistics['tests']
    reject = statistics['holm_reject']
    assert len(tests) == len(reject) == 9
    assert [t['test'] for t in tests] == list(range(1, 10))
    if any(t['state'] != 'COMPLETE' for t in tests):
        return dict(state='INCOMPLETE', reason='a registered hypothesis is incomplete; no substitute verdict')
    families = {}
    for family, names in ARMS.items():
        incumbent = None
        arms = {}
        steps = []
        for arm in names:
            i = ELIGIBILITY[arm]
            t = tests[i]
            assert t['arm'] == arm and t['control'] == f'base-{family}'
            if set(gates.get(arm, {})) != set(GATE_KEYS):
                return dict(state='INCOMPLETE', reason=f'{arm}: every registered gate is required')
            if [d['dataset'] for d in t['dataset_guards']] != ['kadid','tid2013','konfig','cid22_a25']:
                return dict(state='INCOMPLETE', reason=f'{arm}: four dataset guards are required')
            failed = []
            if not reject[i]:
                failed.append('eligibility Holm')
            failed.extend(f'dataset {d["dataset"]}' for d in t['dataset_guards'] if not d['point_guard'])
            if not t['w2']['point_guard']:
                failed.append('two-way W2 point guard')
            for name, entry in gates[arm].items():
                if not entry['pass_']:
                    failed.append(name)
            eligible = not failed
            arms[arm] = dict(eligible=eligible, failed=failed,
                             e33_mean_point_guard_report_only=t['e33_mean_point_guard'])
            previous = incumbent
            rule = 'not eligible'
            if eligible and incumbent is None:
                incumbent, rule = arm, 'first eligible in W, K, L order'
            elif eligible:
                j = IMPROVEMENT[arm, incumbent]
                test = tests[j]
                assert test['arm'] == arm and test['control'] == incumbent
                if reject[j] and test['improvement_point_guard']:
                    incumbent, rule = arm, 'Holm improvement and point delta > +0.002'
                else:
                    rule = 'retain incumbent: improvement Holm or practical point bar fails'
            steps.append(dict(arm=arm, previous=previous, incumbent=incumbent, rule=rule))
        families[family] = dict(arms=arms, incumbent=incumbent or f'base-{family}', steps=steps,
                                meaning='enter full qualification only; production adoption is separate')
    return dict(state='COMPLETE', families=families, production_changed=False)
