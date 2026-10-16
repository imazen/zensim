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


def verdict(statistics, gates, *, mean_guard_decisive=False):
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
            if mean_guard_decisive and not t['e33_mean_point_guard']:
                failed.append('NON-decisive sensitivity: mean delta >= -0.002')
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


def mean_guard_sensitivity(statistics, gates):
    """Counterfactual only; leave the registered nine tests and verdict intact."""
    registered = verdict(statistics, gates)
    counterfactual = verdict(statistics, gates, mean_guard_decisive=True)
    assert registered['state'] == counterfactual['state'] == 'COMPLETE'
    changes = []
    section42_changes = []
    mean_failing = []
    families = {}
    for family, names in ARMS.items():
        before = registered['families'][family]
        after = counterfactual['families'][family]
        changes.extend(arm for arm in names
                       if before['arms'][arm]['eligible'] != after['arms'][arm]['eligible'])
        for arm in names:
            i = ELIGIBILITY[arm]
            test = statistics['tests'][i]
            if not test['e33_mean_point_guard']:
                mean_failing.append(arm)
                if (statistics['holm_reject'][i] and test['w2']['point_guard']
                        and all(g['point_guard'] for g in test['dataset_guards'])):
                    section42_changes.append(arm)
        families[family] = dict(registered=before['incumbent'],
                                counterfactual=after['incumbent'],
                                changed=before['incumbent'] != after['incumbent'])
    return dict(label='NON-decisive sensitivity', rule='mean delta >= -0.002 decisive',
                mean_failing_arms=mean_failing,
                changed_section42_eligibility=section42_changes,
                changed_arm_eligibility=changes, families=families,
                counterfactual_verdict=counterfactual)


# STEER is a versioned experiment; the E34 nine-test entry above is frozen.
STEER_PAIRS = [('s-a', 'base-a'), ('r-a', 'base-a'), ('sr-a', 'base-a'),
               ('s-c', 'base-c'), ('r-c', 'base-c'), ('sr-c', 'base-c'),
               ('r-a', 's-a'), ('sr-a', 's-a'), ('sr-a', 'r-a'),
               ('r-c', 's-c'), ('sr-c', 's-c'), ('sr-c', 'r-c')]
STEER_IDENTICAL = {2, 5, 8, 11}


def steer_verdict(statistics, gates):
    """Q12-policy-pinned S-only qualification; R/SR never become fallbacks."""
    incomplete = dict(state='INCOMPLETE', reason='required STEER evidence invalid; no substitute verdict',
                      production_changed=False)
    if statistics.get('state') == 'PENDING_OWNER_Q12':
        return dict(state='PENDING_OWNER_Q12', production_changed=False)
    if statistics.get('schema') != 'steer-statistics-v1':
        return incomplete
    if statistics.get('alias_policy') == 'e34-strict-b-v1':
        return dict(state='INCOMPLETE', scope='DIAGNOSTIC_ONLY',
                    reason='E34 strict completeness: known identical comparisons are incomplete',
                    production_changed=False)
    if (statistics.get('alias_policy') != 'q12-a-v1'
            or statistics.get('state') != 'COMPLETE_FOR_S'
            or statistics.get('eligibility_margin') != .0065
            or statistics.get('resamples') != 10000 or statistics.get('level') != .95
            or statistics.get('percentile') != 'Hyndman-Fan type 7'):
        return incomplete
    tests, reject = statistics.get('tests', []), statistics.get('holm_reject', [])
    if len(tests) != 12 or len(reject) != 12 or any(type(x) is not bool for x in reject):
        return incomplete
    for i, (test, pair) in enumerate(zip(tests, STEER_PAIRS, strict=True), 1):
        if (test.get('test'), test.get('arm'), test.get('control')) != (i, *pair):
            return incomplete
        expected = 'IDENTICAL_BY_CONSTRUCTION' if i in STEER_IDENTICAL else 'COMPLETE'
        if test.get('state') != expected:
            return incomplete
        if i in STEER_IDENTICAL and (test.get('p', {}).get('decision') != 1 or reject[i-1]):
            return incomplete
    expected_models = {'base-a', 'base-c', 's-a', 's-c', 'r-a', 'r-c', 'sr-a', 'sr-c'}
    if set(gates) != expected_models:
        return incomplete
    for measured in gates.values():
        if set(measured) != set(GATE_KEYS):
            return incomplete
        if any(type(entry.get('pass_')) is not bool for entry in measured.values()):
            return incomplete
    for i in range(12):
        t = tests[i]
        if [d.get('dataset') for d in t.get('dataset_guards', [])] != ['kadid', 'tid2013', 'konfig', 'cid22_a25']:
            return incomplete
        if any(type(d.get('point_guard')) is not bool for d in t['dataset_guards']) or type(t.get('w2', {}).get('point_guard')) is not bool:
            return incomplete
        from math import isfinite
        intervals = [d.get('interval', {}) for d in t['dataset_guards']] + [t.get('w2', {}).get('interval', {})]
        if any(type(iv.get(k)) not in (int, float) or not isfinite(iv[k]) for iv in intervals for k in ('estimate','se')):
            return incomplete
    # The strict R W2 false result is expected, never waived by alias admission.
    if any(tests[i]['w2']['point_guard'] for i in (1, 4)):
        return incomplete
    families = {}
    for family, i in [('a', 0), ('c', 3)]:
        arm, t = f's-{family}', tests[i]
        failed = []
        if not reject[i]:
            failed.append('eligibility Holm')
        failed.extend(f'dataset {d["dataset"]}' for d in t['dataset_guards'] if not d['point_guard'])
        if not t['w2']['point_guard']:
            failed.append('two-way W2 point guard')
        failed.extend(name for name, entry in gates[arm].items() if not entry['pass_'])
        families[family] = dict(eligible=not failed, failed=failed,
                                incumbent=arm if not failed else f'base-{family}',
                                meaning='enter full qualification only')
    return dict(state='COMPLETE_FOR_S', families=families,
                diagnostic_only=['r-a', 'sr-a', 'r-c', 'sr-c'], production_changed=False)
