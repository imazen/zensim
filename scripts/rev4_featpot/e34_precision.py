"""Bind and summarize the report-only E34 gate precision diagnostic."""
import argparse
import csv
import json
import math
from pathlib import Path
import statistics
import struct

from v2_common import sha
from e34_linear_probe import PACKET_SHA


def prepare(root):
    root.mkdir()
    prior = Path('/mnt/v/output/zensim/qual-a-2026-10-10')
    source = prior / 'MODELS.json'
    models = json.loads(source.read_text())
    models['l'] = dict(path='/mnt/v/output/zensim/e34-prep-2026-10-15/part3/l-gates/packed/refit/production-f16.bin',
                       sha256='8ed80a9934573271bdf078f097aa73f890127c73bbda46a8be1e55aff9f4e70d')
    pins = ['7cf4cfd92bb8824405db3ed27287bbec973a01eaeadf09b7ad037db92ab9a275',
            '89cb4861e2651ff8d4a98bb879feed83a0f9eb3f22b83e644e42eb2291d74b2e',
            'f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400',
            models['l']['sha256']]
    records = []
    for name, pin in zip(['a', 'c', 'seed0', 'l'], pins, strict=True):
        m = models[name]
        assert sha(Path(m['path'])) == m['sha256'] == pin
        oracle = prior / f'steer-s0-{name}.json' if name != 'l' else root.parent / 'part3/l-gates/steer-s0-l.json'
        records.append(dict(name=name, path=m['path'], sha256=pin, oracle=str(oracle), oracle_sha256=sha(oracle)))
    packet = Path('/mnt/v/output/zensim/steerfix-2026-10-09/full-packet.json')
    assert sha(packet) == PACKET_SHA
    for label, count in [('pilot', 1), ('full', 135)]:
        job = dict(packet=str(packet), packet_sha256=PACKET_SHA, models=records,
                   output_dir=str(root / label), case_limit=count)
        (root / f'{label}-job.json').write_text(json.dumps(job, indent=2)+'\n')


def summarize(root):
    complete = json.loads((root / 'full/complete.json').read_text())
    assert complete['cases'] == 135 and complete['model_cases'] == 540
    rows = [r for p in sorted((root / 'full').glob('case-*.json')) for r in json.loads(p.read_text())]
    assert len(rows) == 540
    methods = complete['methods']
    result = dict(scope='report-only gate-population diagnostic; no model choice or changed qualification',
                  methods=methods, source_root=str(root), models={})
    table = []
    for model in ['a', 'c', 'seed0', 'l']:
        panel = [r for r in rows if r['model'] == model]
        assert len(panel) == 135
        flips = []
        for r in panel:
            passed = [v >= .99 for v in r['m2']]
            if len(set(passed)) > 1:
                flips.append(dict(key=r['key'], block=r['block'], m2=r['m2'], m2_pass=passed,
                                  registered_m3f=r['registered_m3f']))
            table.append(dict(model=model, key=r['key'], block=r['block'],
                              **{f'm2_{method}': r['m2'][i] for i, method in enumerate(methods)},
                              **{f'm2_f64_gains_{method}': r['m2_f64_gains'][i] for i, method in enumerate(methods)},
                              registered_m3f=r['registered_m3f'], objective=r['sensitivity_objective']))
        errors = {}
        for field in ['gradient_errors', 'raw_gradient_errors']:
            errors[field] = {}
            for i, method in enumerate(methods[:2]):
                values = [v for r in panel for v in r[field][i]]
                errors[field][method] = dict(coordinates=len(values), maximum=max(values),
                                             median=statistics.median(values),
                                             median_case_maximum=statistics.median(max(r[field][i]) for r in panel))
        result['models'][model] = dict(sha256=panel[0]['model_sha256'], cases=135,
            m2_pass_counts=[sum(r['m2'][i] >= .99 for r in panel) for i in range(3)],
            with_registered_m3f_pass_counts=[sum(r['m2'][i] >= .99 and r['registered_m3f'] >= .70 for r in panel) for i in range(3)],
            m2_f64_gains_pass_counts=[sum(r['m2_f64_gains'][i] >= .99 for r in panel) for i in range(3)],
            raw_fallback_cases=[dict(key=r['key'], block=r['block']) for r in panel if r['sensitivity_objective'] == 'PreFloor'],
            hidden_exact_kinks=sum(r['hidden_exact_kinks'] for r in panel),
            max_score_f64_vs_f32_error=max(r['max_score_f64_vs_f32_error'] for r in panel),
            errors=errors, flips=flips)
    run = json.loads((root/'run-provenance.json').read_text())
    result['provenance'] = dict(full_job_sha256=sha(root/'full-job.json'), complete_sha256=sha(root/'full/complete.json'),
                                replay=complete['registered_replay'],
                                binary_sha256=run['binary_sha256'])
    # Explain the specific P2 discrepancy from measured sensitivities and the
    # retained native feature row. No new probe step or model choice is made.
    p2_path = root.parent/'part3/p2-l.json'
    assert sha(p2_path) == '5ef8003f241b01116904b5e2f080e03f4b6f38132f6f39d392a1da5f05e3d3ef'
    old = next(r for r in json.loads(p2_path.read_text())['rows']
               if r['key'] == 'I41_10_05' and r['blocks'][0]['bounds'][2] == 8)
    current = next(r for r in rows if r['model']=='l' and r['key']=='I41_10_05' and r['block']==8)
    slot = 632
    with Path(current['feature_file']).open('rb') as f:
        f.seek(slot*8); x = struct.unpack('<d', f.read(8))[0]
    h=max(abs(x)*1e-3, 1e-5)
    s=old['linear_probe']['s'][slot]; c=old['linear_probe']['c'][slot]
    assert abs(s+c) == current['raw_gradient_errors'][0][current['gradient_ids'].index(slot)]
    ulp=math.ldexp(1.0, math.frexp(abs(old['base_score']))[1]-24)
    result['rounding_example'] = dict(key=old['key'], block=8, slot=slot, feature=x, h=h,
        raw_base_score=old['base_score'], f32_ulp_at_base=ulp, s_fd=s, s_exact=-c,
        fd_numerator_implied=s*2*h, fd_numerator_in_ulps=s*2*h/ulp,
        algebraic_linear_numerator=-c*2*h, algebraic_numerator_in_ulps=-c*2*h/ulp,
        source_p2_sha256=sha(p2_path))
    with (root/'summary.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False);f.write('\n')
    with (root/'cases.csv').open('x', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    print(json.dumps(result, indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'summarize', 'progress'])
    p.add_argument('--root', required=True, type=Path)
    a=p.parse_args()
    if a.mode=='prepare': prepare(a.root)
    elif a.mode=='summarize': summarize(a.root)
    else: print((a.root/'full/progress.json').read_text())


if __name__=='__main__':
    main()
