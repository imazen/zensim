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


def d12_prodqual(root,stage):
    import os
    import subprocess
    models=json.loads((root.parent/'part4/full-job.json').read_text())['models']
    binary=root/'serve-before' if stage=='before' else Path(__file__).resolve().parents[2]/'target/debug/examples/serve_custom_bake'
    paths=[m['path'] for m in models]
    env=dict(os.environ,ZENSIM_FORMULA_REV='5',ZENSIM_NEIGHBOUR_EXACT='1',RAYON_NUM_THREADS='1',OMP_NUM_THREADS='1')
    env.pop('ZENSIM_MAX_TIER',None)
    with (root/f'{stage}-prodqual.json').open('x') as out,(root/f'{stage}-prodqual.log').open('x') as log:
        subprocess.run([str(binary),'--prodqual',*paths],env=env,stdout=out,stderr=log,check=True)
    with (root/f'{stage}-prodqual-provenance.json').open('x') as f:
        json.dump(dict(binary=str(binary),binary_sha256=sha(binary),models=models),f,indent=2)


def d12_summarize(root):
    models=json.loads((root.parent/'part4/full-job.json').read_text())['models']
    result=dict(schema='e34-d12-exact-gradient-v1',method='exact-gradient-d12',bars=dict(m2=.99,m3f=.70,cases=135),models={})
    for model in models:
        path=root/f'cached-gate-{model["name"]}/result.json'
        current=json.loads(path.read_text())['rows']
        assert sha(Path(model['oracle']))==model['oracle_sha256']
        old=json.loads(Path(model['oracle']).read_text())['rows']
        job=json.loads((path.parent/'job.json').read_text())
        assert len(current)==len(old)==len(job['cases'])==135
        flips=[];score_comparisons=0
        for i,(a,b,c) in enumerate(zip(old,current,job['cases'],strict=True)):
            assert a['key']==b['key']==c['key']
            assert b['model']==model['sha256']
            assert b['sensitivity_method']=='exact-gradient-d12' and b['sensitivity_shared_owner_exact']
            assert a['served_base_bits']==b['served_base_bits'],(model['name'],i,'base')
            assert a['served_repair_bits']==b['served_repair_bits'],(model['name'],i,'repairs')
            score_comparisons+=1+len(b['served_repair_bits'])
            assert b['pass']==(b['m2']>=.99 and b['m3f']>=.70)
            assert b['engine_feature_disagreements']==0
            if a['pass']!=b['pass']:flips.append(dict(key=b['key'],block=c['block'],before=a['pass'],after=b['pass'],m2_before=a['m2'],m2_after=b['m2'],m3f_before=a['m3f'],m3f_after=b['m3f']))
        result['models'][model['name']]=dict(model_sha256=model['sha256'],cases=135,before_pass=sum(r['pass'] for r in old),after_pass=sum(r['pass'] for r in current),m2_pass=sum(r['m2']>=.99 for r in current),m3f_pass=sum(r['m3f']>=.70 for r in current),score_bits_compared=score_comparisons,scores_bit_identical=True,flips=flips,result_sha256=sha(path),packet_sha256=sha(path.parent/'job.json'))
    native=json.loads((root/'gate-a/result.json.partial').read_text())
    cached=json.loads((root/'cached-gate-a/result.json').read_text())['rows']
    assert len(native)==107
    for i,(a,b) in enumerate(zip(native,cached)):
        assert all(a[k]==b[k] for k in a), ('native repair cache disagreement', i, [k for k in a if a[k]!=b[k]])
    result['repair_cache_validation']=dict(full_pixel_exact_rows=len(native),total_rows=135,
        every_output_field_equal=True,cache_sha256=sha(root/'native-repair-cache.json'),
        native_partial_sha256=sha(root/'gate-a/result.json.partial'))
    before=[json.loads(x) for x in (root/'before-timing.jsonl').read_text().splitlines()]
    after=[json.loads(x) for x in (root/'after-timing.jsonl').read_text().splitlines()]
    assert len(before)==len(after)==16
    timings=[]
    for a,b in zip(before,after,strict=True):
        assert (a['model'],a['side'],a['model_sha256'],a['bin'],a['tier'])==(b['model'],b['side'],b['model_sha256'],b['bin'],b['tier'])
        for x,y in zip(a['samples'],b['samples'],strict=True):
            assert x['score_bits']==y['score_bits'] and x['features_sha256']==y['features_sha256']
        timings.append(dict(model=a['model'],side=a['side'],before_median_seconds=statistics.median(x['seconds'] for x in a['samples']),after_median_seconds=statistics.median(x['seconds'] for x in b['samples']),before_prepare_seconds=a['prepare_seconds'],after_prepare_seconds=b['prepare_seconds'],repeats=3))
    result['timings']=timings
    a=json.loads((root/'before-prodqual.json').read_text());b=json.loads((root/'after-prodqual.json').read_text())
    comparisons=0;tier_pairs=0
    assert len(a['seeds'])==len(b['seeds'])==4
    for x,y in zip(a['seeds'],b['seeds'],strict=True):
        assert x['model']==y['model'] and len(x['permutations'])==len(y['permutations'])
        for p,q in zip(x['permutations'],y['permutations'],strict=True):
            assert p['label']==q['label'] and len(p['rows'])==len(q['rows'])
            for r,s in zip(p['rows'],q['rows'],strict=True):
                for field in ['pixel_bits','cached_bits','feature_bits','identity_aware_bits','read_bits','feature_mismatches','cached_feature_mismatches']:
                    assert r[field]==s[field],(x['model'],p['label'],field)
                assert s['finite'] and not s['feature_mismatches'] and not s['cached_feature_mismatches']
                comparisons+=1
        # Features have the existing tier parity promise; map gradients must
        # agree bit for bit on the same extracted rows at all permutations.
        base=y['permutations'][0]['rows']
        for permutation in y['permutations'][1:]:
            for r,s in zip(base,permutation['rows'],strict=True):
                assert r['read_bits']==s['read_bits']
                assert r['sensitivity_bits']==s['sensitivity_bits']
                tier_pairs+=1
    result['synthetic_score_parity']=dict(model_tier_geometry_ladder_rows=comparisons,gradient_tier_row_pairs=tier_pairs,score_feature_bits_unchanged=True,gradient_bits_tier_equal=True)
    result['gradient_reference']=json.loads((root/'gradient-reference.json').read_text())
    import re
    logs=Path('/home/lilith/tmp/e34prep')
    result['resource_receipts']={}
    for path in sorted(logs.glob('p6-*.log')):
        receipts=re.findall(r'run-heavy: done[^\n]+',path.read_text(errors='replace'))
        if receipts:result['resource_receipts'][path.name]=receipts

    with (root/'summary.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['timings','gradient_reference']},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'summarize', 'progress','d12-summarize','d12-prodqual-before','d12-prodqual-after'])
    p.add_argument('--root', required=True, type=Path)
    a=p.parse_args()
    if a.mode=='d12-summarize': d12_summarize(a.root)
    elif a.mode.startswith('d12-prodqual-'):d12_prodqual(a.root,a.mode.removeprefix('d12-prodqual-'))
    elif a.mode=='prepare': prepare(a.root)
    elif a.mode=='summarize': summarize(a.root)
    else: print((a.root/'full/progress.json').read_text())


if __name__=='__main__':
    main()
