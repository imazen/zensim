"""Compact receipts from E34 smoke artifacts; no pixels or human labels read."""
import argparse
import csv
import json
from pathlib import Path
import sys

from v2_common import sha


def steering(data):
    rows = data['rows']
    assert len(rows) == 135
    return dict(cases=len(rows), passed=sum(r['pass'] for r in rows),
                failures=[dict(key=r['key'], block=r['blocks'][0]['bounds'][2],
                               m2=r['m2'], m3f=r['m3f']) for r in rows if not r['pass']],
                independent_sensitivities_exact=all(r['sensitivity_independent_probe_exact'] for r in rows),
                feature_comparisons=sum(r['engine_feature_comparisons'] for r in rows),
                feature_disagreements=sum(r['engine_feature_disagreements'] for r in rows))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['serving', 'p2'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    a = p.parse_args()
    files = {}

    def read(name):
        path = a.root / name
        files[name] = sha(path)
        return json.loads(path.read_text())

    if a.mode == 'serving':
        from e33_gates import _verdict_numbers, _pack
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from prodqual_label_free import summarize
        native = summarize(read('prodqual.json'), None)
        native['scope'] = 'one E34 L first-epoch smoke; label-free synthetic SDR only'
        identity_path = a.root / 'identity-s0.jsonl'
        files[identity_path.name] = sha(identity_path)
        identity = json.loads(identity_path.read_text().splitlines()[-1])
        path = a.root / 'served-identity.tsv'
        files[path.name] = sha(path)
        with path.open() as stream:
            rows = list(csv.reader(stream, delimiter='\t'))[1:]
        assert len(rows) == 62
        nearid = read('nearid-s0/candidate-l.GATES.json')
        verdicts = {}
        for grid in ['standard', 'ladder']:
            folder = a.root / 'verdict-s0-l' / grid
            read(str(folder.relative_to(a.root) / 'gaddr.json'))
            verdicts[grid] = _verdict_numbers(folder)
        packing = read('packing.json')
        model = Path(packing['packed_model'])
        assert sha(model) == packing['packed_model_sha256']
        result = dict(model_contract=read('model-contract.json'), identity=identity,
                      served_identity=dict(rows=len(rows), exact_100=all(float(r[2]) == 100 for r in rows)),
                      native_serving=native, nearid={k: nearid[k] for k in ['rows', 'N1', 'N2', 'N3']},
                      verdicts=verdicts, packing=_pack(a.root / 'packed'), model_sha256=sha(model),
                      binaries={name: sha(a.root / 'bin' / name) for name in
                                ['serve_custom_bake', 'bake_verdict', 'steer_instrument']},
                      steering=steering(read('steer-s0-l.json')),
                      unmeasured=['full K4 populations', 'ARM', 'WASM', 'runtime guard', 'worker image/jobset'])
    else:
        data = read('p2-l.json')
        rows = data['rows']
        result = steering(data)
        assert all(r['objective'] == 'PreFloor' and r['linear_probe'] for r in rows)
        result.update(model_sha256=rows[0]['model'],
                      max_abs_s_plus_c=max(r['linear_probe']['max_abs_s_plus_c'] for r in rows),
                      all_zero_sensitivity_cases=sum(all(s == 0 for s in r['linear_probe']['s']) for r in rows),
                      negative_input_cases=[dict(key=r['key'], block=r['blocks'][0]['bounds'][2],
                                                 rows=r['linear_probe']['negative_input_rows'])
                                            for r in rows if r['linear_probe']['negative_input_rows']])
        assert all(r['model'] == result['model_sha256'] for r in rows)
        with a.output.with_suffix('.csv').open('x', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['key', 'block', 'max_abs_s_plus_c', 'all_zero_s', 'negative_rows', 'm2', 'm3f', 'pass'])
            for r in rows:
                probe = r['linear_probe']
                writer.writerow([r['key'], r['blocks'][0]['bounds'][2], probe['max_abs_s_plus_c'],
                                 all(s == 0 for s in probe['s']), len(probe['negative_input_rows']),
                                 r['m2'], r['m3f'], r['pass']])
    result.update(scope='local first-epoch smoke, not a full fit or release qualification',
                  source_root=str(a.root), artifact_sha256=files)
    with a.output.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
