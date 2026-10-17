"""Compare prepared f32 consumed inputs with the retained native Rev5/v3 producer."""
import csv
import json
from pathlib import Path
import numpy as np
from e34_augmentation import sha, write


def legacy():
    from e34_p4_finish import ROOT
    
    root=ROOT
    job=json.loads((root/'repairs/job.json').read_text());ids=job['feature_ids']
    probes=list(map(json.loads,(root/'parity/probe-features.jsonl').read_text().splitlines()))
    probe_map={(r['base_id'],r['block_size'],*r['block']):r for r in probes}
    matched=set();base={}
    with (root/'repairs/features.jsonl').open() as f:
        for line in f:
            r=json.loads(line)
            key=(r['base_id'],r['block_size'],*r['block'])
            if key in probe_map:
                expected=probe_map[key]
                for field in ['base_features','repair_features']:
                    assert np.array_equal(np.asarray(r[field],dtype=np.float32).view(np.uint32),np.asarray(expected[field],dtype=np.float32).view(np.uint32)),(key,field)
                matched.add(key)
            if r['base_id'] not in base:base[r['base_id']]=np.asarray(r['base_features'],dtype=np.float32)[ids]
    assert matched==probe_map.keys()
    expected={f'base-{k}':v for k,v in base.items()}
    expected.update({f'repair-{i}':np.asarray(r['repair_features'],dtype=np.float32)[ids] for i,r in enumerate(probes)})
    seen=set();mismatches=[];comparisons=0
    with (root/'parity/native-paired.csv').open() as f:
        reader=csv.DictReader(f)
        for r in reader:
            key=r['ref_basename'] if 'ref_basename' in r else r['composite_key']
            assert key in expected and key not in seen,key
            seen.add(key)
            actual=np.asarray([float(r[f'f{i}']) for i in ids],dtype=np.float32)
            bad=np.flatnonzero(actual.view(np.uint32)!=expected[key].view(np.uint32))
            comparisons+=len(ids)
            if len(bad):mismatches.append(dict(key=key,columns=[ids[i] for i in bad],max_abs_error=float(np.max(np.abs(actual-expected[key])))))
    assert seen==expected.keys(),(len(seen),len(expected))
    receipt=dict(schema='e34-native-extractor-parity-v1',labels_read=False,formula_revision=5,tier='v3',bases=len(base),repair_probes=len(probes),columns=len(ids),f32_comparisons=comparisons,exact_f32=not mismatches,mismatches=mismatches,native_binary_sha256=sha('/var/tmp/rev5-extract/target/release/examples/extract_features_372col'),prepared_extractor_sha256=json.loads((root/'repairs/extraction-binding.json').read_text())['extractor_sha256'],probe_extractor_sha256=sha('target/debug/examples/e34_pixels'),native_csv_sha256=sha(root/'parity/native-paired.csv'),pairs_sha256=sha(root/'parity/pairs.tsv'),scope='all augmentation base rows; 16 sampled repairs (four codecs x four sizes), no claim of exhaustive repair parity')
    write(root/'parity/comparison.json',receipt)
    print(json.dumps({k:v for k,v in receipt.items() if k!='mismatches'}),flush=True)
    assert not mismatches,'native extractor disagreement'


def e36_prepare(old, out, spec_path):
    import pyarrow.csv as pc
    out.mkdir(exist_ok=False)
    evidence = json.loads((old/'comparison.json').read_text())
    assert sha(old/'pairs.tsv') == evidence['pairs_sha256']
    assert sha(old/'native-paired.csv') == evidence['native_csv_sha256']
    spec = json.loads(spec_path.read_text())
    ids = spec['feature_ids']
    # Project features and keys ONLY; the legacy transport's human_score and
    # target columns are neither loaded nor used by this label-free check.
    table = pc.read_csv(old/'native-paired.csv', convert_options=pc.ConvertOptions(
        include_columns=['ref_basename', *[f'f{i}' for i in ids]]))
    expected = {r['ref_basename']: [r[f'f{i}'] for i in ids] for r in table.to_pylist()}
    pairs = pc.read_csv(old/'pairs.tsv', parse_options=pc.ParseOptions(delimiter='\t'),
        convert_options=pc.ConvertOptions(include_columns=['key','ref_path','dist_path'])).to_pylist()
    by = {r['key']:r for r in pairs}
    keys = ['base-0','base-1','base-2','base-3','base-6','base-9','base-384',
            'base-768','base-1152','repair-0','repair-1','repair-15']
    rows = [dict(key=k,reference=by[k]['ref_path'],reference_sha256=sha(by[k]['ref_path']),
        distorted=by[k]['dist_path'],distorted_sha256=sha(by[k]['dist_path'])) for k in keys]
    write(out/'expected.json',dict(labels_read=False,ids=ids,rows={k:expected[k] for k in keys},
        native_parity_receipt_sha256=sha(old/'comparison.json'),
        feature_projection=['ref_basename', *[f'f{i}' for i in ids]]))
    write(out/'job.json',dict(mode='features',output=str(out/'actual.jsonl'),
        model=spec['model'],model_sha256=spec['model_sha256'],width=1853,feature_ids=ids,
        metric_ids=[],rows=rows))


def e36_verify(root):
    expected = json.loads((root/'expected.json').read_text())
    actual = list(map(json.loads,(root/'actual.jsonl').read_text().splitlines()))
    assert len(actual)==len(expected['rows'])==12
    assert {r['key'] for r in actual}==expected['rows'].keys()
    compared = 0
    for row in actual:
        a=np.asarray(row['features'],dtype=np.float32)[expected['ids']]
        b=np.asarray(expected['rows'][row['key']],dtype=np.float32)
        assert np.array_equal(a.view(np.uint32),b.view(np.uint32)),row['key']
        compared += len(a)
    receipt=dict(labels_read=False,fits=0,rows=len(actual),consumed_columns=len(expected['ids']),
        f32_comparisons=compared,exact_f32=True,
        scope='12 frozen E34 native parity pairs; consumed features only; no synthetic row entered fitting')
    write(root/'verified.json',receipt)
    print(json.dumps(receipt))


if __name__=='__main__':
    import sys
    if len(sys.argv)==1:
        legacy()
    elif sys.argv[1]=='e36-prepare' and len(sys.argv)==5:
        e36_prepare(Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]))
    elif sys.argv[1]=='e36-check' and len(sys.argv)==3:
        e36_verify(Path(sys.argv[2]))
    else:
        raise SystemExit('legacy: no args; or e36-prepare OLD OUT SPEC; or e36-check ROOT')
