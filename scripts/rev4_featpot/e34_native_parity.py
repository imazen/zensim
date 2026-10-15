"""Compare prepared f32 consumed inputs with the retained native Rev5/v3 producer."""
import csv
import json
from pathlib import Path
import numpy as np
from e34_augmentation import sha, write
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
