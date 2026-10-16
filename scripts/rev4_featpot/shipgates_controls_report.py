"""Summarize verified native TRAIN interventions; no release-gate inference."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--markdown', type=Path, required=True)
    args = parser.parse_args()
    results, fingerprints, panels = {}, {}, {}
    for label in ['w-c','diff-frag-820','diff-410','butteraugli']:
        root = args.evidence/('jxl-interventions' if label == 'w-c' else 'jxl-interventions-complete')/label
        data = json.loads((root/'analysis_summary.json').read_text())
        all_oracles = json.loads((root/'ALL_OUTPUTS_LIBJXL.json').read_text())
        map_oracle = json.loads((root/'fixed-judges/INDEPENDENT_MAP_PIXEL_ORACLE.json').read_text())
        assert map_oracle['all_input_pixel_hashes_verified'] and map_oracle['all_map_dedup_keys_verified']
        rows = [json.loads(line) for line in (root/'measurements.jsonl').read_text().splitlines()]
        assert len(rows) == data['judge_pairs_per_metric'] == len(all_oracles['outputs']) == 272
        assert all(row['status'] == 'success' for row in all_oracles['outputs'])
        assert data['verified_neutral_cells'] == len(data['cells']) == 8
        index = {(r['origin'],r['distance'],r['work']['name']):
            (r['work']['encoded_sha256'],r['work']['actual_q_sha256']) for r in rows}
        assert len(index) == len(rows)
        fingerprints[label] = index
        for path in (root/'artifacts/maps').glob('*.json'):
            assert sha(path) == path.stem
        results[label] = dict(neutral_exact=8,judged_encodes=len(rows),independent_map_input_bindings_verified=True,interventions=len(data['interventions']),
            actual_quantizer_inert=sum(c['actual_quantizer_inert'] for c in data['cells']),
            pixel_inert=sum(c['pixel_inert'] for c in data['cells']),
            quantizer_changes_outside_region=sum(c['quantizer_changes_outside_region'] for c in data['cells']),
            pixel_changes_outside_region=sum(c['pixel_changes_outside_region'] for c in data['cells']),
            direction={metric:{side:sum(c['direction'][metric][side] for c in data['cells'])
                for side in ['positive','negative','flat']} for metric in data['cells'][0]['direction']},
            representative_decoder_max_abs_rgb8_difference=max(c['primary_max_abs_rgb8_difference'] for c in data['compatibility']['decodes']),
            analysis_sha256=sha(root/'analysis_summary.json'))
        panels[label] = {(c['origin'],c['distance']):c for c in data['cells']}
    assert all(value == fingerprints['w-c'] for value in fingerprints.values())
    result = dict(scope='four TRAIN families, two distances, native mechanism controls only; no matched-RD or model qualification',
        cross_provider_control_bytes_and_quantizers_identical=True,providers=results)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
        stream.write('\n')
    lines = ['## Native JXL intervention controls','',
        'All four providers complete the same four TRAIN families (photo, document, graphic and screen), distances1/3 and16 coarse regions with0.8/1.2 raw-quantizer interventions. Each provider has272 full encodes,256 interventions and eight exact neutral repeats. Every retained output passes libjxl0.12 decoding and all three fixed judges. The independent map oracle verifies float-map bits, hashes and input-pixel bindings.','',
        'All256 interventions change quantizers and pixels; none changes quantizers outside its selected region. Pixels also change outside every selected region; the responsible stage remains unresolved. Control bytes and quantizer fields match across providers; map predictions and candidate scores differ.','',
        'Representative primary/libjxl U8 comparisons differ by at most one channel level. Pixel parity remains an open finding; decoding success does not establish bit identity.','',
        'The table shows Spearman association between map density and central fixed-B quality improvement per added byte, using only positive byte derivatives. These are descriptive TRAIN mechanism measurements, with no acceptance bar or release qualification.','',
        '| TRAIN origin | Class | Distance | W-C | C | A | Butteraugli map peer |',
        '|---|---|---:|---:|---:|---:|---:|']
    for key in sorted(panels['w-c']):
        cell=panels['w-c'][key]
        values=[panels[label][key]['rank_associations']['map_density_vs_fixed_zensim_quality_delta_per_byte']
                for label in results]
        formatted=['unresolved' if v is None else f'{v:.6f}' for v in values]
        lines.append(f'| {key[0]} | {cell["class"]} | {key[1]:g} | '+' | '.join(formatted)+' |')
    lines += ['',f'Full control summary SHA256 `{sha(args.output)}`; payloads and all per-intervention records remain in the evidence archive.']
    with args.markdown.open('x') as stream:
        stream.write('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


if __name__ == '__main__':
    main()
