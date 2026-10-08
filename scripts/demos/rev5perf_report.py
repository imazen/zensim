#!/usr/bin/env python3
"""Compare frozen SPEEDQ feature bits and report its existing paired analyses."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import speedq_run as speedq


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parity(root):
    paths = list((root / 'parity').glob('PARITY*PASS.json'))
    assert len(paths) == 1, 'one successful full parity receipt required'
    return paths[0], speedq.parity_receipt(paths[0])


def compare_bits(before, after):
    bp, old = parity(before)
    ap, new = parity(after)
    checks = 0
    for key, rec in old.items():
        candidate = new[key]
        assert candidate['input_sha256'] == rec['input_sha256'], 'input identity changed'
        assert candidate['model'] == rec['model'], 'served model identity changed'
        assert candidate['score_bits'] == rec['score_bits'], f'STOP frozen score bits changed: {key}'
        left = b''.join(struct.pack('>d', v) for v in rec['feature_values'])
        right = b''.join(struct.pack('>d', v) for v in candidate['feature_values'])
        assert left == right, f'STOP frozen consumed feature bits changed: {key}'
        if key[-1] in (4, 5):
            checks += 1
    assert checks == 384
    return dict(status='PASS', strict_score_and_420_feature_checks=checks,
                all_revision_frozen_comparisons=len(old),
                before_receipt=str(bp), before_sha256=sha(bp),
                after_receipt=str(ap), after_sha256=sha(ap))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--before', type=Path, required=True)
    ap.add_argument('--after', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--parity-only', action='store_true')
    args = ap.parse_args()
    builds = {label: json.loads((root / 'provenance' / 'instrument.artifact.json').read_text())
              for label, root in [('before', args.before), ('after', args.after)]}
    assert builds['before']['dependencies'] == builds['after']['dependencies'], 'build dependency inventory changed'
    for label, root in [('before', args.before), ('after', args.after)]:
        assert sha(root / 'provenance' / 'instrument') == builds[label]['binary_sha256'], 'frozen executable changed'
    result = dict(parity=compare_bits(args.before, args.after), builds=builds, cells={})
    if not args.parity_only:
        before = {p.parent.name: p.parent for p in (args.before / 'timing').glob('*/COMPLETE.json')}
        after = {p.parent.name: p.parent for p in (args.after / 'timing').glob('*/COMPLETE.json')}
        assert before and set(before) == set(after), 'before/after timing coverage differs'
        for tag in sorted(before):
            row = {}
            for label, root in [('before', before[tag]), ('after', after[tag])]:
                complete = json.loads((root / 'COMPLETE.json').read_text())
                assert complete['status'] == 'PASS' and complete['rounds'] == 32
                assert complete['zenbench_gate_clean'] and complete['paired_alignment_verified']
                header = json.loads((root / 'header.json').read_text())
                assert header['binary_sha256'] == builds[label]['binary_sha256']
                assert header['quiet_gate']['admitted'] and header['quiet_gate']['load1'] < 2
                assert not header.get('gate_trace', False)
                assert header['model_source_sha256'] == speedq.SOURCE_SHA
                assert header['round_cap'] == 64 and header['round_rule'] == speedq.CLEAN_ROUND_RULE
                inner = json.loads((root / 'zenbench.inner.json').read_text())
                values, selection = speedq.select_clean_rounds(inner, 32)
                assert all(complete[k] == v for k, v in selection.items())
                assert set(values) == set(header['arms']) and all(len(v) == 32 for v in values.values())
                assert not inner['zenbench_unreliable']
                interference = json.loads((root / 'interference.json').read_text())
                assert interference['admitted'] and not interference['foreign']
                analysis = json.loads((root / 'paired_analysis.json').read_text())
                row[label] = dict(analysis=analysis, round_selection=complete,
                                  binary_sha256=header['binary_sha256'],
                                  model_source_sha256=header['model_source_sha256'])
            result['cells'][tag] = row
    speedq.write(args.out, result)
    print(json.dumps(result['parity'], indent=2))


if __name__ == '__main__':
    main()
