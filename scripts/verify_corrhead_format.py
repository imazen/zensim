#!/usr/bin/env python3
"""Verify ZCTH export with synthetic rows and the canonical Rust parity tool.

No scientific dataset access. An explicit pre-change exporter is required to
prove byte identity for legacy v1/v2. New revisions test serialization, not
feature extraction or model quality.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-exporter', type=Path, required=True)
    parser.add_argument('--parity-bin', type=Path, required=True)
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    exporter = Path(__file__).parent / 'v_next/train_corruption_head.py'
    new = load(exporter, 'format_current')
    old = load(args.baseline_exporter, 'format_baseline')
    # The exporter pins numerical-library threads before these imports.
    import numpy as np
    from sklearn.isotonic import IsotonicRegression
    from sklearn.ensemble import HistGradientBoostingClassifier

    x = np.random.default_rng(914).uniform(-2, 2, (512, 3)).astype(np.float32).astype(np.float64)
    y = ((x[:, 0] > 0.2) ^ (x[:, 1] + x[:, 2] > 0.1)).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=16, max_leaf_nodes=7,
                                        early_stopping=False, random_state=914).fit(x, y)
    iso = IsotonicRegression(out_of_bounds='clip').fit(clf.predict_proba(x)[:, 1], y)
    parity = args.out_dir / 'synthetic.npz'
    np.savez_compressed(parity, train_X=x, train_raw=clf.decision_function(x),
                        train_p=iso.predict(clf.predict_proba(x)[:, 1]))

    def emit(module, name, **kwargs):
        target = args.out_dir / (name + '.zcth')
        module.emit_zcth(str(target), 372, [13, 91, 146], np.zeros(3), np.ones(3),
                         8., clf, iso, .9, {'synthetic_format_fixture': True}, **kwargs)
        return target

    cases = []
    for precision in ['native', 'f32']:
        before = emit(old, 'old-' + precision, input_precision=precision)
        after = emit(new, 'current-' + precision, input_precision=precision)
        assert before.read_bytes() == after.read_bytes(), 'legacy exporter changed bytes'
        cases.append(after)
    for revision in [1, 2, 3]:
        cases.append(emit(new, f'revision-{revision}', input_precision='f32', formula_revision=revision))
    for revision, precision in [(0, 'f32'), (4, 'f32'), (3, 'native')]:
        try:
            emit(new, 'refused', input_precision=precision, formula_revision=revision)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid revision/precision accepted')
        assert not (args.out_dir / 'refused.zcth').exists()

    receipts = []
    for case in cases:
        command = [str(args.parity_bin.resolve()), '--head', str(case.resolve()),
                   '--parity', str(parity.resolve()), '--set', 'train']
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        log = args.out_dir / (case.stem + '.log')
        log.write_text(result.stdout + result.stderr)
        assert 'corrhead_parity: PASS' in result.stdout
        receipts.append(dict(file=case.name, sha256=sha(case), command=command,
                             parity_log_sha256=sha(log)))
    report = dict(role='synthetic-format-check', scientific_fits=0, synthetic_fits=1,
                  rows_per_case=512, cases=receipts, legacy_v1_v2_bytes_identical=True,
                  invalid_contracts_refused=3, exporter_sha256=sha(exporter),
                  baseline_exporter_sha256=sha(args.baseline_exporter),
                  parity_binary_sha256=sha(args.parity_bin), script_sha256=sha(__file__))
    (args.out_dir / 'RESULT.json').write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: five Rust parity cases; legacy bytes unchanged; three invalid contracts refused')


if __name__ == '__main__':
    main()
