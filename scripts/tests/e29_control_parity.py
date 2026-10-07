"""Fit the registered E30 control recipe with E29 extensions, compare final119.

Uses the strict LODO owner and canonical strip/inspect owners. Only volatile
reproduction metadata is excluded; every other model byte must match E30.
This validation cell never selects/replaces the coordinator's shared control.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rev4_featpot'))
from v2_common import sha


def stable_repro(repro):
    """Normalize only transport/build location, clock and machine identifiers."""
    obj = copy.deepcopy(repro)
    for key in ('timestamp_epoch', 'cwd', 'hostname', 'trainer_source_dir', 'trainer_head_at_train'):
        obj[key] = '<volatile>'
    argv = obj['argv']
    argv[0] = '<trainer>'
    for i, token in enumerate(argv[:-1]):
        if token == '--group':
            name, path, tw, vw, mode = argv[i+1].split(':')
            argv[i+1] = f'{name}:<input>:{tw}:{vw}:{mode}'
        elif token in ('--out', '--keep-features', '--dump-checkpoints-dir'):
            argv[i+1] = '<output>'
    for item in obj['inputs']:
        item['path'] = '<input>'
    for table in obj['table_admission']['tables']:
        path = table['path']
        table['source'] = table['source'].replace(path, '<input>')
        table['path'] = '<input>'
    return obj


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--e30', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True)
    p.add_argument('--fold', choices=('kadid', 'tid2013', 'konfig', 'cid22_a25'), default='kadid')
    p.add_argument('--verify-only', action='store_true', help='audit an existing full validation cell without fitting')
    a = p.parse_args()
    if not a.verify_only:
        a.dest.mkdir(exist_ok=False)
    pinned = json.loads((a.bundle / 'E30_COMPLETE_PINS.json').read_text())
    producer = json.loads((a.bundle / 'PINNED_ARTIFACTS.json').read_text())
    assert sha(a.bundle / 'bin/zensim_mlp_train') == producer['files']['bin/zensim_mlp_train']['sha256']
    jobs = json.loads((a.e30 / 'fit-manifest-fitv2e30-20261007.json').read_text())
    matches = [j for j in jobs if j['kind']['argv'][j['kind']['argv'].index('--heldout')+1] == a.fold and j['kind']['argv'][j['kind']['argv'].index('--seed-index')+1] == '0']
    assert len(matches) == 1
    job = matches[0]
    argv = job['kind']['argv'][:]
    assert argv[argv.index('--heldout')+1] == a.fold
    assert argv[argv.index('--seed-index')+1] == '0'
    oldcell = Path(argv[argv.index('--dest')+1])
    old = oldcell / 'refit/last.bin'
    assert sha(old) == pinned['cells'][f'{a.fold}_s0']['bake_sha256']
    argv[argv.index('--root')+1] = str(a.bundle / 'v2e29')
    argv[argv.index('--data-role-decision')+1] = str(a.bundle / 'v2e29/human_role_decision.json')
    argv[argv.index('--dest')+1] = str(a.dest / 'cell')
    env = dict(os.environ, REV4_V2_BIN_DIR=str(a.bundle / 'bin'),
               ZENSIM_MAX_TIER='v3', RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1')
    cmd = [sys.executable, str(REPO / 'scripts/rev4_featpot/v2_lodo_mlp.py'), *argv[1:]]
    if not a.verify_only:
        with (a.dest / 'driver.log').open('w') as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    newcell = a.dest / 'cell'
    result = json.loads((newcell / 'result.json').read_text())
    original = json.loads((oldcell / 'result.json').read_text())
    assert result['epochs'] == 120 and result['pairs_per_epoch'] == 50000
    assert result['selection']['selected_epoch'] == 119
    assert result['selection']['strict_table_admission'] == original['selection']['strict_table_admission']
    for key in ('init_seed', 'sample_seed', 'train_weights', 'coverage_leg', 'dev_curve'):
        assert result[key] == original[key], key
    repros = []
    checkout_heads = []
    for model in (old, newcell / 'refit/last.bin'):
        inspected = json.loads(subprocess.check_output([str(a.bundle / 'bin/inspect_qualified_checkpoint'), str(model)], text=True))
        assert inspected['checkpoint_epoch'] == '119' and inspected['admitted_tables'] == 7
        checkout_heads.append(inspected['repro']['trainer_head_at_train'])
        repros.append(stable_repro(inspected['repro']))
    assert repros[0] == repros[1], 'nonvolatile reproduction metadata changed'
    files = []
    for label, model in [('e30', old), ('extended', newcell / 'refit/last.bin')]:
        stripped = a.dest / f'{label}-without-repro.bin'
        if not stripped.exists():
            subprocess.run([str(a.bundle / 'bin/bake_dial_refit'), 'strip', '--in', str(model),
                            '--out', str(stripped), '--key', 'zentrain.repro'], check=True)
        files.append(stripped)
    identical = files[0].read_bytes() == files[1].read_bytes()
    report = dict(schema='e29-full-control-parity-v1', status='PASS' if identical else 'MISMATCH',
                  control_choice='coordinator shared fresh v40; unchanged by this result',
                  cell=f'{a.fold}_s0', epochs=120, pairs_per_epoch=50000, selected_epoch=119,
                  tier='v3', rayon_threads=1, trainer_sha256=sha(a.bundle/'bin/zensim_mlp_train'),
                  e30_checkpoint_sha256=sha(old), extended_checkpoint_sha256=sha(newcell/'refit/last.bin'),
                  e30_nonrepro_sha256=sha(files[0]), extended_nonrepro_sha256=sha(files[1]),
                  checkout_revision_at_train={'e30':checkout_heads[0], 'extended':checkout_heads[1]},
                  binary_producer_commit=producer['trainer_build_commit'],
                  normalized_repro_sha256=hashlib.sha256(json.dumps(repros[0], sort_keys=True).encode()).hexdigest(),
                  comparison='all non-repro bytes identical; reproduction metadata identical after documented clock/machine/build/transport-path normalization; strict table receipts, seeds, weights, coverage and dev curve identical',
                  command=cmd, input_program_sha256=job['kind']['program_sha'])
    (a.dest / 'PARITY.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not identical:
        raise AssertionError('baseline model bytes changed; fresh control remains mandatory')


if __name__ == '__main__':
    main()
