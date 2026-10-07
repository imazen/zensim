"""Fit the registered E30 control recipe with E29 extensions, compare final119.

Uses the strict LODO owner and canonical strip/inspect owners. Only volatile
reproduction metadata is excluded; every other model byte must match E30.
This validation cell never selects/replaces the coordinator's shared control.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/rev4_featpot'))
from v2_common import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--e30', type=Path, required=True)
    p.add_argument('--dest', type=Path, required=True)
    a = p.parse_args()
    a.dest.mkdir(exist_ok=False)
    pinned = json.loads((a.bundle / 'E30_COMPLETE_PINS.json').read_text())
    job = json.loads((a.e30 / 'fit-manifest-fitv2e30-20261007.json').read_text())[0]
    argv = job['kind']['argv'][:]
    assert argv[argv.index('--heldout')+1] == 'kadid'
    assert argv[argv.index('--seed-index')+1] == '0'
    oldcell = Path(argv[argv.index('--dest')+1])
    old = oldcell / 'refit/last.bin'
    assert sha(old) == pinned['cells']['kadid_s0']['bake_sha256']
    argv[argv.index('--root')+1] = str(a.bundle / 'v2e29')
    argv[argv.index('--data-role-decision')+1] = str(a.bundle / 'v2e29/human_role_decision.json')
    argv[argv.index('--dest')+1] = str(a.dest / 'cell')
    env = dict(os.environ, REV4_V2_BIN_DIR=str(a.bundle / 'bin'),
               ZENSIM_MAX_TIER='v3', RAYON_NUM_THREADS='1', OMP_NUM_THREADS='1')
    cmd = [sys.executable, str(REPO / 'scripts/rev4_featpot/v2_lodo_mlp.py'), *argv[1:]]
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
    files = []
    for label, model in [('e30', old), ('extended', newcell / 'refit/last.bin')]:
        stripped = a.dest / f'{label}-without-repro.bin'
        subprocess.run([str(a.bundle / 'bin/bake_dial_refit'), 'strip', '--in', str(model),
                        '--out', str(stripped), '--key', 'zentrain.repro'], check=True)
        files.append(stripped)
    identical = files[0].read_bytes() == files[1].read_bytes()
    report = dict(schema='e29-full-control-parity-v1', status='PASS' if identical else 'MISMATCH',
                  control_choice='coordinator shared fresh v40; unchanged by this result',
                  cell='kadid_s0', epochs=120, pairs_per_epoch=50000, selected_epoch=119,
                  tier='v3', rayon_threads=1, trainer_sha256=sha(a.bundle/'bin/zensim_mlp_train'),
                  e30_checkpoint_sha256=sha(old), extended_checkpoint_sha256=sha(newcell/'refit/last.bin'),
                  e30_nonrepro_sha256=sha(files[0]), extended_nonrepro_sha256=sha(files[1]),
                  comparison='all bytes except zentrain.repro; strict table receipts, seeds, weights, coverage and dev curve identical',
                  command=cmd, input_program_sha256=job['kind']['program_sha'])
    (a.dest / 'PARITY.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)
    if not identical:
        raise AssertionError('baseline model bytes changed; fresh control remains mandatory')


if __name__ == '__main__':
    main()
