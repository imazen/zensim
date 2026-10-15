"""Run one explicitly bounded E34 smoke through the existing strict fit owner.

Uses the existing E33 local-smoke budget (one epoch, 49,999 pairs), never a
registered fit. Packing and qualification are separate measured operations.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from v2_common import fx1_declaration, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm', required=True, choices=['base-a', 'base-c', 'w-a', 'w-c', 'l-a', 'k-a', 'k-c'])
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--dest', required=True, type=Path)
    p.add_argument('--bin-dir', required=True, type=Path)
    p.add_argument('--steer-pairs', type=Path)
    p.add_argument('--steer-lambda', type=float, default=0.0)
    a = p.parse_args()
    if a.arm.startswith('k-') != (a.steer_pairs is not None and a.steer_lambda > 0):
        raise ValueError('K smoke requires a table and positive fixed smoke lambda')
    if a.dest.exists():
        raise ValueError('smoke destination must be fresh')
    hidden = 1 if a.arm == 'l-a' else 32 if a.arm.startswith('w-') else 128
    spec = f'sel:3b7bd5ebe929@h32:H{hidden}:cv16:cf98'
    if a.arm.endswith('-c'):
        spec += ':fx1'
    driver = Path(__file__).with_name('v2_confirm_fit.py')
    argv = [sys.executable, str(driver), '--root', str(a.root), '--spec', spec,
            '--columns', ','.join(map(str, fx1_declaration()['direct'])), '--head', 'N',
            '--seed-index', '0', '--strict-admission', '--train-only',
            '--data-role-decision', str(a.root / 'human_role_decision.json'),
            '--local-smoke-budget', '1:49999', '--dest', str(a.dest)]
    if a.arm == 'l-a':
        argv.append('--monotone-cbc')
    if a.arm.startswith('k-'):
        argv += ['--steer-pairs', str(a.steer_pairs), '--steer-lambda', str(a.steer_lambda)]
    env = dict(os.environ, REV4_V2_BIN_DIR=str(a.bin_dir), RAYON_NUM_THREADS='1',
               OMP_NUM_THREADS='1', ZENSIM_FORMULA_REV='5', ZENSIM_MAX_TIER='v3')
    started = time.monotonic()
    result = subprocess.run(argv, env=env)
    receipt = dict(schema='e34-first-epoch-smoke-v1', arm=a.arm, argv=argv,
                   execution_contract='local-smoke', epochs=1, rank_pairs=49999,
                   returncode=result.returncode, wall_seconds=time.monotonic()-started,
                   trainer_sha256=sha(a.bin_dir / 'zensim_mlp_train'),
                   steer_pairs_sha256=sha(a.steer_pairs) if a.steer_pairs else None,
                   steer_lambda=a.steer_lambda, lambda_selection=False)
    if (a.dest / 'result.json').exists():
        receipt['result_sha256'] = sha(a.dest / 'result.json')
    with a.dest.with_suffix('.smoke.json').open('x') as f:
        json.dump(receipt, f, indent=2)
        f.write('\n')
    print(json.dumps(receipt), flush=True)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
