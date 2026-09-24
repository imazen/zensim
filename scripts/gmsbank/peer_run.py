#!/usr/bin/env python3
"""Run the complete exact-pixel GMSD bank peer under the shared heavy lock.

Usage: peer_run.py <implementation-commit> <release-binary> [--allow-cid22-b]
Each set gets its own command record. A bad decode or key aborts the sweep;
the final parquet manifest is written only after all 18 score files pass.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/var/tmp/gmsbank/peer_gmsd')
QUOTA = Path('/home/lilith/tmp/zensim-paper/rev4/CODEX_QUOTA_STOP.md')
HEAVY = '/home/lilith/tmp/devin/heavy'
RECORD = 'scripts/gmsbank/run_record.py'
SCRIPT = 'scripts/gmsbank/peer_bank.py'


def run(label, output, command):
    args = [sys.executable, RECORD, label, '--output', str(output), '--', *command]
    subprocess.run(args, check=True, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})


def main():
    if len(sys.argv) not in (3, 4):
        raise SystemExit('peer_run.py <implementation-commit> <release-binary> [--allow-cid22-b]')
    commit, binary = sys.argv[1:3]
    allow_sealed = len(sys.argv) == 4 and sys.argv[3] == '--allow-cid22-b'
    if len(sys.argv) == 4 and not allow_sealed:
        raise ValueError('unknown option')
    if len(commit) < 8 or not Path(binary).is_file():
        raise ValueError('implementation commit or binary missing')
    inputs = json.loads((ROOT / '_INPUTS.json').read_text())
    if len(inputs['sets']) != 18:
        raise ValueError('expected all 18 promoted bank sets')
    for set_name in sorted(inputs['sets']):
        if set_name == 'cid22_b' and not allow_sealed:
            print('peer cid22_b withheld: DEVIN_COMMON.md seal', flush=True)
            continue
        if QUOTA.exists():
            raise RuntimeError(f'quota stop before {set_name}; no more heavy work')
        score_path = ROOT / f'{set_name}.scores.tsv'
        run(f'peer_{set_name}', score_path,
            [HEAVY, '--mem', '16G', '--jobs', '8', '--', binary,
             str(ROOT / f'{set_name}.keys.tsv'), str(score_path)])
        print(f'peer {set_name} complete', flush=True)
    if not allow_sealed:
        print('17 sets scored; full manifest withheld until CID22-B authorization', flush=True)
        return
    if QUOTA.exists():
        raise RuntimeError('quota stop before peer parquet join')
    run('peer_finish', ROOT / '_MANIFEST.json',
        [HEAVY, '--mem', '16G', '--jobs', '8', '--', sys.executable, SCRIPT, 'finish', commit, binary])
    report = json.loads((ROOT / '_MANIFEST.json').read_text())
    if report['set_count'] != 18 or report['total_rows'] != 249227:
        raise ValueError(f'peer coverage: {report["set_count"]} sets, {report["total_rows"]} rows')
    print(f'peer complete {report["set_count"]} sets {report["total_rows"]} stimuli')


if __name__ == '__main__':
    main()
