#!/usr/bin/env python3
"""Run a lane command with exact UTC bounds, exit code and output hashes."""
import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path('/var/tmp/gmsbank/command_records')


def stamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('label')
    p.add_argument('--output', action='append', default=[])
    try:
        separator = sys.argv.index('--')
    except ValueError:
        raise SystemExit('separate command with --') from None
    a = p.parse_args(sys.argv[1:separator])
    command = sys.argv[separator + 1:]
    if not command:
        raise SystemExit('missing command')
    ROOT.mkdir(parents=True, exist_ok=True)
    log = ROOT / f'{a.label}.log'
    start = stamp()
    with log.open('wb') as f:
        result = subprocess.run(command, cwd=os.getcwd(), stdout=f, stderr=subprocess.STDOUT,
                                check=False)
    end = stamp()
    outputs = {}
    for name in a.output:
        path = Path(name)
        if path.is_file():
            outputs[str(path)] = digest(path)
        else:
            outputs[str(path)] = None
    record = dict(label=a.label, start_utc=start, end_utc=end, cwd=os.getcwd(),
                  command=command, exit_code=result.returncode, log=str(log),
                  log_sha256=digest(log), outputs=outputs)
    (ROOT / f'{a.label}.json').write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
    print(json.dumps(record, sort_keys=True))
    sys.exit(result.returncode)


if __name__ == '__main__':
    main()
