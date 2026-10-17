"""Visible waits for run-heavy's exact pre-execution low-memory refusal."""
from datetime import datetime, timezone
from pathlib import Path
import re
import subprocess
import time


REFUSAL = re.compile(r"run-heavy: only [0-9]+ MiB available — refusing to launch \(free memory first\)\n?")


def run_with_memory_admission(command, log_path, output_path, *, env=None):
    """Every execution error is fatal; only a refused, unstarted command waits."""
    attempt = 0
    while True:
        log = log_path if attempt == 0 else log_path.with_name(log_path.stem+f".memory-admission-{attempt:03d}"+log_path.suffix)
        with log.open("x") as stream:
            result = subprocess.run(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode == 0:
            return
        if result.returncode != 3 or REFUSAL.fullmatch(log.read_text()) is None:
            raise subprocess.CalledProcessError(result.returncode,command)
        assert not Path(output_path).exists(), "low-memory refusal has a populated output; refusing automatic re-execution"
        print(f"{datetime.now(timezone.utc).isoformat()} memory admission refused before execution; waiting30s outside shared lock; receipt={log}",flush=True)
        time.sleep(30)
        attempt += 1
