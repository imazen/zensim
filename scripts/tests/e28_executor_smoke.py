"""Local image test of the real fit-cell-exec entry, extraction and FIT_ROOT link.

Run inside the prepared image with a declared job on disk. Bounded mode uses
an explicitly opted-in site hook; first-epoch mode runs registered constants
and stops only this test's process group after the trainer's first epoch.
"""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("bounded", "first-epoch"), required=True)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--deadline", type=float, default=300)
    args = parser.parse_args()
    job_bytes = args.job.read_bytes()
    job = json.loads(job_bytes)
    arm = job["kind"]["argv"][job["kind"]["argv"].index("--spec") + 1].split(":")[-1]
    name = "EXECUTOR_SHORT" if args.mode == "bounded" else "REAL_" + arm
    out = args.out
    assert out.is_dir()
    env = dict(os.environ)
    env["E28_BOUNDED_SMOKE"] = "1" if args.mode == "bounded" else "0"
    command = ["/usr/local/bin/fit-cell-exec"]
    if args.mode == "bounded":
        command = ["/e28-time", "-v", "-o", str(out / (name + "_TIME.txt")), *command]
    root = Path(job["kind"]["argv"][job["kind"]["argv"].index("--root") + 1])
    train_log = (Path(job["kind"]["argv"][job["kind"]["argv"].index("--dest") + 1]) if "--dest" in job["kind"]["argv"] else root / "cells" / job["cell"]["image_path"]) / "train.log"
    first = None
    with (out / (name + "_OUTPUT.tar.gz")).open("wb") as stdout, (out / (name + "_STDERR.log")).open("wb") as stderr:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, env=env, start_new_session=True)
        started = time.monotonic()
        try:
            proc.stdin.write(job_bytes)
            proc.stdin.close()
            while proc.poll() is None:
                if args.mode == "first-epoch" and train_log.is_file():
                    lines = train_log.read_text(errors="replace").splitlines()
                    first = next((line for line in lines if re.search(r"^\s*epoch\s+0\s+\|.*val\(geomean3\)=", line)), None)
                    if first:
                        command_line = lines[0]
                        assert "--epochs 120 " in command_line and "--pairs-per-epoch 50000 " in command_line
                        assert "E28_BOUNDED_SMOKE" not in (out / (name + "_STDERR.log")).read_text(errors="replace")
                        shutil.copy2(train_log, out / (name + "_TRAIN.log"))
                        os.killpg(proc.pid, signal.SIGTERM)
                        break
                if time.monotonic() - started > args.deadline:
                    raise TimeoutError("fit-cell-exec smoke deadline exceeded")
                time.sleep(0.1)
            try:
                code = proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                code = proc.wait(timeout=5)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=5)
    if args.mode == "bounded":
        assert code == 0, ("fit-cell-exec failed", code, str(out / (name + "_STDERR.log")))
        assert (out / (name + "_OUTPUT.tar.gz")).stat().st_size > 0
    else:
        assert first, ("no first epoch", code, str(out / (name + "_STDERR.log")))
    link = Path("/var/tmp/rev4-featpot")
    target = Path("/scratch/fit-cell") / job["kind"]["data_sha"] / "rev4-featpot"
    assert link.is_symlink() and link.resolve() == target.resolve()
    assert (target.parent / ".verified").read_text().strip() == job["kind"]["data_sha"]
    record = dict(status="PASS", mode=args.mode, arm=arm, actual_entry="/usr/local/bin/fit-cell-exec", declared_argv=job["kind"]["argv"], argv_sha=job["kind"]["argv_sha"], cell=job["cell"]["image_path"], program_sha=job["kind"]["program_sha"], data_sha=job["kind"]["data_sha"], fit_root_link=str(link), verified_extraction_target=str(target), extraction_verified=True, bounded_constants=args.mode == "bounded", first_epoch_line=first, stopped_after_first_epoch=bool(first), child_returncode=code, memory_peak_bytes=int(Path("/sys/fs/cgroup/memory.peak").read_text()))
    (out / (name + "_PATH_PASS.json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "declared_argv"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
