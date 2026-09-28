"""Run disjoint registered P0/P2 MLP fit cells concurrently under one heavy lock.

The original idempotent runners remain the owners of importance and CI. This
batcher selects only absent fit receipts and excludes cells already queued by
those runners, then executes separate per-cell commands with isolated outputs.
"""

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
from pathlib import Path


ROOT = Path("/var/tmp/rev4-featpot")
FLEET_GO = (Path.home() / "tmp/zensim-paper/rev4/FLEET_FITS_GO.md")
SETS = ("kadid_train", "tid2013", "konfig_train", "konjnd_bpg_train",
        "cid22_a25", "aic3", "kadid_select", "konfig_val")


def tasks(scope):
    arms = ("r0", "minus_basic") if scope == "p0" else ("p2", "p2_perm")
    script = "mlp_probe.py" if scope == "p0" else "p2_mlp.py"
    for name in SETS:
        for arm in arms:
            for hidden in (32, 128):
                for outer in (*range(5), None):
                    tag = f"o{outer}" if outer is not None else "full"
                    for rep in range(5):
                        base = ROOT / ("fits" if scope == "p0" else "p2/mlp")
                        dest = base / f"POT_{name}_{arm}_mlp{hidden}" / f"{tag}_r{rep}"
                        cmd = ["python", f"scripts/rev4_featpot/{script}",
                               "--set", name, "--arm", arm, "--hidden", str(hidden)]
                        if outer is not None:
                            cmd += ["--outer", str(outer)]
                        cmd += ["--rep", str(rep)]
                        yield {"scope": scope, "set": name, "arm": arm, "hidden": hidden,
                               "outer": outer, "rep": rep, "dest": dest, "cmd": cmd}


def queued():
    proc = subprocess.run(["ps", "-ww", "-eo", "args="], capture_output=True,
                          text=True, check=True)
    rows = set()
    for line in proc.stdout.splitlines():
        if "mlp_probe.py" not in line and "p2_mlp.py" not in line:
            continue
        try:
            argv = shlex.split(line)
            script = next(x for x in argv if x.endswith("/mlp_probe.py") or x.endswith("/p2_mlp.py"))
            value = lambda flag: argv[argv.index(flag) + 1]  # noqa: E731
            rows.add(("p0" if script.endswith("mlp_probe.py") else "p2",
                      value("--set"), value("--arm"), int(value("--hidden")),
                      int(value("--outer")) if "--outer" in argv else None,
                      int(value("--rep"))))
        except (StopIteration, ValueError, IndexError):
            continue
    return rows


def run_one(task):
    dest = task["dest"]
    if task["scope"] == "p0" and FLEET_GO.is_file():
        return {"task": task["cmd"], "status": "fleet-go-paused-before-fit"}
    if (dest / "result.json").is_file():
        return {"task": task["cmd"], "status": "already-complete"}
    log = ROOT / "mlp_accel" / (f"{task['scope']}_{task['set']}_{task['arm']}_h{task['hidden']}_"
                               f"{task['outer'] if task['outer'] is not None else 'full'}_r{task['rep']}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        proc = subprocess.run(task["cmd"], stdout=output, stderr=subprocess.STDOUT,
                              env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                                   "PYTHONPYCACHEPREFIX": str(ROOT / "pycache")})
    if proc.returncode or not (dest / "result.json").is_file():
        raise RuntimeError(f"fit failed rc={proc.returncode}: {task['cmd']}; see {log}")
    return {"task": task["cmd"], "status": "complete", "result": str(dest / "result.json"),
            "log": str(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.batch < 2 or args.batch > 8:
        raise ValueError("batch must be 2..8 within the heavy --jobs 8 reservation")
    busy = queued()
    def eligible(scope):
        # Serial runners advance from KADID outer0; work from the far end of
        # the grid so a serial runner cannot prequeue one of our cells while
        # this batch holds the heavy lock.
        return [task for task in reversed(list(tasks(scope)))
                if not (task["dest"] / "result.json").is_file() and
                (scope, task["set"], task["arm"], task["hidden"], task["outer"], task["rep"]) not in busy]
    p2 = eligible("p2")
    p0 = [] if FLEET_GO.is_file() else eligible("p0")
    half = args.batch // 2
    chosen = p2[:half] + p0[:args.batch - half]
    if len(chosen) < args.batch:
        extra = [task for task in p2[half:] + p0[args.batch - half:] if task not in chosen]
        chosen += extra[:args.batch - len(chosen)]
    print(json.dumps({"queued_external": len(busy), "selected": [t["cmd"] for t in chosen],
                      "remaining_p0": len(p0), "remaining_p2": len(p2)}), flush=True)
    if not chosen:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        futures = [pool.submit(run_one, task) for task in chosen]
        failures = []
        for future in concurrent.futures.as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:  # keep other independent results
                failures.append(str(error))
                print(json.dumps({"failure": str(error)}), flush=True)
        if failures:
            raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
