"""Parallelize disjoint registered D2 MLP replicates inside one heavy lock."""

import argparse
import concurrent.futures
import json
import os
import shlex
import subprocess
from lodo_bvls import SOURCES
from p2_data import ROOT


def tasks(arm):
    for hidden in (32, 128):
        for heldout in SOURCES:
            for rep in range(5):
                dest = (ROOT / "p2/d2_mlp" / f"LODO_{arm}_mlp{hidden}" /
                        f"without_{heldout}_r{rep}")
                cmd = ["python", "scripts/rev4_featpot/p2_lodo_mlp.py",
                       "--arm", arm, "--hidden", str(hidden),
                       "--heldout", heldout, "--rep", str(rep)]
                yield {"arm": arm, "hidden": hidden, "heldout": heldout,
                       "rep": rep, "dest": dest, "cmd": cmd}


def queued():
    proc = subprocess.run(["ps", "-ww", "-eo", "args="], capture_output=True,
                          text=True, check=True)
    rows = set()
    for line in proc.stdout.splitlines():
        if "p2_lodo_mlp.py" not in line:
            continue
        try:
            argv = shlex.split(line)
            next(x for x in argv if x.endswith("/p2_lodo_mlp.py"))
            def value(flag):
                return argv[argv.index(flag) + 1]
            rows.add((value("--arm"), int(value("--hidden")),
                      value("--heldout"), int(value("--rep"))))
        except (StopIteration, ValueError, IndexError):
            continue
    return rows


def run_one(task):
    result = task["dest"] / "result.json"
    if result.is_file():
        return {"task": task["cmd"], "status": "already-complete"}
    log = (ROOT / "p2/d2_mlp_accel" /
           f"{task['arm']}_h{task['hidden']}_without_{task['heldout']}_r{task['rep']}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as output:
        proc = subprocess.run(task["cmd"], stdout=output, stderr=subprocess.STDOUT,
                              env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
                                   "PYTHONPYCACHEPREFIX": str(ROOT / "pycache")})
    if proc.returncode or not result.is_file():
        raise RuntimeError(f"D2 MLP fit rc={proc.returncode}: {task['cmd']}; see {log}")
    return {"task": task["cmd"], "status": "complete", "result": str(result),
            "log": str(log)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.batch < 2 or args.batch > 8:
        raise ValueError("batch must be 2..8")
    busy = queued()
    def eligible(arm):
        return [task for task in reversed(list(tasks(arm)))
                if not (task["dest"] / "result.json").is_file() and
                (arm, task["hidden"], task["heldout"], task["rep"]) not in busy]
    groups = {arm: eligible(arm) for arm in ("p2", "p2_perm", "r0")}
    quotas = {"p2": args.batch // 3 + (1 if args.batch % 3 > 0 else 0),
              "p2_perm": args.batch // 3 + (1 if args.batch % 3 > 1 else 0),
              "r0": args.batch // 3}
    chosen = [task for arm in groups for task in groups[arm][:quotas[arm]]]
    if len(chosen) < args.batch:
        extra = [task for arm in groups for task in groups[arm][quotas[arm]:]
                 if task not in chosen]
        chosen += extra[:args.batch - len(chosen)]
    print(json.dumps({"queued_external": len(busy), "selected": [t["cmd"] for t in chosen],
                      "remaining": {arm: len(rows) for arm, rows in groups.items()}}), flush=True)
    if not chosen:
        return
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(chosen)) as pool:
        futures = [pool.submit(run_one, task) for task in chosen]
        failures = []
        for future in concurrent.futures.as_completed(futures):
            try:
                print(json.dumps(future.result()), flush=True)
            except Exception as error:
                failures.append(str(error))
                print(json.dumps({"failure": str(error)}), flush=True)
        if failures:
            raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
