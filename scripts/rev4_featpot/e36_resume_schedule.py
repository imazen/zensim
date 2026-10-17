"""Resume the exact native E36 matrix with a scheduling-only worker amendment."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time

from e36_store import sha
from e36_pause_budget import resolve


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def prepare(root, worker, out):
    out.mkdir()
    plan_path = root / "bulk-plan-v6.json"
    plan = json.loads(plan_path.read_text())
    clock_path = root / "bulk-clock-v6-budget2.json"
    _, clock, _ = resolve(clock_path)
    resource = json.loads((root / "resource-admission.json").read_text())
    manifests = {}
    for role in ["fallback", "additions"]:
        entry = plan["phases"][role]
        source = root / "runs" / entry["run"] / "manifest.json"
        if sha(source) != entry["manifest_sha256"]:
            raise ValueError("registered native phase changed")
        jobs = json.loads(source.read_text())
        references = json.loads(
            Path(
                json.loads(Path(plan["runtime"]).read_text())["references"]
            ).read_text()
        )
        refs = {r["reference_key"]: r for r in references["rows"]}
        previous = json.loads(source.read_text())
        for job in jobs:
            view = json.loads(job["kind"]["view"].split(":", 1)[1])
            size = refs[view["reference_key"]]["size"]
            # Observed per-invocation peak + explicit12.5% margin and128MiB
            # persistent-Python allowance; these are scheduling reservations,
            # not a claim of measured bounds for every source or mode.
            measured = resource["observed_native_peak_rss_kib"][str(size)] * 1024
            reservation = max(256 * 1024**2, (measured * 9 + 7) // 8 + 128 * 1024**2)
            if view["panel"] == "mode-panel":
                reservation = max(reservation, resource["mode_panel_bytes"])
            job["hint"] = dict(peak_mem_bytes=reservation, threads=1, vram_bytes=None)
        for a, b in zip(previous, jobs):
            if {k: v for k, v in a.items() if k != "hint"} != {
                k: v for k, v in b.items() if k != "hint"
            }:
                raise ValueError("scheduling amendment changed scientific job identity")
        dest = out / (role + ".json")
        write(dest, jobs)
        manifests[role] = dict(
            path=str(dest),
            sha256=sha(dest),
            original=dict(path=str(source), sha256=sha(source)),
            jobs=len(jobs),
        )
    result = dict(
        schema="e36-native-scheduling-only-v1",
        plan=dict(path=str(plan_path), sha256=sha(plan_path)),
        effective_clock=dict(path=str(clock_path), sha256=sha(clock_path)),
        worker=dict(path=str(worker), sha256=sha(worker)),
        manifests=manifests,
        hard_scope_bytes=38 * 1024**3,
        live_reserve_bytes=8 * 1024**3,
        max_concurrent=24,
        maximum_job_budget_bytes=30 * 1024**3,
        scientific_change=False,
        deadline_changed=False,
        fits=0,
        reservation_basis=dict(
            path=str(root / "resource-admission.json"),
            sha256=sha(root / "resource-admission.json"),
        ),
    )
    if clock["deadline_unix"] <= time.time():
        raise TimeoutError("registered deadline already passed")
    write(out / "amendment.json", result)


def run(root, amendment_path):
    a = json.loads(amendment_path.read_text())
    plan = json.loads(Path(a["plan"]["path"]).read_text())
    _, clock, _ = resolve(Path(a["effective_clock"]["path"]))
    if (
        a["schema"] != "e36-native-scheduling-only-v1"
        or a["scientific_change"] is not False
        or sha(a["worker"]["path"]) != a["worker"]["sha256"]
        or sha(a["plan"]["path"]) != a["plan"]["sha256"]
        or sha(a["effective_clock"]["path"]) != a["effective_clock"]["sha256"]
    ):
        raise ValueError("immutable scheduling-only admission differs")
    if (root / "TOWER_PHASE_HANDOFF.json").exists():
        raise ValueError("another host owns additions")
    owner_path = Path(__file__).with_name("e36_bulk.py")
    spec = importlib.util.spec_from_file_location("e36_native_bulk_owner", owner_path)
    owner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(owner)
    errors = []
    qualified = None
    for role in ["fallback", "additions"]:
        if role == "additions" and qualified != "fallback":
            break
        if time.time() >= clock["deadline_unix"]:
            break
        entry = a["manifests"][role]
        original = entry["original"]
        if (
            sha(entry["path"]) != entry["sha256"]
            or sha(original["path"]) != original["sha256"]
        ):
            raise ValueError("native scheduling manifest changed")
        directory = root / "runs" / plan["phases"][role]["run"]
        available = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        budget = min(a["maximum_job_budget_bytes"], available - a["live_reserve_bytes"])
        if budget < 4 * 1024**3:
            raise RuntimeError("insufficient live RAM beyond8GiB reserve")
        argv = [
            "/usr/bin/timeout",
            "--signal=TERM",
            "--kill-after=120",
            str(clock["deadline_unix"] - time.time()),
            "taskset",
            "--cpu-list",
            "0-23",
            a["worker"]["path"],
            "--manifest",
            entry["path"],
            "--ledger-out",
            str(directory / "ledger.parquet"),
            "--blobs",
            str(directory / "blobs"),
            "--exec",
            str(Path(plan["runtime"]).parent / "executor"),
            "--worker",
            "e36-dev-fair-v7",
            "--provider",
            "local",
            "--max-attempts",
            "1",
        ]
        for ledger in sorted(directory.glob("ledger*.parquet")):
            argv.extend(["--ledger-in", str(ledger)])
        env = {
            **os.environ,
            "ZEN_LOCAL_FAIR_SCHEDULER": "1",
            "ZEN_LOCAL_RESERVE_BYTES": str(a["live_reserve_bytes"]),
            "ZEN_RAM_BYTES": str(budget * 4 // 3),
            "ZEN_CHUNK_WALL_SEC": "300",
            "ZEN_PERSISTENT_EXEC": "1",
            "ZEN_PERSISTENT_KINDS": "bake",
            "RAYON_NUM_THREADS": "24",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "E36_GENERATION_DEADLINE": str(clock["deadline_unix"]),
        }
        started = time.time_ns()
        write(
            amendment_path.parent / (role + "-launch.json"),
            dict(
                argv=argv,
                job_budget_bytes=budget,
                available_bytes=available,
                live_reserve_bytes=a["live_reserve_bytes"],
                started_unix_ns=started,
                effective_deadline_unix=clock["deadline_unix"],
                fits=0,
            ),
        )
        rc = subprocess.run(argv, env=env).returncode
        write(
            amendment_path.parent / (role + "-exit.json"),
            dict(returncode=rc, closed_unix_ns=time.time_ns(), fits=0),
        )
        if rc:
            errors.append(str(rc))
        if owner.qualify(
            root,
            plan["phases"][role]["run"],
            clock["started_unix"],
            clock["deadline_unix"],
        ):
            qualified = "fallback" if role == "fallback" else "full"
        else:
            break
    names = [] if qualified is None else [plan["phases"]["fallback"]["run"]]
    if qualified == "full":
        names.append(plan["phases"]["additions"]["run"])
    write(
        root / "bulk-choice.json",
        dict(
            schema="e36-fixed-matrix-choice-v1",
            plan_path=a["plan"]["path"],
            clock_path=a["effective_clock"]["path"],
            plan_sha256=a["plan"]["sha256"],
            clock_sha256=a["effective_clock"]["sha256"],
            qualified=qualified,
            selected_runs=names,
            errors=errors,
            closed_unix=time.time(),
            fits=0,
            teacher_support_not_used_for_choice=True,
            scheduling_amendment=dict(
                path=str(amendment_path), sha256=sha(amendment_path)
            ),
        ),
    )
    if qualified is None:
        raise ValueError("mandatory matrix incomplete; no fit pool qualifies")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["prepare", "run"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--worker", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--amendment", type=Path)
    args = p.parse_args()
    if args.mode == "prepare":
        prepare(args.root, args.worker, args.out)
    else:
        run(args.root, args.amendment)
