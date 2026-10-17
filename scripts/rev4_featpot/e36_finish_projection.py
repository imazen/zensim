"""Explicit planning model from observed native cell costs; never admits data."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import pyarrow.parquet as pq


def project(root, out):
    now = time.time()
    run = root / "runs/fallback-v6"
    ids = [
        line.split("\t")[1]
        for line in (root / "scheduling-v7/fallback.original.ids.tsv")
        .read_text()
        .splitlines()
    ]
    jobs = json.loads((run / "manifest.json").read_text())
    if len(ids) != len(jobs):
        raise ValueError("native ordered job identity inventory differs")
    done = set()
    ledgers = []
    # Ignore only files whose write could still be in progress in this snapshot.
    # This conservatively counts their work as unfinished, never admits rows.
    for path in sorted(run.glob("ledger*.parquet")):
        if path.stat().st_mtime > now - 2:
            continue
        table = pq.read_table(path, columns=["job_id", "status"])
        for row in table.to_pylist():
            if row["status"].lower() != "done":
                raise ValueError("projection needs explicit failed-status accounting")
            done.add(row["job_id"])
        ledgers.append(str(path))
    costs = json.loads((root / "MEASURED_STAGE_COSTS-v2.json").read_text())["rows"]
    measured = {
        (r["codec"], r["size"], r["run"]): r["unique_complete_cell_median_seconds"]
        for r in costs
    }
    runtime = json.loads((root / "runtime-v6/runtime.json").read_text())
    resource = json.loads((root / "resource-admission.json").read_text())
    refs = {
        r["reference_key"]: r
        for r in json.loads(Path(runtime["references"]).read_text())["rows"]
    }
    modes = {
        codec: [
            json.loads(line)
            for line in (Path(runtime["mode_inventory"]) / (codec + ".jsonl"))
            .read_text()
            .splitlines()
        ]
        for codec in ["jpeg", "rav1e", "svt", "webp", "jxl"]
    }
    serial = 0.0
    longest = 0.0
    weighted = 0.0
    cached = 0
    physical = 0
    strata = defaultdict(
        lambda: dict(
            views=0,
            physical_cells_remaining=0,
            measured_median_seconds=None,
            modeled_serial_seconds=0.0,
        )
    )
    for job_id, job in zip(ids, jobs):
        if job_id in done:
            continue
        text = job["kind"]["view"]
        view = json.loads(text.split(":", 1)[1])
        ref = refs[view["reference_key"]]
        size = ref["size"]
        codec = view["codec"]
        panel = view["panel"]
        directory = root / "progress" / hashlib.sha256(text.encode()).hexdigest()
        checkpoints = (
            {int(p.name.split(".", 1)[0]) for p in directory.glob("*.json.gz")}
            if directory.exists()
            else set()
        )
        if panel == "baseline":
            cells = [json.dumps([q], separators=(",", ":")) for q in view["q_grid"]]
            cost_run = "max-area-v4" if size == 4096 else "cost-v4"
        else:
            cells = [
                json.dumps(
                    [r["q"], json.loads(r["knob_tuple_json"])],
                    sort_keys=True,
                    separators=(",", ":"),
                )
                for r in modes[codec]
            ]
            cost_run = "modes-v4"
        if any(i >= len(cells) or i < 0 for i in checkpoints):
            raise ValueError("unexpected checkpoint ordinal")
        seen = {cells[i] for i in checkpoints}
        needed = {cell for i, cell in enumerate(cells) if i not in checkpoints} - seen
        count = len(needed)
        median = measured[(codec, size, cost_run)]
        duration = count * median
        physical += count
        cached += len(checkpoints)
        serial += duration
        longest = max(longest, duration)
        # A reservation-based planning axis, not a memory-usage estimate.
        mem = resource["observed_native_peak_rss_kib"][str(size)] * 1024
        reservation = max(256 * 1024**2, (mem * 9 + 7) // 8 + 128 * 1024**2)
        if panel != "baseline":
            reservation = max(reservation, 2 * 1024**3)
        weighted += duration * reservation
        key = json.dumps([codec, size, panel], separators=(",", ":"))
        s = strata[key]
        s["views"] += 1
        s["physical_cells_remaining"] += count
        s["measured_median_seconds"] = median
        s["modeled_serial_seconds"] += duration
    launch = json.loads((root / "scheduling-v7/fallback-launch.json").read_text())
    from e36_pause_budget import resolve
    effective = root / "bulk-clock-v7-budget3.json"
    if not effective.exists():
        effective = root / "bulk-clock-v6-budget2.json"
    _, clock, _ = resolve(effective)
    ideal = max(serial / 24, longest, weighted / launch["job_budget_bytes"])
    result = dict(
        schema="e36-planning-projection-v1",
        observed_unix=now,
        observed_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        done_views=len(done),
        remaining_views=len(jobs) - len(done),
        remaining_physical_cells=physical,
        existing_checkpoint_rows=cached,
        modeled_serial_seconds=serial,
        ideal24_slot_seconds=serial / 24,
        longest_modeled_ladder_seconds=longest,
        reservation_weighted_seconds=weighted / launch["job_budget_bytes"],
        optimistic_modeled_seconds_remaining=ideal,
        optimistic_modeled_finish_utc=datetime.fromtimestamp(
            now + ideal, timezone.utc
        ).isoformat(),
        misses_registered_deadline_in_this_model=now + ideal > clock["deadline_unix"],
        deadline_unix=clock["deadline_unix"],
        strata=dict(strata),
        ledger_snapshot_files=ledgers,
        qualifies_data=False,
        fits=0,
        basis="Measured per-codec/per-size prebulk unique-complete-cell medians; true4096-square witness at4096; native mode-panel medians at64/256. Remaining physical requests subtract existing per-view checkpoints and native mode aliases. Ideal24slot and reservation-weighted model, not a measured finish or statistical bound.",
        limitations=[
            "No source-content or quality-dependent cost model.",
            "Excludes cached-row readback, scheduling, per-view ledger overhead and external interference.",
            "Simultaneous in-flight cells are treated as unfinished until their checkpoint lands.",
            "Only mandatory15342views modeled; optional468additions excluded.",
        ],
    )
    with out.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["strata", "ledger_snapshot_files", "limitations"]
            }
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "out"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    project(args.root, args.out)
