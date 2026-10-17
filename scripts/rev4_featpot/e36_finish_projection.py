"""Explicit planning model from observed native cell costs; never admits data."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import gzip
import json
from pathlib import Path
import time
import statistics

import pyarrow.parquet as pq


def observed_costs(root, out):
    """Bounded deterministic diagnostic sample of post-widening checkpoints."""
    jobs=json.loads((root/'runs/fallback-v6/manifest.json').read_text())
    launched=json.loads((root/'scheduling-v7/fallback-launch.json').read_text())['started_unix_ns']/1e9
    freeze=root/'budget-amendment-v3/live/processes.json'
    resumed=json.loads((root/'budget-amendment-v3/live/resumed.json').read_text())['unix_ns']/1e9
    frozen=freeze.stat().st_mtime-1
    groups=defaultdict(list)
    samples=[]
    for job in jobs:
        text=job['kind']['view'];view=json.loads(text.split(':',1)[1])
        if view['panel']!='baseline':continue
        key=hashlib.sha256(text.encode()).hexdigest()
        offset=int(key[:8],16)%len(view['q_grid'])
        for ordinal in sorted({(offset+i*len(view['q_grid'])//8)%len(view['q_grid']) for i in range(8)}):
            p=root/'progress'/key/(str(ordinal)+'.json.gz')
            if not p.exists():continue
            end=p.stat().st_mtime
            if end<launched+60:continue
            row=json.loads(gzip.decompress(p.read_bytes()))
            group=json.dumps([row['codec'],row['size'],view['panel']],separators=(',',':'))
            if len(groups[group])>=128 or row['physical_alias'] or 'cell_total_seconds' not in row:continue
            elapsed=row['cell_total_seconds'];start=end-elapsed
            if start<launched or start<=resumed and end>=frozen:continue
            groups[group].append(elapsed)
            samples.append(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),group=group,wall_seconds=elapsed,completed_unix=end,content=row['content'],q=row['q']))
    result=dict(schema='e36-observed-wide-cell-costs-v1',fits=0,qualifies_data=False,since_unix=launched,
        sampling='Up to8uniformly spaced requested-q ordinals with a deterministic hash offset per baseline view; at most128physical checkpoints per codec/size. Excludes earlier cells and the recorded live resource-amendment pause. Diagnostic sample, not a content/quality fit or population bound.',
        groups={k:dict(n=len(v),median_seconds=statistics.median(v),p75_seconds=statistics.quantiles(v,n=4)[2] if len(v)>1 else v[0]) for k,v in groups.items()},samples=samples)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(dict(groups=len(groups),physical_samples=len(samples),fits=0)))


def project(root, out, cost_snapshot=None, exclude_rav1e=False, phase='fallback'):
    now = time.time()
    run = root / "runs/fallback-v6"
    ids = [
        line.split("\t")[1]
        for line in (root / "scheduling-v7/fallback.original.ids.tsv")
        .read_text()
        .splitlines()
    ]
    jobs = json.loads((run / "manifest.json").read_text())
    if phase=='additions':
        from e36_svt_matrix import configuration
        proof,_,_=configuration(root);entry=proof['phases']['additions'];run=root/'runs'/entry['run']
        jobs=json.loads(Path(entry['manifest']['path']).read_text());ids=[line.split('\t')[1] for line in Path(entry['native_ids']['path']).read_text().splitlines()]
    elif phase!='fallback':raise ValueError('registered diagnostic phase required')
    if len(ids) != len(jobs):
        raise ValueError("native ordered job identity inventory differs")
    original_count = len(jobs)
    excluded_ids = {job_id for job_id, job in zip(ids, jobs)
                    if exclude_rav1e and json.loads(job['kind']['view'].split(':', 1)[1])['codec'] == 'rav1e'}
    selected = [(job_id, job) for job_id, job in zip(ids, jobs) if job_id not in excluded_ids]
    ids = [item[0] for item in selected]
    jobs = [item[1] for item in selected]
    selected_ids = set(ids)
    done = set()
    ledgers = []
    # Ignore only files whose write could still be in progress in this snapshot.
    # This conservatively counts their work as unfinished, never admits rows.
    for path in sorted(run.glob("ledger*.parquet")):
        if path.stat().st_mtime > now - 2:
            continue
        table = pq.read_table(path, columns=["job_id", "status"])
        for row in table.to_pylist():
            if row['job_id'] in excluded_ids:
                continue
            if row['job_id'] not in selected_ids:
                raise ValueError('projection ledger contains an unknown native job')
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
    observed={} if cost_snapshot is None else json.loads(cost_snapshot.read_text())['groups']
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
        group_key=json.dumps([codec,size,panel],separators=(',',':'))
        sample=observed.get(group_key)
        median = sample['median_seconds'] if sample and sample['n']>=8 else measured[(codec, size, cost_run)]
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
    effective = root / "bulk-clock-v8-budget4.json"
    if not effective.exists():
        effective = root / "bulk-clock-v7-budget3.json"
    if not effective.exists():
        effective = root / "bulk-clock-v6-budget2.json"
    _, clock, _ = resolve(effective)
    ideal = max(serial / 24, longest, weighted / launch["job_budget_bytes"])
    result = dict(
        schema="e36-planning-projection-v1",
        phase=phase,
        observed_unix=now,
        observed_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
        done_views=len(done),
        required_views=len(jobs),
        original_required_views=original_count,
        owner_excluded_rav1e_views=len(excluded_ids),
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
        observed_costs=None if cost_snapshot is None else dict(path=str(cost_snapshot),sha256=hashlib.sha256(cost_snapshot.read_bytes()).hexdigest()),
        ledger_snapshot_files=ledgers,
        qualifies_data=False,
        fits=0,
        basis=("Hybrid measured post-widening physical checkpoint medians where at least8diagnostic samples exist; all other strata retain their own measured prebulk cost, with no transfer between sizes. " if cost_snapshot is not None else "")+"Measured per-codec/per-size prebulk unique-complete-cell medians; true4096-square witness at4096; native mode-panel medians at64/256. Remaining physical requests subtract existing per-view checkpoints and native mode aliases. Ideal24slot and reservation-weighted model, not a measured finish or statistical bound.",
        limitations=[
            "No source-content or quality-dependent cost model.",
            "Excludes cached-row readback, scheduling, per-view ledger overhead and external interference.",
            "Simultaneous in-flight cells are treated as unfinished until their checkpoint lands.",
            "Only the selected "+phase+" phase modeled; other phase CPU work excluded.",
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
    p.add_argument('--costs',type=Path)
    p.add_argument('--sample-wide-costs',action='store_true')
    p.add_argument('--exclude-rav1e', action='store_true')
    p.add_argument('--phase',choices=['fallback','additions'],default='fallback')
    args = p.parse_args()
    if args.sample_wide_costs:observed_costs(args.root,args.out)
    else:project(args.root, args.out,args.costs,args.exclude_rav1e,args.phase)
