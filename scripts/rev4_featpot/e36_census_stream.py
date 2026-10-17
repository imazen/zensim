"""Stream the existing native census predicate without retaining feature rows."""

from array import array
from collections import defaultdict
import gzip
import json
from pathlib import Path
import statistics

from e36_data import PREFIX, canonical
from e36_store import sha

STAGES = ["encode_execution", "score_execution", "feature_execution"]


def group():
    return dict(
        logical=0,
        physical=0,
        stage={s: array("d") for s in STAGES},
        basic_peak=0,
        extended_peak=0,
        bundle_bytes=0,
        total=array("d"),
        persist=array("d"),
        maps=array("d"),
        all_total=True,
        extended_missing=False,
    )


def census(root, name):
    out = Path(root) / "runs" / name
    files = []
    completed = []
    groups = defaultdict(group)
    logical = 0
    for path in sorted((out / "blobs").rglob("*")):
        if not path.is_file():
            continue
        try:
            doc = json.loads(gzip.decompress(path.read_bytes()))
        except (ValueError, OSError):
            continue
        if doc.get("schema") != "e36-native-ladder-receipt-v1":
            continue
        if not doc["complete"]:
            raise ValueError("incomplete output")
        files.append(dict(path=str(path), sha256=sha(path)))
        completed.append(PREFIX + canonical(doc["view"]))
        logical += len(doc["rows"])
        for row in doc["rows"]:
            g = groups[f"{row['size']}:{row['codec']}"]
            g["logical"] += 1
            if row["physical_alias"]:
                continue
            g["physical"] += 1
            g["bundle_bytes"] += row["bundle"]["bytes"]
            for stage in STAGES:
                g["stage"][stage].append(row[stage]["wall_seconds"])
                g["basic_peak"] = max(g["basic_peak"], row[stage]["peak_rss_kib"])
            if "cell_total_seconds" not in row:
                g["all_total"] = False
                continue
            g["total"].append(row["cell_total_seconds"])
            if not {"artifact_persistence_seconds", "maps", "oracle_execution"} <= set(
                row
            ):
                g["extended_missing"] = True
                continue
            g["persist"].append(row["artifact_persistence_seconds"])
            g["extended_peak"] = max(
                g["extended_peak"], row["oracle_execution"]["peak_rss_kib"]
            )
            for item in row["maps"]:
                g["maps"].append(item["execution"]["wall_seconds"])
                g["extended_peak"] = max(
                    g["extended_peak"], item["execution"]["peak_rss_kib"]
                )
    stats = {}
    for key, g in groups.items():
        s = dict(logical=g["logical"], physical=g["physical"])
        if g["physical"]:
            s.update(
                stage_median_seconds={
                    stage: statistics.median(g["stage"][stage]) for stage in STAGES
                },
                peak_native_rss_kib=g["basic_peak"],
                bundle_bytes=g["bundle_bytes"],
            )
            if g["all_total"]:
                if g["extended_missing"]:
                    raise ValueError("native census extended timing fields missing")
                s.update(
                    total_median_seconds=statistics.median(g["total"]),
                    total_sum_seconds=sum(g["total"]),
                    persistence_sum_seconds=sum(g["persist"]),
                    map_sum_seconds=sum(g["maps"]),
                    peak_native_rss_kib=max(g["basic_peak"], g["extended_peak"]),
                )
        stats[key] = s
    expected = [
        job["kind"]["view"] for job in json.loads((out / "manifest.json").read_text())
    ]
    admitted = len(completed) == len(set(completed)) and set(completed) == set(expected)
    result = dict(
        schema="e36-data-census-v1",
        fits=0,
        completed_ladders=len(files),
        logical_rows=logical,
        output_receipts=files,
        groups=stats,
        admitted=admitted,
        declared_ladders=len(expected),
    )
    if admitted and (Path(root)/'scheduling-v7/amendment.json').exists():
        from e36_native_ledger_admit import admit
        result['native_publication']=admit(root,name,completed,files)
    with (out / "census.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print(
        canonical(
            {
                k: result[k]
                for k in [
                    "admitted",
                    "completed_ladders",
                    "declared_ladders",
                    "logical_rows",
                    "fits",
                ]
            }
        )
    )
    if not admitted:
        raise ValueError("native ledger outputs do not complete the declared matrix")
