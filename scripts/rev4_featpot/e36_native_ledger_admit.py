"""Require complete, checked native publication for this max-attempts-one run."""

import json
from pathlib import Path
import subprocess

import pyarrow.parquet as pq

from e36_data import canonical
from e36_store import sha


def admit(root, name, completed, files):
    root = Path(root)
    directory = root / "runs" / name
    plan = json.loads((root / "bulk-plan-v6.json").read_text())
    phase = next(k for k, v in plan["phases"].items() if v["run"] == name)
    amendment_path = root / "scheduling-v7/amendment.json"
    amendment = json.loads(amendment_path.read_text())
    entry = amendment["manifests"][phase]
    original = Path(entry["original"]["path"])
    scheduled = Path(entry["path"])
    if (
        sha(original) != entry["original"]["sha256"]
        or sha(scheduled) != entry["sha256"]
        or sha(amendment["worker"]["path"]) != amendment["worker"]["sha256"]
    ):
        raise ValueError("native scheduling/publication inputs changed")
    old = json.loads(original.read_text())
    new = json.loads(scheduled.read_text())
    if len(old) != len(new) or any(
        {k: v for k, v in a.items() if k != "hint"}
        != {k: v for k, v in b.items() if k != "hint"}
        for a, b in zip(old, new)
    ):
        raise ValueError("scheduling amendment changed a scientific job")
    ids = []
    for manifest, label in [(original, "original"), (scheduled, "scheduled")]:
        path = root / "scheduling-v7" / (phase + "." + label + ".ids.tsv")
        if not path.exists():
            with path.open("x") as f:
                subprocess.run(
                    [
                        str(root / "bin/zenfleet-ctl"),
                        "ids",
                        "--manifest",
                        str(manifest),
                    ],
                    stdout=f,
                    check=True,
                )
        ids.append(path)
    if ids[0].read_bytes() != ids[1].read_bytes():
        raise ValueError("native job identities changed")
    native_ids = [line.split("\t")[1] for line in ids[0].read_text().splitlines()]
    if len(native_ids) != len(old) or len(set(native_ids)) != len(native_ids):
        raise ValueError("native identity inventory differs")
    jobs = {job_id: job for job_id, job in zip(native_ids, old)}
    by_view = {job["kind"]["view"]: job_id for job_id, job in jobs.items()}
    published = {}
    ledgers = []
    for path in sorted(directory.glob("ledger*.parquet")):
        for batch in pq.ParquetFile(path).iter_batches(
            columns=["job_id", "status", "attempts", "output_sha", "kind_json"],
            batch_size=256,
        ):
            for row in batch.to_pylist():
                job_id = row["job_id"]
                if (
                    job_id not in jobs
                    or row["status"] != "done"
                    or row["attempts"] != 1
                    or not row["output_sha"]
                    or json.loads(row["kind_json"]) != jobs[job_id]["kind"]
                ):
                    raise ValueError(
                        "native max-attempts-one publication is failed, missing or mismatched"
                    )
                prior = published.setdefault(job_id, row["output_sha"])
                if prior != row["output_sha"]:
                    raise ValueError("conflicting native publication for one job")
        ledgers.append(dict(path=str(path), sha256=sha(path)))
    if set(published) != set(jobs):
        raise ValueError("complete native ledger publication is required")
    if len(completed) != len(files):
        raise ValueError("native census output alignment changed")
    for view, record in zip(completed, files):
        if published[by_view[view]] != record["sha256"]:
            raise ValueError("census output differs from native durable publication")
    log = root / "GENERATION-v7.log"
    if any(
        "ledger write" in line and "failed" in line
        for line in log.read_text().splitlines()
    ):
        raise ValueError(
            "native publication error requires explicit reconciliation before admission"
        )

    def bind(p):
        return dict(path=str(p), sha256=sha(p))

    proof = dict(
        schema="e36-native-publication-admission-v1",
        admitted=True,
        jobs=len(jobs),
        published=len(published),
        native_ids=bind(ids[0]),
        scheduled_ids=bind(ids[1]),
        amendment=bind(amendment_path),
        ledgers=ledgers,
        source_sha256=sha(__file__),
        fits=0,
        producer_attempts_per_job=1,
        census_output_inventory_sha256=__import__("hashlib")
        .sha256(canonical(files).encode())
        .hexdigest(),
    )
    path = root / "scheduling-v7" / (phase + "-native-publication.json")
    with path.open("x") as f:
        json.dump(proof, f, indent=2)
        f.write("\n")
    return bind(path)
