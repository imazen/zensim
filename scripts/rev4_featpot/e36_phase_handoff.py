"""Coordinate one exclusively owned whole native phase; never shard or retry jobs.

The native worker owns execution/attempts/ledger/blobs. This adapter binds its
container and transfers its original evidence with source timestamps intact.
"""

import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import tarfile


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def remote(host, args, **kwargs):
    return subprocess.run(["ssh", "-T", host, shlex.join(args)], check=True, **kwargs)


def inspect_bound(binding):
    records = json.loads(
        remote(
            binding["host"],
            ["docker", "inspect", binding["container_id"]],
            capture_output=True,
            text=True,
        ).stdout
    )
    if len(records) != 1:
        raise ValueError("native phase container identity missing")
    r = records[0]
    if (
        r["Id"] != binding["container_id"]
        or r["Image"] != binding["resolved_image_id"]
        or r["Config"] != binding["config"]
        or r["HostConfig"] != binding["host_config"]
    ):
        raise ValueError(
            "native phase container/image/invocation/resource binding differs"
        )
    return r


def restore_inventory(directory, receipt):
    """Verify native files and restore measured source inode timestamps."""
    expected = {r["relative"]: r for r in receipt["native_files"]}
    if len(expected) != len(receipt["native_files"]):
        raise ValueError("duplicate native inventory path")
    actual = {
        str(p.relative_to(directory))
        for p in directory.rglob("*")
        if p.is_file() and p != directory / "phase-execution.json"
    }
    if actual != set(expected):
        raise ValueError("native phase file inventory differs")
    for name, row in expected.items():
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe native evidence path")
        p = directory / relative
        if p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            raise ValueError("native phase evidence changed")
        # These are the measured source timestamps, not new completion times.
        os.utime(p, ns=(row["mtime_ns"], row["mtime_ns"]))


def harvest(binding_path, out):
    binding_path = Path(binding_path)
    binding = json.loads(binding_path.read_text())
    out = Path(out)
    out.mkdir()
    state = inspect_bound(binding)["State"]
    if state["Status"] != "exited":
        raise ValueError("harvest requires the exact stopped native container")
    directory = Path(binding["manifest"]["path"]).parent
    archive = out / "native-output.tar"
    with archive.open("xb") as stream, (out / "copy.log").open("x") as log:
        remote(
            binding["host"],
            ["docker", "cp", binding["container_id"] + ":" + str(directory), "-"],
            stdout=stream,
            stderr=log,
        )
    extracted = out / "extracted"
    extracted.mkdir()
    with tarfile.open(archive) as source:
        seen = set()
        for member in source.getmembers():
            p = Path(member.name)
            if (
                p.is_absolute()
                or ".." in p.parts
                or (not member.isfile() and not member.isdir())
                or member.name in seen
            ):
                raise ValueError("unsafe/duplicate native harvest member")
            seen.add(member.name)
        source.extractall(extracted, filter="data")
    found = [p for p in extracted.rglob("phase-execution.json")]
    if len(found) != 1:
        raise ValueError("native phase terminal receipt missing or ambiguous")
    execution = json.loads(found[0].read_text())
    result_dir = found[0].parent
    files_path = Path(binding["files_manifest_path"])
    if sha(files_path) != binding["files_manifest_sha256"]:
        raise ValueError("native phase frozen file manifest changed")
    files = json.loads(files_path.read_text())
    expected_worker = next(
        r["sha256"]
        for r in files["files"]
        if r["destination"] == files["root"] + "/bin/zenfleet-worker"
    )
    if (
        execution["schema"] != "e36-whole-native-phase-execution-v1"
        or execution["phase"] != binding["phase"]
        or execution["manifest_sha256"] != binding["manifest"]["sha256"]
        or execution["files_manifest_sha256"] != binding["files_manifest_sha256"]
        or execution["effective_clock_sha256"] != binding["effective_clock"]["sha256"]
        or execution["native_worker_sha256"] != expected_worker
        or execution["fits"] != 0
        or execution["optimizer_steps"] != 0
        or execution["returncode"] != state["ExitCode"]
    ):
        raise ValueError("native phase execution/container/input identity differs")
    restore_inventory(result_dir, execution)
    write(
        out / "harvest.json",
        dict(
            schema="e36-native-phase-harvest-v1",
            binding=dict(path=str(binding_path), sha256=sha(binding_path)),
            execution=dict(path=str(found[0]), sha256=sha(found[0])),
            archive_sha256=sha(archive),
            source_timestamps_restored=True,
            result_directory=str(result_dir),
            returncode=execution["returncode"],
            fits=0,
        ),
    )
    return result_dir, execution


def rows_by_view(directory):
    result = {}
    for path in (Path(directory) / "blobs").rglob("*"):
        if not path.is_file():
            continue
        raw = path.read_bytes()
        doc = json.loads(gzip.decompress(raw))
        if (
            not doc["complete"]
            or doc["fits"] != 0
            or doc["labels_read"] is not False
            or len(doc["rows"]) != 1
        ):
            raise ValueError("complete one-cell first-producer evidence required")
        key = json.dumps(doc["view"], sort_keys=True, separators=(",", ":"))
        if key in result:
            raise ValueError("duplicate first-producer native view")
        result[key] = doc["rows"][0]
    return result


def projection(row):
    # Retain every native scalar variant; omit only operational timing/paths.
    native = [
        {
            k: r[k]
            for k in [
                "metric",
                "score",
                "scores",
                "reference_pixels_sha256",
                "distorted_pixels_sha256",
            ]
            if k in r
        }
        for r in row["native_scores"]
    ]
    f32 = struct.pack("<" + str(len(row["features"])) + "f", *row["features"])

    def bits(value):
        if isinstance(value, float):
            return dict(f64le=struct.pack("<d", value).hex())
        if isinstance(value, dict):
            return {k: bits(v) for k, v in value.items()}
        if isinstance(value, list):
            return [bits(v) for v in value]
        return value

    return dict(
        identity={
            k: row[k]
            for k in [
                "codec",
                "q",
                "reference_sha256",
                "reference_pixels_sha256",
                "distorted_pixels_sha256",
                "encoded_sha256",
                "encoded_bytes",
                "feature_ids",
                "feature_layout_sha256",
                "feature_tool_sha256",
                "tool_sha256",
                "knobs",
                "width",
                "height",
            ]
        },
        features_f32le_sha256=hashlib.sha256(f32).hexdigest(),
        native_scores=bits(native),
        maps={
            m["metric"]: dict(
                sha256=m["sha256"],
                bytes=m["bytes"],
                pixel_pair_metric_key=m["pixel_pair_metric_key"],
            )
            for m in row["maps"]
        },
    )


def parity(root, harvest_directory, out):
    root = Path(root)
    out = Path(out)
    golden = root / "runs/first-producer-v6"
    other = Path(harvest_directory)
    a = rows_by_view(golden)
    b = rows_by_view(other)
    if len(a) != 5 or set(a) != set(b):
        raise ValueError("exact frozen five-producer parity inventory required")
    comparisons = []
    for view in sorted(a):
        left = projection(a[view])
        right = projection(b[view])
        same = left == right
        comparisons.append(dict(view=view, exact=same, dev=left, tower=right))
    result = dict(
        schema="e36-bounded-native-host-parity-v1",
        golden_manifest=dict(
            path=str(golden / "manifest.json"), sha256=sha(golden / "manifest.json")
        ),
        native_execution=dict(
            path=str(other / "phase-execution.json"),
            sha256=sha(other / "phase-execution.json"),
        ),
        admitted=all(c["exact"] for c in comparisons),
        comparisons=comparisons,
        views=5,
        bounded_first_producer_only=True,
        universal_cross_host_parity_proven=False,
        fits=0,
    )
    write(out, result)
    if not result["admitted"]:
        raise ValueError(
            "bounded host parity differs; retain the finding, do not scale tower"
        )
    return result


def supervisor_lock(supervisor, paper_lock, operation, log):
    command = Path(f"/proc/{supervisor}/cmdline").read_bytes().split(b"\0")
    if (
        Path(command[0].decode()).name != "flock"
        or Path(f"/proc/{supervisor}/fd/3").resolve() != Path(paper_lock).resolve()
    ):
        raise ValueError("original paper-lock supervisor descriptor differs")
    with Path(log).open("x") as output:
        subprocess.run(
            [
                "sudo",
                "-n",
                "gdb",
                "-q",
                "-n",
                "-batch",
                "-p",
                str(supervisor),
                "-ex",
                "set $rc=(int)flock(3," + str(operation) + ")",
                "-ex",
                'python assert int(gdb.parse_and_eval("$rc"))==0, "paper flock operation failed"',
                "-ex",
                "detach",
            ],
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )


def wait_and_harvest(root, name, deadline, handoff):
    root = Path(root)
    handoff = Path(handoff)
    spec = json.loads(handoff.read_text())
    binding_path = Path(spec["container"]["path"])
    if (
        sha(binding_path) != spec["container"]["sha256"]
        or spec["schema"] != "e36-whole-native-phase-owner-v1"
        or spec["run"] != name
    ):
        raise ValueError("whole native phase ownership changed")
    binding = json.loads(binding_path.read_text())
    plan = json.loads((root / "bulk-plan-v6.json").read_text())
    clock = Path(binding["effective_clock"]["path"])
    if (
        name != plan["phases"]["additions"]["run"]
        or binding["phase"] != "additions"
        or binding["manifest"]["sha256"]
        != plan["phases"]["additions"]["manifest_sha256"]
        or sha(clock) != binding["effective_clock"]["sha256"]
        or json.loads(clock.read_text())["deadline_unix"] != deadline
    ):
        raise ValueError("native additions phase/clock binding differs")
    state = inspect_bound(binding)["State"]
    out = root / "tower-additions-harvest-v1"
    if state["Status"] == "created":
        write(
            root / "tower-additions-start-failed.json",
            dict(
                container_id=binding["container_id"],
                state=state,
                admitted=False,
                fits=0,
            ),
        )
        return
    if state["Status"] != "exited":
        # No dev computation runs while waiting on this remote native phase.
        # Release the original allocation; reacquire it before copying/census.
        supervisor_lock(
            spec["supervisor_pid"],
            spec["paper_lock"],
            8,
            root / "tower-phase-lock-release.log",
        )
        try:
            with (root / "tower-native-wait.log").open("x") as log:
                remote(
                    binding["host"],
                    ["docker", "wait", binding["container_id"]],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
        finally:
            supervisor_lock(
                spec["supervisor_pid"],
                spec["paper_lock"],
                2,
                root / "tower-phase-lock-reacquire.log",
            )
    directory, execution = harvest(binding_path, out)
    dest = root / "runs" / name
    for record in execution["native_files"]:
        relative = Path(record["relative"])
        source = directory / relative
        target = dest / relative
        if target.exists():
            if sha(target) != record["sha256"]:
                raise ValueError(
                    "native phase install conflicts with retained dev evidence"
                )
            if relative == Path("manifest.json"):
                continue
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            os.utime(target, ns=(record["mtime_ns"], record["mtime_ns"]))
        if (
            sha(target) != record["sha256"]
            or target.stat().st_mtime_ns != record["mtime_ns"]
        ):
            raise ValueError("native phase installed bytes/timestamps differ")
    target = dest / "phase-execution.json"
    shutil.copy2(directory / "phase-execution.json", target)
    write(
        root / "tower-additions-installed.json",
        dict(
            schema="e36-native-phase-installed-v1",
            handoff_sha256=sha(handoff),
            native_execution_sha256=sha(target),
            harvest_sha256=sha(out / "harvest.json"),
            output_source_timestamps_preserved=True,
            native_files=len(execution["native_files"]),
            retained_original_dev_input_manifest_timestamp=True,
            fits=0,
        ),
    )


def admit_installed(root, clock_path, plan):
    """Replay the local immutable whole-phase proof before using tower rows."""
    root = Path(root)
    handoff_path = root / "TOWER_PHASE_HANDOFF.json"
    handoff = json.loads(handoff_path.read_text())
    evidence = [dict(path=str(handoff_path), sha256=sha(handoff_path))]

    def bound(entry):
        p = Path(entry["path"])
        if sha(p) != entry["sha256"]:
            raise ValueError("native whole-phase proof changed")
        evidence.append(entry)
        return json.loads(p.read_text())

    binding = bound(handoff["container"])
    parity_proof = bound(handoff["parity"])
    bound(handoff["live_guard"])
    adapter = handoff["adapter"]
    if sha(adapter["path"]) != adapter["sha256"]:
        raise ValueError("native whole-phase adapter changed")
    evidence.append(adapter)
    phase = plan["phases"]["additions"]
    if (
        handoff["schema"] != "e36-whole-native-phase-owner-v1"
        or handoff["run"] != phase["run"]
        or binding["phase"] != "additions"
        or binding["manifest"]["sha256"] != phase["manifest_sha256"]
        or binding["effective_clock"]["sha256"] != sha(clock_path)
        or parity_proof["admitted"] is not True
        or parity_proof["views"] != 5
    ):
        raise ValueError("native phase ownership/parity/clock differs")
    first = bound(parity_proof["native_execution"])
    files = bound(
        dict(
            path=binding["files_manifest_path"], sha256=binding["files_manifest_sha256"]
        )
    )
    installed_path = root / "tower-additions-installed.json"
    installed = json.loads(installed_path.read_text())
    execution_path = root / "runs" / phase["run"] / "phase-execution.json"
    execution = json.loads(execution_path.read_text())
    harvest_path = root / "tower-additions-harvest-v1/harvest.json"
    if (
        installed["handoff_sha256"] != sha(handoff_path)
        or installed["native_execution_sha256"] != sha(execution_path)
        or installed["harvest_sha256"] != sha(harvest_path)
        or installed["output_source_timestamps_preserved"] is not True
        or first["files_manifest_sha256"] != binding["files_manifest_sha256"]
        or execution["files_manifest_sha256"] != binding["files_manifest_sha256"]
        or execution["effective_clock_sha256"] != sha(clock_path)
        or execution["manifest_sha256"] != phase["manifest_sha256"]
        or execution["returncode"] != 0
        or execution["fits"] != 0
        or execution["optimizer_steps"] != 0
    ):
        raise ValueError("native transferred execution proof differs")
    worker = next(
        r["sha256"]
        for r in files["files"]
        if r["destination"] == files["root"] + "/bin/zenfleet-worker"
    )
    if execution["native_worker_sha256"] != worker:
        raise ValueError("native transferred worker differs")
    for record in execution["native_files"]:
        relative = Path(record["relative"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe installed native evidence path")
        p = execution_path.parent / relative
        if (
            sha(p) != record["sha256"]
            or p.stat().st_size != record["bytes"]
            or (
                relative != Path("manifest.json")
                and p.stat().st_mtime_ns != record["mtime_ns"]
            )
        ):
            raise ValueError("native transferred bytes/source timestamps changed")
    evidence.extend(
        dict(path=str(p), sha256=sha(p))
        for p in [installed_path, execution_path, harvest_path]
    )
    return evidence


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    h = sub.add_parser("harvest")
    h.add_argument("--binding", type=Path, required=True)
    h.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("parity")
    for n in ["root", "harvest", "out"]:
        c.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    if a.mode == "harvest":
        harvest(a.binding, a.out)
    else:
        parity(a.root, a.harvest, a.out)
