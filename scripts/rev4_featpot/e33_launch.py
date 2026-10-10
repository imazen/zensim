"""E33 launch and harvest owner (registration section 10.1; coordinator authorization 2026-10-10).

Launches exactly the frozen packet (`PACKET.json` pinned below) through the existing fleet: live jobset caps
(`/var/tmp/fitv2/jobset_caps.json`), the job store and `/var/tmp/fitv2/fleet_queue`, which the host fillers read.
Jobsets go in the registered sequence (control, A, C, full); each append requires the previous jobset queued.
`placements` reads the filler log after an append (the V40 empty-hosts lesson: a queue line is not a placement).

`harvest` ports the reviewed V40 incremental driver (`v40r4-2026-10-08/harvest_driver_v40.py`) to the packet's
program, inspector and manifests: it verifies every DONE cell with the program archive's own `harvest_fit_cells`
and installs it under the cell's declared destination. It never schedules work. Wall caps are enforced per cell by
the program's `fit_cell_exec`, not here.
"""

import argparse
import collections
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

from shippath10_launch import sha
from v40_launch import placement

PACKET_SHA = "3c96351e51ce6c8e0d214d53b84fcfba83d2f7286adcdd8e975d88340d249693"
ORDER = ("control", "a", "c", "full")
FITV2 = Path("/var/tmp/fitv2")
POT = Path("/var/tmp/rev4-featpot")
CTL = Path("/var/tmp/fleet-fits/fleetbin/zenfleet-ctl")
TOOLS = ("harvest_fit_cells.py", "qualified_fit_contract.py", "fit_paths.py")


def now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def jobset(arm):
    return f"fite33-{arm}-20261010"


def packet(bundle):
    """The frozen packet, refused unless PACKET.json and every file it lists are byte-identical to the freeze."""
    if sha(bundle / "PACKET.json") != PACKET_SHA:
        raise ValueError("PACKET.json differs from the authorized freeze")
    doc = json.loads((bundle / "PACKET.json").read_text())
    for name, pin in doc["files"].items():
        if sha(bundle / name) != pin:
            raise ValueError(f"packet file changed: {name}")
    if sha(bundle / "e33-fit-data.tar.gz") != doc["data_sha"]:
        raise ValueError("data archive differs from the packet")
    return doc


def runtime(bundle, doc):
    """Launch runtime beside the packet: verifier tools from the program archive and the pinned inspector."""
    rt = bundle.parent / "launch"
    tools = rt / "committed-tools"
    tools.mkdir(parents=True, exist_ok=True)
    with tarfile.open(bundle / "program.tar.gz") as archive:
        for name in TOOLS:
            data = archive.extractfile(name).read()
            path = tools / name
            if not path.exists():
                path.write_bytes(data)
            if path.read_bytes() != data:
                raise ValueError(f"committed verifier differs from the program: {name}")
    meta = json.loads((bundle / "build-meta.packer-input.json").read_text())
    inspector = bundle.parent / "bin-final2/inspect_qualified_checkpoint"
    if sha(inspector) != meta["binary_mix"]["inspect_qualified_checkpoint"]["sha256"]:
        raise ValueError("inspector differs from the program's binary pin")
    return rt, tools, inspector


def fleet_hosts(bundle):
    return json.loads((bundle / "PLACEMENT_REHEARSAL.json").read_text())["fleet_hosts"]


def merge_caps(bundle, live=FITV2 / "jobset_caps.json"):
    """Back up the live caps, then add the four E33 entries (hosts listed); never replace another entry."""
    packet(bundle)
    mine = json.loads((bundle / "jobset_caps.json").read_text())
    current = json.loads(live.read_text())
    hosts = fleet_hosts(bundle)
    for js, entry in mine.items():
        placement(entry, hosts)
        if js in current and current[js] != entry:
            raise ValueError(f"live caps already hold a different {js}")
    backup = live.with_name(f"{live.name}.bak-{now().replace(':', '')}-pree33")
    if backup.exists():
        raise ValueError(f"backup exists: {backup}")
    backup.write_bytes(live.read_bytes())
    staging = live.with_name(live.name + ".e33.new")
    staging.write_text(json.dumps({**current, **mine}, indent=2) + "\n")
    os.replace(staging, live)
    return dict(backup=str(backup), added=sorted(set(mine) - set(current)))


def expected_caps(bundle, js):
    """The packet's caps entry plus coordinator amendments recorded in launch/CAPS_AMENDMENTS.json.

    An amendment may only add hosts and restate the reason; memory and existing host slots stay as frozen.
    """
    entry = json.loads(json.dumps(json.loads((bundle / "jobset_caps.json").read_text())[js]))
    path = bundle.parent / "launch" / "CAPS_AMENDMENTS.json"
    if not path.exists():
        return entry
    for amendment in json.loads(path.read_text())["amendments"]:
        change = amendment["jobsets"].get(js)
        if change is None:
            continue
        if set(change) - {"added_hosts", "reason"} or set(change["added_hosts"]) & set(entry["hosts"]):
            raise ValueError(f"amendment {amendment['id']} may only add hosts and restate the reason")
        if any(type(v) is not int or v <= 0 for v in change["added_hosts"].values()):
            raise ValueError(f"amendment {amendment['id']}: positive integer slots required")
        entry["hosts"].update(change["added_hosts"])
        entry["reason"] = change.get("reason", entry.get("reason"))
    return entry


def queued(queue):
    return [line.split()[0] for line in queue.read_text().splitlines() if line.split()]


def launch(bundle, arm, queue=FITV2 / "fleet_queue", live=FITV2 / "jobset_caps.json"):
    doc = packet(bundle)
    js = jobset(arm)
    if js not in doc["jobsets"]:
        raise ValueError(f"{js} is not in the packet")
    actual = subprocess.check_output(["docker", "image", "inspect", "-f", "{{.Id}}", doc["image"]], text=True).strip()
    if actual != doc["image_id"]:
        raise ValueError(f"local image id {actual} differs from the packet's {doc['image_id']}")
    entry = json.loads(live.read_text()).get(js)
    if entry != expected_caps(bundle, js):
        raise ValueError("live jobset cap differs from the packet entry plus recorded amendments (run caps first)")
    placement(entry, fleet_hosts(bundle))
    names = queued(queue)
    if js in names:
        raise ValueError(f"{js} already queued")
    earlier = ORDER[: ORDER.index(arm)]
    if any(jobset(a) not in names for a in earlier):
        raise ValueError("registered sequence: queue the earlier E33 jobsets first")
    rt, _, _ = runtime(bundle, doc)
    record = dict(jobset=js, packet_sha=PACKET_SHA, image=doc["image"], image_id=actual, started=now())
    push = subprocess.run(["docker", "push", doc["image"]], capture_output=True, text=True, check=True)
    record["push_tail"] = push.stdout.strip().splitlines()[-1:]
    control = rt / f"control-{js}.json"
    control.write_text(json.dumps(dict(paused=False, drain=False,
                                       note="E33; coordinator authorization 2026-10-10")) + "\n")
    subprocess.run(["bash", "-c", """set -euo pipefail
. ~/.config/zen/s3env.sh >/dev/null 2>&1
s5cmd --endpoint-url "$EP" cp "$1" "s3://zentrain/jobs/$2/manifest.json"
s5cmd --endpoint-url "$EP" cp "$3" "s3://zentrain/jobs/$2/control.json"
s5cmd --endpoint-url "$EP" cp "$4" "s3://zentrain/jobs/$2/inputs/$5"
""", "--", str(next(bundle.glob(f"fit-manifest-{js}.json"))), js, str(control),
                    str(bundle / "e33-fit-data.tar.gz"), doc["data_sha"]], check=True)
    log = FITV2 / "host_filler.log"
    record["filler_log_offset"] = log.stat().st_size if log.exists() else 0
    queue.with_name(f"fleet_queue.{js}.before").write_bytes(queue.read_bytes())
    staging = queue.with_name(f"fleet_queue.{js}.new")
    manifest = bundle / f"fit-manifest-{js}.json"
    staging.write_text(queue.read_text().rstrip() + f"\n{js} {manifest} {doc['image']}\n")
    os.replace(staging, queue)
    record["queued"] = now()
    (rt / f"LAUNCH_RECORD-{js}.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def placements(bundle, arm):
    """Filler placements of this jobset logged since its queue append."""
    js = jobset(arm)
    record = json.loads((bundle.parent / "launch" / f"LAUNCH_RECORD-{js}.json").read_text())
    with (FITV2 / "host_filler.log").open("rb") as f:
        f.seek(record["filler_log_offset"])
        text = f.read().decode(errors="replace")
    started = [line for line in text.splitlines() if f" -> {js} as " in line]
    hosts = collections.Counter(line.split(" -> ")[0] for line in started)
    envelopes = [line for line in text.splitlines() if line.startswith(f"jobset envelope {js} ")]
    return dict(jobset=js, queued=record["queued"], checked=now(), started=len(started), by_host=dict(hosts),
                envelope_lines=len(envelopes), refusals=sum("refus" in line.lower() for line in text.splitlines()
                                                             if js in line))


def installed(hfc, c):
    k = c["kind"]
    d = POT / hfc.blob_root(k) / c["cell"]["image_path"]
    try:
        receipt = json.loads((d / "fleet_receipt.json").read_text())
        result = json.loads((d / "result.json").read_text())
        argv = k["argv"]
        dest = Path(argv[argv.index("--dest") + 1])
        bake = d / Path(result["selected_bake"]).relative_to(dest)
        return (all(receipt[x] == k[x] for x in ("program_sha", "data_sha", "argv_sha"))
                and result.get("execution_contract") != "local-smoke"
                and receipt["files"]["result.json"] == sha(d / "result.json")
                and result["selected_bake_sha256"] == sha(bake))
    except (OSError, ValueError, KeyError):
        return False


def harvest(bundle, arm, install=False, require_all=False, status_only=False):
    doc = packet(bundle)
    js = jobset(arm)
    manifest = bundle / f"fit-manifest-{js}.json"
    _, tools, inspector = runtime(bundle, doc)
    sys.path.insert(0, str(tools))
    import harvest_fit_cells as hfc
    jobs = json.loads(manifest.read_text())
    for c in jobs:
        k, name = c["kind"], c["cell"]["image_path"]
        argv = k["argv"]
        if (k["program_sha"] != doc["program_sha"] or k["data_sha"] != doc["data_sha"]
                or "--local-smoke-budget" in argv or "--strict-admission" not in argv or "--train-only" not in argv):
            raise ValueError("requires a registered full-budget strict E33 cell")
        if POT / hfc.blob_root(k) / name != Path(argv[argv.index("--dest") + 1]):
            raise ValueError("manifest destination and harvest install root differ")
    work = FITV2 / "harvest-e33" / js
    status = FITV2 / "status-e33" / js
    work.mkdir(parents=True, exist_ok=True)
    status.mkdir(parents=True, exist_ok=True)
    with (work / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        endpoint = os.environ["EP"]
        subprocess.run(["s5cmd", "--endpoint-url", endpoint, "sync", f"s3://zentrain/jobs/{js}/ledger/*",
                        str(status) + "/"], check=True, capture_output=True)
        ids = subprocess.check_output([str(CTL), "ids", "--manifest", str(manifest)], text=True)
        (work / "full-ids.tsv").write_text(ids)
        pairs = hfc.read_ids(work / "full-ids.tsv")
        if len(pairs) != len(jobs) or any(name != c["cell"]["image_path"] for (_, name), c in zip(pairs, jobs)):
            raise ValueError("canonical manifest/IDs mismatch")
        import pyarrow as pa
        import pyarrow.parquet as pq
        relevant = {jid for jid, _ in pairs}
        rows = [r for f in sorted(status.glob("*.parquet")) for r in pq.read_table(f).to_pylist()
                if r["job_id"] in relevant]
        latest, done = {}, set()
        for r in rows:
            if r["status"] == "done" and r.get("output_sha"):
                done.add(r["job_id"])
            if r["job_id"] not in latest or r["ts"] > latest[r["job_id"]]["ts"]:
                latest[r["job_id"]] = r
        counts = collections.Counter("done" if j in done else latest[j]["status"] for j in latest)
        report = dict(jobset=js, ledger=dict(counts), done=len(done), total=len(jobs),
                      installed=sum(installed(hfc, c) for c in jobs), time=now())
        print(json.dumps(report), flush=True)
        if status_only:
            return report
        if require_all and len(done) != len(jobs):
            raise ValueError("all registered cells must be DONE")
        chosen = [(pair, c) for pair, c in zip(pairs, jobs) if pair[0] in done and not installed(hfc, c)]
        if chosen:
            (work / "manifest.json").write_text(json.dumps([c for _, c in chosen]) + "\n")
            pq.write_table(pa.Table.from_pylist(rows), work / "ledger.parquet")
            (work / "ids.tsv").write_text("".join(f"{i}\t{jid}\t{name}\n"
                                                  for i, ((jid, name), _) in enumerate(chosen)))
            cmd = [sys.executable, str(tools / "harvest_fit_cells.py"), "--manifest", str(work / "manifest.json"),
                   "--ids", str(work / "ids.tsv"), "--ledger", str(work / "ledger.parquet"),
                   "--blobs-prefix", f"s3://zentrain/jobs/{js}/blobs", "--endpoint", endpoint,
                   "--scratch", str(work / "scratch"), "--program-archive", str(bundle / "program.tar.gz"),
                   "--checkpoint-inspector", str(inspector)]
            if install:
                cmd += ["--install", "--rescue-root", str(FITV2 / "original-era-e33" / js)]
            subprocess.run(cmd, check=True)
        if install and require_all and not all(installed(hfc, c) for c in jobs):
            raise ValueError("installed cell identity/checkpoint check failed")
        return report


def open_cells(js, manifest):
    """Unclaimed cells, as the filler's picker counts them: no claim object and no done/poison ledger row."""
    import glob
    import hashlib
    import pyarrow.parquet as pq
    endpoint = os.environ["EP"]
    ids = subprocess.run([str(CTL), "ids", "--manifest", str(manifest)], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    chunks = {hashlib.sha256((line.split("\t")[1] + "\n").encode()).hexdigest() for line in ids if line.strip()}
    status = FITV2 / "status-e33" / js
    status.mkdir(parents=True, exist_ok=True)
    subprocess.run(["s5cmd", "--endpoint-url", endpoint, "sync", f"s3://zentrain/jobs/{js}/ledger/*", str(status) + "/"],
                   capture_output=True)
    finished = {hashlib.sha256((r["job_id"] + "\n").encode()).hexdigest() for f in glob.glob(str(status / "*.parquet"))
                for r in pq.read_table(f, columns=["job_id", "status"]).to_pylist() if r["status"] in ("done", "poison")}
    ls = subprocess.run(["s5cmd", "--endpoint-url", endpoint, "ls", f"s3://zentrain/jobs/{js}/claims/*"],
                        capture_output=True, text=True).stdout.splitlines()
    claimed = {line.split()[-1][len("chunk-"):] for line in ls if line.strip() and line.split()[-1].startswith("chunk-")}
    return len(chunks - finished - claimed)


def tower_workers(js, host="root@tower"):
    """Running containers of this jobset on tower (the host whose standing 40g cap bounds overlap)."""
    out = subprocess.run(["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", host,
                          "docker ps -q --filter name=zen-score- | xargs -r docker inspect -f "
                          "'{{range .Config.Env}}{{println .}}{{end}}'"], capture_output=True, text=True, check=True)
    return sum(line == f"ZEN_RUN=jobs/{js}" for line in out.stdout.splitlines())


def advance(bundle, queue=FITV2 / "fleet_queue"):
    """Append the next E33 jobset once the previous has no unclaimed cells and at most one tower worker.

    That bounds tower at 6 E33 cells (36g of 6g caps) under its standing 40g cap while one jobset drains.
    """
    names = queued(queue)
    pending = [a for a in ORDER if jobset(a) not in names]
    if not pending:
        return dict(action="none", reason="all four E33 jobsets queued", time=now())
    arm = pending[0]
    prev = ORDER[ORDER.index(arm) - 1]
    js = jobset(prev)
    unclaimed = open_cells(js, bundle / f"fit-manifest-{js}.json")
    tower = tower_workers(js)
    if unclaimed or tower > 1:
        return dict(action="wait", next=arm, previous=js, unclaimed=unclaimed, tower_workers=tower, time=now())
    record = launch(bundle, arm, queue)
    return dict(action="launched", next=arm, previous=js, unclaimed=unclaimed, tower_workers=tower,
                queued=record["queued"], time=now())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("verify", "caps", "launch", "placements", "harvest", "advance"))
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--arm", choices=ORDER)
    p.add_argument("--install", action="store_true")
    p.add_argument("--require-all", action="store_true")
    p.add_argument("--status", action="store_true")
    a = p.parse_args()
    if a.mode in ("launch", "placements", "harvest") and not a.arm:
        p.error(f"{a.mode} requires --arm")
    if a.mode == "verify":
        doc = packet(a.packet)
        runtime(a.packet, doc)
        out = dict(packet_sha=PACKET_SHA, program_sha=doc["program_sha"], image=doc["image"], jobsets=doc["jobsets"])
    elif a.mode == "caps":
        out = merge_caps(a.packet)
    elif a.mode == "launch":
        out = launch(a.packet, a.arm)
    elif a.mode == "placements":
        out = placements(a.packet, a.arm)
    elif a.mode == "advance":
        packet(a.packet)
        out = advance(a.packet)
    else:
        out = harvest(a.packet, a.arm, a.install, a.require_all, a.status)
    print(json.dumps(out))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"E33 REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)
