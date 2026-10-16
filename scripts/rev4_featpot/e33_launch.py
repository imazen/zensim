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
import signal
import subprocess
import sys
import tarfile
import time
import traceback

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
    return arm if arm.startswith(("fite34-", "fitsteer-")) else f"fite33-{arm}-20261010"


def steer_plan(bundle, doc):
    plan = json.loads((bundle / 'STEER_JOB_PLAN.json').read_text())
    if (plan.get('schema') != 'steer-job-plan-v1' or len(plan['order']) != 8
            or len(set(plan['order'])) != 8 or set(plan['order']) != set(doc['jobsets'])
            or set(plan['unconditional']) != set(plan['order'])
            or sum(doc['jobsets'].values()) != 172 or plan.get('total_executed_cells') != 172):
        raise ValueError('fixed STEER172-cell/eight-jobset inventory differs')
    return plan


def first_steer_admission(bundle):
    proof = json.loads((bundle.parent / 'launch/FIRST_CELL_ADMITTED.json').read_text())
    if (proof.get('packet_sha256') != PACKET_SHA or proof.get('verified') is not True
            or proof.get('inside_pinned_image') is not True or proof.get('worker_drained') is not True
            or sha(Path(proof['dest'])/'result.json') != proof['result_sha256']
            or sha(Path(proof['dest'])/'fleet_receipt.json') != proof['receipt_sha256']):
        raise ValueError('first STEER cell persistence/native-image admission incomplete')
    return proof


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
    if doc.get('schema') == 'steer-fleet-packet-v1' and sha(CTL) != doc['ctl_sha256']:
        raise ValueError('canonical zenfleet controller differs from the STEER binding')
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
    inspector = bundle.parent / ("bin" if doc.get("schema") in ("e34-fleet-packet-v1", "steer-fleet-packet-v1") else "bin-final2") / "inspect_qualified_checkpoint"
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


def launch(bundle, arm, queue=FITV2 / "fleet_queue", live=FITV2 / "jobset_caps.json", *, first_cell=False):
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
    manifest = bundle / f"fit-manifest-{js}.json"
    if doc.get('schema') == 'steer-fleet-packet-v1':
        plan = steer_plan(bundle, doc)
        earlier = plan['order'][:plan['order'].index(js)]
        if first_cell:
            if js != plan['order'][0] or names:
                raise ValueError('first STEER admission requires the first jobset and an empty dedicated queue')
            manifest = bundle / 'FIRST_CELL_MANIFEST.json'
            if json.loads(manifest.read_text()) != json.loads((bundle/f'fit-manifest-{js}.json').read_text())[:1]:
                raise ValueError('first STEER manifest must be the exact first registered cell')
        else:
            first_steer_admission(bundle)
    else:
        if first_cell:
            raise ValueError('first-cell mode is STEER only')
        earlier = () if doc.get("schema") == "e34-fleet-packet-v1" else ORDER[: ORDER.index(arm)]
    if doc.get("schema") == "e34-fleet-packet-v1":
        plan=json.loads((bundle/"E34_JOB_PLAN.json").read_text())
        if js not in plan["unconditional"]:
            chosen=json.loads((bundle.parent/"launch/LAMBDA_SELECTION.json").read_text())
            selected=chosen["selected_branches"]
            if js not in selected.values() or len(selected) != 2:
                raise ValueError("K branch is not selected by the registered pilot")
    if any(jobset(a) not in names for a in earlier):
        raise ValueError("registered sequence: queue the earlier E33 jobsets first")
    rt, _, _ = runtime(bundle, doc)
    record = dict(jobset=js, packet_sha=PACKET_SHA, image=doc["image"], image_id=actual, started=now())
    if doc.get('schema') == 'steer-fleet-packet-v1':
        if '@sha256:' not in doc['image'] or doc['image_ref'] != doc['image']:
            raise ValueError('STEER requires the published immutable image reference')
        record['published_image'] = doc['image']
    else:
        push = subprocess.run(["docker", "push", doc["image"]], capture_output=True, text=True, check=True)
        record["push_tail"] = push.stdout.strip().splitlines()[-1:]
    control = rt / f"control-{js}.json"
    control.write_text(json.dumps(dict(paused=first_cell, drain=False,
                                       note="Frozen registered fit; coordinator authorization")) + "\n")
    subprocess.run(["bash", "-c", """set -euo pipefail
. ~/.config/zen/s3env.sh >/dev/null 2>&1
s5cmd --endpoint-url "$EP" cp "$1" "s3://zentrain/jobs/$2/manifest.json"
s5cmd --endpoint-url "$EP" cp "$3" "s3://zentrain/jobs/$2/control.json"
s5cmd --endpoint-url "$EP" cp "$4" "s3://zentrain/jobs/$2/inputs/$5"
""", "--", str(manifest), js, str(control),
                    str(bundle / "e33-fit-data.tar.gz"), doc["data_sha"]], check=True)
    log = FITV2 / "host_filler.log"
    record["filler_log_offset"] = log.stat().st_size if log.exists() else 0
    queue.with_name(f"fleet_queue.{js}.before").write_bytes(queue.read_bytes())
    staging = queue.with_name(f"fleet_queue.{js}.new")
    staging.write_text(queue.read_text().rstrip() + f"\n{js} {manifest} {doc.get('image_ref', doc['image'])}\n")
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


def sync_ledger(endpoint, js, status):
    """Retry transport only; an empty/missing prefix is never a successful harvest."""
    command = ["s5cmd", "--json", "--endpoint-url", endpoint, "sync",
               f"s3://zentrain/jobs/{js}/ledger/*", str(status) + "/"]
    failures = []
    for attempt, delay in enumerate((0, 1, 3), 1):
        if delay:
            time.sleep(delay)
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            errors = []
            for line in (result.stdout + "\n" + result.stderr).splitlines():
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    if not isinstance(event, dict) or event.get("error") or event.get("success") is False:
                        errors.append(line)
                except ValueError:
                    errors.append(line)
            # s5cmd 2.3 can print a structured error (including an empty prefix)
            # and still exit zero. Require both exit success and error-free events.
            if result.returncode == 0 and not errors:
                if failures:
                    print(f"ledger sync recovered {js} attempt={attempt}", flush=True)
                return
            detail = f"rc={result.returncode} stdout={result.stdout!r} stderr={result.stderr!r}"
        except subprocess.TimeoutExpired as exc:
            detail = f"timeout=30 stdout={exc.stdout!r} stderr={exc.stderr!r}"
        failures.append(f"attempt={attempt} {detail}")
        print(f"ledger sync ERROR {js} {failures[-1]}", file=sys.stderr, flush=True)
    raise RuntimeError(f"ledger sync failed for {js}: " + "; ".join(failures))


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
    suffix = 'steer' if js.startswith('fitsteer-') else 'e34' if js.startswith('fite34-') else 'e33'
    work = FITV2 / ('harvest-'+suffix) / js
    status = FITV2 / ('status-'+suffix) / js
    work.mkdir(parents=True, exist_ok=True)
    status.mkdir(parents=True, exist_ok=True)
    with (work / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        endpoint = os.environ["EP"]
        sync_ledger(endpoint, js, status)
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
            if doc.get('schema') == 'steer-fleet-packet-v1' and os.environ.get('STEER_HARVEST_IMAGE') != doc['image_ref']:
                scratch = bundle.parent/'launch/harvest-scratch'
                (scratch/'tmp').mkdir(parents=True,exist_ok=True)
                cmd = ['docker','run','--rm','--network','host','--memory','6g','--memory-swap','6g',
                       '--entrypoint','/usr/bin/nice','-e','AWS_ACCESS_KEY_ID','-e','AWS_SECRET_ACCESS_KEY',
                       '-e','TMPDIR=/scratch/tmp','-v',str(bundle.parent)+':'+str(bundle.parent)+':ro',
                       '-v',str(FITV2)+':'+str(FITV2),'-v',str(POT)+':'+str(POT),
                       '-v',str(scratch)+':/scratch',doc['image_ref'],'-n19','ionice','-c3',
                       'python3','/opt/fleet-fits/program/harvest_fit_cells.py',*cmd[2:]]
                inspector_index=cmd.index('--checkpoint-inspector')+1
                cmd[inspector_index]='/opt/fleet-fits/program/bin/inspect_qualified_checkpoint'
            subprocess.run(cmd, check=True)
        if install and require_all and not all(installed(hfc, c) for c in jobs):
            raise ValueError("installed cell identity/checkpoint check failed")
        return report


def harvest_attempt(command, log, timeout):
    """Bound one jobset and terminate its whole verifier process group on timeout."""
    with log.open("w") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return proc.wait(timeout=timeout)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise


def harvest_cycle(bundle, queue, status_file=None, jobset_timeout=300):
    """Attempt every queued E34 jobset, keeping failures explicit and independent."""
    doc = packet(bundle)
    if doc.get("schema") not in ("e34-fleet-packet-v1", "steer-fleet-packet-v1"):
        raise ValueError("supervised cycle requires E34 or STEER")
    rt, tools, _ = runtime(bundle, doc)
    sys.path.insert(0, str(tools))
    import harvest_fit_cells as hfc
    plan = steer_plan(bundle,doc) if doc.get('schema') == 'steer-fleet-packet-v1' else json.loads((bundle / "E34_JOB_PLAN.json").read_text())
    allowed = set(plan["unconditional"])
    selection = rt / "LAMBDA_SELECTION.json"
    if doc.get('schema') == 'e34-fleet-packet-v1' and selection.exists():
        selected = json.loads(selection.read_text())
        if selected["packet_sha256"] != PACKET_SHA or len(selected["selected_branches"]) != 2:
            raise ValueError("selection binding differs from frozen packet")
        allowed.update(selected["selected_branches"].values())
    names = queued(queue)
    if len(names) != len(set(names)) or not names or set(names) - allowed:
        raise ValueError("queue duplicates or contains an unreleased jobset")
    with (rt / "harvest-cycle.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        stamp = str(time.time_ns())
        attempts = rt / "harvest-attempts"
        attempts.mkdir(exist_ok=True)
        state = dict(time=now(), cycle=stamp, state="RUNNING", packet_sha256=PACKET_SHA, jobsets={})
        for js in names:
            log = attempts / f"{stamp}-{js}.log"
            jobs = json.loads((bundle / f"fit-manifest-{js}.json").read_text())
            entry = dict(jobset=js, total=len(jobs), done=None, attempt_log=str(log))
            try:
                command = [sys.executable, str(Path(__file__).resolve()), "harvest", "--packet", str(bundle),
                           "--packet-sha", PACKET_SHA, "--arm", js, "--install"]
                rc = harvest_attempt(command, log, jobset_timeout)
                if rc:
                    raise RuntimeError(f"harvest exited {rc}; see {log}")
                reports = []
                for line in log.read_text().splitlines():
                    if line.startswith('{"jobset"'):
                        value = json.loads(line)
                        if value.get("jobset") == js:
                            reports.append(value)
                if not reports or reports[-1]["total"] != len(jobs):
                    raise ValueError("harvest returned no matching manifest-count report")
                entry.update(state="OK", done=reports[-1]["done"], ledger=reports[-1]["ledger"])
            except Exception as exc:
                entry.update(state="ERROR", error=str(exc))
                with log.open("a") as stream:
                    traceback.print_exc(file=stream)
            # Count after installation, including on an error; never admit a failed jobset as complete.
            try:
                entry["installed"] = sum(installed(hfc, cell) for cell in jobs)
            except Exception as exc:
                entry.update(state="ERROR", installed=None, error=f"installed-count check failed: {exc}")
                with log.open("a") as stream:
                    traceback.print_exc(file=stream)
            entry["time"] = now()
            state["jobsets"][js] = entry
            state["updated_at"] = now()
            stage = rt / "HARVEST_STATUS.json.new"
            stage.write_text(json.dumps(state, indent=2) + "\n")
            os.replace(stage, rt / "HARVEST_STATUS.json")
            print(json.dumps(entry), flush=True)
            if status_file:
                with status_file.open("a") as stream:
                    stream.write(f"\n{entry['time']} — HARVEST {js}: {entry['state']}; "
                                 f"installed {entry['installed']}/{entry['total']}; ledger DONE {entry['done']}; "
                                 f"attempt log {log}" + (f"; {entry['error']}" if entry.get("error") else "") + "\n")
        state.update(installed=sum(v["installed"] or 0 for v in state["jobsets"].values()),
                     installed_count_unknown=[js for js, v in state["jobsets"].items() if v["installed"] is None],
                     queued_cells=sum(v["total"] for v in state["jobsets"].values()),
                     errors=[js for js, v in state["jobsets"].items() if v["state"] == "ERROR"])
        target = plan["total_executed_cells"]
        state["state"] = ("DEGRADED" if state["errors"] else "COMPLETE"
                          if state["installed"] == state["queued_cells"] == target else "PROGRESS")
        stage.write_text(json.dumps(state, indent=2) + "\n")
        os.replace(stage, rt / "HARVEST_STATUS.json")
        return state


def open_cells(js, manifest):
    """Unclaimed cells, as the filler's picker counts them: no claim object and no done/poison ledger row."""
    import glob
    import hashlib
    import pyarrow.parquet as pq
    endpoint = os.environ["EP"]
    ids = subprocess.run([str(CTL), "ids", "--manifest", str(manifest)], capture_output=True, text=True,
                         check=True).stdout.splitlines()
    chunks = {hashlib.sha256((line.split("\t")[1] + "\n").encode()).hexdigest() for line in ids if line.strip()}
    status = FITV2 / ("status-e34" if js.startswith("fite34-") else "status-e33") / js
    status.mkdir(parents=True, exist_ok=True)
    subprocess.run(["s5cmd", "--endpoint-url", endpoint, "sync", f"s3://zentrain/jobs/{js}/ledger/*", str(status) + "/"],
                   capture_output=True)
    finished = {hashlib.sha256((r["job_id"] + "\n").encode()).hexdigest() for f in glob.glob(str(status / "*.parquet"))
                for r in pq.read_table(f, columns=["job_id", "status"]).to_pylist() if r["status"] in ("done", "poison")}
    ls = subprocess.run(["s5cmd", "--endpoint-url", endpoint, "ls", f"s3://zentrain/jobs/{js}/claims/*"],
                        capture_output=True, text=True).stdout.splitlines()
    claimed = {line.split()[-1][len("chunk-"):] for line in ls if line.strip() and line.split()[-1].startswith("chunk-")}
    return len(chunks - finished - claimed)


def tower_workers(js, live=FITV2 / "jobset_caps.json"):
    """Running containers of this jobset on tower (the host whose standing 40g cap bounds overlap).

    The SSH placement key comes from the jobset's live caps map (the private fleet configuration).
    """
    hosts = [h for h in json.loads(live.read_text())[js]["hosts"] if h.rsplit("@", 1)[-1] == "tower"]
    if len(hosts) != 1:
        raise ValueError(f"{js}: expected exactly one tower placement key, found {hosts}")
    host = hosts[0]
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


def persist_loan_artifacts(bundle):
    """Copy existing in-flight artifacts off temporary hosts; never resume or admit them.

    Completed cells already persist immediately through the worker's canonical
    blob/ledger path. These immutable generations retain best/epoch bakes and
    logs between completions without changing the exact83 training recipe.
    """
    doc = packet(bundle)
    policy = doc['persistence']
    root = Path(policy['snapshot_root'])
    stamp = now().replace(':', '')
    reports=[]
    for host in policy['hosts']:
        if host not in ['i134','r5600g','i270']:
            raise ValueError('unregistered temporary host')
        source=f'tmp/fitv2-shared/fit-cell/{doc["data_sha"]}/rev4-featpot/'
        exists=subprocess.run(['ssh','-n','-o','BatchMode=yes',host,'test -d '+source],capture_output=True)
        if exists.returncode:
            reports.append(dict(host=host,state='HOST_UNREACHABLE' if exists.returncode==255 else 'INPUT_ROOT_NOT_CREATED',returncode=exists.returncode))
            continue
        dest=root/host/stamp
        dest.mkdir(parents=True,exist_ok=False)
        latest=root/host/'LATEST'
        pattern = 'steer-*-results' if doc.get('schema') == 'steer-fleet-packet-v1' else 'e34-*-results'
        cmd=['rsync','-a','--no-owner','--no-group','--delay-updates','--prune-empty-dirs',
             '--include=/'+pattern+'/***','--exclude=*']
        if latest.exists():cmd += ['--link-dest='+str(root/host/latest.read_text().strip())]
        rsync_available=subprocess.run(['ssh','-n','-o','BatchMode=yes',host,'command -v rsync'],
                                       capture_output=True).returncode==0
        transport='rsync' if rsync_available else 'ssh-tar'
        if rsync_available:
            proc=subprocess.run([*cmd,host+':'+source,str(dest)+'/'],capture_output=True,text=True,
                                env={**os.environ,'RSYNC_RSH':'ssh -o BatchMode=yes -o ConnectTimeout=8'})
            if proc.returncode not in (0,24):
                raise RuntimeError(f'{host}: snapshot failed rc={proc.returncode}: {proc.stderr}')
        else:
            # A host needs only its existing tar; retain a whole immutable raw
            # in-flight archive. It is never admitted as a completed fit.
            archive=dest/'INFLIGHT.tar.gz'
            command='cd '+source+' && set -- '+pattern+'; if [ -e "$1" ]; then nice -n19 ionice -c3 tar -czf - -- "$@"; else tar -czf - --files-from /dev/null; fi'
            with archive.open('wb') as stream:
                proc=subprocess.run(['ssh','-n','-o','BatchMode=yes','-o','ConnectTimeout=8',host,command],
                                    stdout=stream,stderr=subprocess.PIPE)
            if proc.returncode not in (0,1):
                raise RuntimeError(f'{host}: raw snapshot failed rc={proc.returncode}: {proc.stderr!r}')
        files={str(p.relative_to(dest)):sha(p) for p in dest.rglob('*') if p.is_file()}
        record=dict(host=host,time=now(),state='COPIED_UNVERIFIED_INFLIGHT',optimizer_resume=False,
                    transport=transport,copy_returncode=proc.returncode,files=files)
        (dest/'SNAPSHOT.json').write_text(json.dumps(record,indent=2)+'\n')
        latest.write_text(stamp+'\n')
        reports.append(dict(host=host,state=record['state'],files=len(files),snapshot=str(dest)))
    return dict(time=now(),snapshots=reports)


def shorter_queue(bundle, queue, out):
    """Order the canonical picker's queue by measured first-epoch cell cost."""
    doc=packet(bundle)
    costs=doc['smoke_epoch_seconds']
    rows=[line.split() for line in queue.read_text().splitlines() if line.strip()]
    if any(row[0] not in doc['jobsets'] for row in rows):
        raise ValueError('queue contains a jobset outside the frozen packet')
    rows.sort(key=lambda row:costs[row[0] if doc.get('schema') == 'steer-fleet-packet-v1'
                                   else row[0].split('-')[1]])
    stage=out.with_name(out.name+'.new')
    stage.write_text(''.join(' '.join(row)+'\n' for row in rows))
    os.replace(stage,out)
    return dict(queue=str(out),order=[row[0] for row in rows])


def lambda_pilot(bundle, instrument, partial=False):
    """Registered label-free dev96 selection; all score/gain work uses Rust."""
    doc=packet(bundle)
    binding=json.loads((bundle/'DEVELOPMENT_BINDING.json').read_text())
    inputs=Path(binding['assessment_inputs'])
    if sha(inputs)!=binding['assessment_inputs_sha256'] or sha(instrument)!=binding['instrument_sha256']:
        raise ValueError('development packet/instrument differs from freeze')
    panel=json.loads(inputs.read_text())
    if panel.get('labels_read') is not False or len(panel['cases'])!=96 or len({r['origin'] for r in panel['cases']})!=8:
        raise ValueError('registered eight-origin/96-case label-free panel required')
    rt,tools,_=runtime(bundle,doc)
    sys.path.insert(0,str(tools))
    import harvest_fit_cells as hfc
    reports=[];selected={};pending=[]
    for family in ['a','c']:
        choices=[]
        for arm,lam in [('k01',0.1),('k1',1.),('k10',10.)]:
            name=arm+family;js=f'fite34-{name}-pilot-20261015'
            jobs=json.loads((bundle/f'fit-manifest-{js}.json').read_text())
            if len(jobs)!=1 or not installed(hfc,jobs[0]):
                if partial:
                    pending.append(js)
                    continue
                raise ValueError('pilot artifact has not been verified and installed: '+js)
            cell=Path(jobs[0]['kind']['argv'][jobs[0]['kind']['argv'].index('--dest')+1])
            result=json.loads((cell/'result.json').read_text())
            model=Path(result['packed_model'])
            if sha(model)!=result['packed_model_sha256']:
                raise ValueError('packed pilot model differs from result receipt')
            work=rt/'lambda-pilot'/name;work.mkdir(parents=True,exist_ok=True)
            output=work/'dev96.json';jobpath=work/'job.json'
            cases=[{**r,'model':str(model),'model_sha256':sha(model)} for r in panel['cases']]
            job=dict(cases=cases,output=str(output),engine=False,floor_recovery=True,
                     linear_probe=False,labels_read=False,sensitivity_method='exact-gradient-d12')
            data=json.dumps(job,indent=2)+'\n'
            if jobpath.exists() and jobpath.read_text()!=data:
                raise ValueError('pilot job differs from existing immutable assessment')
            if not jobpath.exists():jobpath.write_text(data)
            if not output.exists():
                env=dict(os.environ,ZENSIM_FORMULA_REV='5',ZENSIM_MAX_TIER='v3',
                         RAYON_NUM_THREADS='1',STEERFIX_PACKET=str(jobpath),ZENSIM_NEIGHBOUR_EXACT='0')
                with (work/'native.log').open('w') as log:
                    subprocess.run([str(instrument),'metric::bake::steerfix_packet::engineering_packet',
                                    '--exact','--nocapture'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            assessment=json.loads(output.read_text());rows=assessment['rows']
            if (assessment['packet_sha256']!=sha(jobpath) or len(rows)!=96
                    or {r['key'] for r in rows}!={r['key'] for r in panel['cases']}
                    or any(r['model']!=sha(model) or r['sensitivity_method']!='exact-gradient-d12' for r in rows)):
                raise ValueError('native development assessment identity/coverage mismatch')
            passed=sum(r['m2'] is not None and r['m3f'] is not None and r['m2']>=.99 and r['m3f']>=.70 for r in rows)
            record=dict(family=family,lambda_value=lam,passing_cases=passed,total_cases=96,model=str(model),
                        model_sha256=sha(model),assessment_sha256=sha(output),assessment=str(output),jobset=js)
            reports.append(record);choices.append(record)
            print(json.dumps(record),flush=True)
        if len(choices)!=3:
            continue
        winner=sorted(choices,key=lambda r:(-r['passing_cases'],r['lambda_value']))[0]
        selected[family]=winner['jobset'].replace('-pilot-','-lodo-')
    if pending:
        return dict(state='WAIT',completed_assessments=reports,pending_artifacts=pending,labels_read=False)
    if len(reports)!=6 or len(selected)!=2:
        raise ValueError('all six completed dev96 assessments required for selection')
    selection=dict(schema='e34-registered-lambda-selection-v1',packet_sha256=PACKET_SHA,
                   assessment_inputs_sha256=binding['assessment_inputs_sha256'],labels_read=False,
                   sensitivity_method='exact-gradient-d12',reports=reports,selected_branches=selected,
                   rule='max joint(M2>=.99,M3f>=.70) dev96; tie smaller lambda; selected pilot is full seed0')
    path=rt/'LAMBDA_SELECTION.json'
    if path.exists():raise ValueError('selection already exists; never overwrite a release record')
    path.write_text(json.dumps(selection,indent=2)+'\n')
    return selection


def release_initial(bundle, queue):
    """Release exactly the frozen unconditional cells after first-artifact admission."""
    doc=packet(bundle)
    if doc.get('schema') == 'steer-fleet-packet-v1':
        first_steer_admission(bundle)
        plan=steer_plan(bundle,doc)
        first=plan['order'][0]
        rows=[line.split() for line in queue.read_text().splitlines() if line.strip()]
        if [row[0] for row in rows] != [first]:
            raise ValueError('STEER expansion requires only the admitted first jobset queued')
        manifest=bundle/f'fit-manifest-{first}.json'
        subprocess.run(['bash','-c','set -euo pipefail\n. ~/.config/zen/s3env.sh >/dev/null 2>&1\ns5cmd --endpoint-url "$EP" cp "$1" "s3://zentrain/jobs/$2/manifest.json"',
                        '--',str(manifest),first],check=True)
        rows[0][1]=str(manifest)
        stage=queue.with_name(queue.name+'.steer-expand')
        stage.write_text(''.join(' '.join(row)+'\n' for row in rows));os.replace(stage,queue)
        for js in plan['order'][1:]:
            launch(bundle,js,queue)
        return dict(released=plan['order'],unconditional_cells=172)
    proof=json.loads((bundle.parent/'launch/FIRST_CELL_ADMITTED.json').read_text())
    dest=Path(proof['dest'])
    if (proof['verified'] is not True or proof['packet_sha256']!=PACKET_SHA
            or sha(dest/'result.json')!=proof['result_sha256']
            or sha(dest/'fleet_receipt.json')!=proof['receipt_sha256']):
        raise ValueError('first registered cell has not been admitted')
    plan=json.loads((bundle/'E34_JOB_PLAN.json').read_text())
    order=sorted(plan['unconditional'],key=lambda js:(not '-pilot-' in js, js))
    for js in order:
        if js not in queued(queue):
            print(json.dumps(launch(bundle,js,queue)),flush=True)
    return dict(released=order,unconditional_cells=plan['initial_cells'])


def steer_harvest_command(bundle, doc, arguments):
    """Run the complete ledger reader and archive verifier with baked pyarrow and inspector."""
    repo = Path(__file__).resolve().parents[2]
    scratch = bundle.parent / 'launch/harvest-scratch'
    (scratch/'tmp').mkdir(parents=True, exist_ok=True)
    command = ['docker', 'run', '--rm', '--network', 'host', '--memory', '6g', '--memory-swap', '6g',
               '--entrypoint', '/usr/bin/nice', '-e', 'AWS_ACCESS_KEY_ID', '-e', 'AWS_SECRET_ACCESS_KEY',
               '-e', 'EP', '-e', 'TMPDIR=/scratch/tmp', '-e', 'STEER_HARVEST_IMAGE='+doc['image_ref']]
    for path, mode in [(repo, ':ro'), (bundle.parent, ''), (FITV2, ''), (POT, ''), (CTL.parent, ':ro')]:
        command += ['-v', str(path)+':'+str(path)+mode]
    if '--status-file' in arguments:
        status_parent = Path(arguments[arguments.index('--status-file')+1]).resolve().parent
        command += ['-v', str(status_parent)+':'+str(status_parent)]
    command += ['-v', str(scratch)+':/scratch', doc['image_ref'], '-n19', 'ionice', '-c3',
                'python3', str(Path(__file__).resolve()), *arguments]
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("verify", "caps", "launch", "placements", "harvest", "harvest-cycle", "advance", "persist", "shorter-queue", "lambda-pilot", "release-initial"))
    p.add_argument("--packet", type=Path, required=True)
    p.add_argument("--arm")
    p.add_argument("--packet-sha", help="explicit frozen E34 PACKET.json pin")
    p.add_argument("--queue", type=Path, default=FITV2/"fleet_queue")
    p.add_argument("--out-queue", type=Path)
    p.add_argument("--instrument", type=Path)
    p.add_argument("--status-file", type=Path, help="append per-jobset harvest counts and errors")
    p.add_argument("--partial", action="store_true", help="pilot: explicitly assess available artifacts, never select until all6")
    p.add_argument("--install", action="store_true")
    p.add_argument("--first-cell", action="store_true", help="STEER: release only the pinned first cell before persistence admission")
    p.add_argument("--require-all", action="store_true")
    p.add_argument("--status", action="store_true")
    a = p.parse_args()
    if a.packet_sha:
        global PACKET_SHA
        if len(a.packet_sha) != 64:
            p.error("--packet-sha requires a full SHA-256")
        PACKET_SHA=a.packet_sha
    elif a.arm and a.arm.startswith(("fite34-", "fitsteer-")):
        p.error("E34/STEER requires the frozen --packet-sha")
    if a.mode in ("launch", "placements", "harvest") and not a.arm:
        p.error(f"{a.mode} requires --arm")
    if a.mode in ('harvest', 'harvest-cycle'):
        doc = packet(a.packet)
        if doc.get('schema') == 'steer-fleet-packet-v1':
            marker = os.environ.get('STEER_HARVEST_IMAGE')
            if marker is None:
                sys.exit(subprocess.run(steer_harvest_command(a.packet, doc, sys.argv[1:])).returncode)
            if marker != doc['image_ref']:
                raise ValueError('STEER harvest image differs from frozen digest')
    if a.mode == "harvest-cycle":
        out = harvest_cycle(a.packet, a.queue, a.status_file)
        print(json.dumps(out))
        sys.exit(0 if out["state"] == "COMPLETE" else 4 if out["state"] == "DEGRADED" else 3)
    elif a.mode == "release-initial":
        out=release_initial(a.packet,a.queue)
    elif a.mode == "lambda-pilot":
        if not a.instrument:p.error("lambda-pilot requires --instrument")
        out=lambda_pilot(a.packet,a.instrument,a.partial)
    elif a.mode == "persist":
        out=persist_loan_artifacts(a.packet)
    elif a.mode == "shorter-queue":
        if not a.out_queue:p.error("shorter-queue requires --out-queue")
        out=shorter_queue(a.packet,a.queue,a.out_queue)
    elif a.mode == "verify":
        doc = packet(a.packet)
        runtime(a.packet, doc)
        out = dict(packet_sha=PACKET_SHA, program_sha=doc["program_sha"], image=doc["image"], jobsets=doc["jobsets"])
    elif a.mode == "caps":
        out = merge_caps(a.packet)
    elif a.mode == "launch":
        out = launch(a.packet, a.arm, a.queue, first_cell=a.first_cell)
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
    except (ValueError, OSError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"E33 REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)
