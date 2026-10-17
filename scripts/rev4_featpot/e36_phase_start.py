"""Seal exclusive ownership of one frozen native phase before starting it.

The live dev controller stays debugger-stopped across ownership sealing and
Docker start. Native workers continue; no generation clock is paused or reset.
"""

import argparse
import hashlib
import json
import marshal
from pathlib import Path
import subprocess
import sys
import time

from e36_phase_handoff import inspect_bound, remote, sha, write


def guard_current(root, out):
    root, out = Path(root), Path(out)
    hook = json.loads((root / "live-completion-hook.json").read_text())
    frames = []
    for top in sys._current_frames().values():
        frame = top
        while frame:
            if frame.f_code.co_name == "bulk" and frame.f_locals.get("root") == root:
                frames.append(frame)
            frame = frame.f_back
    if len(frames) != 1:
        raise ValueError("exactly one original bulk controller frame required")
    frame = frames[0]
    if (
        sha(frame.f_code.co_filename) != hook["controller_source_sha256"]
        or hashlib.sha256(marshal.dumps(frame.f_code)).hexdigest()
        != hook["controller_code_sha256"]
        or frame.f_lineno >= hook["branch_line"]
        or frame.f_trace is None
    ):
        raise ValueError(
            "controller has reached its tail or its installed hook differs"
        )
    plan = json.loads((root / "bulk-plan-v6.json").read_text())
    extra = root / "runs" / plan["phases"]["additions"]["run"]
    if (
        (root / "TOWER_PHASE_HANDOFF.json").exists()
        or (root / "bulk-choice.json").exists()
        or any(extra.glob("ledger*.parquet"))
        or (extra / "census.json").exists()
        or any((extra / "blobs").rglob("*"))
    ):
        raise ValueError(
            "phase already has an owner, execution evidence or frozen choice"
        )
    write(
        out,
        dict(
            schema="e36-live-phase-owner-guard-v1",
            admitted=True,
            root=str(root),
            line=frame.f_lineno,
            branch_line=hook["branch_line"],
            checked_unix_ns=time.time_ns(),
            fits=0,
        ),
    )


def resource_admission(binding):
    state = inspect_bound(binding)["State"]
    if state["Status"] != "created":
        raise ValueError("native phase must be a never-started bound container")
    snapshot = remote(
        binding["host"],
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    blockers = [
        n for n in snapshot if n.startswith(("zen-score-", "zen-steer-", "imij-"))
    ]
    if blockers:
        raise ValueError("tower heavy allocation still active: " + repr(blockers))
    meminfo = remote(
        binding["host"], ["cat", "/proc/meminfo"], capture_output=True, text=True
    ).stdout
    available = next(
        int(line.split()[1]) * 1024
        for line in meminfo.splitlines()
        if line.startswith("MemAvailable:")
    )
    config = binding["host_config"]
    cap = config["Memory"]
    if not (0 < cap <= 40 * 1024**3 and available - cap >= 12 * 1024**3):
        raise ValueError(
            "tower memory cap must leave at least12GiB of available memory"
        )
    if (
        config["CpuShares"] != 256
        or config["CpusetCpus"] not in ["1", "0-7,16-23"]
        or config["NetworkMode"] != "none"
    ):
        raise ValueError("tower resource/isolation configuration differs")
    return dict(
        available_bytes=available,
        cap_bytes=cap,
        other_containers=snapshot,
        observed_unix_ns=time.time_ns(),
        leave_available_bytes=12 * 1024**3,
    )


def seal_start(root, binding_path, adapter, parity, guard, supervisor, paper_lock, out):
    root, out, binding_path = Path(root), Path(out), Path(binding_path)
    binding = json.loads(binding_path.read_text())
    resource = resource_admission(binding)
    if binding["phase"] == "additions":
        proof = json.loads(Path(parity).read_text())
        live = json.loads(Path(guard).read_text())
        if (
            proof["admitted"] is not True
            or proof["views"] != 5
            or proof["bounded_first_producer_only"] is not True
            or live["admitted"] is not True
            or live["root"] != str(root)
            or time.time_ns() - live["checked_unix_ns"] > 60 * 10**9
        ):
            raise ValueError(
                "fresh stopped-controller guard and exact first-producer parity required"
            )
        parity_execution = Path(proof["native_execution"]["path"])
        if sha(parity_execution) != proof["native_execution"]["sha256"]:
            raise ValueError("parity execution evidence changed")
        previous = json.loads(parity_execution.read_text())
        if (
            previous["returncode"] != 0
            or previous["phase"] != "first-producer"
            or previous["files_manifest_sha256"] != binding["files_manifest_sha256"]
            or previous["effective_clock_sha256"]
            != binding["effective_clock"]["sha256"]
        ):
            raise ValueError(
                "first producer and additions must use identical frozen inputs and clock"
            )
        plan = json.loads((root / "bulk-plan-v6.json").read_text())
        phase = plan["phases"]["additions"]
        if binding["manifest"]["sha256"] != phase["manifest_sha256"]:
            raise ValueError("whole original additions manifest required")
        handoff = dict(
            schema="e36-whole-native-phase-owner-v1",
            run=phase["run"],
            container=dict(path=str(binding_path), sha256=sha(binding_path)),
            adapter=dict(path=str(adapter), sha256=sha(adapter)),
            parity=dict(path=str(parity), sha256=sha(parity)),
            live_guard=dict(path=str(guard), sha256=sha(guard)),
            supervisor_pid=supervisor,
            paper_lock=str(paper_lock),
            resources=resource,
            ownership_sealed_unix_ns=time.time_ns(),
            fits=0,
        )
        # An exclusive create is the ownership boundary, before remote start.
        write(root / "TOWER_PHASE_HANDOFF.json", handoff)
    elif binding["phase"] != "first-producer":
        raise ValueError("only the frozen first producer or whole additions can start")
    out.mkdir()
    with (out / "start.log").open("x") as log:
        result = remote(
            binding["host"],
            ["docker", "start", binding["container_id"]],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    write(
        out / "start.json",
        dict(
            container_sha256=sha(binding_path),
            resources=resource,
            returncode=result.returncode,
            fits=0,
        ),
    )


def attach_start(args):
    root, out = args.root, args.out
    out.mkdir()
    guard = out / "controller-guard.json"
    code = (
        "import sys,importlib.util; sys.path.insert(0,"
        + repr(str(Path(__file__).resolve().parent))
        + "); s=importlib.util.spec_from_file_location('e36_phase_start',"
        + repr(str(Path(__file__).resolve()))
        + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.guard_current("
        + repr(str(root))
        + ","
        + repr(str(guard))
        + ")"
    )
    argv = [
        sys.executable,
        str(Path(__file__).resolve()),
        "seal-start",
        "--root",
        str(root),
        "--binding",
        str(args.binding),
        "--adapter",
        str(args.adapter),
        "--parity",
        str(args.parity),
        "--guard",
        str(guard),
        "--supervisor",
        str(args.supervisor),
        "--paper-lock",
        str(args.paper_lock),
        "--out",
        str(out / "sealed-start"),
    ]
    script = out / "attach.gdb"
    script.write_text(
        "set confirm off\nset unwind-on-signal on\n"
        "handle SIGALRM nostop noprint nopass\nattach "
        + str(args.controller)
        + "\nhandle SIGSTOP nostop noprint nopass\n"
        "set $gil=(int)PyGILState_Ensure()\nset $rc=(int)PyRun_SimpleString("
        + json.dumps(code)
        + ")\ncall (void)PyGILState_Release($gil)\npython\n"
        "import subprocess\n"
        "assert int(gdb.parse_and_eval('$rc'))==0, 'live ownership guard failed'\n"
        "subprocess.run(" + repr(argv) + ",check=True)\nend\ndetach\n"
    )
    with (out / "attach.log").open("x") as log:
        subprocess.run(
            ["sudo", "-n", "gdb", "-q", "-n", "-batch", "-x", str(script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["first-producer", "attach-start", "seal-start"])
    for name in ["root", "binding", "out", "adapter", "parity", "guard", "paper-lock"]:
        p.add_argument("--" + name, type=Path)
    p.add_argument("--controller", type=int)
    p.add_argument("--supervisor", type=int)
    a = p.parse_args()
    if a.mode == "attach-start":
        attach_start(a)
    else:
        seal_start(
            a.root,
            a.binding,
            a.adapter,
            a.parity,
            a.guard,
            a.supervisor,
            a.paper_lock,
            a.out,
        )
