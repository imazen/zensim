"""Bind the tested census to a waiting resume controller without changing clocks."""

import argparse
import hashlib
import importlib.util
import json
import marshal
from pathlib import Path
import subprocess
import sys
import time

from e36_phase_handoff import supervisor_lock
from e36_store import sha


def install_current(root, receipt, original_sha, supervisor, paper_lock):
    root, receipt = Path(root), Path(receipt)
    source = root / "control-resume-v7/e36_resume_schedule.py"
    frames = []
    for top in sys._current_frames().values():
        frame = top
        while frame:
            if (
                frame.f_code.co_name == "run"
                and Path(frame.f_code.co_filename) == source
                and frame.f_locals.get("root") == root
            ):
                frames.append(frame)
            frame = frame.f_back
    if (
        len(frames) != 1
        or "rc" in frames[0].f_locals
        or (root / "bulk-choice.json").exists()
    ):
        raise ValueError("unique waiting pre-qualification resume controller required")
    frame = frames[0]
    owner = frame.f_locals["owner"]
    original = owner.qualify
    if (
        Path(owner.__file__) != root / "control-resume-v7/e36_bulk.py"
        or sha(owner.__file__) != original_sha
    ):
        raise ValueError("original census/control source changed")
    clock = root / "bulk-clock-v6-budget2.json"
    clock_sha = sha(clock)
    module_path = Path(__file__).with_name("e36_census_stream.py")
    spec = importlib.util.spec_from_file_location(
        "e36_bound_stream_census", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    old_code = hashlib.sha256(marshal.dumps(owner.census.__code__)).hexdigest()

    def window(name, stage):
        directory = root / "step-lock-windows"
        directory.mkdir(exist_ok=True)
        stem = name + "-" + stage + "-" + str(time.time_ns())
        supervisor_lock(supervisor, paper_lock, 8, directory / (stem + "-release.log"))
        supervisor_lock(
            supervisor, paper_lock, 2, directory / (stem + "-reacquire.log")
        )

    def qualify(root_arg, name, start, deadline):
        window(name, "before-census")
        try:
            path = Path(root_arg) / "runs" / name / "census.json"
            if path.exists():
                path.rename(
                    path.with_name("census.retained-" + str(time.time_ns()) + ".json")
                )
            try:
                module.census(root_arg, name)
            except ValueError:
                return False
            # The original expected-window/hash predicate still makes the decision.
            return original(root_arg, name, start, deadline)
        finally:
            window(name, "after-census")

    owner.census = module.census
    owner.qualify = qualify
    if sha(clock) != clock_sha:
        raise ValueError("census binding changed clock bytes")
    with receipt.open("x") as f:
        json.dump(
            dict(
                schema="e36-live-stream-census-binding-v1",
                original_source_sha256=original_sha,
                original_census_code_sha256=old_code,
                new_census=dict(path=str(module_path), sha256=sha(module_path)),
                installer=dict(path=str(Path(__file__)), sha256=sha(__file__)),
                ledger_owner=dict(
                    path=str(Path(__file__).with_name("e36_native_ledger_admit.py")),
                    sha256=sha(Path(__file__).with_name("e36_native_ledger_admit.py")),
                ),
                clock_sha256=clock_sha,
                clock_changed=False,
                scientific_change=False,
                always_fresh_census=True,
                short_job_windows=True,
                fits=0,
            ),
            f,
            indent=2,
        )


def attach(args):
    script = args.receipt.with_suffix(".gdb")
    code = (
        "import sys,importlib.util; sys.path.insert(0,"
        + repr(str(Path(__file__).resolve().parent))
        + "); s=importlib.util.spec_from_file_location('e36_census_install',"
        + repr(str(Path(__file__).resolve()))
        + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.install_current("
        + ",".join(
            repr(v)
            for v in [
                str(args.root),
                str(args.receipt),
                args.original_sha,
                args.supervisor,
                str(args.paper_lock),
            ]
        )
        + ")"
    )
    with script.open("x") as f:
        f.write(
            "set confirm off\nset unwind-on-signal on\nhandle SIGALRM nostop noprint nopass\nattach "
            + str(args.pid)
            + "\nhandle SIGSTOP nostop noprint nopass\nset $gil=(int)PyGILState_Ensure()\nset $rc=(int)PyRun_SimpleString("
            + json.dumps(code)
            + ')\ncall (void)PyGILState_Release($gil)\npython\nassert int(gdb.parse_and_eval("$rc"))==0, "census binding failed"\nend\ndetach\n'
        )
    with args.receipt.with_suffix(".gdb.log").open("x") as log:
        subprocess.run(
            ["sudo", "-n", "gdb", "-q", "-n", "-batch", "-x", str(script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "receipt", "paper-lock"]:
        p.add_argument("--" + name, type=Path, required=True)
    for name in ["pid", "supervisor"]:
        p.add_argument("--" + name, type=int, required=True)
    p.add_argument("--original-sha", required=True)
    attach(p.parse_args())
