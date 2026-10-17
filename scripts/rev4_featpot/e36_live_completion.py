"""Install the reviewed completion tail on an already-running frozen controller.

The tracer jumps over the old tail after computing the corrected result. It
never changes time values or the native qualification predicate. No fit runs.
"""

import ast
import hashlib
import importlib.util
import json
import marshal
from pathlib import Path
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def install_current(root, receipt):
    root = Path(root)
    receipt = Path(receipt)
    if receipt.exists():
        raise ValueError("completion hook was already installed")
    candidates = []
    for top in sys._current_frames().values():
        f = top
        while f:
            if (
                f.f_code.co_name == "bulk"
                and Path(f.f_code.co_filename).name == "e36_bulk.py"
                and f.f_locals.get("root") == root
            ):
                candidates.append(f)
            f = f.f_back
    if len(candidates) != 1:
        raise ValueError("exactly one owned live bulk frame required")
    target = candidates[0]
    source = Path(target.f_code.co_filename)
    tree = ast.parse(source.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "bulk"
    )
    branch = next(
        n
        for n in function.body
        if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call)
        and isinstance(n.test.func, ast.Name)
        and n.test.func.id == "qualify"
        and len(n.test.args) > 1
        and isinstance(n.test.args[1], ast.Name)
        and n.test.args[1].id == "mandatory"
    )
    choice = next(
        n
        for n in function.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "choice" for t in n.targets)
    )
    if branch.lineno >= choice.lineno or sys.gettrace() is not None:
        raise ValueError("unexpected completion source or existing tracer")
    binding = dict(
        schema="e36-live-completion-hook-v1",
        controller_source_path=str(source),
        controller_source_sha256=sha(source),
        controller_code_sha256=hashlib.sha256(marshal.dumps(target.f_code)).hexdigest(),
        adapter_source_path=str(Path(__file__).resolve()),
        adapter_source_sha256=sha(__file__),
        branch_line=branch.lineno,
        choice_line=choice.lineno,
        clock_values_changed=False,
        native_qualification_changed=False,
        fits=0,
    )
    with receipt.open("x") as f:
        json.dump(binding, f, indent=2)

    def trace(frame, event, arg):
        if frame is not target:
            return None
        if event == "line" and frame.f_lineno == branch.lineno:
            local = frame.f_locals
            glob = frame.f_globals
            mandatory = local["mandatory"]
            extra = local["extra"]
            start = local["start"]
            deadline = local["deadline"]
            errors = local["errors"]
            qualified = None
            if glob["qualify"](root, mandatory, start, deadline):
                qualified = "fallback"
                handoff = root / "TOWER_PHASE_HANDOFF.json"
                if handoff.exists():
                    spec = json.loads(handoff.read_text())
                    owner = spec["adapter"]
                    if sha(owner["path"]) != owner["sha256"]:
                        raise ValueError("native phase handoff adapter changed")
                    modspec = importlib.util.spec_from_file_location(
                        "e36_bound_phase_handoff", owner["path"]
                    )
                    module = importlib.util.module_from_spec(modspec)
                    modspec.loader.exec_module(module)
                    module.wait_and_harvest(root, extra, deadline, handoff)
                elif glob["time"].time() < deadline:
                    try:
                        glob["run"](
                            root, extra, local["plan"]["slots"], deadline=deadline
                        )
                    except (SystemExit, TimeoutError) as exc:
                        errors.append(str(exc))
                if glob["qualify"](root, extra, start, deadline):
                    qualified = "full"
            local["qualified"] = qualified
            # CPython's debugger jump requires locals live at the destination
            # to be initialized. The original assignment immediately replaces it.
            local["choice"] = None
            frame.f_lineno = choice.lineno
            frame.f_trace = None
            sys.settrace(None)
            with receipt.with_name(receipt.stem + "-applied.json").open("x") as f:
                json.dump(
                    dict(
                        binding,
                        qualified=qualified,
                        errors=errors,
                        clock_values_changed=False,
                    ),
                    f,
                    indent=2,
                )
            return None
        return trace

    target.f_trace = trace
    target.f_trace_lines = True
    sys.settrace(trace)


def attach(pid, root, receipt):
    import subprocess

    receipt = Path(receipt)
    script = receipt.with_suffix(".gdb")
    code = (
        "import importlib.util; s=importlib.util.spec_from_file_location('e36_completion',"
        + repr(str(Path(__file__).resolve()))
        + "); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.install_current("
        + repr(str(root))
        + ","
        + repr(str(receipt))
        + ")"
    )
    with script.open("x") as f:
        f.write(
            """set confirm off
set unwind-on-signal on
handle SIGALRM nostop noprint nopass
attach """
            + str(pid)
            + """
handle SIGSTOP nostop noprint nopass
set $gil=(int)PyGILState_Ensure()
set $rc=(int)PyRun_SimpleString("""
            + json.dumps(code)
            + """)
call (void)PyGILState_Release($gil)
python
if int(gdb.parse_and_eval('$rc'))!=0:raise RuntimeError('completion hook installation failed')
end
detach
"""
        )
    with receipt.with_suffix(".gdb.log").open("x") as log:
        subprocess.run(
            ["sudo", "-n", "gdb", "-q", "-n", "-batch", "-x", str(script)],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    for n in ["root", "receipt"]:
        p.add_argument("--" + n, type=Path, required=True)
    a = p.parse_args()
    attach(a.pid, a.root, a.receipt)
