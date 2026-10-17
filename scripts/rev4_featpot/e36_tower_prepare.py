"""Prepare a frozen whole native phase's file closure; never launch or split jobs."""

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

from e36_pause_budget import resolve
from e36_store import sha
from e36_data import PREFIX


def write(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def prepare(root, out, image, run_heavy=None, memory_gib=16, packing_gib=12, slots=4):
    import PIL

    if not (
        8 <= memory_gib <= 40 and 0 < packing_gib <= memory_gib - 4 and 1 <= slots <= 8
    ):
        raise ValueError("whole-phase resource cap/packing headroom/slots are invalid")
    out.mkdir()
    clock_path = root / "bulk-clock-v6-budget2.json"
    _, clock, budget = resolve(clock_path)
    plan_path = root / "bulk-plan-v6.json"
    plan = json.loads(plan_path.read_text())
    if sha(plan_path) != clock["plan_sha256"]:
        raise ValueError("frozen native phase plan changed")
    manifests = {}
    jobs = {}
    for role, entry in plan["phases"].items():
        path = root / "runs" / entry["run"] / "manifest.json"
        if sha(path) != entry["manifest_sha256"]:
            raise ValueError("frozen native phase manifest changed")
        items = json.loads(path.read_text())
        views = [j["kind"]["view"] for j in items]
        if len(views) != len(set(views)):
            raise ValueError("duplicate native phase view")
        manifests[role] = dict(path=str(path), sha256=sha(path), views=len(views))
        jobs[role] = set(views)
    if (
        jobs["fallback"] & jobs["additions"]
        or jobs["fallback"] | jobs["additions"] != jobs["full"]
    ):
        raise ValueError("whole native phases are not the exact disjoint frozen union")
    runtime_path = Path(plan["runtime"])
    runtime = json.loads(runtime_path.read_text())
    if sha(runtime_path) != plan["runtime_sha256"]:
        raise ValueError("frozen runtime changed")
    references = json.loads(Path(runtime["references"]).read_text())
    if sha(runtime["references"]) != runtime["references_sha256"]:
        raise ValueError("reference index changed")
    refs = {r["reference_key"]: r for r in references["rows"]}
    files = {}
    executables = set()

    def add(path, expected=None, dest=None, executable=False):
        path = Path(path)
        dest = str(dest or path)
        if not path.is_absolute() or not Path(dest).is_absolute():
            raise ValueError("native file closure requires absolute paths")
        value = dict(
            source=str(path),
            destination=dest,
            sha256=expected or sha(path),
            bytes=path.stat().st_size,
        )
        if dest in files and files[dest] != value:
            raise ValueError("conflicting native phase destination")
        files[dest] = value
        if executable:
            executables.add(str(path))

    add(runtime_path, plan["runtime_sha256"])
    add(runtime_path.parent / "executor")
    for name, expected in runtime["program_files"].items():
        add(runtime_path.parent / "program" / name, expected)
    for key in ["references", "source_budget", "feature_spec", "native_pin"]:
        add(runtime[key])
    add(root / "FEATURE_LAYOUT_BINDING.json", runtime["feature_layout_sha256"])
    add(root / "resource-admission.json", plan["resources_sha256"])
    add(plan_path)
    add(clock_path)
    for proof in budget:
        add(proof["path"], proof["sha256"])
    for name, expected in runtime["mode_inventory_sha256"].items():
        add(Path(runtime["mode_inventory"]) / (name + ".jsonl"), expected)
    pin = json.loads(Path(runtime["native_pin"]).read_text())
    add(runtime["native_tool"], pin["binary_sha256"], executable=True)
    add(runtime["feature_tool"], pin["pixel_binary_sha256"], executable=True)
    for name in ["zenfleet-worker", "zenfleet-ctl"]:
        add(root / "bin" / name, executable=True)
    for name, path in runtime["oracles"].items():
        add(path, runtime["oracle_sha256"][name], executable=True)
    selected = {
        json.loads(v[len(PREFIX) :])["reference_key"] for v in jobs["additions"]
    }
    # The already-frozen first-producer phase is the parity/admission oracle,
    # never a new subset of the bulk matrix.
    first = root / "runs/first-producer-v6/manifest.json"
    if not first.exists():
        raise ValueError("original first-producer declaration is required")
    first_jobs = json.loads(first.read_text())
    for j in first_jobs:
        v = json.loads(j["kind"]["view"][len(PREFIX) :])
        selected.add(v["reference_key"])
        if (
            v["role"] != "first-producer"
            or v["runtime_sha256"] != plan["runtime_sha256"]
        ):
            raise ValueError("first-producer runtime/role differs")
    add(first)
    for role in ["additions"]:
        add(manifests[role]["path"], manifests[role]["sha256"])
    for key in selected:
        r = refs[key]
        add(r["destination"], r["sha256"])
    # Inspection-only Pillow is bundled byte-for-byte, including its wheel's
    # native libraries. No third-party encoder produces a training target.
    site = Path(PIL.__file__).parent.parent
    for package in ["PIL", "pillow.libs"]:
        directory = site / package
        if not directory.is_dir():
            raise ValueError("frozen Pillow package closure missing: " + package)
        for path in directory.rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path, dest=Path("/opt/e36/python") / path.relative_to(site))
    library_paths = set()
    for path in sorted(executables):
        text = subprocess.check_output(["ldd", path], text=True)
        if "not found" in text:
            raise ValueError("native runtime library unresolved")
        for line in text.splitlines():
            match = re.search(r"(?:=>\s+)?(/\S+)\s+\(0x", line)
            if match:
                library_paths.add(match.group(1))
    for path in sorted(library_paths):
        add(path)
    transport = Path(__file__).with_name("e36_local_store_transport.py")
    add(transport, dest="/opt/e36/transport/ssh")
    add(
        Path(__file__).with_name("e36_phase_worker.py"), dest="/opt/e36/phase-worker.py"
    )
    launcher_argv = shlex.split(
        (runtime_path.parent / "executor").read_text().splitlines()[-1]
    )
    if launcher_argv[0] != "exec" or not Path(launcher_argv[1]).is_absolute():
        raise ValueError("unexpected frozen interpreter launcher")
    if launcher_argv[1] != "/opt/venv/bin/python3":
        add(Path(__file__).with_name("e36_python_transport.sh"), dest=launcher_argv[1])
    if run_heavy is not None:
        add(run_heavy, dest="/opt/e36/run-heavy")
    write(
        out / "files.json",
        dict(
            schema="e36-whole-native-phase-files-v1",
            files=list(files.values()),
            library_paths=sorted(library_paths),
            manifests=manifests,
            first_producer=dict(
                path=str(first), sha256=sha(first), jobs=len(first_jobs)
            ),
            image=image,
            root=str(root),
            python_source_root=str(site),
            launcher_interpreter=launcher_argv[1],
            clock=dict(path=str(clock_path), sha256=sha(clock_path)),
            runtime=dict(path=str(runtime_path), sha256=sha(runtime_path)),
            source_views=len(jobs["additions"]),
            reference_files=len(selected),
            file_bytes=sum(r["bytes"] for r in files.values()),
            pillow_version=PIL.__version__,
            scientific_change=False,
            fits=0,
            launch_authorized=False,
            staging_only=True,
            worker_slots=slots,
            packing_ram_bytes=packing_gib * 1024**3,
            container_memory_bytes=memory_gib * 1024**3,
            cpuset="0-7,16-23",
            cpu_shares=256,
        ),
    )
    sources = sorted({r["source"].lstrip("/") for r in files.values()})
    with (out / "sources.files0").open("xb") as f:
        f.write(b"\0".join(p.encode() for p in sources) + b"\0")
    print(
        json.dumps(
            dict(
                phase_views=len(jobs["additions"]),
                references=len(selected),
                files=len(files),
                bytes=sum(r["bytes"] for r in files.values()),
                staging_only=True,
                fits=0,
            )
        )
    )


def stage(preparation, host, container, name, previous=None):
    """Copy and verify input files into the existing storage-only container."""
    preparation = Path(preparation)
    spec = json.loads((preparation / "files.json").read_text())
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError("stage name must be one directory component")

    def remote(args, **kwargs):
        return subprocess.run(
            ["ssh", "-T", host, shlex.join(args)], check=True, **kwargs
        )

    inspected = remote(
        ["docker", "inspect", container, "--format", "{{json .Mounts}}"],
        capture_output=True,
        text=True,
    )
    mounts = json.loads(inspected.stdout)
    runtime = json.loads(Path(spec["runtime"]["path"]).read_text())
    data = [m for m in mounts if m["Destination"] == "/data"]
    if (
        len(data) != 1
        or data[0]["Source"] != runtime["storage"]["root"]
        or host != runtime["storage"]["host"]
    ):
        raise ValueError(
            "resolved storage mount/host differs from the frozen canonical store"
        )
    dest = "/data/" + name
    manifest_name = "files.json"
    verified_name = "verified.json"
    filelist = preparation / "sources.files0"
    copy_files = True
    if previous is None:
        make = (
            "from pathlib import Path; p=Path("
            + repr(dest)
            + "); p.mkdir(); (p/'rootfs').mkdir()"
        )
    else:
        previous = Path(previous)
        old = json.loads((previous / "files.json").read_text())
        old_rows = {r["destination"]: r for r in old["files"]}
        new_rows = {r["destination"]: r for r in spec["files"]}
        if any(new_rows.get(k) != r for k, r in old_rows.items()) or any(
            old[k] != spec[k]
            for k in [
                "root",
                "clock",
                "runtime",
                "image",
                "manifests",
                "first_producer",
            ]
        ):
            raise ValueError(
                "append-only native stage changed an existing input or scientific binding"
            )
        extra = [r for k, r in new_rows.items() if k not in old_rows]
        copy_files = bool(extra)
        filelist = preparation / "append.sources.files0"
        with filelist.open("xb") as f:
            f.write(b"\0".join(r["source"].lstrip("/").encode() for r in extra) + b"\0")
        manifest_name = "files-" + preparation.name + ".json"
        verified_name = "verified-" + preparation.name + ".json"
        make = (
            "from pathlib import Path; import hashlib; p=Path(" + repr(dest) + "); "
            "assert hashlib.sha256((p/'files.json').read_bytes()).hexdigest()=="
            + repr(sha(previous / "files.json"))
            + "; "
            "assert not any((p/'rootfs'/s.lstrip('/')).exists() for s in "
            + repr([r["source"] for r in extra])
            + ")"
        )
    remote(["docker", "exec", container, "python3", "-c", make])
    logfile = preparation / (name + ".log")
    with logfile.open("x") as log:
        # -C precedes -T: tar treats -C as a positional option.
        if copy_files:
            reader = subprocess.Popen(
                [
                    "nice",
                    "-n19",
                    "ionice",
                    "-c3",
                    "tar",
                    "--directory",
                    "/",
                    "--dereference",
                    "--null",
                    "--files-from",
                    str(filelist),
                    "-cf",
                    "-",
                ],
                stdout=subprocess.PIPE,
                stderr=log,
            )
            writer = subprocess.Popen(
                [
                    "ssh",
                    "-T",
                    host,
                    shlex.join(
                        [
                            "docker",
                            "exec",
                            "-i",
                            container,
                            "tar",
                            "--no-same-owner",
                            "-xf",
                            "-",
                            "-C",
                            dest + "/rootfs",
                        ]
                    ),
                ],
                stdin=reader.stdout,
                stdout=log,
                stderr=log,
            )
            reader.stdout.close()
            write_rc = writer.wait()
            read_rc = reader.wait()
            if write_rc or read_rc:
                raise RuntimeError(
                    "staging copy failed; retained complete/partial staging: "
                    + str((read_rc, write_rc))
                )
        put = (
            "from pathlib import Path; import sys; p=Path("
            + repr(dest + "/" + manifest_name)
            + "); f=p.open('xb'); f.write(sys.stdin.buffer.read()); f.close()"
        )
        remote(
            ["docker", "exec", "-i", container, "python3", "-c", put],
            input=(preparation / "files.json").read_bytes(),
            stdout=log,
            stderr=log,
        )
        check = (
            """from pathlib import Path
import json,hashlib
root=Path(STAGE);manifest=root/MANIFEST;spec=json.loads(manifest.read_text());count=0;total=0
for r in spec['files']:
 p=root/'rootfs'/r['source'].lstrip('/')
 with p.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
 if h!=r['sha256'] or p.stat().st_size!=r['bytes']:raise ValueError('staged native input differs: '+r['source'])
 count+=1;total+=p.stat().st_size
receipt=dict(schema='e36-native-phase-staging-v1',files=count,bytes=total,manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),manifest_name=MANIFEST,fits=0,launched=False)
with (root/VERIFIED).open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
""".replace("STAGE", repr(dest))
            .replace("MANIFEST", repr(manifest_name))
            .replace("VERIFIED", repr(verified_name))
        )
        checked = remote(
            ["docker", "exec", container, "python3", "-c", check],
            capture_output=True,
            text=True,
        )
        log.write(checked.stdout)
        log.write(checked.stderr)
    result = {
        **json.loads(checked.stdout),
        "host": host,
        "resolved_physical_root": data[0]["Source"] + "/" + name,
        "storage_container": container,
    }
    write(preparation / (name + ".json"), result)
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["root", "out", "preparation", "append-from"]:
        p.add_argument("--" + name, type=Path)
    p.add_argument("--image")
    p.add_argument("--run-heavy", type=Path)
    p.add_argument("--memory-gib", type=int, default=16)
    p.add_argument("--packing-gib", type=int, default=12)
    p.add_argument("--slots", type=int, default=4)
    for name in ["host", "container", "stage-name"]:
        p.add_argument("--" + name)
    a = p.parse_args()
    if a.preparation is not None:
        stage(a.preparation, a.host, a.container, a.stage_name, a.append_from)
    else:
        if a.root is None or a.out is None or a.image is None:
            p.error("--root/--out/--image required for preparation")
        prepare(
            a.root, a.out, a.image, a.run_heavy, a.memory_gib, a.packing_gib, a.slots
        )
