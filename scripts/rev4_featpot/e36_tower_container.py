"""Create a bound native-phase container or run a read-only dependency probe.

Creating a phase container does not start it. Generation needs separate live
resource admission; this helper's only immediate execution mode is the probe.
"""

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def remote(host, args, **kwargs):
    return subprocess.run(["ssh", "-T", host, shlex.join(args)], check=True, **kwargs)


def container_argv(spec, stage, name, phase):
    physical = Path(stage["resolved_physical_root"])
    root = Path(spec["root"])
    source_root = physical / "rootfs"
    memory = (
        512 * 1024**2
        if phase == "probe"
        else 8 * 1024**3
        if phase == "first-producer"
        else spec["container_memory_bytes"]
    )
    jobs = 1 if phase == "first-producer" else spec["worker_slots"]
    argv = [
        "docker",
        "create",
        "--name",
        name,
        "--network",
        "none",
        "--read-only",
        "--cpu-shares",
        str(spec["cpu_shares"]),
        "--memory",
        str(memory),
        "--cpuset-cpus",
        "1" if phase in ["probe", "first-producer"] else spec["cpuset"],
        "--pids-limit",
        "1024",
    ]

    def mount(source, dest, readonly=True):
        if "," in str(source) or "," in str(dest):
            raise ValueError("container file path contains a mount separator")
        argv.extend(
            [
                "--mount",
                "type=bind,src="
                + str(source)
                + ",dst="
                + str(dest)
                + (",readonly" if readonly else ""),
            ]
        )

    mount(source_root / str(root).lstrip("/"), root, False)
    mount(physical.parent, "/data", False)
    mount(physical / stage["manifest_name"], "/opt/e36/files.json")
    mount(physical / "home-tmp", "/root/tmp", False)
    mount(source_root / spec["python_source_root"].lstrip("/"), "/opt/e36/python")
    for row in spec["files"]:
        dest = Path(row["destination"])
        if dest.is_relative_to(root):
            if row["source"] != row["destination"]:
                raise ValueError("root file remapping requires explicit admission")
            continue
        if dest.is_relative_to("/opt/e36/python"):
            continue
        mount(source_root / row["source"].lstrip("/"), dest)
    for key, value in dict(
        TMPDIR="/root/tmp",
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH="/opt/e36/python",
        OPENBLAS_NUM_THREADS="1",
        OMP_NUM_THREADS="1",
        RAYON_NUM_THREADS="1",
        PATH="/opt/e36/transport:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        E36_STORAGE_SPEC=spec["runtime"]["path"],
    ).items():
        argv.extend(["--env", key + "=" + value])
    if phase == "probe":
        code = (
            "import sys,json,subprocess,platform; sys.path.insert(0,"
            + repr(str(root / "runtime-v6/program"))
            + "); from e36_data import Runtime; import numpy,PIL; r=Runtime("
            + repr(spec["runtime"]["path"])
            + "); subprocess.run(["
            + repr(str(root / "runtime-v6/executor"))
            + ",'capabilities'],check=True); "
            "subprocess.run(["
            + repr(str(root / "bin/zenmetrics"))
            + ",'--version'],check=True); "
            "print(json.dumps(dict(runtime_sha256=r.spec_sha,python=sys.version,numpy=numpy.__version__,pillow=PIL.__version__,libc=platform.libc_ver(),fits=0,generation_started=False)))"
        )
        argv.extend(
            ["--entrypoint", "/usr/local/bin/python3", spec["image"], "-c", code]
        )
    else:
        if memory % (1024**3):
            raise ValueError("native phase memory plan must use whole GiB")
        argv.extend(
            [
                "--entrypoint",
                "/opt/e36/run-heavy",
                spec["image"],
                "--mem",
                str(memory // 1024**3) + "G",
                "--jobs",
                str(jobs),
                "--",
                "python3",
                "/opt/e36/phase-worker.py",
                "--files",
                "/opt/e36/files.json",
                "--phase",
                phase,
            ]
        )
    return argv


def create(preparation, stage_path, name, phase, out):
    preparation = Path(preparation)
    stage_path = Path(stage_path)
    out = Path(out)
    out.mkdir()
    spec = json.loads((preparation / "files.json").read_text())
    stage = json.loads(stage_path.read_text())
    host = stage["host"]
    if (
        sha(preparation / "files.json") != stage["manifest_sha256"]
        or stage["launched"] is not False
    ):
        raise ValueError("verified whole-phase staging receipt required")
    # A separate writable disk-backed scratch mount; HOME is unchanged.
    relative = Path(stage["resolved_physical_root"]).name
    code = (
        "from pathlib import Path; Path("
        + repr("/data/" + relative + "/home-tmp")
        + ").mkdir(exist_ok=True)"
    )
    remote(host, ["docker", "exec", stage["storage_container"], "python3", "-c", code])
    argv = container_argv(spec, stage, name, phase)
    created = remote(host, argv, capture_output=True, text=True)
    cid = created.stdout.strip()
    if len(cid) != 64 or any(c not in "0123456789abcdef" for c in cid):
        raise ValueError("native container identity missing")
    inspected = remote(host, ["docker", "inspect", cid], capture_output=True, text=True)
    record = json.loads(inspected.stdout)[0]
    result = dict(
        schema="e36-whole-native-phase-container-v1",
        container_id=cid,
        container_name=name,
        phase=phase,
        host=host,
        files_manifest_sha256=stage["manifest_sha256"],
        files_manifest_path=str(preparation / "files.json"),
        stage_receipt_sha256=sha(stage_path),
        image=spec["image"],
        create_argv=argv,
        create_argv_sha256=hashlib.sha256(
            json.dumps(argv, separators=(",", ":")).encode()
        ).hexdigest(),
        resolved_image_id=record["Image"],
        config=record["Config"],
        host_config=record["HostConfig"],
        effective_clock=spec["clock"],
        manifest=spec["first_producer"]
        if phase == "first-producer"
        else spec["manifests"]["additions"],
        generation_started=False,
        fits=0,
    )
    with (out / "container.json").open("x") as f:
        json.dump(result, f, indent=2)
    if phase == "probe":
        with (out / "probe.log").open("x") as log:
            remote(
                host,
                ["docker", "start", "--attach", cid],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        state = json.loads(
            remote(
                host,
                ["docker", "inspect", cid, "--format", "{{json .State}}"],
                capture_output=True,
                text=True,
            ).stdout
        )
        if state["ExitCode"] != 0:
            raise RuntimeError(
                "native dependency probe failed; container and log retained"
            )
        with (out / "probe-receipt.json").open("x") as f:
            json.dump(
                dict(
                    container_id=cid,
                    passed=True,
                    exit_code=state["ExitCode"],
                    fits=0,
                    generation_started=False,
                ),
                f,
                indent=2,
            )
    print(
        json.dumps(
            dict(container_id=cid, phase=phase, generation_started=False, fits=0)
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for n in ["preparation", "stage", "out"]:
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument(
        "--phase", choices=["probe", "first-producer", "additions"], required=True
    )
    a = p.parse_args()
    create(a.preparation, a.stage, a.name, a.phase, a.out)
