"""Verify frozen V40 transport and authorization inventories without launching."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True)
    a = p.parse_args()
    b = a.bundle
    pins = json.loads((b / "PACKAGE_PINNED.json").read_text())
    assert digest(b / "program.tar.gz") == pins["program_sha"]
    with tarfile.open(b / "program.tar.gz") as archive:
        metadata = json.loads(archive.extractfile("build_meta.json").read())
        assert json.loads((b / "build-meta.json").read_text()) == metadata
        for path, expected in metadata["files"].items():
            assert hashlib.sha256(archive.extractfile(path).read()).hexdigest() == expected, path
    count, variants = 0, {}
    for jobset, expected in pins["manifests"].items():
        assert digest(b / f"fit-manifest-{jobset}.json") == expected
        jobs = json.loads((b / f"fit-manifest-{jobset}.json").read_text())
        seen = set()
        for job in jobs:
            argv = job["kind"]["argv"]
            spec = argv[argv.index("--spec") + 1]
            fold = argv[argv.index("--heldout") + 1]
            seed = int(argv[argv.index("--seed-index") + 1])
            assert fold in ("kadid", "tid2013", "konfig", "cid22_a25")
            assert 0 <= seed < 10 and "aic3" not in argv
            assert "--strict-admission" in argv and "--train-only" in argv
            assert "--local-smoke-budget" not in argv
            assert job["kind"]["program_sha"] == pins["program_sha"]
            assert (spec, fold, seed) not in seen
            seen.add((spec, fold, seed))
            variants[spec] = variants.get(spec, 0) + 1
        required = json.loads((b / f"AUTHORIZATION_REQUIRED-{jobset}.json").read_text())
        for path, expected in required["identities"]["files"].items():
            assert digest(b / path) == expected, path
        template = json.loads((b / f"AUTHORIZATION_TEMPLATE-{jobset}.json").read_text())
        assert all(template[key] is False for key in (
            "reviewed", "source_landed", "pins_pushed", "E30_completed", "control_choice_frozen"))
        assert not (b / f"LAUNCH_AUTHORIZATION-{jobset}.json").exists()
        count += len(jobs)
    assert count == 160 and len(variants) == 4 and set(variants.values()) == {40}
    prepared = json.loads((b / "prepared-manifest-E31-NOT-LAUNCHABLE.json").read_text())
    assert len(prepared) == 40 and json.loads((b / "E31_PENDING.json").read_text())["launchable"] is False
    contract = json.loads((b / "v40-fit-contract.json").read_text())
    assert "uh4" not in json.dumps(contract)
    smokes = json.loads((b / "EXECUTOR_SMOKES.json").read_text())["smokes"]
    assert len(smokes) == 8
    for item in smokes:
        assert digest(b / item["receipt"]) == item["sha256"]
        record = json.loads((b / item["receipt"]).read_text())
        assert record["status"] == "PASS" and record["program_sha"] == pins["program_sha"]
    actual = subprocess.check_output(["docker", "image", "inspect", "-f", "{{.Id}}", pins["image"]], text=True).strip()
    assert actual == pins["image_id"]
    code = '''import hashlib,json,pathlib,os
b=pathlib.Path('/opt/fleet-fits/program')
m=json.loads((b/'build_meta.json').read_text())
for path,expected in m['files'].items():
 assert hashlib.sha256((b/path).read_bytes()).hexdigest()==expected,path
assert hashlib.sha256(pathlib.Path('/usr/local/bin/zenfleet-worker').read_bytes()).hexdigest()==m['worker']['binary_sha256']
assert os.environ['ZEN_FIT_PROGRAM_SHA']==PROGRAM
print(json.dumps({'program_files':len(m['files']),'worker_build':m['worker']['worker_build_id']}))
'''.replace("PROGRAM", repr(pins["program_sha"]))
    image = json.loads(subprocess.check_output(["docker", "run", "--rm", "--network=none", "--cpus=1", "--memory=512m", "--memory-swap=512m", "--entrypoint", "python3", pins["image"], "-c", code], text=True))
    report = dict(status="PASS", launchable_cells=count, prepared_blocked_cells=40,
        variants=variants, executor_smokes=8, maximum_container_peak_bytes=max(s["memory_peak_bytes"] for s in smokes),
        image=image, image_id=actual, program_sha256=pins["program_sha"],
        action="local archive, metadata and image inventory; no worker entrypoint or queue action")
    with (b / "BUNDLE_CHECK.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
