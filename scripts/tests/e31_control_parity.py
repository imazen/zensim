"""Compare a full extended-trainer control cell with its frozen E30 nA3 cell.

Original bakes are retained. Only zentrain.repro is removed for the complete
model-byte comparison; its inputs and recipe are checked separately.
"""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable_repro(raw):
    r = copy.deepcopy(raw)
    for key in (
        "timestamp_epoch",
        "cwd",
        "hostname",
        "trainer_head_at_train",
        "trainer_source_dir",
    ):
        r.pop(key, None)
    # Bind input identity by the content hashes retained in each input record.
    # Group names, row counts, weights, modes and all numerical flags stay exact.
    for item in r["inputs"]:
        item["path"] = f"<input:{item['name']}>"
    for table in r["table_admission"]["tables"]:
        table["path"] = "<table>"
        source = table.get("source")
        if isinstance(source, str) and source.endswith(": stored table declaration"):
            table["source"] = "<manifest>: stored table declaration"
    argv = r["argv"]
    argv[0] = "<trainer>"
    for i, token in enumerate(argv[:-1]):
        if token in ("--out", "--keep-features", "--dump-checkpoints-dir"):
            argv[i + 1] = f"<{token}>"
        elif token == "--group":
            name, _, tw, vw, mode = argv[i + 1].split(":", 4)
            argv[i + 1] = f"{name}:<input:{name}>:{tw}:{vw}:{mode}"
    return r


def compare(baseline, candidate, stripper, inspector, dest):
    dest.mkdir(parents=True, exist_ok=False)
    old = json.loads((baseline / "result.json").read_text())
    new = json.loads((candidate / "result.json").read_text())
    if (
        old["spec"] != "sel:59f0bbc2f290@h32:H128:cv16:cf98"
        or old["head"] != "N"
        or old["heldout"] != "kadid"
        or old["seed_index"] != 0
        or old["epochs"] != 120
        or old["pairs_per_epoch"] != 50000
        or old["selection"]["selected_epoch"] != 119
    ):
        raise ValueError("expected the frozen E30 nA3 kadid seed 0 cell")
    if (
        old["selected_bake_sha256"]
        != "00d2c8fe699f5c2d3979506bd047d7013f2f1aa9e96d9c3b3883cee8ca007273"
    ):
        raise ValueError("E30 selected checkpoint pin changed")
    files, repros, originals = [], [], []
    for label, cell, result in [("e30", baseline, old), ("extended", candidate, new)]:
        model = cell / "refit/last.bin"
        if sha(model) != result["selected_bake_sha256"]:
            raise ValueError(f"{label} model differs from its result receipt")
        originals.append(sha(model))
        inspected = json.loads(
            subprocess.check_output([str(inspector), str(model)], text=True)
        )
        (dest / f"{label}-inspection.json").write_text(
            json.dumps(inspected, indent=2) + "\n"
        )
        repros.append(stable_repro(inspected["repro"]))
        stripped = dest / f"{label}-without-repro.bin"
        with (dest / f"{label}-strip.log").open("w") as log:
            subprocess.run(
                [
                    str(stripper),
                    "strip",
                    "--in",
                    str(model),
                    "--out",
                    str(stripped),
                    "--key",
                    "zentrain.repro",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        files.append(stripped)
    for result in (old, new):
        result.pop("selected_bake")
        result.pop("selected_bake_sha256")
    checks = dict(
        result_receipts_identical=old == new,
        complete_non_repro_bytes_identical=files[0].read_bytes()
        == files[1].read_bytes(),
        nonvolatile_repro_identical=repros[0] == repros[1],
        kept_feature_bytes_identical=(baseline / "keep_features.txt").read_bytes()
        == (candidate / "keep_features.txt").read_bytes(),
    )
    report = dict(
        schema="e31-control-parity-v1",
        status="PASS" if all(checks.values()) else "FAIL",
        checks=checks,
        original_sha256=originals,
        stripped_sha256=[sha(p) for p in files],
        stripped_bytes=[p.stat().st_size for p in files],
        normalization="remove only zentrain.repro for bytes; normalize declared run/path fields in repro",
        baseline=str(baseline),
        candidate=str(candidate),
    )
    (dest / "PARITY.json").write_text(json.dumps(report, indent=2) + "\n")
    for label, repro in zip(("e30", "extended"), repros):
        (dest / f"{label}-stable-repro.json").write_text(
            json.dumps(repro, indent=2, sort_keys=True) + "\n"
        )
    print(json.dumps(report), flush=True)
    if report["status"] != "PASS":
        raise AssertionError("E31 control parity mismatch; evidence retained")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "stripper", "inspector", "dest"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    a = ap.parse_args()
    compare(a.baseline, a.candidate, a.stripper, a.inspector, a.dest)


if __name__ == "__main__":
    main()
