"""E33 registration §6 parity pre-check: fit the control recipe with the E33 program, compare final119 to V40.

The fresh E33 control (40 cells) runs regardless of the outcome; this cell only records whether the E33 program's
`fx1` extension left the baseline path bit-identical. It reuses the E30/V40 parity owner's reproduction-metadata
normalization (`e29_control_parity.stable_repro`), so the only bytes excluded are clock, machine, build and
transport-path identifiers. V40 cells never replace fresh E33 cells after the fact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/rev4_featpot"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from v2_common import sha  # noqa: E402
from e29_control_parity import stable_repro  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", type=Path, required=True, help="frozen V40 bundle (pins, data root v2e29)")
    p.add_argument("--v40-results", type=Path, required=True, help="harvested V40 control results root")
    p.add_argument("--bin-dir", type=Path, required=True, help="E33 program binaries (zensim_mlp_train, ...)")
    p.add_argument("--dest", type=Path, required=True)
    p.add_argument("--fold", choices=("kadid", "tid2013", "konfig", "cid22_a25"), default="kadid")
    p.add_argument("--verify-only", action="store_true", help="audit an existing cell without fitting")
    a = p.parse_args()
    if not a.verify_only:
        a.dest.mkdir(exist_ok=False)
    pins = json.loads((a.bundle / "V40_CONTROL_PINS.json").read_text())
    if sha(a.bundle / "V40_CONTROL_PINS.json") != "4d7acfc887603b123f9631f38435df5477b6e628a372596cb8beb6128bddc84c":
        raise ValueError("V40 control pins changed")
    spec = json.loads((a.bundle / "fit-spec-fitv40-control-20261007.json").read_text())
    cells = [c for c in spec["cells"]
             if c["argv"][c["argv"].index("--heldout") + 1] == a.fold
             and c["argv"][c["argv"].index("--seed-index") + 1] == "0"]
    assert len(cells) == 1
    argv = cells[0]["argv"][:]
    oldcell = a.v40_results / "cells" / cells[0]["name"]
    old = oldcell / "refit/last.bin"
    want = pins["cells"][f"{a.fold}_s0"]
    if sha(oldcell / "result.json") != want["result_sha256"]:
        raise ValueError(f"{oldcell}: V40 result differs from the frozen control pin")
    original = json.loads((oldcell / "result.json").read_text())
    if original["selected_bake_sha256"] != sha(old) or sha(old) != want["bake_sha256"]:
        raise ValueError(f"{old}: V40 final119 differs from the frozen control pin")
    argv[argv.index("--root") + 1] = str(a.bundle / "v2e29")
    argv[argv.index("--data-role-decision") + 1] = str(a.bundle / "v2e29/human_role_decision.json")
    argv[argv.index("--dest") + 1] = str(a.dest / "cell")
    env = dict(os.environ, REV4_V2_BIN_DIR=str(a.bin_dir), ZENSIM_MAX_TIER="v3",
               RAYON_NUM_THREADS="1", OMP_NUM_THREADS="1")
    cmd = [sys.executable, str(REPO / "scripts/rev4_featpot/v2_lodo_mlp.py"), *argv[1:]]
    if not a.verify_only:
        with (a.dest / "driver.log").open("w") as log:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    newcell = a.dest / "cell"
    result = json.loads((newcell / "result.json").read_text())
    assert result["epochs"] == 120 and result["pairs_per_epoch"] == 50000
    assert result["selection"]["selected_epoch"] == 119
    same = {key: result[key] == original[key]
            for key in ("init_seed", "sample_seed", "train_weights", "coverage_leg", "dev_curve")}
    same["strict_table_admission"] = (result["selection"]["strict_table_admission"]
                                      == original["selection"]["strict_table_admission"])
    repros = []
    for model in (old, newcell / "refit/last.bin"):
        inspected = json.loads(subprocess.check_output(
            [str(a.bin_dir / "inspect_qualified_checkpoint"), str(model)], text=True))
        assert inspected["checkpoint_epoch"] == "119" and inspected["admitted_tables"] == 7
        repros.append(stable_repro(inspected["repro"]))
    same["normalized_repro"] = repros[0] == repros[1]
    files = []
    for label, model in (("v40", old), ("e33", newcell / "refit/last.bin")):
        stripped = a.dest / f"{label}-without-repro.bin"
        if not stripped.exists():
            subprocess.run([str(a.bin_dir / "bake_dial_refit"), "strip", "--in", str(model),
                            "--out", str(stripped), "--key", "zentrain.repro"], check=True)
        files.append(stripped)
    identical = files[0].read_bytes() == files[1].read_bytes()
    status = "PASS" if identical and all(same.values()) else "MISMATCH"
    report = dict(schema="e33-control-parity-v1", status=status,
                  control_choice="fresh matched E33 control runs regardless (registration section 6)",
                  cell=f"{a.fold}_s0", epochs=120, pairs_per_epoch=50000, selected_epoch=119,
                  tier="v3", rayon_threads=1, e33_trainer_sha256=sha(a.bin_dir / "zensim_mlp_train"),
                  v40_program_sha256=json.loads((a.bundle / "V40_CONTROL_PINS.json").read_text())["program_sha"],
                  v40_checkpoint_sha256=sha(old), e33_checkpoint_sha256=sha(newcell / "refit/last.bin"),
                  v40_nonrepro_sha256=sha(files[0]), e33_nonrepro_sha256=sha(files[1]),
                  nonrepro_bytes_identical=identical, fields_equal=same,
                  normalized_repro_sha256=[hashlib.sha256(json.dumps(r, sort_keys=True).encode()).hexdigest()
                                           for r in repros],
                  command=cmd)
    (a.dest / "PARITY.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
