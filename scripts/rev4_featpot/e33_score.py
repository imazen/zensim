"""E33 registered assessment (registration section 9.1 and the 9.4 C-vs-A improvement test).

Verifies every installed E33 cell (control, A and C, 40 each; the six full-data cells for the label-free gates)
against the frozen launch packet before any human label is read, then runs the shared V40 panel owner
(`v40_score.assess`) on D1's four SDR sources. The control is E33's own fresh 40-cell control.

Scores use `predict --score-units` through the program's `bake_dial_refit` for every arm (trainer bakes carry no
spline, so score units equal raw `pin - g`). Missing cells, nonfinite metrics or zero-SE degeneracy are INCOMPLETE.
"""

import argparse
import json
import os
from pathlib import Path
import sys

PACKET = Path("/mnt/v/output/zensim/e33-impl-2026-10-09/packet")
BIN_DIR = PACKET.parent / "bin-final2"
V40 = Path("/mnt/v/output/zensim/v40r4-2026-10-08")
ROOT = V40 / "v2e29"
W2_PINS = V40 / "W2_KEY_PINS.json"
POT = Path("/var/tmp/rev4-featpot")
# The predictor must be the program's own binary; v2_common resolves it at import.
os.environ.setdefault("REV4_V2_BIN_DIR", str(BIN_DIR))

import numpy as np  # noqa: E402
from scipy.stats import t as student_t  # noqa: E402

from e24_rev5 import e29_seed_stat  # noqa: E402
from e33_launch import jobset, packet, runtime  # noqa: E402
from v2_common import sha  # noqa: E402
from v2_human_role import PRODUCTION_SOURCES  # noqa: E402
import v40_score  # noqa: E402

LABELS = ("control", "a", "c")


def complete(bundle):
    """Verified installed cells: {label: {(fold, seed): dir}} for the LODO grids, and the six full-data cells."""
    doc = packet(bundle)
    _, tools, inspector = runtime(bundle, doc)
    meta = json.loads((bundle / "build-meta.packer-input.json").read_text())
    if Path(os.environ["REV4_V2_BIN_DIR"]).resolve() != BIN_DIR.resolve() or sha(BIN_DIR / "bake_dial_refit") != meta[
            "binary_mix"]["bake_dial_refit"]["sha256"]:
        raise ValueError("INCOMPLETE: predictor is not the program's bake_dial_refit")
    sys.path.insert(0, str(tools))
    from qualified_fit_contract import trusted_contract, verify_training
    import harvest_fit_cells as hfc

    program = bundle / "program.tar.gz"
    grids, full = {}, {}
    for label in (*LABELS, "full"):
        jobs = json.loads((bundle / f"fit-manifest-{jobset(label)}.json").read_text())
        rows = {}
        for job in jobs:
            kind = job["kind"]
            argv = kind["argv"]
            spec = argv[argv.index("--spec") + 1]
            seed = int(argv[argv.index("--seed-index") + 1])
            fold = argv[argv.index("--heldout") + 1] if "--heldout" in argv else "full"
            key = (spec, seed) if label == "full" else (fold, seed)
            if key in rows or (label != "full" and (fold not in PRODUCTION_SOURCES or not 0 <= seed < 10)):
                raise ValueError(f"INCOMPLETE: duplicate/foreign cell {label} {key}")
            cell = POT / hfc.blob_root(kind) / job["cell"]["image_path"]
            rp, fleet = cell / "result.json", cell / "fleet_receipt.json"
            if not (rp.is_file() and fleet.is_file()):
                raise ValueError(f"INCOMPLETE: missing {label} {key}")
            receipt = json.loads(fleet.read_text())
            r = json.loads(rp.read_text())
            model = cell / Path(r["selected_bake"]).relative_to(Path(argv[argv.index("--dest") + 1]))
            if (receipt["program_sha"] != kind["program_sha"] or receipt["data_sha"] != kind["data_sha"]
                    or receipt["argv_sha"] != kind["argv_sha"] or receipt["tier"]["effective"] != "v3"):
                raise ValueError(f"INCOMPLETE: fleet identity differs {label} {key}")
            for rel, pin in receipt["files"].items():
                if Path(rel).is_absolute() or ".." in Path(rel).parts or sha(cell / rel) != pin:
                    raise ValueError(f"INCOMPLETE: frozen cell file changed {label} {key}")
            expected = trusted_contract(kind, program, inspector)
            if (verify_training(r, model, kind, expected, inspector) != "registered-fit" or r.get("spec") != spec
                    or r.get("head") != "N" or r.get("execution_contract") == "local-smoke"
                    or sha(model) != r.get("selected_bake_sha256") or sha(model) != receipt["selected_bake_sha"]
                    or sha(rp) != receipt["result_sha"]):
                raise ValueError(f"INCOMPLETE: result/model or full budget differs {label} {key}")
            if label != "full":
                if r.get("heldout") != fold or model != cell / "refit/last.bin":
                    raise ValueError(f"INCOMPLETE: LODO selection differs {label} {key}")
                columns = [int(v) for v in (cell / "keep_features.txt").read_text().split()]
                if columns != expected["columns"]:
                    raise ValueError(f"INCOMPLETE: feature order differs {label} {key}")
            rows[key] = cell
        want = 6 if label == "full" else 40
        if len(rows) != want:
            raise ValueError(f"INCOMPLETE: {label} has {len(rows)} of {want} cells")
        (full if label == "full" else grids)[label] = rows
    return grids, full["full"]


def improvement(report):
    """Registration 9.4: C beats A when mean over sources of C - A signed SROCC > +0.002 and one-sided p < .05."""
    panels = report["panels"]
    delta = [[panels["c"][f"{f}_s{s}"]["signed"] - panels["a"][f"{f}_s{s}"]["signed"] for f in PRODUCTION_SOURCES]
             for s in range(10)]
    stat = e29_seed_stat(delta)
    if not (np.isfinite(stat["delta"]) and np.isfinite(stat["se"]) and stat["se"] > 0):
        raise ValueError("INCOMPLETE: degenerate C-vs-A statistic")
    t = stat["delta"] / stat["se"]
    p = float(student_t.sf(t, df=9))
    return dict(signed=stat, t=t, df=9, p_one_sided=p, c_beats_a=bool(stat["delta"] > 0.002 and p < 0.05),
                source_seed_deltas=delta)


def score(bundle, out):
    grids, full = complete(bundle)
    doc = json.loads((bundle / "PACKET.json").read_text())
    protected = tuple(sorted({cell.parents[2] for grid in grids.values() for cell in grid.values()}))
    report = v40_score.assess(grids, "e33", ROOT, W2_PINS, out, protected,
                              dict(packet_sha256=sha(bundle / "PACKET.json"), program_sha256=doc["program_sha"],
                                   predictor_sha256=sha(BIN_DIR / "bake_dial_refit"), units="score (--score-units)"))
    summary = dict(schema="e33-e21-v1", registration="benchmarks/e33_registration_2026-10-09.md section 9.1",
                   as_good={arm: report["decisions"][arm]["as_good"] for arm in ("a", "c")},
                   decisions=report["decisions"], c_vs_a=improvement(report),
                   decision_sha256=sha(out / "decision.json"),
                   full_cells={f"{spec}__s{seed}": str(cell) for (spec, seed), cell in sorted(full.items())})
    (out / "e33_e21.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: summary[k] for k in ("as_good", "decision_sha256")} | {
        "c_beats_a": summary["c_vs_a"]["c_beats_a"]}, allow_nan=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("check", "e21"))
    p.add_argument("--packet", type=Path, default=PACKET)
    p.add_argument("--out", type=Path)
    a = p.parse_args()
    if a.mode == "check":
        grids, full = complete(a.packet)
        print(json.dumps({**{k: len(v) for k, v in grids.items()}, "full": len(full)}))
    else:
        if a.out is None:
            p.error("e21 requires --out")
        score(a.packet, a.out)


if __name__ == "__main__":
    main()
