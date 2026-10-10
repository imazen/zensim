"""E33 registered label-free gates (registration section 9.2, 8 K1-K5 and 9.3), on full-data seed 0 of A and C.

Every gate uses the owner that produced production seed 0's number, with a positive control on production seed 0
first (`gates/*-ctl`): NEARID (`serve_custom_bake --nearid candidate-<arm>`, N1-N3), `bake_verdict` on the
standard and ladder instrument grids (C1-C6, G-DIAL; `--corpora none`, no labels), the exact-bit identity proof
(`serve_custom_bake --e33-identity`, C5 as registered: exactly 100.0 at every SIMD tier), the STEERFIX
engineering packet with the candidate swapped in (G-STEER), each cell's pack log (K1-K3) plus K4 over every
scored candidate row, and the COSTCMP runtime grid (`costcmp_run.py --e33-grid`). Seeds 1-2 are report-only.

Each subcommand runs one owner; the caller wraps heavy ones in heavy.lock + run-heavy. `summary` reads the
outputs only and writes `GATES.json`.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from v2_common import sha

E = Path("/mnt/v/output/zensim/e33-impl-2026-10-09")
GATES = E / "gates"
BIN = GATES / "bin"
PRODUCTION = Path("/mnt/v/output/zensim/adjudicate-2026-10-07/models/seed0.bin")
PRODUCTION_SHA = "f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400"
NEARID_PACKET = Path("/mnt/v/output/zensim/nearid-2026-10-09/PREREAD_FINAL.json")
STEER_PACKET = Path("/mnt/v/output/zensim/steerfix-2026-10-09/full-packet.json")
INSTRUMENTS = Path("/var/tmp/shippath7/verified/instruments")
NO_HUMAN = Path("/mnt/v/output/zensim/adjudicate-2026-10-07/no-human-corpus")
NO_INTEGRITY = Path("/mnt/v/output/zensim/adjudicate-2026-10-07/no-integrity-grid")
REPO = Path(__file__).resolve().parents[2]
ARMS = {"a": "sel:3b7bd5ebe929@h32:H128:cv16:cf98", "c": "sel:3b7bd5ebe929@h32:H128:cv16:cf98:fx1"}


def candidates(seed=0):
    """{arm: (packed model path, sha256, cell dir)} for full-data seed `seed`, from the verified install."""
    from e33_launch import packet, runtime, jobset
    bundle = E / "packet"
    doc = packet(bundle)
    _, tools, _ = runtime(bundle, doc)
    sys.path.insert(0, str(tools))
    import harvest_fit_cells as hfc
    out = {}
    for job in json.loads((bundle / f"fit-manifest-{jobset('full')}.json").read_text()):
        argv = job["kind"]["argv"]
        spec = argv[argv.index("--spec") + 1]
        if int(argv[argv.index("--seed-index") + 1]) != seed:
            continue
        arm = next(a for a, s in ARMS.items() if s == spec)
        cell = Path("/var/tmp/rev4-featpot") / hfc.blob_root(job["kind"]) / job["cell"]["image_path"]
        r = json.loads((cell / "result.json").read_text())
        dest = Path(argv[argv.index("--dest") + 1])
        model = cell / Path(r["packed_model"]).relative_to(dest)
        if sha(model) != r["packed_model_sha256"] or r.get("execution_contract") != "registered-fit":
            raise ValueError(f"candidate {arm} s{seed} differs from its verified result")
        out[arm] = (model, r["packed_model_sha256"], cell)
    if set(out) != set(ARMS):
        raise ValueError(f"full-data seed {seed} candidates incomplete: {sorted(out)}")
    return out


def run(cmd, log, env=None):
    with open(log, "x") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True, env={**os.environ, **(env or {})})


def done(path):
    """Resumable post-fit: a step whose final output exists is complete; partial outputs refuse (run() opens with x)."""
    if path.exists():
        print(f"complete: {path}")
        return True
    return False


def nearid(arm, seed):
    model, digest, _ = candidates(seed)[arm]
    root = GATES / f"nearid-s{seed}"
    if done(root / f"candidate-{arm}.GATES.json"):
        return
    root.mkdir(parents=True, exist_ok=True)
    run([str(BIN / "serve_custom_bake"), "--nearid", str(NEARID_PACKET), str(root), f"candidate-{arm}", str(model),
         digest], root / f"candidate-{arm}.run.log", {"ZENSIM_FORMULA_REV": "5"})
    run([sys.executable, str(REPO / "scripts/prodqual_label_free.py"), "--nearid-candidate", str(root), arm],
        root / f"candidate-{arm}.gates.log")


def verdict(arm, seed, grid):
    model, _, _ = candidates(seed)[arm]
    out = GATES / f"verdict-s{seed}-{arm}" / grid
    if done(out / "gaddr.json"):
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    run(["just", "--justfile", str(REPO / "benchmarks/adjudicate_land.just"), "--working-directory", str(REPO),
         "bake-verdict-engineering", str(BIN / "bake_verdict"), str(model), str(NO_HUMAN),
         str(INSTRUMENTS / f"{grid}.parquet"), str(INSTRUMENTS / "negtail.parquet"),
         str(INSTRUMENTS / "identity.parquet"), str(NO_INTEGRITY), str(out)], out.parent / f"{grid}.log")


def identity(seed):
    found = candidates(seed)
    out = GATES / f"identity-s{seed}.jsonl"
    if done(GATES / f"identity-s{seed}.log") and len(identity_rows(out)[0]) == 620:
        return
    run([str(BIN / "serve_custom_bake"), "--e33-identity", str(E / "E3_SOURCES.json"),
         str(REPO / "scripts/rev4_featpot/e33_fx1_declaration.json"), str(PRODUCTION), str(out),
         *(str(found[a][0]) for a in sorted(found))], GATES / f"identity-s{seed}.log", {"ZENSIM_FORMULA_REV": "5"})


def steer(arm, seed):
    model, digest, _ = candidates(seed)[arm]
    if done(GATES / f"steer-s{seed}-{arm}.json"):
        return
    doc = json.loads(STEER_PACKET.read_text())
    assert len(doc["cases"]) == 135 and all(c["model_sha256"] == PRODUCTION_SHA for c in doc["cases"])
    for case in doc["cases"]:
        case.update(model=str(model), model_sha256=digest)
    doc["output"] = str(GATES / f"steer-s{seed}-{arm}.json")
    packet = GATES / f"steer-s{seed}-{arm}-packet.json"
    with packet.open("x") as f:
        f.write(json.dumps(doc, indent=1) + "\n")
    run([str(BIN / "steer_instrument"), "metric::bake::steerfix_packet::engineering_packet", "--exact",
         "--nocapture"], GATES / f"steer-s{seed}-{arm}.log",
        {"ZENSIM_FORMULA_REV": "5", "ZENSIM_NEIGHBOUR_EXACT": "1", "STEERFIX_PACKET": str(packet)})


def identity_rows(path):
    """(the 620 source x tier rows, the candidate list) of an --e33-identity output; its last line lists candidates."""
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    rows = [r for r in lines if "reference" in r]
    tail = [r for r in lines if "candidates" in r]
    if len(tail) != 1 or len(rows) + 1 != len(lines):
        raise ValueError(f"unexpected identity output layout: {path}")
    return rows, tail[0]["candidates"]


def runtime_candidates():
    found = candidates(0)
    path = GATES / "runtime-candidates.json"
    want = {arm: {"path": str(model), "sha256": digest} for arm, (model, digest, _) in found.items()}
    if path.exists():
        if json.loads(path.read_text()) != want:
            raise ValueError("runtime candidate pins changed")
    else:
        path.write_text(json.dumps(want, indent=1) + "\n")
    return path


def _verdict_numbers(out):
    g = json.loads((out / "gaddr.json").read_text())
    checks = {c["id"]: dict(measured=c["measured"], state=c["state"], bar=c["bar"]) for c in g["checks"]
              if c["id"] in ("C1", "C2", "C3", "C4", "C5", "C6")}
    grid = g["measured"]["grid"]
    dial = dict(p5=grid["p5"], p95=grid["p95"], mono=grid["mono"])
    dial["pass"] = dial["p5"] <= 25 and dial["p95"] >= 85 and dial["mono"] >= 0.93
    return dict(checks=checks, g_dial=dial)


def _served(out):
    import csv
    with (out / "predictions.tsv").open() as f:
        return [float(r["pred"]) for r in csv.DictReader(f, delimiter="\t")]


def _pack(cell):
    import re
    text = (cell / "pack.log").read_text()
    m = re.search(r"identity-pinned spline: K1 K2 K3 PASS .*?x_floor (\S+) \(floor score (\S+)\).*?calibration rows: "
                  r"(\d+) at/below floor", text)
    if not m:
        raise ValueError(f"INCOMPLETE: no K1-K3 PASS line in {cell / 'pack.log'}")
    return dict(K1_K3="PASS", x_floor=float(m[1]), floor_score=float(m[2]), calibration_rows_at_floor=int(m[3]))


def runtime_summary(dest):
    """Registered guard: no cell slower, i.e. no paired CI wholly above +2% of production."""
    cells, slower = {}, []
    for p in sorted((dest / "timing").iterdir()):
        done = json.loads((p / "COMPLETE.json").read_text())
        if done.get("status") != "PASS":
            raise ValueError(f"INCOMPLETE: timing {p.name}")
        a = json.loads((p / "paired_analysis.json").read_text())
        assert a["baseline_arm"] == "by_v2fy_r5"
        row = {}
        for arm in ("e33_a", "e33_c"):
            c = a["comparisons"][arm]
            bound = 0.02 * c["baseline"]["mean"]
            label = ("slower" if c["ci_lower"] > bound and not c["resolution_limited"] else
                     "faster" if c["ci_upper"] < 0 and not c["resolution_limited"] else "not slower")
            row[arm] = dict(pct_change=c["pct_change"], ci_ns=[c["ci_lower"], c["ci_upper"]], plus2pct_ns=bound,
                            resolution_limited=c["resolution_limited"], label=label)
            if label == "slower":
                slower.append((p.name, arm))
        cells[p.name] = row
    if len(cells) != 6:
        raise ValueError(f"INCOMPLETE: {len(cells)} of 6 timing cells")
    rss = {}
    for arm in ("by_v2fy_r5", "e33_a", "e33_c"):
        rss[arm] = json.loads((dest / "rss" / f"v4x-t1-1024x1024-{arm}.json").read_text())["max_rss_kib"]
    return dict(cells=cells, slower=slower, rss_kib_1024=rss)


def summary():
    found = candidates(0)
    out = {"schema": "e33-label-free-gates-v1", "registration": "benchmarks/e33_registration_2026-10-09.md 9.2/9.3",
           "candidates": {a: dict(path=str(m), sha256=d, model_bytes=m.stat().st_size) for a, (m, d, _) in found.items()},
           "production": dict(sha256=PRODUCTION_SHA, model_bytes=PRODUCTION.stat().st_size), "arms": {}}
    import struct
    hundred = struct.unpack("<q", struct.pack("<d", 100.0))[0]
    id_rows, id_candidates = identity_rows(GATES / "identity-s0.jsonl")
    if len(id_rows) != 620:
        raise ValueError(f"INCOMPLETE: identity proof has {len(id_rows)} of 620 rows")
    order = [c["sha256"] for c in id_candidates]
    for arm, (model, _, cell) in found.items():
        g = {}
        n = json.loads((GATES / "nearid-s0" / f"candidate-{arm}.GATES.json").read_text())
        g["N1"], g["N2"], g["N3"] = ({k: v for k, v in n[x].items() if k != "failing"} | {"failing": n[x].get("failing")}
                                     for x in ("N1", "N2", "N3"))
        std = _verdict_numbers(GATES / f"verdict-s0-{arm}" / "standard")
        lad = _verdict_numbers(GATES / f"verdict-s0-{arm}" / "ladder")
        g["C2"] = dict(standard=std["checks"]["C2"]["measured"], ladder=lad["checks"]["C2"]["measured"],
                       pass_=std["checks"]["C2"]["measured"] <= 0.05 and lad["checks"]["C2"]["measured"] <= 0.05)
        g["C5"] = dict(rule="all 38 raw identities served at exactly 100.0 (section 7 E8, 8 K3), every SIMD tier",
                       rows=len(id_rows),
                       pass_=all(r["candidate_score_bits"][order.index(found[arm][1])] == hundred for r in id_rows),
                       band_report=std["checks"]["C5"])
        steer_rows = json.loads((GATES / f"steer-s0-{arm}.json").read_text())["rows"]
        passed = sum(r["pass"] for r in steer_rows)
        g["G-STEER"] = dict(cases=len(steer_rows), passed=passed, pass_=len(steer_rows) == 135 and passed >= 128,
                            failing=[r["key"] for r in steer_rows if not r["pass"]])
        pack = _pack(cell)
        served = (_served(GATES / f"verdict-s0-{arm}" / "standard") + _served(GATES / f"verdict-s0-{arm}" / "ladder")
                  + [json.loads(line)["served_score"] for line in
                     (GATES / "nearid-s0" / f"candidate-{arm}.jsonl").read_text().splitlines()])
        at_floor = sum(s <= pack["floor_score"] for s in served)
        k5 = all(std["checks"][c]["state"] == "pass" for c in ("C1", "C3", "C4", "C6")) and std["g_dial"]["pass"]
        g["output_stage"] = dict(**pack, K4_rows_at_or_below_floor=at_floor, K4_rows=len(served),
                                 K5_standard=std, K5_ladder=lad, pass_=at_floor == 0 and k5)
        g["eligible_label_free_without_runtime"] = all(g[k]["pass" if k.startswith("N") else "pass_"]
                                                       for k in ("N1", "N2", "N3", "C2", "C5", "G-STEER",
                                                                 "output_stage"))
        out["arms"][arm] = g
    rt = GATES / "runtime"
    if (rt / "timing").is_dir():
        out["runtime"] = runtime_summary(rt)
        for arm in found:
            out["arms"][arm]["runtime_pass"] = not any(a == f"e33_{arm}" for _, a in out["runtime"]["slower"])
    else:
        out["runtime"] = "INCOMPLETE: not measured"
    (GATES / "GATES.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({a: {k: v for k, v in g.items() if k in ("eligible_label_free_without_runtime", "runtime_pass")}
                      for a, g in out["arms"].items()}))


def control_repack():
    """Report-only (registration 9.5 / 8): frozen production s0-s2 re-packed with the E33 output stage."""
    from v2_common import FITBIN
    from v2_production_pack import pack_production
    cells = Path("/var/tmp/rev4-featpot/d1-results/confirm/cells/sel:59f0bbc2f290@h32:H128:cv16:cf98__N")
    anchor = Path("/mnt/v/output/zensim/v40r4-2026-10-08/v2e29/wide/main/real/cid22_fit.parquet")
    root = GATES / "control-repack"
    if sha(FITBIN) != json.loads((E / "packet/build-meta.packer-input.json").read_text())[
            "binary_mix"]["bake_dial_refit"]["sha256"]:
        raise ValueError("bake_dial_refit is not the program's binary (set REV4_V2_BIN_DIR)")
    out = {}
    for seed in range(3):
        cell = cells / f"full_s{seed}"
        r = json.loads((cell / "result.json").read_text())
        dest = root / f"s{seed}"
        (dest / "refit").mkdir(parents=True)
        try:
            packed, status = pack_production(cell / "refit/last.bin", dest, anchor, e33_output_stage=True), "PACKED"
        except Exception as e:  # a K refusal is the report-only result, not a crash
            packed, status = {"error": str(e)}, "REFUSED"
        log = (dest / "pack.log").read_text().splitlines()
        out[f"s{seed}"] = dict(status=status, production_packed_sha256=r["packed_model_sha256"],
                               production_dense_sha256=r["dense_model_sha256"],
                               dense_identical=packed.get("dense_model_sha256") == r["dense_model_sha256"],
                               stage_line=next((x for x in log if "identity-pinned spline" in x), log[-1] if log else None),
                               **packed)
    (root / "CONTROL_REPACK.json").write_text(json.dumps(dict(
        schema="e33-control-repack-v1", report_only=True, anchor_sha256=sha(anchor), fitbin_sha256=sha(FITBIN),
        seeds=out), indent=2) + "\n")
    print(json.dumps({k: (v["status"], v["dense_identical"], v["stage_line"]) for k, v in out.items()}, indent=1))


def train_fragility():
    """Report-only (registration 9.5): TRAIN correlation of each fragility factor f with f^2 (feature columns only)."""
    import numpy as np
    import pyarrow.parquet as pq
    from v2_common import fx1_declaration
    table = Path("/mnt/v/output/zensim/v40r4-2026-10-08/v2e29/wide/main/real/safesyn_fit.parquet")
    ids = sorted({b for _, b in fx1_declaration()["products"]})
    cols = pq.read_table(table, columns=[f"f{i}" for i in ids])
    out = {}
    for i in ids:
        f = cols[f"f{i}"].to_numpy(zero_copy_only=False).astype(np.float64)
        f = f[np.isfinite(f)]
        out[f"f{i}"] = dict(rows=int(f.size), min=float(f.min()), max=float(f.max()),
                            pearson_f_f2=float(np.corrcoef(f, f * f)[0, 1]))
    result = dict(schema="e33-train-fragility-v1", report_only=True, table=str(table), table_sha256=sha(table),
                  columns_read=[f"f{i}" for i in ids], labels_read=False, ids=out)
    (GATES / "TRAIN_FRAGILITY.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: round(v["pearson_f_f2"], 6) for k, v in out.items()}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("candidates", "nearid", "verdict", "identity", "steer", "runtime-candidates",
                                    "control-repack", "train-fragility", "summary"))
    p.add_argument("--arm", choices=sorted(ARMS))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid", choices=("standard", "ladder"))
    a = p.parse_args()
    if a.mode == "candidates":
        print(json.dumps({k: [str(v[0]), v[1]] for k, v in candidates(a.seed).items()}))
    elif a.mode == "nearid":
        nearid(a.arm, a.seed)
    elif a.mode == "verdict":
        verdict(a.arm, a.seed, a.grid)
    elif a.mode == "identity":
        identity(a.seed)
    elif a.mode == "steer":
        steer(a.arm, a.seed)
    elif a.mode == "train-fragility":
        train_fragility()
    elif a.mode == "control-repack":
        control_repack()
    elif a.mode == "runtime-candidates":
        print(runtime_candidates())
    else:
        summary()


if __name__ == "__main__":
    main()
