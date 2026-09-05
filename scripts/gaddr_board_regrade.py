#!/usr/bin/env python3
"""G-ADDR board re-grade: replay the 2026-09-04 board grading under the
2026-09-05 negative-tail pin sets, then graft it onto the board fullevals.

Every invocation is RECONSTRUCTED from the 2026-09-04 run's own as-run logs —
the bake path, the ensemble members, the features root, the corpora and the dial
grid are all printed there verbatim — so the re-grade is the same invocation
with one flag changed rather than a fresh guess.  A cell whose invocation cannot
be reconstructed is REPORTED and skipped, never silently approximated.

Nothing here computes a statistic: `bake_verdict` owns the measurement and
`promote_fulleval.py --graft-gaddr` owns the board write (sha-gated,
same-grid-gated).  This file only orchestrates.
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Every negative-tail / identity probe the 2026-09-04 board run could have used,
# keyed by the sha16 bake_verdict prints in its "probe <sha16> not in the
# registry" note.  Cut by scripts/cut_gaddr_negtail_probe.py (the 944 ones) or
# registered in benchmarks/dial_addressability_floor_2026-09-04.json.
D = Path("/mnt/v/output/zensim/dialgate-2026-09-04")
PROBES = {
    "42f93e61c6e5f562": D / "probes944/negtail_probe_944_2026-08-01era.parquet",
    "b73ce10655cb1c16": D / "probes944/negtail_probe_944_era2r4_foldapp2.parquet",
    "5609d19fa10aef81": D / "negtail_probe_372_2026-09-04.parquet",
}
IDENTITY_372 = D / "identity_probe_372_2026-09-04.parquet"

# Where ensemble members live when they are not beside their anchor bake.
MEMBER_ROOTS = [
    "/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28",
    "/mnt/v/output/zensim/wlin7b-2026-08-30/arms",
    "/mnt/v/output/zensim/wave-r4-2026-09-01/bakes",
]

# Cells the 2026-09-04 run graded with `--cross-regime` (the flag is echoed in
# their logs). Reproduced rather than re-decided: dropping it would change what
# the cell measures, and adding it where it was absent would defeat the
# wrong-regime refusal.
CANONICAL_372_SHA16 = "6546c43e6d9572dc"

# The REFERENCE metric's own per-cell table for each grid that has one. Only
# A9r (report-only) reads it: A7r's per-codec exemptions come from the registry.
S2 = "/mnt/v/output/zensim/ssim2-bar-2026-08-31"
GRID_TRUTH = {
    CANONICAL_372_SHA16: f"{S2}/dialcells_ssim2_qv2grid.tsv",
    "694e16c4520a5d41": f"{S2}/dialcells_ssim2_944grid.tsv",   # 944 POOLS
}


def parse_log(path: Path) -> dict | None:
    """Recover one cell's invocation from its 2026-09-04 as-run log."""
    t = path.read_text(errors="replace")
    hdr = re.search(r"^bake_verdict — (.*)$", t, re.M)
    if not hdr:
        return None
    bake = re.search(r"bake=(\S+)", hdr.group(1))
    root = re.search(r"features-root=(\S+)", hdr.group(1))
    corp = re.search(r"corpora=(\S+)", hdr.group(1))
    grid = re.search(
        r"corpus \*\*dial grid[^|]*\| `([0-9a-f]+)` \| \d+ B \| `([^`]+)`", t)
    if not (bake and root and corp and grid):
        return None
    # `ENSEMBLE: N members, …` followed by one basename per line.
    members: list[str] = []
    ens = re.search(r"^ENSEMBLE: (\d+) members[^\n]*\n((?:  \S+\n)+)", t, re.M)
    if ens:
        # A member does NOT necessarily live beside the anchor (HYA_w084's
        # second member is in a different sweep's `arms/`), so resolve each
        # basename against the anchor's directory first and then the known bake
        # roots. An unresolvable member is left as a bare basename so the run
        # FAILS loudly rather than silently scoring a different ensemble.
        base = Path(bake.group(1)).parent
        roots = [base] + [Path(p) for p in MEMBER_ROOTS]
        for m in ens.group(2).splitlines():
            m = m.strip()
            if not m:
                continue
            hit = next((r / m for r in roots if (r / m).is_file()), None)
            members.append(str(hit) if hit else m)
    # Which negative-tail probe: the "not in the registry" note names its sha16;
    # a REGISTERED probe leaves no such note, so fall back to the 372 one only
    # when the grid is the canonical 372 grid.
    probe = None
    m = re.search(r"\*\*A7\*\*: probe ([0-9a-f]+) not in", t)
    if m:
        probe = PROBES.get(m.group(1))
    elif "no --negtail-probe supplied" not in t:
        if grid.group(1) == CANONICAL_372_SHA16:
            probe = PROBES["5609d19fa10aef81"]
    identity = IDENTITY_372 if "no --identity-probe supplied" not in t else None
    cross_regime = "--cross-regime set" in t
    # Ensemble weights are not echoed; the ONE cell that is not equal-weight is
    # HYA_w084 at 0.84/0.16 (measured + recorded in the gate doc §15.4(b), where
    # equal weights refused the same-grid graft and 0.84/0.16 reproduced the
    # board's dial byte-exactly).
    weights = "0.84,0.16" if path.name.startswith("HYA_w084.") else None
    n_in = re.search(r"bake: n_inputs=(\d+)", t)
    # `--regime` is not printed, but it is RECOVERABLE: it selects the corpus
    # FILENAMES inside the features root, and the log names the file it loaded.
    # `ext_cid22val.parquet` ⇒ a wide regime, `cid22_features_372col_*` ⇒ 372.
    # Which wide one follows from the bake's own caller width. Guessing here
    # would silently re-grade a cell on the wrong corpus set, which is the
    # `--regime 944` mis-scoring bug this repo already carries.
    cid = re.search(r'CID22: loaded [^"]*"([^"]+)"', t)
    regime = None
    if cid and Path(cid.group(1)).name.startswith("ext_"):
        n = int(n_in.group(1)) if n_in else 944
        regime = "944" if n > 720 else "720"
    return {
        "regime": regime,
        "bake": bake.group(1),
        "members": members,
        "root": root.group(1),
        "corpora": corp.group(1),
        "grid": grid.group(2),
        "grid_sha16": grid.group(1),
        "probe": str(probe) if probe else None,
        "identity": str(identity) if identity else None,
        "n_inputs": int(n_in.group(1)) if n_in else None,
        "cross_regime": cross_regime,
        "weights": weights,
    }


def grade(args) -> int:
    src, out = Path(args.src), Path(args.out)
    logs = sorted((src / "logs").glob("*.active.log"))
    if not logs:
        print(f"no as-run logs under {src}/logs", file=sys.stderr)
        return 2
    for pins in ("product", "retired"):
        (out / pins).mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    bad, done = [], 0
    for i, lg in enumerate(logs, 1):
        name = lg.name[: -len(".active.log")]
        inv = parse_log(lg)
        if not inv or not Path(inv["bake"]).is_file():
            bad.append((name, "invocation not reconstructible" if not inv
                        else f"bake missing: {inv['bake']}"))
            continue
        for pins in ("product", "retired"):
            cmd = [args.bv, "--bake", inv["bake"], "--features-root", inv["root"],
                   "--corpora", inv["corpora"], "--dial-grid", inv["grid"],
                   "--gaddr-tail-pins", pins,
                   "--gaddr-json", str(out / pins / f"{name}.json")]
            if inv["regime"]:
                cmd += ["--regime", inv["regime"]]
            if inv["members"]:
                cmd += ["--ensemble", ",".join(inv["members"])]
                if inv["weights"]:
                    cmd += ["--ensemble-weights", inv["weights"]]
            if inv["cross_regime"]:
                cmd += ["--cross-regime"]
            if inv["probe"]:
                cmd += ["--negtail-probe", inv["probe"]]
            if inv["identity"]:
                cmd += ["--identity-probe", inv["identity"]]
            # A9r is report-only; the grid truth only fills its per-codec
            # columns, and only grids with a reference cell table have one.
            gt = GRID_TRUTH.get(inv["grid_sha16"])
            if gt and Path(gt).is_file():
                cmd += ["--gaddr-grid-truth", gt]
            with open(out / "logs" / f"{name}.{pins}.log", "w") as fh:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                bad.append((name, f"{pins} rc={rc}"))
        done += 1
        if i % 10 == 0 or i == len(logs):
            print(f"[{i}/{len(logs)}] {name}", flush=True)
    print(f"\ngraded {done} of {len(logs)} cells")
    for n, why in bad:
        print(f"  SKIPPED/FAILED {n}: {why}")
    return 0


def graft(args) -> int:
    """Graft the `product` reads onto the board, and ASSERT the contract-driven
    NOT SHIPPABLE badge count is unchanged — the ruling touched the regression
    tail only, so a moved badge count would be a defect, not a result."""
    src, out, board = Path(args.src), Path(args.out), Path(args.board)
    repo = Path(__file__).resolve().parent.parent

    def contract_fail_count(d: Path) -> tuple[int, int]:
        n_fail = n_seen = 0
        for f in sorted(d.glob("*.json")):
            g = json.loads(f.read_text())
            n_seen += 1
            if any(c["tier"] == "contract" and c["state"] == "fail" for c in g["checks"]):
                n_fail += 1
        return n_fail, n_seen

    before = contract_fail_count(src / "active")
    after = contract_fail_count(out / "product")
    print(f"CONTRACT-fail cells (the NOT SHIPPABLE badge): "
          f"2026-09-04 {before[0]}/{before[1]}  ->  2026-09-05 {after[0]}/{after[1]}")
    if before[0] != after[0]:
        print("REFUSING TO GRAFT: the contract-fail count moved. The 2026-09-05 tail "
              "re-pin touches the REGRESSION tier only; a changed badge count means "
              "something else moved and must be understood first.", file=sys.stderr)
        return 3
    grafted = skipped = 0
    for f in sorted((out / "product").glob("*.json")):
        target = board / f"{f.stem}.fulleval.json"
        if not target.is_file():
            skipped += 1
            continue
        r = subprocess.run(
            [sys.executable, str(repo / "scripts/promote_fulleval.py"),
             "--graft-into", str(target), "--graft-gaddr", str(f)],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  REFUSED {f.stem}: {r.stdout.strip()} {r.stderr.strip()}")
            skipped += 1
        else:
            grafted += 1
    print(f"grafted {grafted}, skipped/refused {skipped}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grade")
    g.add_argument("--bv", required=True)
    g.add_argument("--src", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--grid-truth", required=True)
    g.set_defaults(fn=grade)
    p = sub.add_parser("graft")
    p.add_argument("--src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--board", required=True)
    p.set_defaults(fn=graft)
    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
