#!/usr/bin/env python3
"""Measure `dial.zones` (ladder inversions by codec x quality zone and by
content class x zone) for every board fulleval, and graft it back.

WHY A DRIVER AND NOT A LOOP: the dial grid must match the bake's regime, and a
mismatched grid does not fail loudly for every width — it can score a plausible
number on the wrong feature root (the `--regime 944` hazard in CLAUDE.md). So
this script does not *guess*: for each cell it re-runs `bake_verdict` under one
or more candidate regimes and ACCEPTS only the run whose pooled dial block
(mono_pct / tied_pct / p5 / p95 / reach / dynamic_range / per_codec / curves) is
BYTE-IDENTICAL to the value already on the board. That identity is the proof
that the fresh run used the grid the board cell was measured on; anything else
is reported as NOT MEASURED with the reason, never grafted.

Nothing here computes a statistic. `bake_verdict` measures, `promote_fulleval.py
--graft-dial-zones` writes under its own byte-identity gate.

Usage:
  measure_dial_zones.py [--fulleval-dir D] [--out-dir D] [--only NAME,NAME]
                        [--bv <bake_verdict>] [--jobs N] [--no-graft]
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

REPO = Path(__file__).resolve().parents[2]
DEFAULT_BV = os.environ.get("ZL_BV") or str(REPO / "target/release/bake_verdict")
# Every dial grid on disk, by feature width. The board's cells were NOT all
# measured on one grid — the campaign carries the 2026-05-29 372 grid and its
# two quarantines, the 720/924/944 grids, the wlin7b POOLS (carriers) grid and
# the era-2 tiling variants — so which grid a cell used is a fact to be
# DISCOVERED (by the byte-identity gate below) and recorded, never assumed.
GRIDS = {
    372: [("372-quarantined-v2",
           "/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet"),
          ("372-quarantined",
           "/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined.parquet"),
          ("372-unquarantined-CORRUPT-LADDERS",
           "/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29.parquet")],
    720: [("720", "/mnt/v/output/zensim/v2-eval-720-2026-07-22/dial_grid_720col_2026-07-22.parquet")],
    924: [("924", "/mnt/v/output/zensim/v2-eval-924-2026-07-27/dial_grid_924col_2026-07-28.parquet")],
    944: [("944", "/mnt/v/output/zensim/v2-eval-944-2026-08-01/dial_grid_944col_2026-08-01.parquet"),
          ("944-POOLS-carriers",
           "/mnt/v/output/zensim/wlin7b-2026-08-30/dial_grid_944col_POOLS_2026-08-30.parquet"),
          ("944-era2-r4", "/mnt/v/output/zensim/era2-rank-2026-08-31/grids/dial_grid_944col_r4.parquet"),
          ("944-era1", "/mnt/v/output/zensim/era2-rank-2026-08-31/grids/dial_grid_944col_era1.parquet"),
          ("944-era2-t1024",
           "/mnt/v/output/zensim/era2-rank-2026-08-31/grids/dial_grid_944col_t1024.parquet"),
          ("944-era2-t32",
           "/mnt/v/output/zensim/era2-rank-2026-08-31/grids/dial_grid_944col_t32.parquet")],
}
FEAT_924 = "/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27"
# regime args carrying the FEATURES root (the rank panel needs one corpus to
# load; the dial panel reads --dial-grid regardless).
REGIME = {372: ["--corpora", "cid22"],
          720: ["--regime", "720", "--corpora", "cid22"],
          924: ["--features-root", FEAT_924, "--corpora", "cid22"],
          944: ["--regime", "944", "--corpora", "cid22"]}
# widths with no registered dial grid — reported, never guessed at
NO_GRID = {504}


def candidates(width: int):
    """[(label, argv)] to try, most likely first. A 372-wide bake whose board
    cell was cut on a folded grid is covered because every width's grids are
    tried against the byte-identity gate."""
    order = [width] if width in GRIDS else [372]
    # a board cell for a narrow bake can still have been measured on a wider
    # grid (the era rows) — try the bake's own width first, then the others.
    order += [w for w in (372, 720, 924, 944) if w not in order]
    out = []
    for w in order:
        for label, g in GRIDS.get(w, []):
            if not os.path.exists(g):
                continue
            base = REGIME[w if w in REGIME else 372]
            out.append((label, [*base, "--dial-grid", g]))
            out.append((label + "+cross-regime",
                        [*base, "--cross-regime", "--dial-grid", g]))
    return out


DIAL_SCALARS = ("mono_pct", "tied_pct", "p5", "p95", "reach", "dynamic_range")


def num_eq(a, b) -> bool:
    """Exact structural equality with NUMERIC comparison of leaves.

    Not a JSON-text compare: a value that round-tripped through the board file
    as `0` and out of a fresh verdict as `0.0` is the SAME number, and treating
    it as a difference rejected every cell whose dial was perfectly reproduced
    (measured — that bug cost the first sweep pass). Floats still compare with
    `==`, so a real 1-ULP difference is still a difference."""
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(num_eq(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(num_eq(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a == b
    return a == b


def board_matches(board_dial: dict, fresh_dial: dict) -> str | None:
    """None when the fresh dial reproduces the board's; else the reason it does not."""
    for k in list(board_dial):
        if k == "zones":
            continue
        if not num_eq(board_dial.get(k), fresh_dial.get(k)):
            bv, fv = board_dial.get(k), fresh_dial.get(k)
            if isinstance(bv, (int, float)) and isinstance(fv, (int, float)):
                return f"dial.{k} {bv!r} != {fv!r}"
            return f"dial.{k} differs"
    return None


def measure_one(fe: Path, out_dir: Path, bv: str, graft: bool) -> dict:
    o = json.loads(fe.read_text())
    name = o.get("name")
    rec = {"name": name, "measured": False, "reason": None, "regime": None,
           "n_inputs": o.get("n_inputs")}
    if (o.get("model") or {}).get("kind") == "reference-metric" or o.get("peer"):
        rec["reason"] = "peer reference metric — not a bake, bake_verdict does not run on it"
        return rec
    if (o.get("model") or {}).get("kind") == "ensemble":
        rec["reason"] = "ensemble — the dial panel scores one ZNPR; the board cell's dial " \
                        "came from the ensemble scorer, which has no zones mode"
        return rec
    bd = o.get("dial") or {}
    if not bd or bd.get("mono_pct") is None:
        rec["reason"] = "board cell carries no dial block"
        return rec
    if isinstance(bd.get("zones"), dict):
        rec.update(measured=True, regime="already-present",
                   reason="board cell already carries dial.zones")
        return rec
    bake = o.get("bake")
    if not bake or not Path(bake).exists():
        rec["reason"] = f"bake file absent on disk ({bake})"
        return rec
    w = o.get("n_inputs")
    if w in NO_GRID:
        rec["reason"] = f"no dial grid exists at {w} features"
        return rec
    tried = []
    for label, extra in candidates(w):
        stem = out_dir / f"{name}.zones.json"
        cmd = [bv, "--bake", bake, *extra, "--full-json", str(stem),
               "--output", str(out_dir / f"{name}.zones.md")]
        if stem.exists():
            stem.unlink()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0 or not stem.exists():
            tried.append(f"{label}: rc={r.returncode} {r.stderr.strip()[-90:]}")
            continue
        v = json.loads(stem.read_text())
        why = board_matches(bd, v.get("dial") or {})
        if why:
            tried.append(f"{label}: {why}")
            continue
        rec.update(measured=True, regime=label, argv=" ".join(extra),
                   n_tried=len(tried) + 1, verdict=str(stem))
        if graft:
            g = subprocess.run([sys.executable, str(REPO / "scripts/promote_fulleval.py"),
                                "--verdict", str(stem), "--graft-into", str(fe),
                                "--graft-dial-zones"], capture_output=True, text=True)
            rec["grafted"] = g.returncode == 0
            if g.returncode != 0:
                rec["reason"] = f"graft failed: {g.stderr.strip()[-200:]}"
        return rec
    rec["reason"] = ("no dial grid on disk reproduces the board cell's own dial block "
                     "(same bake, so the cell was measured on a grid this pass cannot see) — "
                     + " | ".join(tried))
    rec["tried"] = tried
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fulleval-dir", default="/mnt/v/output/zensim/reports/fulleval", type=Path)
    ap.add_argument("--out-dir", default="/mnt/v/output/zensim/failure-profiles-2026-08-31/zones",
                    type=Path)
    ap.add_argument("--bv", default=DEFAULT_BV)
    ap.add_argument("--only", default=None, help="comma-separated board names")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--no-graft", action="store_true")
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(a.fulleval_dir.glob("*.fulleval.json"))
    if a.only:
        want = set(a.only.split(","))
        files = [f for f in files if json.loads(f.read_text()).get("name") in want]
    t0 = time.time()
    recs = []
    done = [0]

    def run(f):
        r = measure_one(f, a.out_dir, a.bv, not a.no_graft)
        done[0] += 1
        print(f"[{done[0]}/{len(files)}] {r['name']}: "
              f"{'OK ' + str(r.get('regime')) if r['measured'] else 'SKIP ' + str(r['reason'])[:110]}",
              flush=True)
        return r

    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        recs = list(ex.map(run, files))
    ok = sum(1 for r in recs if r["measured"])
    log = a.out_dir.parent / "dial_zones_measure_log.json"
    log.write_text(json.dumps({"generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                               "n": len(recs), "measured": ok, "records": recs}, indent=1))
    print(f"\nmeasured {ok}/{len(recs)} in {time.time() - t0:.0f}s -> {log}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
