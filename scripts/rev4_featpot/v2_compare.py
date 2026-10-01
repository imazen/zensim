"""Instrument v2 comparison: seed-paired deltas, permutation excess, hierarchical bootstrap, V1/V2.

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md. For arm A, head h, held-out
source s: per-seed SROCC from the panel owner; reference resamples (B, seed BOOT_SEED) shared by every
model on s; seed resamples (B x 10, seed BOOT_SEED + 1) shared by every arm. Missing cells make a
result INCOMPLETE, never a pass.

  python v2_compare.py --calibration        # instrument-acceptance gate (read before any arm)
  python v2_compare.py --arm c1 [--arm ...]  # candidate arms (only after acceptance passed)
  --human-weight W reads the `<spec>@hW` cells (amendment R3) and writes `*_hW.json`; --jobs N bootstraps cells
  in N processes (each cell's bootstrap is cached beside its result, keyed by the result's sha).
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.zen_stats import panel_batch_indexed, render_indexed_jobs  # noqa: E402
from v2_common import (BOOT_B, BOOT_SEED, CANDIDATES, HEADS, N_PERMS, SOURCE_ORDER, V2, arm_columns, extra_arms, sha)

N_SEEDS = 10
MIN_GAIN = 0.005
REGRESSION = -0.005
SEED_CONSISTENCY = 7
_keys_cache: dict = {}
_ref_draws: dict = {}
_rendered: dict = {}  # source -> the bootstrap jobs' text, rendered once (EFFAUDIT D8)
SUFFIX = ""  # "@h<w>" under --human-weight (amendment R3)
SEED_DRAWS = np.random.default_rng(BOOT_SEED + 1).integers(0, N_SEEDS, size=(BOOT_B, N_SEEDS))


def keys_for(spec: str, source: str):
    family, variant, _ = arm_columns(spec)
    vdir = V2 / "wide" / family / variant
    receipt = json.loads((vdir / "receipt.json").read_text())
    path = vdir / f"{source}.keys.parquet"
    if sha(path) != receipt["legs"][source]["keys_sha256"]:
        raise ValueError(f"{path}: keys changed after the wide receipt")
    keys = pq.read_table(path, columns=["pair_key", "source_row_id", "ref_basename", "target"]).to_pandas()
    ident = (keys.pair_key.astype(str) + "|" + keys.source_row_id.astype(str)).to_numpy()
    base = _keys_cache.setdefault(source, ident)
    if not np.array_equal(base, ident):
        raise ValueError(f"{spec}/{source}: row order differs from the first spec read")
    return keys


def ref_draws(source: str, keys) -> list:
    if source not in _ref_draws:
        refs = keys.ref_basename.astype(str).to_numpy()
        groups = [np.flatnonzero(refs == r) for r in sorted(set(refs))]
        rng = np.random.default_rng(BOOT_SEED)
        _ref_draws[source] = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
                              for _ in range(BOOT_B)]
    return _ref_draws[source]


def rendered_jobs(source: str, keys) -> str:
    if source not in _rendered:
        jobs = [("POINT", "p", "y", None)] + [(f"B{b}", "p", "y", ii) for b, ii in enumerate(ref_draws(source, keys))]
        _rendered[source] = render_indexed_jobs(jobs, ("p", "y"))
    return _rendered[source]


def cell_boot(spec: str, head: str, source: str, i: int, suffix: str | None = None):
    """(point SROCC, bootstrap SROCC array) for one cell, cached beside its result."""
    suffix = SUFFIX if suffix is None else suffix
    cdir = V2 / "cells" / f"{spec}{suffix}__{head}" / f"without_{source}_s{i}"
    res = cdir / "result.json"
    if not res.is_file():
        return None
    cache = cdir / f"boot_B{BOOT_B}_seed{BOOT_SEED}_signed.npz"
    res_sha = sha(res)
    if cache.is_file():
        z = np.load(cache)
        if str(z["result_sha256"]) == res_sha:
            return float(z["point"]), z["boot"]
    cell = json.loads(res.read_text())
    keys = keys_for(spec, source)
    pred = np.asarray(cell["prediction"], dtype=np.float64)
    y = keys.target.to_numpy(dtype=np.float64)
    if len(pred) != len(y):
        raise ValueError(f"{res}: prediction length mismatch")
    rows = panel_batch_indexed({"p": pred, "y": y}, None, stats="srocc", timeout=7200,
                               rendered_jobs=rendered_jobs(source, keys))
    # Signed SROCC: the panel's `srocc` is |rho| (panel.rs), which would hide an inverted model.
    by = {r["label"]: r["srocc_signed"] for r in rows}
    boot = np.asarray([by[f"B{b}"] for b in range(BOOT_B)], dtype=np.float64)
    point = float(by["POINT"])
    np.savez(cache, point=point, boot=boot, result_sha256=res_sha)
    return point, boot


def model(spec: str, head: str, source: str):
    got = [cell_boot(spec, head, source, i) for i in range(N_SEEDS)]
    missing = [i for i, g in enumerate(got) if g is None]
    if missing:
        return None, missing
    return (np.array([g[0] for g in got]), np.stack([g[1] for g in got])), []


def seed_mean(boot: np.ndarray) -> np.ndarray:
    """(10, B) -> (B,) mean over the resampled seed set of each draw."""
    return np.take_along_axis(boot.T, SEED_DRAWS, axis=1).mean(axis=1)


def contrast(arm_spec: str, perm_specs: list[str], head: str, source: str, model_fn=None, keep_boot: bool = False) -> dict:
    """`model_fn(spec, head, source) -> ((point[10], boot[10, B]), missing)`; default: the exploratory cells (`model`).
    v2_confirm_read injects the confirmatory predictions through it; nothing else differs."""
    model_fn = model_fn or model
    base, miss0 = model_fn("r0", head, source)
    arm, miss1 = model_fn(arm_spec, head, source)
    perms, missp = [], []
    for p in perm_specs:
        m, miss = model_fn(p, head, source)
        perms.append(m)
        missp += [f"{p}:{i}" for i in miss]
    missing = [f"r0:{i}" for i in miss0] + [f"{arm_spec}:{i}" for i in miss1] + missp
    if missing:
        return {"status": "INCOMPLETE", "missing": missing}
    (r_pt, r_bt), (a_pt, a_bt) = base, arm
    delta = float(np.mean(a_pt - r_pt))
    d_boot = seed_mean(a_bt - r_bt)
    out = {"status": "OK", "r0_mean": float(r_pt.mean()), "arm_mean": float(a_pt.mean()), "delta": delta,
           "delta_ci95": np.quantile(d_boot, [0.025, 0.975]).tolist(),
           "seeds_arm_above_r0": int((a_pt > r_pt).sum()), "per_seed_delta": (a_pt - r_pt).tolist()}
    if perms:
        p_deltas = [float(np.mean(p[0] - r_pt)) for p in perms]
        p_boot = np.mean([seed_mean(p[1] - r_bt) for p in perms], axis=0)
        out["perm_mean_ci95"] = np.quantile(p_boot, [0.025, 0.975]).tolist()  # R1.2 diagnostic, not a gate
        e_boot = d_boot - p_boot
        e = delta - float(np.mean(p_deltas))
        lo, hi = np.quantile(e_boot, [0.025, 0.975]).tolist()
        out.update({"perm_deltas": p_deltas, "excess": e, "excess_ci95": [lo, hi],
                    "v1_pass": bool(delta >= MIN_GAIN and lo > 0 and delta > max(p_deltas)),
                    "regression": bool(hi < REGRESSION)})
        if keep_boot:  # R2.1: the multiplicity test needs the per-draw excess; off by default (reports stay byte-identical)
            out["excess_boot"] = e_boot
    return out


def family(arm: str, head: str, sources=SOURCE_ORDER, model_fn=None) -> dict:
    perms = [f"{arm}~p{k}" for k in range(1, N_PERMS + 1)]
    per = {s: contrast(arm, perms, head, s, model_fn) for s in sources}
    if any(v["status"] != "OK" for v in per.values()):
        return {"arm": arm, "head": head, "status": "INCOMPLETE", "sources": per}
    passing = [s for s, v in per.items() if v["v1_pass"]]
    regress = [s for s, v in per.items() if v["regression"]]
    v1 = len(passing) >= 2 and not regress
    v2 = v1 and all(per[s]["seeds_arm_above_r0"] >= SEED_CONSISTENCY for s in passing)
    return {"arm": arm, "head": head, "status": "OK", "sources": per, "v1_sources": passing,
            "regressions": regress, "V1": v1, "V2": v2,
            "verdict": ("V1+V2 pass; V3 dial gates next" if v2 and head == "N" else
                        "V1+V2 pass under the free head only; needs a gate-aware design" if v2 else
                        "fails V1" if not v1 else "fails V2 (seed consistency)")}


def calibration() -> dict:
    perms = [f"oracle_lo~p{k}" for k in range(1, N_PERMS + 1)]
    out = {}
    for head in HEADS:
        hi = {s: contrast("oracle_hi", perms, head, s) for s in SOURCE_ORDER}
        lo = {s: contrast("oracle_lo", perms, head, s) for s in SOURCE_ORDER}
        mb = {s: contrast("minus_basic", [], head, s) for s in SOURCE_ORDER}
        complete = all(v["status"] == "OK" for d in (hi, lo, mb) for v in d.values())
        rec = {"oracle_hi": hi, "oracle_lo": lo, "minus_basic": mb, "complete": complete}
        if complete:
            centred = {s: abs(float(np.mean(lo[s]["perm_deltas"]))) < MIN_GAIN for s in SOURCE_ORDER}
            hi_pass = [s for s in SOURCE_ORDER if hi[s]["v1_pass"]]
            # Erratum R1.2 diagnostic (declared before any calibration result was read; does not change "accept"):
            # whether each source's permutation-mean bootstrap CI contains 0, so a centring failure that is within
            # seed/reference noise can be told apart from a biased null. Such a case is reported, not accepted.
            ci_zero = {s: bool(lo[s]["perm_mean_ci95"][0] <= 0 <= lo[s]["perm_mean_ci95"][1]) for s in SOURCE_ORDER}
            accept = bool(len(hi_pass) >= 4 and all(centred.values()))
            rec.update({"oracle_hi_v1_sources": hi_pass, "perm_null_centred": centred,
                        "perm_null_ci_contains_zero": ci_zero,
                        "accept": accept if head == "N" else None,
                        "centring_failure_within_noise": (head == "N" and not accept and len(hi_pass) >= 4
                                                          and all(centred[s] or ci_zero[s] for s in SOURCE_ORDER))})
        out[head] = rec
    out["note"] = ("oracle_hi's permutation null is oracle_lo~p1..p3 (same width, one column); "
                   "acceptance is judged under head N per the amendment")
    return out


def _boot_task(task):
    spec, head, source, i, suffix = task
    cell_boot(spec, head, source, i, suffix)
    return task


def warm(specs: list[str], jobs: int) -> None:
    """Bootstrap every existing cell of `specs` in `jobs` processes, grouped by source so each process renders a
    source's selectors once; results land in the per-cell caches that the contrasts then read."""
    tasks = [(spec, head, source, i, SUFFIX) for source in SOURCE_ORDER for spec in specs for head in HEADS
             for i in range(N_SEEDS)
             if (V2 / "cells" / f"{spec}{SUFFIX}__{head}" / f"without_{source}_s{i}" / "result.json").is_file()]
    with ProcessPoolExecutor(max_workers=jobs) as pool:
        for _ in pool.map(_boot_task, tasks, chunksize=8):
            pass


def main() -> None:
    global SUFFIX
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration", action="store_true")
    ap.add_argument("--arm", action="append", choices=(*CANDIDATES, *extra_arms()["arms"]), default=[])
    ap.add_argument("--root", help="instrument root (default: the Rev3 v2 root); read by v2_common from argv")
    ap.add_argument("--human-weight", type=float, default=None)
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    SUFFIX = "" if args.human_weight is None else f"@h{args.human_weight:g}"
    tag = SUFFIX.replace("@", "_")
    dest = V2 / "compare"
    dest.mkdir(parents=True, exist_ok=True)
    if args.jobs > 1:
        specs = ["r0"]
        if args.calibration:
            specs += ["oracle_hi", "oracle_lo", "minus_basic"] + [f"oracle_lo~p{k}" for k in range(1, N_PERMS + 1)]
        for arm in args.arm:
            specs += [arm] + [f"{arm}~p{k}" for k in range(1, N_PERMS + 1)]
        warm(specs, args.jobs)
    if args.calibration:
        rec = calibration()
        path = dest / f"calibration{tag}.json"
        path.write_text(json.dumps(rec, indent=1) + "\n")
        print(json.dumps({"calibration": str(path), "accept_N": rec["N"].get("accept"),
                          "complete": [rec[h]["complete"] for h in HEADS]}))
    if args.arm:
        gate = dest / f"calibration{tag}.json"
        if not gate.is_file() or not json.loads(gate.read_text())["N"].get("accept"):
            raise SystemExit("instrument acceptance has not passed; no candidate arm may be read")
        for arm in args.arm:
            for head in HEADS:
                rec = family(arm, head)
                (dest / f"{arm}_{head}{tag}.json").write_text(json.dumps(rec, indent=1) + "\n")
                print(json.dumps({"arm": arm, "head": head, "status": rec["status"],
                                  "verdict": rec.get("verdict"), "v1_sources": rec.get("v1_sources")}))


if __name__ == "__main__":
    main()
