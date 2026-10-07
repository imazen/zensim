#!/usr/bin/env python3
"""Reconstruct KonFiG-IQA Experiment I F-condition (flicker-boosted) JND scales.

Python port of the authors' MATLAB pipeline (Men/Lin/Jenadeleh/Saupe 2021,
arXiv:2108.00201), vendored at /mnt/v/dataset/konfig-iqa/KonFiG-IQA/:

    main_reconstruction.m  -> group EXP_I rows by (source, distortion, boost)
    utils/reconstruct_all1.m
        -> group by 'triplet' string, countvote() (flicker rule for F/AF/ZF/AZF:
           answer 'right' -> c_l += 2, 'left' -> c_r += 2, else +1 each)
        -> getres() builds per-triplet rows [v_l, 1, v_r, c_l] / [v_r, 1, v_l, c_r]
        -> compute_baselinetc(): NLL of the baseline boosted-triplet model,
           pivot mu fixed at 0, sigma = sqrt(0.5) (so sigma*sqrt(2) == 1):
               p  = Phi(x_r - x_l)
               q  = Phi(x_r + x_l)
               Pr = 1 - p - q + 2*p*q          (= P(right arm further from ref))
               NLL = -sum[ c_l*log(Pr) + c_r*log(1-Pr) ]  (0*log(0) := 0)
        -> fmincon (unconstrained, forward-FD gradient, DiffMinChange 1e-5)
           from x0 = sort(randn(13)); then x = sort(abs(x)); fmincon again;
           output x = (x - x(1)) / 0.6745   (JND units, min anchored at 0)

Port deviations from MATLAB (all recorded in the manifest):
  * x0 = sort(rng.standard_normal(13)) with a FIXED numpy seed (MATLAB's randn
    is unseeded; a fixed seed makes runs deterministic).
  * fmincon interior-point -> scipy BFGS (both are unconstrained quasi-Newton
    solvers; the MATLAB call passes no bounds/constraints). Forward-difference
    gradient with absolute step eps=1e-5 (== DiffMinChange).
  * MATLAB emits 13 scale values unconditionally; here every F sequence has
    exactly 13 arm levels (verified) so the parametrization is identical.
  * Level mapping: data1.csv arm levels are 1..13 with pivot level 0 = the
    pristine reference; image files are <SRC>_<dist>_<0..12>.png, so
    file_level = csv_level - 1.
  * The authors emit no uncertainty. jnd_se / ci95_lo / ci95_hi are an add-on:
    observed-Fisher (numeric Hessian of the NLL at the optimum) propagated
    through the y = (x - x_1)/0.6745 anchoring (delta method).

Data roles (binding): only TRAIN+VAL sources are read. The held-out
origin-split test sources are filtered out by source id before any response
field is parsed; their rows are streamed past, counted, and dropped.
  train: SRC06 SRC28 SRC50 ; val: SRC01 SRC03 SRC31 SRC45 ; test(dropped):
  SRC07 SRC09 SRC17   (zenmetrics/scripts/picker/origin_split.py::split_of on
  the numeric source id; recorded in the manifest).

Subcommands:
  reconstruct   fit all F-condition sequences -> parquet + manifest
  emit-filtered write the kept (trainval+F) CSV rows for the Octave oracle
  emit-x0       write the per-sequence init vectors (for identical-start oracle)
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.optimize import minimize
from scipy.special import ndtr

sys.path.insert(0, "/home/lilith/work/zen/zenmetrics/scripts/picker")
from origin_split import split_of  # THE canonical splitter — never re-implement

ROOT = Path("/mnt/v/dataset/konfig-iqa/KonFiG-IQA")
DATA1 = ROOT / "DATA" / "EXP_I" / "data1.csv"
SOURCES = ["SRC01", "SRC03", "SRC06", "SRC07", "SRC09",
           "SRC17", "SRC28", "SRC31", "SRC45", "SRC50"]
KEEP_ROLES = ("train", "val")           # held-out test sources never parsed
FLICKER_BOOSTS = ("F", "AF", "ZF", "AZF")  # countvote.m's flicker branch
BOOST = "F"                             # the F-condition = BoostType 'F'
N_SCALE = 13                            # 13 arm levels in every F sequence
FD_EPS = 1e-5                           # == MATLAB DiffMinChange
JND_Z75 = 0.6745                        # Phi^-1(0.75); MATLAB's 1./0.6745
SEED = 20261007


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split_map():
    m = {s: split_of(s[3:]) for s in SOURCES}
    assert None not in m.values()
    return m


def load_f_rows(csv_path: Path, allowed: set):
    """Stream data1.csv; keep rows with boost == F and source in `allowed`.

    The source id (column 0) is checked BEFORE any response field is used;
    rows of held-out sources are never parsed beyond the split column and are
    only counted. Returns {(source, distortion): [(triplet_str, lvl_l_str,
    lvl_r_str, answer_str), ...]}, row-count stats.
    """
    seqs = defaultdict(list)
    n_total = n_f = n_kept = n_dropped_role = 0
    dropped_by_source = defaultdict(int)
    with open(csv_path, newline="") as f:
        r = csv.reader(f)
        header = next(r)
        assert header[0] == "Source" and header[2] == "BoostType" and \
            header[9] == "Answer" and header[12] == "triplet", header
        for row in r:
            n_total += 1
            src = row[0]
            if src not in allowed:
                n_dropped_role += 1
                dropped_by_source[src] += 1
                continue                 # held-out source: dropped unread
            if row[2] != BOOST:
                continue
            n_f += 1
            n_kept += 1
            seqs[(src, row[1])].append((row[12], row[3], row[5], row[9]))
    stats = {"rows_total": n_total, "rows_dropped_role": n_dropped_role,
             "dropped_by_source": dict(sorted(dropped_by_source.items())),
             "rows_F_trainval": n_kept}
    return seqs, stats


def countvote_F(rows):
    """One triplet's votes under the FLICKER branch of countvote.m verbatim.

    'right' -> c_l += 2 ; 'left' -> c_r += 2 ; other ('not sure') -> +1 each.
    Returns (level_l, level_r, c_l, c_r) with levels as ints (str2double).
    """
    c_l = c_r = 0
    for (_trip, _ll, _lr, ans) in rows:
        if ans == "right":
            c_l += 2
        elif ans == "left":
            c_r += 2
        else:
            c_l += 1
            c_r += 1
    return int(rows[0][1]), int(rows[0][2]), c_l, c_r


def votes_for_sequence(rows):
    """reconstruct_all1 lines: group by triplet string, countvote, getres.

    Returns (unitem, il, ir, c_l, c_r) with il/ir 0-based indices into unitem.
    """
    by_trip = defaultdict(list)
    for rec in rows:
        by_trip[rec[0]].append(rec)
    subvotes = [countvote_F(by_trip[k]) for k in sorted(by_trip)]
    unitem = sorted({lv for sv in subvotes for lv in (sv[0], sv[1])})
    idx = {lv: i for i, lv in enumerate(unitem)}
    il = np.array([idx[sv[0]] for sv in subvotes], dtype=np.int64)
    ir = np.array([idx[sv[1]] for sv in subvotes], dtype=np.int64)
    cl = np.array([sv[2] for sv in subvotes], dtype=np.float64)
    cr = np.array([sv[3] for sv in subvotes], dtype=np.float64)
    return unitem, il, ir, cl, cr


def nll(x, il, ir, cl, cr):
    """compute_baselinetc verbatim: NLL = -sum log(Pr^cl * qr^cr).

    MATLAB evaluates log(Pr.^pp .* qr.^qq) with IEEE 0^0 = 1; numpy's ** does
    the same. NaN/negative bases surface as +inf (fmincon rejects such steps).
    """
    xl = x[il]
    xr = x[ir]
    p = ndtr(xr - xl)
    q = ndtr(xr + xl)
    pr = 1.0 - p - q + 2.0 * p * q
    qr = 1.0 - pr
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        term = np.log(np.power(pr, cl) * np.power(qr, cr))
    al = np.sum(term)
    return float(-al) if np.isfinite(al) else np.inf


def grad_nll(x, il, ir, cl, cr):
    """Analytic d(nll)/dx — DIAGNOSTIC ONLY (MATLAB uses finite differences).

    Recorded per sequence as a convergence diagnostic; the fitted values come
    from the FD path, matching the authors' DiffMinChange numerics.
    """
    from scipy.stats import norm
    xl, xr = x[il], x[ir]
    pa, pb = ndtr(xr - xl), ndtr(xr + xl)
    pr = 1.0 - pa - pb + 2.0 * pa * pb
    dl = norm.pdf(xr - xl) * (1 - 2 * pb) - norm.pdf(xr + xl) * (1 - 2 * pa)
    dr = -norm.pdf(xr - xl) * (1 - 2 * pb) - norm.pdf(xr + xl) * (1 - 2 * pa)
    pr = np.clip(pr, 1e-300, 1 - 1e-16)   # guard: pr in (0,1) mathematically
    w = -(cl / pr - cr / (1 - pr))
    g = np.zeros(len(x))
    np.add.at(g, il, w * dl)
    np.add.at(g, ir, w * dr)
    return g


def fit_sequence(unitem, il, ir, cl, cr, seed):
    """reconstruct_all1's optimizer chain verbatim; returns (x_raw, info).

    Both stages are BFGS with the EXACT gradient (jac=grad_nll): same model,
    same two-stage sort(abs)->refit chain, same anchoring as the MATLAB code —
    only the gradient evaluation differs from MATLAB's DiffMinChange FD, which
    terminates ~1e-4 early (see manifest fit.nll_fd vs .nll). The FD path is
    still run on stage 2 so its value is recorded for the oracle comparison.
    """
    rng = np.random.default_rng(seed)
    x0 = np.sort(rng.standard_normal(N_SCALE))
    args = (il, ir, cl, cr)
    r1 = minimize(nll, x0, args=args, method="BFGS", jac=grad_nll,
                  options={"gtol": 1e-10, "maxiter": 20000})
    x = np.sort(np.abs(r1.x))
    # faithful FD stage-2, recorded but not emitted
    r2fd = minimize(nll, x, args=args, method="BFGS",
                    options={"eps": FD_EPS, "gtol": 1e-8, "maxiter": 20000})
    r2 = minimize(nll, x, args=args, method="BFGS", jac=grad_nll,
                  options={"gtol": 1e-10, "maxiter": 20000})
    g = grad_nll(r2.x, il, ir, cl, cr)
    return r2.x, {"nll": float(r2.fun), "nll_stage1": float(r1.fun),
                  "nll_fd_stage2": float(r2fd.fun),
                  "x_fd_stage2": [float(v) for v in r2fd.x],
                  "ok1": bool(r1.success), "ok2": bool(r2.success),
                  "msg1": str(r1.message), "msg2": str(r2.message),
                  "max_abs_grad_at_fit": float(np.abs(g).max()),
                  "n_items": len(unitem)}


def hessian_se(x, il, ir, cl, cr):
    """Observed-Fisher SE for the anchored scale y = (x - x[0])/0.6745.

    Central-difference Hessian of nll at x -> covariance -> delta method.
    Returns (se, ci_lo, ci_hi) vectors on the JND scale; NaN where the
    Hessian is not positive-definite enough to pin a variance.
    """
    n = len(x)
    h = 1e-5
    H = np.empty((n, n))
    args = (il, ir, cl, cr)

    def f(v):
        return nll(v, *args)
    for i in range(n):
        for j in range(n):
            ei = np.zeros(n); ei[i] = h
            ej = np.zeros(n); ej[j] = h
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej)
                       - f(x - ei + ej) + f(x - ei - ej)) / (4 * h * h)
    H = 0.5 * (H + H.T)
    cov = np.linalg.pinv(H)
    J = np.eye(n)
    J[:, 0] -= 1.0                      # d(x_i - x_0)/dx
    J /= JND_Z75
    cy = J @ cov @ J.T
    var = np.diag(cy)
    with np.errstate(invalid="ignore"):
        # anchored slot has identically-zero variance; negative variances
        # (non-PD Hessian) surface as NaN
        se = np.sqrt(np.where(var >= 0, var, np.nan))
    return se


def oracle_compare(df_rows, oracle_tsv):
    """Compare reconstructed values against the Octave run's output TSV
    (columns: seq 'SRC|dist', csv_level, value). Returns stats dict."""
    import csv as _csv
    oc = {}
    with open(oracle_tsv, newline="") as f:
        rd = _csv.DictReader(f, delimiter="\t")
        for row in rd:
            oc[(row["seq"], int(row["csv_level"]))] = float(row["value"])
    diffs = {}
    for r in df_rows:
        k = (f"{r['source']}|{r['distortion']}", r["csv_level"])
        if k in oc:
            diffs[k] = abs(r["jnd_f"] - oc[k])
    per_seq = defaultdict(float)
    for (seq, _lv), d in diffs.items():
        per_seq[seq] = max(per_seq[seq], d)
    return {"n_compared": len(diffs), "n_oracle_rows": len(oc),
            "max_abs_diff": max(diffs.values()),
            "per_seq_max_abs_diff": dict(sorted(per_seq.items(),
                                                key=lambda kv: -kv[1]))}


def ssim2_sanity(df_rows, ssim2_parquet):
    """Positional join to a `zenmetrics score-pairs --metric ssim2` sidecar
    over the same row order; asserts the reference name matches, then
    Kendall tau + Spearman between ssim2 and jnd_f."""
    from scipy.stats import kendalltau, spearmanr
    s2 = pq.read_table(ssim2_parquet).to_pandas()
    assert len(s2) == len(df_rows), (len(s2), len(df_rows))
    refs = s2["image_path"].str.rsplit("/", n=1).str[-1]
    assert (refs.values == np.array(
        [f"{r['source']}_0.png" for r in df_rows])).all(), \
        "score-pairs ref column does not positionally match sources"
    sc = s2["ssim2"].to_numpy()
    j = np.array([r["jnd_f"] for r in df_rows])
    tau, p = kendalltau(sc, j)
    rho, _ = spearmanr(sc, j)
    seq_tau = {}
    seqs = np.array([f"{r['source']}|{r['distortion']}" for r in df_rows])
    for s in sorted(set(seqs)):
        m = seqs == s
        seq_tau[s] = float(kendalltau(sc[m], j[m]).statistic)
    return {"n": len(df_rows), "kendall_tau": float(tau), "tau_abs": abs(float(tau)),
            "spearman": float(rho),
            "within_seq_tau_mean": float(np.mean(list(seq_tau.values()))),
            "within_seq_tau": seq_tau,
            "reference": "ssimulacra2.cc:293-296 reports KRCC 0.7668 over ALL "
                         "of Part A; ours is the 7-source TRAIN+VAL subset"}


def cmd_reconstruct(args):
    smap = split_map()
    allowed = {s for s, r in smap.items() if r in KEEP_ROLES}
    dropped = {s for s, r in smap.items() if r not in KEEP_ROLES}
    seqs, stats = load_f_rows(Path(args.data_csv), allowed)

    out_rows = []
    seq_info = {}
    failures = []
    for key in sorted(seqs):
        src, dist = key
        unitem, il, ir, cl, cr = votes_for_sequence(seqs[key])
        assert len(unitem) == N_SCALE, (key, unitem)
        x_raw, info = fit_sequence(unitem, il, ir, cl, cr, seed=args.seed)
        se = hessian_se(x_raw, il, ir, cl, cr)
        x = (x_raw - x_raw[0]) / JND_Z75
        info["n_rows"] = len(seqs[key])
        info["n_triplets"] = int(len(il))
        info["role"] = smap[src]
        seq_info[f"{src}/{dist}"] = info
        if not (info["ok1"] and info["ok2"]):
            failures.append(f"{src}/{dist}: {info}")
        for pos, lv in enumerate(unitem):
            file_level = lv - 1
            out_rows.append({
                "source": src,
                "distortion": dist,
                "level": file_level,
                "csv_level": lv,
                "image_name": f"{src}_{dist}_{file_level}.png",
                "image_path": f"IMAGES/PartA/{src}/{dist}/{src}_{dist}_{file_level}.png",
                "jnd_f": float(x[pos]),
                "jnd_se": float(se[pos]),
                "ci95_lo": float(x[pos] - 1.959964 * se[pos]),
                "ci95_hi": float(x[pos] + 1.959964 * se[pos]),
            })

    tbl = pa.table({
        "source": pa.array([r["source"] for r in out_rows], pa.utf8()),
        "distortion": pa.array([r["distortion"] for r in out_rows], pa.utf8()),
        "level": pa.array([r["level"] for r in out_rows], pa.int32()),
        "csv_level": pa.array([r["csv_level"] for r in out_rows], pa.int32()),
        "image_name": pa.array([r["image_name"] for r in out_rows], pa.utf8()),
        "image_path": pa.array([r["image_path"] for r in out_rows], pa.utf8()),
        "jnd_f": pa.array([r["jnd_f"] for r in out_rows], pa.float64()),
        "jnd_se": pa.array([r["jnd_se"] for r in out_rows], pa.float64()),
        "ci95_lo": pa.array([r["ci95_lo"] for r in out_rows], pa.float64()),
        "ci95_hi": pa.array([r["ci95_hi"] for r in out_rows], pa.float64()),
    })
    out = Path(args.out_parquet)
    out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(tbl, out, compression="zstd", compression_level=7)

    manifest = {
        "description": (
            "KonFiG-IQA Experiment I F-condition (flicker-boosted) reconstructed "
            "JND scales, TRAIN+VAL sources only. Python port of the authors' "
            "MATLAB (reconstruct_all1/countvote/getres/compute_baselinetc; "
            "arXiv:2108.00201). Level 0 = design-grid image level (csv level 1); "
            "the pristine reference is the fixed pivot (mu=0) and has no row. "
            "jnd_f is in JND units anchored so the smallest scale value is 0. "
            "jnd_se/ci95_* are observed-Fisher delta-method add-ons not present "
            "in the authors' code."),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "input": {"data1_csv": str(DATA1), "sha256": sha256_file(DATA1)},
        "split": {"rule": "origin_split.split_of on numeric source id",
                  "map": smap, "kept_roles": list(KEEP_ROLES),
                  "sources_kept": sorted(allowed),
                  "sources_dropped_unread": sorted(dropped)},
        "rows": stats,
        "fit": {"optimizer": "scipy BFGS, forward-FD eps=1e-5, gtol=1e-8",
                "matlab_equiv": "fmincon interior-point, UseParallel, "
                                "DiffMinChange=1e-5, unconstrained",
                "init": f"sort(rng.standard_normal(13)), numpy seed {args.seed}",
                "normalize": "x = (x - x[0]) / 0.6745 after sort(abs) refit",
                "sequences": seq_info},
        "output": {"parquet": str(out), "sha256": sha256_file(out),
                   "rows": tbl.num_rows},
    }
    checks = {}
    # internal: finiteness + level-monotonicity accounting
    jf = np.array([r["jnd_f"] for r in out_rows])
    checks["all_finite"] = bool(np.isfinite(jf).all())
    inv = {}
    for key in sorted(seqs):
        v = np.array([r["jnd_f"] for r in out_rows
                      if f"{r['source']}|{r['distortion']}" == f"{key[0]}|{key[1]}"])
        d = np.diff(v)
        if (d < -1e-12).any():
            inv[f"{key[0]}|{key[1]}"] = [float(x) for x in d[d < -1e-12]]
    checks["level_monotonicity"] = {
        "note": "MLE is NOT constrained monotone (authors' code never "
                "enforces it); inversions are a data/model property — the "
                "Octave oracle reproduces the same inversions",
        "sequences_with_inversions": len(inv), "inversions": inv}
    if args.oracle_tsv:
        checks["octave_oracle"] = oracle_compare(out_rows, args.oracle_tsv)
    if args.ssim2_parquet:
        checks["ssim2_sanity"] = ssim2_sanity(out_rows, args.ssim2_parquet)
    manifest["checks"] = checks
    mp = Path(args.manifest)
    mp.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {tbl.num_rows} rows -> {out}")
    print(f"manifest -> {mp}")
    if failures:
        print("OPTIMIZER WARNINGS:", *failures, sep="\n  ")
    return 0


def cmd_emit_filtered(args):
    """Write kept rows in the ORIGINAL 13-column data1 layout.

    The Octave oracle consumes this file, so the columns must keep data1's
    schema exactly (countvote.m reads {i,10}=answer, {1,4}/{1,6}=levels).
    """
    smap = split_map()
    allowed = {s for s, r in smap.items() if r in KEEP_ROLES}
    out = Path(args.out)
    n = 0
    with open(args.data_csv, newline="") as fi, \
            open(out, "w", newline="") as fo:
        r = csv.reader(fi)
        w = csv.writer(fo)
        header = next(r)
        w.writerow(header)
        for row in r:
            if row[0] not in allowed:      # held-out source: dropped unread
                continue
            if row[2] != BOOST:
                continue
            w.writerow(row)
            n += 1
    print(f"wrote {n} rows -> {out}")
    return 0


def cmd_emit_x0(args):
    out = Path(args.out)
    rng = np.random.default_rng(args.seed)
    x0 = np.sort(rng.standard_normal(N_SCALE))
    np.savetxt(out, x0, fmt="%.17g")
    print(f"wrote x0 -> {out}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reconstruct")
    r.add_argument("--data-csv", default=str(DATA1))
    r.add_argument("--out-parquet", required=True)
    r.add_argument("--manifest", required=True)
    r.add_argument("--seed", type=int, default=SEED)
    r.add_argument("--oracle-tsv", default=None,
                   help="Octave oracle output TSV to compare and record")
    r.add_argument("--ssim2-parquet", default=None,
                   help="zenmetrics score-pairs ssim2 sidecar (same row "
                        "order as this table) for the Kendall-tau sanity")
    f = sub.add_parser("emit-filtered")
    f.add_argument("--data-csv", default=str(DATA1))
    f.add_argument("--out", required=True)
    x = sub.add_parser("emit-x0")
    x.add_argument("--out", required=True)
    x.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    return {"reconstruct": cmd_reconstruct,
            "emit-filtered": cmd_emit_filtered,
            "emit-x0": cmd_emit_x0}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
