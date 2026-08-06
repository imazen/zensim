#!/usr/bin/env python3
"""Band reliability + banding-scheme instrument (campaign appendix V).

Answers "how wide must a band be to be statistically usable?" from the
board's STORED per-pair predictions — no verdict re-runs, no re-scoring.

Every statistic comes from `zenstats` through the canonical `panel --batch`
owner (via `scripts/lib/zen_stats.py`). There is no Spearman in this file;
per the no-duplication rule, the owner is extended rather than mirrored.

Subcommands
-----------
gv1        Appendix V gate G-V1: re-verify that the published band `srocc`
           is an ABSOLUTE value and census the CID22 B9 sign across every
           board cell that still carries per-pair.
curves     Instruments A/B/C: the pure-n curve (subsample within a band, so
           span is held fixed), the pure-span curve (centred sub-slices at a
           common n), and the joint (n, span) grid. Theory columns
           (Bonett-Wright Fisher-z SE, Thorndike case-II attenuation) are
           emitted alongside as PREDICTIONS, never as substitutes.
discrim    Instrument E: per-band split-half model-ranking reliability
           (Spearman-Brown), paired LSD, and the model spread, over the
           board population.
schemes    Instrument for V.4: occupancy + span of every candidate scheme
           (fixed10 / quantile10 / merge / fixed5) on every banded corpus.

Usage
-----
    ZEN_PANEL_BIN=<path/to/panel> python3 scripts/band_reliability.py <cmd> [opts]
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from scripts.lib import zen_stats  # noqa: E402

# The five corpora with `enable_per_band: true` in bake_verdict, and the
# per-pair key holding their target column.
BANDED_CORPORA = {
    "cid22": "mos",
    "csiq": "mos",
    "kadid": "mos",
    "live": "mos",
    "tid": "mos",
}

DEFAULT_BOARD = "/mnt/v/output/zensim/reports/fulleval"


# ----------------------------------------------------------------------
# Board loading
# ----------------------------------------------------------------------


def load_board(board_dir: str, corpus: str):
    """[(name, target[], pred[], stored_bands)] for cells carrying per-pair.

    Cross-cell band work requires every cell to carry the SAME target
    vector — bands are cut on it, so cells cut on different targets are not
    comparable. Cells are therefore grouped by the exact bytes of their
    target column and only the largest group is returned; the split is
    reported on stderr.

    This is not hypothetical: KADID's stored per-pair is a 5,000-row
    subsample of its 10,031 banded rows and the subsample differs per cell,
    so KADID collapses to groups of one and cannot be recomputed this way
    (appendix V confound 4).
    """
    tkey = BANDED_CORPORA[corpus]
    groups: dict[bytes, list] = {}
    for path in sorted(glob.glob(os.path.join(board_dir, "*.fulleval.json"))):
        with open(path) as fh:
            d = json.load(fh)
        pp = (d.get("per_pair") or {}).get(corpus)
        if not pp or not pp.get(tkey) or not pp.get("pred"):
            continue
        t = np.asarray(pp[tkey], dtype=float)
        p = np.asarray(pp["pred"], dtype=float)
        if t.shape != p.shape:
            continue
        stored = ((d.get("rank") or {}).get(corpus) or {}).get("bands")
        groups.setdefault(t.tobytes(), []).append(
            (d.get("name") or os.path.basename(path), t, p, stored)
        )
    if not groups:
        raise SystemExit(f"no board cell carries per_pair.{corpus} under {board_dir}")
    best = max(groups.values(), key=len)
    total = sum(len(g) for g in groups.values())
    if len(groups) > 1:
        sys.stderr.write(
            f"warn: {corpus} per-pair targets fall into {len(groups)} distinct "
            f"vectors across {total} cells; using the largest group "
            f"({len(best)} cells, n={best[0][1].size})\n"
        )
    return best


# ----------------------------------------------------------------------
# Banding schemes.  Edges are a function of the TARGET column only, so
# every model on a corpus gets identical bands (a hard requirement -- the
# cross-bake band table is meaningless otherwise).
# ----------------------------------------------------------------------


def scheme_fixed(target: np.ndarray, k: int):
    """Fixed [0,1] edges, k equal-width bands, open top (the status quo at k=10)."""
    edges = [i / k for i in range(k + 1)]
    edges[-1] = math.inf
    return [(f"B{i}", edges[i], edges[i + 1]) for i in range(k)]


def scheme_quantile(target: np.ndarray, k: int):
    """Equal-population bands: edges are the corpus's own target quantiles."""
    qs = np.quantile(target, [i / k for i in range(1, k)])
    edges = [-math.inf] + list(qs) + [math.inf]
    return [(f"Q{i}", edges[i], edges[i + 1]) for i in range(k)]


def scheme_merge(target: np.ndarray, n_min: int, span_min: float, k: int = 10):
    """Fixed deciles accumulated into the finest partition whose every band
    satisfies n >= n_min AND span >= span_min.

    MUST stay in lockstep with `zensim_validate::bands::merged_bands` (the
    owner); `tests/band_scheme_parity.rs` gates the two against each other on
    every banded corpus. See that module for why a pairwise "merge the worst
    into its smaller neighbour" greedy was rejected.
    """
    def occ(lo, hi):
        sel = target[(target >= lo) & (target < hi)]
        return (0, 0.0) if sel.size == 0 else (sel.size, float(sel.max() - sel.min()))

    def edge(i):
        return i / k

    def top(j):
        return math.inf if j == k - 1 else (j + 1) / k

    groups, start = [], 0
    for i in range(k):
        n, span = occ(edge(start), top(i))
        if n >= n_min and span >= span_min:
            groups.append([start, i])
            start = i + 1
    if start <= k - 1:
        if groups:
            groups[-1][1] = k - 1
        else:
            groups = [[0, k - 1]]

    return [
        (f"B{a}" if a == b else f"B{a}-B{b}", edge(a), top(b)) for a, b in groups
    ]


def band_members(target: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if math.isinf(hi):
        return np.nonzero(target >= lo)[0]
    return np.nonzero((target >= lo) & (target < hi))[0]


def schemes_for(target: np.ndarray, n_min: int, span_min: float):
    return {
        "fixed10": scheme_fixed(target, 10),
        "quantile10": scheme_quantile(target, 10),
        "merge": scheme_merge(target, n_min, span_min),
        "fixed5": scheme_fixed(target, 5),
    }


# ----------------------------------------------------------------------
# Theory predictions (reported ALONGSIDE the empirical curves, never as a
# substitute -- appendix V instrument D).
# ----------------------------------------------------------------------


def fisher_z_halfwidth(r: float, n: int) -> float:
    """Bonett-Wright 95% CI half-width for Spearman, back-transformed.

    SE_z = sqrt((1 + r^2/2) / (n - 3)); the interval is symmetric in z and
    asymmetric in r, so the half-width reported is (hi - lo)/2 in r-space.
    """
    if n <= 4:
        return float("nan")
    r = max(min(r, 0.999999), -0.999999)
    se = math.sqrt((1.0 + r * r / 2.0) / (n - 3))
    z = math.atanh(r)
    lo, hi = math.tanh(z - 1.959964 * se), math.tanh(z + 1.959964 * se)
    return (hi - lo) / 2.0


def thorndike(r_full: float, u: float) -> float:
    """Case-II range restriction: the correlation a band of sd-ratio u retains."""
    if u <= 0:
        return 0.0
    d = 1.0 + r_full * r_full * (u * u - 1.0)
    if d <= 0:
        return float("nan")
    return r_full * u / math.sqrt(d)


# ----------------------------------------------------------------------
# Bootstrap helpers.  The RNG lives here (the `panel` binary is RNG-free
# by contract); `panel --batch` indexed jobs carry the resamples.
# ----------------------------------------------------------------------


def marginal_ci(pred, target, idxs, B: int, seed: int, cluster=None):
    """Marginal 95% CI half-width of a band's SIGNED Spearman.

    `cluster` (optional) = per-row group id; when given, the bootstrap
    resamples CLUSTERS (references) rather than rows, which is the
    conservative form for reference-clustered corpora (appendix V confound 2).
    """
    rng = np.random.default_rng(seed)
    idxs = np.asarray(idxs)
    bases = {"P": np.asarray(pred)[idxs], "T": np.asarray(target)[idxs]}
    jobs = []
    if cluster is None:
        n = idxs.size
        for b in range(B):
            jobs.append((f"b{b}", "P", "T", rng.integers(0, n, n)))
    else:
        cl = np.asarray(cluster)[idxs]
        uniq = np.unique(cl)
        buckets = [np.nonzero(cl == u)[0] for u in uniq]
        for b in range(B):
            pick = rng.integers(0, len(buckets), len(buckets))
            jobs.append((f"b{b}", "P", "T", np.concatenate([buckets[i] for i in pick])))
    rows = zen_stats.panel_batch_indexed(bases, jobs, stats="srocc")
    vals = np.array([r["srocc_signed"] for r in rows], dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (hi - lo) / 2.0, float(lo), float(hi)


# ----------------------------------------------------------------------
# Subcommands
# ----------------------------------------------------------------------


def cmd_gv1(a):
    cells = load_board(a.board, "cid22")
    target = cells[0][1]
    idxs = band_members(target, 0.90, math.inf)
    jobs = [(f"m{i}", p[idxs], target[idxs]) for i, (_, _, p, _) in enumerate(cells)]
    rows = zen_stats.panel_batch(jobs, stats="srocc")

    out = ["name\tn\tstored_srocc\trecomputed_abs\trecomputed_signed\tidentity_ok\t"
           "pass_abs_0.15\tpass_signed_0.15"]
    n_ident = n_neg = n_pa = n_ps = n_stored = 0
    mism = []
    for (name, _, _, stored), r in zip(cells, rows):
        sv = None
        if stored:
            b9 = next((b for b in stored if b.get("band") == "B9"), None)
            if b9 is not None:
                sv = b9.get("srocc")
        ok = ""
        if sv is not None:
            n_stored += 1
            ok = abs(sv - abs(r["srocc_signed"])) <= 1e-6
            n_ident += bool(ok)
            if not ok:
                mism.append((name, sv, abs(r["srocc_signed"])))
        n_neg += r["srocc_signed"] < 0
        pa = abs(r["srocc_signed"]) >= 0.15
        ps = r["srocc_signed"] >= 0.15
        n_pa += pa
        n_ps += ps
        out.append(
            f"{name}\t{r['n']}\t{'' if sv is None else f'{sv:.6f}'}\t"
            f"{abs(r['srocc_signed']):.6f}\t{r['srocc_signed']:.6f}\t{ok}\t{pa}\t{ps}"
        )
    N = len(cells)
    summary = [
        f"# G-V1 CID22 B9 sign re-verification -- {N} board cells carrying per-pair",
        f"# band = [0.90, inf) under fixed10; n = {idxs.size} pairs",
        f"# G-V1a identity (stored srocc == |recomputed signed| within 1e-6): "
        f"{n_ident}/{n_stored} = {100.0 * n_ident / max(n_stored, 1):.1f}%",
        f"# G-V1b sign census: {n_neg}/{N} = {100.0 * n_neg / N:.1f}% NEGATIVE",
        f"# G-V1c gate consequence: pass |B9|>=0.15 -> {n_pa}/{N}; "
        f"pass signed>=0.15 -> {n_ps}/{N}; delta = {n_pa - n_ps}",
    ]
    for m in mism[:20]:
        summary.append(f"# MISMATCH {m[0]} stored={m[1]:.6f} recomputed_abs={m[2]:.6f}")
    text = "\n".join(summary) + "\n" + "\n".join(out) + "\n"
    _emit(a.out, text)


def cmd_curves(a):
    cells = load_board(a.board, a.corpus)
    target = cells[0][1]
    sd_full = float(np.std(target, ddof=1))
    # Representative models: evenly spaced through the population by
    # aggregate |SROCC| so the curve is not a property of one bake.
    agg = zen_stats.panel_batch(
        [(f"m{i}", p, target) for i, (_, _, p, _) in enumerate(cells)], stats="srocc"
    )
    order = np.argsort([abs(r["srocc_signed"]) for r in agg])
    pick = [int(order[int(round(q * (len(order) - 1)))]) for q in a.model_quantiles]

    rows = ["kind\tmodel\tband\tn\treps\tspan\tsd\tsd_ratio\tsrocc_signed\t"
            "ci_halfwidth\tci_hw_p25\tci_hw_p75\ttheory_fisher_hw\ttheory_thorndike_r"]
    rng = np.random.default_rng(a.seed)

    def cell(kind, name, label, pool, n, r_full):
        """Median over `a.reps` independent subsamples, so a point on the
        curve is a property of (n, span) rather than of one lucky draw."""
        if n > pool.size:
            return None
        hws, rs, spans, sds = [], [], [], []
        for rep in range(a.reps):
            sub = rng.choice(pool, size=n, replace=False)
            hw, _, _ = marginal_ci(pred_of[name], target, sub, a.boot,
                                   int(a.seed + 1009 * n + 17 * rep))
            r = zen_stats.panel_batch(
                [("x", pred_of[name][sub], target[sub])], stats="srocc"
            )[0]["srocc_signed"]
            hws.append(hw)
            rs.append(r)
            spans.append(float(target[sub].max() - target[sub].min()))
            sds.append(float(np.std(target[sub], ddof=1)))
        hw_p25, hw_med, hw_p75 = np.percentile(hws, [25, 50, 75])
        r_med = float(np.median(rs))
        sd_med = float(np.median(sds))
        return (
            f"{kind}\t{name}\t{label}\t{n}\t{a.reps}\t{float(np.median(spans)):.4f}\t"
            f"{sd_med:.4f}\t{sd_med / sd_full:.4f}\t{r_med:.4f}\t{hw_med:.4f}\t"
            f"{hw_p25:.4f}\t{hw_p75:.4f}\t{fisher_z_halfwidth(r_med, n):.4f}\t"
            f"{thorndike(r_full, sd_med / sd_full):.4f}"
        )

    pred_of = {cells[i][0]: cells[i][2] for i in pick}
    rfull_of = {cells[i][0]: agg[i]["srocc_signed"] for i in pick}

    # ---- Instrument A: pure-n (span fixed; subsample WITHIN a band) ----
    for donor_lo, donor_hi, dname in a.donors:
        pool = band_members(target, donor_lo, donor_hi)
        for mi in pick:
            name = cells[mi][0]
            for n in a.n_grid:
                row = cell("pure_n", name, dname, pool, n, rfull_of[name])
                if row:
                    rows.append(row)

    # ---- Instrument B: pure-span (n fixed; centred sub-slices) ----
    centre = float(np.median(target))
    for mi in pick:
        name = cells[mi][0]
        for span in a.span_grid:
            sel = np.nonzero(
                (target >= centre - span / 2) & (target <= centre + span / 2)
            )[0]
            row = cell("pure_span", name, f"centre{span:.2f}", sel, a.span_n,
                       rfull_of[name])
            if row:
                rows.append(row)

    # ---- Instrument C: joint (n, span) grid ----
    for mi in pick:
        name = cells[mi][0]
        for span in a.span_grid:
            sel = np.nonzero(
                (target >= centre - span / 2) & (target <= centre + span / 2)
            )[0]
            for n in a.n_grid:
                row = cell("joint", name, f"span{span:.2f}", sel, n, rfull_of[name])
                if row:
                    rows.append(row)

    _emit(a.out, "\n".join(rows) + "\n")


def _split_half_rsb(preds, target, idxs, shuffles: int, seed: int):
    """Split-half model-ranking SROCC + Spearman-Brown, splitting PAIRS."""
    rng = np.random.default_rng(seed)
    idxs = np.asarray(idxs)
    vals = []
    for _ in range(shuffles):
        perm = rng.permutation(idxs)
        h1, h2 = perm[: perm.size // 2], perm[perm.size // 2 :]
        jobs = []
        for i, p in enumerate(preds):
            jobs.append((f"a{i}", p[h1], target[h1]))
            jobs.append((f"b{i}", p[h2], target[h2]))
        rows = zen_stats.panel_batch(jobs, stats="srocc")
        s1 = np.array([rows[2 * i]["srocc_signed"] for i in range(len(preds))])
        s2 = np.array([rows[2 * i + 1]["srocc_signed"] for i in range(len(preds))])
        ok = np.isfinite(s1) & np.isfinite(s2)
        if ok.sum() < 4:
            continue
        vals.append(zen_stats.panel_batch([("x", s1[ok], s2[ok])], stats="srocc")[0][
            "srocc_signed"
        ])
    if not vals:
        return float("nan"), float("nan"), float("nan")
    r = float(np.mean(vals))
    sb = 2 * r / (1 + r) if r > -1 else float("nan")
    return r, float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0, sb


def cmd_discrim(a):
    cells = load_board(a.board, a.corpus)
    target = cells[0][1]
    preds = [c[2] for c in cells]
    bands = schemes_for(target, a.n_min, a.span_min)[a.scheme]
    rows = ["scheme\tband\tlo\thi\tn\tspan\tsd\tr_halves\tr_sd\tr_sb\t"
            "spread_p10\tspread_p50\tspread_p90\tlsd\tdr\tn_models"]
    rng = np.random.default_rng(a.seed)
    for label, lo, hi in bands:
        idxs = band_members(target, lo, hi)
        if idxs.size < 4:
            rows.append(
                f"{a.scheme}\t{label}\t{lo:.4f}\t{hi}\t{idxs.size}\t\t\t"
                "NOT-MEASURED\t\t\t\t\t\t\t\t" + str(len(preds))
            )
            continue
        r, rsd, sb = _split_half_rsb(preds, target, idxs, a.shuffles, a.seed)
        per = zen_stats.panel_batch(
            [(f"m{i}", p[idxs], target[idxs]) for i, p in enumerate(preds)],
            stats="srocc",
        )
        v = np.array([x["srocc_signed"] for x in per], dtype=float)
        v = v[np.isfinite(v)]
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        # Paired LSD: the 95% half-width of the paired delta between two
        # models on this band, over a sample of model pairs.
        bases = {"T": target[idxs]}
        for i, p in enumerate(preds):
            bases[f"P{i}"] = p[idxs]
        pairs = [
            (int(i), int(j))
            for i, j in rng.integers(0, len(preds), (a.lsd_pairs, 2))
            if i != j
        ]
        halfw = []
        n = idxs.size
        for (i, j) in pairs:
            jobs = []
            for b in range(a.lsd_boot):
                sel = rng.integers(0, n, n)
                jobs.append((f"a{b}", f"P{i}", "T", sel))
                jobs.append((f"b{b}", f"P{j}", "T", sel))
            rr = zen_stats.panel_batch_indexed(bases, jobs, stats="srocc")
            d = np.array(
                [
                    rr[2 * b]["srocc_signed"] - rr[2 * b + 1]["srocc_signed"]
                    for b in range(a.lsd_boot)
                ]
            )
            d = d[np.isfinite(d)]
            if d.size > 2:
                q = np.percentile(d, [2.5, 97.5])
                halfw.append((q[1] - q[0]) / 2.0)
        lsd = float(np.median(halfw)) if halfw else float("nan")
        dr = (p90 - p10) / lsd if lsd and math.isfinite(lsd) and lsd > 0 else float("nan")
        sel = target[idxs]
        rows.append(
            f"{a.scheme}\t{label}\t{lo:.4f}\t{hi}\t{idxs.size}\t"
            f"{float(sel.max() - sel.min()):.4f}\t{float(np.std(sel, ddof=1)):.4f}\t"
            f"{r:.4f}\t{rsd:.4f}\t{sb:.4f}\t{p10:.4f}\t{p50:.4f}\t{p90:.4f}\t"
            f"{lsd:.4f}\t{dr:.3f}\t{len(v)}"
        )
    _emit(a.out, "\n".join(rows) + "\n")


def cmd_discrim2d(a):
    """Split-half model-ranking reliability over the (n, span) grid.

    Discrimination is the BINDING usability condition -- a band can be
    estimable (narrow CI) and still rank models inconsistently -- so both
    registered floors are read off this surface rather than off the CI curve.
    """
    cells = load_board(a.board, a.corpus)
    target = cells[0][1]
    preds = [c[2] for c in cells]
    centre = float(np.median(target))
    rng = np.random.default_rng(a.seed)
    rows = ["span\tn\treps\tspan_actual\tr_halves\tr_sb\tr_sb_p25\tr_sb_p75\tn_models"]
    for span in a.span_grid:
        sel = np.nonzero((target >= centre - span / 2) & (target <= centre + span / 2))[0]
        for n in a.n_grid:
            if n > sel.size:
                continue
            sbs, rs, sa = [], [], []
            for rep in range(a.reps):
                sub = rng.choice(sel, size=n, replace=False)
                r, _, sb = _split_half_rsb(
                    preds, target, sub, a.shuffles, int(a.seed + 1009 * n + 17 * rep)
                )
                if math.isfinite(sb):
                    sbs.append(sb)
                    rs.append(r)
                    sa.append(float(target[sub].max() - target[sub].min()))
            if not sbs:
                continue
            p25, p50, p75 = np.percentile(sbs, [25, 50, 75])
            rows.append(
                f"{span:.3f}\t{n}\t{len(sbs)}\t{float(np.median(sa)):.4f}\t"
                f"{float(np.median(rs)):.4f}\t{p50:.4f}\t{p25:.4f}\t{p75:.4f}\t{len(preds)}"
            )
    _emit(a.out, "\n".join(rows) + "\n")


PARITY_FIXTURE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "benchmarks", "appendixV", "band_scheme_parity.tsv",
)


def _parity_corpus(counts, top_span):
    v = []
    for d, n in enumerate(counts):
        lo = d / 10.0
        width = top_span if (d == len(counts) - 1 and top_span is not None) else 0.0999
        for i in range(n):
            f = 0.0 if n == 1 else i / (n - 1)
            v.append(lo + width * f)
    return np.asarray(v, dtype=float)


def cmd_selfcheck(a):
    """Assert this file's `scheme_merge` matches the OWNER's committed contract.

    The owner is `zensim_validate::bands::merged_bands`; this function is the
    mirror's half of the gate (`zensim-validate/tests/band_scheme_parity.rs` is
    the owner's half). Both read the same fixture, so they cannot drift.
    """
    n_min = span_min = None
    checked = failed = 0
    with open(PARITY_FIXTURE) as fh:
        for line in fh:
            if line.startswith("# floors:"):
                for tok in line.split():
                    if tok.startswith("n_min="):
                        n_min = int(tok.split("=")[1])
                    elif tok.startswith("span_min="):
                        span_min = float(tok.split("=")[1])
            if line.startswith("#") or not line.strip():
                continue
            name, counts, top, expect = line.rstrip("\n").split("\t")
            counts = [int(x) for x in counts.split(",")]
            t = _parity_corpus(counts, None if top == "-" else float(top))
            got = ",".join(b[0] for b in scheme_merge(t, n_min, span_min))
            checked += 1
            if got != expect:
                failed += 1
                print(f"MISMATCH {name}: mirror={got} owner={expect}")
    if n_min is None or span_min is None:
        raise SystemExit("fixture must record `# floors: n_min=.. span_min=..`")
    print(
        f"band-scheme parity: {checked - failed}/{checked} rows match the owner "
        f"(n_min={n_min}, span_min={span_min})"
    )
    if failed:
        raise SystemExit(f"{failed} row(s) diverge from the owner")


def cmd_floor(a):
    """Derive F8's floor on the chosen band, and price it against the board.

    Registered rule (appendix V.5): F8's job is NON-COLLAPSE, so the floor is
    the smallest value at which a model's band ordering is significantly
    positive -- the band's own marginal 95 % CI half-width. Per confound 2 the
    CONSERVATIVE (reference-clustered) bootstrap governs, because CID22's pairs
    are clustered by reference and a pair-level resample understates the
    uncertainty.
    """
    cells = load_board(a.board, a.corpus)
    target = cells[0][1]
    refs = None
    if a.refs:
        refs = np.loadtxt(a.refs, delimiter=",", dtype=int)
        if refs.size != target.size:
            raise SystemExit(f"--refs has {refs.size} ids, corpus has {target.size} rows")
    bands = schemes_for(target, a.n_min, a.span_min)["merge"]
    label, lo, hi = bands[-1] if a.band is None else next(
        b for b in bands if b[0] == a.band
    )
    idxs = band_members(target, lo, hi)

    per = zen_stats.panel_batch(
        [(f"m{i}", c[2][idxs], target[idxs]) for i, c in enumerate(cells)], stats="srocc"
    )
    vals = {c[0]: r["srocc_signed"] for c, r in zip(cells, per)}

    rows = [f"# F8 floor derivation on {a.corpus} band {label} = [{lo:.2f}, {hi})",
            f"# n={idxs.size} pairs, span={float(target[idxs].max()-target[idxs].min()):.4f}, "
            f"refs={len(set(refs[idxs])) if refs is not None else '?'}, "
            f"models={len(cells)}, B={a.boot}, seed={a.seed}",
            "name\tsrocc_signed\tci_hw_pair\tci_hw_refclust"]
    hw_pair, hw_ref = [], []
    # The half-width is a property of the band, not of one model; a stratified
    # subset keeps this affordable while spanning the population's range.
    order = sorted(range(len(cells)), key=lambda i: vals[cells[i][0]])
    probe = sorted({order[int(round(q * (len(order) - 1) / (a.probe - 1)))]
                    for q in range(a.probe)})
    for i in probe:
        name, _, pred, _ = cells[i]
        hp, _, _ = marginal_ci(pred, target, idxs, a.boot, a.seed)
        hr = float("nan")
        if refs is not None:
            hr, _, _ = marginal_ci(pred, target, idxs, a.boot, a.seed, cluster=refs)
        hw_pair.append(hp)
        if math.isfinite(hr):
            hw_ref.append(hr)
        rows.append(f"{name}\t{vals[name]:.4f}\t{hp:.4f}\t{hr:.4f}")

    med_pair = float(np.median(hw_pair))
    med_ref = float(np.median(hw_ref)) if hw_ref else float("nan")
    governing = med_ref if math.isfinite(med_ref) else med_pair
    floor = math.ceil(governing * 100.0) / 100.0
    v = np.array(list(vals.values()))
    rows += [
        "",
        f"# marginal 95% CI half-width  pair-bootstrap median  = {med_pair:.4f}",
        f"# marginal 95% CI half-width  ref-clustered   median = {med_ref:.4f}"
        "   <- GOVERNING (confound 2)",
        f"# DERIVED FLOOR = ceil_2dp(governing) = {floor:.2f}",
        "",
        "# board population on this band:",
        f"#   min {v.min():+.4f}  p10 {np.percentile(v,10):+.4f}  median "
        f"{np.median(v):+.4f}  p90 {np.percentile(v,90):+.4f}  max {v.max():+.4f}",
        f"#   negative: {(v < 0).sum()}/{v.size}",
        "",
        "# pass-count at candidate floors (signed):",
    ]
    for f in sorted({round(floor - 0.05, 2), round(floor - 0.02, 2), floor,
                     round(floor + 0.05, 2), 0.15}):
        if f < 0:
            continue
        rows.append(f"#   floor {f:.2f} -> {(v >= f).sum():3d}/{v.size} pass")
    _emit(a.out, "\n".join(rows) + "\n")


def cmd_schemes(a):
    rows = ["corpus\tscheme\tband\tlo\thi\tn\tspan\tsd\tsd_ratio\tusable_n\tusable_span"]
    for corpus in a.corpora:
        cells = load_board(a.board, corpus)
        target = cells[0][1]
        sd_full = float(np.std(target, ddof=1))
        for sname, bands in schemes_for(target, a.n_min, a.span_min).items():
            for label, lo, hi in bands:
                idxs = band_members(target, lo, hi)
                if idxs.size == 0:
                    rows.append(
                        f"{corpus}\t{sname}\t{label}\t{lo:.4f}\t{hi}\t0\t\t\t\tFalse\tFalse"
                    )
                    continue
                sel = target[idxs]
                span = float(sel.max() - sel.min())
                sd = float(np.std(sel, ddof=1)) if idxs.size > 1 else 0.0
                rows.append(
                    f"{corpus}\t{sname}\t{label}\t{lo:.4f}\t{hi}\t{idxs.size}\t"
                    f"{span:.4f}\t{sd:.4f}\t{sd / sd_full:.4f}\t"
                    f"{idxs.size >= a.n_min}\t{span >= a.span_min}"
                )
    _emit(a.out, "\n".join(rows) + "\n")


def _emit(path, text):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(text)
        sys.stderr.write(f"wrote {path}\n")
    else:
        sys.stdout.write(text)


def _floats(s):
    return [float(x) for x in s.split(",")]


def _ints(s):
    return [int(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--board", default=DEFAULT_BOARD)
    ap.add_argument("--out")
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--n-min", type=int, default=200,
                    help="provisional n floor for the merge scheme / usability columns")
    ap.add_argument("--span-min", type=float, default=0.08,
                    help="provisional span floor for the merge scheme")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gv1")
    g.set_defaults(fn=cmd_gv1)

    c = sub.add_parser("curves")
    c.add_argument("--corpus", default="cid22")
    c.add_argument("--boot", type=int, default=2000)
    c.add_argument("--n-grid", type=_ints,
                   default=[8, 12, 16, 24, 32, 43, 64, 96, 128, 192, 256, 384, 512, 768, 1024])
    c.add_argument("--span-grid", type=_floats,
                   default=[0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20])
    c.add_argument("--span-n", type=int, default=200)
    c.add_argument("--reps", type=int, default=8,
                   help="independent subsample draws per (n, span) cell; the "
                        "reported half-width is their median")
    c.add_argument("--model-quantiles", type=_floats, default=[0.1, 0.5, 0.9])
    c.add_argument("--donors", default="0.70:0.80:B7,0.80:0.90:B8")
    c.set_defaults(fn=cmd_curves)

    d = sub.add_parser("discrim")
    d.add_argument("--corpus", default="cid22")
    d.add_argument("--scheme", default="fixed10")
    d.add_argument("--shuffles", type=int, default=20)
    d.add_argument("--lsd-pairs", type=int, default=30)
    d.add_argument("--lsd-boot", type=int, default=400)
    d.set_defaults(fn=cmd_discrim)

    d2 = sub.add_parser("discrim2d")
    d2.add_argument("--corpus", default="cid22")
    d2.add_argument("--shuffles", type=int, default=12)
    d2.add_argument("--reps", type=int, default=4)
    d2.add_argument("--n-grid", type=_ints,
                    default=[43, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1400])
    d2.add_argument("--span-grid", type=_floats,
                    default=[0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 0.20])
    d2.set_defaults(fn=cmd_discrim2d)

    sc = sub.add_parser("selfcheck")
    sc.set_defaults(fn=cmd_selfcheck)

    fl = sub.add_parser("floor")
    fl.add_argument("--corpus", default="cid22")
    fl.add_argument("--band", default=None, help="default = the top merged band")
    fl.add_argument("--refs", default="benchmarks/appendixV/cid22_ref_ids.csv",
                    help="per-pair reference ids for the conservative "
                         "reference-clustered bootstrap")
    fl.add_argument("--boot", type=int, default=10000)
    fl.add_argument("--probe", type=int, default=25,
                    help="stratified models to bootstrap the half-width on")
    fl.set_defaults(fn=cmd_floor)

    s = sub.add_parser("schemes")
    s.add_argument("--corpora", type=lambda x: x.split(","),
                   default=list(BANDED_CORPORA))
    s.set_defaults(fn=cmd_schemes)

    a = ap.parse_args()
    if a.cmd == "curves":
        a.donors = [
            (float(p[0]), (math.inf if p[1] == "inf" else float(p[1])), p[2])
            for p in (x.split(":") for x in a.donors.split(","))
        ]
    a.fn(a)


if __name__ == "__main__":
    main()
