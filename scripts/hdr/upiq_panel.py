#!/usr/bin/env python3
"""UPIQ-HDR panel for a zensim bake: forward the 372-feature UPIQ extraction
through the bake (dial-grid pred-dump path) and correlate vs JOD.

  usage: upiq_panel.py <bake.bin> [--features F.parquet] [--jod J.csv]
                       [--strata] [--compare OTHER.bin] [--boot 10000]
                       [--verify-scipy]

Prints |SROCC| + |PLCC| (JOD is negative-going; sign conventions differ per
metric, so absolute values — matching scripts/hdr/upiq_corr.py's convention
and the registry reference bars: cvvdp 0.758, iwssim-HDR 0.808,
ssim2-HDR 0.704, Profile A 0.694).

--strata additionally reports per-study |SROCC| (the honest read per
bhdr_improvement_split_lineage §8.1 — pooled UPIQ is scale-misalignment-
confounded across narwaria/korshunov). Strata come from the JOD csv pair-id
prefix (`n-…` narwaria, `k-…` korshunov), not hardcoded row ranges.

--compare OTHER.bin scores a second bake on the identical grid and reports
the per-stratum paired bootstrap on Δ|SROCC| (resampling pairs within the
stratum, `--boot` resamples): p = fraction of resamples where OTHER ≥ bake.

--features-b F2 (2026-08-05, campaign Appendix Q §Q.3): a features table for
the --compare bake, so bakes with DIFFERENT feature contracts compare on the
same 380 conditions (e.g. a 944-regime candidate vs the 372 pulinear BHdr).
Both tables must resolve to the same condition set; the paired bootstrap pairs
by condition, which is regime-independent. Feature tables may be the classic
372 `feat_*` parquet (row-aligned to the JOD csv, unchanged behavior) or an
external-reads CSV (`condition_id,dataset,is_hdr,jod,score228,f0..fN` —
reordered to the JOD csv by condition_id, with per-row jod agreement
asserted). With --compare, a POOLED paired bootstrap over all 380 conditions
is reported alongside the per-stratum ones (its index draws come AFTER the
stratum draws, so previously recorded per-stratum outputs reproduce
bit-for-bit).

Stats provenance (migrated 2026-07-31, decision_surface_audit_2026-07-31.md
gap 4): every correlation comes from the canonical Rust `panel` binary via
`scripts/lib/zen_stats` — |SROCC| is the batch `srocc` column (tie-correct
midrank, `.abs()`), |PLCC| is `abs(plcc_raw)` (raw un-rescaled Pearson, the
statistic this panel has always printed — NOT the aggregate panel's
logistic-rescaled PLCC). The whole bootstrap is ONE `panel --batch` process
(`zen_stats.panel_batch_indexed`); the resampling RNG stays HERE
(np.random.default_rng(20260714), unchanged draw order), so previously
recorded numbers reproduce bit-for-bit at printed precision. Migration
verified against the pre-migration scipy script on identical inputs
(shipped BHdr 7d7f2123: default feats 0.7081/0.7173/0.8992; pulinear
0.7536/0.7834/0.9175; 10k-boot p 0.3950/0.0799 — byte-identical stdout).
--verify-scipy additionally cross-checks every printed stat against scipy
to <=1e-9 (the proven equivalence bound) — optional, off by default.
"""
import argparse, os, subprocess, sys, tempfile
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
from scripts.lib import zen_stats  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("bake")
ap.add_argument("--features", default="/mnt/v/output/zensim-multicodec-probe/upiq_features_372.parquet",
                help="REGIME MATTERS (bhdr_improvement §8.13): this default is the v1 PU21 u8-shell extraction. PU-linear bakes (BHdr production path) need upiq_features_372_pulinear.parquet — mis-feeding understates SROCC by ~0.05 and reshuffles strata.")
ap.add_argument("--jod", default="/mnt/v/output/zenmetrics/upiq-pu/upiq_cid_jod.csv")
ap.add_argument("--strata", action="store_true")
ap.add_argument("--compare", default=None, help="second bake for paired per-stratum bootstrap")
ap.add_argument("--features-b", default=None,
                help="features table for the --compare bake (its OWN regime; default: "
                     "--features). Parquet feat_* or external-reads CSV f0..fN.")
ap.add_argument("--boot", type=int, default=10000)
ap.add_argument("--verify-scipy", action="store_true",
                help="cross-check every printed stat against scipy to <=1e-9 (needs scipy)")
a = ap.parse_args()

lines = [l for l in open(a.jod).read().splitlines() if l.strip()]
jod = np.array([float(l.split(",")[1]) for l in lines])
pair_ids = [l.split(",")[0] for l in lines]
n = len(jod)
STUDY = {"n": "narwaria", "k": "korshunov"}
strata = np.array([STUDY[p.split("-")[0]] for p in pair_ids])


def load_features_table(path: str) -> np.ndarray:
    """(n, W) feature matrix row-aligned to the JOD csv order.

    Two accepted shapes (Appendix Q §Q.3):
    - parquet with `feat_*` columns: the classic upiq extraction, stored in JOD
      csv row order (row-count asserted — unchanged original behavior);
    - external-reads CSV (`condition_id,dataset,is_hdr,jod,score228,f0..fN`,
      e.g. hdr-dmean-2026-07-29/upiq_hdr_944.csv): reordered to the JOD csv by
      condition_id, with exact id-set equality and per-row jod agreement
      (<=1e-6) asserted — a hard alignment pin, not a trust-the-order join.
    """
    if path.endswith(".csv"):
        import csv as _csv
        with open(path, newline="") as f:
            rdr = _csv.reader(f)
            header = next(rdr)
            assert header[0] == "condition_id" and header[3] == "jod", header[:5]
            f0 = header.index("f0")
            by_id = {}
            for r in rdr:
                by_id[r[0]] = (float(r[3]), np.array(r[f0:], dtype=np.float64))
        missing = [p for p in pair_ids if p not in by_id]
        extra = [k for k in by_id if k not in set(pair_ids)]
        assert not missing and not extra, (
            f"{path}: condition set mismatch vs {a.jod} "
            f"(missing {len(missing)}, extra {len(extra)})")
        rows = []
        for i, p in enumerate(pair_ids):
            j, feats = by_id[p]
            assert abs(j - jod[i]) <= 1e-6, f"{p}: csv jod {j} != jod csv {jod[i]}"
            rows.append(feats)
        return np.vstack(rows)
    t = pq.read_table(path)
    fcols = sorted((c for c in t.schema.names if c.startswith("feat_")),
                   key=lambda c: int(c.split("_")[1]))
    assert len(fcols) > 0 and t.num_rows == n, \
        f"bad features table: {t.num_rows} rows (want {n}), {len(fcols)} fcols"
    return np.column_stack([np.asarray(t[c].to_pylist(), dtype=np.float64) for c in fcols])


def build_grid(feats: np.ndarray) -> pa.Table:
    """Shape as a dial grid: one 'codec ladder' per row so the pred dump's
    (group-ordinal, local-index) re-keying is trivially invertible by q."""
    data = {
        "ref_basename": pa.array([f"upiq{i}" for i in range(n)]),
        "human_score": pa.array([0.5] * n),
        "image_path": pa.array([f"upiq{i}" for i in range(n)]),
        "image_id": pa.array(["upiq"] * n),
        "codec": pa.array(["upiq"] * n),
        "q": pa.array([float(i) for i in range(n)]),
        "knob_tuple_json": pa.array(["{}"] * n),
    }
    for i in range(feats.shape[1]):
        data[f"f{i}"] = pa.array(feats[:, i])
    return pa.table(data)


grid = build_grid(load_features_table(a.features))
grid_b = None
if a.compare:
    grid_b = grid if a.features_b in (None, a.features) \
        else build_grid(load_features_table(a.features_b))


def score_bake(bake_path: str, grid_parquet: str) -> np.ndarray:
    with tempfile.TemporaryDirectory() as td:
        pred = os.path.join(td, "pred.tsv")
        env = dict(os.environ, ZENSIM_DIAL_GRID=grid_parquet, ZENSIM_DIAL_PRED_OUT=pred)
        subprocess.run(
            [os.path.join(_REPO, "target/release/bake_verdict"),
             "--bake", bake_path, "--corpora", "aic3", "--output", os.devnull],
            env=env, capture_output=True, check=True)
        rows = [l.split("\t") for l in open(pred).read().splitlines()[1:]]
        hdr_cols = open(pred).readline().rstrip("\n").split("\t")
        qi, pi = hdr_cols.index("q"), hdr_cols.index("pred")
        preds = np.full(n, np.nan)
        for r in rows:
            preds[int(float(r[qi]))] = float(r[pi])
    assert not np.isnan(preds).any(), "pred dump incomplete"
    return preds


def _scipy_check(x, y, srocc_abs, plcc_raw_abs=None):
    """Optional --verify-scipy cross-check (<=1e-9, the proven bound)."""
    if not a.verify_scipy:
        return
    from scipy.stats import pearsonr, spearmanr
    s_ref = abs(float(spearmanr(x, y).statistic))
    assert abs(srocc_abs - s_ref) <= 1e-9, (srocc_abs, s_ref)
    if plcc_raw_abs is not None:
        p_ref = abs(float(pearsonr(x, y).statistic))
        assert abs(plcc_raw_abs - p_ref) <= 1e-9, (plcc_raw_abs, p_ref)


def report(name: str, preds: np.ndarray):
    jobs = [("pooled", preds, jod)]
    strata_names = [s for s in ("narwaria", "korshunov") if (a.strata or a.compare)]
    for s in strata_names:
        m = strata == s
        jobs.append((s, preds[m], jod[m]))
    rows = zen_stats.panel_batch(jobs, stats="full")
    sr, pl = rows[0]["srocc"], abs(rows[0]["plcc_raw"])
    _scipy_check(preds, jod, sr, pl)
    print(f"{name}: UPIQ-HDR |SROCC|={sr:.4f} |PLCC|={pl:.4f} (n={n})")
    out = {}
    for s, row in zip(strata_names, rows[1:]):
        m = strata == s
        ss = row["srocc"]
        _scipy_check(preds[m], jod[m], ss)
        out[s] = ss
        print(f"  {s:10s} |SROCC|={ss:.4f} (n={int(m.sum())})")
    return out


with tempfile.TemporaryDirectory() as td:
    gp = os.path.join(td, "upiq_grid.parquet")
    pq.write_table(grid, gp, compression="zstd")
    preds_a = score_bake(a.bake, gp)
    report(os.path.basename(a.bake), preds_a)
    if a.compare:
        gpb = gp
        if grid_b is not grid:
            gpb = os.path.join(td, "upiq_grid_b.parquet")
            pq.write_table(grid_b, gpb, compression="zstd")
        preds_b = score_bake(a.compare, gpb)
        report(os.path.basename(a.compare), preds_b)
        rng = np.random.default_rng(20260714)
        print(f"paired bootstrap Δ|SROCC| (A − B), {a.boot} resamples, within stratum:")
        # One `panel --batch` process for the WHOLE bootstrap: declare each
        # stratum's (A, B, JOD) vectors once, send every resample as an
        # index set (the same idx applies to the A-leg and B-leg jobs —
        # the paired-resample shape). RNG draw order is IDENTICAL to the
        # pre-migration script: per stratum, `--boot` consecutive
        # rng.integers(0, n_s, n_s) draws.
        bases = {}
        jobs = []
        stratum_meta = []
        for s in ("narwaria", "korshunov"):
            m = np.where(strata == s)[0]
            da, db, dj = preds_a[m], preds_b[m], jod[m]
            tag = s[0]
            bases[f"A{tag}"] = da
            bases[f"B{tag}"] = db
            bases[f"J{tag}"] = dj
            jobs.append((f"pt_a_{tag}", f"A{tag}", f"J{tag}", None))
            jobs.append((f"pt_b_{tag}", f"B{tag}", f"J{tag}", None))
            idx_sets = [rng.integers(0, len(m), len(m)) for _ in range(a.boot)]
            for i, idx in enumerate(idx_sets):
                jobs.append((f"a_{tag}_{i}", f"A{tag}", f"J{tag}", idx))
                jobs.append((f"b_{tag}_{i}", f"B{tag}", f"J{tag}", idx))
            stratum_meta.append((s, tag, da, db, dj, idx_sets))
        # POOLED paired bootstrap over all n conditions (Appendix Q §Q.3/Q.4).
        # Draws come AFTER the per-stratum draws so the per-stratum outputs
        # above reproduce previously recorded numbers bit-for-bit.
        bases["Ap"], bases["Bp"], bases["Jp"] = preds_a, preds_b, jod
        jobs.append(("pt_a_p", "Ap", "Jp", None))
        jobs.append(("pt_b_p", "Bp", "Jp", None))
        pooled_idx = [rng.integers(0, n, n) for _ in range(a.boot)]
        for i, idx in enumerate(pooled_idx):
            jobs.append((f"a_p_{i}", "Ap", "Jp", idx))
            jobs.append((f"b_p_{i}", "Bp", "Jp", idx))
        stratum_meta.append(("pooled", "p", preds_a, preds_b, jod, pooled_idx))
        rows = {r["label"]: r["srocc"] for r in
                zen_stats.panel_batch_indexed(bases, jobs, stats="srocc")}
        for s, tag, da, db, dj, idx_sets in stratum_meta:
            point = rows[f"pt_a_{tag}"] - rows[f"pt_b_{tag}"]
            if a.verify_scipy:
                _scipy_check(da, dj, rows[f"pt_a_{tag}"])
                _scipy_check(db, dj, rows[f"pt_b_{tag}"])
            wins = 0
            for i in range(len(idx_sets)):
                d = rows[f"a_{tag}_{i}"] - rows[f"b_{tag}_{i}"]
                if d <= 0:
                    wins += 1
            # one-sided p for "A > B": fraction of resamples where A does NOT beat B
            print(f"  {s:10s} Δ={point:+.4f}  p(A≤B)={wins / a.boot:.4f}")
