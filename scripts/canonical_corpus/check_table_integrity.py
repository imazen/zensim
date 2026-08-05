#!/usr/bin/env python3
"""Feature-table INTEGRITY gate — the structural checks a canonical table must pass
before it is used to train or eval.

Sibling of `check_target_orientation.py` (which owns the ONE question "does this
table's target point the right way?"). This module owns everything else that can be
established from the table's own bytes:

    A1  target finiteness            no NaN / +-inf in `human_score`
    A2  target range                 within the mix's documented unit convention
    A5  target degeneracy            exact-tie rate (rank-mode groups DROP tied pairs)
    B1  feature finiteness           no NaN / +-inf in any `f<N>` column
    B2  constant columns             min == max, classified against the structural
                                     -zero block (f156..f371 are zero BY DESIGN)
    B4  unguarded heavy tails        max/p99 ratio vs the recipe's winsor_p99 guards
    C1  duplicate feature rows       exact byte-identical 944-vectors within a table
    C4  eval leakage                 reference identity vs the eval corpora

**Why this exists.** On 2026-08-04 an audit of the SOTA-944 training mix (campaign
`benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX G) was the first time
any of these questions had been asked of the tables the project trains on. The
orientation gate had just found a six-week-old inverted KADID target on its first run;
this gate exists so the *other* defect classes cannot sit undetected for six weeks
either. A finding that can recur silently is not closed by documenting it.

Verdicts are three-way, never two-way:
    PASS           the check ran and the table satisfies it
    FINDING        the check ran and the table violates it
    NOT-CHECKABLE  the check could not run (missing key, no ground truth, ...)
A NOT-CHECKABLE is a reported gap, NEVER a pass — same convention as the orientation
gate's SKIPPED.

Usage:
    check_table_integrity.py <parquet> [--name kadid] [--json]
    check_table_integrity.py --mix <recipe.json> [--json] [--tsv-dir DIR]
    check_table_integrity.py --mix-from-spec <bake.bin.spec.json> [--tsv-dir DIR]

Exit 0 = every checked table passed; 1 = at least one FINDING; 2 = usage/IO error.

Statistics come from `zenstats` via `scripts/lib/zen_stats` — no stat math is
implemented here, per the no-duplication rule (zensim/CLAUDE.md).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Regime constants. These describe the 944 folded+append+append2 STREAMING
# layout (`ext944-canonical-2026-08-01/_MANIFEST.json`). f156..f371 are
# STRUCTURAL ZEROS: fold slots that the folded regime deliberately leaves
# empty. A constant column inside that range is BY DESIGN; one outside it is a
# signal gap.
# ---------------------------------------------------------------------------
STRUCTURAL_ZERO_LO = 156
STRUCTURAL_ZERO_HI = 371  # inclusive

# A2: the mix's target-unit convention. The trainer multiplies `human_score` by
# `--target-scale 100`, so canonical tables carry [0,1]. The negative dial tail is
# a deliberate design feature (inputs worse than the worst codec output score below
# zero — zensim/CLAUDE.md "NEGATIVE zensim values MUST work"), so the lower bound is
# permissive; the UPPER bound is not, because nothing legitimately exceeds identity.
TARGET_MIN_HARD = -10.0
TARGET_MAX_HARD = 1.0 + 1e-6
# Advisory band: outside this, the table's negative tail is deep enough that it
# dominates a squared-error term and should be a conscious choice, not a surprise.
TARGET_MIN_ADVISORY = -1.0

# B4: a column whose max is this many times its p99 has a tail that a single row can
# dominate. Reported only when the recipe declares NO winsor_p99 guard for it.
TAIL_RATIO_THRESHOLD = 100.0

# A5: rank-mode groups DROP exactly-tied pairs (mlp_train/mod.rs: `rank_tied` ->
# `continue`), so a high tie rate silently cuts the group's effective sampling mass.
TIE_RATE_THRESHOLD = 0.20

# C1: duplicate feature rows carry no gradient information beyond the first copy.
DUP_MASS_THRESHOLD = 0.05

FEAT_RE = re.compile(r"^f(\d+)$")
COLUMN_BLOCK = 64  # columns read per pass; bounds peak memory


def _feat_cols(schema) -> list[tuple[int, str]]:
    out = []
    for nm in schema.names:
        m = FEAT_RE.match(nm)
        if m:
            out.append((int(m.group(1)), nm))
    out.sort()
    return out


def _is_structural_zero(idx: int) -> bool:
    return STRUCTURAL_ZERO_LO <= idx <= STRUCTURAL_ZERO_HI


# ---------------------------------------------------------------------------
# Per-column statistics. Read in blocks so peak memory is
# COLUMN_BLOCK * n_rows * 8 bytes, never the whole table.
# ---------------------------------------------------------------------------
def column_stats(path: str, feats: list[tuple[int, str]]) -> dict[int, dict]:
    stats: dict[int, dict] = {}
    for i in range(0, len(feats), COLUMN_BLOCK):
        block = feats[i : i + COLUMN_BLOCK]
        tbl = pq.read_table(path, columns=[nm for _, nm in block])
        for idx, nm in block:
            a = tbl[nm].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            n_nan = int(np.isnan(a).sum())
            n_inf = int(np.isinf(a).sum())
            fin = a[np.isfinite(a)]
            if fin.size == 0:
                stats[idx] = dict(
                    n_nan=n_nan, n_inf=n_inf, n=int(a.size), all_nonfinite=True,
                    min=float("nan"), max=float("nan"), p1=float("nan"),
                    p50=float("nan"), p99=float("nan"), p999=float("nan"),
                    constant=True, tail_ratio=float("nan"),
                )
                continue
            mn, mx = float(fin.min()), float(fin.max())
            p1, p50, p99, p999 = (
                float(np.percentile(fin, 1.0)),
                float(np.percentile(fin, 50.0)),
                float(np.percentile(fin, 99.0)),
                float(np.percentile(fin, 99.9)),
            )
            # Tail ratio is scale-free and sign-aware: how far past the 99th
            # percentile the single most extreme value reaches.
            denom = max(abs(p99), abs(p1))
            tail = float(max(abs(mx), abs(mn)) / denom) if denom > 0 else float("inf")
            stats[idx] = dict(
                n_nan=n_nan, n_inf=n_inf, n=int(a.size), all_nonfinite=False,
                min=mn, max=mx, p1=p1, p50=p50, p99=p99, p999=p999,
                constant=bool(mn == mx), tail_ratio=tail,
            )
        del tbl
    return stats


# ---------------------------------------------------------------------------
# Exact-duplicate feature rows. Streams batches; stores only 16-byte digests.
# ---------------------------------------------------------------------------
def duplicate_rows(path: str, feats: list[tuple[int, str]], batch_size: int = 20000):
    names = [nm for _, nm in feats]
    f = pq.ParquetFile(path)
    seen: dict[bytes, int] = {}
    n_rows = 0
    n_dup = 0
    for batch in f.iter_batches(batch_size=batch_size, columns=names):
        cols = [
            batch.column(j).to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
            for j in range(batch.num_columns)
        ]
        mat = np.ascontiguousarray(np.stack(cols, axis=1))
        for r in range(mat.shape[0]):
            d = hashlib.blake2b(mat[r].tobytes(), digest_size=16).digest()
            if d in seen:
                n_dup += 1
                seen[d] += 1
            else:
                seen[d] = 1
            n_rows += 1
    n_unique = len(seen)
    worst = max(seen.values()) if seen else 0
    return dict(
        n_rows=n_rows, n_unique=n_unique, n_dup=n_dup,
        dup_mass=(n_dup / n_rows) if n_rows else 0.0, max_multiplicity=worst,
    )


# ---------------------------------------------------------------------------
# Target checks.
# ---------------------------------------------------------------------------
def target_checks(path: str, target_col: str = "human_score") -> dict:
    t = pq.read_table(path, columns=[target_col])
    a = t[target_col].to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    n = int(a.size)
    n_nan = int(np.isnan(a).sum())
    n_inf = int(np.isinf(a).sum())
    fin = a[np.isfinite(a)]
    mn = float(fin.min()) if fin.size else float("nan")
    mx = float(fin.max()) if fin.size else float("nan")
    # TWO different tie statistics; they are not interchangeable and conflating them
    # overstates the damage by orders of magnitude.
    #   row_tie_rate  = fraction of ROWS sharing their value with >=1 other row.
    #                   Descriptive only: says the target is coarsely quantized.
    #   pair_tie_prob = P(two INDEPENDENTLY drawn rows have equal targets) = sum (n_v/N)^2.
    #                   THIS is the quantity the sampler cares about, because the
    #                   trainer draws two uniform row indices and drops the pair when
    #                   their targets are exactly equal.
    # KADID is the worked example: 241 distinct values over 10,125 rows gives
    # row_tie_rate 99.6% but pair_tie_prob only ~0.5%.
    if fin.size:
        _, counts = np.unique(fin, return_counts=True)
        n_tied = int(counts[counts > 1].sum())
        row_tie_rate = n_tied / fin.size
        p = counts.astype(np.float64) / fin.size
        pair_tie_prob = float((p * p).sum())
        n_distinct = int(counts.size)
    else:
        row_tie_rate, pair_tie_prob, n_distinct = float("nan"), float("nan"), 0
    below = int((fin < TARGET_MIN_ADVISORY).sum()) if fin.size else 0
    # A2 companion: how much SQUARED-ERROR mass the deep-negative tail carries. A group
    # whose loss_mode has an MSE term sees `target * target_scale`, so a row at -7.39
    # contributes (739)^2 against a typical (100)^2 -- two orders of magnitude more
    # gradient than an ordinary row, from 0.01% of the data.
    se_share = float("nan")
    if fin.size:
        se = (fin * 100.0) ** 2
        tot = float(se.sum())
        if tot > 0:
            se_share = float(se[fin < TARGET_MIN_ADVISORY].sum() / tot)
    return dict(
        n=n, n_nan=n_nan, n_inf=n_inf, min=mn, max=mx,
        row_tie_rate=row_tie_rate, pair_tie_prob=pair_tie_prob, n_distinct=n_distinct,
        n_below_advisory=below,
        frac_below_advisory=(below / fin.size) if fin.size else 0.0,
        se_share_below_advisory=se_share,
    )


# ---------------------------------------------------------------------------
# Verdict assembly.
# ---------------------------------------------------------------------------
def audit_table(path: str, name: str, winsor_guarded: set[int] | None = None,
                loss_mode: str = "both", do_dups: bool = True) -> dict:
    winsor_guarded = winsor_guarded or set()
    pf = pq.ParquetFile(path)
    schema = pf.schema_arrow
    feats = _feat_cols(schema)
    res: dict = {
        "name": name, "path": path, "rows": pf.metadata.num_rows,
        "n_features": len(feats), "loss_mode": loss_mode, "checks": {},
    }
    if not feats:
        res["checks"]["_schema"] = {"verdict": "NOT-CHECKABLE",
                                    "detail": "no f<N> feature columns"}
        return res

    tc = target_checks(path)
    res["target"] = tc

    # A1 target finiteness
    res["checks"]["A1_target_finite"] = {
        "verdict": "PASS" if (tc["n_nan"] == 0 and tc["n_inf"] == 0) else "FINDING",
        "detail": f"nan={tc['n_nan']} inf={tc['n_inf']}",
    }
    # A2 target range. The hard bound is a unit-sanity test; the deep-negative tail is
    # reported by its SQUARED-ERROR share, which is what an MSE-carrying group feels.
    a2_bad = (tc["min"] < TARGET_MIN_HARD) or (tc["max"] > TARGET_MAX_HARD)
    if (not a2_bad) and loss_mode != "rank" and tc["se_share_below_advisory"] > 0.05:
        a2_bad = True  # outlier rows dominating the absolute term
    res["checks"]["A2_target_range"] = {
        "verdict": "FINDING" if a2_bad else "PASS",
        "detail": (f"[{tc['min']:.4f}, {tc['max']:.4f}]; "
                   f"{tc['n_below_advisory']} rows ({tc['frac_below_advisory']*100:.3f}%) "
                   f"below advisory {TARGET_MIN_ADVISORY}; "
                   f"they carry {tc['se_share_below_advisory']*100:.2f}% of the "
                   f"group's squared-error mass"),
    }
    # A5 target degeneracy. Only rank-mode groups DROP tied pairs, and the quantity
    # that matters is the PAIR-collision probability, not the per-row tie rate.
    if loss_mode == "rank":
        res["checks"]["A5_target_ties"] = {
            "verdict": "FINDING" if tc["pair_tie_prob"] > TIE_RATE_THRESHOLD else "PASS",
            "detail": (f"pair_tie_prob={tc['pair_tie_prob']*100:.3f}% "
                       f"(row_tie_rate={tc['row_tie_rate']*100:.2f}%, "
                       f"{tc['n_distinct']} distinct / {tc['n']} rows); "
                       f"rank-mode DROPS tied pairs"),
        }
    else:
        res["checks"]["A5_target_ties"] = {
            "verdict": "PASS",
            "detail": (f"pair_tie_prob={tc['pair_tie_prob']*100:.3f}% "
                       f"(row_tie_rate={tc['row_tie_rate']*100:.2f}%); "
                       f"loss_mode={loss_mode} carries an MSE term so tied pairs "
                       f"are NOT dropped"),
        }

    cs = column_stats(path, feats)
    res["_column_stats"] = cs

    # B1 feature finiteness
    bad = {i: s for i, s in cs.items() if s["n_nan"] or s["n_inf"]}
    res["checks"]["B1_feature_finite"] = {
        "verdict": "PASS" if not bad else "FINDING",
        "detail": ("all finite" if not bad
                   else f"{len(bad)} columns with non-finite values: "
                        f"{sorted(bad)[:12]}"),
        "columns": sorted(bad),
    }
    # B2 constant columns outside the structural-zero block
    const_all = sorted(i for i, s in cs.items() if s["constant"])
    const_struct = [i for i in const_all if _is_structural_zero(i)]
    const_other = [i for i in const_all if not _is_structural_zero(i)]
    res["checks"]["B2_constant_columns"] = {
        "verdict": "PASS" if not const_other else "FINDING",
        "detail": (f"{len(const_all)} constant total; {len(const_struct)} are the "
                   f"structural-zero block f{STRUCTURAL_ZERO_LO}..f{STRUCTURAL_ZERO_HI} "
                   f"(by design); {len(const_other)} OUTSIDE it: {const_other[:24]}"),
        "constant_all": const_all,
        "constant_structural": const_struct,
        "constant_other": const_other,
    }
    # B4 unguarded heavy tails. BOTH declared transform kinds tame a tail:
    # `winsor_p99` clips it outright, `signed_cbrt` compresses it (a 776x excursion
    # becomes ~9x under a cube root). Counting only winsor would mis-flag every
    # signed_cbrt column -- f38 is the worked example (776x raw, cbrt-guarded).
    heavy = [
        i for i, s in cs.items()
        if (not s["constant"]) and np.isfinite(s["tail_ratio"])
        and s["tail_ratio"] > TAIL_RATIO_THRESHOLD and i not in winsor_guarded
    ]
    heavy.sort(key=lambda i: -cs[i]["tail_ratio"])
    res["checks"]["B4_unguarded_tails"] = {
        "verdict": "PASS" if not heavy else "FINDING",
        "detail": (f"{len(heavy)} columns with max/p99 > {TAIL_RATIO_THRESHOLD} and NO "
                   f"declared transform"
                   + (f"; worst: " + ", ".join(
                       f"f{i}={cs[i]['tail_ratio']:.3g}x" for i in heavy[:8]) if heavy else "")),
        "columns": heavy,
    }
    # B4b: guards declared for columns that are CONSTANT in this table. A winsor
    # guard on a never-populated slot is inert -- it says the screen that produced it
    # was fit somewhere the column was live, or fit without checking.
    wasted = sorted(i for i in winsor_guarded
                    if i in cs and cs[i]["constant"])
    res["checks"]["B4b_inert_guards"] = {
        "verdict": "PASS" if not wasted else "FINDING",
        "detail": (f"{len(wasted)} declared transforms target columns that are CONSTANT "
                   f"here (inert): {wasted}"),
        "columns": wasted,
    }

    if do_dups:
        d = duplicate_rows(path, feats)
        res["duplicates"] = d
        res["checks"]["C1_duplicate_rows"] = {
            "verdict": "FINDING" if d["dup_mass"] > DUP_MASS_THRESHOLD else "PASS",
            "detail": (f"{d['n_dup']} duplicate rows / {d['n_rows']} "
                       f"({d['dup_mass']*100:.2f}%); {d['n_unique']} unique; "
                       f"max multiplicity {d['max_multiplicity']}"),
        }
    else:
        res["checks"]["C1_duplicate_rows"] = {
            "verdict": "NOT-CHECKABLE", "detail": "skipped (--no-dups)"}
    return res


def twin_check(teacher_path: str, base_path: str, teacher_name: str,
               base_name: str, target_col: str = "human_score") -> dict:
    """A6/A7/E1 — a DISTILLATION TEACHER twin against the leg it twins.

    A teacher twin carries the base leg's feature rows VERBATIM with `human_score`
    replaced by a teacher model's prediction, so two things must hold:

      A6  row correspondence  the `ref_basename` sequence is identical, row for row.
                              A mismatch is a join error and voids every number from
                              every bake trained with that twin.
      A7  target agreement    the twin's target ranks its rows roughly like the base
                              target does. A twin is ALLOWED to disagree (that is the
                              point of distillation), but a twin that agrees at rho
                              ~0.25 is not teaching a refinement of the base signal,
                              it is teaching a different and largely unrelated one --
                              while consuming the same sampling mass.

    This check exists because it found exactly that on 2026-08-04: `tkadis` twins
    `kadis` at signed SROCC +0.2485 with a systematic +0.579 median offset and 55% of
    rows past |delta| > 0.5, at 7.87% of the mix's sampling mass (Appendix G).
    """
    from scripts.lib.zen_stats import panel_batch  # local: keeps import cost off the
                                                   # table-only path

    tt = pq.read_table(teacher_path, columns=["ref_basename", target_col])
    bt = pq.read_table(base_path, columns=["ref_basename", target_col])
    trb = tt["ref_basename"].to_pylist()
    brb = bt["ref_basename"].to_pylist()
    ts = np.asarray(tt[target_col].to_pylist(), dtype=float)
    bs = np.asarray(bt[target_col].to_pylist(), dtype=float)

    row_match = len(trb) == len(brb)
    seq_ok = row_match and trb == brb
    out = {"teacher": teacher_name, "base": base_name,
           "rows_teacher": len(trb), "rows_base": len(brb),
           "checks": {}}
    out["checks"]["A6_twin_row_correspondence"] = {
        "verdict": "PASS" if seq_ok else "FINDING",
        "detail": ("ref_basename sequence identical row-for-row" if seq_ok else
                   f"MISMATCH: rows {len(trb)} vs {len(brb)}, "
                   f"sequence_identical={seq_ok} — this is a JOIN ERROR"),
    }
    if not row_match:
        out["checks"]["A7_twin_target_agreement"] = {
            "verdict": "NOT-CHECKABLE", "detail": "row counts differ"}
        return out

    d = ts - bs
    # |SROCC| from zenstats' batch fast path (full-panel PWRC is O(n^2) and these
    # vectors reach 208k rows); SIGN from midrank covariance, as the orientation
    # gate does. No stat math is implemented here.
    mag = float(panel_batch([("twin", list(ts), list(bs))], stats="srocc")[0]["srocc"])

    def _midrank(v):
        o = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=float)
        r[o] = np.arange(1, len(v) + 1, dtype=float)
        return r

    cov = np.cov(_midrank(ts), _midrank(bs))[0, 1]
    srocc = mag * (1.0 if cov >= 0 else -1.0)
    mean_abs = float(np.abs(d).mean())
    med = float(np.median(d))
    frac_big = float((np.abs(d) > 0.5).mean())
    bad = (srocc < 0.5) or (mean_abs > 0.1) or (abs(med) > 0.05) or (frac_big > 0.01)
    out["srocc_signed"] = srocc
    out["mean_abs_delta"] = mean_abs
    out["median_delta"] = med
    out["frac_abs_delta_gt_0p5"] = frac_big
    out["checks"]["A7_twin_target_agreement"] = {
        "verdict": "FINDING" if bad else "PASS",
        "detail": (f"srocc={srocc:+.4f} mean|d|={mean_abs:.4f} median_d={med:+.4f} "
                   f"frac|d|>0.5={frac_big*100:.3f}%"),
    }
    return out


def leakage_check(train: dict[str, str], evalsets: dict[str, str],
                  key: str = "ref_basename") -> dict:
    """C4/C5 — reference identity, not filename equality.

    Two guards keep this from reporting nonsense:

    1. SELF-COMPARISON. Several training legs are themselves files in the canonical
       root (`safesyn` IS `ext_safesyn_full.parquet`). A table always shares 100% of
       its references with itself; that is identity, not leakage. Compared by resolved
       real path, not by name.
    2. NAMESPACE COLLISION. KADID and TID both label their references `I01`..`I25`.
       Those are DIFFERENT images that happen to share a label, so a bare set
       intersection between them is meaningless and is reported NOT-CHECKABLE rather
       than as 25 hits. This is exactly why the check is specified on reference
       IDENTITY and not on filename equality.
    """
    def _norm(n: str) -> str:
        return n[4:] if n.startswith("ext_") else n

    COLLIDING = {("kadid", "tid"), ("tid", "kadid")}

    def refs(p):
        return set(pq.read_table(p, columns=[key])[key].to_pylist())

    T = {k: (refs(v), os.path.realpath(v)) for k, v in train.items()}
    E = {k: (refs(v), os.path.realpath(v)) for k, v in evalsets.items()}
    out = {"key": key, "train_ref_counts": {k: len(v[0]) for k, v in T.items()},
           "eval_ref_counts": {k: len(v[0]) for k, v in E.items()},
           "hits": {}, "matrix": {}, "skipped": {}}
    for tn, (ts, tp) in T.items():
        out["matrix"][tn] = {}
        for en, (es, ep) in E.items():
            if tp == ep:
                out["matrix"][tn][en] = "SELF"
                out["skipped"][f"{tn}->{en}"] = "same file"
                continue
            if (_norm(tn), _norm(en)) in COLLIDING:
                out["matrix"][tn][en] = "NOT-CHECKABLE"
                out["skipped"][f"{tn}->{en}"] = "colliding I01..I25 label namespace"
                continue
            inter = ts & es
            out["matrix"][tn][en] = len(inter)
            if inter:
                out["hits"][f"{tn}->{en}"] = sorted(inter)[:50]
    return out


def parse_winsor_from_spec(spec_path: str) -> set[int]:
    """Feature indices the recipe declares ANY tail-taming transform for.

    Both registered kinds count: `winsor_p99:<idx>:<lo>,<hi>` clips the tail,
    `signed_cbrt:<idx>:` compresses it. A column carrying either is guarded.
    """
    d = json.load(open(spec_path))
    argv = d.get("argv", [])
    guarded = set()
    for i, a in enumerate(argv):
        if a == "--feature-transform" and i + 1 < len(argv):
            v = argv[i + 1]
            parts = v.split(":")
            if len(parts) >= 2 and parts[0] in ("winsor_p99", "signed_cbrt"):
                try:
                    guarded.add(int(parts[1]))
                except ValueError:
                    pass
    return guarded


def parse_mix_from_spec(spec_path: str) -> list[dict]:
    """The 11 groups of a bake's embedded repro, as (name, path, weights, loss_mode)."""
    d = json.load(open(spec_path))
    argv = d.get("argv", [])
    inputs = {e["name"]: e for e in d.get("inputs", [])}
    groups = []
    for i, a in enumerate(argv):
        if a == "--group" and i + 1 < len(argv):
            spec = argv[i + 1]
            parts = spec.split(":")
            nm = parts[0]
            g = {
                "name": nm,
                "path_recorded": parts[1] if len(parts) > 1 else "",
                "train_w": float(parts[2]) if len(parts) > 2 else 1.0,
                "val_w": float(parts[3]) if len(parts) > 3 else 0.0,
                "loss_mode": parts[4] if len(parts) > 4 else "rank",
            }
            if nm in inputs:
                g["rows_recorded"] = inputs[nm].get("rows")
                g["sha256_recorded"] = inputs[nm].get("sha256")
            groups.append(g)
    return groups


def sampling_mass(groups: list[dict], row_counts: dict[str, int],
                  tie_rates: dict[str, float]) -> list[dict]:
    """D1/D2 — derived FROM the trainer source, not guessed.

    `zensim-validate/src/mlp_train/mod.rs` builds the group CDF from
    `train_weight / sum(train_weight)` and then draws two row indices UNIFORMLY inside
    the chosen group. So the expected pair share is `train_w / sum(train_w)` and is
    INDEPENDENT of row count. Two draws are then wasted rather than redrawn:
      * `ia == ib`               -> `continue`             (probability 1/n)
      * rank-mode, tied target   -> `continue`             (probability P(tie pair))
    """
    tw = {g["name"]: g["train_w"] for g in groups}
    total = sum(v for v in tw.values() if v > 0)
    out = []
    for g in groups:
        w = g["train_w"]
        n = row_counts.get(g["name"], 0)
        nominal = (w / total) if (w > 0 and total > 0) else 0.0
        p_same = (1.0 / n) if n else 0.0
        # For a rank-mode group a pair is dropped when the two targets are exactly
        # equal. Under uniform draws that probability is the collision probability of
        # the target-value distribution; the per-row tie RATE is an upper bound on it
        # and is what the table reports, so the effective share is bracketed.
        p_tie = tie_rates.get(g["name"], 0.0) if g["loss_mode"] == "rank" else 0.0
        eff = nominal * (1.0 - p_same) * (1.0 - p_tie)
        out.append({
            "name": g["name"], "rows": n, "train_w": w, "val_w": g["val_w"],
            "loss_mode": g["loss_mode"],
            "row_share": (n / sum(row_counts.values())) if row_counts else 0.0,
            "nominal_pair_share": nominal,
            "effective_pair_share": eff,
            "wasted_same_index": p_same,
            "wasted_tied_target": p_tie,
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parquet", nargs="?")
    ap.add_argument("--name", default=None)
    ap.add_argument("--mix-from-spec", help="a bake's .bin.spec.json; audits every group")
    ap.add_argument("--data-root", action="append", default=[],
                    help="prefix rewrite SRC=DST for recorded paths (repeatable)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--tsv-dir")
    ap.add_argument("--no-dups", action="store_true")
    ap.add_argument("--twin", action="append", default=[],
                    metavar="TEACHER=BASE",
                    help="A6/A7 teacher-twin check by group name (repeatable). "
                         "Only meaningful with --mix-from-spec.")
    ap.add_argument("--leak-eval-root",
                    help="C4/C5: directory of ext_*.parquet eval corpora to test every "
                         "training group's references against")
    args = ap.parse_args()

    rewrites = []
    for r in args.data_root:
        if "=" in r:
            rewrites.append(tuple(r.split("=", 1)))

    def resolve(p: str) -> str:
        for src, dst in rewrites:
            if p.startswith(src):
                return dst + p[len(src):]
        return p

    results = []
    paths_by_name: dict[str, str] = {}
    if args.mix_from_spec:
        guarded = parse_winsor_from_spec(args.mix_from_spec)
        groups = parse_mix_from_spec(args.mix_from_spec)
        for g in groups:
            p = resolve(g["path_recorded"])
            paths_by_name[g["name"]] = p
            if not os.path.exists(p):
                results.append({"name": g["name"], "path": p,
                                "checks": {"_io": {"verdict": "NOT-CHECKABLE",
                                                   "detail": "path not found"}}})
                continue
            results.append(audit_table(p, g["name"], guarded, g["loss_mode"],
                                       do_dups=not args.no_dups))
    elif args.parquet:
        # Single-table mode also feeds the C4 leakage check (2026-08-05,
        # Appendix L): a NEW training leg is exactly the case where "this one
        # table vs every eval corpus" must be answerable without a bake spec.
        paths_by_name[args.name or Path(args.parquet).stem] = args.parquet
        results.append(audit_table(args.parquet, args.name or Path(args.parquet).stem,
                                   set(), "both", do_dups=not args.no_dups))
    else:
        ap.error("need a parquet or --mix-from-spec")

    findings = 0
    for r in results:
        cs = r.pop("_column_stats", None)
        line = f"{r['name']:16s} rows={r.get('rows','?'):>8}"
        print(line)
        for cid, c in sorted(r["checks"].items()):
            v = c["verdict"]
            if v == "FINDING":
                findings += 1
            mark = {"PASS": "OK      ", "FINDING": "FINDING ",
                    "NOT-CHECKABLE": "NOTCHECK"}[v]
            print(f"   {mark} {cid:24s} {c['detail']}")
        if args.tsv_dir and cs:
            os.makedirs(args.tsv_dir, exist_ok=True)
            out = Path(args.tsv_dir) / f"colstats_{r['name']}.tsv"
            with open(out, "w") as fh:
                fh.write("feat_idx\tn\tn_nan\tn_inf\tmin\tp1\tp50\tp99\tp999\tmax"
                         "\tconstant\ttail_ratio\n")
                for i in sorted(cs):
                    s = cs[i]
                    fh.write(f"{i}\t{s['n']}\t{s['n_nan']}\t{s['n_inf']}\t{s['min']!r}"
                             f"\t{s['p1']!r}\t{s['p50']!r}\t{s['p99']!r}\t{s['p999']!r}"
                             f"\t{s['max']!r}\t{int(s['constant'])}\t{s['tail_ratio']!r}\n")
        print()

    # A6/A7 teacher twins.
    twins = []
    for spec in args.twin:
        if "=" not in spec:
            ap.error(f"--twin wants TEACHER=BASE, got {spec!r}")
        tn, bn = spec.split("=", 1)
        tp, bp = paths_by_name.get(tn) or tn, paths_by_name.get(bn) or bn
        if not (os.path.exists(tp) and os.path.exists(bp)):
            print(f"{tn} vs {bn}: NOT-CHECKABLE (missing table)")
            continue
        tw = twin_check(tp, bp, tn, bn)
        twins.append(tw)
        print(f"TWIN {tn} vs {bn}")
        for cid, c in sorted(tw["checks"].items()):
            if c["verdict"] == "FINDING":
                findings += 1
            mark = {"PASS": "OK      ", "FINDING": "FINDING ",
                    "NOT-CHECKABLE": "NOTCHECK"}[c["verdict"]]
            print(f"   {mark} {cid:24s} {c['detail']}")
        print()
    if twins:
        results.append({"twins": twins})

    # C4/C5 leakage: every training group's references vs every eval corpus.
    if args.leak_eval_root and paths_by_name:
        root = Path(args.leak_eval_root)
        train_paths = {n: p for n, p in paths_by_name.items() if os.path.exists(p)}
        # The canonical root holds TRAINING legs alongside the held-out eval corpora
        # (`ext_safesyn_full.parquet` is a training input, not a holdout). Testing a
        # training leg against itself -- or against its own teacher twin, which
        # carries the same references by construction -- reports identity as leakage.
        # Exclude by resolved path so the rule is structural, not a name allowlist.
        train_real = {os.path.realpath(p) for p in train_paths.values()}
        evalsets = {p.stem: str(p) for p in sorted(root.glob("ext_*.parquet"))
                    if os.path.realpath(str(p)) not in train_real}
        lk = leakage_check(train_paths, evalsets)
        hits = {k: v for k, v in lk["hits"].items()}
        print("LEAKAGE (reference identity, key=ref_basename)")
        if hits:
            findings += 1
            for k, v in hits.items():
                print(f"   FINDING  {k}: {len(v)}+ shared references e.g. {v[:5]}")
        else:
            print(f"   OK       no training reference appears in any of "
                  f"{len(evalsets)} eval corpora "
                  f"({len(lk['skipped'])} pairs skipped: self / colliding namespace)")
        print()
        results.append({"leakage": lk})

    if args.json:
        print(json.dumps(results, indent=1, default=str))
    print(f"TOTAL FINDINGS: {findings}")
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
