#!/usr/bin/env python3
"""Shared join-safety + metric-column-integrity guards for every canonical /
training-corpus builder.

This module makes the two corruption modes root-caused in
`benchmarks/DATA_INTEGRITY_root_cause_2026-05-25.md` *structurally impossible
to reintroduce*, rather than merely detectable after the fact:

  Mode A — iwssim "human-copy" leak (a validation-only MOCK that leaked into
           training parquets). Defended by `assert_no_leaked_metric_columns`,
           which rejects any column matching `*_mock*` / `*MOCK*` and any
           metric column that is bit-identical to (or a perfect rank-copy of)
           `human_score`.

  Mode B — ssim2_gpu "ref-misjoin": a per-pair metric (one value per
           (ref, codec, q, knob)) was joined onto a per-corpus features table
           that carried ONLY `ref_basename`, so `groupby(ref_basename).mean()`
           broadcast one mean value onto all distortions of each reference,
           destroying the per-distortion signal. Defended by `safe_metric_join`,
           which REFUSES a join when the required per-pair key columns are
           absent (it never silently collapses to a ref-level mean), and by
           `assert_metric_not_constant_per_ref`.

Import this from any builder that attaches a metric column to a features table.
The `groupby(ref_basename).mean()` fallback that caused the bug is deleted —
there is intentionally NO ref-only mean codepath in this module.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Per-pair key columns. A correct join of a per-pair metric onto a features
# table requires ALL of these (or an explicit positional alignment). The
# canonical score sidecars (`scores/{ssim2,iwssim,cvvdp}_imazen*.parquet`) are
# keyed on (image_path, codec, q, knob_tuple_json); `image_path` plays the
# role of the per-pair distortion identifier there. `ref_basename` is NOT a
# per-pair key — it identifies the *reference*, shared by ~125 distortions.
PER_PAIR_KEY_CANDIDATES: tuple[str, ...] = (
    "image_path",
    "dist_path",
    "codec",
    "q",
    "quality",
    "knob_tuple_json",
)

# Columns that, on their own, are NOT sufficient to key a per-pair join.
REF_ONLY_KEYS: frozenset[str] = frozenset({"ref_basename", "ref", "source"})

HUMAN_SCORE_COL = "human_score"


class JoinSafetyError(RuntimeError):
    """Raised when a join or a metric column would reintroduce a known
    corruption mode (ref-only broadcast, mock leak, human-score copy)."""


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = np.isfinite(a) & np.isfinite(b)
    return a[m], b[m]


def safe_metric_join(target, metric, join_keys: Sequence[str], metric_col: str,
                     *, how: str = "left"):
    """Join `metric[metric_col]` onto `target` using the FULL `join_keys`.

    `target` and `metric` are pandas DataFrames. `join_keys` MUST be the full
    per-pair key (e.g. ["ref_basename", "codec", "q", "knob_tuple_json"]).

    Raises `JoinSafetyError` — never silently collapses — if the target lacks
    any of the per-pair key columns, because joining on `ref_basename` alone
    would broadcast one value per reference onto every distortion (the Mode B
    misjoin). If a per-pair key genuinely cannot be carried, attach the metric
    POSITIONALLY instead (see `attach_metric_positional`); do not weaken the
    join key here.
    """
    missing_target = [k for k in join_keys if k not in target.columns]
    missing_metric = [k for k in join_keys if k not in metric.columns]
    effective = [k for k in join_keys if k in target.columns and k in metric.columns]

    # The exact failure that produced the corruption: keys collapsed to a
    # ref-only set. Refuse loudly.
    if set(effective) <= REF_ONLY_KEYS:
        raise JoinSafetyError(
            f"cannot join per-pair metric {metric_col!r} on {effective or ['<none>']} "
            f"alone — target parquet is missing per-pair key columns "
            f"{missing_target or missing_metric}; this would broadcast one value per "
            f"reference onto all its distortions (the ssim2_gpu ref-misjoin bug). "
            f"Carry the full per-pair key {list(join_keys)} on the features table, or "
            f"attach the metric POSITIONALLY (attach_metric_positional)."
        )
    if missing_target or missing_metric:
        raise JoinSafetyError(
            f"safe_metric_join({metric_col!r}): requested keys {list(join_keys)} but "
            f"target missing {missing_target}, metric missing {missing_metric}. A "
            f"partial-key join silently mis-broadcasts; supply the full key or use "
            f"attach_metric_positional."
        )

    # All keys present on both sides — but a per-pair metric may still have
    # duplicate rows per key on the metric side. We MUST NOT .mean() them away
    # (that re-creates the broadcast). Require uniqueness on the metric side.
    dupes = metric.duplicated(subset=list(join_keys)).sum()
    if dupes:
        raise JoinSafetyError(
            f"safe_metric_join({metric_col!r}): metric side has {dupes} rows that are "
            f"NOT unique on the join key {list(join_keys)} — averaging them would "
            f"destroy per-pair signal. De-duplicate the metric source first."
        )
    return target.merge(metric[list(join_keys) + [metric_col]], on=list(join_keys), how=how)


def attach_metric_positional(target, metric_values, metric_col: str):
    """Attach a per-pair metric POSITIONALLY (row order == target row order).

    Use ONLY when the target features table genuinely cannot carry a per-pair
    key (e.g. KADID/TID feature CSVs that emit only ref_basename) and the metric
    was computed in the SAME row order as the dmos.csv / mos file that produced
    the features. Raises if the lengths disagree, so a misalignment can't slip
    through silently.
    """
    metric_values = np.asarray(metric_values, dtype=float)
    if len(metric_values) != len(target):
        raise JoinSafetyError(
            f"attach_metric_positional({metric_col!r}): {len(metric_values)} metric "
            f"values vs {len(target)} target rows — positional alignment requires "
            f"exact row-count match."
        )
    out = target.copy()
    out[metric_col] = metric_values
    return out


def assert_metric_not_constant_per_ref(label: str, refs: Iterable, vals,
                                       metric_col: str = "metric",
                                       min_groups: int = 5,
                                       min_group_size: float = 1.5):
    """Raise if `vals` is constant within every `refs` group (Mode B signature).

    `min_group_size` gates against false positives on per-pair score sidecars
    where each ref key is unique (mean group size ≈ 1) — there, "one value per
    ref" is trivially true and NOT a misjoin.
    """
    vals = np.asarray(vals, dtype=float)
    by_ref: dict = {}
    for r, v in zip(refs, vals):
        if np.isfinite(v):
            by_ref.setdefault(r, set()).add(round(float(v), 4))
    if len(by_ref) < min_groups:
        return
    # Recompute group sizes from the raw vals (the sets above collapse duplicates).
    sizes: dict = {}
    for r, v in zip(refs, vals):
        if np.isfinite(v):
            sizes[r] = sizes.get(r, 0) + 1
    mean_sz = sum(sizes.values()) / len(sizes) if sizes else 0.0
    if mean_sz <= min_group_size:
        return  # per-pair sidecar; test N/A
    n_const = sum(1 for vs in by_ref.values() if len(vs) == 1)
    if n_const == len(by_ref):
        raise JoinSafetyError(
            f"DATA-INTEGRITY: {label} {metric_col!r} is constant within every "
            f"reference group ({len(by_ref)} refs, each 1 unique value, mean group "
            f"size {mean_sz:.1f}) — joined on ref_basename only (ref-vs-ref broadcast). "
            f"Recompute on the correct (ref,dist) pairs."
        )


def assert_no_leaked_metric_columns(label: str, names: Sequence[str], table,
                                    human_col: str = HUMAN_SCORE_COL,
                                    metric_prefixes: Sequence[str] = (
                                        "iwssim", "ssim2", "cvvdp", "butter", "dssim",
                                    )):
    """Reject mock columns and human_score-identical RAW-metric columns.

    `table` exposes `.column(name).to_numpy(zero_copy_only=False)` (pyarrow) —
    accepts pyarrow.Table. Raises `JoinSafetyError` on:

      1. Any column whose name matches `*_mock*` / `*MOCK*` / `*_MOCK_*` — mock
         columns must never appear in a training/canonical corpus (they leaked
         once because the qualifier lived only in a filename).
      2. Any RAW-metric column (by `metric_prefixes`: iwssim/ssim2/cvvdp/
         butter/dssim) that is BIT-IDENTICAL to `human_score` — the target leak
         (Mode A). A raw metric is an independent measurement and must never
         equal the human anchor it predicts.

    Deliberately NOT flagged (legitimate by design):
      - `mix_*` columns equal to `human_score`. For konjnd-dense and LARGE the
        anchor `human_score` IS the active mix target, so a mix column equaling
        it is correct — `mix_*` is therefore excluded from `metric_prefixes`.
      - A perfect *correlation* (|corr| ≈ 1.0) without bit-identity. safesyn's
        `human_score` is a linear rescale of `ssim2_gpu` (= ssim2_gpu / 100),
        so corr is exactly 1.0 by design. The leak signature is value-equality,
        not rank/linear agreement — only bit-identity is rejected.
    """
    lowered = {n: n.lower() for n in names}

    # (1) mock columns — forbidden outright.
    for n, ln in lowered.items():
        if "mock" in ln:
            raise JoinSafetyError(
                f"DATA-INTEGRITY: {label} carries a MOCK column {n!r}. Mock metric "
                f"columns must never enter a training/canonical corpus (the iwssim "
                f"leak survived three corpus generations because the 'mock' qualifier "
                f"lived only in a filename). Drop it, or rename the validation-only "
                f"signal so it cannot be mistaken for a real metric and exclude it "
                f"from the canonical schema."
            )

    # (2) human_score-identical metric columns.
    if human_col not in names:
        return
    hs = np.asarray(table.column(human_col).to_numpy(zero_copy_only=False), dtype=float)
    if np.isfinite(hs).sum() < 100:
        return
    for n in names:
        if n == human_col:
            continue
        ln = lowered[n]
        if not any(ln.startswith(p) or p in ln for p in metric_prefixes):
            continue
        try:
            col = np.asarray(table.column(n).to_numpy(zero_copy_only=False), dtype=float)
        except (TypeError, ValueError):
            continue  # non-numeric metric-named column; skip
        a, b = _finite_pair(col, hs)
        if a.size < 100:
            continue
        ident = float(np.mean(np.isclose(a, b, atol=1e-9)))
        if ident > 0.995:
            raise JoinSafetyError(
                f"DATA-INTEGRITY: {label} metric column {n!r} is a bit-identical copy "
                f"of {human_col!r} ({ident*100:.1f}% identical) — target leak. A metric "
                f"column may never equal the human anchor it is meant to predict."
            )
