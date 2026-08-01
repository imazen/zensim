"""zen_stats — the canonical Python entry point for the Mohammadi 2025
IQA statistical panel.

This module is a THIN SHIM over the Rust `panel` binary
(`zensim-validate/src/bin/panel.rs`, which wraps
`zensim_validate::panel`). It exists so Python pipelines that cannot
easily restructure to call the binary directly still get bit-identical
stats — every number comes from the same Rust code path that
`bake_verdict` / `bake_compare` use, NOT from a hand-rolled Python
reimplementation.

## Why this module exists

The dedup audit (`benchmarks/dedup_VERIFIED_synthesis_2026-05-26.md`
Tier-1 #2) found ~14 scattered Python reimplementations of SROCC / PLCC
/ KROCC / OR / PWRC / Z-RMSE, each with its own tie-handling, NaN-drop
policy, PWRC weighting convention, and OR residual rule. Those silently
changed ship/no-ship verdicts. This module replaces all of them with a
single call into the canonical Rust home.

## Verified equivalence

`scripts/verify_panel_parity.py` proves the Rust `panel` agrees with the
scipy reference (`spearmanr`/`kendalltau`/`pearsonr` + logistic fit) to
<= 1e-9 on SROCC / PLCC / KROCC / PWRC across 36 synthetic cases. (OR
and Z-RMSE are definition-dependent — see that script's footer.)

## Usage

    from scripts.lib.zen_stats import panel

    stats = panel(predicted, target)              # dict of 6 stats + n
    stats = panel(predicted, target, sigma=sig)   # + per-sample Z-RMSE
    print(stats["srocc"], stats["plcc"], ...)

## Polarity convention

Matches `panel::compute_panel`: SROCC / KROCC / PWRC are reported as
`abs()` (polarity is treated as a nuisance, since metric outputs can be
distance- or score-shaped). PLCC is computed after a 4-parameter
logistic rescale. Pass raw predicted / target — do NOT pre-flip.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from typing import Optional, Sequence

# Resolve the `panel` binary once. Prefer release, then debug. Override
# with the ZEN_PANEL_BIN env var (e.g. for CI / vast.ai images that bake
# the binary at a known path).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _find_panel_bin() -> str:
    env = os.environ.get("ZEN_PANEL_BIN")
    if env and os.path.exists(env):
        return env
    for cand in (
        os.path.join(_REPO_ROOT, "target", "release", "panel"),
        os.path.join(_REPO_ROOT, "target", "debug", "panel"),
    ):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "zen_stats: `panel` binary not found. Build it with "
        "`cargo build --release -p zensim-validate --bin panel`, or set "
        "$ZEN_PANEL_BIN to its path."
    )


def panel(
    predicted: Sequence[float],
    target: Sequence[float],
    sigma: Optional[Sequence[float]] = None,
    band: Optional[Sequence] = None,
) -> dict:
    """Compute the full Mohammadi panel via the canonical Rust `panel`.

    Args:
        predicted: metric / model outputs.
        target:    human MOS / reference quality.
        sigma:     optional per-stimulus observer σ (enables the
                   per-sample Z-RMSE; the global Z-RMSE is always
                   returned).
        band:      optional grouping key; when present the return value
                   carries a "bands" list in addition to the aggregate.

    Returns:
        For the no-band case: a dict with keys
        {n, n_dropped, srocc, plcc, krocc, or, pwrc, z_rmse,
         z_rmse_per_sample (if sigma)}.
        For the band case: the aggregate dict plus a "bands" key mapping
        each band label -> the same per-group dict.
    """
    predicted = list(predicted)
    target = list(target)
    if len(predicted) != len(target):
        raise ValueError(
            f"predicted ({len(predicted)}) and target ({len(target)}) "
            "must be the same length"
        )
    has_sigma = sigma is not None
    has_band = band is not None
    if has_sigma and len(sigma) != len(predicted):
        raise ValueError("sigma must match predicted/target length")
    if has_band and len(band) != len(predicted):
        raise ValueError("band must match predicted/target length")

    bin_path = _find_panel_bin()

    # Write a TSV the Rust bin can parse. repr(float(...)) is the
    # shortest round-trippable decimal — Rust reads it back bit-exactly.
    cols = ["predicted", "target"]
    if has_sigma:
        cols.append("sigma")
    if has_band:
        cols.append("band")
    with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
        tmp = f.name
        f.write("\t".join(cols) + "\n")
        for i in range(len(predicted)):
            row = [repr(float(predicted[i])), repr(float(target[i]))]
            if has_sigma:
                row.append(repr(float(sigma[i])))
            if has_band:
                row.append(str(band[i]))
            f.write("\t".join(row) + "\n")
    try:
        out = subprocess.run(
            [bin_path, "--input", tmp, "--json"],
            capture_output=True, text=True, timeout=300, check=True,
        )
    finally:
        os.unlink(tmp)

    groups = json.loads(out.stdout)["groups"]
    agg = next(g for g in groups if g["label"] == "ALL")
    result = {k: (float("nan") if v is None else v) for k, v in agg.items()
              if k not in ("label",)}
    if has_band:
        result["bands"] = {
            g["label"]: {k: (float("nan") if v is None else v)
                         for k, v in g.items() if k != "label"}
            for g in groups if g["label"] != "ALL"
        }
    return result


# ----------------------------------------------------------------------
# Batch mode (decision_surface_audit_2026-07-31.md gap 4)
#
# Per-call shelling is fine for aggregates but prohibitive inside
# bootstrap loops (10k resamples = 10k process spawns). These wrappers drive
# `panel --batch`: N (x, y) vector pairs in, N stat rows out, ONE
# process. The caller keeps ownership of any resampling RNG (send the
# index sets), so recorded bootstrap numbers stay bit-reproducible and
# the Rust side stays RNG-free/deterministic.
# ----------------------------------------------------------------------

_BATCH_FULL_COLS = ("n", "n_dropped", "srocc", "srocc_signed", "plcc",
                    "plcc_raw", "krocc", "or", "pwrc", "z_rmse")
_BATCH_SROCC_COLS = ("n", "n_dropped", "srocc", "srocc_signed")


def _run_batch(text: str, stats: str, timeout: float) -> list[dict]:
    if stats not in ("full", "srocc"):
        raise ValueError(f"stats must be 'full' or 'srocc', got {stats!r}")
    bin_path = _find_panel_bin()
    with tempfile.NamedTemporaryFile("w", suffix=".batch.tsv", delete=False) as f:
        tmp = f.name
        f.write(text)
    try:
        out = subprocess.run(
            [bin_path, "--batch", tmp, "--stats", stats],
            capture_output=True, text=True, timeout=timeout, check=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"panel --batch failed: {e.stderr.strip()}") from e
    finally:
        os.unlink(tmp)
    lines = out.stdout.splitlines()
    header = lines[0].split("\t")
    want = ("label",) + (_BATCH_SROCC_COLS if stats == "srocc" else _BATCH_FULL_COLS)
    if tuple(header) != want:
        raise RuntimeError(f"unexpected batch header {header!r}")
    rows = []
    for line in lines[1:]:
        parts = line.split("\t")
        d = {"label": parts[0]}
        for k, v in zip(header[1:], parts[1:]):
            d[k] = int(v) if k in ("n", "n_dropped") else float(v)
        rows.append(d)
    return rows


def _fmt_vec(v) -> str:
    # repr(float(...)) is the shortest round-trippable decimal — the
    # Rust side reads it back bit-exactly (same convention as panel()).
    return ",".join(repr(float(x)) for x in v)


def panel_batch(jobs, stats: str = "full", timeout: float = 1800.0) -> list[dict]:
    """N explicit (x, y) pairs -> N stat rows, one `panel` process.

    Args:
        jobs:  iterable of (label, x, y) with x/y equal-length float
               sequences. x = predicted / metric, y = target / human
               (same convention as `panel`).
        stats: "full" (everything the aggregate panel emits, plus
               srocc_signed + plcc_raw) or "srocc" (bootstrap fast path).

    Returns one dict per job, in input order. Non-finite (x, y) rows are
    dropped per job and counted in `n_dropped` (aggregate-mode policy).
    """
    parts = []
    for label, x, y in jobs:
        if "\t" in str(label) or "\n" in str(label):
            raise ValueError(f"label {label!r} must not contain tab/newline")
        if len(x) != len(y):
            raise ValueError(f"job {label!r}: x ({len(x)}) != y ({len(y)})")
        parts.append(f"{label}\t{_fmt_vec(x)}\t{_fmt_vec(y)}\n")
    return _run_batch("".join(parts), stats, timeout)


def panel_batch_indexed(bases: dict, jobs, stats: str = "full",
                        timeout: float = 1800.0) -> list[dict]:
    """N index-set resamples over shared base vectors -> N stat rows.

    The paired-bootstrap shape: declare each base vector ONCE, then each
    job references two bases plus an index set applied to BOTH (or None
    for all rows). ~n integers per job instead of ~2n floats.

    Args:
        bases: {name: float sequence}. Names must be tab/colon-free.
        jobs:  iterable of (label, x_name, y_name, indices_or_None).
        stats: "full" or "srocc".
    """
    parts = []
    for name, v in bases.items():
        if any(c in str(name) for c in "\t\n:@"):
            raise ValueError(f"base name {name!r} must not contain tab/colon/@")
        parts.append(f"#def {name}\t{_fmt_vec(v)}\n")
    for label, xn, yn, idx in jobs:
        if "\t" in str(label) or "\n" in str(label):
            raise ValueError(f"label {label!r} must not contain tab/newline")
        if xn not in bases or yn not in bases:
            raise ValueError(f"job {label!r}: undefined base {xn!r} or {yn!r}")
        sel = "*" if idx is None else ",".join(str(int(i)) for i in idx)
        parts.append(f"{label}\t@{xn}:@{yn}\t{sel}\n")
    return _run_batch("".join(parts), stats, timeout)


# Convenience single-stat accessors for drop-in replacement of the
# retired one-off `def srocc(...)` / `def spearman(...)` helpers. Each
# delegates to `panel` so there is still exactly ONE stat code path.
def srocc(predicted, target) -> float:
    return panel(predicted, target)["srocc"]


def plcc(predicted, target) -> float:
    return panel(predicted, target)["plcc"]


def krocc(predicted, target) -> float:
    return panel(predicted, target)["krocc"]


def pwrc(predicted, target) -> float:
    return panel(predicted, target)["pwrc"]


def outlier_ratio(predicted, target) -> float:
    return panel(predicted, target)["or"]


def z_rmse(predicted, target) -> float:
    return panel(predicted, target)["z_rmse"]
