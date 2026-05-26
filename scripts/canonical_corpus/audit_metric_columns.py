#!/usr/bin/env python3
"""Census of iwssim / ssim2_gpu metric-column integrity across every
training/validation parquet under /mnt/v/zen/zensim-training/.

Detects the two corruption modes root-caused in
benchmarks/DATA_INTEGRITY_root_cause_2026-05-25.md:

  * iwssim "human-copy" leak — `iwssim` (or `iwssim_log_norm` derived
    from it) is a verbatim copy of `human_score`. Origin:
    scripts/v_next/v0_22_iw_make_mock_val_csvs.sh:56 (`iwssim := human_score`)
    created as a VALIDATION-ONLY mock, then mispropagated into training
    parquets with the "mock" qualifier lost.

  * ssim2_gpu "ref-misjoin" — `ssim2_gpu` is constant within every
    reference group (≈ one unique value per ref). Origin:
    scripts/v_next/build_ex3_mix_corpus.py:add_ssim2_to_372feat_corpus()
    groupby(["ref_basename"]).mean() + merge-on-ref_basename-alone,
    because the 372-feat targets parquet has NO codec/q column to key on.

For each parquet with an iwssim and/or ssim2_gpu column, emit:
  parquet_path, rows, has_iwssim, iwssim_corr, iwssim_identical_pct,
  has_ssim2, ssim2_corr, ssim2_constant_per_ref, verdict

Read-only. Writes a TSV to stdout (and an optional --out file).
"""
from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np
import pyarrow.parquet as pq

ROOT = "/mnt/v/zen/zensim-training"

# Treat these as the human-anchor column candidates per corpus.
HUMAN_COLS = ("human_score",)
# iwssim candidate columns (raw + derived).
IWSSIM_COLS = ("iwssim", "iwssim_imazen_v0_0_1")
SSIM2_COLS = ("ssim2_gpu",)
REF_COLS = ("ref_basename", "image_path")


def _np(col):
    a = np.asarray(col, dtype=np.float64)
    return a


def _corr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    aa, bb = a[m], b[m]
    if np.std(aa) == 0 or np.std(bb) == 0:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def _constant_per_ref(refs, vals) -> tuple[float, float]:
    """Returns (frac_const, mean_group_size).

    frac_const = fraction of reference groups that hold exactly one unique
    value. 1.0 => ssim2 is broadcast per-ref (the misjoin signature) — BUT
    only meaningful when groups actually bundle >1 distortion. When every
    `ref` value is unique (mean_group_size ≈ 1, e.g. a per-pair score
    sidecar keyed on the full distorted path), the test is a false positive;
    callers must gate on mean_group_size > 1.5."""
    import collections

    groups = collections.defaultdict(list)
    for r, v in zip(refs, vals):
        if np.isfinite(v):
            groups[r].append(round(float(v), 4))
    if not groups:
        return float("nan"), float("nan")
    n_const = sum(1 for vs in groups.values() if len(set(vs)) == 1)
    mean_sz = sum(len(vs) for vs in groups.values()) / len(groups)
    return n_const / len(groups), mean_sz


def audit_one(path: str) -> dict | None:
    try:
        schema = pq.read_schema(path)
    except Exception as e:
        return {"path": path, "verdict": f"UNREADABLE ({e})"}
    names = set(schema.names)

    iw_col = next((c for c in IWSSIM_COLS if c in names), None)
    s2_col = next((c for c in SSIM2_COLS if c in names), None)
    if iw_col is None and s2_col is None:
        return None  # not relevant

    human_col = next((c for c in HUMAN_COLS if c in names), None)
    ref_col = next((c for c in REF_COLS if c in names), None)

    cols = [c for c in (iw_col, s2_col, human_col, ref_col) if c]
    t = pq.read_table(path, columns=cols).to_pandas()
    n = len(t)

    out = {
        "path": path.replace(ROOT + "/", ""),
        "rows": n,
        "has_iwssim": int(iw_col is not None),
        "iwssim_corr": "",
        "iwssim_identical_pct": "",
        "has_ssim2": int(s2_col is not None),
        "ssim2_corr": "",
        "ssim2_constant_per_ref": "",
        "verdict": "",
    }

    verdicts = []
    hs = _np(t[human_col]) if human_col else None

    if iw_col is not None:
        iw = _np(t[iw_col])
        if hs is not None:
            out["iwssim_corr"] = f"{_corr(iw, hs):.4f}"
            ident = float(np.mean(np.isclose(iw, hs, atol=1e-9, equal_nan=False)) * 100)
            out["iwssim_identical_pct"] = f"{ident:.1f}"
            if ident >= 99.5:
                verdicts.append("iwssim=HUMAN-COPY")
            elif abs(_corr(iw, hs)) > 0.999:
                verdicts.append("iwssim≈HUMAN-COPY")
            else:
                verdicts.append("iwssim=real?")
        else:
            verdicts.append("iwssim=present(no-human-col)")

    if s2_col is not None:
        s2 = _np(t[s2_col])
        if hs is not None:
            out["ssim2_corr"] = f"{_corr(s2, hs):.4f}"
        if ref_col is not None:
            cper, mean_sz = _constant_per_ref(t[ref_col].to_numpy(), s2)
            out["ssim2_constant_per_ref"] = f"{cper:.3f}"
            # The misjoin signature requires groups that actually bundle
            # multiple distortions. A per-pair score sidecar (one row per
            # ref key) trivially has cper=1.0 but is NOT misjoined.
            if mean_sz <= 1.5:
                verdicts.append("ssim2=per-pair-sidecar(test-N/A)")
            elif cper >= 0.95:
                verdicts.append("ssim2=REF-MISJOIN")
            elif hs is not None and abs(_corr(s2, hs)) > 0.30:
                verdicts.append("ssim2=real?")
            else:
                verdicts.append("ssim2=suspect")
        else:
            verdicts.append("ssim2=present(no-ref-col)")

    out["verdict"] = "; ".join(verdicts) if verdicts else "OK"
    return out


# Verdict substrings that mean a training/canonical parquet is CORRUPT.
# (per-pair-sidecar + "real?" + "OK" are clean; mock/human-copy/misjoin are not)
CORRUPT_VERDICT_MARKERS = (
    "HUMAN-COPY",
    "REF-MISJOIN",
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None, help="write TSV here too")
    ap.add_argument(
        "--fail-on-corruption",
        action="store_true",
        help="exit nonzero if any audited parquet carries a HUMAN-COPY iwssim "
        "or REF-MISJOIN ssim2_gpu column. Use as a CI / pre-train gate.",
    )
    ap.add_argument(
        "--paths",
        nargs="*",
        default=None,
        help="explicit parquet paths to audit (overrides --root glob). Use for "
        "CI fixtures or a targeted train-set census.",
    )
    args = ap.parse_args()

    if args.paths:
        paths = sorted(args.paths)
    else:
        paths = sorted(
            p for p in glob(os.path.join(args.root, "**", "*.parquet"), recursive=True)
        )
    cols = [
        "path",
        "rows",
        "has_iwssim",
        "iwssim_corr",
        "iwssim_identical_pct",
        "has_ssim2",
        "ssim2_corr",
        "ssim2_constant_per_ref",
        "verdict",
    ]
    rows = []
    for p in paths:
        try:
            r = audit_one(p)
        except Exception as e:
            r = {"path": p.replace(args.root + "/", ""), "verdict": f"ERROR ({e})"}
        if r is not None:
            rows.append(r)

    lines = ["\t".join(cols)]
    for r in rows:
        lines.append("\t".join(str(r.get(c, "")) for c in cols))
    text = "\n".join(lines)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n[wrote {len(rows)} relevant parquets to {args.out}]", file=sys.stderr)

    corrupt = [
        r for r in rows
        if any(m in str(r.get("verdict", "")) for m in CORRUPT_VERDICT_MARKERS)
    ]
    if corrupt:
        print(
            f"\n[CORRUPTION CENSUS] {len(corrupt)} parquet(s) carry a leaked / "
            f"misjoined metric column:",
            file=sys.stderr,
        )
        for r in corrupt:
            print(f"  - {r['path']}: {r['verdict']}", file=sys.stderr)
        if args.fail_on_corruption:
            print(
                "\nERROR: --fail-on-corruption set and corruption detected. "
                "Recompute the affected columns (fix_kadid_tid_*.py) before training.",
                file=sys.stderr,
            )
            return 1
    elif args.fail_on_corruption:
        print(
            f"\n[CORRUPTION CENSUS] clean — {len(rows)} audited parquet(s), no "
            f"HUMAN-COPY / REF-MISJOIN columns.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
