#!/usr/bin/env python3
"""EXP-CROSS-CODEC multi-band cross-codec consistency check — any generation.

Replaces eval_v{5,6,7,8}_multi_band_check.py (2026-07-15), 943 lines. Their
real shape, once the version token is normalized away:

  - v5 and v6 are the SAME PROGRAM. Same anchor parquet, same logic. The whole
    diff is `s/v5/v6/` plus a docstring — and v6 additionally DELETED three of
    v5's explanatory comments, so the copy is worse-documented than its source.
    Forks do not just drift; they lose information as they go.
  - v7 is a genuine variant: an empirical anchor whose target_score is the
    per-(codec, band) median ssim2, plus a per-codec achieved-vs-target table.
  - v8 points at a third anchor AND QUIETLY CHANGED THE GATE. v5/v6/v7 pass a
    band on `cc_std_median <= 5`. v8 passes only on `cc_std_median <= 5 AND
    |achieved_mean - target| <= 5`. Both reports print the word "PASS". Anyone
    comparing a v6 report against a v8 report was comparing two different
    criteria under one label, with nothing on the page to say so. A fork chain
    does not just drift in prose — it can move the gate.

So the axes are three flags, not four files: which anchor, whether targets are
per-codec, and whether the target error gates. That is what this takes, and it
makes the gate difference visible in the report instead of buried in a copy.

The prose is deliberately NOT reproduced verbatim. Each copy's header described
itself relative to the previous one ("V6 mirrors V5's multi-band gate check
but...") — chain-prose that means nothing once the chain is gone, and that is
exactly where the sibling eval_v*_pjnd_check.py family accumulated three
outright false statements. Numbers are reproduced exactly; the framing is
rewritten to stand alone.

The gate: for each anchor band, the cross-codec score std per source must be
<= 5.0 — the metric should agree on a band regardless of which codec got there.

Usage:
    python3 scripts/v_next/cross_codec_multi_band_check.py v8 \
        /mnt/v/zen/zensim-eval/exp_cross_codec_v8_2026-05-19
    python3 scripts/v_next/cross_codec_multi_band_check.py v7 <dir>   # per-codec targets
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
# Repo-relative, NOT a sibling-worktree path — see CLAUDE.md "NEVER hardcode a
# sibling-worktree path in a committed script".
SCORE_BIN = REPO / "target/release/ensemble_score_rows"

TRAIN = Path("/mnt/v/zen/zensim-training")
# Per-generation anchor. `per_codec` marks the generation whose target_score is
# empirical per-(codec, band) rather than one value for the whole band.
ANCHORS: dict[str, dict] = {
    "v5": {"path": TRAIN / "2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet",
           "per_codec": False},
    "v6": {"path": TRAIN / "2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet",
           "per_codec": False},
    "v7": {"path": TRAIN / "2026-05-19-empirical-band-anchors/anchors_empirical_372col.parquet",
           "per_codec": True},
    "v8": {"path": TRAIN / "2026-05-19-v8-anchors/anchors_v8_372col.parquet",
           "per_codec": False, "target_gate": True},
}

GATE_CC_STD = 5.0
ACHIEVEMENT_TOL = 5.0


def score_bake(bake_path: Path, anchor_path: Path) -> np.ndarray:
    """Run ensemble_score_rows on the anchor parquet, return per-row scores."""
    if not SCORE_BIN.exists():
        raise SystemExit(
            f"missing {SCORE_BIN}\n"
            f"  build it:  cargo build --release -p zensim-validate --bin ensemble_score_rows"
        )
    with tempfile.NamedTemporaryFile(suffix=".tsv", delete=False, mode="w") as tmp:
        tmp_path = tmp.name
    result = subprocess.run(
        [str(SCORE_BIN), "--bake", str(bake_path),
         "--parquet", str(anchor_path), "--output", tmp_path],
        capture_output=True, text=True, check=False,
    )
    if result.returncode != 0:
        print(f"score_rows stderr: {result.stderr}", file=sys.stderr)
        raise RuntimeError(f"ensemble_score_rows failed: {result.returncode}")
    scores: list[float] = []
    with open(tmp_path) as f:
        f.readline()  # header
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            try:
                scores.append(float(parts[2]))
            except ValueError:
                scores.append(float("nan"))
    Path(tmp_path).unlink(missing_ok=True)
    return np.asarray(scores, dtype=np.float64)


def band_row(band, band_scores, band_sources, band_codecs, band_targets,
             per_codec, target_gate):
    """One band's stats for one bake."""
    target_min = float(np.nanmin(band_targets))
    target_med = float(np.nanmedian(band_targets))
    target_max = float(np.nanmax(band_targets))

    achievement: dict = {}
    if per_codec:
        # The empirical target varies per codec, so a single pooled
        # `achieved_mean - target` delta is misleading — break it out.
        for codec in np.unique(band_codecs):
            m = band_codecs == codec
            if m.sum() == 0:
                continue
            c_t = band_targets[m]
            achievement[str(codec)] = {
                "target": float(c_t[0]) if len(c_t) else float("nan"),
                "achieved_mean": float(band_scores[m].mean()),
                "achieved_std": float(band_scores[m].std()),
                "n": int(m.sum()),
            }

    # The gate stat: within one source, how much does the score move when only
    # the codec changes? Within a band it should not move.
    cc = [band_scores[band_sources == s].std()
          for s in np.unique(band_sources)
          if (band_sources == s).sum() >= 2]
    cc_arr = np.asarray(cc) if cc else np.array([np.nan])
    cc_median = float(np.nanmedian(cc_arr))

    achieved_mean = float(band_scores.mean())
    abs_err = abs(achieved_mean - target_med)
    # v8 gates on BOTH cc_std and target error; every other generation gates on
    # cc_std alone. Keeping this explicit is the point — see the module
    # docstring on how the copy chain moved this silently.
    passed = cc_median <= GATE_CC_STD and (not target_gate or abs_err <= ACHIEVEMENT_TOL)

    return {
        "band": band,
        "target": target_med,
        "abs_err": abs_err,
        "target_min": target_min,
        "target_max": target_max,
        "n": int(len(band_scores)),
        "achieved_mean": achieved_mean,
        "achieved_std": float(band_scores.std()),
        "cc_std_median": cc_median,
        "cc_std_mean": float(np.nanmean(cc_arr)),
        "cc_std_p95": float(np.nanpercentile(cc_arr, 95)),
        "gate": "PASS" if passed else "FAIL",
        "per_codec_achievement": achievement,
    }


def write_report(out_md: Path, exp: str, summary: list, per_codec: bool,
                 target_gate: bool) -> None:
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.write(f"# EXP-CROSS-CODEC-{exp.upper()} multi-band cross-codec consistency check\n\n")
        f.write(
            "For each anchor band, measure cross-codec score std per source\n"
            f"within that band. Gate: cc_std_median <= {GATE_CC_STD} at EVERY band"
            + (f", AND |achieved_mean - target| <= {ACHIEVEMENT_TOL:g}.\n"
               if target_gate else ".\n")
        )
        if per_codec:
            f.write(
                "Anchor targets are per-(codec, band) empirical ssim2 medians.\n"
            )
        f.write("\n")

        f.write("## Bake-level summary\n\n")
        f.write("| bake | passing_bands | total_bands | all_pass |\n")
        f.write("| --- | ---: | ---: | :-: |\n")
        for s in summary:
            f.write(f"| {s['name']} | {s['passing_bands']} | {s['total_bands']} | "
                    f"{'PASS' if s['all_pass'] else 'FAIL'} |\n")

        detail = "Per-bake per-band detail (codec-pooled)" if per_codec else "Per-bake per-band detail"
        f.write(f"\n## {detail}\n\n")
        tcol = ("target (med, min..max)" if per_codec
                else ("target" if target_gate else "target_score"))
        talign = "---" if per_codec else "---:"
        err_h, err_a = ("abs_err | ", "---: | ") if target_gate else ("", "")
        for s in summary:
            f.write(f"### {s['name']}\n\n")
            f.write(f"| band (butter) | {tcol} | n | achieved_mean | {err_h}"
                    "achieved_std | cc_std_median | cc_std_p95 | gate |\n")
            f.write(f"| ---: | {talign} | ---: | ---: | {err_a}---: | ---: | ---: | :-: |\n")
            for r in s["per_band"]:
                tgt = (f"{r['target']:.1f} ({r['target_min']:.1f}..{r['target_max']:.1f})"
                       if per_codec else f"{r['target']:.1f}")
                err = f"{r['abs_err']:.2f} | " if target_gate else ""
                f.write(f"| {r['band']:.2f} | {tgt} | {r['n']} | "
                        f"{r['achieved_mean']:.2f} | {err}{r['achieved_std']:.2f} | "
                        f"{r['cc_std_median']:.2f} | {r['cc_std_p95']:.2f} | {r['gate']} |\n")
            f.write("\n")

        if not per_codec:
            return
        f.write("## Per-bake per-(codec, band) achievement vs empirical target\n\n")
        f.write(f"Gate (advisory): per-(codec, band) `|achieved_mean - target| <= {ACHIEVEMENT_TOL:g}`.\n\n")
        for s in summary:
            f.write(f"### {s['name']}\n\n")
            f.write("| band | codec | target | achieved_mean | achieved_std | Δ | within ±5 |\n")
            f.write("| ---: | --- | ---: | ---: | ---: | ---: | :-: |\n")
            for r in s["per_band"]:
                for codec, i in sorted(r["per_codec_achievement"].items()):
                    d = i["achieved_mean"] - i["target"]
                    f.write(f"| {r['band']:.2f} | {codec} | {i['target']:.2f} | "
                            f"{i['achieved_mean']:.2f} | {i['achieved_std']:.2f} | "
                            f"{d:+.2f} | {'Y' if abs(d) <= ACHIEVEMENT_TOL else 'N'} |\n")
            f.write("\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exp", help=f"generation: one of {sorted(ANCHORS)}")
    ap.add_argument("dir", type=Path, help="experiment dir holding the bakes")
    ap.add_argument("--anchor", type=Path, default=None, help="override the anchor parquet")
    ap.add_argument("--bake-glob", default=None, help="default: cc4<exp>_*.bin")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: <dir>/<exp>_multi_band_check.md")
    ap.add_argument("--per-codec-achievement", action="store_true",
                    help="force the per-(codec, band) target table (default: per the anchor)")
    ap.add_argument("--target-gate", action="store_true",
                    help="also gate each band on |achieved_mean - target| <= 5 (v8 behaviour)")
    a = ap.parse_args()

    if a.exp not in ANCHORS and a.anchor is None:
        print(f"unknown generation {a.exp!r}; pass --anchor explicitly", file=sys.stderr)
        return 2
    spec = ANCHORS.get(a.exp, {"path": a.anchor, "per_codec": False})
    anchor = a.anchor or spec["path"]
    per_codec = a.per_codec_achievement or spec["per_codec"]
    target_gate = a.target_gate or spec.get("target_gate", False)
    out_md = a.out or (a.dir / f"{a.exp}_multi_band_check.md")

    bakes = sorted(a.dir.glob(a.bake_glob or f"cc4{a.exp}_*.bin"))
    if not bakes:
        print(f"no {a.bake_glob or f'cc4{a.exp}_*.bin'} under {a.dir}", file=sys.stderr)
        return 1
    if not Path(anchor).exists():
        print(f"missing anchor parquet: {anchor}", file=sys.stderr)
        return 1

    print(f"loading anchor parquet: {anchor}")
    df = pq.read_table(anchor).to_pandas()
    print(f"  n={len(df)}, codecs={sorted(df['codec'].unique())}, "
          f"sources={df['ref_basename'].nunique()}, bands={sorted(df['butter_target'].unique())}")

    codecs_arr = df["codec"].to_numpy()
    sources_arr = df["ref_basename"].to_numpy()
    bands_arr = df["butter_target"].to_numpy()
    targets_arr = df["target_score"].to_numpy()
    unique_bands = sorted(df["butter_target"].unique())

    summary = []
    for bake_path in bakes:
        name = bake_path.stem
        print(f"\n=== {name} ===")
        scores = score_bake(bake_path, Path(anchor))
        if len(scores) != len(df):
            print(f"  ERROR: got {len(scores)} scores, expected {len(df)}", file=sys.stderr)
            continue
        valid = np.isfinite(scores)
        if not valid.all():
            print(f"  WARN: {(~valid).sum()} non-finite scores; dropping")

        rows = []
        for band in unique_bands:
            m = (bands_arr == band) & valid
            if m.sum() == 0:
                rows.append({"band": band, "target": float("nan"),
                             "target_min": float("nan"), "target_max": float("nan"),
                             "abs_err": float("nan"),
                             "n": 0, "achieved_mean": float("nan"),
                             "achieved_std": float("nan"), "cc_std_median": float("nan"),
                             "cc_std_mean": float("nan"), "cc_std_p95": float("nan"),
                             "gate": "no-data", "per_codec_achievement": {}})
                continue
            r = band_row(band, scores[m], sources_arr[m], codecs_arr[m],
                         targets_arr[m], per_codec, target_gate)
            rows.append(r)
            errs = f"abs_err={r['abs_err']:5.2f} " if target_gate else ""
            print(f"  band butter={band:.2f} target={r['target']:5.1f} n={r['n']:5d} "
                  f"achieved={r['achieved_mean']:6.2f}±{r['achieved_std']:5.2f} "
                  f"{errs}cc_std_median={r['cc_std_median']:5.2f} "
                  f"(p95={r['cc_std_p95']:5.2f}) {r['gate']}")

        passing = sum(1 for r in rows if r["gate"] == "PASS")
        total = sum(1 for r in rows if r["gate"] != "no-data")
        summary.append({"name": name, "passing_bands": passing, "total_bands": total,
                        "all_pass": passing == total and total > 0, "per_band": rows})

    write_report(out_md, a.exp, summary, per_codec, target_gate)
    print(f"\nwrote {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
