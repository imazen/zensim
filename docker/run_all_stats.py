#!/usr/bin/env python3
"""zensim-stats orchestrator (2026-05-24).

Runs the full evaluation suite for the shipped `PreviewV0_3` bake
(Tuner v5) inside the docker image and pretty-prints the results.

Phases:
  1. Aggregate Mohammadi panel per val corpus (bake_verdict).
  2. CID22 per-band SROCC + Z-RMSE table (10-band MOS grid).
  3. JPEG q-sweep monotonicity (qsweep_eval).
  4. Per-corpus headline summary.

Optional via `--full`:
  5. Cross-codec consistency on the 68k-pair equivalence parquet
     (requires the parquet to be volume-mounted at
     /data/cross_codec_equivalence.parquet).

Output:
  - Default: ANSI-coloured terminal report to stdout.
  - `--json`: machine-readable JSON dump.
  - `--md`: markdown report (suitable for GitHub paste).
"""
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

BIN_DIR = Path(os.environ.get("BIN_DIR", "/usr/local/bin"))
BAKE_VERDICT = BIN_DIR / "bake_verdict"
QSWEEP_EVAL = BIN_DIR / "qsweep_eval"
PREDICT = BIN_DIR / "predict_features_with_bake"

BAKE = Path(os.environ.get("BAKE_PATH", "/app/weights/codec_target.bin"))
VAL_PARQUETS = Path(os.environ.get("VAL_PARQUETS", "/app/val_parquets"))
QSWEEP_FEATURES = Path(os.environ.get("QSWEEP_FEATURES", "/app/qsweep/qsweep_features.csv"))
QSWEEP_MANIFEST = Path(os.environ.get("QSWEEP_MANIFEST", "/app/qsweep/qsweep_manifest.tsv"))
BASELINE_PANELS = Path(os.environ.get("BASELINE_PANELS", "/app/baseline_panels.md"))

# ANSI helpers (skipped when not a TTY or --no-color is passed).
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    CYAN = "\033[36m"
    MAGENTA = "\033[35m"
    RED = "\033[31m"

NO_COLOR = not sys.stdout.isatty()


def c(text: str, code: str) -> str:
    if NO_COLOR:
        return text
    return f"{code}{text}{C.RESET}"


def parse_summary_row(verdict_md: str, corpus: str) -> dict | None:
    """Extract one row from the bake_verdict ## Summary table."""
    in_summary = False
    for line in verdict_md.splitlines():
        if line.startswith("## Summary"):
            in_summary = True
            continue
        if in_summary and line.startswith(f"| {corpus} |"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            # Format: Corpus | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE
            if len(parts) < 8:
                return None
            return {
                "corpus": parts[0],
                "n": int(parts[1]),
                "srocc": float(parts[2]),
                "plcc": float(parts[3]),
                "krocc": float(parts[4]),
                "or": float(parts[5]),
                "pwrc": float(parts[6]),
                "z_rmse": float(parts[7]),
            }
    return None


def parse_band_rows(verdict_md: str, corpus_section: str) -> list[dict]:
    """Extract per-band rows from a `### {corpus_section} 10-band ...` section."""
    bands = []
    in_section = False
    in_table = False
    for line in verdict_md.splitlines():
        if line.startswith(f"### {corpus_section} 10-band"):
            in_section = True
            continue
        if in_section and line.startswith("### ") and not line.startswith(f"### {corpus_section}"):
            break
        if in_section and line.startswith("| Band |"):
            in_table = True
            continue
        if in_section and in_table and line.startswith("|---"):
            continue
        if in_section and in_table and line.startswith("|"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            # | Band | range | n | SROCC | PLCC | KROCC | OR | PWRC | Z-RMSE | MAE |
            if len(parts) < 9 or parts[3].lower() == "n/a":
                continue
            try:
                bands.append({
                    "band": parts[0].lstrip("⚠ "),
                    "range": parts[1],
                    "n": int(parts[2]),
                    "srocc": float(parts[3]),
                    "plcc": float(parts[4]),
                    "krocc": float(parts[5]),
                    "or": float(parts[6]),
                    "pwrc": float(parts[7]),
                    "z_rmse": float(parts[8]),
                })
            except (ValueError, IndexError):
                continue
        if in_section and in_table and not line.startswith("|") and bands:
            in_table = False
    return bands


def parse_baseline_panels(corpus: str, baseline_md: str) -> dict[str, dict]:
    """Extract ssim2/cvvdp/iwssim aggregate stats from baseline_panels.md
    for a given corpus heading (e.g. 'CID22', 'AIC-3 CTC')."""
    out = {}
    in_corpus = False
    for line in baseline_md.splitlines():
        if line.startswith(f"## {corpus}"):
            in_corpus = True
            continue
        if in_corpus and line.startswith("## ") and not line.startswith(f"## {corpus}"):
            break
        if in_corpus and line.startswith("| ssim2") or in_corpus and line.startswith("| cvvdp") or in_corpus and line.startswith("| iwssim"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) < 7:
                continue
            name = parts[0].split(" ")[0].lower()
            try:
                out[name] = {
                    "srocc": float(parts[2]),
                    "plcc": float(parts[3]),
                    "krocc": float(parts[4]),
                    "or": float(parts[5]),
                    "pwrc": float(parts[6]),
                    "z_rmse": float(parts[7]),
                }
            except (ValueError, IndexError):
                pass
    return out


def parse_band_panels(corpus: str, baseline_md: str) -> dict[str, list[dict]]:
    """For a corpus, extract per-band rows for ssim2/cvvdp/iwssim from
    baseline_panels.md's '### {corpus} 10-band panels' section."""
    out: dict[str, list[dict]] = {}
    in_corpus_bands = False
    current_metric = None
    for line in baseline_md.splitlines():
        if line.startswith(f"### {corpus} 10-band"):
            in_corpus_bands = True
            continue
        if in_corpus_bands and line.startswith("## "):
            break
        if in_corpus_bands and line.startswith("#### "):
            metric_label = line[5:].strip().lower()
            for k in ("ssim2", "cvvdp", "iwssim"):
                if k in metric_label:
                    current_metric = k
                    out.setdefault(current_metric, [])
                    break
            else:
                current_metric = None
            continue
        if in_corpus_bands and current_metric and line.startswith("|") and not line.startswith("| Band") and not line.startswith("|---"):
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) < 8 or parts[3].lower() == "n/a":
                continue
            try:
                out[current_metric].append({
                    "band": parts[0].lstrip("⚠ "),
                    "range": parts[1],
                    "n": int(parts[2]),
                    "srocc": float(parts[3]),
                    "z_rmse": float(parts[8]) if len(parts) > 8 else None,
                })
            except (ValueError, IndexError):
                continue
    return out


def run_bake_verdict(bake: Path, features_root: Path) -> str:
    """Run bake_verdict and return the markdown output."""
    out_file = Path("/tmp/_bake_verdict.md")
    cmd = [str(BAKE_VERDICT), "--bake", str(bake),
           "--features-root", str(features_root),
           "--output", str(out_file)]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out_file.read_text()


def run_qsweep_eval(bake: Path) -> dict:
    """Run qsweep_eval and parse the monotonicity rate."""
    out_file = Path("/tmp/_qsweep.md")
    cmd = [str(QSWEEP_EVAL),
           "--features", str(QSWEEP_FEATURES),
           "--manifest", str(QSWEEP_MANIFEST),
           "--bake", f"v0.3={bake}",
           "--out", str(out_file)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    # Parse "monotonicity=0.XXXX (50/900 curves)" from stderr/stdout.
    text = proc.stdout + proc.stderr
    m = re.search(r"monotonicity=([\d.]+)\s*\((\d+)/(\d+)", text)
    if not m:
        return {"monotonicity": None}
    return {
        "monotonicity": float(m.group(1)),
        "violations": int(m.group(1).split('.')[0] if False else (1 - float(m.group(1))) * int(m.group(3))),
        "n_pairs": int(m.group(3)),
    }


def pretty_print_text(report: dict) -> None:
    print()
    print(c("zensim PreviewV0_3 — full evaluation report", C.BOLD + C.CYAN))
    print(c(f"  bake: {BAKE.name}", C.DIM))
    print()
    print(c("══════════════════════════════════════════════════════════════", C.DIM))
    print(c(" Aggregate Mohammadi panel — fair holdouts", C.BOLD))
    print(c("══════════════════════════════════════════════════════════════", C.DIM))
    print()
    print(f"{'Corpus':<28} {'n':>5}  {'SROCC':>7} {'PLCC':>7} {'KROCC':>7} {'PWRC':>7} {'Z-RMSE':>7}")
    print("-" * 80)
    for corpus, payload in report["aggregate"].items():
        v03 = payload["v03"]
        fair = "✓ fair holdout" if payload["fair_holdout"] else "△ in training"
        c_corpus = c(corpus, C.GREEN if payload["fair_holdout"] else C.YELLOW)
        n_str = f"{v03['n']:>5}" if v03 else "  n/a"
        srocc_str = f"{v03['srocc']:7.3f}" if v03 else "    n/a"
        plcc_str = f"{v03['plcc']:7.3f}" if v03 else "    n/a"
        krocc_str = f"{v03['krocc']:7.3f}" if v03 else "    n/a"
        pwrc_str = f"{v03['pwrc']:7.3f}" if v03 else "    n/a"
        zrmse_str = f"{v03['z_rmse']:7.3f}" if v03 else "    n/a"
        print(f"{c_corpus:<37}  {n_str}  {srocc_str} {plcc_str} {krocc_str} {pwrc_str} {zrmse_str}  {c(fair, C.DIM)}")
    print()
    print(c(" v0.3 vs baselines on fair holdouts (SROCC)", C.BOLD))
    print(c("──────────────────────────────────────────────────────────────", C.DIM))
    for corpus, payload in report["aggregate"].items():
        if not payload["fair_holdout"]:
            continue
        v03 = payload.get("v03")
        if not v03:
            continue
        baselines = payload.get("baselines", {})
        print()
        print(c(f"  {corpus} (n={v03['n']})", C.CYAN))
        # Sort metrics by SROCC descending.
        all_metrics = [("v0.3", v03["srocc"])] + [
            (k, baselines[k]["srocc"]) for k in baselines
            if baselines[k]
        ]
        all_metrics.sort(key=lambda kv: kv[1], reverse=True)
        for i, (name, srocc) in enumerate(all_metrics):
            tag = c("★", C.YELLOW) if i == 0 else " "
            label = c(name, C.GREEN if name == "v0.3" else C.RESET)
            print(f"    {tag} {label:<20}  SROCC = {srocc:.3f}")

    print()
    print(c("══════════════════════════════════════════════════════════════", C.DIM))
    print(c(" CID22 per-band SROCC (10-band MOS grid)", C.BOLD))
    print(c("══════════════════════════════════════════════════════════════", C.DIM))
    bands = report.get("cid22_bands", [])
    baselines_bands = report.get("cid22_baseline_bands", {})
    if bands:
        print()
        print(f"{'Band':<8} {'range':<14} {'n':>5}   {'v0.3':>7}  {'ssim2':>7}  {'cvvdp':>7}  {'iwssim':>7}")
        print("-" * 80)
        for b in bands:
            band = b["band"]
            ssim2 = next((x for x in baselines_bands.get("ssim2", []) if x["band"] == band), None)
            cvvdp = next((x for x in baselines_bands.get("cvvdp", []) if x["band"] == band), None)
            iwssim = next((x for x in baselines_bands.get("iwssim", []) if x["band"] == band), None)
            srocc_row = [b["srocc"]]
            cells = [f"{b['srocc']:7.3f}"]
            for x in (ssim2, cvvdp, iwssim):
                srocc_row.append(x["srocc"] if x else None)
                cells.append(f"{x['srocc']:7.3f}" if x else "    n/a")
            # Highlight the winner in this band.
            best_idx = max(range(len(srocc_row)), key=lambda i: srocc_row[i] if srocc_row[i] is not None else -1)
            winners = [c(cell, C.GREEN + C.BOLD) if i == best_idx else cell
                       for i, cell in enumerate(cells)]
            print(f"{band:<8} {b['range']:<14} {b['n']:>5}   {winners[0]}  {winners[1]}  {winners[2]}  {winners[3]}")

    print()
    print(c("══════════════════════════════════════════════════════════════", C.DIM))
    print(c(" JPEG q-sweep monotonicity", C.BOLD))
    print(c("══════════════════════════════════════════════════════════════", C.DIM))
    mono = report.get("monotonicity", {})
    if mono.get("monotonicity") is not None:
        mr = mono["monotonicity"]
        color = C.GREEN if mr >= 0.9278 else C.YELLOW if mr >= 0.85 else C.RED
        status = "PASS (≥ 0.928 gate)" if mr >= 0.9278 else "BELOW GATE"
        print(f"  v0.3 monotonicity: {c(f'{mr:.4f}', color + C.BOLD)} ({status})")
        print(f"  Measured on 50 imgs × 19 q-values = 900 adjacent pairs.")
    else:
        print("  monotonicity unavailable (qsweep_eval failed)")

    print()
    print(c("══════════════════════════════════════════════════════════════", C.DIM))
    print(c(" Read", C.BOLD))
    print(c("══════════════════════════════════════════════════════════════", C.DIM))
    print()
    print("  Green corpora are unambiguously fair holdouts (no training-set overlap).")
    print("  Yellow corpora were in v0.3's training mix and are NOT fair anchors.")
    print()
    print("  v0.3 is the shipping PreviewV0_3 bake (Tuner v5,")
    print(f"  {BAKE.name}, 54 KB packed). Trained on safesyn +")
    print("  cid22_train + kadid + tid + konjnd_dense.")
    print()
    print("  For machine-readable output: docker run --rm zensim-stats --json")
    print()


def pretty_print_markdown(report: dict) -> None:
    print("# zensim PreviewV0_3 — full evaluation report")
    print()
    print(f"- Bake: `{BAKE.name}`")
    print()
    print("## Aggregate Mohammadi panel — fair holdouts")
    print()
    print("| Corpus | Fair? | n | SROCC | PLCC | KROCC | PWRC | Z-RMSE |")
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    for corpus, payload in report["aggregate"].items():
        v03 = payload["v03"]
        if not v03:
            continue
        fair = "✓" if payload["fair_holdout"] else "△ train"
        print(f"| {corpus} | {fair} | {v03['n']} | {v03['srocc']:.3f} | {v03['plcc']:.3f} | "
              f"{v03['krocc']:.3f} | {v03['pwrc']:.3f} | {v03['z_rmse']:.3f} |")
    print()
    print("## CID22 per-band SROCC")
    print()
    print("| Band | range | n | v0.3 | ssim2 | cvvdp | iwssim |")
    print("|---|---|--:|--:|--:|--:|--:|")
    bands = report.get("cid22_bands", [])
    baselines_bands = report.get("cid22_baseline_bands", {})
    for b in bands:
        ssim2 = next((x for x in baselines_bands.get("ssim2", []) if x["band"] == b["band"]), None)
        cvvdp = next((x for x in baselines_bands.get("cvvdp", []) if x["band"] == b["band"]), None)
        iwssim = next((x for x in baselines_bands.get("iwssim", []) if x["band"] == b["band"]), None)
        s_str = f"{ssim2['srocc']:.3f}" if ssim2 else "n/a"
        c_str = f"{cvvdp['srocc']:.3f}" if cvvdp else "n/a"
        i_str = f"{iwssim['srocc']:.3f}" if iwssim else "n/a"
        print(f"| {b['band']} | {b['range']} | {b['n']} | {b['srocc']:.3f} | {s_str} | {c_str} | {i_str} |")
    print()
    print("## JPEG q-sweep monotonicity")
    mono = report.get("monotonicity", {})
    if mono.get("monotonicity") is not None:
        print(f"- v0.3 monotonicity: {mono['monotonicity']:.4f} on 50 imgs × 19 q (900 adjacent pairs)")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--md", action="store_true", help="emit markdown report")
    ap.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    args = ap.parse_args()

    global NO_COLOR
    if args.no_color or args.json or args.md:
        NO_COLOR = True

    if not BAKE.exists():
        print(f"error: bake not found at {BAKE}", file=sys.stderr)
        sys.exit(2)
    if not VAL_PARQUETS.is_dir():
        print(f"error: val parquets dir not found at {VAL_PARQUETS}", file=sys.stderr)
        sys.exit(2)

    # Phase 1: aggregate panel.
    if not args.json:
        print(c("[1/3] Running bake_verdict (Mohammadi panel + per-band) ...", C.DIM), file=sys.stderr)
    verdict_md = run_bake_verdict(BAKE, VAL_PARQUETS)
    baseline_md = BASELINE_PANELS.read_text() if BASELINE_PANELS.exists() else ""

    # Fair-holdout taxonomy (per AIC-3 verification 2026-05-24).
    fair_holdouts = {"CID22", "AIC-3 CTC", "AIC-4 sample"}
    in_training = {"KADIK10k", "TID2013", "KonJND-1k (full)"}

    aggregate = {}
    for corpus in ["CID22", "KADIK10k", "TID2013", "KonJND-1k (full)", "AIC-3 CTC", "AIC-4 sample"]:
        v03 = parse_summary_row(verdict_md, corpus)
        # Map corpus -> baseline-panels heading. AIC-4 has no
        # baseline-panel data (the AIC-3 PTC subset is a DIFFERENT
        # dataset, not AIC-4 baselines). Map to None → skip lookup.
        baseline_corpus_map = {
            "CID22": "CID22 (n=4292)",
            "KADIK10k": "KADID-10k (n=10125)",
            "TID2013": "TID2013 (n=3000)",
            "KonJND-1k (full)": "KonJND-1k (n=1008)",
            "AIC-3 CTC": "AIC-3 CTC per-pair sweep (n=600)",
            "AIC-4 sample": None,  # no AIC-4 baseline panel data
        }
        heading = baseline_corpus_map[corpus]
        baselines = parse_baseline_panels(heading, baseline_md) if (baseline_md and heading) else {}
        aggregate[corpus] = {
            "v03": v03,
            "baselines": baselines,
            "fair_holdout": corpus in fair_holdouts,
            "in_training": corpus in in_training,
        }

    cid22_bands = parse_band_rows(verdict_md, "CID22")
    cid22_baseline_bands = parse_band_panels("CID22", baseline_md) if baseline_md else {}

    # Phase 2: monotonicity.
    if not args.json:
        print(c("[2/3] Running qsweep_eval (JPEG q-sweep monotonicity) ...", C.DIM), file=sys.stderr)
    try:
        mono = run_qsweep_eval(BAKE)
    except subprocess.CalledProcessError as e:
        mono = {"monotonicity": None, "error": e.stderr}

    if not args.json:
        print(c("[3/3] Formatting report ...", C.DIM), file=sys.stderr)

    report = {
        "bake": BAKE.name,
        "aggregate": aggregate,
        "cid22_bands": cid22_bands,
        "cid22_baseline_bands": cid22_baseline_bands,
        "monotonicity": mono,
    }

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    elif args.md:
        pretty_print_markdown(report)
    else:
        pretty_print_text(report)


if __name__ == "__main__":
    main()
