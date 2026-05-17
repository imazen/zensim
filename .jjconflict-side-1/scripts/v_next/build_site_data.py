#!/usr/bin/env python3
"""Build JSON data files for the zensim GitHub Pages site (Goal 6).

Reads zensim-bench eval log files (produced by
`dataset_metric_baseline`), extracts per-band SROCC + per-bake
aggregate numbers, and writes them as one JSON per bake under
`site/data/bakes/`. Also writes a `site/data/index.json` listing
all bakes + their metadata (md5, train CSV, hyperparams) so the
HTML can populate dropdowns / chart inputs.

The site itself (in `site/index.html` + `site/js/*.js`) reads these
JSON files and renders the Plotly.js charts.

Usage:
    python3 build_site_data.py --eval-log <path>.log --bake <path>.bin \\
                               --label V0_X --train-csv <path>.csv \\
                               --out-dir site/data

Multi-bake mode: pass `--manifest manifest.tsv` where each row is
`label\\teval_log\\tbake_path\\ttrain_csv`.
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

# Regex patterns matching dataset_metric_baseline's output format.
# Example: "| CID22 | 4292 | 0.8676 | 0.8839 | 0.8895 | 0.7412 |"
AGGREGATE_LINE = re.compile(
    r"^\|\s*(?P<dataset>\w[\w-]*)\s*\|\s*(?P<n>\d+)\s*\|\s*(?P<v02>[\d.]+|n/a)\s*\|\s*(?P<v04>[\d.]+|n/a)\s*\|\s*(?P<ssim2>[\d.]+|n/a)\s*\|\s*(?P<butter>[\d.]+|n/a)\s*\|"
)

# Per-band line:
# "| B0 below medium (<50) | 324 | 0.4072 | 0.4088 | [0.32, 0.49] | 0.4418 | 0.2041 | 59.00 | 52.65 |"
BAND_LINE = re.compile(
    r"^\|\s*(?P<band>B\d[^|]*?)\s*\|\s*(?P<n>\d+)\s*\|\s*(?P<v02>[\d.-]+)\s*\|\s*(?P<v04>[\d.-]+)\s*\|\s*\[(?P<cilo>[\d.-]+),\s*(?P<cihi>[\d.-]+)\]\s*\|\s*(?P<ssim2>[\d.-]+)\s*\|\s*(?P<butter>[\d.-]+)\s*\|\s*(?P<mae>[\d.-]+)\s*\|"
)

# Track which dataset block we're in
DATASET_HEADER = re.compile(r"^===\s*(?P<dataset>[A-Za-z0-9_-]+)\s*\(n=(?P<n>\d+)\)\s*===")


def parse_eval_log(log_path: Path) -> dict:
    """Parse a dataset_metric_baseline output log into a dict shape:
    {
      'aggregate': {dataset: {n, v02, v04, ssim2, butter}},
      'per_band': {dataset: [{band, n, v02, v04, ci, ssim2, butter, mae}, ...]},
    }
    The `v04` column carries whichever bake was passed to --v04-bake.
    """
    out = {"aggregate": {}, "per_band": {}}
    current_dataset = None
    in_per_band_section = False
    text = log_path.read_text()
    for line in text.splitlines():
        # Dataset header (Section start) — also reset per-band state
        dh = DATASET_HEADER.match(line.strip())
        if dh:
            current_dataset = dh.group("dataset")
            in_per_band_section = False
            continue
        # Per-band section starts with "### <Dataset> per-band SROCC"
        if line.startswith("### ") and "per-band" in line:
            in_per_band_section = True
            continue
        # Aggregate row
        agg = AGGREGATE_LINE.match(line.strip())
        if agg and not in_per_band_section:
            dataset = agg.group("dataset")
            if dataset in ("Dataset",):  # header
                continue
            try:
                out["aggregate"][dataset] = {
                    "n": int(agg.group("n")),
                    "v02": float(agg.group("v02")) if agg.group("v02") != "n/a" else None,
                    "v04": float(agg.group("v04")) if agg.group("v04") != "n/a" else None,
                    "ssim2": float(agg.group("ssim2")) if agg.group("ssim2") != "n/a" else None,
                    "butter": float(agg.group("butter")) if agg.group("butter") != "n/a" else None,
                }
            except ValueError:
                pass
            continue
        # Per-band row
        if in_per_band_section:
            bd = BAND_LINE.match(line.strip())
            if bd:
                ds = current_dataset or "UNKNOWN"
                out["per_band"].setdefault(ds, []).append({
                    "band": bd.group("band").strip(),
                    "n": int(bd.group("n")),
                    "v02": float(bd.group("v02")),
                    "v04": float(bd.group("v04")),
                    "ci_lo": float(bd.group("cilo")),
                    "ci_hi": float(bd.group("cihi")),
                    "ssim2": float(bd.group("ssim2")),
                    "butter": float(bd.group("butter")),
                    "mae": float(bd.group("mae")),
                })
    return out


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def build_bake_json(label: str, eval_log: Path, bake_path: Path | None,
                    train_csv: Path | None, notes: str | None) -> dict:
    parsed = parse_eval_log(eval_log)
    return {
        "label": label,
        "eval_log_path": str(eval_log),
        "bake": {
            "path": str(bake_path) if bake_path else None,
            "md5": md5_of(bake_path) if bake_path and bake_path.exists() else None,
            "size_bytes": bake_path.stat().st_size if bake_path and bake_path.exists() else None,
        },
        "train_csv": str(train_csv) if train_csv else None,
        "notes": notes,
        "aggregate": parsed["aggregate"],
        "per_band": parsed["per_band"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-log", type=Path)
    ap.add_argument("--bake", type=Path)
    ap.add_argument("--label", type=str)
    ap.add_argument("--train-csv", type=Path)
    ap.add_argument("--notes", type=str, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--manifest", type=Path,
                    help="TSV with columns: label, eval_log, bake, train_csv, notes")
    args = ap.parse_args()

    (args.out_dir / "bakes").mkdir(parents=True, exist_ok=True)

    bakes: list[dict] = []
    if args.manifest:
        for line in args.manifest.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            label = parts[0]
            eval_log = Path(parts[1]) if len(parts) > 1 and parts[1] else None
            bake = Path(parts[2]) if len(parts) > 2 and parts[2] else None
            train_csv = Path(parts[3]) if len(parts) > 3 and parts[3] else None
            notes = parts[4] if len(parts) > 4 else None
            if eval_log is None or not eval_log.exists():
                print(f"WARN: missing eval log for {label}", file=sys.stderr)
                continue
            b = build_bake_json(label, eval_log, bake, train_csv, notes)
            bakes.append(b)
    elif args.eval_log and args.label:
        b = build_bake_json(args.label, args.eval_log, args.bake, args.train_csv, args.notes)
        bakes.append(b)
    else:
        ap.error("Provide --manifest OR (--eval-log + --label)")

    # Write per-bake JSON + index.
    for bake in bakes:
        out = args.out_dir / "bakes" / f"{bake['label']}.json"
        out.write_text(json.dumps(bake, indent=2))
        print(f"wrote {out}", file=sys.stderr)

    index = {
        "bakes": [
            {
                "label": b["label"],
                "notes": b.get("notes"),
                "json": f"bakes/{b['label']}.json",
                "datasets": sorted(b["aggregate"].keys()),
            }
            for b in bakes
        ],
    }
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {args.out_dir}/index.json with {len(bakes)} bakes", file=sys.stderr)


if __name__ == "__main__":
    main()
