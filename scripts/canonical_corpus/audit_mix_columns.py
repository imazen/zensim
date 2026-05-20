"""Audit canonical training parquets — count non-null mix target columns."""
import pyarrow.parquet as pq
import pyarrow.compute as pc
from pathlib import Path

CANONICAL = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18")
TRAIN = CANONICAL / "train"

# All possible target columns per CLAUDE.md schema
MIX_COLS = [
    "human_score",
    "cvvdp_score", "cvvdp_log_norm",
    "iwssim", "iwssim_log_norm",
    "ssim2_gpu", "ssim2_log_norm",
    "pjnd_target",
    "mix_cv25_iw75", "mix_cv30_iw70", "mix_cv35_iw65",
    "mix_cv40_iw60", "mix_cv45_iw55", "mix_cv50_iw50",
    "mix_cv55_iw45", "mix_cv55_iw44",
    "mix_cv60_iw40", "mix_cv65_iw35", "mix_cv70_iw30", "mix_cv75_iw25",
    "mix_cv33_iw33_sm33",
    "mix_target",
]

parquets = sorted(TRAIN.glob("*.parquet"))
print(f"\nAuditing {len(parquets)} training parquets in {TRAIN}\n")

# Build a matrix: corpus -> col -> (non_null, total)
results = {}
totals = {}
all_cols_seen = set()
for pq_path in parquets:
    corpus = pq_path.stem
    pf = pq.ParquetFile(pq_path)
    n_rows = pf.metadata.num_rows
    totals[corpus] = n_rows
    schema_names = set(pf.schema_arrow.names)
    cols_in_file = [c for c in MIX_COLS if c in schema_names]
    all_cols_seen.update(cols_in_file)
    print(f"Reading {corpus} ({n_rows:,} rows, {len(cols_in_file)}/{len(MIX_COLS)} target cols present)…")
    t = pf.read(columns=cols_in_file)
    results[corpus] = {}
    for col in cols_in_file:
        arr = t.column(col)
        nn = arr.length() - arr.null_count
        results[corpus][col] = (nn, arr.length())
    # Mark missing cols
    for col in MIX_COLS:
        if col not in schema_names:
            results[corpus][col] = (None, n_rows)  # column missing

# Build markdown table
ordered_corpora = ["safesyn", "kadid", "tid", "konjnd-dense", "cvvdp_iwssim_LARGE"]
ordered_corpora = [c for c in ordered_corpora if c in results]
# Print header
hdr = "| target_col | " + " | ".join(f"{c} ({totals[c]:,})" for c in ordered_corpora) + " |"
sep = "|" + "---|" * (len(ordered_corpora) + 1)
print()
print(hdr)
print(sep)
for col in MIX_COLS:
    cells = []
    for corp in ordered_corpora:
        v = results[corp].get(col, (None, totals[corp]))
        if v[0] is None:
            cells.append("MISSING")
        else:
            nn, total = v
            pct = 100.0 * nn / total if total else 0.0
            if nn == 0:
                cells.append(f"0 / {total:,} (ALL-NULL)")
            elif nn == total:
                cells.append(f"{nn:,} (100.0%)")
            else:
                cells.append(f"{nn:,} / {total:,} ({pct:.1f}%)")
    print(f"| {col} | " + " | ".join(cells) + " |")

# Special audit: highlight all-null and missing cols
print("\n## Anomalies\n")
for col in MIX_COLS:
    for corp in ordered_corpora:
        v = results[corp].get(col, (None, totals[corp]))
        if v[0] is None:
            print(f"- {corp}: column `{col}` is MISSING from schema")
        elif v[0] == 0:
            print(f"- {corp}: column `{col}` is ALL-NULL ({v[1]:,} rows)")
