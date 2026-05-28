#!/usr/bin/env bash
# Cross-metric scoring of the corruption corpus (codec-corpus#7).
#
# Output: /tmp/corruption_multimetric_2026-05-28.tsv with columns:
#   name, family, region, severity, anchor (corruption/q20/q10),
#   ssim2_gpu, butter_max_gpu, butter_pnorm3_gpu, cvvdp, dssim_gpu
#
# Total: 672 corruptions × 3 variants × 5 metric outputs = a panel of
# 10,080 metric scores across the full corpus. Runs through
# zen-metrics compare for batched (cached reference) scoring.

set -euo pipefail

CORPUS_DIR="${CORPUS_DIR:-/mnt/v/output/zensim/corruption_gate}"
REF="${CORPUS_DIR}/gb82_dog__reference.png"
ZEN_METRICS="${ZEN_METRICS:-/home/lilith/work/zen/zenmetrics/target/release/zen-metrics}"
OUT="${OUT:-/tmp/corruption_multimetric_2026-05-28.tsv}"

if [[ ! -f "$REF" ]]; then
    echo "Reference image not found at $REF" >&2
    exit 1
fi
if [[ ! -x "$ZEN_METRICS" ]]; then
    echo "zen-metrics binary not found at $ZEN_METRICS" >&2
    exit 1
fi

# Collect every variant (corruption + q20 + q10) for the reference.
mapfile -t VARIANTS < <(ls "$CORPUS_DIR" | grep -E '__(corruption|q20|q10)\.png$' | sort)
echo "Collected ${#VARIANTS[@]} variant images for reference $(basename $REF)" >&2

# Build a CSV mapping path → (name, family, region, sev, kind) for parsing.
# Filename format: <ref>__<family>__<region>__<severity>__<kind>.png
declare -A KIND NAMES FAMILIES REGIONS SEVS
for v in "${VARIANTS[@]}"; do
    base="${v%.png}"
    kind="${base##*__}"
    rest="${base%__*}"
    sev="${rest##*__}"
    rest="${rest%__*}"
    region="${rest##*__}"
    rest="${rest%__*}"
    family="${rest##*_}"
    # rest now ends in the family. Compose the original name (without __kind suffix).
    name="${base%__*}"   # strip trailing kind, leave ref__family__region__sev
    KIND[$v]="$kind"
    NAMES[$v]="$name"
    FAMILIES[$v]="$family"
    REGIONS[$v]="$region"
    SEVS[$v]="$sev"
done

echo "metric_run starts: $(date -Iseconds)" >&2

# Batch all variants in one zen-metrics compare call. zen-metrics caches the
# reference decode, so per-pair cost is just metric compute.
TMP_OUT="$(mktemp -t zen-metrics-out.XXXX)"
trap 'rm -f "$TMP_OUT"' EXIT

# Build the cmdline: --variant /path × N
ARGS=(--reference "$REF" --metric ssim2-gpu --metric butteraugli-gpu --metric cvvdp --metric dssim-gpu)
for v in "${VARIANTS[@]}"; do
    ARGS+=(--variant "$CORPUS_DIR/$v")
done

echo "Running zen-metrics compare with ${#VARIANTS[@]} variants and 4 metrics..." >&2
"$ZEN_METRICS" compare "${ARGS[@]}" 2>/dev/null > "$TMP_OUT"

echo "metric_run finishes: $(date -Iseconds)" >&2
echo "Raw output: $TMP_OUT ($(wc -l < $TMP_OUT) lines)" >&2

# Parse the human-readable output into a TSV.
# zen-metrics compare prints groups separated by blank lines:
#   ref.png vs variant.png:
#     metric_name      score
#     ...
# Convert into one row per (variant, metric).
python3 << PYEOF > "$OUT"
import sys, os, re, pathlib

raw_path = "$TMP_OUT"
corpus = pathlib.Path("$CORPUS_DIR")
ref = "$REF"

# Reconstruct the metadata map from filenames (must mirror the bash parsing above).
def parse_meta(fname):
    base = fname[:-4]  # strip .png
    parts = base.split("__")
    # ref__family__region__sev__kind  (kind = corruption|q20|q10)
    if len(parts) < 5:
        return None
    ref_prefix, family, region, sev, kind = parts[0], parts[1], parts[2], parts[3], parts[4]
    name = "__".join(parts[:4])  # ref__family__region__sev (without kind)
    # Re-merge any family that contained underscores. Format examples:
    #   gb82_dog__channel_zero_r__whole__op100__corruption
    # The base has 6+ parts in those cases; family = parts[1..-3], region/sev/kind from end.
    if len(parts) > 5:
        kind = parts[-1]
        sev = parts[-2]
        region = parts[-3]
        family = "_".join(parts[1:-3])
        name = "__".join(parts[:-1])
    return name, family, region, sev, kind

# Read raw output
text = open(raw_path).read()
# Groups separated by blank line
groups = re.split(r"\n\s*\n", text.strip())

print("name\tfamily\tregion\tsev\tkind\tssim2_gpu\tbutteraugli_max_gpu\tbutteraugli_pnorm3_gpu\tcvvdp\tdssim_gpu")

rows_written = 0
for g in groups:
    lines = [l.strip() for l in g.splitlines() if l.strip()]
    if not lines:
        continue
    # First line: "/path/ref.png vs /path/variant.png:"
    m = re.match(r"^(.+?) vs (.+?):$", lines[0])
    if not m:
        continue
    variant_path = m.group(2)
    variant_fname = os.path.basename(variant_path)
    meta = parse_meta(variant_fname)
    if not meta:
        continue
    name, family, region, sev, kind = meta
    scores = {}
    for line in lines[1:]:
        # Possible formats:
        #   "metric_name  value" (when score succeeds)
        #   "metric_name  ERROR: ..."
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        k, vraw = parts[0], parts[1].strip()
        if vraw.startswith("ERROR"):
            scores[k] = None
        else:
            try:
                scores[k] = float(vraw)
            except ValueError:
                scores[k] = None
    cols = [
        name, family, region, sev, kind,
        f"{scores.get('ssim2_gpu', float('nan')):.4f}" if scores.get('ssim2_gpu') is not None else "nan",
        f"{scores.get('butteraugli_max_gpu', float('nan')):.4f}" if scores.get('butteraugli_max_gpu') is not None else "nan",
        f"{scores.get('butteraugli_pnorm3_gpu', float('nan')):.4f}" if scores.get('butteraugli_pnorm3_gpu') is not None else "nan",
        f"{scores.get('cvvdp_imazen_v0_0_1', float('nan')):.4f}" if scores.get('cvvdp_imazen_v0_0_1') is not None else "nan",
        f"{scores.get('dssim_gpu', float('nan')):.4f}" if scores.get('dssim_gpu') is not None else "nan",
    ]
    print("\t".join(cols))
    rows_written += 1

print(f"# wrote {rows_written} rows", file=sys.stderr)
PYEOF

echo "Wrote $OUT ($(wc -l < $OUT) lines)" >&2
