#!/usr/bin/env bash
# Mirror /mnt/v/zen/zensim-training/2026-05-07/unified/ to R2 for reproducibility.
#
# Usage: bash scripts/v_next/sync_unified_to_r2.sh [--dry-run]
#
# Requires R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ACCOUNT_ID env vars
# (already set in ~/.config/cloudflare/r2-env.sh, sourced by ~/.bashrc).
set -euo pipefail

LOCAL_DIR="/mnt/v/zen/zensim-training/2026-05-07/unified"
R2_PREFIX="s3://zentrain/v_next-training/2026-05-07/unified"
DRY=""
[ "${1:-}" = "--dry-run" ] && DRY="--dryrun"

if [ -z "${R2_ACCESS_KEY_ID:-}" ]; then
  echo "ERROR: R2_ACCESS_KEY_ID not set; source ~/.config/cloudflare/r2-env.sh first." >&2
  exit 1
fi

export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
ENDPOINT="https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com"

# Build a small manifest documenting the snapshot.
MANIFEST="$LOCAL_DIR/_MANIFEST.json"
GIT_COMMIT=$(cd "$(dirname "$0")/../.." && git rev-parse HEAD)
GIT_BRANCH=$(cd "$(dirname "$0")/../.." && git rev-parse --abbrev-ref HEAD)
TS=$(date -u +%Y-%m-%dT%H:%M:%SZ)

python3 - <<EOF > "$MANIFEST"
import json, glob, os
out = {
    "generated_at": "$TS",
    "git_commit": "$GIT_COMMIT",
    "git_branch": "$GIT_BRANCH",
    "parquets": [],
}
for p in sorted(glob.glob("$LOCAL_DIR/unified_*.parquet")):
    name = os.path.basename(p)
    sz = os.path.getsize(p)
    try:
        import pyarrow.parquet as pq
        rows = pq.ParquetFile(p).metadata.num_rows
    except Exception as e:
        rows = -1
    out["parquets"].append({"name": name, "rows": rows, "bytes": sz})
print(json.dumps(out, indent=2))
EOF

echo "manifest:"
cat "$MANIFEST"

aws s3 sync $DRY \
  --endpoint-url="$ENDPOINT" \
  --exclude '*' \
  --include 'unified_*.parquet' \
  --include 'adversarial_pairs_*.parquet' \
  --include 'monotonicity_violations.tsv' \
  --include 'bumpiness_per_curve.tsv' \
  --include 'score_quality_summary_*.log' \
  --include '_MANIFEST.json' \
  "$LOCAL_DIR/" "$R2_PREFIX/"

echo "done — R2 prefix: $R2_PREFIX"
