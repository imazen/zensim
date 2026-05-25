#!/usr/bin/env bash
# Inject a spline sidecar payload into an existing bake via zenpredict.
#
# Usage: bash scripts/inject_spline.sh <bake.bin> <spline.bin> <output.bin>
#
# The spline payload is the raw bytes from ZENSIM_SPLINE_SIDECAR.
# This script hex-encodes it and adds it as a zentrain.output_calibration_spline
# metadata entry via the zenpredict JSON roundtrip.

set -euo pipefail

BAKE=${1:?usage: inject_spline.sh <bake.bin> <spline.bin> <output.bin>}
SPLINE=${2:?}
OUTPUT=${3:?}

ZENPREDICT="${ZENPREDICT:-/home/lilith/work/zen/zenanalyze/target/release/zenpredict}"

if [ ! -f "$ZENPREDICT" ]; then
    echo "zenpredict CLI not found at $ZENPREDICT"
    exit 1
fi

# Hex-encode the spline payload
SPLINE_HEX=$(xxd -p "$SPLINE" | tr -d '\n')
echo "Spline payload: $(wc -c < "$SPLINE") bytes → ${#SPLINE_HEX} hex chars"

# Inspect the bake to get JSON
JSON=$("$ZENPREDICT" inspect --json "$BAKE" 2>/dev/null || true)
if [ -z "$JSON" ]; then
    echo "Failed to inspect bake — trying direct JSON dump"
    # Fallback: just dump raw and add metadata
    "$ZENPREDICT" inspect "$BAKE"
    exit 1
fi

# Add the spline metadata entry to the JSON
# The metadata array needs a new entry:
# {"key": "zentrain.output_calibration_spline", "type": "numeric", "hex": "<SPLINE_HEX>"}
MODIFIED=$(echo "$JSON" | python3 -c "
import sys, json
j = json.load(sys.stdin)
if 'metadata' not in j:
    j['metadata'] = []
# Remove existing spline entry if present
j['metadata'] = [m for m in j['metadata'] if m.get('key') != 'zentrain.output_calibration_spline']
# Add new spline entry
j['metadata'].append({
    'key': 'zentrain.output_calibration_spline',
    'type': 'numeric',
    'hex': '$SPLINE_HEX'
})
json.dump(j, sys.stdout)
")

# Write modified JSON to temp file
TMPJSON=$(mktemp /tmp/inject_spline_XXXXXX.json)
echo "$MODIFIED" > "$TMPJSON"

# Bake the modified JSON
"$ZENPREDICT" bake "$TMPJSON" "$OUTPUT"
rm -f "$TMPJSON"

echo "Wrote spline-injected bake to $OUTPUT"
ls -la "$OUTPUT"
