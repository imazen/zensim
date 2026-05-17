#!/bin/bash
# Runtime entrypoint for zensim-repro container.
#
# Roles (selectable via $1 or $ZENSIM_REPRO_MODE):
#
#   validate  (default) — Run 10-band SROCC + KonJND validation
#                         against corpora mounted at $CORPORA_BASE,
#                         write report + bakes to $OUTPUT_BASE.
#   bake-only           — Skip validation; just copy the trained
#                         bakes to $OUTPUT_BASE. Useful when the
#                         caller will validate elsewhere.
#   download            — Run the corpus downloader script. The
#                         caller must mount a writable volume at
#                         $CORPORA_BASE (default /data).
#   shell               — Drop into bash for manual exploration.
#
# Volume mounts (see docker/README.md for the canonical invocation):
#
#   -v /your/data:/data       — test corpora (CID22, KADID, TID, ...)
#   -v /your/synth:/synth     — synthetic feature CSVs (optional;
#                                already in image if Stage 3 downloaded)
#   -v /your/output:/output   — write bake + report here
#   -v /your/cache:/cache     — (optional) train cache; survives
#                                rebuilds, persists across ARG tweaks

set -euo pipefail

CORPORA_BASE=${CORPORA_BASE:-/data}
SYNTH_BASE=${SYNTH_BASE:-/synth}
OUTPUT_BASE=${OUTPUT_BASE:-/output}
WORK_BASE=${WORK_BASE:-/work}

MODE="${1:-${ZENSIM_REPRO_MODE:-validate}}"

ensure_writable() {
  local dir="$1"
  mkdir -p "$dir" 2>/dev/null || true
  if ! [ -w "$dir" ]; then
    echo "ERROR: $dir is not writable. Did you forget '-v /host/path:$dir'?"
    exit 2
  fi
}

case "$MODE" in
  download)
    ensure_writable "$CORPORA_BASE"
    exec /usr/local/bin/download_corpora.sh
    ;;

  validate)
    ensure_writable "$OUTPUT_BASE"
    if ! [ -d "$CORPORA_BASE/cid22" ]; then
      echo "WARN: no $CORPORA_BASE/cid22/ — validation will skip CID22 corpus."
    fi
    echo "[validate] Running 10-band evaluation against mounted corpora..."
    "$WORK_BASE/target/release/examples/dataset_metric_baseline" \
      ${CORPORA_BASE:+--cid22 "$CORPORA_BASE/cid22"} \
      ${CORPORA_BASE:+--kadid "$CORPORA_BASE/kadid10k"} \
      ${CORPORA_BASE:+--tid "$CORPORA_BASE/tid2013"} \
      --v04-bake "$WORK_BASE/benchmarks/v0_X_calibrated.bin" \
      --max-pairs "${MAX_PAIRS:-50000}" \
      > "$OUTPUT_BASE/v0_X_10band.md" || {
        echo "WARN: validation exited non-zero — partial output saved."
      }
    cp "$WORK_BASE/benchmarks/v0_X_calibrated.bin" "$OUTPUT_BASE/" 2>/dev/null || true
    cp "$WORK_BASE/benchmarks/v0_X_concat_3way.bin" "$OUTPUT_BASE/" 2>/dev/null || true
    cp "$WORK_BASE/benchmarks/v0_X_base_seed"*.bin   "$OUTPUT_BASE/" 2>/dev/null || true
    cp "$WORK_BASE/benchmarks/v0_X_cycle14_s"*.bin   "$OUTPUT_BASE/" 2>/dev/null || true
    ( cd "$OUTPUT_BASE" && md5sum *.bin *.md > CHECKSUMS 2>/dev/null || true )
    echo "[validate] Done. Output at $OUTPUT_BASE/:"
    ls -la "$OUTPUT_BASE/"
    ;;

  bake-only)
    ensure_writable "$OUTPUT_BASE"
    cp "$WORK_BASE/benchmarks/v0_X_calibrated.bin" "$OUTPUT_BASE/" 2>/dev/null || true
    cp "$WORK_BASE/benchmarks/v0_X_concat_3way.bin" "$OUTPUT_BASE/" 2>/dev/null || true
    cp "$WORK_BASE/benchmarks/v0_X_base_seed"*.bin   "$OUTPUT_BASE/" 2>/dev/null || true
    cp "$WORK_BASE/benchmarks/v0_X_cycle14_s"*.bin   "$OUTPUT_BASE/" 2>/dev/null || true
    ( cd "$OUTPUT_BASE" && md5sum *.bin > CHECKSUMS 2>/dev/null || true )
    echo "[bake-only] Done. Bakes at $OUTPUT_BASE/:"
    ls -la "$OUTPUT_BASE/"
    ;;

  shell|bash)
    exec /bin/bash
    ;;

  *)
    echo "Unknown mode: $MODE"
    echo "Valid modes: validate | bake-only | download | shell"
    exit 1
    ;;
esac
