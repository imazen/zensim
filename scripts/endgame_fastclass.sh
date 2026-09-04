#!/bin/bash
# endgame_fastclass.sh — fastclass distillation wave (2026-09-04) endgame.
# Runs IN THE DRIVER on chain completion (playbook step 5 `--then`).
# Idempotent, judgment-free, bounded: it assembles tables and never commits.
set -euo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO"
O=/mnt/v/output/zensim/fastclass-2026-09-04
W="${FCD_HB_DIR:-$HOME/tmp/fastclass}"
mkdir -p "$W"
FC=${ZL_FC:-/mnt/v/zen/cargo-targets/waver4/release/freeze_check}
D=benchmarks/fastclass_distill_wave_2026-09-04

# 1. per-arm exam table (reads fullevals; computes no statistic)
python3 "$D/exam_table.py" --dir "$O" > "$W/exam_table.txt" 2>&1 || true
# 2. the exam's own paired-bootstrap instrument, all six axes + the HF band
bash "$D/boot_fastclass_arms.sh" > "$W/boot.log" 2>&1 || true
# 3. the registered selection rule, run by its owner
"$FC" --select "$O"/*.fulleval.json --tsv > "$W/select.tsv" 2>&1 || true
# 4. one draft for the reviewer
{
  echo "## FASTCLASS WAVE RESULTS (DRAFT $(date -u +%FT%TZ) — review before folding in)"
  echo; echo '### exam table (per-seed + per-arm, read from fullevals)'; echo '```'
  cat "$W/exam_table.txt"; echo '```'
  echo; echo '### freeze_check --select'; echo '```'
  cat "$W/select.tsv" 2>/dev/null; echo '```'
  echo; echo '### paired bootstrap (W1/W2), tail'; echo '```'
  grep -E '^(#####|candidate|[A-Z][0-9A-Za-z_.]*\s)' "$O/paired_boot_fastclass.txt" 2>/dev/null | tail -200
  echo '```'
} > "$W/doc_append.draft.md"
echo "ENDGAME COMPLETE $(date -u +%FT%TZ) -> $W/doc_append.draft.md"
