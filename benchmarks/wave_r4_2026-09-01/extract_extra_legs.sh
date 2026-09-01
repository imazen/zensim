#!/bin/bash
# wave-r4: the legs the canonical 11-leg driver does not cover.
#   - tbig200k : the bigcodec 200k stride view the flagship recipe consumes,
#                localized to decoded PNGs by the W-LIN 7b lane
#   - konjnd_bpg_{train,val} : the wave-7 KonJND BPG legs (the kon lever)
#   - imazen26 / nonphoto / hfnlproxy : the three eval slices whose 944 rows
#                come from bigcodec test views, localized in the r1b root
# Same extractor, same mode, same regime as extract_944_canonical.sh.
# Env: ZM944_BIN, ZM944_OUT, ZM944_MODE (default foldapp2pools), ZM944_LEGS
set -u
BIN="${ZM944_BIN:?ZM944_BIN required}"
OUT="${ZM944_OUT:?ZM944_OUT required}"
MODE="${ZM944_MODE:-foldapp2pools}"
LEGS="${ZM944_LEGS:-}"
[ -x "$BIN" ] || { echo "ABORT: extractor missing: $BIN"; exit 1; }
mkdir -p "$OUT"
ts() { date -u +%H:%M:%SZ; }
echo "== extract_extra_legs MODE=$MODE OUT=$OUT LEGS=${LEGS:-<all>}"

run_leg() {
  local name="$1" pairs="$2"
  if [ -n "$LEGS" ]; then
    case " $LEGS " in *" $name "*) ;; *) echo "== $name SKIPPED"; return 0;; esac
  fi
  [ -f "$pairs" ] || { echo "ABORT: $name pairs TSV missing: $pairs"; exit 1; }
  echo "== $name start $(ts)"
  local t0=$SECONDS
  ZENSIM_AB_MODE="$MODE" "$BIN" "$pairs" "$OUT/$name.csv"
  local rc=$?
  local rows=-1 cols=-1
  if [ -f "$OUT/$name.csv" ]; then
    rows=$(( $(wc -l < "$OUT/$name.csv") - 1 ))
    cols=$(head -1 "$OUT/$name.csv" | awk -F, '{print NF}')
  fi
  local want=$(( $(wc -l < "$pairs") - 1 ))
  echo "== $name done rc=$rc rows=$rows/$want cols=$cols $((SECONDS-t0))s $(ts)"
  if [ "$rc" -ne 0 ] || [ "$rows" -ne "$want" ] || [ "$cols" -ne 946 ]; then
    echo "ABORT: $name failed"; exit 1
  fi
}

R1B=/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/pairs
run_leg ext_konjnd_bpg_train /mnt/v/output/zensim/wave7/konjnd_bpg_train_pairs.tsv
run_leg ext_konjnd_bpg_val   /mnt/v/output/zensim/wave7/konjnd_bpg_val_pairs.tsv
run_leg ext_imazen26         "$R1B/pairs_imazen26_png.tsv"
run_leg ext_nonphoto         "$R1B/pairs_nonphoto_png.tsv"
run_leg ext_hfnlproxy        "$R1B/pairs_hfnlproxy_png.tsv"
run_leg ext_tbig200k         /mnt/v/zen/zensim-training/wlin7-pools944-2026-08-30/pairs/pairs_tbig_png.tsv
echo "EXTRA-LEGS-DONE $(ts)"
