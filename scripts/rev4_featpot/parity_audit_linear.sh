#!/usr/bin/env bash
# Bit-parity audit of the deterministic BVLS and lasso probes: pre-rule vs clean-main
# binaries. Fixed cells (registered before running): konfig_train r0 for both probes.
# The original dirs are copied first; if the rerun differs, the originals are restored
# and the new result kept under parity/det/ so eras are never mixed silently.
# Run under ~/tmp/devin/heavy. POTENTIAL - ceiling, not a model score.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CARGO_TARGET_DIR=/var/tmp/rev4-featpot/target CARGO_HOME=/var/tmp/rev4-featpot/cargo_home
export ZEN_PANEL_BIN=/var/tmp/rev4-featpot/target/debug/panel OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
root=/var/tmp/rev4-featpot; par=$root/parity/det; mkdir -p "$par"
ok=1
for probe in bvls linear; do
    dir=$root/fits/POT_konfig_train_r0_$probe
    cp -a "$dir" "$par/orig_$probe"
    old=$(sha256sum "$dir/result.json" | cut -d' ' -f1)
    python scripts/rev4_featpot/${probe}_probe.py --set konfig_train --arm r0 > "$par/rerun_$probe.log" 2>&1
    new=$(sha256sum "$dir/result.json" | cut -d' ' -f1)
    if [[ "$old" == "$new" ]]; then
        printf 'PARITY %s identical result.json sha256=%s\n' "$probe" "$new"
    else
        ok=0
        printf 'PARITY %s DIFFERENT old=%s new=%s\n' "$probe" "$old" "$new"
        cp "$dir/result.json" "$par/new_${probe}_result.json"
        mv "$dir" "$par/new_$probe" && cp -a "$par/orig_$probe" "$dir"
    fi
done
(( ok )) && echo PARITY_ALL_IDENTICAL || { echo PARITY_DIFFERENCES; exit 3; }
