#!/usr/bin/env bash
# Explicit corpus producer for the frozen C8 ratio and XYB dumps.
# Usage: scripts/gmsbank/calibration_instrument.sh /var/tmp/gmsbank/new-calibration
# The input directory must contain planes.tsv and its decoded RGB8 files.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo 'usage: calibration_instrument.sh <fresh-planes-directory>' >&2
    exit 2
fi
root=$(realpath -e -- "$1")
if [[ ! -f "$root/planes.tsv" ]]; then
    echo "missing $root/planes.tsv" >&2
    exit 2
fi
if [[ -e "$root/ratios.tsv" || -e "$root/xyb8.tsv" ]]; then
    echo 'refusing to overwrite calibration outputs' >&2
    exit 2
fi
target=${CARGO_TARGET_DIR:-/var/tmp/gmsbank/target-calibration}
case "$target" in
    /var/tmp/*) ;;
    *) echo 'CARGO_TARGET_DIR must be under /var/tmp/' >&2; exit 2 ;;
esac

/home/lilith/tmp/devin/heavy --mem 16G --jobs 8 -- \
    env CARGO_TARGET_DIR="$target" \
        RUSTFLAGS='--cfg gmsbank_calibration_instrument' \
        GMSBANK_CALIB_DIR="$root" \
        cargo test --locked -p zensim --release --all-features --lib \
        gmsbank_calibration_scale1_y_dump -- --nocapture
