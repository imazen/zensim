# hdrgrid944 linear baseline (floor) — 2026-08-26

`scripts/hdrgrid944_linear_control.py` (BVLS twin, deterministic, ZLIN_NFEAT=944,
mix = the hdrgrid944 leg alone) → `bakes/hdrgrid944-probe/hdrgrid944_lin944.bin`.

Owner read (`scripts/hdrgrid944_val_read.py`, n=6,510):
**val SROCC 0.7609, PLCC 0.8065, ZRMSE 0.5912** (652/944 active BVLS weights;
instrument shape: 372-screen transforms on the first 372 slots + identity above).

**FINDING**: the 944 (Folded720Append2) floor sits WELL BELOW the 372-leg floor
(0.8105 on the identical 24,750-cell population) — the folded front-end (zeroed
f156-371 + append blocks) is less LINEARLY predictive of the Appendix-Q cvvdp-mix
target than the v1 with-iw 372 HDR route. Whether MLPs close the gap is exactly
what wave-1 measures; do not read this as a regime verdict by itself.
