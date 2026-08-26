# hdrgrid372 linear baseline — the new leg's first number (2026-08-26)

**Deterministic BVLS baseline on the hdrgrid372 cvvdp-mix leg** (the first model
measurement on the 2026-08-26 hdrgrid harvest lineage; selection/sanity class —
never a UPIQ substitute, not a ship candidate).

| read | value |
|---|---|
| val SROCC | **0.8105** |
| val PLCC | 0.8371 |
| val ZRMSE | 0.5470 |
| n (val) | 6,510 (valdigits; test digits held out) |

Instrument: `scripts/hdrgrid372_linear_control.py` — the pinned
`linear_projections_2026-07-03.py` driven at ZLIN_NFEAT=372 (leg-native width,
372-screen + identity, v1 sign mask, tau 0; BVLS active weights 96/372;
deterministic, no seed). Read: `scripts/hdrgrid372_val_read.py` (forward =
predict_features_with_bake wire format; stats = zen_stats.panel → the canonical
Rust panel bin).

Provenance chain: bake `hdrgrid372_lin372.bin` sha256 02f63006a300f223… ← leg
`/mnt/v/output/zensim/hdrgrid372-leg/` (train 473879d6…, val 644fbeda…,
orientation-gated OK +0.8127/+0.8161) ← harvest
`/mnt/v/output/hdrgrid-2026-08-06/harvest-2026-08-26/` (writeback e461f96d) ←
the 8 drained sf/sf2 score waves on the LAN store (2026-08-26).

Context: the appendix-Q 944 leg's equivalent linear control (Q_lin944_hdr,
different lineage + width) is NOT comparable — different corpus, arms, grid.
This number is the hdrgrid372 leg's own floor for future MLP/wave arms to beat.
