# Dial-reach eval — EXPANDED grid (dense near-lossless + JND + q0 + jxl-in-butter)

Follow-up to `dial_dynamic_range_reach_2026-05-29.md`. Per request, densified the
q-sweep where dial precision matters most. Built by
`scripts/v_next/build_qsweep_expanded.py` via `zen-metrics sweep --metric
zensim-gpu --zensim-features-regime with-iw --feature-output` (encode + 372-feature
extract in one shot), then `qsweep_eval`.

## Grid

| codec | axis | values |
|---|---|---|
| jpeg / webp / avif | quality | **q0** + low (5,10,15,20,25,30,40,50,60) + **JND zone 70..90 step 2** + **near-lossless 90..100 step 1** = 33 q |
| jxl | **butteraugli distance** (native near-lossless axis) | 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 1.8, 2.2, 2.6, 3.0, 3.5, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0 — relabeled q_equiv = round(100 − 7·d) so lower distance = higher quality on the monotonicity axis |

40 source images. ~25% of cells drop to NaN on odd-dimension images (GPU zensim
path); **3,230 valid rows** remain: jpeg 759, webp 759, avif 1122, jxl 590.
Features CSV + manifest: `/mnt/v/output/zensim/qsweep_expanded_2026-05-29/`.

## Result — the dense near-lossless zone separates the two dials

| bake | n_curves | strict monotonicity↑ | **tied↓** | G3 tied ≤5% |
|---|--:|--:|--:|---|
| **A (v47)** | 114 | 0.9525 | **0.031** | ✓ pass |
| **Cell5 (PreviewV0_5Linear)** | 114 | 0.9531 | **0.131** | ✗ FAIL |

Both keep ~95% strict monotonicity (no inversions). But on the dense step-1
near-lossless grid, **Cell5's tied rate quadruples to 13.1% vs A's 3.1%** — and
this was invisible on the coarse 16-q grid (where both showed <1% tied).

### Two findings the coarse grid hid

1. **Cell5 has dead-zones at fine near-lossless granularity.** At step-1 q90→q100,
   Cell5 produces equal scores on adjacent q for many individual (image, codec)
   curves — it can't reliably distinguish q96 from q97. A resolves those steps.
   For a codec binary-searching "give me score 92," Cell5's ties create ambiguity
   in exactly the near-lossless band where web-delivery quality decisions live.
2. **Cell5 floors to 0 on some images even at q100** (per-q `min = 0.00` across
   the entire near-lossless range), whereas A never drops below ~42 at high q.
   A few images sit in a Cell5 dial dead-zone. A's dial is more robust per-image.

Median climb is smooth for both (the medians hide it — different images tie at
different q), which is why only the dense per-curve adjacent-pair analysis exposes
the gap.

## Implication

For the **codec-target dial** use case, **A (v47) is the more reliable dial in the
near-lossless band** — finer resolution (3.1% tied vs 13.1%) and no per-image
floor. Cell5 remains the better held-out *ranker* (panel) and is fine as a coarse
dial, but the dense grid shows its near-lossless dial resolution is weaker. This
strengthens keeping A as the default codec-target profile; Cell5 is a
ranking/footprint sibling, not a near-lossless-dial replacement.

The expanded eval is also the right standing harness for any future dial bake: the
JND-zone + step-1 near-lossless + q0 + jxl-in-distance grid is where dial defects
actually surface.

## Reproduce

```bash
python3 scripts/v_next/build_qsweep_expanded.py   # ~few min on GPU box; writes the grid
./target/release/qsweep_eval \
  --features /mnt/v/output/zensim/qsweep_expanded_2026-05-29/expanded_features.csv \
  --manifest /mnt/v/output/zensim/qsweep_expanded_2026-05-29/expanded_manifest.tsv \
  --bake A_v47=zensim/weights/v47_strict_qat_native_2026-05-27.bin:clamp \
  --bake Cell5=zensim/weights/v02_372feat_cell5_2026-05-28.bin:clamp \
  --out /mnt/v/output/zensim/qsweep_expanded_2026-05-29/dialreach_expanded.md
```

Full per-q histograms: `/mnt/v/output/zensim/qsweep_expanded_2026-05-29/dialreach_expanded.md`.
