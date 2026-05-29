# Dial dynamic range + reach across codec configurations — 2026-05-29

User asked for "a metric that represents dial dynamic range and reach across
codec configurations." The metric is `qsweep_eval`'s **monotonicity_rate +
tied_rate + per-q dial span**, run on a **multi-codec q-sweep grid**. This is
the codec-target (G1/G3/G4) view that `bake_verdict`'s static-corpus panel
does NOT capture.

## The grid (real, reused on-disk variants — no re-encoding)

`scripts/v_next/build_qsweep_372_grid.py` → 40 source images × 4 codec
families (JPEG=mozjpeg-rs-420, WebP=zenwebp, JXL=zenjxl, AVIF=zenavif) × 16 q
values (q5..q100) = **2,560 pairs**, 372 features extracted via
`extract_features_372col --corpus qsweep`. Features CSV:
`/mnt/v/output/zensim/qsweep_372_grid_features.csv`.

Note: a 300-input bake scores the 372-feature grid correctly — `bake_runtime`
copies `min(n_inputs, row.len())` = its `f0..f299` prefix (the IW-pool block
`f300..f371` is just the tail). So all profiles are comparable here.

## Result — dial dynamic range + codec reach

| profile | n_in | monotonicity↑ | tied↓ | dial span (q5 med → q100 med) | codec-target dial? |
|---|--:|--:|--:|---|---|
| **Cell5 (PreviewV0_5Linear)** | 372 | **0.961** | 0.008 | 2.6 → 93.1 (**~90 units**) | ✓ best — widest low-end reach |
| **A (v47, default)** | 372 | 0.930 | 0.007 | 23.3 → 93.0 (~70 units) | ✓ proper; floors ~23 at q5 |
| Balanced (V0_5) | 300 | 0.642 | 0.608 | 9.9 → **0.0 ∀ q≥50** | ✗ collapsed |
| Compression (V0_5) | 300 | 0.528 | 0.511 | 14.7 → **0.0 ∀ q≥50** | ✗ collapsed |

Per-q median climb (the dial), JPEG+WebP+JXL+AVIF pooled:

| q | A (v47) | Cell5 | Balanced | Compression |
|--:|--:|--:|--:|--:|
| 5 | 23.3 | 2.6 | 9.9 | 14.7 |
| 20 | 40.7 | 30.4 | 4.5 | 7.7 |
| 50 | 64.0 | 57.8 | 0.0 | 0.0 |
| 80 | 82.6 | 78.6 | 0.0 | 0.0 |
| 95 | 91.4 | 89.5 | 0.0 | 0.0 |
| 100 | 93.0 | 93.1 | 0.0 | 0.0 |

## Findings

1. **The metric discriminates dial-calibrated from rank-only profiles
   cleanly.** A and Cell5 climb monotonically q5→q100 across all four codec
   families and pass G3 (≥93% strict monotonicity); Cell5 is best (96.1%,
   ~90-unit span, deepest low-q reach).
2. **Balanced + Compression are NOT codec-target dials.** Their median score
   pins to **0 for every q ≥ 50** with 51–61% tied pairs. They are
   SOTA-rank-trail bakes with **no tanh-pin + no dial spline** (`spline=no`
   in the eval), so `clamp(raw,0,100)` floors their rank-shaped raw output.
   No post-mode rescues them — they are ranking tools, not dials. This is the
   intended discrimination, not a feature-count artifact (the 300→372 prefix
   slice is correct).
3. **Confirms the minimal-profile instinct:** of the hidden V0_5 wave, none
   is dial-viable. A and Cell5 are the codec-target keepers; the V0_5*
   variants add binary bloat without a dial-usable contribution.

## Reproduce

```bash
python3 scripts/v_next/build_qsweep_372_grid.py --out /tmp/grid.tsv --n-images 40
./target/release/examples/extract_features_372col --corpus qsweep \
    --path /tmp/grid.tsv --out /mnt/v/output/zensim/qsweep_372_grid_features.csv
./target/release/qsweep_eval \
    --features /mnt/v/output/zensim/qsweep_372_grid_features.csv \
    --manifest /tmp/grid.tsv \
    --bake A_v47=zensim/weights/v47_strict_qat_native_2026-05-27.bin:clamp \
    --bake Cell5_linear=zensim/weights/v02_372feat_cell5_2026-05-28.bin:clamp \
    --out /mnt/v/output/zensim/qsweep_372_dialreach_2026-05-29.md
```

Full per-q histograms + calibration-linearity tables:
`/mnt/v/output/zensim/qsweep_372_dialreach_2026-05-29.md`.
