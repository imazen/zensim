# V_20 input-shaping methodology + results (2026-05-15)

**Status**: training in flight; this doc captures methodology now and
will be filled with results once the trainer finishes.

## Hypothesis

zenpredict 0.2.0 (commit `8cae13b`) shipped the first 3 V_20
`FeatureTransform` variants (`signed_log1p`, `signed_sqrt`,
`signed_cbrt`). Today (zenpredict `ea217f2`) shipped the 3
parameterized variants (`clip_then_log1p`, `winsor_p99`,
`quantile_bins`). zensim's trainer now accepts these via
`--feature-transform TOKEN:IDX[:PARAMS]` (commits `6fb79bb1` +
`dc98058b`).

The V_20 design doc (`benchmarks/v0_20_v0_21_design_2026-05-14.md`)
predicted that per-feature transforms applied BEFORE the
standardize step would let the MLP see features whose post-transform
distributions are closer to linear-with-MOS — improving training
efficiency and final SROCC at constant model size.

The greedy screen
(`benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv`,
v2 with safety gates) showed **MASSIVE Pearson lifts** across 249
of 300 features after best-of-6-transforms — some features go from
Pearson 0.15 with MOS to 0.90 after `winsor_p99`. Top-feature
distribution:

| Best transform | Count (of 300) | % |
|---|---:|---:|
| winsor_p99 | 170 | 57% |
| clip_then_log1p | 43 | 14% |
| signed_sqrt | 35 | 12% |
| signed_cbrt | 26 |  9% |
| quantile_bins | 16 |  5% |
| log1p | 5 | 2% |
| identity | 5 | 2% |

Pearson lift across all 300 features: p50=0.050, p75=0.153, p90=0.402, max=0.746.

**Caveat (the screen does NOT prove training lift)**: Pearson on
raw features is a SCREEN, not a sufficient condition. The actual
question — "does the trainer converge to a better minimum with
transformed features?" — needs an MLP train+eval to answer.

## Experiment

Train one V_20-input-shaping candidate single-MLP with the V_18
base recipe (4 corpora, h=128, seed=1, 300 epochs), applying the
**top 98 transforms** from the greedy screen (those with Pearson
lift ≥ 0.05, capped to feat_idx < 228 to match the runtime input
width).

### Recipe

```sh
CLEAN=/mnt/v/zen/zensim-training/2026-05-14-clean
ARGS=$(grep -v '^#' /tmp/v0_20_transforms_top.txt | tr '\n' ' ')
./target/release/zensim_mlp_train \
  --group safesyn:$CLEAN/safe_synth_v19_clean_features.csv:1.0:0.0 \
  --group kadid:$CLEAN/kadid_features.csv:0.3:1.0 \
  --group tid:$CLEAN/tid_features.csv:0.3:1.0 \
  --group konjnd:$CLEAN/konjnd_aligned_features.csv:0.5:1.0 \
  --hidden 128 --epochs 300 --seed 1 \
  $ARGS \
  --out benchmarks/v0_20_input_shaping_seed1_2026-05-15.bin
```

Args list lives at `/tmp/v0_20_transforms_top.txt` (regenerated via
`scripts/v_next/v0_20_screen_to_trainer_args.py`).

## Comparison baselines

| Bake | CID22 SROCC | Status |
|---|---:|---|
| V_18 ship (3-way concat 0.65/0.30/0.05) | 0.8933 | Shipped baseline |
| V_18 base_seed1 alone | 0.8880 | Same-architecture comparison |
| **V_20 input-shaping seed1** | **TBD** | This experiment |

V_18 base_seed1 is the apples-to-apples comparison for a single
228→128→1 MLP. If V_20 input-shaping CID22 SROCC > 0.8880 by
≥ 0.005 with no >0.005 KADID/TID regression, that's evidence the
mechanism transfers and warrants a full 3-way concat sweep.

## Results

_(Pending — training in flight as of 2026-05-15)_

| Bake | KADID | TID | CID22 | CID22 B3 [30,40) | n |
|---|---:|---:|---:|---:|---:|
| V_18 ship | 0.9427 | 0.9526 | 0.8933 | 0.0246 | (ref) |
| V_18 base_seed1 | 0.9326 (est) | 0.9521 (est) | 0.8880 | (TBD) | (ref) |
| V_20 input-shaping seed1 | TBD | TBD | TBD | TBD | 138872+10125+3000+76104 |

## Acceptance gate

Per `benchmarks/v0_18_ship_reference_card_2026-05-14.md`:

- CID22 agg ≥ 0.8880 (V_18 base_seed1 floor for single-MLP comparison)
- CID22 B3 [30, 40) ≥ 0.13 (fast-ssim2 floor — V_18's weakest band)
- KADID and TID within −0.005 of V_18 ship's numbers
- No NaN losses during training (already gated via screen v2)

## Next steps if positive

1. Full 3-way concat: train the cycle-14 TV-regularized components
   with the same transforms, concat with same 0.65/0.30/0.05 mix,
   compare against V_18 ship.
2. Methodology doc + ship swap decision per
   `zensim/CLAUDE.md` shipping policy.
3. Bake size optimization: lz4+zerobias (the V_18 ship-form swap
   path) compresses 5.2× — apply same.

## Next steps if negative / null

1. Try a more aggressive screen threshold (lift ≥ 0.10 = 98 features
   → likely 50 features).
2. Try fewer feature transforms (top 10 only, isolate which
   transforms matter).
3. Conclude: greedy-Pearson-screen does NOT predict MLP-training
   lift; the MLP already absorbs the input non-linearity. Pivot to
   V_20b (distortion-manifold pre-training) for B0..B5 lift.

## Repro

- Screen: `benchmarks/v0_20_feature_transform_greedy_screen_2026-05-15.tsv`
- Args gen: `scripts/v_next/v0_20_screen_to_trainer_args.py`
- Train log: `/tmp/v0_20_input_shaping_2026-05-15.log`
- Bake: `benchmarks/v0_20_input_shaping_seed1_2026-05-15.bin`

## References

- V_20 design doc: `benchmarks/v0_20_v0_21_design_2026-05-14.md`
- V_18 reference card: `benchmarks/v0_18_ship_reference_card_2026-05-14.md`
- zenpredict commits: `8cae13b` (non-param variants), `ea217f2` (params)
- zensim commits: `6fb79bb1` (--feature-transform flag), `dc98058b` (params extension)
