# Per-band weighted TV regularizer — design

**Authored**: 2026-05-12, after V0_8 ship. User-directed direction:
"Start with per-band-weighted TV first".

## Problem

V0_8 (TV=15 seed=1, current ship) trades smoothness for B1 closure
relative to V0_7 (TV=10 seed=1):

| Metric | V0_7 (TV=10) | V0_8 (TV=15) | Δ |
|---|--:|--:|--:|
| B1 SROCC vs ssim2 | -0.027 | -0.014 | **+0.013 (better)** |
| B0 SROCC vs ssim2 | -0.005 | -0.010 | -0.005 (worse) |
| Near-PJND vs ssim2 | -0.017 | -0.024 | -0.007 (worse) |
| Non-mono % | 5.46% | 5.87% | -0.41% (worse) |

V0_8 sweep findings: higher TV (15/20) **helps B1** but **hurts B0
and non-mono**. **A constant TV weight is suboptimal** — different
bands want different TV strengths.

## Proposed: per-band TV weight

Replace the constant `--tv-weight` with a per-band table:

```
tv_weight = {
  B0 (<50):    10.0,  // low-q: avoid over-regularizing
  B1 [50,65):  20.0,  // medium-q: strong TV closes the gap
  B2 [65,90):  15.0,  // high-q: moderate
  B3 (>=90):   10.0,  // visually-lossless: avoid over-fit on n=43
}
```

When TV update sample picks pair `(lo_idx, hi_idx)`, compute the
**band of the pair** by:
- Use ssim2 score of the SAFESYN training-row corresponding to the
  pair's `hi_idx` (high-quality member, MCOS-aligned proxy)
- Look up `tv_weight[band]` and apply that as the scale factor
- Default to a "fallback weight" for KonJND / other groups whose
  scores aren't MCOS-aligned

## TV pairs file format change

Current TV pairs file format (per `zensim/scripts/v_next/regen_tv_pairs.py`):

```tsv
lo_trainer_idx  hi_trainer_idx
0   1
1   2
...
```

Extended format:

```tsv
lo_trainer_idx  hi_trainer_idx  band_id
0   1   2
1   2   2
...
```

`band_id` is one of `{0, 1, 2, 3}` mapped to B0/B1/B2/B3, computed
at TV-pair-generation time from the ssim2 score of the hi-idx row.

For backward compat: trainer treats absent `band_id` column as
"all band 2" (B2), preserving prior single-weight behavior.

## Trainer change

In `zensim-validate/src/mlp_train.rs` `TvRegularizer` struct:

```rust
pub struct TvRegularizer {
    pub pairs: Vec<(usize, usize)>,
    pub features: Vec<Vec<f64>>,
    pub band_id: Option<Vec<u8>>,  // NEW: optional per-pair band
    pub weight: f64,                // baseline (used if band_id is None)
    pub band_weights: Option<[f64; 4]>,  // NEW: per-band weights B0..B3
    pub apply_every: usize,
    pub batch: usize,
}
```

In `train_mlp_with_tv` inner loop, when applying TV gradient:

```rust
let pair_weight = match (&tv.band_id, tv.band_weights) {
    (Some(bands), Some(weights)) => weights[bands[pair_idx] as usize],
    _ => tv.weight,
};
// scale TV gradient by pair_weight
```

## CLI extensions

`zensim_mlp_train` gains:

```
--tv-band-weights B0,B1,B2,B3   e.g. --tv-band-weights 10,20,15,10
```

If specified, the TV pairs file must include the `band_id` column.

## Expected outcome

If hypothesis holds:
- B1 SROCC matches or exceeds V0_8 (-0.014 → ≤ -0.014, possibly 0.0)
- B0/Near-PJND match V0_7 (-0.005 / -0.017, not V0_8's worse values)
- Non-mono stays below 5.5%-6.0% range

Risk: maybe the TV regime change disrupts cyclic-LR convergence, or
the per-band weighting fights with cross-band rank consistency.

Empirical validation: train (seed=1, band-weights `[10, 20, 15, 10]`)
and compare to V0_8 baseline.

## Implementation steps

1. **Extend `regen_tv_pairs.py`** to emit `band_id` column based on
   ssim2 of the hi-idx row.
2. **Patch `TvRegularizer`** in mlp_train.rs.
3. **Add `--tv-band-weights` flag** to `zensim_mlp_train`.
4. **Build a TV-pairs-with-bands file** for the cleaned safesyn corpus.
5. **Train (seed=1, band-weights `[10, 20, 15, 10]`)** on cleaned data.
6. **Eval CID22 + per-band + non-mono** and compare to V0_8.

Total estimated work: ~3-5 hours implementation + ~30 min training +
~5 min eval.

## Next steps after per-band TV

If per-band TV closes B1 to within +/-0.005 of ssim2 without breaking
other bands, V_9 ships. If not, alternative directions:
- Targeted training-pair densification at B1 quality (50-65 MCOS)
- Architecture: more scales / asymmetric features
- Image-type-aware MLP dispatch (next-next priority)
