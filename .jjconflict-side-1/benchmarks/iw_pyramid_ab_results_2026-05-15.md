# A/B comparison: spatial-variance vs steerable-pyramid IW weights (2026-05-15)

**Spike**: `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md`
**Binary**: `target/release/examples/iw_pyramid_ab` (in
`zensim-validate/examples/iw_pyramid_ab.rs`)
**Worktree**: `claude-session-iw-pyramid-spike`
**Commit**: see jj `@` change at end of spike

## What was measured

For three KADID reference images, computed the IW weight map under
both estimators:

- `IwWeightKind::LocalVariance` (current V_20a-shipped path)
- `IwWeightKind::SteerablePyramidLogGsm` (new spike variant; 4-orientation
  oriented gradient + max-across-orientations local variance)

Both with `info_log_sigma_e_sq: None` (raw variance, no log
transform — the log is a monotonic post-step that doesn't change
correlation or top-K overlap). Both at `kernel_half = 2` (5×5 patch,
current default) and `kernel_half = 4` (9×9 patch, paper-closer).

## Results — three KADID images

### kernel_half = 2 (5×5 patch — current default)

| Image | Pearson(LV, SP) | Spearman | Top-256 overlap | Top-1024 overlap | Top-4096 overlap |
|---|---:|---:|---:|---:|---:|
| I01_01_01 | **0.8454** | 0.8825 | 0.164 | 0.300 | 0.578 |
| I03_01_01 | **0.8303** | 0.9356 | 0.070 | 0.214 | 0.400 |
| I15_01_01 | **0.8393** | 0.9553 | 0.066 | 0.228 | 0.457 |

### kernel_half = 4 (9×9 patch — paper-closer)

| Image | Pearson(LV, SP) | Spearman | Top-256 overlap | Top-1024 overlap | Top-4096 overlap |
|---|---:|---:|---:|---:|---:|
| I01_01_01 | 0.8985 | 0.9127 | 0.125 | 0.251 | 0.532 |
| I03_01_01 | 0.8834 | 0.9493 | 0.105 | 0.303 | 0.436 |
| I15_01_01 | 0.8804 | 0.9638 | 0.133 | 0.184 | 0.412 |

## Interpretation

**Pearson at 5×5 is consistently < 0.85** (range 0.83–0.85, mean
0.838) → the methodology doc's decorrelation threshold is met.
**Pearson at 9×9 is 0.88–0.90 (mean 0.887)** → mixed signal, just
above the 0.85 threshold. The 9×9 path smooths out the
directional-max signal — larger patches average across multiple
orientations and lose some of the directional discrimination.

**Top-K overlap is the most striking diagnostic**: across all three
images at the 5×5 patch, only **~7–16 % of the top 256 most-weighted
pixels** overlap between the two methods. Even the top 1024 only
overlaps at ~21–30 %, and the top 4096 at ~40–58 %. This means the
two methods are selecting fundamentally different regions of the
image as "high information content" — exactly the behavior the
paper's wavelet-domain approach was meant to capture.

The Spearman correlation (rank correlation) is uniformly higher
than Pearson (0.88–0.96 at 5×5). The Pearson-Spearman gap (typically
0.04–0.13 across configs) means the two methods agree *on which
pixels are high-weight on average*, but disagree on the *magnitude*
of those weights. The directional-max amplifies edge pixels (the
spike's max-across-orientation rises faster than the
scalar-variance which averages all axes). That magnitude difference
is what an MLP could learn to exploit.

## Decision per methodology doc

At 5×5 (the current `IwWeightKind::LocalVariance` default kernel),
Pearson < 0.85 on every test image. Per the methodology doc:

> r < 0.85 (decorrelated) → wavelet path carries different signal.
> Worth training a 372-feat bake against the new estimator (~$0.30
> GPU).

**Recommendation: train a 372-feat bake** against
`IwWeightKind::SteerablePyramidLogGsm` (with `kernel_half = 2`,
`info_log_sigma_e_sq` swept across {0.1, 0.4, 1.0}). The decision
gate is met at the 5×5 patch size that matches the existing
production default.

At 9×9, the Pearson rises into the [0.85, 0.95] "mixed" band; this
is consistent with the methodology doc's hypothesis that **larger
patches smooth out the directional signal**. A 5×5 patch (current
default) is the right operating point — not the paper's 11×11.

## Caveats

- These are **synthetic test images** (KADID 512×384 references) on
  the **luminance channel only**. Production zensim runs on XYB
  with 3 channels and 4 pyramid scales. The decorrelation seen
  here may shrink on coarser pyramid scales (everything blurs) and
  may differ on the B/Y channels (less edge content). The training
  cost ($0.30) is low enough that the bake itself is the right
  next experiment, not a more elaborate sensitivity sweep.
- The 4-orientation approximation is NOT the paper's full
  steerable pyramid. If the bake trained against this estimator
  underperforms V_18 ship on CID22, the full Simoncelli pyramid
  may still be worth building (~200 LOC) — different filter shape,
  different signal. But if the cheap approximation wins (even
  marginally), the full pyramid is over-engineered.
- The paper's `σ²_e` calibration is critical. The default
  `info_log_sigma_e_sq: None` (no log transform) keeps the raw
  variance scale; with log transform applied, the saturation curve
  changes the shape of the weight distribution. The bake training
  should sweep `info_log_sigma_e_sq ∈ {None, 0.1, 0.4, 1.0}`.

## Cost summary

- **Spike compute spend so far**: ~10s per image × 3 images × 2
  patch sizes = 60s total. Trivial.
- **Recommended next experiment**: 372-feat bake against
  `SteerablePyramidLogGsm` on safe-synthetic + KADID + TID, seed=1
  first per CLAUDE.md "principled experiment workflow Step 3".
  Cost: ~30 min CPU on AMD 7950X, or ~$0.15 on vast.ai. Total:
  ~$0.30 across {None, 0.4} σ²_e settings.

## Reproducing

```sh
cd ~/work/zen/zensim/.claude/worktrees/agent-ae0e10c9dcc6401ac
cargo build --release -p zensim-validate --example iw_pyramid_ab
./target/release/examples/iw_pyramid_ab \
    /mnt/v/dataset/kadid10k/images/I01_01_01.png \
    /mnt/v/dataset/kadid10k/images/I01_01_03.png
```

## Files

- Spike methodology: `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md`
- A/B binary: `zensim-validate/examples/iw_pyramid_ab.rs`
- New estimator: `zensim/src/iw_pool.rs` (`IwWeightKind::SteerablePyramidLogGsm`)
- A/B logs: `/tmp/iw_pyramid_ab_kadid_I01.log`, `/tmp/iw_pyramid_ab_kadid_I03.log`,
  `/tmp/iw_pyramid_ab_kadid_I15.log`
