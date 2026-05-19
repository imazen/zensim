# `PreviewV0_5TunerV2` from V3 candidates — FALSIFIED on monotonicity (2026-05-19)

## Hypothesis (parent task)

V2 (`cc4v2`) candidates collapsed to constant output ~63 score, failing
the Tuner trail's dynamic range gate by 500× (range 0.10 vs ≥ 50). V3
adds two architectural counterweights to prevent collapse:

1. **Cross-codec rank-preserve term** weighted by `|butter_diff|` —
   pushes (y_a, y_b) ordering to match pivot-metric ordering on equiv
   pairs. Collapse to `y_a = y_b` violates rank → gradient grows with
   collapse severity.
2. **Dynamic-range floor probe** — `L = w · max(0, σ_thresh − σ_obs)²`
   on N=40 random A-side equiv rows, fires 5 % of pair-steps.
   Structurally forbids constant-output failure mode at the σ level.

Plus stronger monotonicity regularizer (`--monotonicity-reg 5.0` vs
V2's 0.5).

Sweep: 3 seeds × W ∈ {0.5, 1.0} = 6 bakes.

## Verdict: **falsified.**

**0 of 6 V3 candidates pass the Tuner trail gate.** The architectural
counterweights successfully solved V2's collapse problem — every V3
candidate has dynamic range 89.94-90.02 score units post-calibration
(gate ≥ 50). But the same machinery that prevents collapse also
prevents the network from learning a smooth monotonic per-image curve.
**Every V3 candidate has lower strict monotonicity than the
baseline tuner** (best V3 = 0.9100 vs baseline 0.9278 vs gate 0.9378).

## Per-bake gate scorecard

Gates: `strict_mono ≥ 0.9378`, `tied ≤ 5 %`, `range ≥ 50`,
`T=63 butter_max OR butter_p3 < 2.5`.

| Bake | strict_mono | tied | q5 med | q95 med | range | T=63 butter_max | T=63 butter_p3 | mono | tied | range | xc | ALL |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|
| baseline_tuner (existing ship) | 0.9278 | 0.0044 | 4.96 | 94.64 | 89.68 | (8.07 / 2.11)* | — | ✗ | ✓ | ✓ | ✗ | (current) |
| cc4v3_s1_w0_5 | 0.8622 | 0.0556 | 4.99 | 94.98 | 89.99 | 6.76 | 2.57 | ✗ | ✗ | ✓ | ✗ | FAIL |
| cc4v3_s1_w1_0 | 0.8933 | 0.0822 | 5.00 | 94.97 | 89.97 | 6.09 | **2.26** | ✗ | ✗ | ✓ | ✓ | FAIL |
| cc4v3_s2_w0_5 | **0.9100** | 0.0278 | 5.01 | 95.00 | 89.99 | 7.70 | 2.91 | ✗ | ✓ | ✓ | ✗ | FAIL |
| cc4v3_s2_w1_0 | 0.8233 | 0.0989 | 5.01 | 95.03 | 90.02 | 6.10 | **2.21** | ✗ | ✗ | ✓ | ✓ | FAIL |
| cc4v3_s3_w0_5 | 0.8711 | 0.0522 | 5.00 | 94.99 | 89.99 | 8.99 | 3.27 | ✗ | ✗ | ✓ | ✗ | FAIL |
| cc4v3_s3_w1_0 | 0.8733 | 0.0600 | 5.03 | 94.97 | 89.94 | 7.00 | 2.61 | ✗ | ✗ | ✓ | ✗ | FAIL |

*baseline T=63 figures are from prior `v_cross_codec_findings_2026-05-19.md`
(n=20 on the same image set). Note: the baseline tuner already fails
its own mono gate (0.9278 vs gate 0.9378) — the gate is "1 pp better
than baseline", which baseline itself doesn't satisfy.

## What V3 actually solved

| Failure mode | V2 result | V3 result | Verdict |
|---|---|---|---|
| Constant-output collapse | range 0.10-0.92 | **range 89.94-90.02** | **SOLVED** |
| σ_obs floor across q-sweep | ~0 across q | 10-13 raw units within sweep | **SOLVED** |
| Cross-codec T=63 butter_p3 | 0.54 (collapse artifact) | 2.21-3.27 (real signal) | gate passes on 2/6 |
| Strict monotonicity | (n/a, collapsed) | 0.82-0.91 | **NEW failure mode** |
| Tied rate | 0 % (collapsed) | 2.8-9.9 % | borderline |

## Why V3 broke monotonicity

Three mechanisms compounded:

### 1. Raw output range is too tight (10-13 score units pre-affine)

V3 bakes produce raw q5/q95 medians in (50.99..58.39, 66.45..68.08).
That's only 10-13 raw score units spanning the full q range. The σ-floor
probe targets `σ ≥ 15` ACROSS A-side equiv pool features (which span
many codecs/refs/q levels) — but for a SINGLE source image's q-sweep
(50 images × 19 q values), the within-image raw spread is much smaller
than σ across the heterogeneous probe substrate.

After affine `β` (5-10×) maps that 10-13 raw range to 0-100, the
per-pair raw jitter (~0.05-0.2 raw units between adjacent q) gets
amplified to ~0.5-2 score units. Floating-point rounding artifacts at
the bake's output layer become visible "ties" at adjacent q steps,
inflating the tied rate to 5-10 %.

### 2. Monotonicity-reg 5.0 fights σ-floor when curves are short

The per-curve mono-reg pushes adjacent-q outputs apart by ≥ 0
(margin=0). The σ-floor pushes ALL outputs to spread σ ≥ 15 across the
probe — including outputs that for a SINGLE image's q-sweep should
naturally be in a narrow band (e.g., q=80 vs q=85 differ by only ~1
butter unit). The two losses compete: mono pushes (q_lo, q_hi) outputs
apart smoothly, σ-floor pushes the bake to manufacture spread between
unrelated probe rows. The net effect is non-smooth output curves.

### 3. Affine calibration's β-amplifier exposes raw jitter

Pre-affine, the V3 bakes are reasonably monotonic on raw outputs (the
raw-mode `qsweep_eval` row would show better mono rate, but the
runtime ships post-affine because we need 0-100 score range). The
β=5-10× scaler that maps 10-13 raw to 0-100 also scales every per-pair
raw inversion by the same factor — turning a 0.05-raw inversion into a
0.5-score-unit violation, which trips strict mono detection.

The baseline tuner avoids this because its raw range is ALREADY
89.68 units wide (β ≈ 1.0), so jitter doesn't get amplified.

## What V3 did succeed at

- **No collapse**: every V3 candidate has dynamic range > 89, range
  gate passes.
- **CID22 SROCC competitive**: 0.852-0.883 vs baseline tuner 0.879
  (V3 candidates carry the rank-preserve signal that helps cross-corpus
  ranking).
- **Cross-codec gate passes on 2/6 candidates** (W=1.0 seeds 1 + 2,
  butter_p3 = 2.26, 2.21 respectively). The cross-codec mechanism
  works when not collapsed.

The "pass cross-codec + non-collapsed range" combination is itself a
new Pareto point that the existing PreviewV0_5CrossCodec doesn't
achieve — but it does NOT pass the Tuner trail gate, which requires
mono > 0.9378.

## V4 direction proposals

The V3 falsification rules out the "rank-preserve + σ-floor on equiv
pool" recipe at these hyperparams. The conceptual approach
(architectural anti-collapse) is sound — V3 solved the σ-floor failure
mode cleanly. The next iteration should target the new failure mode
(mono-loss from raw-range compression + β amplification):

### V4-A: σ-floor over per-curve substrate (recommended first try)

Replace the equiv-pool A-side probe with a **per-image q-sweep
substrate**: for each of 8 random training images, forward features at
q ∈ {5, 25, 50, 75, 95}, compute σ across those 5 outputs (NOT 40
heterogeneous equiv rows). σ-threshold lowers to ~3-5 raw units (the
natural within-image spread). This decouples the σ requirement from
cross-image variance and aligns the σ-floor objective with the
monotonic-curve objective.

Cost: need a per-image-grouped q-sweep parquet (we have one at
`/mnt/v/zen/picker-training/2026-05-19/butter/*.parquet`, 1000 refs ×
19 q values per codec). The trainer would need a `--q-sweep-parquet`
loader and a grouped sampler. ~2 hr trainer work.

### V4-B: lower mono-reg, higher rank-preserve

V3 used mono-reg=5.0 + rank-preserve=0.2. The mono-reg was set high to
push back against collapse, but V3's σ-floor already prevents collapse
structurally — so mono-reg can be relaxed (back to V_tuner-v2's 1.0)
without re-introducing collapse risk. Meanwhile rank-preserve at 0.2
gives ~0.022 gradient per pair; bumping to 0.5-1.0 would let
rank-preserve dominate the equiv MSE more aggressively, which may
reduce the within-curve jitter that breaks mono.

Cost: hyperparam sweep only (no code change). ~30 min × 6 bakes.

### V4-C: train without aggressive affine

The β=5-10× amplification is the proximate cause of the strict-mono
collapse. If the trainer produces raw outputs in the [0, 100] range
natively (β ≈ 1), we avoid the amplification. Approach: replace the
score-shape head's identity output layer with a tanh-scaled output,
hard-pinned to `[0, 100]`. This changes the bake architecture (not
just hyperparams) but may decouple "learn cross-codec consistency" from
"compress to narrow raw range."

Cost: trainer architectural change. ~1 day.

### V4-D: dual training — Tuner head + CrossCodec head

Train two heads in parallel from the same backbone, one with Tuner-v1
recipe (preserves mono), one with V3 cross-codec recipe (preserves
cross-codec consistency). Ship as ensemble routing: monotonicity-sensitive
queries (binary-search) use Tuner head, codec-comparison queries use
CrossCodec head. This admits the two objectives are mechanically
incompatible on a single head and ships them separately.

Cost: 2-head trainer + ensemble runtime. ~1 day.

## Decision

**No PreviewV0_5TunerV2 ship from V3.** PreviewV0_5Tuner (baseline
tuner, V_tuner-v2-s2 calibrated, range 89.68, strict mono 0.9278)
remains the dial profile. PreviewV0_5CrossCodec (V2 W=1.0 seed=1, T=63
butter 5.52) remains the cross-codec profile.

The V3 line is **closed** for Tuner-trail consideration at the
hyperparameter combination tested. Re-opening requires:
1. Different probe substrate (per-curve, not heterogeneous-pool).
2. OR architectural change to the bake's output head.
3. OR explicit recognition that mono + cross-codec are incompatible
   on a single bake (V4-D ensemble approach).

V4-A is the recommended first try because it preserves the V3
counterweight machinery while addressing the diagnosed root cause.

## Files produced this session

- `benchmarks/v_tuner_v3_methodology_2026-05-19.md` — hypothesis +
  loss math + recipe.
- `benchmarks/v_tuner_v3_eval_2026-05-19.md` — sweep grid + ship table
  template (populated below).
- `benchmarks/v_tuner_v3_falsification_2026-05-19.md` — this doc.
- `scripts/v_next/run_cross_codec_v3_seed.sh` — driver.
- `scripts/v_next/eval_cross_codec_v3.sh` — 5-phase eval pipeline.
- `scripts/v_next/run_cross_codec_v3_consistency.sh` — T=63 driver.
- `/mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19/` — bakes +
  calibrated + verdicts + cross-codec TSVs.

## Reproduction

```bash
cd /home/lilith/work/zen/zensim--cross-codec-metric
# 1. Build trainer
cargo build --release --bin zensim_mlp_train -p zensim-validate
# 2. Train 6 bakes in parallel (~33 min on 7950X)
for seed in 1 2 3; do
  for w in 0.5 1.0; do
    bash scripts/v_next/run_cross_codec_v3_seed.sh $seed $w &
  done
done; wait
# 3. Eval pipeline (raw qsweep → affine → calib qsweep → verdicts, ~3 min)
bash scripts/v_next/eval_cross_codec_v3.sh
# 4. Cross-codec T=63 (n=20) per bake — ~1 min each
bash scripts/v_next/run_cross_codec_v3_consistency.sh
# 5. Summary
python3 /tmp/v3_summary.py
```

Total compute: ~40 min on 7950X. Total data: ~1.5 MB bakes + ~10 MB
eval artifacts.

## Trainer commits

- `de097f1c` (main, 2026-05-19): trainer + CLI for range floor +
  rank-preserve.
- `579229eb` (main, 2026-05-19): driver + eval + methodology.

## Sister artifact

`/mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19/qsweep_calibrated.md` —
full per-q histogram per bake.
