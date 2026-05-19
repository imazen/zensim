# V_tuner-v2 — cross-codec JND anchor loss FALSIFICATION (2026-05-19)

**Status: FALSIFIED at both `--anchor-loss-weight 1.0` AND
`--anchor-loss-weight 0.05`.** Two 3-seed runs with a 20× weight
spread both fail. The approach is **structurally inadequate** for
the cross-codec consistency problem; see "Why the approach was
structurally wrong" below.

## Round 1 (weight = 1.0)

The anchor MSE loss at weight 1.0 collapsed the bake's dynamic
range — every input gets predicted as ~63 — killing rank fidelity on
held-out corpora AND making cross-codec butter at T=63 essentially
unchanged.

## Round 2 (weight = 0.05, 1/20× of round 1)

Lighter weight preserved most of CID22 SROCC (0.867 mean vs 0.879
baseline, s3 individual = 0.8794 — within noise of baseline). But
the bake **still saturates at ~63 for q≥25**, so cross-codec butter
at T=63 stays at 5.99 (1.0× round-1's 6.59). KonJND essentially
unchanged at 0.255 (baseline 0.235). 2 of 4 criteria pass (CID22 +
butter T=70). KonJND-SROCC and butter-T=63 both fail.

## Why the approach was structurally wrong

The hypothesis was: "force feature vectors with ssim2≈63 / KonJND
PJND to score 63 → cross-codec butter at T=63 collapses." But:

1. **Cross-codec butter is a property of encoder outputs, not the
   scorer.** When JPEG q=N, WebP q=M, AVIF q=K all hit zensim=63,
   the butter distance between their *decoded outputs* doesn't
   change because the bake assigns them all "63". The encoder
   choice (which q at which codec) is downstream of the bake; the
   bake decides which q each codec picks, but the codecs' decoded
   outputs at those q values stay visually different.

2. **The anchor loss collapses dynamic range to win MSE on anchor
   rows.** Both rounds show the bake learning "predict 63 for
   anything roughly at PJND quality." The bake can't simultaneously
   (a) place anchors at 63 and (b) predict score=80, 90 on cleaner
   pairs — without the second behavior, score-to-codec-q targeting
   above T=70 becomes impossible (`mean_dist_from_target` > 16 at
   T=80, > 26 at T=90 in both rounds).

3. **The product question "score=63 means PJND across codecs" is
   already partly addressed by the existing Tuner.** Today's Tuner
   achieves dist_from_target < 1.0 at T=63 (codec-q searches land
   the codecs within 1 score unit of 63). The remaining 6.68
   pairwise butter at T=63 isn't a bake-shape problem — it's
   inherent to PIL JPEG q ≈ 30 vs PIL WebP q ≈ 50 vs PIL AVIF
   q ≈ 25 producing genuinely different decoded outputs at the
   PJND-equivalent score.

4. **To meaningfully tighten cross-codec butter at T=63, we'd need
   a different intervention** — either (a) per-codec score
   calibration that compensates for inherent codec differences
   (e.g. "WebP score is 5 less than JPEG score at the same butter"),
   or (b) the bake learning butter directly rather than ssim2/MOS,
   or (c) production replacing PIL encoders with calibrated
   encoders (zenjpeg + zenwebp + zenavif with shared q semantics).

## Falsification table — both rounds vs baseline

| Criterion | Threshold | Round 1 (w=1.0) | Round 2 (w=0.05) | Pass? (R1 / R2) |
|---|---|---:|---:|:---:|
| butter T=63 | < 3.0 | 6.59 | 5.99 | FAIL / FAIL |
| butter T=70 | < 2.5 | 2.04 | 1.43 | PASS / PASS |
| CID22 SROCC | ≥ 0.85 | 0.8354 | 0.8666 | FAIL / PASS |
| KonJND SROCC | ≥ 0.80 | 0.1321 | 0.2554 | FAIL / FAIL |

Round 2 picks up CID22 (loosening the anchor enough that rank
fidelity recovers near baseline) but still fails the two
load-bearing criteria for the *original* hypothesis (cross-codec at
T=63 and KonJND).

## Hypothesis (recap)

User directive: *"don't we want jnd anchoring at a certain number cross codec"*.

Today's Tuner (PreviewV0_5Tuner, V_tuner-v2-s2 calibrated) trains MSE on
`mix_cv40_iw60` from safesyn. Cross-codec consistency eval (2026-05-19)
measured **mean pairwise butter_max at T=63 = 6.68** across JPEG/WebP/AVIF
on 10 source images — above the 4.0 "broken" threshold. The hypothesis:
adding an explicit anchor MSE loss that ties feature vectors with
ssim2≈63 (synthetic) and KonJND PJND (real human) to score=63 will
collapse the cross-codec spread to < 3.0 while preserving CID22 SROCC
≥ 0.85, monotonicity ≥ 0.92, and lifting KonJND from 0.235 to ≥ 0.80.

## Falsification table

| Criterion | Threshold | Mean (3 seeds) | Pass? |
|---|---|---:|:---:|
| butter T=63 | < 3.0 | **6.59** | FAIL |
| butter T=70 | < 2.5 | **2.04** | PASS |
| CID22 SROCC | ≥ 0.85 | **0.8354** | FAIL |
| KonJND SROCC | ≥ 0.80 | **0.1321** | FAIL (worse than baseline 0.2351) |

1 of 4 hard criteria passes — falsification is decisive.

## What went wrong: the anchor pulled too hard

The recipe `--anchor-loss-weight 1.0 --anchor-step-p 0.10` interleaves
one anchor MSE step per ~10 pair-steps. With 9,373 anchor rows all
targeting score=63 and a per-row weight of 1.0–1.5, the anchor's
cumulative gradient force significantly exceeded the pair-loss gradient.

The bake's strict-mono `q`-sweep histogram shows the collapse plainly
(`benchmarks/qsweep_v2_vs_baseline.md`):

- **TunerV2 s1**: q=5 → median 54.26, q=95 → median 63.71. **Range
  = 9.45 zq.** Almost everything is ~63.
- **TunerV2 s2**: q=5 → median 32.96, q=95 → median 71.53. **Range
  = 38.57 zq.** Better range but max(95) is only 71 — capped at the
  anchor target's neighborhood.
- **TunerV2 s3**: similar saturation pattern.

Compare to today's Tuner: q=5 → ~30, q=95 → ~92, **range = ~62 zq.**

Strict-mono violation rates (from `qsweep_v2_vs_baseline.md`):

| Bake | violations / 900 | strict_mono | tied_rate |
|---|--:|---:|---:|
| baseline_tuner | 65 | 0.9278 | 0.0044 |
| tuner_v2_s1 | 76 | 0.9156 | 0.0000 |
| tuner_v2_s2 | 129 | 0.8567 | 0.0011 |
| tuner_v2_s3 | 149 | 0.8344 | 0.0000 |

Only s1 stays close to baseline mono. s2 and s3 are 5–9 pp worse.

## What the cross-codec numbers tell us

The recipe DID achieve cross-codec consistency at T=70 (2.04 vs
baseline 5.00, well under the 2.5 gate). But T=63 stayed at 6.59 —
essentially unchanged from baseline 6.68. Why?

The answer is in the `mean_dist_from_target` rows:

| Target | baseline_tuner | TunerV2 s1 | TunerV2 s2 | TunerV2 s3 |
|---|---:|---:|---:|---:|
| T=30 | 8.10 | 24.79 | 9.21 | 11.99 |
| T=50 | 2.26 | 5.01 | 3.39 | 4.18 |
| T=63 | 0.90 | 0.04 | 0.87 | 1.41 |
| T=70 | 0.65 | 6.31 | 0.31 | 1.78 |
| T=80 | 0.87 | 16.31 | 8.57 | 11.78 |
| T=90 | 0.89 | 26.31 | 18.57 | 21.78 |

s1's dist_from_target at T=63 is 0.04 — every codec hits 63
exactly. But the corresponding butter is still 5.61 because the
bake's output is now SO insensitive to codec choice that the codecs
all collapse to picking the maximum q (q=95) AND that q produces
~63 regardless of codec — but the underlying decoded images are
still butter-different by ~5-6 units. The anchor squashed the
bake's response curve so flat that codec-distinctness can't be
expressed in the score.

In other words: the anchor produced consistency in the wrong way —
not "different codecs land at similar JND-distance from source when
the score is the same" but "the score is constant regardless of
codec or q above some floor."

## What still works: the data + trainer infrastructure

The anchor parquet
(`/mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet`,
9,373 rows = 504 KonJND JPEG + 504 KonJND BPG + 8,365 safesyn
ssim2≈63) is a useful future asset. The trainer's
`--anchor-loss-weight / --anchor-target-score / --anchor-step-p`
flags are wired and tested.

A follow-on cycle could:

1. **Sweep `--anchor-loss-weight` ∈ {0.01, 0.03, 0.10, 0.30}** at
   `--anchor-step-p 0.10` to find the weight where the anchor
   constrains cross-codec spread WITHOUT collapsing the dynamic
   range. The current 1.0 weight is ~30× too aggressive given the
   `pairs_per_epoch=50000` cardinality.
2. **Use anchor as a regularizer, not a primary loss.** A weight
   of ~0.001-0.01 likely preserves rank fidelity while nudging
   the score=63 anchor.
3. **Constrain the anchor target's force per-row, not globally.**
   Currently `dL/dy = 2·w·row_w·(y - target)`. A bounded variant
   like `2·w·row_w·tanh((y - target)/τ)` with τ=10 prevents the
   anchor from drowning out non-anchor signal when y is far from
   target.
4. **Per-codec anchors instead of cross-codec averaging.** The
   anchor pool right now treats every codec's PJND as the same
   target=63. A future variant could compute per-codec anchor
   targets (e.g. KonJND-JPEG mean PJND score → some value, KonJND-
   BPG mean PJND score → another) — but the CID22 paper's
   universal PJND=63 calibration is the simpler default.

## Data lineage (preserved)

| Path | Status |
|---|---|
| `2026-05-19-jnd-anchors/anchors_372col.parquet` | 9373 rows, built from canonical val/konjnd + train/safesyn |
| `scripts/v_next/build_jnd_anchors.py` | builder script |
| `zensim-validate/src/mlp_train.rs::AnchorRows` | trainer public API |
| `zensim-validate/src/mlp_train.rs::train_mlp_with_tv_anchored` | trainer entrypoint |
| `zensim-validate/src/bin/zensim_mlp_train.rs --anchor-parquet/--anchor-loss-weight/--anchor-target-score/--anchor-step-p` | CLI flags |
| `scripts/v_next/run_tuner_v2_seed.sh` | recipe (`-w 1.0 -p 0.10`) |
| `scripts/v_next/cross_codec_jnd_eval_bake.py` | eval driver |
| `scripts/v_next/aggregate_tuner_v2.py` | 3-seed aggregator |

All committed under this experiment's WIP change. NO ship — production
`PreviewV0_5Tuner` remains the dial profile.

## Reproduction

```bash
cd ~/work/zen/zensim--exp-tuner-v2
cargo build --release --bin zensim_mlp_train -p zensim-validate
python3 scripts/v_next/build_jnd_anchors.py
for s in 1 2 3; do
  bash scripts/v_next/run_tuner_v2_seed.sh $s
done
# Eval (~10 min)
for s in 1 2 3; do
  ./target/release/bake_verdict \
    --bake /mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19/tuner_v2_s${s}_h128.bin \
    --output /mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19/verdict_s${s}.md
  python3 scripts/v_next/cross_codec_jnd_eval_bake.py \
    --bake /mnt/v/zen/zensim-eval/exp_tuner_v2_2026-05-19/tuner_v2_s${s}_h128.bin \
    --label tuner_v2_s${s} --bake-post clamp
done
python3 scripts/v_next/aggregate_tuner_v2.py
```

Wall time: ~22 min training (3 seeds parallel on 7950X) + ~12 min eval
(3 cross-codec runs sequential, ~1 min bake_verdict for all 3).

## Decision

**No `PreviewV0_5TunerV2` ship.** Today's `PreviewV0_5Tuner` remains
the dial profile. The anchor-loss infrastructure is committed for
future exploration of fundamentally different cross-codec strategies
(per-codec calibration, butter-direct training, or encoder swap).

The 2 of 4 criteria passing in round 2 is encouraging — CID22 stays
near baseline AND butter T=70 improves 3.5× — but the load-bearing
T=63 criterion (which is the user-visible PJND operating point) needs
a fundamentally different approach.
