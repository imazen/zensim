# V_tuner-v3 EVAL — 6-bake matrix RESULTS (2026-05-19)

Per `benchmarks/v_tuner_v3_methodology_2026-05-19.md`. **Verdict:
falsified.** See
`benchmarks/v_tuner_v3_falsification_2026-05-19.md` for the full
analysis.

## Sweep grid

| Seed | W (cross-codec-eq) | Mono-reg | Rank-preserve | DR floor | σ_thresh |
|---|---:|---:|---:|---:|---:|
| 1 | 0.5 | 5.0 | 0.2 | 0.2 | 15.0 |
| 1 | 1.0 | 5.0 | 0.2 | 0.2 | 15.0 |
| 2 | 0.5 | 5.0 | 0.2 | 0.2 | 15.0 |
| 2 | 1.0 | 5.0 | 0.2 | 0.2 | 15.0 |
| 3 | 0.5 | 5.0 | 0.2 | 0.2 | 15.0 |
| 3 | 1.0 | 5.0 | 0.2 | 0.2 | 15.0 |

Plus held-fixed: `--dynamic-range-step-p 0.05 --dynamic-range-probe-n 40`.

## Tuner-trail gate ship table

Gates: `strict_mono ≥ 0.9378`, `tied ≤ 5 %`, `range ≥ 50`,
`T=63 butter_max OR butter_p3 < 2.5`.

### Pre-affine (raw output)

| Bake | raw q5 med | raw q95 med | raw range | α (calibration) | β (calibration) |
|---|---:|---:|---:|---:|---:|
| baseline_tuner | 4.96 | 94.64 | 89.68 | 0.022 | 1.004 |
| cc4v3_s1_w0_5 | 55.07 | 68.08 | 13.01 | -375.961 | 6.918 |
| cc4v3_s1_w1_0 | 57.79 | 67.21 | 9.42 | -547.134 | 9.554 |
| cc4v3_s2_w0_5 | 50.99 | 67.66 | 16.67 | -270.291 | 5.399 |
| cc4v3_s2_w1_0 | 58.39 | 66.99 | 8.60 | -606.058 | 10.465 |
| cc4v3_s3_w0_5 | 52.18 | 66.45 | 14.27 | -324.096 | 6.307 |
| cc4v3_s3_w1_0 | 56.37 | 66.77 | 10.40 | -482.817 | 8.654 |

### Post-calibration

| Bake | strict_mono | tied | q5 med | q95 med | range | T=63 butter_max | T=63 butter_p3 | mono ≥ 0.9378 | tied ≤ 5% | range ≥ 50 | xc < 2.5 | ALL |
|---|---:|---:|---:|---:|---:|---:|---:|:-:|:-:|:-:|:-:|:-:|
| baseline_tuner | 0.9278 | 0.0044 | 4.96 | 94.64 | 89.68 | (8.07) | (2.11) | ✗ | ✓ | ✓ | ✗ | (current ship) |
| cc4v3_s1_w0_5 | 0.8622 | 0.0556 | 4.99 | 94.98 | 89.99 | 6.76 | 2.57 | ✗ | ✗ | ✓ | ✗ | FAIL |
| cc4v3_s1_w1_0 | 0.8933 | 0.0822 | 5.00 | 94.97 | 89.97 | 6.09 | **2.26** | ✗ | ✗ | ✓ | ✓ | FAIL |
| cc4v3_s2_w0_5 | **0.9100** | 0.0278 | 5.01 | 95.00 | 89.99 | 7.70 | 2.91 | ✗ | ✓ | ✓ | ✗ | FAIL |
| cc4v3_s2_w1_0 | 0.8233 | 0.0989 | 5.01 | 95.03 | 90.02 | 6.10 | **2.21** | ✗ | ✗ | ✓ | ✓ | FAIL |
| cc4v3_s3_w0_5 | 0.8711 | 0.0522 | 5.00 | 94.99 | 89.99 | 8.99 | 3.27 | ✗ | ✗ | ✓ | ✗ | FAIL |
| cc4v3_s3_w1_0 | 0.8733 | 0.0600 | 5.03 | 94.97 | 89.94 | 7.00 | 2.61 | ✗ | ✗ | ✓ | ✗ | FAIL |

## Cross-corpus context (Mohammadi panel CID22 SROCC)

| Bake | CID22 SROCC |
|---|---:|
| baseline_tuner | 0.879 |
| cc4v3_s1_w0_5 | 0.853 |
| cc4v3_s1_w1_0 | **0.883** |
| cc4v3_s2_w0_5 | 0.852 |
| cc4v3_s2_w1_0 | 0.863 |
| cc4v3_s3_w0_5 | 0.882 |
| cc4v3_s3_w1_0 | 0.872 |

V3 bakes preserve cross-corpus CID22 ranking (0.852-0.883 vs baseline
tuner 0.879). The rank-preserve mechanism does what it's designed to
do for cross-corpus rank — but doesn't help the within-curve
monotonicity that the Tuner trail gate measures.

## Range gate: SOLVED across all V3 candidates

The V2 collapse mode (range 0.10-0.92) is gone. Every V3 candidate has
post-calibration range 89.94-90.02 — the σ-floor probe + rank-preserve
machinery successfully prevent constant-output failure. **The V3
architectural counterweights work as designed.**

## Monotonicity gate: failed on all V3 candidates

Best V3 = cc4v3_s2_w0_5 at 0.9100 strict mono. Baseline = 0.9278. Gate
= 0.9378.

Root cause analysis in `benchmarks/v_tuner_v3_falsification_2026-05-19.md`
§ "Why V3 broke monotonicity":

1. Raw output range tight (10-13 score units) → β=5-10× affine
   amplification.
2. Strong mono-reg=5.0 fights σ-floor when within-curve spread is
   small.
3. Pre-affine jitter ~0.05-0.2 raw units × β ≈ 0.5-2 score units →
   visible tied + non-monotonic at adjacent q.

## Cross-codec gate: passes on 2/6 (W=1.0 seed=1, seed=2)

cc4v3_s1_w1_0 (butter_p3 = 2.26) and cc4v3_s2_w1_0 (butter_p3 = 2.21)
**pass** the < 2.5 cross-codec gate. The V3 mechanism produces real
cross-codec consistency (vs V2's collapse-artifact "consistency").

This is the FIRST candidate family in the EXP-CROSS-CODEC- chain to
achieve cross-codec gate ≤ 2.5 with non-collapsed dynamic range. Per
SOTA_TRAILS.md cross-codec-trail § (a separate trail from Tuner), this
might rotate `PreviewV0_5CrossCodec` (currently V2 W=1.0 seed=1 at
butter_p3 5.52 / mono N/A) — but that requires evaluating against the
cross-codec trail's specific gate, which is OUT OF SCOPE for this
session's Tuner-trail focus.

## Decision

**No `PreviewV0_5TunerV2` ship.** V3 line closed for Tuner trail at
the (mono-reg=5.0, rank-preserve=0.2, σ-floor=0.2 σ=15) hyperparam
combination. See falsification doc for V4 direction proposals
(V4-A: per-curve σ substrate, recommended first).

## Reproduction

See `benchmarks/v_tuner_v3_falsification_2026-05-19.md` § Reproduction
for the full command sequence (~40 min compute on 7950X).
