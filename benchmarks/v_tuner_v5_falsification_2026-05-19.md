# EXP-CROSS-CODEC-V5 falsification — 2026-05-19

**Status:** FALSIFIED on median-range gate / SHIP-DECISION-PENDING.
3 of 3 V5 seeds pass 5 of 6 ship gates; same kind of "near-miss" as
V4b's 4-of-5 result (which was deferred to user).

## Verdict

V5 piecewise multi-band anchor (6 bands per source × codec) achieves
the foundational property — **spectacular cross-codec parity across
the entire output range**, not just at PJND — but **does NOT restore
the median q-sweep dynamic range** that V4 collapsed. All 3 V5 seeds
fail the same gate that all 6 V4b bakes failed: `range ≥ 50` (using
the median-range definition that produced V4b's 35.25 result).

V5 wins decisively over V4b on every OTHER gate, including:
- **Monotonicity**: V5 = 0.9578–0.9767 (best cc4v5_s1 = 0.9767) vs
  V4b best = 0.9578. Best V5 seed beats best V4b by 1.9 pp.
- **Tied rate**: V5 = 0.0000 (all 3 seeds) vs V4b best = 0.0000
  (3 of 6 bakes). Same.
- **Cross-codec parity at EVERY anchor band** (V4 had only T=63
  parity): V5 cc_std_median ≤ 1.04 at every band ≤ gate 5.0;
  V4 had no multi-band gate to compare.
- **T=63 cross-codec consistency**: V5 mean butter_p3 = 1.48–1.53;
  V4b best was ~1.5; tied.

But the median-range failure mode is identical to V4b: the network
clusters its outputs in a [37, 70] band on the JPEG q-sweep instead
of spanning [10, 95]. **V5 piecewise anchor was hypothesized to fix
this, but it does not.**

## Mechanism — why piecewise didn't restore range

The anchor parquet was correctly built with 18,459 rows at 6 distinct
butter bands × 4 codecs × ~770 sources/band (with codec coverage
filtering). The trainer correctly read per-row `target_score` from the
parquet (verified at runtime: `PER-ROW target_score: min=10.0
median=63.0 max=90.0`). The per-row target was correctly applied in
the anchor MSE step.

**However**, the anchor pressure was structurally too weak to drive the
network's outputs to the full [10, 90] band targets:

- `anchor_loss_weight = 0.05`, `anchor_step_p = 0.15` → ~7.5% of pair
  steps trigger an anchor step, each contributing weight 0.05.
- The cross-codec equivalence loss runs at `weight=1.0, step_p=0.10`
  → 10% of pair steps trigger an equiv step at weight 1.0.
- Per-pair MSE on safesyn `mix_cv40_iw60` target dominates the
  remaining 82.5% of steps.

The cross-codec-eq + per-pair MSE jointly favor a compressed output
range (predictions cluster near the mix_cv40_iw60 distribution
center), while the anchor steps only nudge each (image, codec) row
toward its band-specific target with a 0.05-weighted gradient.

Per-band achievement (cc4v5_s1):

| butter_target | target_score | achieved_mean | delta |
|---:|---:|---:|---:|
| 0.30 | 90.0 | 70.61 | −19.4 |
| 0.80 | 75.0 | 68.26 | −6.7 |
| 1.50 | 63.0 | 61.14 | −1.9 |
| 2.50 | 45.0 | 52.73 | +7.7 |
| 4.00 | 25.0 | 45.34 | +20.3 |
| 6.00 | 10.0 | 40.50 | +30.5 |

The network's outputs are compressed toward [40, 70] regardless of
which band the anchor row targets. The cross-codec parity property
(all 4 codecs predict similar values within each band) is preserved
because the equiv loss is large — but those similar values are
nowhere near the target_score endpoints.

## Why the range gate fails (median_range = 30.7, gate = 50)

Same as V4b. Output is pinned in [37.7, 70.2] median range on the
JPEG 50-image × 19-q sweep. Hidden behind this:
- **full_range** (max across all q − min across all q) = 71.67 — this
  IS above 50. So the network DOES express scores between 0 and 70 in
  outliers (q=5 some images scored 0.22).
- **median_range** (p50 at q=95 − p50 at q=5) = 30.73. This is the
  V4b-defined metric and the gate measure.

The shift from V4b: V5's median band is [37, 70], V4b's was [11, 47].
V5 shifted higher (closer to the band targets) but still 30.73-wide.

## Per-seed full Mohammadi panel (held-out validation)

| Bake | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| cc4v5_s1 | **0.8818** | 0.6163 | 0.2583 | 0.4168 | 0.7827 |
| cc4v5_s2 | **0.8889** | 0.7365 | **0.7940** | 0.3033 | 0.7853 |
| cc4v5_s3 | 0.8761 | 0.6933 | 0.7067 | 0.3633 | 0.7779 |

Tuner trail explicitly doesn't have SROCC ship gates (per CLAUDE.md
SOTA_TRAILS); panel reported here for completeness only.

## All 6 ship gates per V5 bake

| Bake | mono ≥ 0.9378 | tied ≤ 5% | medRange ≥ 50 | T63 butter < 2.5 | cc_std PJND ≤ 5 | cc_std ALL bands ≤ 5 | passed |
|---|:---:|:---:|:---:|:---:|:---:|:---:|---:|
| cc4v5_s1 | PASS (0.9767) | PASS (0.0000) | **FAIL (30.73)** | PASS (1.53) | PASS | PASS (max 1.04) | 5 of 6 |
| cc4v5_s2 | PASS (0.9578) | PASS (0.0000) | **FAIL (32.17)** | PASS (1.51) | PASS | PASS (max 1.04) | 5 of 6 |
| cc4v5_s3 | PASS (0.9700) | PASS (0.0000) | **FAIL (31.40)** | PASS (1.48) | PASS | PASS (max 1.04) | 5 of 6 |

## V6 direction proposals

Three independent levers, ranked by likelihood of restoring range
without breaking cross-codec parity:

1. **Stronger anchor pressure** (top candidate). Raise
   `anchor_loss_weight` from 0.05 → 0.5–1.0 with `anchor_step_p` 0.30
   (V5 used 0.15). At 6 bands × 0.5 weight × 0.30 step, total anchor
   gradient bandwidth is 12× V5's. Hypothesis: at higher anchor weight,
   the network reaches the band targets (score=90 at butter=0.3,
   score=10 at butter=6.0), and the resulting [10, 90] spread on the
   anchor data forces an equally wide output range on the q-sweep.
   Risk: too high a weight may destroy the cross-codec parity that V5
   achieves (the same trade-off V4→V4b explored, but in the OPPOSITE
   direction).

2. **Smaller tanh scale** (parallel candidate). V5 used scale=15.0.
   Drop to scale=5.0 — narrower linear region forces extreme y_pre
   values to map to score endpoints. At scale=5.0 the σ(y/5) function
   reaches 99% saturation at y_pre=±15, so the network can express
   [1, 99] without exploding y_pre.

3. **Tighter cross-codec equiv loss anchored to butter_diff scale**.
   The cross-codec-eq loss currently uses
   `(y_a − y_b)² · w` regardless of butter level. Replace with a band-
   aware variant that only enforces equivalence when the two codecs
   are close in butter_pnorm3 (e.g., |Δbutter| < 0.5 → tight equiv;
   larger Δ → relax). This avoids the over-collapse mechanism while
   keeping the cross-codec consistency at each band.

4. **Multi-stage training**. Phase 1: train safesyn pair MSE + per-row
   anchor only (no equiv loss) until range hits 50. Phase 2: freeze
   feature scaler, add equiv loss, fine-tune for cross-codec parity.
   The two pressures don't compete in the same loss step.

## Ship decision

3 bakes pass 5 of 6 gates. Per task brief instructions:
"If V5 candidate is 'close' (passes 5 of 6 gates), surface to user as
ship-decision-pending (like V4b's 4-of-5 result that we deferred)."

**Action**: surface to user; do NOT auto-ship as `PreviewV0_5TunerV2`.
The cross-codec multi-band parity is a real, foundationally important
property (decisive 6/6 pass at cc_std ≤ 1.04). The range gap to
V4b is small enough that user may prefer V5 anyway for the cross-codec
property, accepting the 30-unit median range.

If user opts to ship cc4v5_s1 (best mono of the 3 seeds), the bake
bytes are at
`/mnt/v/zen/zensim-eval/exp_cross_codec_v5_2026-05-19/cc4v5_s1.bin`.

If user opts to falsify and pursue V6: candidate is the
`anchor_loss_weight=0.5 + scale=5.0` joint sweep (V6 plan above).
