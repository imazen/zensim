# EXP-CROSS-CODEC-V2 — methodology

**Date:** 2026-05-19
**Status:** In progress.
**Prior:** EXP-CROSS-CODEC-METRIC W=1.0 seed=1 shipped as
`PreviewV0_5CrossCodec` (opt-in). T=63 6-img butter 4.82 / 20-img
5.52 — falls short of strict <2.5 gate. Seed 2 hit 2.81/2.97 but
rank-collapsed (KADID 0.308, TID 0.367).

## Hypothesis

Three substrate changes + one regularization change should tighten
the cross-codec consistency without triggering the rank-collapse
mode that killed seed 2 at W=3.0:

1. **Tighter butter gap (0.3 vs 0.5)** — each equivalence pair is
   a more precise statement of "perceptually equivalent across
   codecs at this butter level". Cleaner training signal.
2. **Avif↔X oversampling (2× row weight)** — the original pool had
   only ~25% avif pairs (rejected pool: avif's q=5 lower bound
   limits the upper butter range). Doubling avif-pair weight
   compensates for the smaller pool size at training time.
3. **30 butter levels (vs 20)** — extends from butter 0.3 to 12.0
   with finer granularity, giving more equivalence-pair samples
   per source.
4. **W central sweep + 3-seed CI** — train at W ∈ {1.0, 1.5, 2.0}
   to find the new operating point. W=3.0 collapsed in the original
   sweep; W=2.0 may also collapse on the new data (need to verify).

## Falsification

If no candidate (across W ∈ {1.0, 1.5, 2.0} × seeds {1, 2, 3})
achieves **all** of:
- T=63 6-img cross-codec butter < 3.0 (relaxed from < 2.5 strict
  to < 3.0 nice-to-have per Phase 2 directive)
- CID22 SROCC ≥ 0.86 (within 0.02 of current ship 0.880)
- KADID SROCC ≥ 0.70 (within 0.10 of current ship 0.800)
- TID SROCC ≥ 0.72 (within 0.10 of current ship 0.822)

then V2 is falsified and the cross-codec trail keeps its opt-in
W=1.0 seed=1 ship. The mechanism may need joint butter+features
training or the documented rank-preservation regularizer (deferred
to V3 if V2 fails).

## Cost ceiling

- Build tighter equivalence parquet: ~5 min (done).
- 5 trainer runs in parallel (seed×W): ~25-30 min wall.
- bake_verdict eval on canonical val parquets: ~3-5 min × 5 bakes.
- Cross-codec consistency eval at T=63 (n=20): ~15 min × N best
  candidates.
- Total: ~1.5 hr.

## Reporting panel

- **Primary metric (ship decision):** T=63 mean pairwise
  butter_max on the 6-img feature subset AND 20-img cohort.
- **Secondary metrics (don't catastrophically regress):**
  - CID22 aggregate SROCC vs W=1.0 ship (tolerate −0.02).
  - KADID/TID/KonJND aggregate SROCC vs W=1.0 ship (tolerate
    −0.10 on any single corpus per the cross-codec trail policy).
  - AIC-3 aggregate SROCC vs W=1.0 ship (tolerate −0.02).

## Substrate changes vs EXP-CROSS-CODEC-METRIC

| Aspect | METRIC (v1, shipped W=1.0) | V2 |
|---|---|---|
| Equivalence parquet | `cross_codec_equivalence.parquet` (57,972 pairs, gap ≤ 0.5, 20 butter levels 0.5..12.0, no avif rebalance) | `cross_codec_equivalence_tight_v3.parquet` (68,788 pairs, gap ≤ 0.3, 30 butter levels 0.3..12.0, avif rows 2× row-weight) |
| Trainer flags | `--cross-codec-eq-weight 1.0` (single point) | `--cross-codec-eq-weight {1.0, 1.5, 2.0}` × seeds {1, 2, 3} |
| Other | tuner-v2 recipe + KonJND anchor (target=63, weight=1.0) | unchanged |

## Data pipeline

1. **Re-build tighter equivalence parquet** via
   `scripts/v_next/build_cross_codec_equivalence.py --max-pair-gap
   0.3 --avif-oversample 2.0 --n-levels 30 --butter-low 0.3
   --butter-high 12.0 --out cross_codec_equivalence_tight_v3.parquet`.
   See `benchmarks/build_equiv_v3_2026-05-19.log`.
2. **Training driver:** `scripts/v_next/run_cross_codec_v2_seed.sh
   <seed> <W>`. Composes with the existing tuner-v2 recipe +
   anchor + cross-codec-eq parquet.
3. **Eval:**
   - `bake_verdict --bake <bake> --corpora cid22,kadid,tid,konjnd,aic3`
     for the full Mohammadi panel on canonical val parquets.
   - `scripts/v_next/run_cross_codec_metric_eval.sh <bake> <label>`
     for T=63 cross-codec butter consistency (uses
     `cross_codec_consistency.py` per the v1 ship eval pipeline).

## Ship policy

- **Pass strict (T=63 butter < 2.5, all secondary gates clear)**:
  rotate `PreviewV0_5CrossCodec` to V2 (swap include_bytes in
  `zensim/src/profile.rs`).
- **Pass relaxed (T=63 butter in [2.5, 3.0])**: rotate
  `PreviewV0_5CrossCodec` to V2; document the gap.
- **Fail (T=63 butter ≥ 3.0 or any secondary gate breach)**:
  document falsification, keep W=1.0 ship.

## Outstanding work (deferred to V3 if V2 fails)

- **Rank-preservation regularizer** (not implemented in V2 trainer
  yet). Per phase 2 directive: add a `--cross-codec-rank-preserve-weight`
  flag that adds `−log(sigmoid(score_B − score_A))` to each
  equivalence pair gradient, with the butter-derived
  ordering hint (lower-butter codec should score higher). This
  would prevent seed-2-style rank-degenerate collapse at higher W
  values. Substantial trainer change; defer to V3 if V2 substrate
  changes alone don't close the gap.
- **Joint butter+features training**: fold butter into the loss
  directly via a learnable adapter, rather than using butter only
  as a pivot at data construction time.
- **Higher per-pair weight cap**: currently capped at 20.0 (after
  avif-2x = 40.0). May benefit from higher cap for tight-gap pairs.
