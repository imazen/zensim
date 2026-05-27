# V47 masked-monotone — STRICT fixes the correctness defect (2026-05-27)

First retrain using the per-feature sign mask (300 pin≥0 / 72 free, from
benchmarks/feature_sign_mask_2026-05-26.tsv). Two variants on the
V32-faithful recipe + --monotone-cbc.

## Blur-ladder monotonicity (OOD synthetic, raw post-spline)

| Bake | inversions | above-identity | identity=max? |
|---|--:|--:|---|
| V39 (shipped A) | 27 | 31 | NO (blur scores ≥ identity) |
| **V47-strict** | **0** | **0** | **YES** ✓ |
| V47-partial | 0 | 0 | (degenerate, all ~constant) |

V47-strict is **monotone-by-construction** — the V39 correctness defect
(blur ≥ identity on OOD content) is FIXED. identity is the unique max at
every content; score strictly decreases with blur.

## Held-out panel (bake_verdict)

| Corpus | V47-strict | V47-partial | V39 |
|---|--:|--:|--:|
| CID22 | 0.8547 | 0.0 (dead) | 0.8793 |
| KADID | 0.8030 | 0.0 | 0.9251 |
| TID | 0.7965 | 0.0 | 0.9317 |
| KonJND | **0.4850** | 0.0 | 0.4197 |
| AIC-3 | 0.7700 | 0.0 | 0.8023 |
| AIC-4 | 0.8902 | 0.0 | 0.9051 |

- **STRICT**: CID22 0.855 (−0.024 vs V39, above the 0.85 G7 floor),
  KonJND **+0.065** (monotonicity helps near-lossless rank), AIC-4 −0.015.
  Cost: KADID −0.122, TID −0.135 — the analytic-distortion-ranking signal
  carried by the 72 dropped sign-flip features. Trained healthily
  (val 0.90, no collapse).
- **PARTIAL is degenerate** — the 72 free features let the optimizer
  cancel the spread; raw preds collapse to ~50 (range 0.05) → spline maps
  everything to ~0 → all-zero panel. Out.

## Remaining: dial calibration (G1)

STRICT ranks well but its dial is COMPRESSED on real content: `y_pre`
clusters near 0 (tanh-pin score ~50) and only spreads wide on extreme
synthetic blur. The auto-spline fit only 2 degenerate knots
(pred≈49.97/49.99) and maps the val distribution to a negative band
(G1 p5=−42, p95=−15). The bake is monotone with real spread — it needs
the dial stretched to [0,100].

Fix (in flight, v48): retrain strict with a SMALLER `--tanh-output-head-scale`
(30 → ~5) so the narrow real-content `y_pre` spread maps across the full
[0,100] dial. Monotone amplification preserves the correctness guarantee.

## Status vs the goal

The #1 goal — Profile::A monotone-by-construction (never blur > identity)
— is ACHIEVED IN PRINCIPLE by V47-strict. Two items before it could
ship as A: (1) dial calibration (v48 tanh-scale), (2) the KADID/TID rank
cost (−0.12/−0.14) is the price of strict monotonicity — a user call,
since CID22 (the compression gold standard) stays competitive and KADID/
TID are integrity guards, not the primary compression target.
