# CORRECTION (from parent, 2026-05-19)

The user just clarified: **use `butteraugli_pnorm3_gpu` NOT `butteraugli_max_gpu`** for cross-codec equivalence pairing.

Reasoning: butter_max is hyper-sensitive to single-pixel outliers (single edge pixels, isolated artifacts). butter_pnorm3 (3-norm aggregation) balances peak with mean — gives a smoother, content-aware quality measure. Equivalence pairs picked via pnorm3 will be more perceptually meaningful.

Both metrics are in the R2 sidecars per DATA_PROVENANCE.md:
- `butteraugli-max-gpu`  ← DON'T use for equivalence
- `butteraugli-pnorm3-gpu` ← USE this

**Replace in your equivalence-pair builder**:
```python
# WAS:
# pivot_butter = row['butteraugli_max_gpu']
# IS NOW:
pivot_butter = row['butteraugli_pnorm3_gpu']
```

Butter-level grid: log-spaced butter_pnorm3 ∈ [0.3, 5.0] is more sensible than the [0.5, 30] for butter_max (pnorm3 ranges roughly 5-10× smaller). Pick K=24 log-spaced levels in pnorm3 space.

Reference points (rough, from session memory):
- butter_pnorm3 ≈ 0.5 → near-lossless
- butter_pnorm3 ≈ 1.5 → PJND threshold (the CID22 paper anchor at ssim2 ≈ 63 corresponds to roughly this butter_pnorm3 level)
- butter_pnorm3 ≈ 3.0 → noticeably degraded
- butter_pnorm3 ≈ 5.0 → clearly distorted

Use these to set the butter-level grid sensibly.

All other phases of the brief remain valid.
