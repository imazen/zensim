# V0_24 v2 dssim co-training — also failed (-0.067 CID22), running control

**Trained 2026-05-12 23:20Z. Eval 23:22Z.**
**Bake**: `/tmp/zensim_loop/bakes/v0_24_v2_dssim03_2026-05-13.bin`

## Result

V0_24 v2 was launched with the same recipe as V0_16, with the
critical TV-regularizer issue from v1 FIXED (TV pairs constructed
from real synth-CSV codec/quality values; 104,319 adjacent-q pairs
active at weight=20).

Despite TV being properly active, V0_24 v2 still regressed:

| Metric | CID22 SROCC (n=4292) |
|---|---:|
| V0_2 (linear baseline) | 0.8676 |
| **V0_24 v1 (no TV)** | 0.8315 |
| **V0_24 v2 (TV active + dssim)** | **0.8254** ⚠ worse than v1 |
| V0_16 (ship) | **0.8919** |
| fast-ssim2-gpu | 0.8895 |

V0_24 v2 is **-0.067 below V0_16** — actually slightly worse than
v1 (-0.060). Both v1 and v2 are clearly below V0_16 by the same
ballpark.

Hypothesis update: **the regression is NOT primarily recipe-drift**
(v2 has TV but still regresses). Two possibilities remain:

1. **dssim_weight=0.3 is too aggressive** — the dssim auxiliary
   loss is pulling the network away from the ssim2 ranking
   objective enough to hurt CID22.
2. **The training data differs from V0_16's**: V0_16 may have
   used additional supervision sources (KADID + TID), KonJND
   anchor, different concordance filtering. The recipe stored in
   our docs is incomplete.

## Control run (V0_25)

Launched V0_25 = "same as V0_24 v2 except `--dssim-weight 0.0`".
If V0_25 lands near V0_16's 0.8919, the regression is squarely
caused by dssim_weight=0.3. If V0_25 also lands below 0.88, then
recipe drift remains and we need to surface V0_16's true training
command (KADID/TID/KonJND inputs).

Run: `/mnt/v/zen/zensim-training/2026-05-07/runs/20260512T172428_v0_25_control_dssim0_seed1_2026-05-13`
Log: `/tmp/zensim_loop/v0_25_control_train.log`. ETA ~10 min.

## Per-band V0_24 v2

| Band | n | V0_2 | V0_24 v2 | ssim2 |
|---|---:|---:|---:|---:|
| B0 (<50)           | 324  | 0.4072 | 0.3710 | 0.4418 |
| B1 [50,65)         | 1010 | 0.4119 | 0.3368 | 0.4694 |
| B2 [65,90)         | 2915 | 0.7359 | 0.6713 | 0.7722 |
| B3 (≥90)           | 43   | 0.0035 | 0.0769 | 0.1121 |
| Near-PJND [58,68]  | 787  | 0.3551 | 0.2994 | 0.3908 |

V0_24 v2 loses in every band on CID22.

## Implications

If V0_25 (control) shows V0_16-equivalent performance, then:
- The dssim co-training approach with `--dssim-weight 0.3` is too
  aggressive. Need lower weight (try 0.05) or different
  formulation (rank-based instead of MSE).

If V0_25 also regresses:
- V0_16's recipe wasn't fully captured in our docs; need to
  archeologically reconstruct from the actual V0_16 run dir.
