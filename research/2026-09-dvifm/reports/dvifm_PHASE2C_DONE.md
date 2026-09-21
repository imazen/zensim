# DVIFM Phase 2c — DONE (2026-09-19)

Commits (jj, no push):
- 4d6946db preregistration (before any fit)
- 8ea2e7ef driver extensions: in-driver bake stamping, codec admission reuse-verified cell copy, per-family codec panels, screen2c report
- <record commit> benchmarks/dvifm_screen2c_2026-09-19.{md,json,pointer.md}, board append, gates outcome

Verdicts:
- Q1 (human rows, 8,327 fit / 3,125 dev / 436 dev2): NOT-A-SUBSTITUTE.
  Δ(y60dvifm − y60) = −0.00454 (2·SD/√10 = 0.0015), signs +0/−10.
  Δ(y60dvifm − y60perm30) = +0.00795 (+10/−0, real signal).
  GAP CLOSED = −2.46 per-ref / −1.27 pooled (widened the gap).
  dvifm30 alone = 0.857 per-ref mean vs perm30 0.005 (Δ +0.852, +10/−0).
  dev2 ordering identical: basic228 0.8715 > y60 0.8643 > y60dvifm 0.8480.
- Q2 (codec proxy labels — score_ssim2, NOT human): NEGATIVE.
  Δ(basic228dvifm − basic228perm30) = −0.0021, |Δ| ≤ 2·SD/√10 = 0.0085 (+5/−5).
  Δ(basic228dvifm − basic228) = +0.0047 noise (+6/−4).
  y60 pair on codec: NOT-A-SUBSTITUTE (−0.0132, +0/−10).
  Per-codec within-ladder (E=100, mean over seeds):
    jpeg: y60dvifm−y60 = −0.1105 ± 0.0146 (largest loss — blockiness fails first on JPEG)
    webp/jxl: saturated at 1.0000 both arms (Δ exactly 0, uninformative)
    avif_svt: +0.0183 ± 0.0104 (small, noisy)
  Ancillary: y60 > basic228 on codec proxies (+0.0191 per-ref, +8/−2).

What was NOT done: no E=150 (no Q2 arm rising; Q1 flag = dvifm30 standalone,
non-decision arm), no frozen EVAL/test/terminal (6 codec test families
excluded, never opened), no CID22/AIC/KonJND-val/KonFiG-test/KADID{7,9}/
holdout reads, no optimization, no qualification, no push. dvifm_block
stays default-OFF.

Evidence: /mnt/v/output/zensim/dvifm-screen2c-2026-09-19/
  q1-human/ (120 cells; 40 reused byte-identical 2b bakes)
  q2-codec/ (100 cells; 494-row panel extracted in 70.8s < 500-row gate)
  segments/ codec-{train,eval}-{segment,admission}.json (373/121 rows)
  admit_2c.py, recipe_q1.json (448ac697…), recipe_q2.json (320b732b…)
