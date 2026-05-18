# V_24-α sweep table (vs V_22-mix-LARGE+iwssim seed=3)

**Generated**: 2026-05-18, branch `feat/v24-alpha-sweep`.
**Baseline B**: `v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128.bin` panel SROCC
CID22=0.8323, KADID=0.9677, TID=0.9729, KonJND=0.8928, AIC-3=0.7831.
Baseline weighted score (CID22 + 0.5·mean(KADID,TID) + 0.5·KonJND + 0.25·AIC-3) = 1.9596.

| α | CID22_A | ΔCID22 | KADID_A | ΔKADID | TID_A | ΔTID | KonJND_A | ΔKonJND | AIC-3_A | ΔAIC-3 | score_A | Δscore | Adec / Bdec / pND / tied / noisy | Winner |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0.025 | 0.8687 | +0.0365 | 0.9119 | −0.0557 | 0.8900 | −0.0829 | 0.7988 | −0.0940 | 0.7878 | +0.0046 | 1.9155 | −0.0441 | 3/17/3/6/1 | B |
| 0.05 | 0.8683 | +0.0361 | 0.9056 | −0.0620 | 0.8878 | −0.0852 | 0.8146 | −0.0782 | 0.7915 | +0.0083 | 1.9219 | −0.0377 | 3/16/3/7/1 | B |
| **0.10** | 0.8676 | +0.0353 | 0.9061 | −0.0616 | 0.8902 | −0.0828 | 0.8348 | −0.0581 | 0.7898 | +0.0067 | **1.9315** | **−0.0281** | 2/16/3/8/1 | B |
| 0.15 | 0.8629 | +0.0306 | 0.9008 | −0.0668 | 0.8896 | −0.0833 | 0.8443 | −0.0485 | 0.7916 | +0.0085 | 1.9306 | −0.0290 | 3/16/1/9/1 | B |
| 0.20 | 0.8614 | +0.0292 | 0.8960 | −0.0717 | 0.8825 | −0.0904 | 0.8309 | −0.0620 | 0.7913 | +0.0081 | 1.9193 | −0.0403 | 2/16/3/8/1 | B |
| 0.25 | 0.8665 | +0.0343 | 0.9026 | −0.0651 | 0.8888 | −0.0841 | 0.8156 | −0.0773 | 0.7843 | +0.0012 | 1.9183 | −0.0413 | 3/16/3/7/1 | B |
| 0.30 | 0.8713 | +0.0390 | 0.8916 | −0.0761 | 0.8850 | −0.0879 | 0.8182 | −0.0746 | 0.7847 | +0.0016 | 1.9207 | −0.0389 | 3/16/3/7/1 | B |
| 0.35 | 0.8628 | +0.0306 | 0.8868 | −0.0808 | 0.8820 | −0.0910 | 0.8239 | −0.0689 | 0.7845 | +0.0014 | 1.9131 | −0.0465 | 3/16/3/7/1 | B |

(V_24-full α=1/3 from EX-3-followup, included for reference: CID22 0.8702 (+0.0379), KADID 0.8776 (−0.090), TID 0.8782 (−0.095), AIC-3 0.7846 (+0.0015), 2/16/4/7/1, B>>A.)

**Key reads:**

- **No Pareto-better α exists.** Every α loses 16 decisive cells to B.
- α=0.10 is the **weighted-best** (Δscore = −0.028, vs −0.038..−0.046 for others).
- α=0.30 has highest CID22 (+0.039) but lowest KADID (-0.076) — CID22 specialist.
- AIC-3 +0.008 max across all α — confirms structural gap.
- The trade is **discontinuous at α=0**: even α=0.025 already loses ~0.083 TID + 0.056 KADID. The damage is constant, not proportional to α.

## Packed α=0.10 seed=3 bake (candidate ship)

Path: `/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18/v24_alpha010_s3_h128_packed.bin`
Source: `/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18/v24_alpha010_s3_h128.bin`
Pack command: `rebake_v3_1 --compress --zerobias 0.005 --dtype i8`

| Metric | Source (f32, 157,252 B) | Packed (i8 + LZ4, 38,850 B) | Drift |
|---|---:|---:|---:|
| Size | 157,252 B | 38,850 B (24.7%) | — |
| CID22 SROCC | 0.86759 | 0.86762 | **0.0000365** (well under 0.001 threshold) |

Pack is ship-grade quality; SROCC drift is 30× under the documented 0.001 ceiling.
