# External held-out sets for the Rev4 featpot instrument (2026-10-03)

Registered 13:29 MT in the design log and in `docs/DATA_SPLITS.md` §3e, before any external score was read. NITS-IQA, LIVE
release 2 and MCIQA-2K are open external reads. KonIQ-10k is a source-image pool. None is a training input.

Owner: `scripts/rev4_featpot/external_sets.py` (extract / table / score). Pairs: `scripts/canonical_corpus/build_fr_corpus_pairs.py
{nits,mciqa}` (LIVE's `live_r2_pairs.tsv` predates this). Pool: `scripts/canonical_corpus/build_koniq_pool.py`.

## 1. Tables

Features f0–f1824 come from the r4 bank extractor with the E14/E15 arguments (feature_set_id `…#d57e9571`, Rev4 `tiercanon_c3negfold`);
f1825–f1852 are NaN (texgain/satsign not extracted). Every distorted image's decoded pixels are hashed (`--audit-jsonl`);
`pair_key = sha256(ref_px_sha + dist_px_sha + "legacy-rgb8")`. No pixel-identical pairs. The orientation gate
(`check_target_orientation.py`, raw ground truth) passes on all three sets with signed SROCC 1.0.

| Set | Rows | Groups | `human_score` | Table sha256 | Keys sha256 |
|---|--:|---|---|---|---|
| NITS-IQA | 405 | 9 distortions D1–D9 | MOS/100 (`Score.xlsx`) | `cebd2573…ed91472` | `b71e328a…a02118` |
| LIVE R2 | 779 | 5 (jp2k, jpeg, wn, gblur, fastfading) | 1 − dmos_new/100 | `ef703de0…fa72c103` | `29bebd93…e6828ca` |
| MCIQA-2K | 2,000 | 5 colorization models | global-naturalness z, min-max | `717ce4b2…bc5fac5` | `f5cca58c…ab189a` |

Tables: `/var/tmp/rev4-featpot/v2c/external/<set>.parquet` (+ `.keys.parquet`, manifests with pairs sha256, extractor sha256 and
gate verdict). Pairs sha256: NITS `8f1fd4da…`, LIVE `49fcd7d2…`, MCIQA `fb0489c4…`. MCIQA pairs each colorized image with its COCO
test2017 original (400 originals in `/mnt/v/datasets/mciqa-2k_extracted/coco_test2017_refs/`; the builder refuses a size mismatch).

## 2. Content-overlap audit

**dHash-64** (`check_holdout_overlap`, binary sha256 `3d14f6c5…`, threshold 16):
- 438 new references (NITS 9, LIVE 29, MCIQA 400) vs 17,251 training sources: the 8,370 of the AIC2026 audit (SafeSyn,
  imazen-26-png-v3, KonJND, KonFiG, CID22-train, TID2013), KADID's 81 and the 8,800 KADIS references used by E14/E15.
  33 flags at d≤10. All were adjudicated unrelated by inspection: 14 of them hit one COCO image with a horizon-like hash.
- The new references vs the existing T0 estate (CID22-49, AIC-3, AIC-4): min d=13, no flags.

**Supplementary checks** (scripts and montages in `/mnt/v/output/zensim/extaudit-2026-10-03/`). dHash misses crop+rescale copies:
- Whole-image 32×24 thumbnail NCC, new references vs the same 17,251 sources: max 0.969, 1 ≥ 0.95, 7 ≥ 0.9. The top two are
  real: LIVE `sailing1` and `plane` against TID2013 `I06`/SafeSyn `6_512sq` and SafeSyn `20_512sq`.
- Multi-scale FFT template match (TID2013 reference slid inside each rescaled LIVE reference): **19 of 29 LIVE references
  contain a TID2013 reference**, NCC 0.90–0.995 with runner-up ≤ 0.82. The other 10 score ≤ 0.64. The 19 are Kodak scenes
  (TID I03, I04, I05, I06, I08, I09, I10, I11, I13, I14, I16, I17, I18, I19, I20, I21, I22, I23, I24). The same match finds
  SafeSyn's `<kodak>_512sq.png` crops inside 17 of them (NCC ≥ 0.85; buildings 0.76 and bikes 0.72 likely too). SafeSyn has
  no crop of Kodak 4, 12 or 22.
- NITS references vs TID2013: max 0.71, runner-up equal, so no match.

**Consequence.** TID2013 trains 4 of 5 LODO folds and SafeSyn every fold, so LIVE is not unseen content. This is the NNCD
precedent. Every LIVE read is reported overall and on the 10 content-disjoint references (`disjoint:` metrics in the compare
JSON; 267 pairs): building2, carnivaldolls, cemetry, churchandcapitol, coinsinfountain, dancers, flowersonih35, manfishing,
monarch, studentsculpture. The subset is fixed by this audit, never by a score. The thumbnail and template checks finished
after the first scored read: the registered dHash owner ran first and missed the overlap.

Residual for every set: a crop of a reference inside a larger training image other than TID/SafeSyn is not excluded.

**KonIQ-10k** vs 3,101 evaluation references (the T0 estate, the three new sets, KADID, TID, KonFiG, CID22-train, KonJND):
- dHash: 367 flags at d≤10, 52 at d≤6. Every closest pair is unrelated (dark or flat images, degenerate hashes).
- Thumbnail NCC: max 0.97. The top 20 are all unrelated smooth-gradient scenes.
- No duplicate found.

## 3. Scoring

Cell bakes are identity-width (1,853 inputs). A zero-weight input still multiplies its NaN, so every prediction on these tables
was NaN until the scorer started routing each bake through the owner `bake_dial_refit densify`. That rewrite reports
predictions BIT-IDENTICAL on its 512 probe rows. The scorer also refuses a cell whose keep list reads f1825+, and refuses any
non-finite prediction.

`python3 external_sets.py score --root /var/tmp/rev4-featpot/v2c --specs <41 specs> --seeds 0-4 --out external_e13_e16_2026-10-03`
→ `/var/tmp/rev4-featpot/v2c/compare/external_e13_e16_2026-10-03.json`. Every (fold, seed) bake is scored and the deltas are
seed-paired against the control. E15's cv1 arms have seeds 0–2 (n=15); everything else has n=25.

**Control `set:v2+basic@h32:H128`:**

| Set | SROCC | Detail |
|---|--:|---|
| NITS | 0.7141 | contrast change D4 0.184, motion blur D6 0.612, pixelate D5 0.787 |
| LIVE | 0.9205 | disjoint 0.9100; white noise 0.706, disjoint white noise 0.660 |
| MCIQA | 0.2690 | expected to be low: humans rated plausibility, not fidelity |

R0 944 (`r0@h32:H128`) is level with the control on NITS and LIVE (+0.0011, −0.0006). It is better on D4 (+0.040) and worse
on MCIQA (−0.023).

| Arm | NITS Δ | LIVE Δ | LIVE disjoint Δ | D4 contrast Δ | D5 pixelate Δ | MCIQA Δ | rule |
|---|--:|--:|--:|--:|--:|--:|---|
| tsnone (no SafeSyn) | −0.0136±0.0025 | −0.0276±0.0102 | −0.0481±0.0121 | −0.046±0.022 | −0.021±0.013 | −0.059±0.016 | – |
| tsfloor0 | +0.0063±0.0029 | −0.0367±0.0131 | −0.0417±0.0150 | −0.006±0.023 | −0.048±0.010 | −0.029±0.015 | – |
| tsxxyb | +0.0059±0.0029 | −0.0207±0.0101 | −0.0247±0.0118 | −0.018±0.024 | −0.009±0.008 | −0.034±0.010 | – |
| ko4 | +0.0021±0.0023 | +0.0301±0.0093 | +0.0363±0.0105 | −0.052±0.025 | −0.006±0.007 | −0.015±0.014 | – |
| ko16 | +0.0050±0.0029 | +0.0377±0.0092 | +0.0465±0.0107 | −0.064±0.016 | −0.001±0.007 | −0.042±0.017 | – |
| cv1:cf4 (noise) | −0.0047±0.0029 | +0.0213±0.0101 | +0.0251±0.0112 | −0.013±0.030 | −0.006±0.007 | −0.007±0.013 | – |
| cv1:cf7f | +0.0025±0.0029 | +0.0307±0.0132 | +0.0375±0.0151 | +0.034±0.029 | −0.005±0.009 | +0.002±0.014 | – |
| cv4:cf98 | +0.0050±0.0027 | +0.0104±0.0084 | +0.0100±0.0098 | +0.028±0.020 | +0.002±0.008 | −0.004±0.012 | SUPPORT |
| cv4:cfbd | +0.0045±0.0024 | +0.0305±0.0084 | +0.0350±0.0098 | +0.004±0.022 | −0.000±0.009 | −0.010±0.011 | – |
| cv4:cffd | +0.0065±0.0033 | +0.0279±0.0077 | +0.0316±0.0089 | +0.022±0.020 | +0.000±0.007 | −0.024±0.010 | SUPPORT |
| cv16:cf98 | +0.0094±0.0025 | +0.0046±0.0089 | +0.0025±0.0106 | −0.003±0.024 | +0.003±0.007 | +0.003±0.011 | – |
| cv16:cfbd | +0.0073±0.0027 | +0.0370±0.0094 | +0.0427±0.0112 | −0.016±0.028 | −0.003±0.008 | −0.018±0.009 | – |
| cv16:cffd | +0.0087±0.0023 | +0.0431±0.0089 | +0.0519±0.0101 | −0.024±0.020 | +0.003±0.007 | −0.033±0.013 | – |

The full 40-arm table is `~/tmp/featpot-audit/external_rule_table.txt`; the compare JSON holds every group.

**Registered reading rule:** an arm is "supporting" if its NITS Δ ≥ 0, LIVE Δ ≥ 0 and the NITS D4 and D5 deltas do not fall. It
cannot overturn a registered decision by itself. Only cv4:cf98 and cv4:cffd pass. The E16 winner cv16:cf98 has the largest NITS
gain (+0.0094), but D4 is −0.003±0.024. That is a fall of about 0.1 SE, and under the strict rule it still fails.

## 4. What the external reads say

- **White noise is the control's external tail.** LIVE wn is 0.706 overall and 0.660 on disjoint references. Noise coverage
  repairs it: ko16 +0.228, cv1:cf4 +0.130, cv16:cffd +0.256, cv16:cfbd +0.212 (overall wn). cf98 has no noise family and gains
  only +0.015.
- **NITS contrast change stays at about 0.1–0.2 under every recipe.** The coverage contrast family does not move it (cv1:cf20
  +0.001), and the KADIS ordinal leg hurts it (ko16 −0.064). Measured cause (DATA_SPLITS §3e): NITS levels 1–2 *lower*
  contrast (luma std ratio 0.89, 0.95) and levels 3–5 *raise* it (1.06–1.23). Pixel change is symmetric (mean |Δ| 7.9 at levels
  1 and 4), yet MOS is 0.76 vs 0.31: NITS observers penalise an increase more than an equal decrease. TID2013 type 17 shows
  the opposite (increase 6.39 vs decrease 4.52). The control follows TID: its mean predictions over NITS levels 1–5 are 45.9 /
  63.9 / 69.6 / 57.4 / 42.1, an inverted U ranked by magnitude with a mild preference for the increase. This is a direction
  preference the training sources contradict, not missing coverage.
- **SafeSyn is valuable on unseen data too.** Dropping it (tsnone) loses on all three sets, which agrees with E13.
- **MCIQA is a weak signal at 0.27.** Most coverage and teacher arms lower it slightly. It is reported as a colour-sensitivity
  diagnostic, not as accuracy.
- The content-disjoint LIVE subset gives the same signs as the full set, with slightly larger deltas. The Kodak overlap does not
  drive any conclusion here.

## 5. KonIQ-10k pool

`/mnt/v/datasets/koniq10k_extracted/koniq10k_pool.parquet`, sha256 `c0e7ad2d98e856b16108f57586c681700b9ac395f08ae8699b80baf867b07472`:
- 10,073 scored 1024×768 images, each verified at that size.
- Columns: file sha256, MOS/SD/z, vote counts c1–c5, and the authors' indicators (brightness, contrast, colourfulness,
  sharpness, JPEG quality factor, bitrate).
- Split: `sha256(image_name) % 10 < 8` gives 8,019 train / 2,054 holdout.
- The 1024×768 zip carries 300 further JPEGs absent from the score file; they are excluded and listed in the manifest.
- Zip sha256 values are in `/mnt/v/datasets/koniq10k/SHA256SUMS` (1024×768 `ea73b96d…`, scores `d895af94…`, indicators `c037abea…`).

## 6. Every feature-set cell on the external sets (exploratory)

`external_sets.py score --specs <control, R0, by_v2fy, every set: spec> --seeds 0-4 --out external_sets_e9_e10_e12_2026-10-03`
(`/var/tmp/rev4-featpot/v2c/compare/`). 262 specs requested, 224 scored. The 38 that read texgain/satsign (f1825+, not
extracted for the external tables) are listed as refused in the JSON. Ranking: `~/tmp/featpot-audit/external_sets_ranking.txt`.

| Spec | NITS (Δ vs v2 + basic) | LIVE (Δ) | LIVE disjoint Δ | MCIQA Δ |
|---|--:|--:|--:|--:|
| v2 + basic (E9″ LODO winner, control) | 0.7141 | 0.9205 | — | 0.2690 |
| R0 944 | +0.0011±0.0022 | −0.0006±0.0152 | −0.0011 | −0.023 |
| by_v2fy (cost candidate) | −0.0109±0.0028 | +0.0233±0.0105 | +0.0265 | −0.034 |
| v2 alone | +0.0150±0.0030 | −0.0013±0.0094 | +0.0031 | −0.079 |
| append + append2 | +0.0151±0.0029 | +0.0301±0.0099 | +0.0397 | +0.054 |
| append + c4 | +0.0137±0.0036 | +0.0306±0.0104 | +0.0408 | +0.043 |
| append alone | +0.0064±0.0031 | +0.0331±0.0094 | +0.0431 | +0.050 |
| v2 + csfw (best NITS) | +0.0193±0.0020 | −0.0016±0.0095 | +0.0021 | −0.073 |
| b2 + c2 (best LIVE) | −0.0241±0.0026 | +0.0469±0.0092 | +0.0593 | −0.095 |

- **The LODO winner is mid-pack on unseen data.** Six sets beat v2 + basic on both NITS and LIVE by more than 2 SE, and every
  one of them contains the `append` group: append + append2, append + c4, append + csfw, a1 + append, append alone,
  append + c4 + masked. If the 224 specs were independent, fewer than one would pass both by chance. They are correlated, so the
  shared `append` group is the likely common cause.
- NITS favours v2-based sets and LIVE favours the b/c families. The two sets do not agree on a single best set.
- by_v2fy keeps LIVE (+0.023) but loses NITS (−0.011): it is not "as good" out of sample on NITS.
- This read is exploratory (open sets, chosen after looking, many comparisons). It does not overturn E9″. A claim that
  append-based sets generalise better needs a registered read on data no one has looked at; R6 stays held.
