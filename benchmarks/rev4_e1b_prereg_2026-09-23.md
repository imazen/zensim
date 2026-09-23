# Rev4 E1b — preregistration: is zensim's human deficit a cross-codec ordering problem? (2026-09-23)

Lane `rev4-e1b`. Program: `docs/REV4_EXPERIMENTS_2026-09-23.md` §Step 2 "E1b". Brief:
`~/tmp/zensim-paper/rev4/E1b_brief.md`. Committed before any statistic in this lane reads a
human label. Anything in the record that is not listed here is marked **exploratory**.

**Question.** On references scored under more than one codec, is zensim's deficit to the best
peer concentrated in cross-codec pairs (same reference, different codec), with same-codec pairs
(same reference, same codec, different quality) tying or winning?

**Scope.** Existing per-stimulus scores and human labels only. Nothing is trained, extracted,
encoded or scored. The statistics owner is `panel --pairwise` (zensim-validate), the same binary
E1 used (sha256 `f11857c27563bfc0ca242d07ef52ba78526b9545d02a3ab9f0754e4a34d05688`, copied to
`/var/tmp/rev4-e1b/panel_f11857c2`; E1 verified it reproduces 21 recorded board SROCCs to 4 dp).

## 1. What was read before this commit (disclosure — this is NOT an independent confirmation)

- **E1's exploratory table** `benchmarks/rev4_e1_regime_2026-09-23_explore.md` (and E1_DONE):
  within-band same-/cross-codec pairwise accuracy, including an `ALL` row per corpus, on the same
  CID22-A, CSIQ, AIC-3 and AIC-4 tables used here. That table *generated* the lead this lane
  tests. E1b therefore re-tests the lead on the **same data** under a fixed rule; the only
  splits not seen by anyone before this commit are: the forced-choice same/cross split with a
  paired difference (E1's fc run computed unpaired question-type cells I have not opened), the
  CSIQ codec-only subset, the CID22 three-way encoder split, the TID2013 codec subset, the
  strata, and the per-pair error tables. Confirmation here means "the lead survives a
  preregistered rule", not "replicated on fresh data".
- Label-free structure only otherwise: stimulus names per corpus (codec tokens and counts,
  below), the forced-choice CSV columns and the codec-code → filename mapping, TID2013 row
  alignment of C's fulleval to `tid_iwssim.tsv` (element-wise equality of the stored target
  vectors, max |diff| 4.4e-11; no statistic).

## 2. Units, codec identity and roles

Input tables: E1's assembled per-stimulus tables, byte-identical to E1's worklog (copied to
`/var/tmp/rev4-e1b/tables/`, sha256 in §8). Every metric is oriented higher = better quality.

| unit | population | codec identity (how read; verified on the counts listed) | label → `t` | role |
|---|---|---|---|---|
| **CID22-A(25)** | 1,992 stimuli / 25 A refs (CID22-B(24) sealed, filtered out by E1's assembler before any target) | path `compressed/<ref>/<enc>/<file>`: **encoder config** = `<enc>` + the non-quality prefix of `<file>` (`aom/s1`, `aom/s7`, `cld_avif`, `vis_avif`, `cld_heic`, `cld_jp2`, `cld_webp`, `libjxl/e7`, `mozjpeg`); **format** = AVIF {aom/s1, aom/s7, cld_avif, vis_avif}, HEIC, JP2, WebP, JXL, JPEG. Counts: aom 504 (s1 229, s7 275), cld_avif 275, cld_heic 200, cld_jp2 225, cld_webp 225, libjxl 272, mozjpeg 273, vis_avif 218 | MCOS | T0 holdout (A only) |
| **CSIQ codec subset** | stimuli of types `JPEG` (150) and `jpeg2000` (150), 30 refs; AWGN/BLUR/contrast/fnoise dropped | filename token 2 (`<ref>.<type>.<level>.png`) | 1−DMOS | T0 eval-only |
| **AIC-3 CTC** | 600 / 10 refs. Rendering: decoded CTC PNGs at the CTC source resolution (`/mnt/v/dataset/aic3_ctc_epfl/decoded/`) | filename token 1: AVIF, HM (HEVC), JPEG-1, JPEG-2000, JPEGXL, VVC (100 each) | design JND `−0.25 × level` (per-codec levels calibrated by the AIC-3 subjective study; cross-codec order is therefore the study's calibrated design, not per-stimulus votes) | T0 family holdout |
| **AIC-4 crop** | 300 / 5 refs, 620×800 PTC crops as shown to observers | filename token 2: AVIF, JPEG-1, JPEG-2000, JPEG-AI, JPEG-XL, VVC (50 each) | −human JND | T0 family holdout |
| AIC-4 full (secondary reading) | same 300 labels, full-resolution renderings | as crop | as crop | same |
| **JPEG-AIC forced choice `btc_native`** | AIC-3 BTC + SDR25 BTC responses, trap/bias/undecided rows excluded (as E1/hfhuman) | `codec_left`/`codec_right` codes; verified 1:1 with filename token 3 on every response row: 0=0ref, 1=AVIF, 2=JPEG-1, 3=JPEG-2000, 4=JPEG-XL, 5=VVC, 6=JPEG-AI | per-question response counts | T0 family holdout |
| TID2013 codec subset (**descriptive only**) | types 10 (JPEG) and 11 (JPEG2000), 25 refs × 2 × 5 | filename token 2 (`I<ref>_<type>_<level>.png`; numbering from the dataset readme) | MOS/9 (`tid_iwssim.tsv` `human_score`) | **TRAIN-role, memorised** in several zensim eras; not counted in the decision |

**Not included:** KADID-10k — its fulleval `per_pair` blocks carry no stimulus keys and are
different subsets for C (5,000 rows) and B (3,125 rows), so a codec split cannot be aligned
without new work → MISSING. KonJND and SDR25 q_jnd tables are single-codec. LIVE excluded
(registered target defect).

Every reference in every unit has stimuli from ≥ 2 codecs (asserted by the script).

## 3. Models and peers (identical to E1 §3; eras labelled per unit in the record)

zensim: **B** (served default), **C** = `W10L9PH_s4004_packed`, D, Rev3 fast `R915_y60_h32_ens5`,
Rev3 rich `R915_basic228_h128_ens5`, PreviewV0_2 — where the unit has rows (CSIQ: B = 2026-07-07
bake on the 08-30 root, C, D; forced choice: C and B (07-07 bake, v1 regime); TID: C only).

Eligible peers per unit (best recorded configuration; anything else labelled):
SSIMULACRA2; butteraugli 3-norm and max-norm; IW-SSIM (ours; on AIC-4 crop also the organisers'
column); DSSIM (CID22-A, AIC-3, AIC-4 crop); MS-SSIM and PSNR-Y (AIC-4 crop organisers' columns);
CVVDP at `standard_fhd` on AIC-4 crop (our port, and the organisers' column); CVVDP elsewhere is
our port at `standard_4k`, **labelled "not the documented display"** (TID: GPU port column,
display unrecorded, labelled likewise). Forced choice: SSIMULACRA2 is the only peer with
per-stimulus scores. Context columns (SSIM, VMAF-neg, HDR-VDP) are not eligible. GMSD has no
per-stimulus rows on any unit → MISSING.

zensim was trained on SSIMULACRA2 labels; agreement with SSIMULACRA2 is partly distillation.

## 4. Pairs

- **Table units.** All within-reference pairs of stimuli (no quality band restriction) whose
  targets differ at the stored resolution (exact ties dropped). Classes:
  - **same** — same codec (CID22-A: same *encoder config*, i.e. one quality ladder);
  - **cross** — different codec (CID22-A: different *format*);
  - CID22-A only, **mid** — same format, different encoder config (e.g. cld_avif vs aom/s7):
    descriptive, not in the decision.
  `panel --pairwise` rows: `group = reference`, `s_left/s_right` = metric quality, `choice` = the
  side with the lower `t`, weight 1; metric ties score 0.5 (owner rule); statistic `acc_response`.
- **Forced choice.** Triplets with a distorted image on both sides: **same** = `codec_left ==
  codec_right`, **cross** = different codecs (`vs_original` questions excluded). Rows as in E1
  fc.py: `group = question_id`, response-weighted, the response names the side judged more
  different.

## 5. Statistics and uncertainty

Per unit × pair class × metric: `acc_response` from `panel --pairwise`, point and per-resample.

- **Reference-clustered bootstrap, B = 2000, seed 20260923** (E1's draws: `random.Random(seed)`,
  each resample draws the unit's references with replacement, count = number of references).
  The **same draws** are used for every metric and for the same and cross classes of a unit
  (paired). Forced choice: image-clustered over the union of images, one draw set shared by
  both classes (seed 20260923). Percentile 95 % CIs.
- **Best peer** of (unit, class): the eligible peer with the highest point accuracy on that
  class's pairs. Δx = M − best-peer(cross) on cross pairs; Δs = M − best-peer(same) on same
  pairs; **DiD = Δx − Δs** per resample. Δ vs SSIMULACRA2 reported alongside.
- n for every cell: pairs, references (and responses / questions for forced choice).
- Ceiling note: same-codec ladders are often ordered correctly by every metric; the record
  reports the fraction of same-class pairs every eligible metric orders correctly.

**Descriptive strata (preregistered, not in the decision)**, cross pairs only, best peer fixed at
the unit's cross best peer:
- *human gap:* JND units (AIC-3, AIC-4) |Δt| < 1 JND = small, ≥ 1 = large; MOS units
  (CID22-A, CSIQ, TID) split at the median |Δt| of that unit's cross pairs (≤ median = small);
  forced choice |dlevel_left − dlevel_right| ≤ 1 = small, ≥ 2 = large.
- *codec pair:* unordered pair of formats (CID22-A: formats; others: codec tokens).
- *error tables:* per unit × codec pair (same-codec pairs as X–X), per metric: count of pairs the
  metric orders wrong while the unit's best cross peer orders right, the converse, and metric
  ties. Plain tallies on the pair lists (strict sign of the score difference vs the human
  choice; forced choice uses the per-question strict majority, exact splits excluded). These
  are raw material for cross-codec supervision, not a statistic.

## 6. Decision rule (brief, then operational)

> Confirmed: zensim's deficit to the best peer excludes zero on cross-codec pairs, while it ties or
> wins on same-codec pairs, on at least two corpora. Refuted: cross-codec and same-codec deficits
> are similar. Unresolved: otherwise, naming the missing data. Report B and C separately.

Per zensim model M and unit U (a unit where M has no rows is MISSING for M):
- **CONF** — CI(Δx) entirely < 0 **and** upper CI(Δs) ≥ 0.
- **NONSPEC** (deficit not cross-specific) — CI(Δs) entirely < 0 **and** DiD CI not entirely < 0.
- **NONE** — neither CI(Δx) nor CI(Δs) entirely < 0.
- Anything else (both deficits, DiD CI < 0: cross worse but same-codec also losing) — **PARTIAL**,
  counted toward neither.

**Independent units:** CID22-A, CSIQ codec subset, and the JPEG-AIC family counted **once**
(members AIC-3, AIC-4 crop, forced choice `btc_native`; AIC-4 full and TID are reported, never
counted). Family = CONF if ≥ 1 member is CONF and none is NONSPEC; NONSPEC if ≥ 1 member is
NONSPEC and none is CONF; otherwise MIXED (counted toward neither).

- **Confirmed:** #CONF ≥ 2 and #NONSPEC = 0.
- **Refuted:** #NONSPEC ≥ 1 and #NONSPEC ≥ #CONF.
- **Unresolved:** everything else (including #CONF = 1, or no deficit anywhere); the record names
  the data that would resolve it.

Headline: **B** and **C** separately. D, R915 fast/rich and V0_2 are evaluated under the same
rule and reported without changing the headline. Sensitivity (reported, not decisive):
(i) family members counted as separate units; (ii) best peer of a unit fixed to the cross-class
best peer for both classes; (iii) CID22-A with E1's encoder-directory definition of codec
(aom s1/s7 same, other formats' encoders cross).

## 7. Exposure

One evaluation read of CID22-A(25), CSIQ, AIC-3, AIC-4 (crop, full) and the JPEG-AIC BTC
responses, purpose **"rev4 step-1 e1b"**, appended to the ledger in `docs/DATA_SPLITS.md` in this
commit. TID2013 (TRAIN-role) read for description only. No fitting, calibration or selection.

## 8. Inputs (sha256)

```
ac02ce4c87286c46d2405235ba2c3fc98d86ad337c152b654f7409a1aeb0e36f  /var/tmp/rev4-e1b/tables/cid22a.tsv
ad609c663bd57cef7d2351b406c2b56b2e0ed3ef191374e9b23d46690315ccbc  /var/tmp/rev4-e1b/tables/csiq.tsv
f2421a98b20fb40f171152f6d6a0c711e601bdc2e8bf3c1042518a3b8bab42ff  /var/tmp/rev4-e1b/tables/aic3.tsv
486cf77e0473de6f2fec8f8d04a24f5087012c48908e5d932cdba5bf621877ed  /var/tmp/rev4-e1b/tables/aic4crop.tsv
6064e9d4e7eea7192cebd1eee1f3fae83f1c489385e8ce1026741c911cc88458  /var/tmp/rev4-e1b/tables/aic4full.tsv
7771c4b178d3e440631dd9eeb0151c8e744c7d4cea262a9bfa68450a2579b337  /var/tmp/rev4-e1b/tables/manifest.json
edd1835dfc4e5eeb6dddb6a81ea45c8f2fdf020b5c6f00dcc12d34b20fb1e6d7  /mnt/v/output/zensim/hfhuman-2026-09-01/btc_native_scores.tsv
d3da52a14745a883ab338e029a616606c11459d5d217ffc1a684d015a3884216  /mnt/v/datasets/aic3-btc-ptc/JPEG-AIC_BTC_final_response_data_2024.01.10.csv
dd3f7050c83553282bc34f0812a81a96cd14745d3492c39ae5b721f87a686365  /mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data/JPEG_AIC_SDR_BTC_JPEG_AI_responses_2025.02.28_v1.csv
473d98806d0190e5a0abc6d73fa3c2e47748c98b5e8399beba94c11567cd0b7a  /mnt/v/output/zensim/reports/fulleval/W10L9PH_s4004_packed.fulleval.json
4e0e637dafc3d70a2381eea130c9132d3c3274fd977d5f44d757d322108a3aae  /mnt/v/output/zensim/reports/refmetrics/tid_ssim2_gpu.tsv
4358fafa54cadef5ff432a68186cf94eeb3060a78d6ded14148fbe22073e0546  /mnt/v/output/zensim/reports/refmetrics/tid_butteraugli_gpu.tsv
009d5492e809a31831fa4f1c97ddeebcbfda4dc964dd1cca362cc8f901992e4d  /mnt/v/output/zensim/reports/refmetrics/tid_iwssim.tsv
41ffbce633560520a4a0dff9bf8def55884df9343689568a99950fd98697dbe1  /mnt/v/output/zensim/reports/refmetrics/tid_cvvdp_gpu.tsv
f11857c27563bfc0ca242d07ef52ba78526b9545d02a3ab9f0754e4a34d05688  /var/tmp/rev4-e1b/panel_f11857c2
```
The upstream sources of the E1 tables are hashed in `benchmarks/rev4_e1_prereg_2026-09-23.md` §8.
