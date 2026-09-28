# DATA_SPLITS.md — canonical train/val/test conventions (locked 2026-07-02)

## September 14 clarification: test evaluation when no eval split exists

The user's later clarification supersedes the September 13 blanket ban:
"when there is no eval you can eval against test, but use guards from
overfitting and know that there are secret holdout sets".

Keep original dataset roles. CID22's 201-reference SSIMULACRA2-oracle training
population (including its safesyn use) is distinct from the 49-reference
human-scored test population. Human test labels never become training targets.
Use an existing EVAL split when present; otherwise the published TEST population
may assess a frozen candidate. This permits CID22 gold, AIC-3 and the AIC-4 public
sample for assessment; it does not authorize reading secret holdouts.

Before a read, freeze model bytes/composition, population, metrics and gates.
Record each batch's exposure and retain all candidate results and failures.
Never use these results for fitting, calibration, feature/hyperparameter search,
checkpoint selection or repeated adaptive tuning. Develop changes on TRAIN;
subsequent public-test assessments must disclose prior exposure and cannot be
claimed as fresh independent holdout confirmation. Do not scan secret holdouts.
Existing historical admissions remain immutable; new batches cite this ruling.

<a id="september-13-user-ruling-train--eval-only-never-touch-test"></a>

## September 13 user ruling (superseded where clarified above)

This later explicit instruction supersedes every historical permission below
to read a test/terminal segment, including "touch once" or frozen-finalist reads.
Training, transforms, calibration and checkpoint selection use train data only.
Eval data are for gates/evaluation after the candidate is frozen, not fitting or
checkpoint selection. Test segments are never opened, extracted, scored, used
for audits, or silently renamed to eval. Preserve canonical source/family
assignments and immutable historical evidence.

The feature-screen owner now accepts only the explicit v2 train/eval protocol
in [FULL_EVAL](FULL_EVAL.md#strict-train-eval-feature-screens-september-13).
Legacy fit/dev/test recipes and mixed preparation caches are refused before
dataset access. Its strict admission route uses source-only sidecars, not the
historical split checker's terminal-table scans. Sidecars must be reviewed
against their named canonical split authority; schema/hash validation alone
does not prove that an arbitrary assignment is canonical.

Earlier September 13 studies used an internal role named `test` for KADID
SELECT and inner splits of training-origin codec/corruption data. Those labels
are historical, not authorization to reuse the segments under this ruling.
No existing `test` segment is migrated or retagged by this change.

**This is the ONE registry of how every dataset in the zensim/picker/metric
stack is split, what the rest of the field does with the same data, and which
rules are load-bearing for replicable science.** Locked per user directive
2026-07-02 ("document train vs val vs test sets and their conventions
everywhere … so our science can be replicated"). When a new dataset lands,
add its section here IN THE SAME COMMIT that first uses it. When a rule here
conflicts with an older doc, THIS FILE WINS — fix the older doc.

Companion docs: `~/work/zen/DATA_PROVENANCE.md` (where data lives),
`docs/EVAL_PANEL_REQUIREMENT.md` (two-panel eval), CLAUDE.md ("CID22 is
VALIDATION-ONLY", contamination rules).

September 14 derived TRAIN entry: the [product packet](../benchmarks/product_train_2026-09-14.md)
inherits W-LIN7's original TRAIN key authority and the September 8 family map.
It excludes all canonical validation/test families, historical codec-screen
reservations 1220/1634/7004/7050/7058/8134 and their relatives, historical
corruption-screen reservations 8462/9066 and relatives, and every suffix-8
origin/family to preserve the earlier AVIF eval8 reservation. Metadata admission
precedes any pixel read. Its 10,499 pairs are split by source-family SHA-256
modulo ten into internal development (0/1), calibration (2), and fit (3–9).
All three roles remain TRAIN; they cannot qualify a frozen model. Counts are
7,947 fit, 1,629 development and 923 unused calibration. The reused 8,000-row
human TRAIN packet contributes 7,000 fit rows and 1,000 internal-development
rows from eight KADID TRAIN sources selected by source hash order. No previous
test segment is renamed or admitted. Original authorities, row assignments,
source/pixel hashes and model results are pinned in the linked record.

September 8 derived-input entry: the [canonical corruption packet](CANONICAL_CORRUPTION_2026-09-08.md)
inherits the existing native-targeting 12 training / 8 validation origin and
family assignments, including all corruption attempts, anchors and honest
codec renditions. It does not define a new random split. Raw catalog tables
retain duplicates for audit and are not yet admitted training views. Any fit
must separately register probability-calibration origins within the training
families; all eight validation origins stay evaluation-only. No terminal origin
is present. [Counts, hashes and pending admission gates](../benchmarks/canonical_corruption_2026-09-08.md).

---

## 1. Principles (apply to every dataset)

1. **Split by CONTENT, never by row.** A "content unit" is an origin image /
   reference / source — every rendition, crop, encode, severity level, or
   metric-scored cell derived from it inherits its bucket. Splitting rows
   leaks near-duplicates across the boundary. This matches the standard IQA
   literature protocol (by-reference splits) and is non-negotiable.
2. **Deterministic arithmetic rules, never seeded shuffles.** Our two split
   forms (least-significant-digit, modulo-10) are reproducible across blind
   sessions with zero state — no seed files, no stored index lists to lose.
3. **Holdout tiers.** Every dataset is exactly one of:
   - **T0 SACRED human holdout** — human labels NEVER in training, content
     dHash-audited against training corpora. (CID22-49, AIC-3, AIC-4, SDR25.)
   - **T1 integrity guard** — present in training; its eval numbers detect
     pipeline breakage/memorization, NEVER used for ranking candidates or
     scoreboarding vs external metrics. (KADID, TID, konjnd-dense.)
   - **T2 training** — metric-anchored or weak labels, freely trainable.
     (safesyn, kadis-700k train, bigcodec/canonical-picker, cid22-train-201.)
   - **T3 instrument** — eval-only grids for dial/safety/zone panels; must
     document their CONTENT overlap with training tiers (see §4).
4. **Frozen inputs.** A file referenced by a ship manifest is immutable —
   schema additions create a NEW dated file (the 2026-05-28 konjnd in-place
   rewrite destroyed byte-provenance across all three mirrors; never again).
5. **Reproducibility gates.** Manifests record input sha256 + `trainer_commit`;
   the trainer verifies both (see `train_manifest.rs`). Profile A is
   byte-reproducible under these gates (verified 2026-07-01).
6. **Dedup by content at corpus build.** Sweep-derived training corpora MUST
   dedup on (ref, target, feature-prefix) — knob no-ops produce byte-identical
   encodes under different `knob_tuple_json` keys (measured 2026-07-02: 22.2%
   duplicate rows in canonical-2026-06-27-derived training data). The
   validator's C10 gate (<1% sampled dup rate) enforces this.
7. **Contamination audits.** Any new training corpus is dHash-64-audited
   against every T0 holdout's references at d≤10 (strict) before first use;
   the d≤16 tail is screening-only (flat/graphic content false-positives).
8. **Target ORIENTATION is gated at build time.** Every table with recoverable
   human labels must satisfy `sign(SROCC(human_score, raw_human_truth)) > 0`
   before it is trained or evaluated on, via
   `scripts/canonical_corpus/check_target_orientation.py` (`--all-roots` sweeps
   every known root). A corpus with no recoverable raw truth reports SKIPPED,
   which means "not checked", never "passed". Added 2026-08-05 after the ext
   lineage carried an inverted KADID target for six weeks (campaign appendix F).

### 1.4 — the ONE registered exception to frozen inputs (2026-08-05)

Principle 4 says a file referenced by a ship manifest is immutable. The
2026-08-05 KADID orientation correction **rewrote `ext_kadid.parquet` in place**
at all three ext roots, which is a deviation from the letter of that rule. It is
recorded here rather than buried, because a locked rule that gets quietly bent is
worse than one that gets openly amended.

**What the rule exists to prevent, and whether it happened.** Principle 4 was
written after the 2026-05-28 konjnd in-place rewrite *destroyed byte-provenance
across all three mirrors* — the old bytes were simply gone. That did **not**
happen here: the inverted originals are preserved as
`ext_kadid_INVERTED_2026-08-04.parquet` in **all three mirrors** (local
`/mnt/v`, `s3://zentrain/<root>/`, `/mnt/tower/output/zensim-<root>/`), each
sha256-recorded in the root `_MANIFEST.json`, and the ext944 preserved sha
(`4dde6be2…`) is byte-for-byte the sha every affected bake's embedded
`zentrain.repro` already carries for that input. No provenance was lost.

**Why the canonical name, and not a new dated file.** A new dated file leaves
`ext_kadid.parquet` — the name every recipe, driver and manifest already points
at — permanently inverted, so the orientation gate can never exit 0 and every
future recipe has to *remember* to override the path. That is precisely the
failure mode that let this defect live six weeks. The correction is worth more
at the canonical name than the immutability is worth on a file that was wrong.

**The hazard this creates, stated plainly.** Re-running any pre-2026-08-05
bake's embedded `zentrain.repro` argv **verbatim** now trains against the
corrected bytes and will **NOT** reproduce that bake. The `sha256` field in the
repro is the discriminator; substitute `ext_kadid_INVERTED_2026-08-04.parquet`
to reproduce. Registry entry: `kadid-ext-root-corrected-2026-08-05` in
`benchmarks/eval_annotations.json`.

**Scope.** This exception covers exactly the three `ext_kadid.parquet` files and
nothing else. Principle 4 is otherwise unchanged: any FUTURE schema addition,
row change, or target change creates a new dated file.

---

## 2. The two canonical split FORMS

### 2a. Least-significant-digit origin rule (imazen-26 family)

**September 8 chronology correction for new steering calibration:** imazen-26
moved to `imazen/imazen-26` on August 23; the `codec-corpus` copy is explicitly
superseded. The newer August 27 `manifests/split_map_family.tsv` groups shared
content (patent pages, screenshot viewports, generator families), assigning a
family by its lowest origin ID. It changes 175 origin assignments; totals are
1084 train / 661 validate / 415 test. `origin_split.py` below implements the
older individual-origin rule, not that family extension. The September 8
steering experiment requires both assignments to agree and enforces disjoint
families, a conservative intersection which preserves existing restrictions.
Use the canonical manifests and measured render-URL index at pinned revision
`187fbf338ce08e8e6654db7f04ddae58d5263da2` (also retained in the
experiment's source manifest; the corpus repository is the authority).
Historical tables are not retrospectively relabeled by this correction.

**Source of truth: `zenmetrics/scripts/picker/origin_split.py::split_of` —
import it, never re-implement.** Set by the user 2026-06-26.

```
last digit of the origin's numeric id ∈ {0,2,4,6,8} → TRAIN
                                      ∈ {1,3,5}     → VALIDATION
                                      ∈ {7,9}       → TEST
```

- Origin-level: every rendition/crop/encode of `o_1004.*` inherits o_1004's
  bucket. The origin stem must LEAD the filename (imazen-26 + dense-rendition
  conventions guarantee this); a name with no leading numeric stem → None.
- Used by: canonical-picker-2026-06-27 (all 7 datasets; builder asserts zero
  cross-split origins), picker training, corpus segmentation.
- 414 origins → 212 train / 128 val / 74 test.

### 2b. Modulo-10 source_id rule (KADIS-700k)

**Source of truth: the dataset README (`s3://zentrain/kadis-700k*/README.md`,
`~/work/kadis-distort/docs/DATASET.md`).**

```
source_id % 10 < 8   → TRAIN   (112,000 sources / 560,000 cells)
source_id % 10 == 8  → VAL     ( 14,000 sources /  70,000 cells)
source_id % 10 == 9  → TEST    ( 14,000 sources /  70,000 cells)
```

- `source_id` = stable 0..139,999, assigned by sorting unique
  `source_filename`; all 5 severity levels of a reference share it —
  splitting on it is leakage-free by construction. Split on source_id,
  **never on row**.
- Used by: `kadis_cvvdp_train.parquet` (train split), the held-out KADIS
  monotonic-safety grid (`kadis_test_safetygrid.parquet` = test split,
  signed types 7/18/25 excluded), clean TV pairs (train split).

---

## 3. Per-dataset registry

| Dataset | Tier | Our split | What others do | Leakage status |
|---|---|---|---|---|
| **CID22** (Cloudinary, 4,292 val pairs / 49 refs + 201 train refs) | T0 (49-ref) + T2 (201-ref, ssim2-anchored) | 49-ref set = sacred eval-only; 201 disjoint refs trainable with **ssim2 targets only** (verified: `cid22_train_norm.human_score == ssim2_gpu/100` exactly; human MCOS never trains) | The CID22 paper itself: 201 refs tuned SSIMULACRA2, 49 held out — we mirror the authors' own split | dHash-audited; synth corpus purged 2026-05-12; imazen-26 clean at d≤10 (2026-07-02); **⚠ contains one picture under two names: `844297.png` ≡ `3316926_opo25u.png` (dHash 0, NCC 0.9999; audit 2026-09-22) — any split of CID22-49 by filename leaks it; split by content hash** |
| **KADID-10k** (10,125 pairs, 81 refs, DMOS) | T1 | Full set trains (v47 w0.5) AND full set evaluates → train==val integrity guard | No official split; literature: random by-reference 80/20 (or 60/20/20) × 10 repeats, median SROCC. **ssim2 tuned on ALL of it** → never scoreboard vs ssim2 here | 6 training sources flagged d≤10 vs KADID refs (2026-05-14, mostly flat-content FPs, user review pending) |
| **TID2013** (3,000 pairs, 25 refs, MOS) | T1 | Same as KADID (v47 w0.5, train==val) | Same literature convention (by-reference CV); ssim2 tuned on all of it | 1 source d=10 (flat-content FP, review pending) |
| **KADIS-700k** (700k cells, 140k sources, NO human labels) | T2 + T3 | §2b modulo rule; train=<8, safety-grid=9; targets = GPU metrics (cvvdp/10 primary) | Authors (Lin/Hosu/Saupe): weak-label TRAINING set for FR-metric distillation (DeepFL-IQA) — no human labels, no eval role. Our train-on-metric use matches the authors' intent; our %10 split adds held-out safety eval they didn't define | Reference pool is KADIS (Pixabay), disjoint from KADID's 81 refs per the authors; our safety grid excludes signed types 7/18/25 (severity≠quality there) |
| **imazen-26 / canonical-picker-2026-06-27** (5,742,660 cells, 414 origins) | T2 | §2a LSD rule for picker work. For ZENSIM training (bigcodec_5p7M) all three buckets train — zensim's holdouts are T0 corpora, not picker buckets. NOTE the consequence: picker-val/test origins are seen by zensim bakes | N/A (our corpus) | imazen-26 origins vs CID22-49: **CLEAN at d≤10** (min d=12, 2026-07-02, `imazen26_vs_cid22_dhash_t16.tsv`); 16/1067 decode-failures unaudited (odd screen PNGs) |
| **safesyn** (196,086 pairs) | T2 | All train; ssim2-derived targets; CID22-leak-purged 2026-05-12 | N/A (our synthetic corpus) | Purged at d≤16 (loose-threshold caveat documented) |
| **KonJND-1k** (1,008 refs; JPEG+BPG PJND) | T1 (semi) | train = konjnd-dense (20,160 rows, per-pair active-mix target) AND val = per-ref mean PJND — same 1,008 refs both sides → ref-level train==val; treat KonJND eval as guard+anchor, not holdout. **MEASURED 2026-08-04 (wave 6): the set is 504 JPEG refs ∪ 504 BPG refs, intersection 0**; the 944 eval leg `ext_konjnd_jpeg_val.parquet` is **exactly the JPEG 504**, and `konjnd-dense − eval` is **exactly the BPG 504**. So the ONLY reference-disjoint KonJND training mass is the BPG half. **CORRECTED + RESOLVED 2026-08-04 (wave 7, campaign amendment 7):** the "no BPG decoder ⇒ cannot be extracted" claim was wrong — the KonJND-1k distribution ships the BPG half **pre-decoded** (`KonJND-1k/bpg/` = 25,704 valid 640×480 RGB8 PNGs, 504 refs × 51 QPs, upstream 2021 mtimes; zero `.bpg` bitstreams exist on disk), the 372 dense build had already extracted those very pixels (10,080 BPG rows), and the dense build's pair list + target rule WERE recovered exactly from `konjnd_full_scored.csv` (20 rank-evenly-spaced picks/ref over the ssim2-sorted ladder; `human_score` = raw `gpu_ssimulacra2`; verified 1008/1008 refs <1e-9). The reference-disjoint 944 training leg now exists: `ext944-canonical-2026-08-01/konjnd_bpg_{train,val}_944.parquet` (403/101 refs, srcnum%10∈{8,9}→val, target = ssim2/100; `_MANIFEST_konjnd_bpg.json`) | Authors: whole-set JND benchmark, no split defined | — |
| **AIC-3 CTC** (600 pairs, 10 refs) | T0 | Eval-only, never train | JPEG-AIC committee test set; Mohammadi 2025 evaluates metrics on it | Holdout by construction |
| **AIC-4 sample** (300 pairs, 5 refs) | T0 | Eval-only, never train; do NOT recipe-search to win it (holdout-fishing ban, 2026-05-25 #10). **⚠ TARGET IS DISTORTION-ORIENTED** (`q_jnd`, same reconstruction family as SDR25): all 188 board fullevals report negative `srocc_signed`. Correct for a JND study; negate before any training use. **⚠ SDR25 ⊂ this corpus** — SDR25's 50 rows are the JPEG-AI subset of these same 300 rows / 5 crops (verified 50/50 on `ref_basename`+f0..f5), so scoring both is not two independent reads. See campaign Appendix I | The CfP keeps the larger set committee-hidden — public sample is eval-only for everyone | Holdout by construction |
| **NNCD-IQA** (16 Kodak refs × 5 codecs × 4 rates = 320, MOS; registered 2026-09-22, zensim#62) | **EVAL-only** | Never trained, fitted, calibrated or selected on. Target quality-oriented (MOS). Local `/mnt/v/datasets/nncd-iqa/` (sha256 in `SHA256SUMS`) | Authors (Khan, Dardouri, Kaaniche, Dauphin, MTAP 2022): benchmark set; one of the five sets in the DVIFM talk's list | dHash vs 4,544 training sources: no duplicates (all flags adjudicated unrelated). **Not content-disjoint from TID2013: all 16 NNCD references contain an unscaled crop that is a TID2013 reference (16 of TID's 25; NCC 1.0000)** — any TID-trained model has seen NNCD's scenes. DATASET_HISTORY 2026-09-22 |
| **JPEG-AI-SDR25** (5 src × 10 levels, 95k raw triplets) | **T0 (BUILT 2026-07-02)** | Eval-only. Reconstructed: `sdr25_jnd_reconstructed_2026-07-02.parquet` (116 stimuli, ordered-probit triplet MLE, `scripts/v_next/reconstruct_sdr25_jnd.py`; response = MORE-DISTORTED side, trap-verified). Scoreable subset = 5×10 JPEG-AI PTC crops (anchor codecs not in the **JPEG-AI** zip — but they ARE on disk in the AIC-3 package, `aic3-btc-ptc/test-images/{BTC,PTC}_images.zip`, 5 refs × {AVIF,JPEG-1,JPEG-2000,JPEG-XL,VVC} × 10 levels; corrected 2026-08-04). **⚠ TARGET IS DISTORTION-ORIENTED** — `human_score` = `q_jnd`, a JND *distance* from the original (rises with distortion). Verified three ways (source, raw ladder, and signed SROCC **−0.9757** vs 67,714 raw votes); all 171 board fullevals report negative `srocc_signed`. This is CORRECT for a JND study and **must NOT be flipped** (the seed-selection oracle consumes `\|SROCC\|`; flipping silently inverts it). **Negate before any training use.** Gated by `check_target_orientation.py` (declares `distortion`). **⚠ SDR25 ⊂ AIC-4**: all 50 rows are present in `ext_aic4.parquet` (300 rows, same 5 crops) — they are NOT independent eval corpora. **NOT TRAINABLE** — T0 + it is the seed-selection oracle (+0.752 → CID22 over 35 bakes) + 5 refs. Full determination: campaign **Appendix I**. Baseline: within-image SROCC A 0.998 / ssim2 1.000 (both ceiling); pooled A 0.904 vs ssim2 0.958 → A currently FAILS the "SDR25 ≥ ssim2" gate | Authors: subjective study behind the QoMEX'25 SVQA paper (arXiv:2504.06301), Jenadeleh/Sneyers/Jia/Mohammadi/Ascenso/Saupe — cite arXiv:2504.06301 | Postdates ssim2 tuning — honest holdout for both sides |
| **JPEG AIC2026** (70 src × 17 codec configs × 20 levels = 9,618 distorted; full-res + 840×944 PTC/BTC crops; **NO human scores in this release**) | **T0-family, EVAL-ONLY (INGESTED 2026-09-19)** | Member of the registered holdout family `jpeg-aic-family-holdout-2026-09-01` (§3d) — **never a training input, membership by CONTENT**. It carries **no human labels at all**: the release ships 71 *objective* IQA-method score columns and nothing else, so every use is a **metric-agreement / ladder-behaviour panel**, never an accuracy-vs-humans measurement. It cannot produce a `rank.<corpus>` board axis in the human sense and must not be given a synthesised `human_score`. **Target orientation is per column and mixed**: the `JND_*` columns and the distance-like metrics (`proposal-Butteraugli`, `DSSIM`, `LPIPS-*`, `DISTS`, `GMSD`, `NLPD`, `CIEDE2000`, `WD_s*`, …) **RISE with distortion**; `SSIMULACRA2`, `CVVDP` (JOD), `PSNR*`, `SSIM`, `MS-SSIM`, `VMAF*`, `IW-SSIM`, `VIF`, `TOPIQ`, `HaarPSI`, `FSIM*`, `AHIQ`, `VSI`, `CW-SSIM` **FALL** — sign-normalise per column before any correlation. **Levels were PLACED using CVVDP**, so CVVDP is monotone-by-construction on these ladders and is not a fair contestant on the monotonicity axis. Local: `/mnt/v/datasets/aic2026/` | Authors: fine-grained high-fidelity benchmark for the JPEG AIC activity (Jenadeleh, Sneyers, Ascenso, Richter, Karabutov, Jia, Alshina, Watanabe, Pinheiro, Ebrahimi, Saupe), DaRUS doi:10.18419/DARUS-6156, arXiv:2607.22783, CC BY-SA 4.0. The subjective study over these stimuli was still pending at release | First read by any zensim model **2026-09-19**; all scored models were frozen before that date. dHash audit (`check_holdout_overlap`) run before first use — see `benchmarks/aic2026_2026-09-19.pointer.md` |
| **KonFiG-IQA** (10 src × 7 dist × 12-30 levels over 3 JND; 1.7M triplets) | **T2 (INGESTED 2026-07-02; 944 LEG BUILT 2026-08-05)** | 944 leg: `ext944-canonical-2026-08-01/konfig_944.parquet` (1,090 rows, 85+24 per source, + native `q_jnd`; multiset-identical to the 372-era `konfig_train_2026-07-02.parquet`; builder `scripts/canonical_corpus/build_konfig_944.py`; campaign **Appendix L**, pre-reg `e93eba04`). human_score = 1−q_jnd/3.2 — **QUALITY-oriented, gated**: `check_target_orientation.py` declares `quality`, verified signed SROCC **+0.5645** vs the 75,519 raw EXP_III DCR votes (n=850 PartA; PartB shares the formula). Origin-split views `konfig_originsplit_{train,val,test}_944.parquet` (327/436/327; `split_of` on numeric src id) exist for any future within-KonFiG instrument; the registered probe leg is the FULL table (L.6 design decision — training on it forecloses those views as eval for those models). **ssim2 tuned on it** → never a ssim2-comparison corpus | Authors: fine-grained JND-unit scales via boosted triplet comparisons (Men 2021) | 10 sources. **dHash spot-audit RUN 2026-08-05 (Appendix L G-L1/G-L2, commit `7ed6ac4b`): CLEAN PASS** — 0 exact hits + zero d≤10 flags vs KonJND-1008 / CID22-49 / CSIQ-30 / LIVE-29 / AIC3-10; global min d=17 (dHash is crop-blind — residual stated in L.11.8) |
| **PIPAL** (local, unused) | — | Not in pipeline (SR/GAN domain) | Official NTIRE train/val/test splits | — |
| **UPIQ / HDR** | T0-eval for HDR track | Held-out UPIQ eval per HDR plan | Mikhailiuk 2021: consolidated dataset, JOD-rescaled | — |
| **AIC-HDR2025** (5 HDR src × 4 codecs × 5 levels, JND) | — | **UNOBTAINABLE (user ruling 2026-08-05): data was never publicly released and the authors are unresponsive — STOP live-checking `github.com/jpeg-aic/AIC-HDR2025`.** README-only clone at `/mnt/v/datasets/aic-hdr2025/`; dropped from HDR anchor plans (`HDR_PLAN.md`, `PLAN_HDR.md`) | Paper: QoMEX'25, arXiv:2506.12505 | — (no data on disk) |
| **hdr_v3mix @944 (hdr944-leg)** (17,100 zenjxl HDR-PQ cells → 7,410 train + 3,900 val after dedup; 58 imazen-26-hdr origins) | **T2 (944 LEG BUILT 2026-08-03; REGISTERED 2026-08-05)** | Digit origin split on the leading numeric stem (`origin_split.split_of`): 38 train / 20 val origins, overlap 0, `split_of` agrees 870/870 refs (campaign **Appendix Q** G-Q3). Features = chunk-2 HDR route at `Folded720Append2` (`compute_folded720_append2_features_hdr`, PQ 10k nits); target = cvvdp-mix `0.5·clip01(ssim2/100)+0.5·clip01((JOD−6)/4)`, **quality-oriented, gated** (`check_target_orientation.py hdr_v3mix` in-table mode: train +0.8494 / val +0.8606; caveat: consistency vs carried JOD, not independent). NEW-REGIME leg — never column-mix with SDR tables or the v3 pu-linear 372 corpus (same targets, different feature space). Manifest: `/mnt/v/output/zensim/hdr944-leg/_MANIFEST.json`; Tower `zensim-hdrp1-2026-08-05/hdr944-leg/` | N/A (our corpus; targets are metric teachers) | G-Q4 PASS: imazen-26 zfold7 2026-03 personal captures are temporally+authorially disjoint from every HDR eval source (UPIQ narwaria/korshunov, SI-HDR, HDR-VDC, AVT, CHUG, Rousselot); id-containment 0 hits |

| **avif944 / avifgen-2026-08-06** (564,300 cells, 1,455 renditions, 1,082 origins — ALL train-side by construction) | **T2 (CLOSED 2026-08-21; campaign appendix Z.R)** | AC.R1 amendment 2: the corpus reuses train_renditions_2026-06-14 (every origin ends 0/2/4/6/8 — the June even/odd rule puts all of it train-side), so validate/test views are structurally EMPTY. Views: `train_944` = origins ending 0/2/4/6 (459,780 rows / 873 origins; the wave-12 TRAIN-ONLY leg, target `ssim2_gpu`) + `eval8_944` = origins ending 8 (104,520 rows / 209 origins; leg-side eval holdout — never trained on; the G-AC2 AVIF instrument population). Emitter: `zenmetrics scripts/jobsys/avifgen_training_views.py` (origin_split owner; hard-errors on any non-train-side digit). Rank validation for wave-12 stays the T0 estate (CID22 etc.) | N/A (our corpus; targets are metric teachers) | 4 byte-identical rendition pairs — all train/train, zero leakage (`duplicate_renditions.json`). GPU-score VRAM corruption caught at birth by G-Z5 + cured by the AC.R1 rescue (verdict 3/2,000 mismatches; re-gate 0.9993) — see `/mnt/v/output/avifgen-2026-08-06/_MANIFEST.json` |

| **avif-autotune-2026-09-04** (79,368 cells, 143 configs, 32 references x 2 corpora — ALL train-side by construction) | **T2 (BUILT 2026-09-04)** | Same shape as `avif944` above, same cause: the 32 refs were k-means-selected under `--parity 0` (`imazen26_recluster_even.py`), so every origin ends 0/2/4/6/8 and the canonical `{1,3,5}`/`{7,9}` buckets are **structurally EMPTY**. Registered even-only sub-split: `{0,2,4,6}` = train (26 origins), `{8}` = **`eval8`**, the leg-side holdout (6 origins — `1008` photo, `6018`+`6038` scan, `7058` plot, `8288` screenshot, `9118` ai-gen), never trained on. Invoked through `train_hybrid.py`'s new `SPLIT_RULE = "even_only_eval8"` hook, which hard-errors if ANY origin is not canonical-train and hard-errors if it ever emits a test row. **`eval8` is a leg-side holdout and must NOT be cited as the canonical one** — a real generalization estimate for an AVIF picker needs an odd-origin encode wave, which does not exist. Emitter: `zenmetrics scripts/jobsys/avif_autotune_view.py`; manifest `/mnt/v/zen/avif-autotune-2026-09-04/_MANIFEST.json` | N/A (our corpus; target is `ssim2`, the only corpus-wide scalar the AVIF DOE produced) | **The two corpora share all 32 filenames and 13 are different pixels** — rows carry `corpus` and features are per (corpus, image); the 19 pixel-shared refs have byte-identical feature rows in both tables (2,223/2,223). `backend` and `chroma` are perfectly collinear (svt 4:2:0, zenrav1e 4:4:4, 1,114 `av1C` boxes, 0 exceptions). HDR (`cells_hdr.parquet`) is a SEPARATE regime — never column-mix |

### §3d. The JPEG-AIC study family — one holdout family (registered 2026-09-01)

The three T0 rows above (**AIC-3 CTC**, **AIC-4 sample**, **JPEG-AI-SDR25**)
govern the *derived pair corpora*. They did not govern the material behind
them, which is the same content: the **10 AIC-3 CTC full-resolution sources**
(`00001`…`00010`) and their 600 encodes, every **620×800 BTC / PTC / IPTC
crop**, the **50 JPEG-AI VM full-resolution encodes**, and the **567,120 raw
triplet responses**.

**Registered rule `jpeg-aic-family-holdout-2026-09-01`
(`benchmarks/eval_annotations.json`): all of it is ONE holdout family, never a
training input, membership by CONTENT — a crop of a member is a member.**
Measured basis (`benchmarks/hfhuman_2026-09-01.md`): every PTC crop is a
pixel-exact crop of a CTC source (G1) and every PTC distorted stimulus a
pixel-exact crop of its CTC decode (G5, 250/250); BTC crops are 2× magnified,
~1.8× distortion-amplified renderings of a 310×400 CTC region (G2/G7). So
training on any member contaminates `aic3` **and** `aic4` **and** `sdr25` at
once — and `sdr25` is `bake_verdict`'s selection comparator, so that leak
changes which model *ships*. It would buy ≤ **900 labelled rows on 10 images**.
The alternative 6/4 reference partition is priced in the doc §3.3 (it leaves
`aic4` at 2 references and `sdr25` at 20 rows) and is **recommended against**.

Two corrections to the rows above, both registered in
`benchmarks/eval_annotations.json`:

* **`aic3-target-is-design-jnd`** — `ext_aic3`'s `human_score` is the CTC
  **design** value: `score.jnd == −0.25 × quality_level` **exactly on 600/600
  rows**. It is a 10-level ordinal ladder with 60 tied targets per level, and
  **121/600** of the levels are `method: estimated` (interpolated), image
  `00004` being 31/60. A cross-codec *equalization* test, not a MOS
  correlation.
* **`sdr25-is-aic4-subslice`** — the containment was already registered
  (Appendix I); what is new is that the two axes carry **different**
  reconstructions of the same stimuli, differing by up to **1.79 JND**.

A third, registered 2026-09-22:

* **`sdr25-372-root-table-orientation-unverified-2026-09-22`** — the 372-width
  eval roots (2026-05-15, both 2026-08-30 roots, 2026-09-05 post-C) carry a
  DIFFERENT `ext_sdr25.parquet` (sha256 `4f567646dcc6…`): 50 rows = **10**
  references × 5 rows, `human_score` ∈ {2, 6, 7, 9, 10} on every reference, no
  builder or provenance recorded. It is not the 5-reference × 10-level `q_jnd`
  table the JPEG-AI-SDR25 row above describes, so the `distortion` declaration
  does not cover it and its orientation is unverified. The 63 board cells that
  read it keep their stored per-reference values; a fresh 372-root verdict
  prints the opposite per-reference sign from them until the table is
  identified (`benchmarks/board_orientation_fix_2026-09-22.md` §1.4).

**~~NOT-REACHABLE~~ RECOVERED 2026-09-01** (`aic3-iptc-stimuli-recovered-2026-09-01`,
doc APPENDIX A): the 130 `IPTC_*` stimulus files were never a separate artifact
— the `IPTC` response table is the source paper's **PTC** experiment and its
stimuli are the plain `PTC_*` crops already in `test-images/PTC_images.zip`
(gate **G8**: 130/130 name map, 0 disagreements in 155,610 filename-vs-field
checks, level set exactly the published `{0,2,4,6,8,10}`, campaign shape
1,050 / 352 / 494 = the paper's own PTC row). No upstream archive exists and
none is needed; there is no licence or registration wall (CC BY 4.0). The
superseded entry `aic3-iptc-stimuli-not-reachable` is kept and points here.
**These 130 crops are HOLDOUT members like every other member of this family**
— they always were, being the same `PTC_*` pixels the rule already covered.

**EVAL axes built on it (2026-09-01, holdout-legal):** six arms over the
**567,120** scoreable responses — `ptc_native` / `btc_displayed` / `btc_native`
/ `aic4_all`, plus (APPENDIX A) **`iptc_native`** (130 stimuli, 900 triplets,
35,044 decided responses) and the pooled **`native_all`** (175 stimuli, 1,080
triplets, 41,973) — at all three live regimes, statistic
`panel --pairwise` = `zensim_validate::pairwise::agreement`. Artifacts
`/mnt/v/output/zensim/hfhuman-2026-09-01/` (+ `…/iptc/`). The unboosted,
native-scale leg is now 62,160 raw judgments rather than 10,290.

**AIC2026 joined this family 2026-09-19.** It is the same JPEG AIC activity,
by the same authors, and its 70 sources are a *different* content pool from the
AIC-3 CTC ten — but the family rule is about role, not pedigree: AIC2026 is
**eval-only, never a training input**, and a crop of a member is a member (its
own `PTC_`/`BTC_` 840×944 crops are members of it). Two things make it unlike
the other four members and must be said every time it is cited:

1. **There are no human labels in it.** The release contains objective IQA
   scores only. Nothing measured on it is "accuracy" — it is agreement between
   metrics, plus ladder behaviour (monotonicity, cross-codec consistency).
2. **CVVDP placed the levels.** Every ladder was built to be evenly spaced in
   CVVDP-estimated JND, so CVVDP (and, to a lesser degree, anything strongly
   correlated with it) is monotone on these ladders **by construction**. Reading
   a monotonicity ranking that puts CVVDP first as evidence about CVVDP is a
   category error.

What it *is* uniquely good for: **cross-codec consistency** — 17 codec
configurations over the same 70 sources at matched JND, which is the population
needed to ask whether one dial value means the same thing on JPEG, JXL, AVIF,
JPEG-AI and the learned codecs.

### SSIMULACRA2's own data usage (for fair comparisons)

Per the README (read 2026-07-02): ssim2 was Nelder-Mead-tuned on **CID22
(201/250 refs) + TID2013 + KADID-10k + KonFiG-IQA**. Consequences:
- CID22-49 is held out for BOTH us and ssim2 → fair comparison corpus.
- KADID/TID are **in-sample for ssim2** and train==val for us → integrity
  guards only; never scoreboard either metric there.
- ssim2's 70/80/85/90 anchors are JND-graded (side-by-side / in-place /
  flicker) — and our HQ instrument measures ssim2's within-band rank at
  cvvdp-agreement 0.48 in 85-100 → do NOT densify ssim2-labeled supervision
  in the ≥0.85 band (amplifies saturation); use cvvdp/butteraugli/human-JND
  there instead.

---

### §3b. Derived training corpora (registered builds)

| Build | Rows | Contract |
|---|--:|---|
| `bigcodec_hqdedup_{train,val}digits_2026-07-02` | 2,322,579 / 114,871 | canonical 7-dataset + jxl-hqfill, content-deduped (22.2% knob-no-op dups removed), LSD splits, C10<1% |
| `bigcodec_mm6_traindigits_2026-07-02` | 1,565,469 | 6 sidecar-covered datasets + hqfill, 4 metric columns joined (patched sidecar; mask-per-metric NaN 0.35%), deduped, LSD-train only. Bet-1 input; avif joins after its fleet fill |

## 4. Instruments (T3) — provenance + known content overlap

| Instrument | Content source | Overlap caveat |
|---|---|---|
| KADIS safety grid (`kadis_test_safetygrid.parquet`) | KADIS source_id%10==9, signed types excluded | Clean: test-split sources never train. Oracle mono ceiling 0.980 (cvvdp's own step-inversions) |
| HQ codec grid (`hq_codec_grid_2026-07-01.parquet` + refs sidecar) | 2026-06-24 GPU corpus = `train_renditions_2026-06-14` (1,482 imazen-26-family renditions) | **In-domain for bigcodec-trained bakes** (same content family, different encodes). Valid as a diagnostic; NOT a content holdout. Fix queued: rebuild on test-digit origins ({7,9}) only |
| Standard dial grid (`dial_grid_372col_2026-05-29.parquet`) | 2026-05-29 densified multi-codec q-sweep | Pre-dates the LSD rule; provenance vs training content not audited — treat as in-domain diagnostic until re-derived |

**Rule going forward: every instrument grid documents its content source and
its overlap class (holdout-content vs in-domain) here at creation time.**

---

## 5. Training-side val (checkpoint selection) — the kb25 lesson

The v47 recipe's val groups (safesyn/cid22_train/kadid/tid/konjnd) are all
T1/T2 — **train==val at the content level — so checkpoint selection is BLIND
to holdout collapse** (v50 kb25 collapsed to CID22 0.64 with val(geomean3)
0.909 looking healthy). Locked fix for new recipes: include at least one
truly-held-out val group (e.g. KADIS %10==8 val split, or picker val-digit
origins) with val_w > 0 so selection/early-stop can see generalization
failures. (Do NOT use T0 corpora for this — selection on T0 is training on
T0.)

---

## 5b. The weights ARE the mix — pair share is INDEPENDENT of row count (measured 2026-08-04)

Read from the trainer source (`zensim-validate/src/mlp_train/mod.rs:1892-2062`), not from
intuition: a training step picks a **group** by `train_weight / Σ train_weight`, then draws
two row indices **uniformly inside that group**. So a group's expected share of the epoch's
pairs is `train_w / Σ train_w` and **does not depend on how many rows it has.**

Measured on the incumbent SOTA-944 recipe (`H_co3abpg_s2507`), full table in
`benchmarks/data_integrity_sampling_mass_2026-08-04.tsv`:

| group | rows | row share | **pair share** | ratio |
|---|---:|---:|---:|---:|
| konjnd_bpg | 8,060 | 1.03% | **18.90%** | 18.3× |
| tid | 3,000 | 0.39% | 7.86% | 20.4× |
| cid22_train | 17,611 | 2.26% | 15.75% | 7.0× |
| bigcodec | 208,169 | 26.71% | 7.87% | **0.29×** |
| kadis | 50,000 | 6.42% | 2.36% | **0.37×** |

The extremes are 70× apart. `bigcodec`'s 208k rows are covered only ~4.5 times across a
whole 120-epoch run; `tid`'s 3,000 rows are re-covered ~2.6 times *per epoch*.

**Consequences for anyone writing or reading a recipe:**

1. **Never reason about a mix by row counts.** "bigcodec dominates the mix" is false — it
   is 26.7% of the rows and 7.9% of the gradient.
2. **Quote the pair share, not the row count**, whenever a recipe's composition is
   discussed in a doc or a commit message.
3. Two small corrections that fall out of the same source read: `ia == ib` is a *wasted*
   draw (`continue`), not a redraw; and a `rank`-mode group additionally drops
   **exactly-target-tied** pairs. Both are negligible here (kadid loses 0.9% of its share,
   tid 0.15%) — but the quantity that governs the drop is the **pair-collision
   probability** `Σ (n_v/N)²`, *not* the fraction of rows sharing a value. KADID reads
   99.60% by the wrong statistic and 0.876% by the right one.
4. **The weights have never been swept.** No measurement in the campaign varied them
   against held-out score. Until one does, no claim that the mix is well-balanced is
   supported — see `benchmarks/data_integrity_audit_2026-08-04.md` §7.

---

## 6. Compute placement — current LAN workflow

**September 7 chronology correction:** the July Hetzner-first instruction
below is historical. Use current LAN/local capacity through the resource-capped
workspace `run-heavy` owner; use [WAVE_PLAYBOOK.md](WAVE_PLAYBOOK.md) for
orchestration. Four obsolete Hetzner launchers were retired; the unique derived-
table builder remains. No external fleet is provisioned or retired implicitly.

### Historical July placement rule

Per user 2026-07-02 (twice): **ALL slow work runs on Hetzner train boxes by
default** — trainer cells, sweeps, corpus/parquet builds, anything minutes-
scale. The workstation is for orchestration, seconds-scale evals, analysis,
and commits only. Rationale (learned the hard way same day): local heavy jobs
contend with each other (an uncapped parquet build OOM'd next to two 40G
training cgroups), die with harness crashes (nohup'd chains lost twice), and
occupy the interactive box; the CCX63 has 48 dedicated cores/192GB, runs 6+
cells concurrently, and its nohup jobs survive local crashes. Agents rsync
data + ssh-control boxes directly (identity: `~/.ssh/zen-arm-dev`). vast.ai
remains GPU-metrics-only. Standard flow: rsync the canonical parquets +
manifests + a static trainer binary → run cells under nohup with per-cell
logs → rsync verdicts/bakes back → all results land in
`benchmarks/`-committed docs exactly like local runs. zenfleet-hetzner is the
scaled alternative when a grid is big enough to warrant the job system.

---

## 7. Action items opened by this doc (tracked)

1. ~~Fix the wrong "human_score = MCOS/100" note in v47/v48/v49/v50 manifests~~
   (fixed in the commit introducing this file — it is ssim2_gpu/100).
2. Rebuild the HQ instrument grids on test-digit ({7,9}) origins → true
   content-holdout diagnostics (wave-4 prerequisite).
3. Add a held-out val group to the next recipe generation (§5).
4. SDR25 JND reconstruction → T0 corpus ingest (+ dHash audit).
5. Audit the 16 decode-failed imazen-26 screen PNGs (or exclude them from
   training corpora).
6. KADID/TID d≤10 flagged-pair user review (pending since 2026-05-14).
7. **Carry the quality/severity key into every canonical table** (opened by the
   2026-08-04 integrity audit, F-5). `bigcodec` kept `encoded_filename` and
   `kadis` kept `source_id`+`score_ssim2_gpu`, and both proved ladder
   monotonicity cleanly; `safesyn`, `cid22_train`, and `konjnd_bpg` carry only
   `ref_basename` + `human_score`, so 17.5% of the mix's rows have no auditable
   ladder at all. This is a promotion-script change, not a re-extraction.
8. **Sweep the 11 mix weights against held-out score** (F-2). Pair share is
   independent of row count (§5b), so the weights *are* the mix, and they have
   never been varied experimentally. Highest-leverage knobs the audit surfaced:
   `konjnd_bpg` (18.9% of pairs off 1.03% of rows), `bigcodec`+`ttbig` (53.4% of
   rows, 15.7% of pairs), `tkadis` (item 9).
9. **Resolve the `tkadis` conflict** (F-1). The kadis teacher twin ranks its own
   rows at ρ=0.25 vs the base leg while outweighing it 3.3×; the clip/affine
   explanation is falsified. Either zero its weight or rebuild it from a teacher
   that generalizes to the KADIS distribution.
10. **Give the 9 metric/teacher-target legs an internal-consistency gate.** They
    cannot be orientation-checked against humans (F-3), so A4 ladder monotonicity
    is their only handle — and item 7 is its prerequisite.
7. **Multi-metric backfill of the 5.7M canonical corpus — LANDED as sidecars
   (2026-07-02; status corrected 2026-08-04, was "IN FLIGHT")** — the
   authoritative per-encode metric sidecar is
   `/mnt/v/datasets/fill4-6codec-2026-07-01/fill4metrics_sidecar_patched_2026-07-02.parquet`
   (4.18M rows, 6 codecs, key=`encoded_filename`/`encode_sha`) + the JXL
   near-lossless top-up `hqfill_7metric_sidecar_2026-07-02.parquet`; joined
   probe table: `bigcodec_mm6_traindigits_2026-07-02.parquet` (audited
   2026-07-16 — CLAUDE.md "RECURRING PRIORITIES" carries the paths + column
   naming caveats). The remaining sub-items below are still open where a
   rebuilt canonical-view parquet is what they need: (a) it must arrive as NEW
   dated files/sidecars joined on content-addressed keys, never in-place
   rewrites of files a manifest references (§1.4); (b) rebuild
   `bigcodec_multimetric_<date>.parquet` with per-zone targets — the wave-4
   HQ-band supervision substrate (cvvdp/butteraugli in the ≥0.85 band where
   ssim2 saturates); (c) re-derive the digit-split train/val files from it.
   Until then, ssim2-target waves (v51) validate the held-out-val selection
   fix, NOT the final supervision design.

### §3c. HF-near-lossless family (registered 2026-08-29 — was missing from this registry)

| Asset | Tier | Split | Notes |
|---|---|---|---|
| `hf_nearlossless_{train,val}` (372-col, 1,200 rows / 200 refs; canonical-2026-07-15) | T2 train + within-family val | REF-level: sorted refs, every 4th → val (150/50, zero ref overlap) | jxl near-lossless refit corpus; target = ssim2/100; manifest `_MANIFEST_hf_nearlossless.json` |
| `tbig_hf_pure` (944, 11,941 rows / 195 origins; sdr-pure-2026-08-28) | T2 train | LSD **train digits {0,2,4,6,8}** — verified 2026-08-29, zero origin overlap with any eval slice | The l944_hf gram source = jeweler-loupe's training data + north-anchor's `tbig_hf` group. Was UNMANIFESTED until 2026-08-29 (now in the root `_MANIFEST.json` with shas, incl. the `_jw`/`_jwfold` variants) |
| `ext_hfnlproxy` @ ext944 root (7,717 rows / 87 origins) | T3 selection instrument | LSD **VALIDATE digits {1,3,5}** — the 2026-08-28 D1 re-slice (methodology audit: selection had been consuming TEST families; fixed) | ⚠ the root's `_MANIFEST_eval_slices.json` still records the pre-D1 11,356-row cut — STALE for this file. ⚠ bake_verdict's display string "944 TEST views" is now WRONG for this slice (it reads validate views) |
| `ext_hfnlproxy` @ 372 root (11,356 rows / 74 origins) | T3 touch-once terminal | LSD **TEST digits {7,9}** (pre-D1 cut, built 2026-08-27) | What shipped-B's 2026-08-29 remeasure consumed. Retired from selection per D1; terminal reads only |

**⚠ Cross-generation comparability (measured 2026-08-29):** three slice
generations are simultaneously live on the board for imazen26/nonphoto/
hfnlproxy — era-candidate fullevals (pre-D1 test-family: n 7,869/7,255/9,167),
the 2026-08-28 validate-family cuts (6,953/6,142/7,717), and B's 372-root
test-family cuts (7,844/8,241/11,356). Numbers across generations are NOT
same-ruler; hfnl shifts up to 0.09 between populations (north-anchor 0.752
test-family vs 0.699 validate-family). Same-ruler comparisons must rescore on
one generation — see the balance campaign's split-audit section for the
validate-slice rescores of the era candidates.

---

## 8. SPLIT POLICY v2 (registered 2026-08-29, user directive: "test and eval won't be trained on")

**Universal invariant:** any content (ref/origin/source) that appears on an
eval surface (board rank row, gate population, selection instrument, terminal
read) is NEVER in a train-side group of the same feature family. Deterministic
per-dataset rules; enforcement is mechanical, not disciplinary.

**Owner tool:** `scripts/canonical_corpus/check_split_compliance.py` —
audits any recipe (`--group …`) or any bake's embedded repro
(`--from-fulleval …`) against the registered surfaces; hard-errors on
overlap; registered train==val guards print WARN. Run it before every
new-recipe train; the WAVE_PLAYBOOK gains this as a step.

**Tier names:** TRAIN / SELECT (selection eval; digits {1,3,5} where the LSD
rule applies) / TERMINAL (touch-once; digits {7,9}). kadis keeps its
registered %10 rule (<8 / 8 / 9) as a documented exception.

**Per-dataset v2 assignments (views built 2026-08-29, manifests
`_MANIFEST_splitv2_views_2026-08-29.json` at the ext944 root + the 372
mirror manifest):**

| dataset | TRAIN | SELECT | TERMINAL | measured reasonableness |
|---|---|---|---|---|
| KADID | 40 refs / 5,000 rows (`ext_kadid_train`) | 25 refs / 3,125 | 16 refs / 2,000 | **REASONABLE** — 6-model rescore: buckets agree within 0.01–0.03, rankings identical |
| TID | 12 refs / 1,440 | 9 refs / 1,080 | 4 refs / 480 | SELECT reasonable (rankings hold); **4-ref TERMINAL is level-shifted (+0.03–0.05 for every model)** — sanity guard only, never a ranking surface |
| KonJND | BPG half (403/101 refs %10 — unchanged; JPEG never trains) | JPEG 404 refs | JPEG 100 refs | SELECT tracks the full set; **100-ref TERMINAL is VOLATILE (amber 0.50→0.24, models reorder)** — touch-once sanity only |
| KADIS | `kadis_944_ssim2_train_2026-08-29.parquet` (40,040 rows, %10<8) | %10==8 | %10==9 (safety grid) | **the 50k leg VIOLATED the rule (19.92% val/test sources; every 944-era bake trained on them)** — fixed view built; recipes switch next generation |
| imazen-26 family | digits {0,2,4,6,8} | {1,3,5} (the D1 slices) | {7,9} | already compliant (verified: 0 overlap for every tbig/hf leg) |
| CID22 | 201 refs, ssim2 targets | — | 49-ref gold (also reused for selection — documented deviation) | verified 0 ref intersection |
| KonFiG | `konfig_originsplit_train` ONLY for new recipes | originsplit_val | originsplit_test | full-table-trained models keep their annotation |
| hdrmix | 38 train-digit origins (verified {0,2,4,6,8}) | val file carries {1,3,5}+{7,9} — carve TERMINAL {7,9} at next HDR wave | — | queued |
| safesyn / teachers | all-train | — | — | rule: an all-train dataset may NOT gain an eval role without splitting first |
| T0 estate (aic3/aic4/sdr25/csiq/live/UPIQ…) | NEVER | — | eval-only | checker hard-errors on any T0 name in a train group |

**Recipe migration:** existing bakes are NOT retrained; they carry the audit
verdict (north-anchor: compliant except the kadis leg + kadid/tid guards).
Every NEW recipe uses the `*_train` views (kadid/tid/kadis/konfig) and must
pass the checker. Board kadid/tid rank rows migrate to the SELECT views
(honest generalization for future models; equal-footing memorization reads
for existing ones — both sides trained on those refs).

### §8.1 TID RETIRED TO TRAIN-ONLY (USER RULING 2026-08-29)

TID2013 is **T2 train-only** effective immediately: all 3,000 rows / 25 refs
trainable; **no TID eval surface exists** — board tid rows are historical
guard reads, never ranking signal (the gauntlet legend says so). The
`ext_tid_{select,terminal}` views built earlier the same day are RETIRED
UNUSED for eval (files kept; they demonstrated the reasonableness
measurement). Rationale: 25 refs is too small to fund an honest eval tier
AND a train share; its distortion mix is ~95% non-compression; ssim2 is
in-sample on all of it.

### §8.2 REGISTERED DEFECT: `live` targets are NOT rank-identical across roots (found 2026-08-29)

The single-ruler audit's target-identity gate caught it: the 372-root LIVE
table's `human_score` ordering disagrees with `ext_live`'s (|SROCC| < 1
between the two target vectors on the same 779 pairs). One of them carries a
target defect or a different label version. LIVE is EXCLUDED from
cross-regime comparisons until audited (owner: canonical_corpus; annotate
any cross-root live citation). Registered in `eval_annotations.json`.

## Exposure ledger — 2026-09-19: CID22-49 A/B split (DVIFM Phase-2d)

Per "Record each batch's exposure" above: the CID22 49-reference
human-scored set was split into **A (25 refs, fit-allowed for the
Phase-2d standalone-DVIFM constants fit only)** and **B (24 refs, sealed
until one frozen descriptive read)** under recorded seed 20260919. The
full ref lists, the user direction, what is fitted (≤~90 scalar constants
+ per-domain affine; no MLP/feature-selection/checkpoint-selection), and
the consequence (artefacts consuming A-derived constants quote CID22 only
as `CID22-B(24)`) are recorded in
`docs/DATASET_HISTORY.md` under 2026-09-19 and preregistered in
`benchmarks/dvifm_screen2d_prereg_2026-09-19.md` §4–§5. The CID22 registry
row above is otherwise unchanged: the 49-ref set remains holdout-only for
every other purpose, and no zensim model fit consumes these labels.

## Exposure ledger — 2026-09-22: AIC-4 frozen read, crops + full resolution

Per the September 14 clarification (AIC-4 public sample = published TEST,
assessable by frozen candidates with recorded exposure). Registered BEFORE the
read (board-orientation lane, `benchmarks/board_orientation_fix_2026-09-22.md`):

- **Population:** the AIC-4 public sample, 5 sources x 6 codecs x 10 levels =
  300 stimuli, scored twice — on the `PTC_images` crops the study showed and on
  the `full_resolution_images` encodes they were cut from. Labels: committed
  `site/data/parquet/aic4_sample.parquet` (`human_jnd`, distortion-oriented).
- **Models (all frozen before this read, no member changed):** named profiles
  `PreviewV0_2`, B, C, D; `R915_y60_h32_ens5` and `R915_basic228_h128_ens5`
  (member hashes as in `recovery/calibrated/FROZEN.json`); the MT914 matched
  B/D bakes; our fast-ssim2 and butteraugli (`peer_metric_pairs`); the
  organisers' published columns (crops only).
- **Statistics:** global |SROCC| / |KROCC| and the orientation-aligned sign,
  plus the same per source, all from the Rust `panel` owner.
- **Use:** descriptive only. Nothing is fitted, calibrated, selected or tuned on
  it. Five sources is a sanity check (is any ladder backwards?), never a
  selection axis. Prior AIC-4 exposure of these models (the 2026-09-14/15
  public-test panels on the feature tables) is disclosed; this is not a fresh
  independent holdout confirmation.

## Exposure ledger — 2026-09-23: Rev4 step-1 E1 regime analysis (read-only)

Purpose "rev4 step-1 e1". Registered before the read in
`benchmarks/rev4_e1_prereg_2026-09-23.md`. Existing per-pair predictions of frozen
models and peers only; nothing is fitted, calibrated, selected or tuned on these reads.

- **Populations read (labels, evaluation only):** CID22-A(25) rows only (CID22-B(24)
  sealed and not read; its rows are dropped by `ref_path` before any target value is
  extracted); CSIQ (866); KonJND JPEG SELECT (404; TERMINAL-100 not read); AIC-3 CTC (600);
  AIC-4 sample crops and full resolution (300 each); SDR25 q_jnd table (50); the JPEG-AIC
  forced-choice responses (AIC-3 BTC, AIC-3 IPTC, SDR25 BTC/PTC) as already scored by
  `hfhuman_2026-09-01`.
- **Not read:** LIVE (target defect §8.2), KADID/TID (TRAIN-role), any secret holdout.
- **Models:** B, C (`W10L9PH_s4004`), D, R915 fast/rich, PreviewV0_2 and the context arms
  already scored in the forced-choice tables; all frozen before this read, all with prior
  exposure to these populations (disclosed; not a fresh holdout confirmation).
- **Statistics:** within-band pairwise ordering accuracy and band SROCC by quality band,
  reference-clustered bootstrap, all from the `panel` owner.

## Exposure ledger — 2026-09-23: rev4 step-1 e4

Purpose: "rev4 step-1 e4" (`docs/REV4_EXPERIMENTS_2026-09-23.md` §E4; prereg
`benchmarks/rev4_e4_prereg_2026-09-23.md`). Frozen existing board candidates
only (ladder-board 2026-09-06: 359 width-944 cells, 67 width-372 cells); no
fitting, calibration, feature/hyperparameter or checkpoint selection consumes
these reads. Held-out reads, evaluation only:

- **CID22-A(25)** human MCOS — per-candidate SROCC from A-only rows (rows
  filtered by `ref_basename` membership before any label is loaded). CID22-B(24)
  is not scored and not read.
- **AIC-3, KonJND-504, CSIQ** — the stored per-candidate `rank.<corpus>` SROCCs
  already in each fulleval (no re-scoring).
- **LIVE, AIC-4, imazen26, nonphoto** — stored per-candidate SROCCs, read only
  as terms of the registered composite.
- KADID (TRAIN-role) is re-scored only as a features-root identity check.

No secret holdout is read.

## Exposure ledger — 2026-09-23: rev4 step-1 e1b (cross-codec split, read-only)

Purpose "rev4 step-1 e1b". Registered before the read in
`benchmarks/rev4_e1b_prereg_2026-09-23.md`. Existing per-stimulus scores of frozen models and
peers only (E1's assembled tables); nothing is fitted, calibrated, selected or tuned.

- **Populations read (labels, evaluation only):** CID22-A(25) rows only (CID22-B(24) sealed;
  dropped by `ref_path` in E1's assembler before any target is extracted); CSIQ (JPEG and
  JPEG2000 stimuli); AIC-3 CTC (600); AIC-4 sample crops and full resolution (300 each); the
  JPEG-AIC BTC responses (AIC-3 BTC, SDR25 BTC) as scored by `hfhuman_2026-09-01`.
- **TRAIN-role, description only:** TID2013 JPEG/JPEG2000 stimuli.
- **Not read:** CID22-B, LIVE, KADID, any secret holdout.
- **Models:** B, C (`W10L9PH_s4004`), D, R915 fast/rich, PreviewV0_2; all frozen, all with prior
  exposure to these populations (not a fresh holdout).
- **Statistics:** same-codec vs cross-codec within-reference pairwise ordering accuracy,
  reference-clustered bootstrap, from the `panel` owner.

## Exposure ledger — 2026-09-23: rev4 E5a rendering-regression corruptions (e5a-render)

Purpose "rev4 e5a-render" (`docs/REV4_EXPERIMENTS_2026-09-23.md` §E5; prereg
`benchmarks/e5a-render_prereg_2026-09-23.md`). The lane synthesizes
correct/broken rendering-implementation pairs and a benign-drift set, then
scores fixed metric arms. **No human labels exist or are read anywhere in this
lane** — every ground truth is the generator's own twin pair.

- **Read (pixels, generation only):** the 12 canonical TRAIN imazen-26 sources of
  `canonical-corruption-2026-09-08` (`train-sources.json` sha256
  `4f7ee719520d2e71a672ddc5b83aba9aa24960c0684c8361d306aea119c03a71`), at the
  canonical longest-side-256 rendition plus the longest-side-512 cleanpicker
  rendition of the same 12 origins (per-file sha256 in the prereg §2).
- **Not read:** CID22-B (sealed), CID22-A labels, AIC-3/4, AIC2026, KonJND
  validation, KonFiG test, KADID terminal references, SDR25, LIVE, any secret
  holdout, and the 8 canonical validation sources.
- **Models scored:** frozen profiles B and D and the frozen Rev3 ensembles
  `R915_y60_h32_ens5` / `R915_basic228_h128_ens5` (member sha256s in prereg
  §11), plus ssim2/butteraugli/DSSIM/GMSD peers and a TRAIN-calibrated testing
  arm. No model is trained, refit or selected on held-out data.
- **Statistics:** origin-clustered bootstrap (B=2000, seed 20260923) and
  `panel --batch` SROCC; thresholds from the benign-drift set only.

## Addendum — 2026-09-20: `joint-core-v1` registered as a TRAIN-role view

`joint-core-v1` (`/mnt/v/output/zensim/joint-core-v1/`, 52,963 pairs,
`_MANIFEST.json` at the root carries `build_commit` + per-input sha256 +
cluster/seed rules + per-leg kernel provenance) is a **TRAIN-role view**.
It assigns no new role to any corpus: every source corpus keeps its
registered role and the view only consumes rows already admissible for
training.

Composition (measured, `coverage_report.json`): mid band 55.9% /
small 24.4% / tiny 19.7% (rung ≤1024 px); photography 81.6%; screen,
document, line-art guards 5.6% each; AI-labelled 1.5%. Legs:
fresh_imazen26 62.2% / cid22 14.2% / human 9.6% / fresh_safesyn 8.7% /
hdr 4.7% / konfig 0.6%. HDR rows are PQ-regime features — stored under
`features/hdr_pq/` with their own manifest, never column-mixed into SDR
tables or SDR DVIFM caches.

Selection: k-means (seed 17) centroid-nearest over each leg's `feat_*`
embedding; singleton clusters kept whole; imazen-26 contributes TRAIN
manifest ids only (even last digit), verified ref-disjoint from
`codec_development`, `safesyn_development`, `cid22_development`,
`human_development` and every eval corpus. dHash-64 audit against
CID22-49, AIC-3, AIC-4, AIC2026, SDR25, KonJND-val and KonFiG-test
produced 21 flags, each adjudicated false-positive by pixel RMSE/NCC —
flags are recorded, nothing auto-quarantined.

Kernel provenance per leg is recorded in the manifest: fresh legs are
plain Mitchell `sharpen=0`; cid22/human/konfig are native (no resample);
hdr is Mitchell `resize_sharpen=10` on linear PQ; the earlier PIL-Lanczos
picker renditions were rejected and re-rendered. A per-leg
`kernel` column is carried in `pairs/pairs_core.tsv`.

## Ruling — 2026-09-23: Rev4 feature-bank potential and leave-one-dataset-out roles (user decision)

These rulings answer the decisions in `docs/REV4_FEATURE_BANK_PLAN_2026-09-23.md`.

- **D2, leave-one-dataset-out (LODO).** The user accepted the design's proposal.
  - **Withheld from every fold** (untouched Rev4 confirmation): CID22-B(24), the AIC-4 sample, KonJND JPEG
    (SELECT and TERMINAL), CSIQ, KADID TERMINAL, LIVE (target defect) and every secret holdout.
  - **Fold sets:** KADID, TID, KonFiG, KonJND-BPG, CID22-A(25), AIC-3 and KADID SELECT. Each is recorded
    here as **LODO-exposed** when its fold runs.
  - **MCL-JCI** joins the folds only if D3 gives it a fitting role. Until then it is confirmation-only.
  - **Quarantine:** fold models live under `/var/tmp/rev4-featpot/lodo/` with the prefix `LODO_`. They are never
    packed, never on the board, and never used to select a shipped recipe.
- **D1, in-sample potential.**
  - **Fitted in-sample for potential estimates:** CID22-A(25), AIC-3 CTC, KADID SELECT and KonFiG originsplit
    val. They become **potential-exposed** and can never again be quoted as held out for any choice the potential
    run informs.
  - **Untouched for Rev4 confirmation:** CID22-B, the AIC-4 sample, KonJND JPEG, CSIQ and secret holdouts.
- **D4, feature extraction of held-out pixels.** Allowed now, pixels only. Features may be extracted for every
  set, including CID22-B, the AIC-4 sample, KonJND, CSIQ and MCL-JCI. **No label is read** until confirmation,
  and every extraction is logged. This relaxes the 2026-09-13 no-extraction ruling for these sets only.
- **D5, adoption bar for a candidate feature family.** All of these must hold:
  - stability-selection frequency ≥ 0.6;
  - nested-CV gain ≥ +0.005 SROCC, with the CI excluding zero on ≥ 2 human sets;
  - ~~it pays its measured runtime cost;~~ **Removed 2026-09-24 by the user:** "remember not to reject things for
    the cost budget, and track all things and code and results of those you have. we can optimize and make things
    optional". Cost is measured and reported for every family. It never accepts or rejects one. Every candidate
    stays in the evaluation, its code stays landed (default off), and its results stay recorded.
  - it keeps the dial contract under the non-negative-distance head.
- **D3 (MCL-JCI's role)** is pending the datasets-lane proposal. The default is confirmation-only, as the natural
  test set for the JPEG response-shape question.

## Exposure ledger — 2026-09-23: MCL-JCI datasets-lane orientation and DSSIM panel

The datasets lane parsed **all 5,000 MCL-JCI JND labels** at about 12:42 UTC (06:42 -06:00) and read them for per-source label/QF orientation checks. It then compared DSSIM(QF100 JPEG → QFq JPEG) with the human `jnd_dist` label on the full 4,950-pair QF1–99 grid (`zen_stats.panel`: SROCC 0.8664, PLCC 0.9048, KROCC 0.7144, PWRC 0.9880; signed SROCC +0.8664). No zensim candidate was scored, fitted or selected on these labels. This read was authorized for orientation in the datasets brief, but the lane did **not** commit the required preregistration before reading labels; that process defect is recorded in `benchmarks/datasets_WORKLOG.md` and `benchmarks/rev4_datasets_inventory_2026-09-23.md`. No preregistration was backdated. MCL-JCI remains confirmation-only pending D3; this exposure must accompany future confirmation claims.

## Exposure ledger — 2026-09-23: Rev4 feature-bank pixels-only extraction of held-out sets (ruling D4)

The featbank-extract lane ran the pinned extractor (`extract-native-admission`, sha256 `7c7ffbbf…`, formula revision 3, root form sqrt, `--full-944`) over the **pixels only** of the held-out sets that ruling D4 allows: CID22-B (24 references, 2,100 pairs), the AIC-4 sample (300), KonJND JPEG SELECT (404) and TERMINAL (100), CSIQ (866), MCL-JCI (5,000) and KADID TERMINAL (2,000). Outputs are `keys.parquet` plus feature parquets under `/var/tmp/rev4-featbank/bank/<set>/`; **no labels file exists for these sets.** The source pair, audit and feature CSVs that replicated these sets' `human_score` columns were sealed under `/var/tmp/rev4-featbank/_sealed/` the same day (review correction 4, option a). The assembler only copied and equality-bound those columns; **no label was analysed, and no statistic was computed on them.** Record: `benchmarks/rev4_featbank_extract_2026-09-23.md`; worklog `benchmarks/featbank-extract_WORKLOG.md`. These sets keep their confirmation-only roles.

## Exposure ledger — 2026-09-24: C8 gmsbank peer GMSD scoring, pixels only, all 18 bank sets

The gmsbank lane scored exact GMSD/GMSM (zenmetrics `gmsd` crate, scorer sha256 `2ed8f676…`) on the **pixels only** of
all 18 Rev4 bank sets (248,983 pixel keys, 249,227 stimulus rows). That includes the held-out and confirmation sets:
CID22-B (authorised pixel-only under ruling D4), the AIC-4 sample, KonJND JPEG SELECT/TERMINAL, CSIQ, MCL-JCI and
KADID TERMINAL. Output: `/var/tmp/gmsbank/peer_gmsd/<set>.parquet`, keyed by `pair_key` and verified against the
bank's decoded-pixel hashes. **No label file was opened and no statistic was computed on held-out labels.** The
columns feed the preregistered P2 arm only on its admitted D1/D2 sets. The C8 chroma calibration used TRAIN pixels
only and needs no entry.

## Exposure ledger — 2026-09-25: restore-cuts families, pixels-only extraction, all 18 bank sets

The restore-cuts lane ran the extractor `extract_features_372col` (source landed as `f3f021f6`, quarantine id
`ec5b1821`; binary sha256 `4ea8f333…`; formula revision 3, root form sqrt, legacy-rgb8, `--restore-cuts`) over the
**pixels only** of all 18 Rev4 bank sets (248,983 pixel keys, 249,227 stimulus rows), producing the default-off
families `mapdev`, `z1max`, `gmsnative` and `dvifmgate` (f1502..f1824). That includes the held-out and confirmation
sets under ruling D4: CID22-B, the AIC-4 sample, KonJND JPEG SELECT/TERMINAL, CSIQ, MCL-JCI and KADID TERMINAL. The
pair TSVs carry `human_score=0`; **no label or `_sealed` file was opened and no statistic was computed on held-out
labels.** Output: `/var/tmp/restore-cuts/bank/<set>/features__restore_*.parquet`, keyed by `pair_key`. Record:
`benchmarks/rev4_restore_cuts_2026-09-24.md`.

## Exposure ledger — 2026-09-22: CID22-49 duplicate, B(23) correction, full-set DVIFM fits, NNCD

Recorded in `docs/DATASET_HISTORY.md` under 2026-09-22, committed before the
reads they govern:

- **CID22-A/B split leak.** `844297.png` (A) and `3316926_opo25u.png` (B) are
  the same picture; CID22-B(24) → **CID22-B(23)** (2,011 pairs). The spent
  2026-09-21 read was re-scored on 23 references from its own per-row scores —
  a correction of the same read, not a second read; conclusions unchanged.
- **Full CID22-49 human labels fitted** for the dvifmish "full CID22 human,
  author-style" constants source (DVIFM constants + level/channel weights +
  3-parameter output map only). Those presets carry zero held-out CID22 claim;
  their CID22 numbers are fit-domain.
- **CID22-A** additionally serves as the human selection leg of the dvifmish
  variant screen (SafeSyn-fitted arms only); screen winners quote CID22 as
  CID22-B(23).
- **KonFiG-IQA `originsplit_val`** (436 pairs, 8 references; its registered
  SELECT view) is the screen's second human selection leg. Recorded here on
  2026-09-23 after the screen had used it (rounds 1 and 2); the use is the
  view's registered purpose. Screen winners have no held-out KonFiG claim on
  those references; `originsplit_test` was not read.
- **NNCD-IQA** registered EVAL-only (row in §3). Not content-disjoint from
  TID2013.
- **CID22-B(23)** read a second time, by frozen dvifmish presets not fitted on
  CID22-49 (September 14 rule; the 2026-09-21 read disclosed as prior exposure).
- **AIC-4 public sample** (300 pairs; PTC crops and full resolution) read by
  frozen dvifmish presets and frozen peers under the September 14 rule; target
  distortion-oriented, |SROCC| reported; prior zensim exposure disclosed (SDR25
  slice = seed-selection oracle). Nothing fitted, calibrated or selected on it.

## Exposure ledger — 2026-09-23: NNCD-IQA first read by the frozen dvifmish presets and peers (reported 2026-09-25)

The frozen peers (fast-ssim2, zensim B and D, R915 Rev3 ensembles) were scored on NNCD at 2026-09-23T03:50:26Z;
another session's join audit saw their SROCCs around 05:10Z (before the dvifmish variant screen closed at
06:19:34Z), and the dvifmish lane recomputed them at 05:14Z only to verify the corrected join. The 27 frozen
dvifmish presets (frozen at dvifmish `f562c519`, 06:55:12Z) were first scored on NNCD at 09:50:17Z, inside the
final repro run. Nothing was fitted, calibrated or selected on NNCD, and no preset changed after the read. NNCD is
EVAL-only and **not content-disjoint from TID2013**: its 16 references contain the 16 TID2013 references as
unscaled crops (NCC 1.0000), so a model fitted on TID2013 rows has seen NNCD's scenes. Detail: DATASET_HISTORY
2026-09-22 NNCD entry and addenda; record `benchmarks/dvifmish_eval_2026-09-22.md`.

## Exposure ledger — 2026-09-23: canonical corruption packet 2026-09-08 read by the dvifmish presets and peers (reported 2026-09-25)

Both splits of the canonical corruption packet (validate 8 origins, train 12; imazen-26 origins at longest side
256; 713 corruption attempts plus q10/q20 native JPEG anchors per origin and the accepted honest native supplement)
were scored by the frozen dvifmish presets and the frozen peers with the owner protocol (`corruption_gate_eval.py`
summarize, via `corruption_eval.py`). The packet carries no human labels; the only label read was its own
`is_corruption` flag. Characterisation only (fraction below the honest q20/q10 anchors, detection at a matched
honest false-positive rate); nothing was fitted, calibrated or selected on it. Peer scores were re-joined through
the extractor's reference-sorted row order first (a positional join had been wrong). Record:
`benchmarks/dvifmish_eval_2026-09-22.md` §6.

## Exposure ledger — 2026-09-24: fleet transport of Rev4 potential-fit inputs (ruling D1)

The `fleet-fits` lane is preparing a content-addressed input archive for the 960 preregistered P0 feature-potential MLP
cells. Under D1, the archive will copy the admitted feature/label bytes for CID22-A(25) (2,192 rows), AIC-3 CTC (600),
KADID SELECT (3,125), and KonFiG originsplit val (436), along with the four TRAIN-role sets. This entry precedes that
archive's creation. The transport step verifies source receipts and copies bytes without decoding label columns,
computing statistics, or changing the potential lane's fit procedure. The potential lane's fit script subsequently reads
these labels for the already-authorized in-sample potential estimates; these four populations remain potential-exposed,
never Rev4 confirmation holdouts for choices informed by these fits. CID22-B, the AIC-4 sample, KonJND JPEG, CSIQ, and
secret holdouts are outside the archive. The exact archive SHA-256, source manifest hashes, and fleet program identity
will be recorded in `benchmarks/fleet-fits_WORKLOG.md` in the quarantine zenmetrics workspace before the fleet
declaration.

## Exposure ledger — 2026-09-25: fleet transport of P2 and D2 potential-fit inputs (ruling D1)

The `fleet-fits` lane, under the coordinator's GO of 2026-09-25, is preparing a second content-addressed input archive
for the preregistered P2 MLP grid (960 cells) and the D2 source-held-out MLP replicate grid (210 cells). It is the P0
archive's populations (the four TRAIN-role sets, CID22-A(25), AIC-3 CTC, KADID SELECT, KonFiG originsplit val) plus
KonJND BPG val (2,020 rows, the D2 evaluation substitute for konjnd_bpg_train), the label-free reviewed peer GMSD/GMSM
tables for exactly those nine populations, and the D2 fold tables with their receipts for the arms r0, p2 and p2_perm.
The transport step verifies source receipts, table hashes and peer parquet hashes, and copies bytes without decoding
label columns or computing statistics. The potential lane's fit scripts subsequently read these labels for the
already-authorized in-sample potential and source-held-out estimates. All nine populations remain potential-exposed,
never Rev4 confirmation holdouts for choices informed by these fits. CID22-B, the AIC-4 sample, CSIQ, KonJND JPEG, KADID
terminal, the remaining peer-bank sets and secret holdouts are outside the archive; the packer refuses any file name
naming them. The exact archive SHA-256 and source manifest hashes will be recorded in `benchmarks/fleet-fits_WORKLOG.md`
in the quarantine zenmetrics workspace before the fleet declaration.

Addendum, recorded at landing (2026-09-26), after the archive existed: the P2/D2 archive is
`/var/tmp/fleet-fits/data-p2d2.tar.gz`, 703,345,489 bytes, SHA-256
`4b2c0434c5eb7edf8fb0813677ccdb9867a5a4792f2d2f082fda59c26e6d8839` (the `data_sha` of every cell in
`/var/tmp/fleet-fits/fit-manifest-p2.json`; recomputed with `sha256sum` at landing). The promised worklog entry was not
made: `benchmarks/fleet-fits_WORKLOG.md` lives in the separate zenmetrics fleet-fits workspace and was not updated, so
this record is the zensim-side copy. Timing: this entry's commit time is 05:37:50 (-06:00) and the archive's file mtime
is 05:38:23 (-06:00), 33 s later; the timestamps are consistent with the entry preceding the archive's completion but do
not demonstrate it, so treat 'prerecord' as unproven for this archive.

## Exposure ledger — 2026-09-23: rev4 E2b cvvdp-safesyn (TRAIN-role selection read)

Purpose "rev4 e2b cvvdp display" (lane `cvvdp-safesyn`; prereg
`benchmarks/cvvdp-safesyn_prereg_2026-09-23.md`, registered before the read).
Frozen metric predictions only — CVVDP is a fixed scorer, nothing is fitted,
calibrated or tuned on these reads; the registered selection rule consumes
them.

- **Populations read (TRAIN-role labels, selection per prereg):** KADID-train
  (5,000 pairs / 40 refs, `kadid_train_pairs.tsv`), TID-train (1,440 / 12,
  `tid_train_pairs.tsv`), KonFiG-train (327 / 6, `konfig_train_pairs.tsv`,
  q_jnd consumed only as triplet ordering). These corpora are TRAIN-role;
  using them for the registered display selection is their registered role.
- **Not read:** CID22-A/B, LIVE, AIC-3/AIC-4 labels (Part 0 reproduced stored
  score TSVs only, never labels), any secret holdout, any eval-role corpus.
- **Scorer:** CVVDP `cvvdp_cpu_imazen_v0_1_0` at 7 named display presets via
  `--display-model` (zenmetrics `19d6dd8e`, binary sha256 `f8c352ee…`);
  `modern_oled_phone_indoor` parity-checked vs pycvvdp 0.5.7 (max |Δ| 0.0001
  JOD, 8-pair probe).
- **Statistics:** pooled + per-reference SROCC and KonFiG triplet accuracy,
  reference-clustered bootstrap B=2000 seed 20260923, all from the `panel`
  owner (`zensim-validate`). Deliverables:
  `benchmarks/rev4_e2b_cvvdp_display_2026-09-23.{md,json}`.
- **SafeSyn (Part 2):** codec-variant pixels only, no human labels exist;
  descriptive rank agreement vs stored features/labels per brief — pending
  the fleet gate (FLEET_READY.md; not yet run).
- **Status at landing (2026-09-26):** Part 2 ran as fleet run
  `cvvdp-safesyn-20260923` (3,218 jobs, 196,086 pairs, metric labels only; no
  human label was read). Its record is zenmetrics
  `benchmarks/cvvdp_safesyn_2026-09-23.md`; sidecar sha256 `775bdb8f…`.

## Exposure receipt — 2026-09-23: rev4 featpot baseline (preread reservation)

**POTENTIAL — ceiling, not a model score.** Preregistration: `benchmarks/rev4_featpot_prereg_2026-09-23.md`, committed as `d2169f5b` before any label value was decoded by this lane. Purpose: D1 fitted in-sample potential diagnostics and D2 LODO rotation, baseline Rev3 944 arm only. Fitted models are diagnostic only, under `/var/tmp/rev4-featpot/`; they never qualify a recipe or a held-out score.

| Population | Whole source / planned admitted rows | Status before read | Status after read |
|---|---|---|---|
| CID22-A(25) | `ext_cid22val.parquet` 4,292 mixed A/B rows; admit only A rows by pinned ref allowlist | planned | pending; update to potential-exposed / LODO-exposed with exact A row count only after the corresponding read |
| AIC-3 CTC | `ext_aic3.parquet` 600 rows | planned | pending; update after read |
| KADID SELECT | `ext_kadid.parquet` 3,125 rows | planned | pending; update after read |
| KonFiG originsplit_val | `ext_konfig.parquet` 436 rows, origin split to be verified | planned | pending; update after read |

TRAIN-role KADID, TID, KonFiG originsplit_train and KonJND BPG are subject to their original roles; every TRAIN label read also gets a worklog receipt. CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE and secret holdouts stay unread. A missing/incompatible f64 cache does not authorize a substitute population or a different feature era.

## Exposure receipt — 2026-09-23: rev4 featpot promoted-bank baseline (preread)

**POTENTIAL — ceiling, not a model score.** This receipt follows addendum commit `044f00dc043f` (22:59:18 UTC). Source is only `/var/tmp/rev4-featbank/bank/<set>/labels__*.parquet`, joined by `pair_key` to the same set's keys and Rev3 feature sidecar. The coordinator's D1/D2 ruling authorizes the named diagnostic fits and fold reads. All statuses below are **pending until the first actual label-value read**; they are updated with command/output hashes after admission. Fitted in-sample results are diagnostic only.

| Population | Stimuli / unique keys | Planned use | Status before read |
|---|---:|---|---|
| KADID TRAIN | 5,000 / 4,880 | TRAIN fit, D2 fold | pending TRAIN read |
| TID2013 | 3,000 / 3,000 | TRAIN fit, D2 fold | pending TRAIN read |
| KonFiG originsplit_train | 327 / 327 | TRAIN fit, D2 fold | pending TRAIN read |
| KonJND BPG TRAIN | 8,060 / 8,060 | oracle-target TRAIN fit, D2 fold | pending TRAIN read |
| KonJND BPG VAL | 2,020 / 2,020 | oracle-target D2 fold eval only | pending LODO exposure |
| CID22-A(25) | 2,192 / 2,192 | D1 in-sample diagnostic, D2 fold | pending potential/LODO exposure |
| AIC-3 CTC | 600 / 600 | D1 in-sample diagnostic, D2 fold | pending potential/LODO exposure |
| KADID SELECT | 3,125 / 3,050 | D1 in-sample diagnostic, D2 fold | pending potential/LODO exposure |
| KonFiG originsplit_val | 436 / 436 | D1 in-sample diagnostic, D2 fold eval | pending potential/LODO exposure |

The only admitted held-out human label files are the `labels__human.parquet` files for CID22-A, AIC-3, KADID SELECT and KonFiG VAL. **Never read held-out `human_score` from the bank `pairs/` or `raw/` copies or their `_sealed/` destinations.** CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE, MCL-JCI pending D3, and every secret holdout stay unread.

### Admission update — 2026-09-23 23:07:33–23:07:35 UTC

The nine populations above were admitted by `scripts/rev4_featpot/admit_bank.py` after preregistration commits `044f00dc043f` and `2195532f`. The adapter read only the listed bank `labels__*.parquet` values, joined on `pair_key`, and wrote `/var/tmp/rev4-featpot/admitted/POT_<set>_rev3_944.parquet`. Its initial nine-line receipt had SHA-256 `5344cbefb4e42188f3c3b66bc4cc5bdb61ee6e60fdba5fd54dd777c0d0d2a23f`; the current receipt and target-free tables are recorded below. Exact admitted rows and keys are the counts in the table above; the receipt also records each output SHA-256 and reference count. KADID TRAIN, TID2013, KonFiG TRAIN and KonJND BPG TRAIN are **TRAIN-read**. CID22-A(25), AIC-3 CTC, KADID SELECT and KonFiG originsplit_val are now **potential-exposed** by the admitted label read; their D2 LODO folds have not yet run. KonJND BPG VAL is **oracle-label-read for D2**, with its fold pending. None is a Rev4 confirmation result.

### Label-source correction — 2026-09-23 23:31 UTC

Before any model fit, the nine admitted tables were rewritten to contain **no target column**. The adapter still validates source-row multiplicity and finite targets from the allowed bank label files, but stores only `pair_key`, `source_row_id`, reference/codec metadata and f0–f943. The new nine-line receipt `/var/tmp/rev4-featpot/admit_bank.jsonl` has SHA-256 `aaf927dc2a8c8dd6066ce740bef53f137a61cee8e26bb3210d440d38465cdbcb`; its lines identify every rewritten output hash. All fit/stat drivers now call `scripts/rev4_featpot/data.py`, which rechecks the pinned manifest/file hashes and reads values directly from the nine named bank `labels__*.parquet` files, joining on both `pair_key` and `source_row_id`. It refuses an admitted table containing a target column. The queued AIC-3 fit was interrupted before acquiring the shared heavy lock; **no model fit ran against the earlier copied-target tables**.

### LODO rotation reservation — 2026-09-24 00:06 UTC, before any LODO fit

The baseline R0 LODO runner will fit seven quarantined diagnostic folds over KADID TRAIN (5,000 rows), TID2013 (3,000), KonFiG originsplit_train (327), KonJND BPG TRAIN (8,060, **SSIMULACRA2 oracle /100**), CID22-A(25) (2,192, **human MCOS/100**), AIC-3 CTC (600), and KADID SELECT (3,125). It will evaluate the KonFiG held-out fold on the reference-disjoint originsplit_val (436) and the BPG held-out fold on its reference-disjoint oracle VAL (2,020), with equal total weight per *training* source. All label values are read only from the nine pinned bank `labels__*.parquet` files through `scripts/rev4_featpot/data.py`; the admitted feature tables carry no target. The listed seven training populations become **LODO-exposed** only after this command actually fits; KonFiG VAL and BPG VAL receive LODO evaluation exposure only after their folds actually score. CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE, MCL-JCI pending D3, and secret holdouts remain unread. No fold output is a model score or a Rev4 confirmation result.

### LODO rotation exposure — 2026-09-24 00:12:21 UTC

The reserved baseline R0 BVLS D2 rotation completed through `scripts/rev4_featpot/lodo_bvls.py --arm r0` under the shared heavy runner (start `00:08:13Z`, end `00:12:21Z`, exit 0). Result `/var/tmp/rev4-featpot/lodo/LODO_r0_bvls/result.json` SHA-256 `7d7d9427737d79e3d1cf522a04a4b86c15a4de21f4d633fdfc9cc7619b070af1`; command log SHA-256 `d147d25911d30ec84d8538db6b087355342c2b691da9dc1ba3dd7b0c334aab59`. KADID TRAIN, TID2013, KonFiG originsplit_train, KonJND BPG TRAIN, CID22-A(25), AIC-3 CTC and KADID SELECT are now **LODO-exposed** diagnostic training populations. KonFiG originsplit_val (436) and KonJND BPG VAL (2,020, SSIMULACRA2 oracle /100) are **LODO-evaluation-exposed**. All source targets were normalized separately on their own training rows before equal-source weighting; CID22-A remains human MCOS/100 and BPG remains an oracle in /100 units. The fold models live only under `/var/tmp/rev4-featpot/lodo/LODO_*`. No secret, confirmation or D3-pending population was read. These results are potential ceilings, never model scores.

The R0 lasso seven-fold rotation subsequently re-read the same nine already exposed bank label files through `scripts/rev4_featpot/data.py`; it added no population. The shared-heavy command `python scripts/rev4_featpot/lodo_lasso.py --arm r0` exited 0 by `2026-09-24T00:32:02Z`. Its result `/var/tmp/rev4-featpot/lodo/LODO_r0_linear/result.json` has SHA-256 `0c01528806375c912ea25e9ed5eb4cd42d32b372d5980e5d93ae2b9c3807f109`, and its command log SHA-256 is `6d2981771c80209c17f5875bf83d8600fc650f16d005a3cb06114d1f888f89fd`. The reference-clustered CI receipt is in `benchmarks/rev4_featpot_lodo_2026-09-23.md`.

## Exposure receipt — 2026-09-25: rev4 featpot restore-cuts arms (preread reservation)

**POTENTIAL — ceiling, not a model score.** Follows amendment commit `72b35b43439b` (2026-09-25T15:57:29Z). Populations, roles and forbidden sets are exactly those of the promoted-bank baseline receipt above (KADID TRAIN, TID2013, KonFiG train/val, KonJND BPG train/val, CID22-A(25), AIC-3 CTC, KADID SELECT); nothing new is admitted, and CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE and every secret holdout stay unread. Arms A1, A1m, A1w, B2, B2m (and, once their Part B sidecars exist, B1, B1s, C8n, ALL) with size-matched permuted controls are fitted on these populations as **potential / LODO diagnostics: fitted in-sample, diagnostic only.** Restore-cuts sidecars carry features only; this lane opened no label for these arms before the pin commit. Status before read: pending potential/LODO re-exposure of the same nine populations (already potential-exposed by the baseline). MLP fits of these arms run in the fleet AVX2 era; deterministic BVLS/lasso run locally.

### Post-read update — 2026-09-26 (rev4 featpot baseline preread reservation)

The first featpot receipt above ("rev4 featpot baseline (preread reservation)") lists four `rev3-public-human-eval/features-rev3/ext_*.parquet` f64-cache files, including the mixed A/B `ext_cid22val.parquet`, as "pending; update after read". **Superseded by the promoted-bank receipt** that follows it. Those files were only hashed and footer-counted at preregistration, and `ext_konfig.parquet` was probed for `ref_basename`/`f0` only (`benchmarks/rev4_featpot_2026-09-23/probe_cache.py`, `"labels_read": false`). No label value was decoded from any of them.

### Post-read update — 2026-09-26 (rev4 featpot restore-cuts arms)

The restore-cuts receipt above ends at "Status before read: pending". Reads since: from 2026-09-25T16:05Z the deterministic D1/D2 fits of the 18 arms (c1–c4, all, csfw, c7, p1, p3, b1, b1s, c8n, rall, a1, a1w, a1m, b2, b2m) read labels of exactly the same nine populations; 648 results, the last before 2026-09-26T00:56Z; the arm stability runs followed. So did the two disclosed VOID sets (8 `a1` BVLS D1 cells, 2026-09-25T16:05–16:26Z; 18 registry-mask results, 16:29–16:40Z), whose outputs are excluded from every table; valid results start at 16:43Z. No new population was read and no status changed: the nine populations stay potential/LODO-exposed as recorded.

## Exposure ledger — 2026-09-26: accidental display of holdout human scores (audit lane `cvvdpaudit`)

A read-only data audit lane (the wgpu CVVDP ≥ 4,194,240-pixel audit, zenmetrics fix `9a8326fd`) ran `head` on
`/var/tmp/rev4-e1*/tables/*.tsv` while locating CVVDP columns. That printed 2 rows each of the aic3, aic4crop,
cid22a, sdr25 and csiq tables to its terminal, including the human-score column `t`.
- The values were not recorded, compared or used. No statistic was computed from them, and no model, feature,
  hyperparameter, checkpoint or selection decision was informed by them.
- All later reads by that lane selected only the CVVDP, id and dimension columns.
- AIC-3 and CID22-A are already potential-exposed under D1 (entries above). For AIC-4 (crop), SDR25 and CSIQ this
  is a 2-row incidental display, recorded here so it is never silent. Their holdout status is unchanged.
- Disclosure: `~/tmp/zensim-paper/rev4/CVVDP_WGPU_AUDIT_DONE.md`, MISSING item 6.
