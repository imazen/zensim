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

October 4 R5INTEG derived TRAIN entry: the explicit lane brief reuses the
September8 canonical packet's eight fit and four calibration origins, including
1214/6064/9066/8462 as historical inner calibration. This does not migrate the
later product packet or its reservations. No canonical validation pair is
decoded or scored; no protected label or sealed data is read. Existing admission
provenance checks rehash protected reference bytes, as disclosed in the worklog.
All9,036 TRAIN attempts are re-extracted at Rev4/Rev5;526 verified identity
attempts (eight unique pairs) are separately retained and excluded from head
fitting because the public identity shortcut bypasses model inference. The
remaining8,510 attempts deduplicate to8,205 nonidentity pairs. Recipe, feature
regimes, source hashes and activation failures are recorded in
[the lane worklog](../benchmarks/r5integ_WORKLOG.md). Historical catalog positives
are not reviewed catastrophic labels; no threshold, seed or cost is selected.

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
| **KonFiG-IQA** (10 src × 7 dist × 12-30 levels over 3 JND; 1.7M triplets) | **T2 (INGESTED 2026-07-02; 944 LEG BUILT 2026-08-05)** | 944 leg: `ext944-canonical-2026-08-01/konfig_944.parquet` (1,090 rows, 85+24 per source, + native `q_jnd`; multiset-identical to the 372-era `konfig_train_2026-07-02.parquet`; builder `scripts/canonical_corpus/build_konfig_944.py`; campaign **Appendix L**, pre-reg `e93eba04`). human_score = 1−q_jnd/3.2 — **QUALITY-oriented, gated**: `check_target_orientation.py` declares `quality`, verified signed SROCC **+0.5645** vs the 75,519 raw EXP_III DCR votes (n=850 PartA; PartB shares the formula). Origin-split views `konfig_originsplit_{train,val,test}_944.parquet` (327/436/327; `split_of` on numeric src id) exist for any future within-KonFiG instrument; the registered probe leg is the FULL table (L.6 design decision — training on it forecloses those views as eval for those models). **F-condition (flicker-boosted) reconstructed JND scale for TRAIN+VAL sources (2026-10-07):** `/mnt/v/dataset/konfig-iqa/derived/konfig_fscale_trainval_2026-10.parquet` (+`.manifest.json`), 637 rows = 7 src × 7 dist × 13 levels, builder `scripts/canonical_corpus/konfig_fscale.py` (Python port of the authors' EXP_I MATLAB chain; Octave-fmincon oracle max abs diff 0.0050, bitwise deterministic, ssim2 τ = −0.7724 on the subset vs published 0.7668 Part A). Held-out test sources' raw responses were never parsed. **ssim2 tuned on it** → never a ssim2-comparison corpus | Authors: fine-grained JND-unit scales via boosted triplet comparisons (Men 2021) | 10 sources. **dHash spot-audit RUN 2026-08-05 (Appendix L G-L1/G-L2, commit `7ed6ac4b`): CLEAN PASS** — 0 exact hits + zero d≤10 flags vs KonJND-1008 / CID22-49 / CSIQ-30 / LIVE-29 / AIC3-10; global min d=17 (dHash is crop-blind — residual stated in L.11.8) |
| **PIPAL** (local, unused) | — | Not in pipeline (SR/GAN domain) | Official NTIRE train/val/test splits | — |
| **NITS-IQA** (9 refs × 9 distortions × 5 levels = 405 pairs, 512×512, MOS 0–100, 162 observers) | **T0, EVAL-ONLY (INGESTED 2026-10-03)** | Open external held-out set for the featpot instrument (§3e): never a training input. Pairs `build_fr_corpus_pairs.py nits` → `/mnt/v/datasets/nits-iqa_extracted/nits_iqa_pairs.tsv`; `human_score` = MOS/100, QUALITY-oriented (`check_target_orientation` declares `quality`, ground truth = raw `Score.xlsx`). Distortions D1–D9: Gaussian blur, chromatic Gaussian noise, chromatic uniform noise, **contrast change**, **pixelate mosaic**, motion blur, JPEG, JPEG2000, JPEG-XT. The D7 files are JPEG bitstreams named `.bmp` (as released) | Authors' release (Ruikar & Chaudhury, Sensors 2023, doi:10.3390/s23042279, CC BY): no split; whole-set correlation | 9 references never in any training corpus; dHash audit 2026-10-03 (§3e): 1 flag at d≤10 (I8 vs a KADIS cat photo, unrelated); no TID2013 crop |
| **MCIQA-2K** (2,000 colorized images = 5 colorization models × 400 COCO test2017 images; z-scored MOS for colour smearing, semantic colour misalignment, global naturalness) | **T0, EVAL-ONLY, EXPLORATORY (INGESTED 2026-10-03)** | No-reference by design (humans rated plausibility, not fidelity), so the full-reference read pairs each colorized image with its COCO test2017 original (`coco_test2017_refs/`, same size). Pairs `build_fr_corpus_pairs.py mciqa`; `human_score` = global naturalness min-max scaled (QUALITY-oriented, declared `quality`); CS/SCM/GN z kept as columns. Reported as a colour-sensitivity diagnostic only, never as accuracy-vs-humans for a fidelity metric | Authors (arXiv:2609.14495, CC BY 4.0): official train (1,600) / test (400) split for NR models | COCO test2017 images are not in any training corpus; dHash audit 2026-10-03 (§3e): 31 flags at d≤10, all unrelated (degenerate horizon-like hashes); thumbnail NCC max < 0.9 |
| **KonIQ-10k** (10,073 in-the-wild photos, 1024×768 + 512×384, MOS) | **Source-image pool (INGESTED 2026-10-03)** | No reference images, so its MOS cannot label a full-reference pair and is never used as a target. Role: content-diverse, license-clean pristine sources for synthetic coverage ladders (design log E15 follow-ups); MOS used only to pick high-quality sources. Any coverage leg built on it is split by source image (`sha256(filename) % 10 < 8` train) | Authors (Hosu et al., TIP 2020; CC-licensed images allowing edits): NR benchmark, 7,058 / 1,000 / 2,015 split | Flickr sources. Audit 2026-10-03 vs 3,101 eval references (T0 estate, NITS, LIVE, MCIQA, KADID, TID, KonFiG, CID22-train, KonJND): dHash 367 flags at d≤10 (52 at d≤6), every closest pair unrelated (dark / flat degenerate hashes); thumbnail NCC max 0.97, top 20 all unrelated smooth-gradient scenes — no duplicate found. Pool table `koniq10k_pool.parquet` (`scripts/canonical_corpus/build_koniq_pool.py`; 10,073 scored images, 8,019 train / 2,054 holdout; the zip's 300 unscored JPEGs excluded) |
| **UPIQ / HDR** | **UPIQ-380 (HDR subset): T2 TRAIN since 2026-10-07 (owner decision, ledger below); rest of UPIQ: T0** | UPIQ-380 human JOD scores may train HDR legs (its holdout value was spent: ~21 looks, DATASET_HISTORY); any other UPIQ portion stays held-out eval | Mikhailiuk 2021: consolidated dataset, JOD-rescaled. HDR-VDP-3 was calibrated on UPIQ, so UPIQ never independently tests an HDR-VDP-3-trained model | — |
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

### §3e. External held-out evaluation sets for the featpot instrument (registered 2026-10-03)

NITS-IQA, LIVE release 2 and MCIQA-2K are added to the Rev4 instrument as **open external reads**: every trained cell's bake is
scored on them (a LODO bake never saw them), arms are seed-paired against their control, and the reads are reported beside the
five LODO sources. They are **not** part of the sealed confirmatory set (R6 stays as registered; CSIQ/AIC-4/the terminal splits
keep their sealed labels). Being open, they inform development from their first read on: a claim of generalization to them
is exploratory unless a later registered read is declared before looking. Owner: `scripts/rev4_featpot/external_sets.py`
(features from the r4 bank extractor, decoded-pixel audit, orientation gate at build). LIVE's §8.2 target-identity defect concerns
the older 372/ext roots; the Rev4 external table is built fresh from `live_r2_pairs.tsv` (`1 − dmos_new/100`).

**Content-overlap audit (2026-10-03, `benchmarks/external_sets_2026-10-03.md` §2).** dHash-64 (`check_holdout_overlap`, t=16)
of the 438 new references vs 17,251 training sources (the 8,370 of the AIC2026 audit + KADID's 81 + the 8,800 KADIS references
of E14/E15): 33 flags at d≤10, all adjudicated unrelated by inspection; vs the existing T0 estate min d=13. A supplementary
whole-image thumbnail correlation and a multi-scale template match then found what dHash cannot (crop + rescale): **19 of LIVE's
29 references are Kodak scenes that contain a TID2013 reference** (NCC 0.90–0.995, runner-up ≤ 0.82; the other 10 ≤ 0.64), and
17 of those also contain a SafeSyn `<kodak>_512sq.png` source. TID2013 trains 4 of 5 LODO folds and SafeSyn every fold, so
**LIVE is not content-disjoint from training** (the NNCD precedent). Every LIVE read is therefore reported overall and on the
10 content-disjoint references (building2, carnivaldolls, cemetry, churchandcapitol, coinsinfountain, dancers, flowersonih35,
manfishing, monarch, studentsculpture; 267 pairs) — the subset is fixed by the audit, never by a score. NITS references contain
no TID2013 reference (max 0.71, runner-up equal). Residual for every set: a crop of a reference inside a larger training image
other than TID/SafeSyn is not excluded.

**Contrast direction: NITS and TID2013 observers disagree (measured 2026-10-03).** NITS D4 levels 1–2 lower contrast (luma std
ratio 0.89 / 0.95) and levels 3–5 raise it (1.06 / 1.11 / 1.23), with equal pixel change at levels 1 and 4 (mean |Δ| 7.9) and at
2 and 3 (4.0). NITS MOS falls monotonically with level (0.76 → 0.13), so its observers rank a contrast **increase below a decrease**
of the same size (0.31 vs 0.76). TID2013 type 17 shows the opposite: increases MOS 6.39 (n 50) vs decreases 4.52 (n 75) at
similar magnitude. TID trains 4 of 5 LODO folds and the models follow it (control predictions are an inverted U over NITS levels,
peaking at level 3), so NITS contrast change (SROCC ≈ 0.18) measures a direction preference that the training sources contradict,
not a coverage gap. Treat contrast direction like the KADIS signed types (§2b): never a monotone "more change = worse" target.

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
| T0 estate (aic3/aic4/sdr25/csiq/live/UPIQ except UPIQ-380…) | NEVER | — | eval-only | checker hard-errors on any T0 name in a train group; UPIQ-380 re-designated T2 on 2026-10-07 |

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

## Exposure receipt — 2026-09-30: rev4 featpot Instrument v2 (preread reservation)

**POTENTIAL — ceiling, not a model score.** Governing record: `benchmarks/rev4_featpot_v2_amendment_2026-09-30.md`
(this commit). Populations read by v2 cells: `kadid_train`, `kadid_select`, `tid2013`, `konfig_train`, `konfig_val`,
`cid22_a25`, `aic3` — the same bank label files the promoted-bank baseline receipt already lists; no new
population. KonJND-BPG is not read by v2. Forbidden and unchanged: CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID
TERMINAL, LIVE, MCL-JCI, secret holdouts. Status before read: pending; update after read.

Takeover reads the same day (already reserved populations, registered analyses): P0 / P2 / D2 MLP pooled aggregates
and compares were computed on 2026-09-30 21:59–22:05Z from the fleet-era cells (`mlp_aggregate/`,
`p2/mlp_aggregate/`, `p2/mlp_compare/`, `p2/d2_mlp_compare/` under `/var/tmp/rev4-featpot/`). Record:
`~/tmp/zensim-paper/rev4/FEATPOT_AUDIT_2026-09-30.md`.

Revision R1 (same day, before any v2 cell result): v2 also reads the TRAIN-role teacher labels (SSIMULACRA2 oracle)
of `safesyn` and `cid22_train` from the promoted bank, restricted to R915's recorded fit/dev reference split, as
training-only legs; never evaluated. Pin: `benchmarks/rev4_featpot_v2_teacher_pin_2026-09-30.json`. Status before
read: pending; update after read.

## Exposure ledger — 2026-09-30: fleet transport of Instrument v2 inputs (ruling D1)

The v2 wide tables (`/var/tmp/rev4-featpot/v2/wide/{real,p1,p2,p3}`: the five admitted human sources' features +
targets and the two TRAIN-role teacher legs, per `benchmarks/rev4_featpot_v2_amendment_2026-09-30.md` revision R1)
are packed into one content-addressed archive (sha256 `97a0c517f88efb34e217403388f36f5804e30c89a93745901ec5899e11cc3551`,
8,900,170,387 bytes, `zenmetrics scripts/jobsys/pack_fit_data_v2.py`) and uploaded to the LAN object store under
`s3://zentrain/jobs/<v2 jobset>/inputs/` for zenfleet fit cells (program sha
`e791fcc310981f2f0661be4a819e7d8d69e1ee800b6d0e44cd67922a8de7121b`, image `ghcr.io/imazen/zenfleet-worker:fit-v2-v1`).
No new population; the LAN store and workers are operator-controlled machines on the home network. No holdout table
is in the archive (CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE, MCL-JCI and secret holdouts excluded).

## Exposure ledger — 2026-10-01: fleet transport of Instrument v2 inputs, erratum R1.1 layout (ruling D1)

The jobset that used the 2026-09-30 archive (`97a0c517…`) is retired (erratum R1.1 in
`benchmarks/rev4_featpot_v2_amendment_2026-09-30.md`). Its replacement packs the two-family tables
(`/var/tmp/rev4-featpot/v2/wide/{main,aux}/{real,p1,p2,p3}`, same populations, same rows, same splits; only the column
layout changed) into one content-addressed archive (sha256
`71e05dd29f7205bc915b531dada88d916f8d23f520ad06fdc4e32a98d858eb13`, 14,725,362,668 bytes, 353 members,
`zenmetrics scripts/jobsys/pack_fit_data_v2.py`), uploaded to the LAN object store under
`s3://zentrain/jobs/<v2 jobset>/inputs/` for zenfleet fit cells (program sha
`4c2b064a68753289095af6ad4fe1a2866abd25dce19cf19ac7a68a8f9dad4a1f`, image `ghcr.io/imazen/zenfleet-worker:fit-v2-v2`).
No new population; the LAN store and workers are operator-controlled machines on the home network. No holdout table
is in the archive (CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE, MCL-JCI and secret holdouts excluded).

Addendum (2026-09-30 18:52 MT / 2026-10-01 00:52 UTC): the jobset using program `4c2b064a…` was stopped and retired (its image carried a start-up race
in the fit-cell executor's link creation). The same archive (`71e05dd2…`, unchanged) is copied server-side to the
replacement jobset, run with program `ff2e3ec63c4ecaf07ffd612bb9352ab35d42b89ada43ae354400b3aa96064f09` and image
`ghcr.io/imazen/zenfleet-worker:fit-v2-v3`. No new population and no new transport destination.

## Exposure ledger — 2026-09-30 22:18 MT: Rev4 confirmation holdouts designated (user decision)

The user chose to use human labels on the five Instrument v2 sources for Rev4 design and designated CID22-B, the AIC-4
sample, KonJND-JPEG (SELECT as the confirmatory surface; TERMINAL-100 as a touch-once sanity guard), CSIQ and MCL-JCI as the
confirmatory holdouts (`benchmarks/rev4_featpot_v2_amendment_2026-09-30.md` revision R2). Their labels stay sealed and
unread until the single confirmatory read of frozen candidates; no label of these sets was read in making this
designation. KADID TERMINAL, LIVE and secret holdouts are not part of it and stay untouched.

## Exposure ledger — 2026-09-25/26: featcanon tier-parity audit and its fix (pixels only)

The featcanon lane (`quarantine/devin/featcanon`, 2026-09-25) and its fix lane (featcanon-fix, 2026-09-26) read
**pixels only**. No label file was opened, and every quantity is a bit comparison between SIMD tiers or an error
magnitude against an f64 oracle. Nothing was fitted, calibrated, selected or tuned. Recorded because the audit
sample included T0 content and SELECT references, which the lane's own record called "TRAIN".

The lane's audit sample (`~/tmp/devin/featcanon/audit_pairs.tsv`, restated exactly, 11 pairs):

| label | reference / distorted | role |
|---|---|---|
| kadid512x384 | KADID `I01` / `I01_01_01` | SELECT (last digit 1). **Pixel-identical pair.** |
| kadid64crop | KADID `I01` / `I01_02_05`, crop x=64 y=224 64×64 (identified 2026-09-26 by exact pixel match; the generator was not committed) | SELECT |
| tid512x384 | TID2013 `I01` / `I01_01_1` | TRAIN (TID is train-only, §8.1) |
| konfig384x512 | KonFiG `SRC01_PartA` / `SRC01_colordiffusion_0` | SELECT (`originsplit_val`, digit 1). **Pixel-identical pair.** |
| konjnd640x480 | KonJND `SRC0505` / `SRC0505_BPG_051` | TRAIN (BPG half, 5 ∉ {8, 9}) |
| aic3-853x945, aic3-945x840, aic3-1192x832, aic3-2000x2496, aic3-2592x1946 | AIC-3 CTC originals `00002`, `00003`, `00001`, `00004`, `00010` with their `AVIF_*_1` decodes | **T0, eval-only** |
| mosaic4096 | a 2×2 mosaic of AIC-3 content (the lane's worklog: "disjoint ~2048² real AIC3 crops"; the generator and the crop boxes were not recorded) | **T0-derived** |

So the sample was 2 TRAIN, 3 SELECT and 6 T0 pairs, not "9 real TRAIN pairs". The review
(`REVIEW_FEATCANON.md` D8) listed TID `I01` as SELECT; under §8.1 TID is train-only. The two pixel-identical pairs
are the "0" in the lane's "0–297 slots/pair" and the source of its `append_texture_dissim` exact-0.0 outliers.

The fix lane's own pixel reads: KADID `I01` and its 125 distortions (SELECT), read once to identify the
`kadid64crop` source above; and the review's probe set (`/var/tmp/review-featcanon/probe/pairs.tsv`, KADID `I02`/`I24`
and 16 even references, TID `I14`/`I01`, KonJND `SRC0510`/`SRC0505`, KonFiG `SRC06`, all TRAIN) for the Rev1–3
bit-identity and Rev4 tier gates. No AIC-3 file was opened by the fix lane. AIC-3's holdout status is unchanged.

## Exposure ledger — 2026-10-01 02:29 MT: fleet transport of the R3 retune inputs

Amendment revision R3 rebuilt the aux family with distance-oriented oracle columns (same populations, rows, splits,
keys and every other column, byte-identical to the superseded aux tables). The retune sweep and the acceptance re-run
read only `main/real` and `aux/{real,p1,p2,p3}`, packed into one content-addressed archive (sha256
`6e8460713b5b8512f34517b917f5257e7a7dc4a585dde0adab8e64a2c1499b48`, 8,069,171,677 bytes, 221 members,
`zenmetrics scripts/jobsys/pack_fit_data_v2.py --select`), uploaded to the LAN object store under
`s3://zentrain/jobs/<v2 R3 jobset>/inputs/` for zenfleet fit cells (program sha
`ec5273653165092835378c0ed823427ccbd047bf427e3fcf983297fcc1b2b2f2`, image `ghcr.io/imazen/zenfleet-worker:fit-v2-v4`).
No new population; the LAN store and workers are operator-controlled machines on the home network. No holdout table
is in the archive (CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE, MCL-JCI and secret holdouts excluded).

Addendum (2026-10-01 02:45 MT): when the retune sweep completes and the R3 rule selects a weight, the same archive
(`6e846071…`, unchanged) is copied server-side to the acceptance re-run jobset (`fitv2acc-20261001`, same program and
image). No new population and no new transport destination.

## Exposure ledger — 2026-10-01 07:12 MT: fleet transport of the v2-canon LODO inputs (amendment R2.3)

The v2-canon root (`/var/tmp/rev4-featpot/v2c`, width 1853 = the Rev4 bank's f0–f1824 plus the 28 SIGNEDFEAT sidecar
columns f1825–f1852; frozen record `wide/frozen.json`, sha256
`f432995f31a3de32a17f7defbaccecd3c22114130db107a050bbdd8fdc404040`; all 129 verify gates pass) is packed for the R2.3
calibration + screen cells with `scripts/rev4_featpot/v2c_pack.py --kind lodo`: one content-addressed archive (sha256
`6ec0357bccc2b243150e333e1b14b11eb032b21505609cc4753a3d1f37639e83`, 10,103,175,330 bytes, 418 members:
`wide/{main,aux}/{real,p1,p2,p3}` with the five admitted human sources' tables + keys, the per-held-out
`human_without_<source>_{fit,dev}` and `human_all_{fit,dev}` legs, the two TRAIN-role teacher legs `safesyn_{fit,dev}`
and `cid22_{fit,dev}`, receipts, `keep_lists.json` and `extra_arms.json`). Every table was hash-checked against its
receipt before it was copied. Uploaded to the LAN object store under `s3://zentrain/jobs/fitv2canon-20261001/inputs/`
for zenfleet fit cells (program sha `9607eada738ede364a46f4b0363b4f0fc90a99afb02af1eaa64f9af659741022`, image
`ghcr.io/imazen/zenfleet-worker:fit-v2-v8`). No new population; the LAN store and workers are operator-controlled
machines on the home network. No holdout table is in the archive: the member list was checked for the confirmatory
sets and none is present (CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE, MCL-JCI and secret holdouts
excluded). The confirmatory archive (`--kind confirm`) is not built yet and gets its own entry when it is.

## Exposure ledger — 2026-10-01 11:25 MT: fleet transport of the v2-canon confirmatory inputs (amendment R2)

For the R2 full-data confirmatory fits (`v2_confirm_fit.py`), `scripts/rev4_featpot/v2c_pack.py --kind confirm` packs from the
frozen v2-canon root (`f432995f…`) one content-addressed archive (sha256
`5c5780340de081f939987ee78b63f0f32c811e0db9ae0d439155e9318de9d1c7`, 7,752,615,791 bytes, 283 members): the two TRAIN-role
teacher legs and `human_all` fit/dev for both families and all four variants, receipts and keep lists, and the
**features-only** confirmatory tables of CID22-B, AIC-4, KonJND JPEG SELECT, KonJND JPEG TERMINAL, CSIQ and MCL-JCI
(`wide/confirm/{main,aux}[/p1-p3]`). The confirmatory tables carry no label: `human_score` is written as the constant 0
by the builder (`v2c_wide.py` confirm: `np.zeros(...)`), their keys hold only `pair_key, row_id, ref_basename,
member_set`, and the aux oracle columns are zero there (an oracle needs a label). Verified from the code and the
parquet schemas; no value of these columns was read. Uploaded to the LAN object store under
`s3://zentrain/jobs/fitv2confirm-20261001/inputs/`. No new population; the LAN store and workers are operator-controlled
machines on the home network. The labels stay sealed until the single confirmatory read (label pins: CID22-B(23),
AIC-4, KonJND JPEG SELECT/TERMINAL, CSIQ, MCL-JCI; `~/tmp/zensim-paper/rev4/LABELPIN_specs.json`).

Addendum (2026-10-01 11:32 MT): the archive above lacked `wide/frozen.json`, which `v2_confirm_fit` requires
(`v2_common.load_frozen`); found before any confirmatory cell ran, so no cell ever used it. `v2c_pack.py` now ships the
freeze record. The replacement archive (sha256 `cf9b83179376c78471905c8e97e3280b3d45ea008123f9febdbd660f9046f99d`,
7,752,619,775 bytes, 284 members = the same 283 plus `wide/frozen.json`) goes to the same store prefix; same populations,
no new transport destination, labels still sealed.

## Finding — 2026-10-04: KonFiG-IQA's ten references are crops of MCL-JCI sources (open, owner decision)

KonFiG-IQA (T2, ingested 2026-07-02; a training source of the Rev4 feature-potential fits, including all four R7 entries) takes its
ten references from MCL-JCI, a confirmation-only holdout: KonFiG `SRC01/03/06/07/09/17/28/31/45/50` are zoomed crops (384×512) of
MCL-JCI `ImageJND_SRC##` with the same numbers. Source: the KonFiG paper ("ten source images from the MCL-JCI dataset"); three of ten
pairs (SRC01, SRC03, SRC45) checked by eye in the chromatic-studies survey (`~/tmp/zensim-paper/rev4/CHROMA_STUDIES_survey.md`
item 5). The KonFiG row's dHash audit did not include MCL-JCI and is crop-blind, so its "CLEAN PASS" does not cover this overlap.
No label was read for this finding. Interim handling: amendment R7a (`benchmarks/rev4_featpot_v2_amendment_2026-09-30.md`) requires
every R7 verdict to hold also with MCL-JCI restricted to its 40 other sources. Open for the owner: whether KonFiG stays in training,
and whether MCL-JCI (or its 40-source subset) stays a holdout for KonFiG-trained models. A crop-aware duplicate check (feature or
template matching) of every training reference against every holdout source is not yet run.

## Exposure ledger — 2026-10-04 03:31 MDT: set-compare confirmatory read (amendment R7)

Status: **read 2026-10-04 03:31–03:32 MDT, once** (pin `v2c_setcompare_pin_r7a_2026-10-04.json`, R7 + R7a; full output `/var/tmp/rev4-featpot/v2c/compare/r7a_setcompare_read.json` sha256 `3f0b7f9f699ad1d3…`, tower `output/zensim/featpot-r7a-read-2026-10-04/`; summary `benchmarks/rev4_featpot_effaudit/r7a_setcompare_read_2026-10-04.summary.json`). Outcome: Q1 A vs C **not confirmed** (4-set mean Δ +0.0001, p 0.46); Q2 A vs D **not confirmed** (+0.0022, p 0.18); Q3 B vs A **as good (non-inferior)** (+0.0010, 5th percentile −0.0006; KonJND-JPEG SELECT Δ −0.0171, CI upper −0.004, 0/10 seeds above — no veto, 0.001 inside the margin). Every verdict is the same on the R7a clean primary (MCL-JCI without the ten KonFiG sources). These sets' labels are now exposed for the four entries; no design change may follow, and a later change needs a new holdout. `v2_confirm_read.py --set-compare` read the sealed labels of: cid22_b (`/mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv`, sha256 `3ce0f7438ea0…`), aic4 (`/mnt/v/output/zensim/v2-backfill-2026-07-20/aic4_pairs.tsv`, sha256 `955a9601e94c…`), csiq (`/mnt/v/dataset/csiq/csiq_pairs.tsv`, sha256 `78b1dac5f74e…`), mcljci (`/var/tmp/datasets/mcl-jci/mcljci_labels.csv`, sha256 `36f17dd8bac2…`), konjnd_jpeg_select (`/mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv`, sha256 `70148a39c90d…`), konjnd_jpeg_terminal (`/mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv`, sha256 `70148a39c90d…`).
Frozen entries: A = `set:v2+basic@h32:H128:cv16:cf98`; B = `sel:59f0bbc2f290@h32:H128:cv16:cf98`; C = `set:v2+basic@h32:H128`; D = `r0@h32:H128` (head N); pin `../../benchmarks/rev4_featpot_effaudit/v2c_setcompare_pin_r7a_2026-10-04.json` sha256 `701964b7014f8d6b33854f2db253574a83db9f3046c4630746c6d0cd74c422f8`; frozen root `f432995f31a3…`.
Primary: superiority A vs C, A vs D (Holm at 0.05); non-inferiority B vs A. No design change may follow from this read; a later change needs a new holdout.

## Exposure ledger — 2026-10-04: R5CONFIRM pixels-only extraction and fleet transport

Authorized artifact production by `R5CONFIRM_brief.md`; no held-out label or `_sealed` directory was opened.
The existing `rev5_bank.py` owner extracted the six R7 confirmatory populations from old-bank keys/pixels
with binary `c649e810…` (build `1a9d5a8a`, descendant of frozen arithmetic `60174678`), requested basic+peaks+v2,
feature identity `basic+peaks+v2@w1825/rev5_localwin#36c3f3af`, absent slots NaN. Confirmatory tables
contain constant-zero `human_score`; keys contain no label/target. Rows: CID22-B 2100, AIC-4 300,
KonJND JPEG SELECT/TERMINAL 404/100, CSIQ 865, MCL-JCI 5000. The original scientific roles are unchanged.

`v2c_pack.py --kind confirm --select main/real --name v2c5` packed admitted TRAIN teacher legs and
full exploratory `human_all` fit/dev plus these features-only tables, receipts/keys/manifests/keep list/freeze.
SHA-256 `1707879d64581d60450e2d6978988f2d77007fc0814e1b513ac6a1076cfe35c1`, 389904514 bytes,
38 members plus inventory, transported to the existing operator-controlled fleet store at
`s3://zentrain/jobs/fitv2r5confirm-20261004/inputs/`. Program v25b carries the Rev5 E15 coverage pool.
Six R7 A/B recipe cells, head N, seeds 0–2; no confirmation read and no Rev5-vs-Rev4 decision.
Provenance and final serving evidence are owned by `benchmarks/r5confirm_WORKLOG.md`; E24 owns selection.

## Exposure ledger — 2026-10-04: HDRCORR frozen Rev4 transfer and corruption read

Explicit evaluation-lane brief `~/tmp/zensim-paper/rev4/HDRCORR_brief.md`; no fit, calibration,
feature/checkpoint selection or threshold tuning. Frozen by_v2fy and v2+basic full-data seeds
0/1/2 at cv16:cf98 plus shipped B/BHdr; hashes in `/var/tmp/hdrcorr/FROZEN.json`.

- HDR TRAIN: September15 corrected 7,425 native-PQ pairs, 495 variants / 33 source families;
  fresh native CVVDP/PU-SSIM2 truth retained by that packet. Candidate scores freshly computed
  through `BakeScorer::compute_hdr` at Rev4; shipped controls at their own Rev1 in a separate process.
- HDR VAL: registered hdr_v3mix 3,900 rows / 300 reference variants / 20 origins, explicitly
  authorized by this brief, preserving its historical VAL role (including odd terminal digits).
  Original stored features used only to recover row identity against the retained producer TSV;
  every alias checked for bitstream-byte identity. New scores use native PQ16/codestream cICP.
  Carried historical cvvdp-mix targets are not fresh common-primary judge outputs.
- Corruption: canonical September8 TRAIN only, 8,580 catalog attempts plus 456 retained
  honest native JXL/AVIF outputs on the 12 TRAIN origins. Deduplicate source/pixel identity;
  inert controls, honest low-quality anchors and historical catalog positives remain distinct.
  No canonical corruption validation rows used.
- No UPIQ label/distorted image, secret holdout, or other T0 label read. UPIQ request and
  missing Rev4-compatible integrity companion recorded in `HDRCORR_decisions.md`.

Evidence: [HDRCORR worklog](../benchmarks/hdrcorr_WORKLOG.md). These are descriptive
transfer/catalog diagnostics; no model is qualified or promoted.

## Exposure ledger — 2026-10-04: HDRTEACH native teacher labels

Explicit follow-on `~/tmp/zensim-paper/rev4/HDRTEACH_brief.md` authorizes labeling the
same corrected TRAIN (7,425 rows / 495 reference variants / 33 origins) and registered
hdr_v3mix VAL (3,900 / 300 / 20) admitted by HDRCORR. Their original roles and row IDs
remain fixed; no digit-based re-splitting, feature joins, deduplication or model fitting.

HDR-VDP-3.0.7 q_jod is produced by the zenmetrics owner at one preregistered condition:
ppd60, absolute BT.709 RGB nits via declared-PQ common-primary ingress,
led-lcd-srgb emission, surround none, age24, reference quality options. All native
file hashes are verified against HDRCORR. TRAIN's second teacher is its fresh native
CVVDP JOD; VAL's is carried historic CVVDP JOD, with historic cvvdp-mix auxiliary.
The per-row `agree` flag uses exact-reference averaged tied midranks, absolute rank
difference <=1 position, minimum two finite rows. Fixed before new scores; it is not
human validation, fit selection or an adaptive threshold. VAL flags remain VAL-only.

Only the HDR-VDP-3 publication text in zenpapers was read concerning UPIQ. It explicitly
states quality calibration on UPIQ; consequently a student trained on these labels
cannot treat UPIQ as an independent human test. No UPIQ label/image or secret holdout
is opened by this task. No model is trained, qualified, promoted or pushed.
Evidence and final artifact admission: [worklog](../benchmarks/hdrcorr_WORKLOG.md),
[teacher report](../benchmarks/hdrteach_2026-10-04.md).

## Exposure ledger — 2026-10-05: SHIPPATH Rev5 teacher admission smoke

Explicit SHIPPATH brief; own jj workspace and local commits only. Reads only
SafeSyn and CID22 201-reference oracle TRAIN fit/internal-development tables
from frozen `/var/tmp/rev4-featpot/v2c5/wide/main/real`. New view
`/var/tmp/shippath/teachers` preserves the four Parquet files byte for byte:
141,054 / 38,757 SafeSyn fit/dev rows, 12,163 / 3,785 CID22 fit/dev rows.
Internal development remains TRAIN. No T0/human/holdout labels, `_sealed`
files, human_all tables or coverage targets were read. Bank features/keys and
producer manifests were hashed for the two teacher sets only.

The canonical trainer ran one epoch / 500 pairs, N head / H128 / the existing
420 by_v2fy IDs, init 1101 / sample 101, no auto evaluation, no historical
replay. Embedded admission reports qualified table provenance at Rev5.
This is software plumbing validation: it does not reproduce cv16:cf98, change
design populations into production TRAIN, assess a holdout, select a model,
or establish production qualification. Receipt and commands:
[SHIPPATH worklog](../benchmarks/shippath_WORKLOG.md).

## Exposure ledger — 2026-10-05: SHIPPATH2 recipe metadata views and TRAIN export smoke

Coordinator authorized complete metadata-only fresh copies of frozen Rev5 v2c5
main/real human and oracle teacher recipe tables, plus the E15 ordinal pool.
All 21 original recipe Parquets and the pool are byte-preserved. Human key reads
project only pair_key/source_row_id/ref_basename/member_set; table reads project
only reference names. Human target columns are not decoded, fitted, assessed or
selected on. Five-source design release/exposure remains the September 30 ruling
above; no production TRAIN reclassification is made. The explicit decision
request is `benchmarks/SHIPPATH_decisions.md` (also delivered with the report).
Both strict wrapper entry points were checked to refuse the actual pending
human role before trainer invocation; no real approved decision was created.

The registered local plumbing smoke uses only SafeSyn and CID22 oracle TRAIN
fit/internal-development, plus 17,600 cf98 rungs of the existing KADIS TRAIN ordinal
pool (light/spatial/new). H128/N/420 IDs, init 1101/sample 101, 3 epochs × 500 draws;
last-epoch export, canonical densify and f16 pack with explicit CID22 oracle TRAIN
fit spline anchor, then oracle TRAIN-development packed inference. No human fit,
full recipe/seed selection, confirmation/T0/secret label, EVAL, protected read,
fleet, publishing or scientific quality claim. Coverage-family filtering does
not alter feature or ordinal target values. Frozen source roots stay immutable.

## 2026-10-05 — SHIPPATH7 features-only preparation; exposure pending

The coordinator authorized Rev5 assessment features only, using registered gate
populations and by_v2fy's existing420 IDs. Fifteen declared tables retain original
keys/order, identity rows and byte/decoder pins. Confirmation/terminal banks are
projected without sealed labels; no protected/confirmation/T0 labels were read.
This preparation does not release labels or reassign data roles. R7 panels remain
spent; KADID terminal remains subject to its terminal read contract.

Before a future label read, freeze the complete final composition/companion and
calibration/thresholds, TRAIN recipe/choices, all instrument/decoded pixel/decoder/
evaluator hashes, populations and original roles, statistics/multiplicity and
prior exposure, then obtain owner authorization. The five-source human union's
production-role decision remains pending in SHIPPATH_decisions.md. No decision,
training, candidate selection or qualification is implied by feature admission.
See [worklog](../benchmarks/shippath7_WORKLOG.md) and its explicit exposure receipt.

## 2026-10-05 E26 native Rev5 teacher leg (registered execution)

[Registration](../benchmarks/e26_hdr_teacher_registration_2026-10-05.md) fixes
by_v2fy, headN, hd4/hd16, ten seeds/five folds, unchanged E24 Rev5 controls.
TRAIN alone supplies7390 agreement rows from the immutable HDRTEACH TRAIN
table; target exactly10*q_jod, no clipping, fit-only withinref,rank with
coverage acceptance weighting. Transform committed before any fit at8e8f3170.
VAL retains all3900 rows/300 references, including disagreements, for the
frozen registered two-teacher assessment. No HDR dev, calibration, early
stopping or checkpoint search. Every fit selects zero-based epoch119.
Native PQ16 PNG/JXL, declared actual primaries and PQ10000 use the production
planned HDR fold through the existing research/extractor owners, exact420
IDs, f64 features, absent NaNs; fitting casts those measured slots to f32 as
the SDR owner does. Original file hashes, order, keys, extraction executable
and source are pinned in native producer manifests. Explicit IDs/per-slot
provenance identify the channel subset; its reconstructible family-token
FeatureSetId is null. Never replace that with a width-inferred identity.

[Native bank pins](../benchmarks/e26_native_features_2026-10-05.pointer.json).
Program v28=7e8ac08b6b97fc78057251bdfa4af2a605543d8a36c0b2e1be4aec458634d5a0;
TRAIN-only data pack=2eb85985a69a255a784c4b8f50a1a9180034c8ccc69d92f95d179c5ba723d3c2.
P2 admission fix checks record/sidecar contract before any candidate payload
opens; actual loader/packer forbidden-VAL read tripwires pass. Original failed
smoke retained. No UPIQ or sealed read. Encoder commits remain the inherited
HDRTEACH unknowns; new byte hashes cannot establish missing codec commits.

Coordinator-authorized serving binding, identical for50 controls and100 arms:
existing dense_bake BIT-IDENTICAL512-row gate then pinned bake_stamp_revision
to separate Rev5 outputs with per-bake source/dense/stamped/gate receipts.
Immutable cells remain untouched. v2c5 trainer logs formula_revision null and
qualified_provenance false are an explicit admission gap, to fix forward
separately; serving revision comes from the stamp, not retrospective trainer
qualification. Report-only external SDR, canonical corruption TRAIN and one
fixed HDR TRAIN steering panel cannot change the registered adoption rule.
[Execution worklog](../benchmarks/E26_WORKLOG.md).

Final registered E26 result: hd4 is the lowest passing weight; both arms pass
SDR as-good and the two within-reference HDR conditions over all50 matched
cells. Hd4 HDR-VDP-3 delta+0.0014655677655677746, SE0.0003411001471767929;
CVVDP delta+0.0015161172161172077, SE0.0003392474364166754. Hd16 deltas
+0.0014747252747252793 / +0.0015007326007325662, SEs0.0003378213879858726 /
0.00033559844516359565. Pooled HDR-VDP-3 falls(-0.044864262128246485 hd4,
-0.12476287879994075 hd16); it remains reported-only under the preregistration.
Human HDR, encoder RD/spatial and full product qualification remain MISSING.
All150 full VAL panels retain3900 rows/300 references and per-bake bindings;
whole native/cache control parity is bit-identical with0 pixel-identical rows.
Full records: output/zensim/e26-2026-10-05 (tower); large JSONs are external.
V28's actual packed Rust executables are the existing fleet-v2 set
6b28576f/81ec2207/c9c610b8, not the v25b hashes in copied binary_mix prose.
The embedded inventory is correct; supplemental provenance erratum records
actual hashes and bounded canonical parity evidence without rewriting the pack.


## 2026-10-05 E27 registered preparation on the E26 landing chain

Registration070b8247 fixes hp4 pooled rank and ha4 within-reference MSE+rank,
nominal HDR weight4 through E26's unchanged weighting convention. Every other
fit parameter and original TRAIN role/authority stays E26's:7390 agreement
rows, unclipped10*q_jod, Rev5 native420 by_v2fy IDs, headN, seeds0–9/five folds,
120 epochs/final119. Fresh v2e27 transport preserves all56 E26 payload hashes;
no HDR VAL/development/confirmation payload is added. A one-cell hp4 smoke
checks mechanics, not scientific adoption. E27 HDR VAL is not read for prep.

The assessment owner now supports the registered E27 pooled and within-reference
gates for both teachers, SDR E21, and larger passing pooled HDR-VDP-3 gain;
E26 hd4 and full raw/cell geometry remain report-only. Serving binding stays
approved dense prediction gate followed by receipt-bound Rev5 stamps, identically
for controls and new arms. Exact E26 executables/shared program data are reused;
merged runtime scripts and their v2c_wide dependency are explicitly pinned.
The old trainer's null revision/unqualified admission remains an inherited gap;
no stamp upgrades training qualification. Human HDR, original encoder commits,
encoder RD/spatial and product/runtime qualification remain MISSING.

E26 landing review found no defects. No E27 enqueue or source push; the launch
owner requires coordinator confirmation of E26 landing and matching local
zenmetrics profile push. No tail controller. A broad filename-only tool lookup
reached protected _sealed directory enumeration and child access was denied;
no label payload was opened and no data came from that attempt. Further tool
lookup is restricted to named tool directories. See the E27 worklog and
[preparation pointer](../benchmarks/e27_preparation_2026-10-05.pointer.json).

E27 preparation smoke completed: full hp4/kadid/s0, final119,7390 HDR TRAIN,
7869SDR predictions, zeroHDRdev; native/cache one-TRAIN-row score bit-identical
74.63024139404297 after dense gate+Rev5 stamp. Default-Rev1 native refusal was
retained and corrected by explicit Rev5 invocation, with no refit. No HDR VAL
assessment or fleet enqueue during preparation. Readiness is separately gated
on coordinator E26 landing and matching zenmetrics push confirmations.

## 2026-10-06 E27 registered assessment completed

All100 final119 arm cells and200 complete HDR VAL panels were verified. Neither
arm passes E27: hp4 passes SDR and pooled HDR-VDP-3 improvement but fails pooled
CVVDP noninferiority; ha4 fails SDR and HDR guards. Retain the control. E26 hd4,
all external categories and raw geometry remain report-only. Historical table
admission stays unqualified; approved Rev5 stamps supply serving binding. Human
HDR judgment, encoder provenance completion and shipping gates remain missing.
See [E27 worklog](../benchmarks/E27_WORKLOG.md),
[complete measured summary](../benchmarks/e27_result_summary_2026-10-05.json) and
[immutable Tower evidence](../benchmarks/e27_final_2026-10-05.pointer.json).

## Exposure ledger — 2026-10-07: owner decisions on production human data, KADID TERMINAL and UPIQ-380

Owner decisions, verbatim (2026-10-07, in reply to the decision brief https://claude.ai/artifact/8DxtQmJErgsxTfHRytaCHF):

> D1 (human data): B — approve KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL, CID22-A; AIC-3 stays in the holdout family
> D2 (KADID TERMINAL): B — register now, read once on the final qualified model
> D3 (UPIQ): B — re-designate UPIQ-380 as HDR training data

* **D1 — production human training population.** KADID-10k TRAIN + SELECT, TID2013, KonFiG-IQA TRAIN + VAL and CID22-A (25 references)
  may train the qualified production model (`shippath-human-role-decision-v1`, `allowed_use = qualified-recipe-training`). **AIC-3 is
  excluded** and stays in the JPEG-AIC holdout family (rule `jpeg-aic-family-holdout-2026-09-01`, §3d), together with the AIC-4 sample
  and SDR25. Consequence recorded: the R7 confirmatory fits (2026-10-04) trained on all five design sources including AIC-3 and were read
  on the AIC-4 sample, which shares AIC-3's content; that part of the R7 read is flagged as possibly contaminated by the family rule. The
  production recipe drops the AIC-3 human leg; the change is checked by a registered comparison before the production fit.
* **D2 — KADID TERMINAL** (2,000 pairs, never read) is reserved for exactly one confirmatory read of the final qualified model; the read
  is registered now (`benchmarks/kadid_terminal_registration_2026-10-07.md`) and no label is opened before that model exists and passes
  its release gates.
* **D3 — UPIQ-380** (UPIQ's HDR subset) is re-designated from T0 to T2 training data for HDR legs. Basis: it is already burned as a
  test (~21 looks, `docs/DATASET_HISTORY.md`), and E26/E27 showed teacher-only HDR legs cannot settle cross-image HDR calibration. Any
  other UPIQ portion stays T0. A new independent human HDR test is needed later (the planned Squintly HDR study).


## 2026-10-07 E28 registered SSIM2-recipe preparation

[Registration](../benchmarks/e28_ssim2recipe_registration_2026-10-07.md)
fixes two bake arms and one grouped POTENTIAL diagnostic on the Rev5 by_v2fy
420-ID subset. [Teacher pin](../benchmarks/e28_teacher_pin_2026-10-07.json)
records the unchanged SafeSyn/CID22 tables and the exact human member sets.
The new dataset legs are derived from the existing admitted exploratory full
source tables, retaining original labels and the R1 reference-hash dev split.
Pooled rank and Pearson comparisons never cross dataset/teacher legs.

`s2o` retains the R1 teachers, four non-held-out human sources and coverage;
shared-scale SafeSyn/CID22/KADID/TID legs receive the new pooled terms.
`s2m` admits only CID22 fit, KADID TRAIN, TID2013 and KonFiG TRAIN, excluding
the held-out source where applicable. Its four-dataset restriction excludes
SafeSyn, KADID SELECT, KonFiG VAL, cid22_a25, AIC-3 and ordinal coverage.
KonFiG uses the design-grid label, with the registered reconstruction caveat.

The local kadid/seed-0 smoke of each bake arm opens admitted held-out KADID
labels only after the fixed final-epoch fit. The NM smoke standardizes included
fit rows only and opens held-out KADID only after freezing its parameters.
E24 KADID control-cell statistics supply report-only NM deltas. No protected
bank labels, HDR VAL or confirmation payloads are inputs to E28 preparation.
No external SDR labels are opened during preparation. All results remain
POTENTIAL; historical replay does not establish strict training qualification.
The data transport includes admitted exploratory assessment tables, excludes
HDR/confirmation members, and has not been uploaded or enqueued.

[Preparation evidence](../benchmarks/E28_WORKLOG.md) records fixed choices,
local full-budget smokes, finite diagnostic non-convergence, source pins and
resource measurements. The complete registered scientific verdict requires
100 harvested bake cells and 50 matched E24 control cells.


### E28 admission followup (2026-10-07)

The new E28 human fit/dev views are additionally bound to the frozen
`benchmarks/e28_admission_inventory_2026-10-07.json` inventory. MLP and NM admit
all active manifests before opening training labels, then validate the four
label-free key columns, ordered row identities, allowed member sets and the
reference-hash split. Preparation copies an explicit source inventory only and
refuses confirmation/HDR directories before copying. The 37 numerical tables
are unchanged; key sidecars, manifest bindings and receipt provenance changed.
Producer and artifact receipts: `benchmarks/e28_admission_2026-10-07.pointer.md`.
The existing E28 exploratory population and qualification limits still apply.

## Exposure ledger — 2026-10-07: PALETTE research-only feature extraction

The PALETTE lane decoded pixels and projected keys/paths/pixel hashes only;
no human labels, protected/holdout payloads or AIC-3 row enumeration were read.
D1 roles remain unchanged. KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL and
CID22-A25 contribute ordered design views; SafeSyn/CID22 train fit/dev and
coverage companions retain their existing roles. NITS/LIVE/MCIQA sidecars
are features-only, report-only. No E32 fit or qualification was performed.

Current independent arithmetic is palette_v2, IDs f1825..f1866, producer set
`palette@w1867/palette_v2#30b09cd1`, full-width research width 1867.
Build `e60a6ad74a47981f93f969d63b605c7a88ab09b0` produced 254,778 bank rows and
209,576 ordered instrument observations under
`/mnt/v/output/zensim/palette-2026-10-07-v2/`. The instrument uses explicitly
mapped `palette_f1825`..`palette_f1866` columns to avoid overwriting existing
auxiliary f1825+ owners; repeated KADID observations remain in source order.
Coverage path-derived keys are a separate domain from canonical pixel-pair
keys. All bank hashes and every instrument feature join were independently
verified. The initial palette_v1 tree is superseded and incompatible with v2;
it remains preserved and cannot be mixed into a fit. Serving reads are refused.

Bank/instrument/verification SHA-256s respectively:
`46587338cc74ba59e38fe96e637776bfe74135fde29f3f70e22b51605fc50917`,
`9f7523bf7d3aaa32418d40d83adb44edccff9e70cc75acc32eb8d5711fe89934`,
`0f28b79a64061a62748f5f41c2cae35061f7875e3ff8d991b5ad25289307ad48`.
Archive: `/mnt/tower/output/zensim-palette-archive-2026-10-07/palette_v2/`;
three random file hashes match the mirror. R2 mirror absent. Full provenance,
measured diagnostic limits and the inherited library failures are recorded in
[PALETTE pointer](../benchmarks/palette_2026-10-07.pointer.md).

## Exposure ledger — 2026-10-07: PALETTE2 identity admission correction

No new pixel extraction, human-label read, model fit or role change. Canonical
palette_v2 features retain producer e60a6ad7 and the original pinned bank and
instrument manifest bytes. New instrument admission requires the consumer's
frozen instrument manifest SHA-256 and exact palette_v2 identity, ordered
integer IDs, map, producer commit and false serving flag before feature reads.
Strict palette_v1 and swapped-map negative controls reject consistently
rehashed wrong identities as well as changed bytes. The original value-join
receipt (hash recorded above) is retained as `_VERIFIED.round1.json`; the new
`_VERIFIED.json` SHA-256 is `5424cd0d015ca48e94d3a04fc94a2507eadd75e805e977ec39842208756465b1`.
All 254,778 bank rows and 209,576 instrument observations passed the new
verification. External roles and D1 stay unchanged; AIC-3 remains unread.

The supported public token/API snapshots are unchanged from rebased main.
E32's [final registration proposal](../benchmarks/e32_palette_registration_2026-10-07.md)
requires E30's complete frozen 40-cell nA3 control and exact parity, or one
fresh matched control registered before any E32 fit. The four-source seed
composite and fixed KADID/TID W2 reductions are defined before outcomes.
No launch is authorized until the coordinator registers and freezes all
transport/program/data/control pins. See the [round-two pointer](../benchmarks/palette2_2026-10-07.pointer.md).

## 2026-10-07 E28 registered assessment completed

All 100 final-119 cells (`fitv2e28b-20261007`) were harvested and audited; the registered E28 assessment read the five
exploratory held-out panels already admitted in the v2c5 instrument (kadid, tid2013, konfig, cid22_a25, aic3) — the
same panels E21–E27 read. Neither arm passes; control retained. This is the pre-D1 five-source exploratory design: its
s2o arm trained on AIC-3 rows when AIC-3 was not held out. E28 models are research-only, are never read on AIC-4 or
SDR25, and never ship; D1 production qualification excludes AIC-3. No protected, sealed, KADID TERMINAL or T0 payload
was read. Records: `benchmarks/e28_result_summary_2026-10-07.json`, `benchmarks/e28_final_2026-10-07.pointer.md`.

## 2026-10-07 E30 report and production fit launch

E30's 40 four-source cells (`fitv2e30-20261007`) trained only on the D1 population (no AIC-3 row) and were scored on the
four D1 held-out panels against the pinned E24 control cells; no AIC-family, protected, sealed, KADID TERMINAL or T0 label
was read. The AIC-3 drop shows no measured cost (signed +0.0003 ± 0.0015). The D1 production fit (`fitv2d1-20261007`,
three full-data seeds on KADID TRAIN+SELECT, TID2013, KonFiG TRAIN+VAL and CID22-A) launched afterwards. Records:
`benchmarks/e30_result_summary_2026-10-07.json`.

## Exposure ledger — 2026-10-07: E29 preparation read a legacy HDR VAL panel (incident)

During E29 implementation, a new unit test imported `scripts/hdr/hdr_route_panel.py`, whose CLI ran unguarded at import and
read the entire legacy HDR VAL panel `/mnt/v/zen/zensim-training/hdrgrid-mc944-t1-2026-08-27/hdrgrid_mc944_t2_val.parquet`
(22,860 rows × 952 columns, including its teacher-score `human_score` column) and printed a target-swing diagnostic before
failing. No student prediction, no model fit, no threshold or target choice used it; the registered hdr_v3mix 3,900-row VAL,
confirmation, AIC and human-label payloads were not opened. The module now has a main guard and zero-open import tripwires.
This legacy panel's teacher targets count as seen by the E29 lane from this date. Receipt:
`/mnt/v/output/zensim/e29-2026-10-07/UNINTENDED_EXPOSURE.json` (build `a9b92db3`).

## 2026-10-07 D1 production fit complete

`fitv2d1-20261007`: three full-data seeds (0–2), by_v2fy at Rev5 on the D1 population, each 120 epochs × 50,000 pairs, epoch
119, densified, packed to f16 and TRAIN-calibrated in the fit; harvest bound to the registered budget and admission. No
evaluation label was read by the fit. Packed models: s0 `f803b74c…`, s1 `1bf8f3af…`, s2 `54118c54…` (tower
`/mnt/tower/output/zensim-production-d1-2026-10-07/`). Release gates that read evaluation data wait for the owner to freeze the
final composition (which seed or ensemble) before any read, per the scorecard.


### E29 four-source HDR-consensus preparation (2026-10-07)

Local amendment only; no full E29 fit, launch or assessment. D1 source/role
approval is unchanged. Preparation copies the exact frozen four-source SDR
inventory and 7,390 existing agreement-only HDR TRAIN rows; hb4 changes only
the target and hc4 adds the complete two-teacher cross-reference pair list.
An import-test mistake opened the entire 22,860-row/952-column legacy
`hdrgrid_mc944_t2_val.parquet` before mocks were installed. It computed target
swings, then failed before student prediction; no model fit used it. The
panel import is now guarded and zero-open tripwires pass. The unintended
legacy HDR VAL exposure violates the preparation limit and is disclosed;
registered 3,900-row hdr_v3mix VAL, AIC, confirmation and protected labels
remain unopened. Native HDR
subset provenance remains explicit and unqualified. All 40 completed E30 nA3
model/result artifacts were verified and pinned without label reads; exact
program reuse parity is not asserted. One matched-control manifest remains a
registration proposal before any full fitting. Role/exposure receipt, staged
root, executor smokes, producer pins and tower mirror:
[E29 worklog](../benchmarks/E29_WORKLOG.md) and
[preparation pointer](../benchmarks/e29_preparation_2026-10-07.pointer.md).
