# DATA_SPLITS.md — canonical train/val/test conventions (locked 2026-07-02)

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
| **CID22** (Cloudinary, 4,292 val pairs / 49 refs + 201 train refs) | T0 (49-ref) + T2 (201-ref, ssim2-anchored) | 49-ref set = sacred eval-only; 201 disjoint refs trainable with **ssim2 targets only** (verified: `cid22_train_norm.human_score == ssim2_gpu/100` exactly; human MCOS never trains) | The CID22 paper itself: 201 refs tuned SSIMULACRA2, 49 held out — we mirror the authors' own split | dHash-audited; synth corpus purged 2026-05-12; imazen-26 clean at d≤10 (2026-07-02) |
| **KADID-10k** (10,125 pairs, 81 refs, DMOS) | T1 | Full set trains (v47 w0.5) AND full set evaluates → train==val integrity guard | No official split; literature: random by-reference 80/20 (or 60/20/20) × 10 repeats, median SROCC. **ssim2 tuned on ALL of it** → never scoreboard vs ssim2 here | 6 training sources flagged d≤10 vs KADID refs (2026-05-14, mostly flat-content FPs, user review pending) |
| **TID2013** (3,000 pairs, 25 refs, MOS) | T1 | Same as KADID (v47 w0.5, train==val) | Same literature convention (by-reference CV); ssim2 tuned on all of it | 1 source d=10 (flat-content FP, review pending) |
| **KADIS-700k** (700k cells, 140k sources, NO human labels) | T2 + T3 | §2b modulo rule; train=<8, safety-grid=9; targets = GPU metrics (cvvdp/10 primary) | Authors (Lin/Hosu/Saupe): weak-label TRAINING set for FR-metric distillation (DeepFL-IQA) — no human labels, no eval role. Our train-on-metric use matches the authors' intent; our %10 split adds held-out safety eval they didn't define | Reference pool is KADIS (Pixabay), disjoint from KADID's 81 refs per the authors; our safety grid excludes signed types 7/18/25 (severity≠quality there) |
| **imazen-26 / canonical-picker-2026-06-27** (5,742,660 cells, 414 origins) | T2 | §2a LSD rule for picker work. For ZENSIM training (bigcodec_5p7M) all three buckets train — zensim's holdouts are T0 corpora, not picker buckets. NOTE the consequence: picker-val/test origins are seen by zensim bakes | N/A (our corpus) | imazen-26 origins vs CID22-49: **CLEAN at d≤10** (min d=12, 2026-07-02, `imazen26_vs_cid22_dhash_t16.tsv`); 16/1067 decode-failures unaudited (odd screen PNGs) |
| **safesyn** (196,086 pairs) | T2 | All train; ssim2-derived targets; CID22-leak-purged 2026-05-12 | N/A (our synthetic corpus) | Purged at d≤16 (loose-threshold caveat documented) |
| **KonJND-1k** (1,008 refs; JPEG+BPG PJND) | T1 (semi) | train = konjnd-dense (20,160 rows, per-pair active-mix target) AND val = per-ref mean PJND — same 1,008 refs both sides → ref-level train==val; treat KonJND eval as guard+anchor, not holdout. **MEASURED 2026-08-04 (wave 6): the set is 504 JPEG refs ∪ 504 BPG refs, intersection 0**; the 944 eval leg `ext_konjnd_jpeg_val.parquet` is **exactly the JPEG 504**, and `konjnd-dense − eval` is **exactly the BPG 504**. So the ONLY reference-disjoint KonJND training mass is the BPG half. **CORRECTED + RESOLVED 2026-08-04 (wave 7, campaign amendment 7):** the "no BPG decoder ⇒ cannot be extracted" claim was wrong — the KonJND-1k distribution ships the BPG half **pre-decoded** (`KonJND-1k/bpg/` = 25,704 valid 640×480 RGB8 PNGs, 504 refs × 51 QPs, upstream 2021 mtimes; zero `.bpg` bitstreams exist on disk), the 372 dense build had already extracted those very pixels (10,080 BPG rows), and the dense build's pair list + target rule WERE recovered exactly from `konjnd_full_scored.csv` (20 rank-evenly-spaced picks/ref over the ssim2-sorted ladder; `human_score` = raw `gpu_ssimulacra2`; verified 1008/1008 refs <1e-9). The reference-disjoint 944 training leg now exists: `ext944-canonical-2026-08-01/konjnd_bpg_{train,val}_944.parquet` (403/101 refs, srcnum%10∈{8,9}→val, target = ssim2/100; `_MANIFEST_konjnd_bpg.json`) | Authors: whole-set JND benchmark, no split defined | — |
| **AIC-3 CTC** (600 pairs, 10 refs) | T0 | Eval-only, never train | JPEG-AIC committee test set; Mohammadi 2025 evaluates metrics on it | Holdout by construction |
| **AIC-4 sample** (300 pairs, 5 refs) | T0 | Eval-only, never train; do NOT recipe-search to win it (holdout-fishing ban, 2026-05-25 #10). **⚠ TARGET IS DISTORTION-ORIENTED** (`q_jnd`, same reconstruction family as SDR25): all 188 board fullevals report negative `srocc_signed`. Correct for a JND study; negate before any training use. **⚠ SDR25 ⊂ this corpus** — SDR25's 50 rows are the JPEG-AI subset of these same 300 rows / 5 crops (verified 50/50 on `ref_basename`+f0..f5), so scoring both is not two independent reads. See campaign Appendix I | The CfP keeps the larger set committee-hidden — public sample is eval-only for everyone | Holdout by construction |
| **JPEG-AI-SDR25** (5 src × 10 levels, 95k raw triplets) | **T0 (BUILT 2026-07-02)** | Eval-only. Reconstructed: `sdr25_jnd_reconstructed_2026-07-02.parquet` (116 stimuli, ordered-probit triplet MLE, `scripts/v_next/reconstruct_sdr25_jnd.py`; response = MORE-DISTORTED side, trap-verified). Scoreable subset = 5×10 JPEG-AI PTC crops (anchor codecs not in the **JPEG-AI** zip — but they ARE on disk in the AIC-3 package, `aic3-btc-ptc/test-images/{BTC,PTC}_images.zip`, 5 refs × {AVIF,JPEG-1,JPEG-2000,JPEG-XL,VVC} × 10 levels; corrected 2026-08-04). **⚠ TARGET IS DISTORTION-ORIENTED** — `human_score` = `q_jnd`, a JND *distance* from the original (rises with distortion). Verified three ways (source, raw ladder, and signed SROCC **−0.9757** vs 67,714 raw votes); all 171 board fullevals report negative `srocc_signed`. This is CORRECT for a JND study and **must NOT be flipped** (the seed-selection oracle consumes `\|SROCC\|`; flipping silently inverts it). **Negate before any training use.** Gated by `check_target_orientation.py` (declares `distortion`). **⚠ SDR25 ⊂ AIC-4**: all 50 rows are present in `ext_aic4.parquet` (300 rows, same 5 crops) — they are NOT independent eval corpora. **NOT TRAINABLE** — T0 + it is the seed-selection oracle (+0.752 → CID22 over 35 bakes) + 5 refs. Full determination: campaign **Appendix I**. Baseline: within-image SROCC A 0.998 / ssim2 1.000 (both ceiling); pooled A 0.904 vs ssim2 0.958 → A currently FAILS the "SDR25 ≥ ssim2" gate | Authors: subjective study behind the QoMEX'25 SVQA paper (arXiv:2504.06301), Jenadeleh/Sneyers/Jia/Mohammadi/Ascenso/Saupe — cite arXiv:2504.06301 | Postdates ssim2 tuning — honest holdout for both sides |
| **KonFiG-IQA** (10 src × 7 dist × 12-30 levels over 3 JND; 1.7M triplets) | **T2 (INGESTED 2026-07-02; 944 LEG BUILT 2026-08-05)** | 944 leg: `ext944-canonical-2026-08-01/konfig_944.parquet` (1,090 rows, 85+24 per source, + native `q_jnd`; multiset-identical to the 372-era `konfig_train_2026-07-02.parquet`; builder `scripts/canonical_corpus/build_konfig_944.py`; campaign **Appendix L**, pre-reg `e93eba04`). human_score = 1−q_jnd/3.2 — **QUALITY-oriented, gated**: `check_target_orientation.py` declares `quality`, verified signed SROCC **+0.5645** vs the 75,519 raw EXP_III DCR votes (n=850 PartA; PartB shares the formula). Origin-split views `konfig_originsplit_{train,val,test}_944.parquet` (327/436/327; `split_of` on numeric src id) exist for any future within-KonFiG instrument; the registered probe leg is the FULL table (L.6 design decision — training on it forecloses those views as eval for those models). **ssim2 tuned on it** → never a ssim2-comparison corpus | Authors: fine-grained JND-unit scales via boosted triplet comparisons (Men 2021) | 10 sources. **dHash spot-audit RUN 2026-08-05 (Appendix L G-L1/G-L2, commit `7ed6ac4b`): CLEAN PASS** — 0 exact hits + zero d≤10 flags vs KonJND-1008 / CID22-49 / CSIQ-30 / LIVE-29 / AIC3-10; global min d=17 (dHash is crop-blind — residual stated in L.11.8) |
| **PIPAL** (local, unused) | — | Not in pipeline (SR/GAN domain) | Official NTIRE train/val/test splits | — |
| **UPIQ / HDR** | T0-eval for HDR track | Held-out UPIQ eval per HDR plan | Mikhailiuk 2021: consolidated dataset, JOD-rescaled | — |
| **hdr_v3mix @944 (hdr944-leg)** (17,100 zenjxl HDR-PQ cells → 7,410 train + 3,900 val after dedup; 58 imazen-26-hdr origins) | **T2 (944 LEG BUILT 2026-08-03; REGISTERED 2026-08-05)** | Digit origin split on the leading numeric stem (`origin_split.split_of`): 38 train / 20 val origins, overlap 0, `split_of` agrees 870/870 refs (campaign **Appendix Q** G-Q3). Features = chunk-2 HDR route at `Folded720Append2` (`compute_folded720_append2_features_hdr`, PQ 10k nits); target = cvvdp-mix `0.5·clip01(ssim2/100)+0.5·clip01((JOD−6)/4)`, **quality-oriented, gated** (`check_target_orientation.py hdr_v3mix` in-table mode: train +0.8494 / val +0.8606; caveat: consistency vs carried JOD, not independent). NEW-REGIME leg — never column-mix with SDR tables or the v3 pu-linear 372 corpus (same targets, different feature space). Manifest: `/mnt/v/output/zensim/hdr944-leg/_MANIFEST.json`; Tower `zensim-hdrp1-2026-08-05/hdr944-leg/` | N/A (our corpus; targets are metric teachers) | G-Q4 PASS: imazen-26 zfold7 2026-03 personal captures are temporally+authorially disjoint from every HDR eval source (UPIQ narwaria/korshunov, SI-HDR, HDR-VDC, AVT, CHUG, Rousselot); id-containment 0 hits |

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

## 6. Remote training (Hetzner) — HETZNER-FIRST for all slow work

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
