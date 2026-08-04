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
| **KonJND-1k** (1,008 refs; JPEG+BPG PJND) | T1 (semi) | train = konjnd-dense (20,160 rows, per-pair active-mix target) AND val = per-ref mean PJND — same 1,008 refs both sides → ref-level train==val; treat KonJND eval as guard+anchor, not holdout. **MEASURED 2026-08-04 (wave 6): the set is 504 JPEG refs ∪ 504 BPG refs, intersection 0**; the 944 eval leg `ext_konjnd_jpeg_val.parquet` is **exactly the JPEG 504**, and `konjnd-dense − eval` is **exactly the BPG 504**. So the ONLY reference-disjoint KonJND training mass is the BPG half — **and zensim has no BPG decoder**, so it cannot be extracted at 944/924/720. ⇒ *there is no legitimate KonJND training leg at any post-372 regime*; adding one needs a BPG decode path (re-encoding those refs with a supported codec changes the distortion type and voids their PJND targets). The 372 dense build's pair list + active-mix target are not recoverable from any committed artifact (parquet carries `ref_basename` only) | Authors: whole-set JND benchmark, no split defined | — |
| **AIC-3 CTC** (600 pairs, 10 refs) | T0 | Eval-only, never train | JPEG-AIC committee test set; Mohammadi 2025 evaluates metrics on it | Holdout by construction |
| **AIC-4 sample** (300 pairs, 5 refs) | T0 | Eval-only, never train; do NOT recipe-search to win it (holdout-fishing ban, 2026-05-25 #10) | The CfP keeps the larger set committee-hidden — public sample is eval-only for everyone | Holdout by construction |
| **JPEG-AI-SDR25** (5 src × 10 levels, 95k raw triplets) | **T0 (BUILT 2026-07-02)** | Eval-only. Reconstructed: `sdr25_jnd_reconstructed_2026-07-02.parquet` (116 stimuli, ordered-probit triplet MLE, `scripts/v_next/reconstruct_sdr25_jnd.py`; response = MORE-DISTORTED side, trap-verified). Scoreable subset = 5×10 JPEG-AI PTC crops (anchor codecs not in the public zip). Baseline: within-image SROCC A 0.998 / ssim2 1.000 (both ceiling); pooled A 0.904 vs ssim2 0.958 → A currently FAILS the "SDR25 ≥ ssim2" gate | Authors: subjective study behind the QoMEX'25 SVQA paper (arXiv:2504.06301) | Postdates ssim2 tuning — honest holdout for both sides |
| **KonFiG-IQA** (10 src × 7 dist × 12-30 levels over 3 JND; 1.7M triplets) | **T2 (INGESTED 2026-07-02)** | `konfig_train_2026-07-02.parquet` (1,090 rows after identity+content dedup; human_score = 1−q_jnd/3.2, native `q_jnd` col; targets from the DESIGN grid — levels calibrated to 0.25/0.1 JND spacing, no reconstruction needed). All-train (v53 replicate-axis group). **ssim2 tuned on it** → never a ssim2-comparison corpus | Authors: fine-grained JND-unit scales via boosted triplet comparisons (Men 2021) | 10 sources, disjoint from all T0 refs by construction (Konstanz set); dHash spot-audit pending |
| **PIPAL** (local, unused) | — | Not in pipeline (SR/GAN domain) | Official NTIRE train/val/test splits | — |
| **UPIQ / HDR** | T0-eval for HDR track | Held-out UPIQ eval per HDR plan | Mikhailiuk 2021: consolidated dataset, JOD-rescaled | — |

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
7. **Multi-metric backfill of the 5.7M canonical corpus is IN FLIGHT
   (2026-07-02)** — cvvdp/butteraugli/iwssim/dssim being added to the
   2026-06-27 datasets (backfill recipe: score persisted variants via
   `variant_r2_url`, no re-encode). When it lands: (a) it must arrive as NEW
   dated files/sidecars joined on content-addressed keys, never in-place
   rewrites of files a manifest references (§1.4); (b) rebuild
   `bigcodec_multimetric_<date>.parquet` with per-zone targets — the wave-4
   HQ-band supervision substrate (cvvdp/butteraugli in the ≥0.85 band where
   ssim2 saturates); (c) re-derive the digit-split train/val files from it.
   Until then, ssim2-target waves (v51) validate the held-out-val selection
   fix, NOT the final supervision design.
