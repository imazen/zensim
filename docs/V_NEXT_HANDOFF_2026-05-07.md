# zensim v_next — handoff for a fresh session

**Date:** 2026-05-07
**Author context:** dropped after 12 h of v15-series sweep work; this handoff
captures the data state + concrete TODOs needed to take zensim past V0_6/V0_7.

This doc is **complementary** to
[`docs/NEXT_TIER_DATA_PLAN.md`](./NEXT_TIER_DATA_PLAN.md) (the 2026-05-04
research baseline). Read that first for the architectural context (MLP
runtime, ZNPK format, V0_2..V0_6 history, CID22/KADID/TID validation
splits). This doc lists **new data that landed since then** + the **next
training cycle's concrete actions**.

---

## 1. New training data that landed 2026-05-06 → 2026-05-07

All sweeps emit per-cell zensim 300-feature parquet sidecars (the
`compute_extended_features()` output, schema documented in
`/home/lilith/work/zen/zenmetrics/crates/zen-metrics-cli/src/sweep/feature_writer.rs`).
Joins back to the matching TSV on `(image_path, codec, q, knob_tuple_json)`.

| Sweep run_id (R2 prefix) | Codec | Cells | Source corpus | Notes |
|---|---|---|---|---|
| `s3://zentrain/sweep-v12-2026-05-06/` | webp + jxl + avif | ~12K | 16 MP archival | mixed-codec, sparse zensim metric coverage |
| `s3://zentrain/sweep-v13-2026-05-06/` | zenjpeg | ~36K | 16 MP archival | dense q-grid, full knobs |
| `s3://zentrain/sweep-v14-2026-05-06/` | zenpng | ~2.4K | 16 MP archival | lossless reference cells |
| `s3://zentrain/sweep-v15-2026-05-06/` | zenjpeg | ~635K | 16 MP archival | **abandoned** — 16MP made cells 50× slower than necessary |
| `s3://zentrain/sweep-v15r-2026-05-06/` | zenjpeg | **~1.79M** | **1024 px Lanczos3** | full grid, primary new corpus |
| `s3://zentrain/sweep-v15r-2026-05-06/zenjpeg/baseline.tsv` | zenjpeg | 18,639 | 1024 px Lanczos3 | **no expert overrides** — true encoder-default reference |
| `s3://zentrain/sweep-v15rc-2026-05-07/` | zenjpeg | ~514K | 1024 px Lanczos3 | **chroma deep-dive**: chroma_distance_scale × 10 ∈ [0.4..2.0] × subsampling × 3 |

**Total new cells with feature parquets:** ~2.3M (image, distorted, 300-feat,
4-target) tuples. Targets in TSV: `score_zensim`, `score_ssim2[_gpu]`,
`score_butteraugli_max[_gpu]`, `score_butteraugli_pnorm3[_gpu]`.

**Source corpus**: 1024 px (max-dim) Lanczos3 re-encodes of 981 OpenAI-tagged
images from `/mnt/v/output/corpus-builder/curated_manifest_2026-04-16.tsv`
spanning 5 content classes:
- `illustration_or_logo` (150)
- `illustration_or_screen` (300)
- `photo_natural_or_detailed` (200)
- `photo_or_illustration` (250)
- `photo_wide_gamut` (83)

Resized stage at `/tmp/v15-prep/stage_1024/` (981 PNGs, 348 MB total — but
this lives on local /tmp and may be wiped; canonical mirror at
`s3://zentrain/sweep-v15r-2026-05-06/sources/`). **2 source images
permanently fail decode-back** (`png-8__web_kickstarter_com_*` — broken
PNG headers); ignore them in joins.

**zenanalyze named features** for the same corpus:
`/tmp/v15r-prep/features_v15r_combined.tsv` — 981 rows × 33 `feat_<name>_*`
columns + `content_class`. Source: re-extracted via `extract_features_for_picker`
example in zenanalyze. **Mirror this file to a durable location** before
relying on it for retraining (see TODO §4.1).

---

## 2. Gaps blocking zensim v_next

Six gaps, ordered by impact:

### 2.1 Codec coverage is JPEG-heavy (highest priority)

v15r/v15rc dominate the 2.3M-cell pool and they're zenjpeg-only. Block-
boundary artefacts (WebP), transform-edge ringing (AVIF), gaborish patterns
(JXL) are **structurally different distortions** the current model has not
been recently exposed to at scale. v12 has cross-codec data but only ~12K
cells with sparse zensim metric coverage.

**Acceptance:** ~500K cells per codec on the same 1024 px corpus, matched
zensim band coverage.

### 2.2 Scale coverage is bimodal, not continuous

We have 1024 px (~2.3M cells) and 16 MP archival (~50K cells). **Nothing
in 256–768 px**, which is exactly where post-srcset web traffic sits.
zensim's DCT-band features are scale-sensitive (different image scales
have different per-band energy distributions); a model trained only at
1024 px will mis-rank distortions at 256 px because the relative band
weights shift.

**Acceptance:** sweep at sizes [128, 256, 384, 512, 768, 1024, 1536, 2048]
on a representative subset (~20 cluster-centroid images per content class
× 8 sizes × full q × few knob configs ≈ 100K cells). Validate scale
invariance: same `(source, distortion-type, target_zensim)` should
produce ±2 zensim points across scales.

See also [`docs/scale-invariance.md`](./scale-invariance.md) for prior
experiments on this axis.

### 2.3 Content class is OpenAI's 5-tag set — narrow

Missing entirely from training:
- **anime / manga** (line-heavy, flat-color regions, severe ringing
  sensitivity)
- **pixel art** (every pixel matters; bilinear upsampling lies)
- **satellite / aerial** (texture-heavy, no semantic salience map)
- **medical** (DICOM-style 16-bit grayscale; rare in web but high-stakes
  in some markets — flag as **out of scope** unless a customer asks)
- **charts / graphs** (sparse high-contrast text + lines)
- **document scans** (text-on-paper, high-frequency content)
- **HDR / wide-gamut** (only 83 cells in v15r-corpus; need ~5×)

**Acceptance:** corpus-builder gets 200+ images per missing class, then
re-sweep to give ~50K cells per class.

### 2.4 No human MOS anchor (architectural)

All ground truth in our 2.3M cells is **butteraugli** (or zensim itself,
self-referential). Using butteraugli as supervisor means v_next zensim
asymptotes to "predict butteraugli", **not** "predict human perception".

The current `NEXT_TIER_DATA_PLAN.md §6` already prescribes CID22/KADID-10k/
TID2013 as MOS validation sets. They're held out of training so we can
measure how well a butteraugli-supervised zensim transfers to human
judgments. **This handoff doesn't change that plan** — but flag it for the
v_next training to **report transfer SROCC explicitly**, because the
interesting failure mode is "v_next gets better on butteraugli but worse
on CID22 MOS."

### 2.5 No mined adversarial examples

The most informative training samples are pairs where current zensim
disagrees with butteraugli by >5 points. These are the model's failure
modes. Our 2.3M-cell pool contains them but we haven't identified them.

**Acceptance:** a script (TODO §4.4) that joins the parquet sidecars to
the TSV per-sweep, computes `|zensim - 100*(1 - butteraugli_max/threshold)|`
or similar normalized residual, and emits the top 5K disagreement pairs
as a separate parquet for over-sampling in v_next training.

### 2.6 No "ordering-failure" pairs

For ranking-aware training (Plackett-Luce / pairwise hinge), we need
pairs where zensim ranks `(distortion_A, distortion_B)` opposite to
butteraugli's ranking. Not in the corpus today.

**Acceptance:** per (source, q-band) compute pairwise rank-disagreements
between zensim and butteraugli on the 96/72/24 in-band knob variants.
Emit the rank-flipped pairs as a sidecar.

---

## 3. Synthetic / programmatic data we already have

| Source | Path | Use |
|---|---|---|
| Synthetic-tile training corpus | refer to `NEXT_TIER_DATA_PLAN.md §3` for the codec-corpus generation | base v_next pretraining |
| Sweep-v15r baseline | `s3://zentrain/sweep-v15r-2026-05-06/zenjpeg/baseline.tsv` (18,639 rows) | encoder-default reference for "what zenjpeg does without overrides" |
| Sweep-v15rc chroma curve | 514K cells across `(class, subsampling, chroma_scale, q)` | **rich source for studying how chroma quantization affects zensim** — every cell has the 300-feature vector + butteraugli reference |
| corpus-builder OpenAI tags | `/mnt/v/output/corpus-builder/curated_manifest_2026-04-16.tsv` (981 entries, 5 classes) | content-class labels for stratified training/eval |

**To bolster** (highest leverage, in priority order):

1. **Codec-cross sweep** on the same 1024 px corpus, matching cell count
   per codec — see TODO §4.2. Gives 4× our distortion diversity.
2. **Multi-scale subset sweep** — TODO §4.3. ~100K cells, validates scale
   invariance.
3. **Adversarial mining** — TODO §4.4. Pure analysis on existing data,
   no new sweep needed.
4. **Content-class expansion** — TODO §4.5. Requires corpus-builder runs,
   slower path.

---

## 4. TODOs (concrete, ordered)

Each item lists: prerequisite, path to artefact, acceptance criterion.

### 4.1 Mirror v15r/v15rc/v15rc-baseline + features TSV to durable storage

**Prereq:** none.
**Action:**

```bash
# Sources (already on R2):
#   s3://zentrain/sweep-v15r-2026-05-06/sources/  (981 PNGs)
# Mirror to /mnt/v (durable Windows volume):
mkdir -p /mnt/v/zen/zensim-training/v15r-2026-05-07
aws s3 sync s3://zentrain/sweep-v15r-2026-05-06/  /mnt/v/zen/zensim-training/v15r-2026-05-07/v15r/   --endpoint-url=https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
aws s3 sync s3://zentrain/sweep-v15rc-2026-05-07/ /mnt/v/zen/zensim-training/v15r-2026-05-07/v15rc/  --endpoint-url=https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
# Copy zenanalyze features TSV (currently only in /tmp):
cp /tmp/v15r-prep/features_v15r_combined.tsv /mnt/v/zen/zensim-training/v15r-2026-05-07/features_v15r_combined.tsv
```

**Acceptance:** `/mnt/v/zen/zensim-training/v15r-2026-05-07/` contains all
TSVs, parquets, and the features TSV. /tmp paths can be safely wiped.

### 4.2 Cross-codec sweep on 1024 px corpus (~500K cells per codec)

**Prereq:** the rebuilt rayon binary at
`s3://coefficient/binaries/zen-metrics-0.6.8-linux-x86_64-gpu` — verified
running 53× faster than the serial 0.6.5 binary. Code at
`/home/lilith/work/zen/zenmetrics/crates/zen-metrics-cli/src/sweep/run.rs`.

**Action:** for each of `zenwebp`, `zenavif`, `zenjxl`, run a sweep mirroring
v15r's grid shape (full knob coverage at default-adjacent values) on the
same 981-image 1024 px corpus.

Reference launcher: `/home/lilith/work/zen/zenmetrics/scripts/sweep/v15/launch_gpu.sh`
(adapt: change `--codec`, set new `SWEEP_RUN_ID`, copy sources to
`s3://zentrain/<run_id>/sources/`).

Expected wall-clock per codec on 50 vast.ai workers: 30–60 min (rayon
binary makes this fast).

**Acceptance:** `s3://zentrain/sweep-v16w-…/`, `…v16a-…/`, `…v16j-…/`
each with ~500K cells. Codec-cross training-validation table in
benchmark/ writeup.

### 4.3 Multi-scale subset sweep

**Prereq:** §4.2 done; cross-codec data lets us test scale invariance
across codecs not just JPEG.

**Action:** pick 20 centroid images per content class (k-means on
zenanalyze features) → 100 sources × Lanczos3 resize to
[128, 256, 384, 512, 768, 1024, 1536, 2048] → sweep at
`q ∈ {30, 50, 70, 85, 95}` × default knobs only. ~100 × 8 × 5 = 4 000
cells per codec; 4 codecs = 16 000 cells. Tiny.

**Acceptance:** for each `(source_image, codec, target_zensim_band)`,
zensim score across 8 scales should agree to within ±2 points after
controlling for scale-dependent feature normalization.
[`docs/scale-invariance.md`](./scale-invariance.md) has the prior method.

### 4.4 Mine adversarial pairs from existing 2.3M cells

**Prereq:** §4.1 done (data on durable storage).

**Action:** join feature-parquets to TSVs per sweep; compute residual

```
delta_zensim_butteraugli = zensim_score - 100 * (1 - butteraugli_max / 6.0)
```

(or whatever zensim→butteraugli mapping the current model implements).
Emit:
- Top 5 000 high-positive-residual pairs (zensim says good, butteraugli
  says bad)
- Top 5 000 high-negative-residual pairs (zensim says bad, butteraugli
  says good)

Save to `s3://zentrain/v_next-training/adversarial_pairs.parquet` keyed
by `(image_path, codec, q, knob_tuple_json, residual)`.

**Acceptance:** parquet exists; sample 50 pairs and visually inspect
some — many should be content-class outliers (anime, charts, lineart)
that would confirm the OpenAI-5-class corpus is too narrow.

### 4.5 Content-class corpus expansion

**Prereq:** corpus-builder pipeline understanding.

**Action:** run corpus-builder against:
- Anime/manga sources (Danbooru / Pixiv / wallhaven anime tags — get
  user authorization before scraping)
- Charts (Statista publicly downloadable charts; matplotlib gallery
  rendered to PNG; Wikipedia chart figures via Commons API)
- Pixel art (Lospec gallery, OpenGameArt 16/32-px canonical)
- Documents (Common-Crawl PDF first-page renders @ 1024 px)

Aim for 200 sources per missing class.

**Acceptance:** new `curated_manifest_<date>.tsv` adds 600+ entries
across 3+ new classes with `suspected_category` populated.

### 4.6 Add ssim2 & butteraugli targets to zensim training pipeline

**Prereq:** none — purely a training-script change.

**Action:** the existing zensim training (V0_5/V0_6 paths in
`NEXT_TIER_DATA_PLAN.md`) takes a single target (SSIMULACRA2 proxy or
butteraugli). v_next should multi-task:

```
loss = α · MSE(zensim_pred, butteraugli_target)
     + β · MSE(zensim_pred, ssim2_target)
     + γ · pairwise_rank_loss(zensim_pred, butteraugli_target)
```

with α=1, β=0.3, γ=0.5 as reasonable starting weights. The TSV has both
butteraugli and ssim2 columns. Multi-task supervision yields a more
robust feature mapping.

**Acceptance:** v_next training script consumes both targets;
holdout SROCC on CID22/KADID/TID measured against both ssim2 *and*
butteraugli; SROCC vs. each target should be ≥ V0_6's single-target
numbers.

### 4.7 Multi-scale feature normalization

**Prereq:** §4.3 done (scale-invariance validation).

**Action:** zensim's 300 features include DCT-band energies that scale
with image dimensions. v_next should normalize features by `log(pixels)`
before the MLP, OR include `log_pixels` as a feature input. Already
tracked at `NEXT_TIER_DATA_PLAN.md §5` — promote to a v_next must-have.

**Acceptance:** v_next ZNPK has explicit scale-input or pre-normalization
documented in the binary header / README.

### 4.8 Bake v_next + run held-out CID22/KADID/TID validation

**Prereq:** all training TODOs done.

**Action:** train v_next as ZNPK file, run
`/home/lilith/work/zen/zensim/zensim-validate/` against:
- CID22 (held out 49 of 215 images per `NEXT_TIER_DATA_PLAN.md §6`)
- KADID-10k full
- TID2013 full

Report SROCC + KROCC + Pearson per validation set vs. published V0_2
through V0_6 numbers.

**Acceptance:** v_next holds ≥ V0_6 on all three validation sets.

### 4.9 Update `NEXT_TIER_DATA_PLAN.md` + `CHANGELOG.md`

After each TODO completes, update both docs with measured numbers and
data paths. The plan doc should be living; this handoff is a snapshot.

---

## 5. Rebuilt zen-metrics binary (for any new sweep)

The `0.6.8` binary added inner-loop rayon parallelism (53× local speedup
vs. serial) and `catch_unwind` per-cell isolation. Code path:
`/home/lilith/work/zen/zenmetrics/crates/zen-metrics-cli/src/sweep/run.rs`.
Build:

```bash
cd /home/lilith/work/zen/zenmetrics
cargo build --release -p zen-metrics-cli --features sweep,gpu,gpu-cuda
# 76 MB binary — upload as new version:
aws s3 cp target/release/zen-metrics \
    s3://coefficient/binaries/zen-metrics-<version>-linux-x86_64-gpu \
    --endpoint-url=https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
```

Worker scripts at
`/home/lilith/work/zen/zenmetrics/scripts/sweep/onstart_v3.sh` (now with
read-back-verified atomic claim) and
`/home/lilith/work/zen/zenmetrics/scripts/sweep/sweep_diag.py` (per-worker
waste analyzer) and
`/home/lilith/work/zen/zenmetrics/scripts/sweep/sweep_janitor.py` (auto-
reaper for slow workers). All committed on a detached HEAD as of
2026-05-07; **rebase onto a real branch before merging**.

---

## 6. Quick start for the fresh-context successor

1. Read `docs/NEXT_TIER_DATA_PLAN.md` (read-only research baseline,
   2026-05-04).
2. Read this doc (handoff, 2026-05-07) for what's new.
3. Run TODO §4.1 first — mirror the perishable /tmp data to /mnt/v.
4. Pick the first TODO that fits the available compute budget:
   - 1 hr local: TODO §4.4 (adversarial mining) — pure analysis.
   - 2 hr vast.ai: TODO §4.2 (one cross-codec sweep).
   - 1 day: TODOs §4.2 + §4.6 + §4.8 chain — full v_next iteration.
5. Update `CHANGELOG.md` and this doc as work lands.

---

## 7. Things explicitly out of scope for v_next

- HDR (PQ/HLG) — separate effort, different metric design
- Video — zensim is a still-image metric; no temporal extension here
- Custom MOS collection (Squintly app) — covered separately in `NEXT_TIER_DATA_PLAN.md §7`
- Mobile-tier optimization (i8 quant, fp16 weights) — that's an inference
  performance task, orthogonal to "what does zensim need to learn better"
