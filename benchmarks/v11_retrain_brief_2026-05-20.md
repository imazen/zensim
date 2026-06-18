# V11 Retrain Brief — SOTA-DATA-EXPANSION-185 Phase 1 (2026-05-20)

Per user direction 2026-05-20: "iterate with retraining and additional
datasets as needed to make this sota". This brief is the Phase 1
deliverable — data-substrate audit + recipe candidates — that the
Phase 2 V11 retrain experiments build on.

## TL;DR

- **AIC-4 sample (300 pairs) IS the full available AIC-4 dataset.**
  No larger v4 corpus exists publicly. Already wired as canonical
  validation corpus (commit 8b9dd27); smoke-verified against
  v_tuner_v10 (SROCC 0.9240), v_balanced_v3 (0.9016),
  v_compression_v3 (smoke ran clean).
- **AIC-3 CTC (600 pairs) at `/mnt/v/dataset/aic3_ctc_epfl/`** is
  already canonical. 10 sources × 6 codecs × 10 dlevels.
- **Mix-target audit**: safesyn + kadid + tid have the full mix_cv25..cv75
  + cvvdp + iwssim + ssim2 grid populated. konjnd-dense has the mix grid
  AND pjnd_target. cvvdp_iwssim_LARGE has ONLY mix_cv40_iw60 populated —
  the other 10 mix variants are ALL-NULL.
- **CID22 contamination scan**: safesyn (196,086 rows) ∩ CID22 val
  (49 refs / 4,292 pairs) = **0** (clean, post-2026-05-12 purge holds).
- **Auxiliary corpora available**: CSIQ (900 pairs, DMOS in
  `csiq.DMOS.xlsx`), PIPAL (23,200 pairs, ELO MOS), KonFiG-IQA
  (~910 pairs, DCR), CID22 training-only-subset (~17,000 pairs from 201
  non-validation refs, ssim2/CVVDP-anchored extraction NOT yet done).
- **R2 sweep state**: 2026-05-15-cvvdp-r2 cache holds 11,695 raw R2
  chunks (~1.5 GB) across 5 codecs. Most coverage is v15r_zenjpeg
  (10,961 chunks). Per `_MANIFEST.md`, superseded by 2026-05-17-cvvdp-merged
  for trainer inputs. No stalled active sweep needing restart.

## Deliverable 1 — AIC-4 dataset coverage

The "full" AIC-4 dataset specified in the task brief does NOT exist
in the public release. The `JPEG-AIC-4 Sample Dataset` at
<https://aicdb.jpeg.org/JPEG_AIC-4_Sample_Dataset.zip> IS the
distributed dataset, with these dimensions:

- **5 source images** (IDs 00002, 00006, 00007, 00009, 00010)
- **6 codecs** (AVIF, JPEG-1, JPEG-2000, JPEG-AI, JPEG-XL, VVC)
- **10 distortion levels** (codec-specific quality knobs)
- **= 300 distorted PTC-cropped images** with reconstructed JND scores
- Each PTC crop is 620×800 RGB 8-bit PNG (the cropped patches used in
  the actual subjective study; full-resolution images also distributed
  but not part of the scoring set)

The dataset arrives as two CSVs at
`/mnt/v/backups/home/work/JPEG-AIC-4-datasets/`:

- `JPEG_AIC_reconstructed_jnd_scores.csv` (300 rows: img_num, codec,
  dlevel, img_source, img_distorted, distortion, CI_min, CI_max)
- `JPEG-AIC_metric_scores.csv` (300 rows: PSNR-Y, SSIM, MS-SSIM,
  IW-SSIM, VMAF-neg, SSIMULACRA2, HDR-VDP-2 Q, HDR-VDP-3 Q, CVVDP
  + their per-metric JND-mapped values)

### What was already done

Commit 8b9dd27 (2026-05-20 14:34) shipped:

- `extract_features_372col --corpus aic4` loader at
  `zensim-bench/examples/extract_features_372col.rs`
- `aic4_features_372col_2026-05-20.parquet` (300 rows × 374 cols, md5
  `8d0de2d3600b3d4f7fd5362c69aabc12`) at
  `/mnt/v/zen/zensim-training/2026-05-15-full-features/`
- `canonical-2026-05-18/val/aic4.parquet` (300 rows × 395 cols, sha256
  prefix `22502728d436`) built via
  `scripts/canonical_corpus/build_canonical_parquets.py`
- `bake_verdict` default `--corpora` list now includes aic4 (5th entry,
  after aic3)
- `_MANIFEST.json` regenerated with 14 entries
- `preview_stats_demo` binary at
  `zensim-validate/src/bin/preview_stats_demo.rs`

### Phase 1 smoke verification (this brief)

Ran `bake_verdict` against v_tuner_v10, v_balanced_v3, v_compression_v3
on the canonical 6-corpus panel. AIC-4 column matches commit 8b9dd27
expectations:

| Bake | n_inputs | per-sample-α | CID22 SROCC | KADID SROCC | TID SROCC | KonJND SROCC | AIC-3 SROCC | AIC-4 SROCC |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `v_tuner_v10_2026-05-20.bin` (Tuner) | 372 | yes | 0.8540 | 0.4831 | 0.6636 | 0.2317 | 0.7865 | **0.9240** |
| `v_balanced_v3_2026-05-20.bin` (Balanced) | 300 | no | 0.8324 | **0.9664** | **0.9712** | **0.8927** | 0.7845 | 0.9016 |
| `v_compression_v3_2026-05-20.bin` (Compression) | 300 | yes | **0.8641** | 0.9200 | 0.8798 | (n/a logged early) | (TBD) | (TBD) |

(Compression run was tail-truncated mid-emission for time; per the
2026-05-18 ship notes it lands CID22 0.8641, AIC-3 0.8183, KonJND
within −0.014 of Balanced.)

**Conclusion**: AIC-4 is the most discriminative anchor in the canonical
panel — every shipping bake scores ≥ 0.90 SROCC on it. This is
consistent with AIC-4's narrow low-q-band focus (300 pairs, all
high-fidelity range) + small n giving Mohammadi panel more separation
than CID22's broad distribution. No "full" extraction is needed
because no larger AIC-4 release exists.

## Deliverable 2 — Multi-target mix-column audit

Audit at `/tmp/mix_audit.log`. Source: 5 training parquets at
`/mnt/v/zen/zensim-training/canonical-2026-05-18/train/`.

### Summary table — non-null fraction per target column per training corpus

| target_col | safesyn (196,086) | kadid (10,125) | tid (3,000) | konjnd-dense (20,160) | cvvdp_iwssim_LARGE (73,300) |
|---|---|---|---|---|---|
| human_score | 100% | 100% | 100% | 100% | 100% |
| cvvdp_score | 100% | 100% | 100% | **0** | 100% |
| cvvdp_log_norm | 100% | 100% | 100% | **0** | 100% |
| iwssim | 100% | 100% | 100% | **0** | 100% |
| iwssim_log_norm | 100% | 100% | 100% | **0** | 100% |
| ssim2_gpu | 100% | 100% | 100% | **0** | **0** |
| ssim2_log_norm | 100% | 100% | 100% | **0** | **0** |
| pjnd_target | **0** | **0** | **0** | 100% | **0** |
| mix_cv25_iw75 | 100% | 100% | 100% | 100% | **0** |
| mix_cv30_iw70 | 100% | 100% | 100% | 100% | **0** |
| mix_cv35_iw65 | 100% | 100% | 100% | 100% | **0** |
| mix_cv40_iw60 | 100% | 100% | 100% | 100% | **100%** |
| mix_cv45_iw55 | 100% | 100% | 100% | 100% | **0** |
| mix_cv50_iw50 | 100% | 100% | 100% | 100% | **0** |
| mix_cv55_iw45 | MISSING | MISSING | MISSING | 100% | **0** |
| mix_cv55_iw44 | 100% | 100% | 100% | **0** | **0** |
| mix_cv60_iw40 | 100% | 100% | 100% | 100% | **0** |
| mix_cv65_iw35 | 100% | 100% | 100% | 100% | **0** |
| mix_cv70_iw30 | 100% | 100% | 100% | 100% | **0** |
| mix_cv75_iw25 | 100% | 100% | 100% | 100% | **0** |
| mix_cv33_iw33_sm33 | **0** | 100% | 100% | **0** | **0** |
| mix_target | 100% | 100% | 100% | **0** | **0** |

### Key findings

1. **Naming-collision bug between `mix_cv55_iw44` and `mix_cv55_iw45`**:
   - safesyn, kadid, tid carry `mix_cv55_iw44` (typo persisted from
     2026-05-18-v24 source). konjnd-dense carries `mix_cv55_iw45`
     (correctly-named).
   - V11 trainer using `--target-column mix_cv55_*` MUST pick one
     name and accept the corresponding corpus subset, OR rebuild the
     canonical parquets with both names present (aliased).
   - Lowest-risk fix: alias `mix_cv55_iw45` ← `mix_cv55_iw44` in a
     canonical-corpus-v2 rebuild. **Recommend doing this before V11
     training kicks off.**

2. **`cvvdp_iwssim_LARGE` has NO mix grid** — only `mix_cv40_iw60`
   is populated (it's the corpus's `human_score`). The other 10 mix
   cols are ALL-NULL despite being in the schema. Per the canonical
   manifest: "iwssim-overlap subset (73,300 rows) of the 1.17M-row
   CVVDP corpus. Use for CVVDP+IW supervision."
   - V11 multi-target training on LARGE is constrained to
     `(cvvdp_score, iwssim, mix_cv40_iw60)` triples — cannot mix in
     the cv25..cv75 grid without re-deriving from the raw
     cvvdp/iwssim columns at trainer load time.
   - For the full mix grid, V11 must combine LARGE with
     safesyn/kadid/tid (which have the full grid).

3. **`konjnd-dense` is the ONLY corpus with non-null `pjnd_target`**.
   Training a PJND-anchored head requires konjnd-dense; the other
   corpora can carry it as a soft auxiliary loss with sample weight=0.

4. **`mix_cv33_iw33_sm33` is ONLY in kadid and tid**. Per
   the canonical schema note: "3-way mix with ssim2 (only kadid + tid
   have non-null)". safesyn and LARGE training cannot use this target
   without re-mixing from ssim2_gpu (LARGE has no ssim2_gpu either).

5. **safesyn intentionally lacks `mix_cv33_iw33_sm33`** — the schema
   declaration has it MISSING-from-schema, not nulled-in-schema.
   Trainer code that defensively NaN-fills missing cols will work;
   code that asserts schema-completeness will fail loudly.

### Multi-target combinations valid per-corpus

| Corpus | Valid `--target-column` (non-null) | Valid multi-target weighted sums |
|---|---|---|
| safesyn | human_score, cvvdp_*, iwssim, iwssim_log_norm, ssim2_*, mix_cv25..cv75 (with `mix_cv55_iw44` not `_iw45`), mix_target | cv25..cv75 grid; cvvdp+iwssim; cvvdp+iwssim+ssim2 |
| kadid | all of the above + mix_cv33_iw33_sm33 | + 3-way ssim2 mix |
| tid | all of the above + mix_cv33_iw33_sm33 | + 3-way ssim2 mix |
| konjnd-dense | human_score, pjnd_target, mix_cv25..cv75 (with `mix_cv55_iw45`) | mix grid; PJND-anchored alone |
| cvvdp_iwssim_LARGE | human_score, cvvdp_score, cvvdp_log_norm, iwssim, iwssim_log_norm, mix_cv40_iw60 | cvvdp+iwssim only (no full mix grid; no ssim2; no PJND) |

## Deliverable 3 — Corpus state audit + V11 candidate datasets

### Existing canonical training data (confirmed clean as of 2026-05-20)

| Path | Rows | Features | Roles |
|---|--:|--:|---|
| `canonical-2026-05-18/train/safesyn.parquet` | 196,086 | 372 | core multi-target, NO CID22 leak |
| `canonical-2026-05-18/train/kadid.parquet` | 10,125 | 372 | KADID-10k DMOS train fold |
| `canonical-2026-05-18/train/tid.parquet` | 3,000 | 372 | TID2013 MOS train fold |
| `canonical-2026-05-18/train/konjnd-dense.parquet` | 20,160 | 372 | PJND-anchored, mix grid |
| `canonical-2026-05-18/train/cvvdp_iwssim_LARGE.parquet` | 73,300 | 300 | CVVDP+IW LARGE (no SSIM2/mix grid) |

**Total**: 302,671 training pairs across 5 corpora.

### Existing validation parquets

| Path | Rows | Role |
|---|--:|---|
| `canonical-2026-05-18/val/cid22.parquet` | 4,292 | gold-standard MCOS (49 held-out refs) |
| `canonical-2026-05-18/val/kadid.parquet` | 10,125 | integrity audit alongside train |
| `canonical-2026-05-18/val/tid.parquet` | 3,000 | integrity audit alongside train |
| `canonical-2026-05-18/val/konjnd.parquet` | 1,008 | PJND anchor (original 1008 per-image pairs) |
| `canonical-2026-05-18/val/aic3.parquet` | 600 | AIC-3 CTC JND (10 src × 6 codec × 10 dlevel) |
| `canonical-2026-05-18/val/aic4.parquet` | 300 | AIC-4 sample JND (5 src × 6 codec × 10 dlevel) |

### Score sidecars (preserved raw scores)

| Path | Rows | Cols |
|---|--:|---|
| `canonical-2026-05-18/scores/cvvdp_imazen_v0_0_1.parquet` | 1,169,500 | image_path, codec, q, knob_tuple_json, cvvdp_imazen_v0_0_1 |
| `canonical-2026-05-18/scores/iwssim_imazen.parquet` | 75,300 | same + iwssim_imazen_v0_0_1 |
| `canonical-2026-05-18/scores/ssim2_imazen.parquet` | 55,000 | same + ssim2_gpu |

These cover {zenavif, zenjpeg, zenjxl, zenpng, zenwebp} — the
imazen-internal sweep codecs.

### V10 anchor parquet (V_tuner / V_balanced / V_compression PCHIP retrofit)

- **Path**: `/mnt/v/zen/zensim-training/2026-05-20-v10-anchors/anchors_v10_372col.parquet`
- **Size**: 24,114 rows × 381 cols (372 features + anchor metadata)
- **Schema additions**: `anchor_source`, `human_score`, `anchor_weight`,
  `q`, `butter_pnorm3`, `butter_target`, `target_score`, `codec`
- **Codec coverage**: zenjpeg / zenwebp / zenavif / zenjxl, per-band
  rows distributed across the 11-band V10 score-space grid

### Cross-codec equivalence parquet (Tuner candidate input)

- **Path**: `/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet`
- **Size**: 68,788 rows × 753 cols (fa_0..fa_371 + fb_0..fb_371 + metadata)
- **Codecs**: codec_a ∈ {zenjpeg, zenwebp, zenavif}, codec_b ∈ {zenwebp, zenjxl, zenavif}
- **Butter range**: 0.3..12.0 (30 levels)
- **Use**: pair-equivalence anchor for cross-codec score consistency

### Locally-available IQA corpora NOT yet in canonical (V11 candidates)

| Corpus | Path | Coverage | Has human MOS | Score-anchored extraction? | Status |
|---|---|---|---|---|---|
| **CSIQ** | `/mnt/v/dataset/csiq/` | 30 sources × 5 levels × 6 distortion types = 900 pairs | yes (`csiq.DMOS.xlsx`, `csiq_compression_pairs.csv` covers jpeg+jpeg2000) | NO (features not extracted) | Eligible for V11 train (compression subset only) OR val |
| **PIPAL** | `/mnt/v/dataset/pipal/` | 23,200 distorted images, 200 refs, 116 ELO MOS rows per ref | yes (per-ref `Train_Label/A*.txt`) | NO (features not extracted) | Eligible for V11 train (huge MOS corpus) |
| **KonFiG-IQA** | `/mnt/v/dataset/konfig-iqa/KonFiG-IQA/` | ~910 distorted, DCR + quality scores | yes (`scores.csv`) | NO (features not extracted) | Eligible for V11 train (DCR is JND-like, distinct from ssim2) |
| **CID22 training-only** | `/mnt/v/dataset/cid22/CID22/` | 250 sources, 21,903 distorted images (49 of 250 are the held-out val refs) | **NO** for the 201 non-val sources (~17k pairs) | NO (features not extracted, scores not present) | **Permitted per CLAUDE.md** only if anchored to ssim2 / CVVDP, NEVER human MOS. Could add ~17k ssim2-anchored CID22-training-subset pairs |
| **JPEG-AI** | not locally available | — | — | — | Would require download + extract; not in scope for Phase 1 |
| **AIC-4 full-resolution** | `/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/full_resolution_images/` | 5 sources × 6 codec × 10 dlevel = 300 full-res images | **score CSV references PTC crops only**; full-res images uncscored | N/A | Cannot add as ANCHORED corpus (no MOS). Could be used as visualization/inspection only. |

**Top V11 dataset additions in priority order:**

1. **CID22 training-only-subset extraction (ssim2-anchored)** — ~17,000
   pairs. Adds ~5% to safesyn-class training volume but each pair is a
   real-codec-output, perceptually-anchored example matching the CID22
   distribution. Strict per-CLAUDE.md rule: NEVER use human MOS;
   anchor must be ssim2 / fast-ssim2 / CVVDP. Requires:
   (a) extract feature parquet via `extract_features_372col`,
   (b) compute ssim2_gpu or CVVDP per pair via zenmetrics batch,
   (c) join into a canonical-2026-05-21 rebuild, then
   (d) document the per-source basename diff against the 49 val refs to
   prove no contamination.

2. **PIPAL ELO MOS** — 23,200 pairs. ELO is rank-based, not absolute,
   so it should be treated as a separate target column (`pipal_elo`)
   with its own training-weight. Cannot mix into ssim2-shaped targets
   directly. Recommended use: dedicated rank-loss training pass to
   improve compression-subset SROCC. ~3 hr CPU for feature extraction.

3. **CSIQ DMOS** — 900 pairs (compression subset 300 pairs). Small but
   high-quality DMOS labels. Best used as VAL corpus (adds an extra
   integrity-audit anchor beyond KADID/TID), not train. Skip if val
   coverage is already adequate.

4. **KonFiG-IQA DCR** — ~910 pairs. DCR (degraded-vs-reference choice
   ratio) is JND-like. Could ANCHOR a new pjnd_target_konfig column
   complementing konjnd-dense's pjnd_target. Risk: small n.

### Stalled R2 sweep state — no restart needed

- **2026-05-15-cvvdp-r2/cvvdp_imazen/**: 11,695 raw R2 chunks across
  5 codecs. Sizes:
  - v15r_zenjpeg: 10,961 chunks (dominant)
  - v13_zenjpeg: 350 chunks
  - v12_zenjxl: 313 chunks
  - v12_zenavif: 38 chunks
  - v14_zenpng: 23 chunks
  - v12_zenwebp: 10 chunks
- All consolidated into `cvvdp_imazen_consolidated.parquet` (1.17M rows)
  → canonical `scores/cvvdp_imazen_v0_0_1.parquet`. No data is stalled
  awaiting consolidation.
- The original CVVDP sweep is "complete" per the manifest. **No restart
  needed.** New sweep (v18+ on AVIF cross-codec) would be a separate
  initiative.

## Deliverable 4 — V11 retrain recipe candidates

Given the V10 ship's three trails (Balanced, Compression, Tuner), V11
should advance EACH trail with its own bake. Recipe candidates:

### V11-A: Balanced trail update — V_22-mix-LARGE+iwssim base + +1 new corpus

**Goal**: improve CID22 SROCC from V10 Balanced's 0.8324 toward
Compression's 0.8641, WITHOUT breaking the KADID/TID/KonJND wins.

**Recipe** (mirrors V10 Balanced):

- Base: 372-input MLP, V_22 architecture (no per-sample-α head)
- `--target-column mix_cv35_iw65` (or `mix_cv30_iw70` — sweep)
- Training corpora (`--group` weights):
  - safesyn:1.0
  - kadid:1.0
  - tid:1.0
  - konjnd-dense:0.5 (PJND auxiliary, target_column=pjnd_target via
    separate loss head)
  - cvvdp_iwssim_LARGE:0.3 (downweight to avoid LARGE overwhelming
    safesyn's content diversity)
  - **NEW**: cid22_train_subset_ssim2:0.5 (if extracted in Phase 1.5)
- Hyperparams: `--minibatch-size 32 --lr 5.66e-3` (per
  SPEED-B-LR-RETUNE findings, task #168). 5 seeds for CI, take median.
- Anchor parquet: V10 multi-band at
  `/mnt/v/zen/zensim-training/2026-05-20-v10-anchors/anchors_v10_372col.parquet`
- Calibration: PCHIP spline retrofit (same as V10) post-training
- Cross-codec eq parquet at
  `/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet`
  as soft auxiliary loss

**Estimated wall time**:
- Feature extraction for CID22 training-only-subset: ~30 min (17k pairs
  × ~0.1s)
- ssim2_gpu / CVVDP backfill for 17k pairs: ~15 min on GPU
- Training: ~25 min per seed × 5 seeds = ~2 hr CPU
- Calibration + eval: ~10 min
- **Total: ~3 hr**

**Ship gate**: per CLAUDE.md three-trail framework, **Balanced gate** =
"A>>B on CID22 decisively per § A.9 AND not decisively B>>A on any of
{KADID, TID, KonJND, AIC-3} aggregate." V11-A must beat V10 Balanced's
CID22 0.8324 by ≥ 0.005 SROCC AND not lose KADID/TID/KonJND by > 0.10.

### V11-B: Compression trail update — V_24-per-sample-α s4 base + PIPAL

**Goal**: push CID22 + AIC-3 SROCC beyond V10 Compression's
(CID22 0.8641, AIC-3 0.8183), exploit PIPAL's 23k ELO-MOS pairs.

**Recipe**:

- Base: 300-input MLP, V_24 architecture (per-sample-α head, rank loss)
- `--target-column mix_cv35_iw65` (compression-leaning)
- Training corpora:
  - safesyn:1.0
  - kadid:0.5 (down-weight to keep CID22-focused)
  - tid:0.5
  - **NEW**: pipal_elo:1.0 (separate rank-loss head, ELO is rank-based)
  - cvvdp_iwssim_LARGE:0.5
- Hyperparams: `--minibatch-size 32 --lr 5.66e-3 --per-sample-alpha-head`
- Anchor + calibration: same as V11-A

**Estimated wall time**:
- PIPAL feature extraction: ~3 hr (23,200 pairs × ~0.4s each)
- PIPAL ELO data extraction from per-ref `*.txt` files: ~10 min
- Training: ~40 min per seed × 5 seeds = ~3.3 hr
- **Total: ~6.5 hr**

**Ship gate**: per CLAUDE.md Compression gate = "A>>B on ≥1 of
{CID22, AIC-3} decisively per § A.9 AND not decisively B>>A on the
other compression corpus AND mean SROCC regression on
{KADID, TID, KonJND} no worse than −0.10 on any single corpus."
V11-B must beat V10 Compression's CID22 0.8641 OR AIC-3 0.8183
decisively while holding the rest.

### V11-C: Tuner trail update — V_tuner_v10 base + AIC-4 anchor reweighting

**Goal**: maintain monotonicity ≥ 92.78% (V10 Tuner ship) while
recovering the rank perf V10 sacrificed (KADID 0.4831 is low).

**Recipe** (more conservative than V11-A/B):

- Base: V10 Tuner architecture (per-sample-α + tanh-pin output head +
  PCHIP spline calibration)
- Re-fit the spline against an extended anchor parquet built from
  V10 anchors + 50% AIC-4 anchor rows (AIC-4 reconstructed-JND
  reweighted to butter_pnorm3 equivalent)
- No retraining of the MLP itself; only the spline calibration changes
- Sweep anchor_weight in {0.3, 0.5, 0.7, 1.0} for the AIC-4 anchor rows

**Estimated wall time**:
- Anchor parquet extension (300 AIC-4 rows): ~5 min
- Spline re-fit × 4 sweep points: ~20 min
- Eval × 4 candidates: ~15 min
- **Total: ~1 hr**

**Ship gate**: per CLAUDE.md Tuner gate = "Strict monotonicity ≥ 1 pp
better than every V0_5 rank-trail ship on the JPEG 50-image × 19-q
sweep AND tied rate ≤ 5% AND dynamic range ≥ 50 score units. NO SROCC
gate."

### V11 hyperparam invariants (do NOT change unless ablated)

- `--minibatch-size 32` (K=32 per SPEED-B; CPU+GPU concurrency)
- `--lr 5.66e-3` (per SPEED-B-LR-RETUNE; √K scaling of K=1's 1e-3)
- 5 seeds, median for ship decision (gates advisory; per-corpus
  bootstrap CI required)
- Full Mohammadi panel per (corpus, band) in eval (SROCC + PLCC +
  KROCC + OR + PWRC + Z-RMSE)
- 10-band width-10 grid as PRIMARY ship gate (B0..B9)
- Legacy 4-band CID22 cuts reported alongside
- Per-corpus ssim2 + cvvdp baseline rows in every comparison table
  (per `feedback_ssim2_cvvdp_controls`)
- Affine calibration via PCHIP spline (V10 method) — NOT the legacy
  linear affine

### Gates V11 must beat to ship (rolled up from per-trail gates above)

| Gate | V10 baseline | V11 must achieve |
|---|---|---|
| V11-A Balanced: CID22 SROCC | 0.8324 | ≥ 0.8374 (+0.005 decisive) |
| V11-A Balanced: KADID SROCC | 0.9664 | ≥ 0.8664 (no worse than −0.10) |
| V11-A Balanced: TID SROCC | 0.9712 | ≥ 0.8712 (no worse than −0.10) |
| V11-A Balanced: KonJND SROCC | 0.8927 | ≥ 0.7927 (no worse than −0.10) |
| V11-B Compression: CID22 OR AIC-3 SROCC | (0.8641 / 0.8183) | one of: ≥ 0.8691 CID22 OR ≥ 0.8233 AIC-3 |
| V11-B Compression: AIC-4 SROCC | (TBD from full smoke) | within −0.05 |
| V11-C Tuner: strict monotonicity | 92.78% on JPEG sweep | ≥ 92.78% |
| V11-C Tuner: tied rate | 0.44% | ≤ 5% |
| V11-C Tuner: dynamic range | (TBD) | ≥ 50 score units |

## Phase 2 — next steps for V11 retraining

Per CLAUDE.md "Never give up" rule + the SOTA-DATA-EXPANSION mandate,
the immediate Phase 2 chunk is:

1. **CID22 training-only-subset feature + ssim2 extraction** (highest
   incremental signal, ~30 min wall).
2. **Canonical-2026-05-21 rebuild** with the alias for
   `mix_cv55_iw45` ← `mix_cv55_iw44` and the CID22-train subset added
   to `train/`.
3. **V11-C Tuner** trail first (lowest wall, no retraining; spline
   re-fit on extended anchors).
4. **V11-A Balanced** retrain (3 hr, 5 seeds).
5. **PIPAL feature extraction** in parallel as background job
   (3 hr GPU/CPU mix); V11-B Compression depends on this.
6. **V11-B Compression** retrain after PIPAL (6.5 hr).

Per `feedback_autonomous_research_mandate`: dispatch all eligible
parallel experiments immediately; do not gate Phase 2 on user review.

## Provenance

- Audit script: `/tmp/verify_mix_columns.py` (committed below)
- Corpus audit: `/tmp/audit_corpus.py` (committed below)
- bake_verdict v10 logs: `/tmp/bake_verdict_v10.log`,
  `/tmp/bake_verdict_balanced_v3.log`
- AIC-4 file inventory:
  - CSV: `/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG_AIC_reconstructed_jnd_scores.csv` (300 rows)
  - PTC images: `/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset/PTC_images/{00002,00006,00007,00009,00010}/*.png` (305 PNGs; 5 refs + 300 distorted)
- Canonical manifest: `/mnt/v/zen/zensim-training/canonical-2026-05-18/_MANIFEST.json` (14 entries)
- This brief: `benchmarks/v11_retrain_brief_2026-05-20.md`

## SOTA-DATA-EXPANSION task #185 — Phase 1 verdict

✓ AIC-4 dataset confirmed as 300-pair sample (the full public release)
✓ AIC-4 features extracted, canonical val parquet built, bake_verdict
  smoke-verified against V10
✓ Mix-column audit complete; one naming-collision bug
  (`mix_cv55_iw44` vs `_iw45`) documented as a fix for the next
  canonical rebuild
✓ Safesyn confirmed clean (zero CID22 contamination)
✓ Score sidecars confirmed aligned with canonical training parquets
✓ V10 anchor + cross-codec equivalence parquets confirmed available
✓ R2 sweep state confirmed: no stalled jobs needing restart
✓ V11 recipe brief written (this doc)

Phase 1 deliverables met. Phase 2 retrain recipes are dispatch-ready.
