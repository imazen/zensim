# DATA INTEGRITY BUG: kadid/tid metric columns are corrupt (2026-05-25)

Surfaced by the AIC-3 CVVDP-feature spike (commit 2804c86a). The
`iwssim` and `ssim2_gpu` columns in the canonical KADID and TID train
parquets are **not real metric scores**.

## Evidence

`/mnt/v/zen/zensim-training/canonical-2026-05-21/train/{kadid,tid}.parquet`:

| corpus | column | corr(human_score) | identical | std |
|---|---|--:|--:|--:|
| kadid | iwssim | **1.0000** | **100%** | 0.2707 |
| kadid | ssim2_gpu | 0.0125 | 0% | 0.7219 |
| tid | iwssim | **1.0000** | **100%** | 0.1377 |
| tid | ssim2_gpu | -0.0020 | 0% | 3.5357 |

- **iwssim = a literal copy of `human_score`** (100% identical, corr
  1.0). NOT IW-SSIM scores. Training with `--target-column iwssim` on
  these corpora trains on a leaked target = the eval label.
- **ssim2_gpu = near-zero correlation with human_score** and wrong
  scale (std 0.72 / 3.5 vs the expected [0,100]). Garbage, not SSIMULACRA2.

safesyn's columns look legitimate by contrast (iwssim corr 0.896,
ssim2_gpu corr 1.0 because safesyn's human_score IS ssim2-derived).

## Impact

- **Shipped bakes are SAFE**: V39 and all production bakes train with
  `--target-column human_score`, never iwssim/ssim2_gpu. No shipped
  bake is contaminated.
- **Multi-target experiments are NOT safe**: any bake that used
  `--target-column iwssim` (or a mix including it) on kadid/tid trained
  on the eval label — those results are invalid and must be re-run
  after the columns are rebuilt. The CLAUDE.md "Multi-target training
  corpus" plan depends on real iwssim/ssim2 columns that don't exist
  for kadid/tid.

## Root cause (traced 2026-05-25)

The corruption originates in the upstream build of
`/mnt/v/zen/zensim-training/2026-05-18-v24/{kadid,tid}_4target_372col.parquet`
(an ad-hoc step, not a committed script). Both canonical builders faithfully
propagate it: `scripts/canonical_corpus/build_canonical_parquets.py`
(`build_kadid()` L132-146 / `build_tid()` L148-162 just `pq.read_table` the
v24 source + `select_canonical_schema`), then
`scripts/canonical_corpus/build_canonical_2026_05_21.py:rename_typo_columns`
(L71+) copies them forward. Neither builder introduces the bug; neither
validated against it.

The v24 parquet's pandas metadata shows the construction order: the base
DataFrame was `ref_basename, human_score, iwssim, f0..f371` and the other
target columns (`iwssim_log_norm, cvvdp_score, …, ssim2_gpu, ssim2_log_norm,
mix_*`) were appended afterward. This pins both failure modes:

- **`iwssim` = literal copy of `human_score`** — created as a placeholder in
  the base DataFrame and never overwritten with real IW-SSIM. A target leak.
- **`ssim2_gpu` joined on `ref_basename` ALONE** — confirmed: it is **constant
  within every reference group** (81 KADID refs, 1 unique ssim2 value each;
  for I01 = 99.992 across all 125 distorted variants regardless of MOS). By
  contrast `cvvdp_score` correctly varies (~122 unique per ref). So ssim2 was
  effectively `ssim2(ref, ref) ≈ 99.99` broadcast onto every distortion of
  that ref → ~0 corr with MOS. **Failure mode (c): right metric on the right
  scale, but computed/joined on misaligned (ref, dist) pairs (ref-vs-ref).**

Downstream poisoning: `iwssim_log_norm`, `ssim2_log_norm`, and every `mix_*`
column (each is `Σ w·{cvvdp,iwssim,ssim2}_log_norm`) are derived from the
corrupt raw values and are therefore ALSO corrupt.

## Scope of affected parquets

| Parquet | iwssim | ssim2_gpu | Affected? |
|---|---|---|---|
| `canonical-2026-05-21/train/{kadid,tid}.parquet` | human-copy | ref-misjoin | **YES** |
| `canonical-2026-05-18/train/{kadid,tid}.parquet` | human-copy | ref-misjoin | **YES** |
| `2026-05-18-v24/{kadid,tid}_4target_372col.parquet` (upstream) | human-copy | ref-misjoin | **YES (origin)** |
| `canonical-2026-05-{18,21}/val/{kadid,tid}.parquet` | all-NULL | all-NULL | no (only carry human_score) |
| `2026-05-15-full-features/{kadid,tid}_features_372col*.parquet` | column ABSENT | column ABSENT | no |
| `canonical-*/train/safesyn.parquet` | real (corr 0.896) | real (corr 1.0) | no |

## Resolution (2026-05-25)

Recomputed real IW-SSIM + SSIMULACRA2 on the correct (ref, dist) pairs via
`zen-metrics batch --metric {ssim2-gpu,iwssim}` and re-joined positionally.

Row→image mapping recovered + verified (max_err 0):
- KADID: parquet row order == `dmos.csv` row order (full ref match);
  `human_score == (dmos − 1) / 4`; row i → `dist_img`/`ref_img`.
- TID: parquet row order == `mos_with_names.txt` row order (full ref match);
  `human_score == mos / 9`; row i → `names[i]` distorted PNG, ref `I{NN}.png`.

Tools (committed):
- `scripts/canonical_corpus/fix_kadid_tid_build_pairs.py` — recovers per-row
  (ref_path, dist_path) and emits pairs TSVs.
- `scripts/canonical_corpus/fix_kadid_tid_apply_scores.py` — re-joins
  recomputed scores positionally, replaces iwssim / ssim2_gpu / their
  *_log_norm / all mix_* (transforms verified-exact: iwssim_log_norm
  SLOPE_IW=7.2837 INT_IW=0.0302; ssim2_log_norm `(s2+30)/1.3`;
  mix_cv33_iw33_sm33 = mean of the three log_norms). Writes
  `*_fixed_2026-05-25.parquet` SIBLINGS; never overwrites originals.

Build-script guard (committed): `build_canonical_2026_05_21.py` now calls
`_validate_metric_columns()` which raises if iwssim is a human_score copy or
ssim2_gpu is constant within every ref group — so a rebuild can't silently
re-ship the corrupt columns. (Verified: fires on the corrupt kadid, passes on
clean safesyn.)

PoC (50 stratified KADID pairs): recomputed ssim2 SROCC 0.830, iwssim SROCC
0.935 vs DMOS (corrupt baseline 0.013 / leaked-copy).

Full-corpus post-fix SROCC vs human MOS (no NaN, row order + all other
columns preserved):

| corpus | iwssim before | iwssim after | ssim2_gpu before | ssim2_gpu after |
|---|--:|--:|--:|--:|
| kadid | 1.0000 (leak) | 0.8498 | 0.0125 | 0.8133 |
| tid | 1.0000 (leak) | 0.7794 | −0.0020 | 0.8460 |

iwssim==human identical% is now 0 (was 100). ssim2_gpu range restored to a
proper SSIMULACRA2 scale (KADID −367..100, TID −96..90).

Outputs: per-pair score TSVs at
`/mnt/v/output/zensim/data-fix-2026-05-25/{kadid,tid}_{ssim2,iwssim}.tsv`;
corrected parquets at
`canonical-2026-05-{18,21}/train/{kadid,tid}_fixed_2026-05-25.parquet`.
**Originals NOT overwritten — user reviews before promotion.** To promote:
`mv {corp}_fixed_2026-05-25.parquet {corp}.parquet`, rebuild `_MANIFEST.json`,
re-sync R2, then re-run any multi-target bake that used
`--target-column iwssim` / a mix including it on kadid/tid (those results were
trained on the leaked eval label and are invalid).

## Provenance

Found during the AIC-3 CVVDP-feature spike (general-purpose agent,
2026-05-25). The spike itself only used CVVDP (honest) on these
corpora; it did not use the poisoned iwssim column. Root-caused + fixed by
the data-integrity repair agent, 2026-05-25.
