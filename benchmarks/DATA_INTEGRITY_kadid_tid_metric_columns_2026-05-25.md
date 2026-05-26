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

## Fix (next session)

Recompute real IW-SSIM and SSIMULACRA2 on the KADID/TID (ref, dist)
image pairs and overwrite these columns. The CLAUDE.md "Canonical
training/validation corpora" section claims these columns are real;
the build script `scripts/canonical_corpus/build_canonical_parquets.py`
either copied human_score into iwssim by mistake or never populated it.
Audit that script's iwssim/ssim2_gpu join before rebuilding.

## Provenance

Found during the AIC-3 CVVDP-feature spike (general-purpose agent,
2026-05-25). The spike itself only used CVVDP (honest) on these
corpora; it did not use the poisoned iwssim column.
