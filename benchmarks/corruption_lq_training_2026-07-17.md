# Corruption + LQ training — zensim's own features BEAT butteraugli (2026-07-17)

User: "why do we need butteraugli — we built a corruption corpus worse than kadis"
(= kadis_negrich) + "keep held-out rows for gating". Result: adding a corruption
group + kadis_negrich to the 2-layer psa recipe closes every low-end gap at once.

## The setup (held-out gating, per user directive)

- `kadis_negrich` (266k severe-KADIS, ssim2 to -18): split 85% train / 15% gate
  (`/mnt/v/output/zensim/corr-lq/kadis_negrich_{train,gate}.parquet`). train-only,
  uniform pairing (no source_id needed — NO rebuild; the Rust loader reads
  ref_basename "when present", the Python trainers already consume it).
- structural corruption (gb82 grid, 44 families): split by TYPE — 30 train / **14
  HELD-OUT** types (`corruption_{train,gate}.parquet`), withinref (corruption vs its
  q10/q20). Ordered synthetic target (corruption 0.02 < q10 0.30 < q20 0.55).
- manifest `depth_v2_corrlq.toml` = depth_v2 (2-layer psa+tanh) + both groups
  train-only. Trainer: `zensim_mlp_train`. Two configs: min-lift 0.20 (cl_tfm) and
  none (cl_notfm). Eval: `scripts/v_next/corr_lq_eval.py`.

## Result — held-out numbers

| bake | held-out corr-gate | kadis_negrich-SROCC | CID22 | imazen26 | nonphoto | KonJND | HF |
|---|---|---|---|---|---|---|---|
| fs_minlift020 (no corr/LQ) | 22.5% | 0.697 | 0.858 | 0.949 | 0.949 | 0.723 | 0.00 |
| **cl_tfm** | **100.0%** | **0.912** | 0.883 | 0.940 | 0.939 | 0.761 | 0.06 |
| cl_notfm | 100.0% | 0.945 | 0.864 | 0.929 | 0.927 | 0.778 | 0.05 |
| refs | B 18.8% / butter 72% | — | B 0.876 | B 0.841 | B 0.861 | B 0.547 | B 0.61 |

**cl_tfm: held-out corruption gate 22.5→100% (> butteraugli 72%, zensim features
only), LQ SROCC 0.70→0.912, CID22 0.883 (BEATS B), imazen26 0.940 (ssim2 ceiling
held), KonJND 0.761 (clears G5).** The ssim2 ceiling cost ~0.009 for all of it.

**The reframe held:** butteraugli isn't needed — zensim's edge/peak-max features
carry the corruption signal; B failed only because it never TRAINED on corruption.
Give it the exposure (corruption + kadis_negrich groups) and it beats butteraugli.

## Caveats
- 100% corruption is held-out BY TYPE on ONE source (gb82). Generalizes across
  corruption families; deployment needs the corruption corpus across many sources
  + a source-held-out gate. (Mechanism proven; scale-up is data-gen.)
- HF near-lossless still ~0 (the winsor/yeo-johnson transform compression, a
  separate issue — see hf_nearlossless_saturation_2026-07-17.md).
- cl_tfm > cl_notfm on ssim2/CID22 (transforms help mid-q); cl_notfm > on LQ/KonJND.

Bakes: `/mnt/v/output/zensim/corr-lq/cl_{tfm,notfm}.bin`.
