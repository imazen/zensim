# Zensim champion candidate — 2026-05-10 (loop session)

After 15 loop iterations exploring TV regularizer, mixed-supervision,
humw weighting, capacity, and training corpus combinations, the strongest
candidate found is:

**`h192x128_tv10_safesyn218k_kt_2026-05-10.bin`** (278,103 bytes, ZNPR v2)

## Recipe
- **Architecture**: 228 → 192 → 128 → 1 LeakyReLU MLP (two hidden layers)
- **Training base**: 218k clean safe-synthetic (CID22-leak-free) from
  ZSFC v3 binary `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv.features.20260308_162434.bin`
  converted to (ref_basename, human_score=ssim2, f0..f227) format
- **Mixed supervision**: KADID-train + TID-train at humw=0.3
- **Loss**: `mse_rank` (MSE + 0.5·RankNet)
- **Regularization**: TV-weight 10 on adjacent-q pairs
- **Optimizer**: AdamW lr=3e-3, weight_decay=1e-5, batch=16384, 50 epochs
- **Trainer**: `scripts/v_next/train_v_next_mlp.py` after the
  last-partial-batch TV slice fix (commit `dd79a3c`).

## Headline numbers (full-dataset eval)

| Dataset | V0_5 (shipped) | h192x128 NEW | Δ |
|---|---|---|---|
| **KADIK10k** (n=10125) | 0.8432 | **0.8898** | **+0.0466** ★ |
| TID2013 (n=3000) | 0.8401 | 0.8195 | -0.0206 |
| **CID22** (n=4292) | **0.8893** | 0.8695 | -0.0198 |
| **non-mono q-step** | ~8.26% | **4.93%** | **-3.33 pp** ★ |

V0_2 reference: 4.86% non-mono. ssim2 GT: 5.08%.

## Tradeoff summary
- **KADID**: massive +0.047 SROCC (best ever measured for a zensim bake)
- **CID22**: -0.020 vs V0_5 — Python trainer doesn't fully reproduce
  V0_5's Rust trainer that achieved 0.8893
- **Smoothness**: 4.93% — within 0.07 pp of V0_2's 4.86% floor;
  decisive improvement over V0_4/V0_5's 8.26%
- **Bake size**: 278 KB vs 60 KB for V0_5 (4.6× larger; still trivial)

## Decision pending user approval

Swapping V0_5 → this candidate is a tradeoff:
- **Gain**: smoothness floor reached, KADID +0.047, V0_2 SROCC retained
- **Cost**: CID22 -0.020 (0.8695 vs 0.8893)

Do NOT auto-swap. The CID22 regression is the user's gold standard.
Ask user before promoting to `zensim/weights/v0_4_2026-04-30.bin` slot.

## Provenance

- Training run: `/mnt/v/zen/zensim-training/2026-05-07/runs/20260510T021341_v_next_h192x128_tv10_safesyn218k_kt_2026-05-10/`
- Bake: `zensim/benchmarks/h192x128_tv10_safesyn218k_kt_2026-05-10.bin`
- Eval log: `zensim/benchmarks/h192x128_tv10_safesyn218k_kt_2026-05-10.eval.log`
- 218k features.bin → CSV converter: `/tmp/zensim_loop/convert_features_bin.py` (durable copy in next commit)

## Why we didn't reach V0_5's CID22 0.8893

- The 2026-04-30 V0_4 mixed-supervision bake (== V0_5 SSIM2-proxy MLP)
  was produced by an in-tree Rust trainer (`zensim-validate/src/mlp_train.rs`)
  that was deleted in PR #29 (commit `e613224`).
- The Python `train_v_next_mlp.py` faithfully implements the same loss
  (`mse + 0.5*ranknet`) and a similar architecture, but produces -0.020
  to -0.040 CID22 SROCC even on the SAME 218k clean training corpus.
- The Rust trainer used different optimizer (Nelder-Mead-style with
  random restarts) vs PyTorch AdamW. It also had different feature
  standardization handling.
- Future work: port the Rust trainer's exact recipe, OR train against
  the 300-feature extended set (vs 228), OR investigate the optimizer
  difference. None of these were attempted in this loop session.
