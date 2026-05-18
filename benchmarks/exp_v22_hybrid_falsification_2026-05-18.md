# EXP-V22-HYBRID — falsified vs BOTH trail gates (2026-05-18)

## Hypothesis (Step 1)

1. **Hypothesis**: V_22-mix-LARGE+iwssim recipe + `hybrid_head` rank+pool fusion (a shared learned scalar α gate) should land between V_22-mix-LARGE+iwssim (Balanced) and V_24-per-sample-α s4 (Compression) on CID22/AIC-3 while preserving the Balanced ship's strong KADID/TID/KonJND coverage. Targeting CID22 ≥ 0.860 AND KonJND ≥ 0.85.
2. **Falsification**: If neither trail gate passes — fails balanced (B>>A on any of KADID/TID/KonJND with KonJND ≤ −0.10 ceiling) and fails compression (no A>>B decisive on either CID22 or AIC-3 vs current persample-α ship) — hypothesis is dead.
3. **Cost ceiling**: 5 seeds × `mix_cv40_iw60` target, 300 epochs, early-stop 60, ~10 min wall total.
4. **Ship form**: PreviewV0_5 trail rotation (no crate version bump) IF either gate passes.

## Reporting panel (Step 2)

Mohammadi full panel (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE) at aggregate + 10-band level, on all 5 corpora (CID22, KADID, TID, KonJND, AIC-3). bake_compare with 1000-bootstrap §A.9 verdict. Controls in `benchmarks/baseline_panels_2026-05-18.md`.

| Corpus | Role | When inspected |
|---|---|---|
| CID22 | gold-standard holdout | only after 5-seed packed |
| AIC-3 CTC | compression holdout | only after 5-seed packed |
| KADID | integrity guard | only after 5-seed packed |
| TID | integrity guard | only after 5-seed packed |
| KonJND | PJND anchor | only after 5-seed packed |

## Recipe & lineage

- **Architecture**: 300 → 128 → 128 (LeakyReLU α=0.01, identity-passthrough final layer) + `zentrain.hybrid_head` metadata (shared learned scalar α-gate fusing rank head + pool head, NOT NiN; identical to V_24-hybrid no-NiN architecture but trained on V_22 recipe).
- **Trainer**: `zensim-validate/src/bin/zensim_mlp_train.rs`, `--hybrid-head --target-column mix_cv40_iw60`.
- **Groups + weights** (V_22-mix-LARGE+iwssim recipe verbatim):
  - safesyn (cvvdp_iwssim_300col) — w_train 1.0, w_val 0.0
  - kadid_mix_300col — w_train 0.3, w_val 1.0
  - tid_mix_300col — w_train 0.3, w_val 1.0
  - konjnd_mix_300col — w_train 0.02, w_val 1.0
  - cvvdp_iwssim_large_300col_v2 — w_train 0.5, w_val 0.0
- **Hyperparameters**: `--hidden 128 --max-features 300 --epochs 300 --pairs-per-epoch 50000 --lr 0.001 --l2 0.00001 --leaky-alpha 0.01 --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0`
- **Seeds**: 1..5 (PIDs 876889–876893 in parallel on workspace `zensim--exp-v22-hybrid`).

## Training input data

| Group | Path | Rows |
|---|---|--:|
| safesyn | `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet` | (LARGE 300-col cvvdp+iwssim mix) |
| kadid | `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet` | 10,125 |
| tid | `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet` | 3,000 |
| konjnd | `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet` | 1,008 |
| cvvdp_iwssim_LARGE | `/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet` | 73,300 |

CID22-leak status: cvvdp_iwssim_LARGE join is the same merge used by the current Balanced ship. No CID22 human MOS in any training group.

## 5-seed CI (CID22 SROCC target)

| Seed | CID22 | KADID | TID | KonJND | AIC-3 | Bake md5 |
|--:|---:|---:|---:|---:|---:|---|
| 1 | 0.8585 | 0.9255 | 0.8875 | 0.7571 | 0.8000 | — |
| 2 | 0.8739 | 0.9329 | 0.8906 | 0.7859 | 0.8088 | — |
| 3 (**median**) | **0.8662** | **0.9314** | **0.8906** | **0.7812** | **0.8033** | `516ffba9555f28c2a67eba993f4f458e` |
| 4 | 0.8694 | 0.9274 | 0.8850 | 0.7573 | 0.8066 | — |
| 5 | 0.8436 | 0.9208 | 0.8914 | 0.7413 | 0.7992 | — |

| Stat | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| mean | 0.8623 | 0.9276 | 0.8890 | 0.7646 | 0.8036 |
| std | 0.0119 | 0.0048 | 0.0027 | 0.0186 | 0.0041 |
| min | 0.8436 | 0.9208 | 0.8850 | 0.7413 | 0.7992 |
| max | 0.8739 | 0.9329 | 0.8914 | 0.7859 | 0.8088 |

KonJND is the highest-variance corpus (σ=0.0186) — none of the 5 seeds breaks 0.80, and the best seed (s2 at 0.7859) is still 0.107 below the Balanced ship's 0.8927.

## Packed bake

Median seed (s3) packed via `zenpredict repack --dtype i8 --zerobias 0.005 --compress`:
- Raw: 223,354 bytes, md5 `516ffba9555f28c2a67eba993f4f458e`
- Packed: 43,387 bytes (19.4% of input), md5 `bc20284e75412e5ba82375fbda1271bd`
- CID22 SROCC drift: raw 0.8662 → packed 0.8657 (Δ=−0.0005, exactly at the 0.0005 i8 ceiling)
- No fallback to f16 needed.
- Bake byte at offset 4: `0x03` (ZNPR v3 ✓)
- Carries `zentrain.hybrid_head` metadata — scoreable via the `forward_one_bake` hybrid-head dispatch landed earlier 2026-05-18

## A.9 1000-bootstrap verdicts

### vs Balanced ship (V_22-mix-LARGE+iwssim s3, `b703c9cfc7e1908faf5b0e78dc823221`)

Full report at `/mnt/v/zen/zensim-eval/exp_v22_hybrid_2026-05-18/verdicts/vs_balanced.md`. Aggregate verdicts (1000-bootstrap, § A.9):

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8657 | 0.8324 | 0.503 | 0.559 | 0.9173 | 0.9006 | +41.973 | +127.071 | +34.978 | **A>>B** |
| KADIK10k | 10125 | 0.9315 | 0.9677 | 0.362 | 0.249 | 0.9596 | 0.9804 | −90.810 | −792.667 | (B>>A) | **B>>A** |
| TID2013 | 3000 | 0.8906 | 0.9729 | 0.431 | 0.236 | 0.9181 | 0.9832 | −54.098 | −311.269 | (B>>A) | **B>>A** |
| KonJND-1k | 1008 | 0.7814 | 0.8927 | 0.568 | 0.376 | 0.8284 | 0.9178 | −46.356 | −141.309 | (B>>A) | **B>>A** |
| AIC-3 CTC | 600  | 0.8034 | 0.7845 | 0.583 | 0.606 | 0.8758 | 0.8630 | +17.444 | +35.483 | +14.537 | **A>>B** |

**Balanced trail gate** (A>>B on CID22 decisively AND not decisively B>>A on any of {KADID, TID, KonJND, AIC-3}):
- Step 1 PASS — A>>B decisive on CID22 (+0.0333, h=+41.97).
- Step 2 FAIL — B>>A decisive on KADID (−0.0362), TID (−0.0823), AND KonJND (−**0.1113**).
- Step 3 FAIL — KonJND regression −0.1113 EXCEEDS the −0.10 noise tolerance.

**Balanced trail verdict: FAIL by wide margin (3 of 4 anti-corpora are B>>A; KonJND alone breaches the tolerance ceiling).**

### vs Compression ship (V_24-per-sample-α s4, `f09a9abdce00805000c1d112c2421b2d`)

Full report at `/mnt/v/zen/zensim-eval/exp_v22_hybrid_2026-05-18/verdicts/vs_compression.md`. Aggregate verdicts:

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8657 | 0.8641 | 0.503 | 0.508 | 0.9173 | 0.9157 | +6.841 | +40.406 | +0.000 | tied |
| KADIK10k | 10125 | 0.9315 | 0.9316 | 0.362 | 0.362 | 0.9596 | 0.9602 | −3.337 | −24.943 | −0.000 | promising |
| TID2013 | 3000 | 0.8906 | 0.8893 | 0.431 | 0.432 | 0.9181 | 0.9173 | +40.200 | +54.364 | +13.400 | promising |
| KonJND-1k | 1008 | 0.7814 | 0.8080 | 0.568 | 0.502 | 0.8284 | 0.8505 | −25.577 | −112.110 | (B>>A) | **B>>A** |
| AIC-3 CTC | 600  | 0.8034 | 0.8183 | 0.583 | 0.565 | 0.8758 | 0.8856 | −79.916 | −161.376 | (B>>A) | **B>>A** |

**Compression trail gate** (A>>B on ≥1 of {CID22, AIC-3} decisively AND not decisively B>>A on the other of {CID22, AIC-3} AND no single corpus −0.10 regression on {KADID, TID, KonJND}):
- Step 1 FAIL — neither CID22 (tied, DecScore +0.000) nor AIC-3 (B>>A) is A>>B decisive.
- Step 2 FAIL — B>>A decisive on AIC-3 (−0.0149).
- Step 3 PASS — KADID −0.0001, TID +0.0013, KonJND −0.0266 all within −0.10 tolerance.

**Compression trail verdict: FAIL on steps 1 AND 2. CID22 is statistically tied with the current Compression ship (h=+6.8 is small relative to A.9 cutoff but DecScore says tied; aggregate Δ = +0.0016). AIC-3 falls decisively.**

## Both trails: FALSIFIED

Neither trail gate passes. **No ship action.** Candidate is added to `SOTA_TRAILS.md` candidate matrix as a falsification record.

## Honest gaps

- **KonJND collapse**: konjnd weight 0.02 (V_22 recipe's standard) is too low for the hybrid-head architecture to learn the PJND surface — every seed lands in [0.7413, 0.7859], well below ssim2's 0.8927 and Balanced's 0.8927.
- **KADID/TID parity with Compression ship**: hybrid_head + cvvdp_iwssim_LARGE merge gives the architecture KADID/TID regression-free vs V_24-per-sample-α s4, but the resulting CID22/AIC-3 trade is not strong enough to flip steps 1+2 of the compression gate.
- **CID22 plateau**: 5-seed mean 0.8623, max 0.8739 — comparable to V_22-mix-LARGE+iwssim baseline (0.8324) by +0.030 but matches V_24-per-sample-α s4's 0.8641 only at the median. Hybrid-head's contribution over plain MLP on this recipe is small (and architecture-confounded vs the V_24 family).

## Reproducibility

- Workspace: `/home/lilith/work/zen/zensim--exp-v22-hybrid/` (jj colocated)
- Trainer run logs: `/tmp/exp_v22_hybrid_logs/seed{1..5}.log`
- Post-pipeline log: `/tmp/exp_v22_hybrid_cont_post.log`
- Verdict tree: `/mnt/v/zen/zensim-eval/exp_v22_hybrid_2026-05-18/verdicts/`
- 5-seed summary CSV: `verdicts/5seed_summary.csv`
- Median seed selection: `verdicts/median_seed.txt`
- Packed bake: `/mnt/v/zen/zensim-eval/exp_v22_hybrid_2026-05-18/v22_hybrid_s3_h128_packed.bin`

## See also

- Architecture sibling V_24-hybrid no-NiN s4 (V_24 recipe + hybrid_head): `benchmarks/v24_hybrid_no_nin_s4_vs_{balanced,compression_ship}_2026-05-18.md`. V_24-hybrid no-NiN s4's packed CID22 0.8657 was identical to this experiment's, but on the V_24 recipe; it also fails both gates. The architectural choice (hybrid_head shared α-gate) does not by itself flip either gate — the trail-relevant signal is in the recipe (V_22 vs V_24) and the per-sample α-gate (which this experiment does not use).
