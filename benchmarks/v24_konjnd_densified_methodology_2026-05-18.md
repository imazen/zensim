# V_24 KonJND-1k densified training — methodology + verdict

**Date:** 2026-05-18
**Branch:** `feat/ex2-stdpool-head`
**Workspace:** `/home/lilith/work/zen/zensim--konjnd-densify`
**Agent:** claude-konjnd-dense
**Status:** **FALSIFIED on Pareto gate — KonJND-1k validation collapses from 0.8927 → 0.31..0.42**

---

## Hypothesis (recap)

After the prior agent (commit `132cddd9`) found KonJND++ unpublished, the no-regret alternative was to **densify the KonJND-1k training group** from the legacy 1,008 rows (one row per source, anchored on per-source PJND threshold) to ~20k rows by adding per-distortion-level pairs from the same 1,008 sources. Hypothesis:

> Adding 20× more (source, distorted) pairs at varied quality levels gives the trainer richer JND-axis gradient signal, lifting held-out KonJND-1k SROCC without breaking the V_22-LARGE Pareto on CID22 / KADID / TID / AIC-3.

**Pareto gate (from the task spec):**
- CID22 ≥ V_22 + 0.005
- KonJND-1k ≥ V_22's 0.893 − 0.01 (= 0.883)
- AIC-3 ≥ V_22 + 0.015
- KADID/TID within −0.01 of V_22

---

## KonJND-1k full-corpus audit

Per the on-disk corpus at `/mnt/v/datasets/KonJND-1k/`:

| Element | Count |
|---|---|
| Total source images | 1,008 (504 unique to JPEG + 504 unique to BPG) |
| Distortion levels per JPEG source | 100 (q = 001..100) |
| Distortion levels per BPG source | 51 (q = 001..051) |
| Total distorted variants | 76,104 (50,400 JPEG + 25,704 BPG) |
| Subjective ratings file | `subjective_ratings.csv` — 1,008 rows, one PJND threshold + std + raw ratings list per (source, codec) |

PJND annotation schema: each row carries `image_id, Compression type, No. of ratings, mean, std, ratings`. The `mean` column is the **mean perceived-just-noticeable-difference threshold on the 1..100 q scale**, hand-annotated per CID22 paper Table 4. **There is NO per-distortion-level PJND** — the only human-derived signal is one threshold per (source, codec). The prior agent's "1,008 × ~131 × full PJND annotations" figure is incorrect: it's **504 × 100 (JPEG) + 504 × 51 (BPG) distorted variants** with **one PJND scalar per source**.

Per-pair ssim2/butteraugli/dssim scores for all 76,104 variants are pre-computed in `/mnt/v/datasets/KonJND-1k/konjnd_full_scored.csv` (gpu_ssimulacra2 + gpu_butteraugli + dssim columns).

The full per-pair 372-feature extraction was also already on disk at `/mnt/v/zen/zensim-training/2026-05-15-full-features/konjnd_full_features_372col_2026-05-15.csv` (76,105 rows × 374 cols, anchored on `human_score = gpu_ssimulacra2 / 100`).

---

## Densified parquet build

**Builder:** `scripts/v_next/build_konjnd_dense_parquet.py`

**Sampling strategy:** stratified per-source — for each of the 1,008 `ref_basename`s, sort that source's rows by `human_score` (ssim2/100), pick 20 evenly-spaced ranks across the JND ladder.

**Output:** `/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_features_mix_targets_{372col,300col}.parquet`
- 20,160 rows × 385 cols (372col) / 313 cols (300col)
- `human_score`: ssim2 score × 100 (range [-65.7, 96.2], median 67.7 — at the PJND ssim2 boundary per Table 4)
- `mix_cv40_iw60` and all `mix_*` columns: copies of `human_score` (no real mix for KonJND)
- 67.7 MB (372col) / 54.0 MB (300col) on disk, zstd-compressed

**Schema match against legacy 1008-row konjnd:** identical column names + types; only row count differs (1,008 → 20,160).

---

## Training

Two variants, both seed=3, identical recipe except `konjnd_w`:

**Recipe (mirrors V_22-LARGE + per-sample-α):**
```
zensim_mlp_train \
  --group safesyn:.../safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:.../kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:.../tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_features_mix_targets_300col.parquet:${KONJND_W}:1.0 \
  --group cvvdp_iwssim_large:.../cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 --target-scale 100.0 \
  --val-policy min --minibatch-size 256 \
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed 3 --early-stop-patience 0
```

| Variant | konjnd_w | best val SROCC | training konjnd SROCC final |
|---|---|---|---|
| dense_002 | 0.02 (V_22 default) | 0.9867 | 0.9940 |
| dense_010 | 0.10 (boost) | 0.9862 | 0.9964 |

Training time: ~8 min per variant on the local 7950X (300 epochs × 50k pairs/epoch × 256-batch).

---

## Verdict — full Mohammadi 2025 panel via `bake_compare`

**Tool:** `/home/lilith/work/zen/zensim/target/release/bake_compare` with 500 bootstrap resamples.
**Reports:** `/mnt/v/zen/zensim-eval/v24_konjnd_dense_2026-05-18/cmp_*.md`

### dense_002 (konjnd_w=0.02) vs V_22-mix-LARGE

| Corpus | n | SROCC_A | SROCC_B | Δ SROCC | PWRC_A | PWRC_B | Z-RMSE_A | Z-RMSE_B | Verdict |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22 | 4292 | **0.8832** | 0.8324 | **+0.0508** | 0.9258 | 0.9006 | 0.469 | 0.559 | A>>B |
| KADID-10k | 10125 | 0.9330 | **0.9677** | -0.0347 | 0.9611 | 0.9804 | 0.358 | 0.249 | B>>A |
| TID2013 | 3000 | 0.8896 | **0.9729** | -0.0833 | 0.9172 | 0.9832 | 0.432 | 0.236 | B>>A |
| **KonJND-1k** | 1008 | **0.3147** | **0.8927** | **-0.5780** | 0.4287 | 0.9178 | 0.948 | 0.376 | B>>A |
| AIC-3 CTC | 600 | **0.8215** | 0.7845 | **+0.0370** | 0.8889 | 0.8630 | 0.556 | 0.606 | A>>B |

### dense_010 (konjnd_w=0.10) vs V_22-mix-LARGE

| Corpus | n | SROCC_A | SROCC_B | Δ SROCC | Verdict |
|---|--:|---:|---:|---:|---|
| CID22 | 4292 | **0.8861** | 0.8324 | **+0.0537** | A>>B |
| KADID-10k | 10125 | 0.9316 | **0.9677** | -0.0361 | B>>A |
| TID2013 | 3000 | 0.8896 | **0.9729** | -0.0833 | B>>A |
| **KonJND-1k** | 1008 | **0.4163** | **0.8927** | **-0.4764** | B>>A |
| AIC-3 CTC | 600 | **0.8197** | 0.7845 | +0.0352 | A>>B |

### dense_002 vs per-sample-α baseline (`persample_seed4_packed.bin`)

| Corpus | n | SROCC_A | SROCC_B | Δ SROCC | Verdict |
|---|--:|---:|---:|---:|---|
| CID22 | 4292 | 0.8832 | 0.8641 | +0.0191 | A>>B |
| KADID-10k | 10125 | 0.9330 | 0.9316 | +0.0014 | A>>B |
| TID2013 | 3000 | 0.8896 | 0.8893 | +0.0003 | tied |
| **KonJND-1k** | 1008 | **0.3147** | **0.8080** | **-0.4933** | B>>A |
| AIC-3 CTC | 600 | 0.8215 | 0.8183 | +0.0032 | tied |

### Pareto gate evaluation

| Criterion | Target | dense_002 | dense_010 | Pass? |
|---|---|---|---|---|
| CID22 ≥ V_22 + 0.005 | 0.8374 | 0.8832 | 0.8861 | **PASS** |
| KonJND ≥ 0.883 (V_22 − 0.01) | 0.883 | **0.3147** | **0.4163** | **FAIL by -0.57 / -0.47** |
| AIC-3 ≥ V_22 + 0.015 | 0.7995 | 0.8215 | 0.8197 | PASS |
| KADID within −0.01 of V_22 | ≥ 0.9577 | 0.9330 | 0.9316 | **FAIL by -0.025** |
| TID within −0.01 of V_22 | ≥ 0.9629 | 0.8896 | 0.8896 | **FAIL by -0.073** |

**Pareto gate: FAILED on KonJND, KADID, and TID.**

The dense_002 bake **wins CID22 and AIC-3 decisively** (paneled bootstrap; 5 cells A>>B aggregate vs 17 B>>A). But the KonJND collapse is so severe (-0.578 SROCC vs V_22; -0.493 vs per-sample-α) that it disqualifies the variant as a ship candidate.

**No winner to pack via `zenpredict repack` — both variants fail the load-bearing KonJND gate.**

---

## Why densification destroyed KonJND held-out SROCC — load-bearing finding

The KonJND-1k validation parquet at `2026-05-15-full-features/konjnd_features_372col_2026-05-15.parquet` has:

- **`human_score`** = per-source **mean PJND threshold** (one scalar per source, 22..70 on the 1..100 q scale)
- **`f0..f371`** = features from the **(source, distorted-at-round(mean_PJND))** pair — one specific quality level per source

The densified training parquet has:

- **`human_score`** = per-pair **ssim2 / 100** (continuous through the JND zone, range -65..96)
- **`f0..f371`** = features from each (source, distorted-at-q_i) pair across 20 evenly-spaced quality levels per source

**The two `human_score` columns measure different things:**
1. The legacy 1008-row "training" file uses (source, dist-at-PJND) features paired with PJND threshold as the target — teaching the model "given features at the PJND, predict the PJND threshold." The MLP learns a per-source-difficulty proxy.
2. The densified file uses per-pair features paired with per-pair ssim2 — teaching the model "given features at any quality level, predict the ssim2 score." The MLP learns a continuous per-pair quality predictor.

When `bake_compare` evaluates the dense-trained MLP on the legacy KonJND val set:
- Input: features from (source, dist-at-PJND) — these are mid-band quality features
- MLP output: predicted ssim2 score at that quality level (range ~50..70 typically, since PJND is mid-band)
- Compared against: per-source PJND threshold (1..100 q scale)

The MLP's per-source variation in output is small (all sources at q=PJND have similar mid-band ssim2 ~60), so the output rank doesn't track the wide variation in PJND thresholds across sources. SROCC collapses to ~0.31.

**The legacy 1008-row training succeeds because its features encode source difficulty implicitly (a source with low PJND is "harder to compress without visible distortion", and the (source, dist-at-PJND) features carry signal about that), and the MLP regresses that signal directly to the PJND scalar.**

Densification with per-pair ssim2 targets is a **fundamentally different supervision signal** — it solves a different problem and cannot satisfy the legacy val set's encoding without explicit per-source aggregation in the bake forward path (which the runtime doesn't do).

### What WOULD work — for future agents

To make a densified KonJND training group lift the legacy KonJND val SROCC, the per-row target needs to be the **per-source PJND threshold** (broadcast across all 20 distortion variants from that source). Then the MLP would learn: "given features at distortion level q, predict the source's PJND threshold." This forces the model to extract source-difficulty signal from per-pair features at arbitrary q — a different and harder task than per-pair ssim2 prediction, but one that directly matches the val set's encoding.

That is **NOT a simple parquet rebuild**. It would require:
1. A new target column `pjnd_threshold` populated per source from `subjective_ratings.csv`
2. The trainer's `--target-column pjnd_threshold` reading the broadcast value
3. The val-policy semantics still using ssim2-shaped features (so the model also continues to predict ssim2 on safesyn). This would require a dual-target head or separate per-group target columns

Alternative — closer to "just denser pairs": train on the densified file with `human_score = per-source PJND threshold (broadcast)` AND keep the val set's PJND-threshold encoding. The 20 rows per source would all share the same target. The trainer's PJND-pair-weighting would fire harder (more pairs near threshold=45 in the per-source-PJND distribution). **This is the recommended next experiment.**

---

## PJND-target retry — broadcast threshold across distortion levels

After the ssim2-target falsification, I built a second densified parquet:
`konjnd_dense_pjndtarget_300col.parquet` (20,160 rows, identical features,
but `human_score` = per-source PJND threshold broadcast across all 20
distortion variants per source). This directly matches the legacy
KonJND val set's encoding.

Two variants trained (seed=3, konjnd_w ∈ {0.02, 0.10}):

### dense_pjnd_010 (konjnd_w=0.10) vs V_22-mix-LARGE

| Corpus | n | SROCC_A | SROCC_B | Δ SROCC | Verdict |
|---|--:|---:|---:|---:|---|
| CID22 | 4292 | **0.6223** | 0.8324 | **-0.2101** | B>>A |
| KADID-10k | 10125 | 0.9277 | 0.9677 | -0.0400 | B>>A |
| TID2013 | 3000 | 0.8879 | 0.9729 | -0.0850 | B>>A |
| KonJND-1k | 1008 | **0.8896** | 0.8927 | **-0.0031** | tied |
| AIC-3 CTC | 600 | 0.7678 | 0.7845 | -0.0167 | promising |

### dense_pjnd_010 vs per-sample-α baseline

| Corpus | n | SROCC_A | SROCC_B | Δ SROCC | Verdict |
|---|--:|---:|---:|---:|---|
| CID22 | 4292 | 0.6223 | 0.8641 | **-0.2418** | B>>A |
| KADID-10k | 10125 | 0.9277 | 0.9316 | -0.0039 | B>>A |
| TID2013 | 3000 | 0.8879 | 0.8893 | -0.0014 | promising |
| **KonJND-1k** | 1008 | **0.8896** | 0.8080 | **+0.0816** | **A>>B (decisive)** |
| AIC-3 CTC | 600 | 0.7678 | 0.8183 | -0.0505 | B>>A |

### Pareto gate for dense_pjnd_010

| Criterion | Target | dense_pjnd_010 | Pass? |
|---|---|---|---|
| CID22 ≥ V_22 + 0.005 (= 0.8374) | 0.6223 | **FAIL by -0.215** |
| KonJND ≥ 0.883 | 0.8896 | PASS |
| AIC-3 ≥ V_22 + 0.015 (= 0.7995) | 0.7678 | **FAIL by -0.032** |
| KADID within −0.01 of V_22 | 0.9277 | **FAIL by -0.040** |
| TID within −0.01 of V_22 | 0.8879 | **FAIL by -0.085** |

**Pareto gate: FAILED on CID22, AIC-3, KADID, TID.**

### Interpretation — load-bearing finding (PJND-target side)

The PJND-target variant **recovers the KonJND signal to within 0.003 of
V_22** (and beats per-sample-α on KonJND by a decisive +0.082) — proving
that "densification + correct supervision target = KonJND lift is real."

But the recovery comes at the cost of **−0.21 CID22**, **−0.04 KADID**,
**−0.085 TID**. The mechanism: at konjnd_w=0.10 over 20,160 rows
(vs the legacy 0.02 over 1,008 rows), the konjnd group contributes
~10× more loss to the trainer. The MLP over-fits to the
"20-identical-targets-per-source" konjnd encoding (which forces the
model to ignore per-pair signal in favor of per-source aggregation),
and this aggregation behavior generalizes poorly to corpora where
per-pair quality matters (CID22, KADID, TID).

### Why konjnd_w=0.02 didn't work either

Symmetric to the boost case, dense_pjnd_002 (konjnd_w=0.02) UNDER-fits
the konjnd group: training val SROCC plateaus at 0.56, held-out KonJND
SROCC drops to 0.79 (vs V_22's 0.89). The smaller train_w means too
few konjnd pairs in each epoch's 50k pair budget — the model can't
internalize the broadcast-PJND encoding from the limited per-source
sampling rate. Other corpora regress less but still drop (CID22 = 0.72
vs V_22's 0.83).

### Summary of the four trained variants

| Bake | konjnd target | konjnd_w | CID22 | KADID | TID | KonJND | AIC-3 | Pareto |
|---|---|---|---|---|---|---|---|---|
| V_22-mix-LARGE (baseline) | per-source-PJND (1k rows) | 0.02 | **0.8324** | **0.9677** | **0.9729** | 0.8927 | 0.7845 | — |
| per-sample-α (baseline) | per-source-PJND (1k rows) | 0.02 | 0.8641 | 0.9316 | 0.8893 | 0.8080 | 0.8183 | — |
| dense_002 (ssim2 target, 20k) | per-pair-ssim2 | 0.02 | **0.8832** | 0.9330 | 0.8896 | **0.3147** | **0.8215** | FAIL (KonJND) |
| dense_010 (ssim2 target, 20k) | per-pair-ssim2 | 0.10 | **0.8861** | 0.9316 | 0.8896 | **0.4163** | **0.8197** | FAIL (KonJND) |
| dense_pjnd_002 (PJND target, 20k) | per-source-PJND-broadcast | 0.02 | 0.7237 | 0.9301 | 0.8887 | 0.7949 | 0.7745 | FAIL (CID22) |
| dense_pjnd_010 (PJND target, 20k) | per-source-PJND-broadcast | 0.10 | 0.6223 | 0.9277 | 0.8879 | **0.8896** | 0.7678 | FAIL (CID22) |

---

## Conclusion — densification ALONE is not the right lever

Both target encodings produce **single-axis improvements** but neither
holds the Pareto gate:

- **ssim2 targets** (per-pair) → wins CID22 + AIC-3 by 0.05+, but
  KonJND collapses because the model now predicts per-pair quality
  instead of per-source PJND threshold.
- **PJND targets** (per-source broadcast) → recovers KonJND, but the
  broadcast encoding starves per-pair gradient signal and tanks CID22.

The two variants are roughly **CID22 ↔ KonJND tradeoff axes** — neither
gives a single bake that ships.

### What would actually work — for future agents

1. **Dual-target head** — a separate output for per-pair quality
   (trained on safesyn/kadid/tid/cvvdp) AND per-source PJND
   (trained on konjnd). This requires a multi-head MLP architecture
   the current zensim runtime does not support.
2. **KonJND++ acquisition** — the unpublished 300 sources × ~129
   spatial click maps × ~43 PJND ratings encode per-pair JND signal
   directly, eliminating the target-mismatch tension entirely. The
   prior agent's commit `132cddd9` documents the blocked acquisition
   path; emailing Chen/Lin remains the unblocking action.
3. **Augmented per-pair labels** — derive per-pair PJND-distance from
   the 1-pjnd-per-source signal (e.g., `pjnd_dist[i] = sign(q[i] -
   pjnd_threshold)` or a logistic-mapped version). This preserves
   per-pair gradient while anchoring on the human-derived threshold.
   Not attempted in this experiment; reasonable next step.

The user's strategic pivot ("use more JND data, not just more JND
weight") IS valid — densification gives 20× more JND-axis samples —
but a single-target scalar regression cannot encode both per-pair
quality and per-source PJND simultaneously. Multi-target
supervision (or genuinely per-pair PJND labels via KonJND++) is the
unblocker.

---

## Artifacts

- Densified parquets: `/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_features_mix_targets_{300col,372col}.parquet`
- Train logs: `/mnt/v/zen/zensim-eval/v24_konjnd_dense_2026-05-18/persample_dense_konjnd0.{02,10}_seed3.log`
- Bakes: `/mnt/v/zen/zensim-eval/v24_konjnd_dense_2026-05-18/persample_dense_konjnd0.{02,10}_seed3.bin`
- bake_compare reports: `/mnt/v/zen/zensim-eval/v24_konjnd_dense_2026-05-18/cmp_*.md` (6 reports total: dense_{002,010} + dense_pjnd_{002,010} × {V_22, per-sample-α})
- Builder scripts: `scripts/v_next/build_konjnd_dense_parquet.py` (ssim2-target builder + PJND-broadcast variant inline in commit message)
- Train scripts: `scripts/v_next/run_persample_konjnd_dense_seed.sh` (ssim2-target), `scripts/v_next/run_persample_konjnd_dense_pjndtarget_seed.sh` (PJND-broadcast)
