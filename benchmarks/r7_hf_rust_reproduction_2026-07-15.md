# Round-7 HF near-lossless — Rust-trainer reproduction (2026-07-15)

Answers the standing question *"did you try a bake including it using the
optimal pairwise/per-ref"* and the directive *"ensure key bakes are
reproducible with the rust paths and update rust when needed to do so."*

**Verdict: the FINDING reproduces; one number does not, and the cause is a
missing Rust feature, not a disagreement.**

## What was run

| | Python round-7 (`blend_search.py` ROUND7) | Rust (`zensim_mlp_train`) |
|---|---|---|
| main groups | `safesyn 1, bigcodec 1.5, kadis 0.3` (`_HON`) | same |
| HF group | `hf_rank_weight 0.1`, within-ref RankNet | `hf_nearlossless:…:0.1:0.0:withinref` |
| arch | 2-layer, H=128 | 2-layer, H=128 (`--n-hidden-layers 2 --hidden 128`) |
| **main-group loss** | **`smooth_l1_loss` (Huber — absolute target)** | **RankNet (`mse_weight: 0.0`)** |
| **seeds** | **2 (1, 7), averaged** | **1** |
| epochs | 400 full-batch | 120 × 50k sampled pairs |

Rust command (from `/mnt/v/output/zensim/r7_rust/train_hf01.log`):

```
zensim_mlp_train --out r7_hf01.bin \
  --group safesyn:canonical-2026-05-21/train/safesyn.parquet:1.0:0.0 \
  --group bigcodec:bigcodec_hqdedup_traindigits_2026-07-02.parquet:1.5:0.0 \
  --group kadis:canonical-2026-07-15/train/kadis_negrich.parquet:0.3:0.0 \
  --group hf_nearlossless:canonical-2026-07-15/train/hf_nearlossless_train.parquet:0.1:0.0:withinref \
  --group cid22_val:canonical-2026-05-21/train/cid22_train.parquet:0.0:1.0 \
  --target-column human_score --target-scale 100 \
  --hidden 128 --n-hidden-layers 2 --epochs 120 --seed 1
```

`hf0` is the identical command with the `hf_nearlossless` group removed.

## Split integrity (checked, not assumed)

`hf_nearlossless_train` = 900 rows / **150 refs**; `hf_nearlossless_val` =
300 rows / **50 refs**; `train ∩ val = 0 refs`; `train ∪ val` is exactly the
200-ref parent. The held-out readout is honest.

Per-ref n = **6** for every held-out ref (a 6-rung ladder). SROCC on 6 points
is coarse — read `%bwd` (share of refs ranked backwards) as the primary, since
it is robust to that.

## Result — the HF axis

| HF near-lossless, 50 held-out refs | hf0 | hf0.1 |
|---|---:|---:|
| **per-ref SROCC** | +0.2539 | **+0.9500** |
| **% refs ranked BACKWARDS** | **35%** | **0%** |
| PLCC | 0.3184 | 0.6275 |
| PWRC | 0.6510 | 0.8241 |
| Z-RMSE | 0.948 | 0.779 |
| DS-AUC | 0.5073 | 0.5842 |
| pooled SROCC | 0.2093 | 0.2810 |

A 900-pair group at weight 0.1 takes held-out near-lossless ordering from *a
third of refs backwards* to *none*. The pooled-vs-per-ref gap (0.281 vs 0.950)
is the documented within-vs-cross-image confound — the HF ladder moves ~0.92
ssim2 pts within an image vs ~6 pts between images, so the pooled number is
carried by cross-image scale. It is not a defect, and it is exactly why the
group is consumed rank-only and within-ref.

## Result — the cost

| Δ (hf0 → hf0.1) | Python r7 | Rust r7 | |
|---|---:|---:|---|
| CID22 | −0.0041 | **−0.0047** | reproduces (within 0.0006) |
| non-photo | −0.0017 | **−0.0016** | reproduces (within 0.0001) |
| KonJND | **+0.0334** | **−0.0349** | **does NOT reproduce** |

Absolute values — Python `hf0`: CID22 0.8862, non-photo 0.9549, KonJND 0.5187;
`hf0.1`: CID22 0.8821, non-photo 0.9532, KonJND 0.5521. Rust `hf0`: CID22
0.8842, non-photo 0.9669, KonJND 0.5024; `hf0.1`: CID22 0.8795, non-photo
0.9653, KonJND 0.4675.

Two of the three cost axes reproduce to within 0.0006 across two *different*
training objectives. KonJND is the lone outlier — which is what points at the
objective, below, rather than at the HF group.

## Why KonJND does not reproduce — a missing feature, not a contradiction

The two runs **optimize different objectives.** `mse_weight` in the Rust
trainer is wired **only inside `train_mlp_per_sample_alpha_head`**
(`zensim-validate/src/mlp_train/mod.rs:5698` panics when `mse_weight > 0 &&
!per_sample_alpha_head`). This run used the standard `train_mlp` path, which
is **RankNet-only by construction**. So:

- Python: `huber(main groups) + 0.1 · ranknet(HF)` — main groups carry an
  absolute target; the HF term is scale-free and, per its own comment,
  *"cannot drag the absolute dial of the MSE groups."*
- Rust: `ranknet(everything)` — there is no absolute dial to protect.

KonJND is the corpus most sensitive to absolute calibration (its `human_score`
is a PJND threshold, not a ladder position). A pure-rank objective having a
different effect on it than a Huber+rank objective is the expected outcome,
not a disagreement between implementations. The CID22 delta agreeing to 0.0006
across two different objectives is the more surprising half.

The Rust run is also **1 seed** against Python's 2-seed average, and KonJND
seed variance is known to be large (v48 multi-seed: 2/9 runs collapse), so
±0.03 on a single seed is not separable from noise regardless.

**Conclusion: do not treat `KonJND −0.035` as a measurement of the HF group's
effect.** It measures a rank-only objective. The HF finding itself (per-ref,
%bwd, CID22) reproduces.

## Honest scope of the HF win

The 50 held-out refs are held out **by reference**, but share the corpus's
distortion family and its 6-rung ladder design. So the result says: *the model
generalizes this ladder's ordering to unseen images.* It does **not** show that
near-lossless ordering improved anywhere else — CID22/KonJND/non-photo all moved
slightly down or sideways. No broad near-lossless transfer is claimed or
measured here.

Measuring transfer needs a near-lossless-specific readout on the other corpora.
`blend_lib.panel` has one (`srocc_hightail` / `srocc_lowtail`, added
2026-07-15); `zenstats` and `bake_verdict` do **not**. That is a Python-only
stat which, per the standing directive, belongs in `zenstats` — queued.

Neither bake is a ship candidate: both are raw rank output with no calibration
spline (G1 dynamic range 0.00, p5=−16.1 p95=2.5). They are research bakes.

## Gap to close

`within_ref` is a **per-group** modifier; `mse_weight` is a **global**
hyperparameter that the plain path rejects outright. So the Python r7 recipe —
*MSE on the main groups, rank-only on HF* — is **not currently expressible in
the Rust trainer.** Closing that is the per-group-loss-mode work; until it
lands, a Rust/Python comparison on any absolute-dial corpus is apples to
oranges.

## Reproduce

```
cargo build --release --bin bake_verdict -p zensim-validate
./target/release/bake_verdict --bake /mnt/v/output/zensim/r7_rust/r7_hf01.bin \
  --corpora hf_nearlossless,cid22,konjnd,nonphoto \
  --output /mnt/v/output/zensim/r7_rust/v2_hf01.md
```

Both arms must be scored by the **same** `bake_verdict` build — the binary that
auto-emitted `r7_hf01.verdict.md` at train time predates the `Orientation` fix
and the `hf_nearlossless` corpus, so its per-ref column is not comparable.

- Bakes: `/mnt/v/output/zensim/r7_rust/r7_hf{0,01}.bin`
- Panels: `/mnt/v/output/zensim/r7_rust/v2_hf{0,01}.md`
- Train logs: `/mnt/v/output/zensim/r7_rust/train_hf{0,01}.log`
- Python reference: `benchmarks/blend_search_r7_2026-07-15.tsv`
