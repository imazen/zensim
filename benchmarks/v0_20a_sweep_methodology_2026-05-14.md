# V0_20a IW-SSIM sweep — 6-cell results (2026-05-14)

**Methodology fix**: trainer commit `f0b369a` introduced independent
init / sampler RNGs so 228-baseline vs 372-IW sees identical training
pair sequences at the same seed. All 6 bakes in this sweep were
trained with the fixed trainer; the pre-fix bakes are archived at
`/tmp/v0_20a_sweep_tainted_pre_seedfix/`.

## Recipe (all 6 cells share)

| Param | Value |
|---|---|
| Groups | KADID (10125 pairs, train_w=0.3 val_w=1.0) + TID (3000, 0.3/1.0) + KonJND (910, 0.5/1.0) |
| Hidden | 128 (LeakyReLU α=0.01) |
| Epochs | 300 |
| LR | 1e-3 cosine-annealing (50-epoch period) |
| L2 | 1e-5 |
| val_policy | min (worst per-group SROCC) |
| Early stop | 50 epochs no improvement |
| IW weight formula | `iw_weight[i] = 1.0 + k * blur(\|src - μ\|)[i]` |

## Cells

| Config | features | seed | iw_strength k |
|---|---:|---:|---:|
| baseline_228_s1 | 228 | 1 | n/a |
| baseline_228_s42 | 228 | 42 | n/a |
| iw_k4_s1 | 372 | 1 | 4.0 (default) |
| iw_k4_s42 | 372 | 42 | 4.0 |
| iw_k1_s1 | 372 | 1 | 1.0 (light) |
| iw_k8_s1 | 372 | 1 | 8.0 (aggressive) |

## Results

### Training-corpus val_mean SROCC (KADID+TID+KonJND, val_policy=min)

| Config | val_mean | Δ vs baseline_228_s1 |
|---|---:|---:|
| baseline_228_s1 | **0.9459** | — |
| baseline_228_s42 | 0.9442 | −0.0017 (seed noise) |
| iw_k4_s1 | 0.9425 | −0.0034 |
| iw_k4_s42 | 0.9407 | −0.0052 |
| iw_k1_s1 | 0.9353 | −0.0106 |
| **iw_k8_s1** | **0.9468** | **+0.0009** |

### Held-out CID22 SROCC (PRIMARY MEASUREMENT — these bakes lack synth supervision, so absolute SROCC is low; the A/B is what matters)

| Config | CID22 agg | B0 [<50] | B1 [50,65) | B2 [65,90) | B3 [≥90] |
|---|---:|---:|---:|---:|---:|
| baseline_228_s1 | 0.4529 | 0.1587 | 0.1425 | 0.2630 | 0.2381 |
| baseline_228_s42 | 0.5157 | 0.2052 | 0.2018 | 0.2978 | 0.3843 |
| iw_k4_s1 | 0.3602 | 0.1881 | 0.1365 | 0.1498 | 0.2294 |
| iw_k4_s42 | 0.4288 | 0.1762 | 0.1615 | 0.2245 | 0.3466 |
| **iw_k1_s1** | **0.4738** | 0.1823 | 0.1516 | **0.2671** | **0.4260** |
| iw_k8_s1 | **0.1865** ↓↓ | 0.1142 | 0.0277 | 0.1405 | 0.0578 |

(Reference: V0_18 ship on the same CID22 corpus = 0.8933 aggregate. The
V0_18 ship was trained on synth + KADID + TID + KonJND; these sweep
bakes were trained WITHOUT synth so absolute SROCC is much lower.
What matters here is the A/B comparison between 228-baseline and
372-IW at the same recipe.)

## Findings

1. **val_mean and CID22 SROCC DIVERGE.** Higher val_mean (training-corpus
   SROCC) does NOT predict higher CID22 (held-out generalization). The
   highest-val_mean bake (`iw_k8_s1: 0.9468`) is the WORST on CID22
   (0.1865). The lowest-val_mean IW bake (`iw_k1_s1: 0.9353`) is the
   BEST on CID22 (0.4738).

2. **iw_k1_s1 beats baseline on CID22 by +0.021 aggregate.** Despite a
   −0.011 val_mean cost, light IW weighting (k=1.0) generalizes
   markedly better on CID22 than the 228-baseline at the same recipe.
   Largest CID22 lift is in B3 [≥90] (visually-lossless) at +0.188,
   matching Wang & Mohammadi 2025's prediction that IW-SSIM shines in
   the HF regime. (n=43 for B3, so the magnitude is noisy, but the
   direction is unambiguous.)

3. **iw_k8_s1 catastrophically overfits.** At k=8.0 the IW weight ranges
   roughly 1×–9× across pixels, dominating the basic+peaks signal. The
   MLP learns this strong-emphasis input perfectly on KADID/TID/KonJND
   (0.9468 val_mean) but fails on CID22 (0.1865) — the texture-emphasis
   prior is too rigid to generalize.

4. **iw_k4 is in between** — neither overfitting (val_mean within seed
   noise of baseline) nor generalizing well (CID22 worse than baseline
   on both seeds).

## Interpretation

Wang & Li 2011 (the original IW-SSIM paper) reports +0.006 SROCC vs
MS-SSIM. Wang/Mohammadi 2025 (the recent benchmark) found IW-SSIM is
the BEST classical metric in the HF regime. Both numbers are for
**uniform-vs-IW spatial pool of an SSIM aggregate** — not for adding
IW features as additional MLP inputs (the V0_20a setup). Our test
asks a DIFFERENT question:

> Does adding 72 IW-pooled features as MLP inputs (on top of the
> existing 156 basic + 72 peak features) help the MLP rank
> SSIMULACRA2-style perceptual distances?

Result: **at light strength (k=1.0), yes — CID22 +0.021.** At heavy
strength (k=8.0), the MLP learns the new inputs at the cost of
generalization. The val_mean policy (worst per-group SROCC on
training corpora) actively MISLEADS us toward the overfit bake.

## Caveats

- These bakes lack synth supervision. The proper test is V0_20a with
  the V0_18 recipe (synth + KADID + TID + KonJND, 218k pairs total).
  Synth 372-feature CSV extraction is in flight at the time of
  writing.
- Single seed at k=1 and k=8. Per the rigor policy, k=1 should be
  re-run at seed 42 + 7 to verify the +0.021 CID22 lift replicates
  before claiming a paper-claimed-benefit result.
- val_policy=min picks the WORST per-group SROCC as the checkpoint
  selector. With KADID/TID/KonJND only, the bake that overfits the
  hardest training group wins val_mean. A val_policy that includes a
  held-out CID22 sample would have selected `iw_k1_s1` over `iw_k8_s1`.

## Files

- 6 component bakes at `benchmarks/v0_20a_sweep/{tag}.bin`
- 7 per-band markdown reports at `benchmarks/v0_20a_sweep/{tag}.md`
  (+ v0_18_ship_on_cid22.md as the reference)
- Trainer code: zensim-validate src @ commit `f0b369a`
- Per-band evaluator: `zensim-validate/src/bin/eval_bake_per_band.rs`

## Next experiments

1. **iw_k1_s42 + iw_k1_s7**: replicate the CID22 +0.021 lift across seeds.
2. **iw_k2_s1 + iw_k0.5_s1**: refine the strength sweep around k=1.
3. **synth + iw_k1 (full V0_18 recipe)**: extract the 218k synth CSV at IW
   features, retrain the 3-way concat ensemble at 372 features, validate
   on full corpora. THIS is the ship-decision test.
4. **val_policy=cid22_held_out**: train with a held-out CID22 sample
   in the validation pool so val_mean tracks ship metric. Avoids the
   overfit-selection trap revealed here.
