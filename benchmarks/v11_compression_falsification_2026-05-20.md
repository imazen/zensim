# V11-B Compression-trail ship — FALSIFIED on all 3 gate criteria (task #191, 2026-05-20)

**Status: FALSIFIED. V11-clean candidate bakes fail every Compression-trail gate
criterion against the correct V_24-per-sample-α s4 baseline. Ship decision: NO.**

The V11-SUBSTRATE-V2 work (task #189, commits 9223706 + 84ce339) produced 5 candidate
bakes claiming to recover from the prior V11-substrate-v1 ship gate failure by switching
to the V_24 per-sample-α head + V11 substrate + V_24 hparams recipe (NOT the previous
falsified brief recipe). The prior agent confirmed Balanced-trail gate failure on
KonJND (-0.49 vs Balanced baseline). This re-eval asks whether the same candidates
PASS on the Compression-trail's looser gate.

**Answer: no, and the failure is decisive on all three gate criteria.**

## Baseline-identification correction vs the task brief

The task brief cited V_24 s4 Compression baseline CID22 0.8641 / AIC-3 0.8183 /
KonJND 0.8080. Those numbers correspond to `v_compression_persample_2026-05-18.bin`
(md5 `f09a9abdce00805000c1d112c2421b2d`), which is the **actual current
Compression ship** (per `zensim/CLAUDE.md` § Three-trail SOTA).

The brief-referenced "V_24 s4 Compression baseline" at
`v_compression_2026-05-18.bin` (md5 `3be4f781238dcb35f32c964cb218a8a4`)
is in fact the historical V_22-372feat s5 bake (per `zensim/SOTA_TRAILS.md` line
541), kept in `weights/` for reproducibility per the matrix note. This bake
scores CID22 0.8580 on the same parquet root — close to but distinct from
the V_24 s4 baseline.

This doc evaluates V11-clean against the **correct V_24-per-sample-α s4
baseline** (`v_compression_persample_2026-05-18.bin`), which is what the
Compression-trail gate is anchored against.

## Median candidate selection

5 V11-clean candidates at
`/mnt/v/zen/zensim-eval/exp_v11_balanced_v2_clean_2026-05-20/cc4v11a_v2clean_s{1..5}.bin`.

Sort by CID22 SROCC ascending (median selection methodology per V6-RESHIP):

| Seed | CID22 SROCC | KADID | TID | KonJND | AIC-3 | AIC-4 |
|---|---:|---:|---:|---:|---:|---:|
| s2 | 0.8683 | 0.9136 | 0.8968 | 0.4033 | 0.7721 | 0.8730 |
| s5 | 0.8698 | 0.9214 | 0.8934 | 0.3280 | 0.7976 | 0.9060 |
| **s3** | **0.8754** | **0.9196** | **0.8921** | **0.4627** | **0.7943** | **0.8907** |
| s1 | 0.8812 | 0.9225 | 0.8878 | 0.2874 | 0.8229 | 0.9289 |
| s4 | 0.8899 | 0.9221 | 0.8882 | 0.4392 | 0.8013 | 0.9086 |

Median by CID22 SROCC: **s3 (0.8754)**. The KonJND SROCC ranges 0.29..0.46 across all
5 seeds — every seed catastrophically below the 0.81 V_24 baseline. KonJND collapse
is structural, not seed-dependent.

## Full Mohammadi panel vs V_24-per-sample-α s4 Compression ship

Baseline: `zensim/weights/v_compression_persample_2026-05-18.bin` (the actual
shipped Compression bake, per CLAUDE.md). Candidate: `cc4v11a_v2clean_s3.bin`.
Both evaluated via `bake_verdict --features-root /mnt/v/zen/zensim-training/
2026-05-15-full-features` — same parquet root, apples-to-apples.

### Aggregate SROCC

| Corpus | V_24 s4 baseline | V11-clean s3 | Δ |
|---|---:|---:|---:|
| CID22 | 0.8641 | 0.8754 | **+0.0113** |
| KADIK10k | 0.9316 | 0.9196 | -0.0120 |
| TID2013 | 0.8893 | 0.8921 | +0.0028 |
| KonJND-1k | 0.8080 | 0.4627 | **-0.3453** |
| AIC-3 CTC | 0.8183 | 0.7943 | **-0.0240** |
| AIC-4 sample | 0.9538 | 0.8907 | -0.0631 |

### Aggregate PWRC (Pearson-weighted rank correlation)

| Corpus | V_24 s4 baseline | V11-clean s3 | Δ |
|---|---:|---:|---:|
| CID22 | 0.9157 | 0.9239 | +0.0082 |
| KADIK10k | 0.9602 | 0.9518 | -0.0084 |
| TID2013 | 0.9173 | 0.9194 | +0.0021 |
| KonJND-1k | 0.8505 | 0.6015 | **-0.2490** |
| AIC-3 CTC | 0.8856 | 0.8693 | -0.0163 |
| AIC-4 sample | 0.9766 | 0.9326 | -0.0440 |

### Aggregate Z-RMSE (lower = better calibration)

| Corpus | V_24 s4 baseline | V11-clean s3 | Δ |
|---|---:|---:|---:|
| CID22 | 0.508 | 0.481 | -0.027 (better) |
| KADIK10k | 0.362 | 0.392 | +0.030 |
| TID2013 | 0.432 | 0.436 | +0.004 |
| KonJND-1k | 0.502 | 0.912 | **+0.410** |
| AIC-3 CTC | 0.565 | 0.591 | +0.026 |
| AIC-4 sample | 0.309 | 0.484 | +0.175 |

Mohammadi triangulation:
- **CID22**: SROCC +0.0113, PWRC +0.0082, Z-RMSE -0.027 — all directionally A,
  but no single stat clears its decisive cut (SROCC +0.015, PWRC +0.010, Z-RMSE
  -0.030). **Promising-A, NOT decisive A>>B per § A.9.**
- **AIC-3**: SROCC -0.0240, PWRC -0.0163, Z-RMSE +0.026 — all over the
  decisive-B cut (-0.015 SROCC, -0.010 PWRC, +0.020 Z-RMSE). **Decisive B>>A.**
- **KonJND**: SROCC -0.3453, PWRC -0.2490, Z-RMSE +0.410 — catastrophic
  collapse in all three dimensions.

## Compression-trail gate scorecard

Per `zensim/CLAUDE.md` § Three-trail SOTA and `zensim/SOTA_TRAILS.md` — the
Compression-trail gate criteria:

> A>>B on ≥1 of {CID22, AIC-3} decisively per § A.9 AND not decisively B>>A on the
> other compression corpus AND mean SROCC regression on {KADID, TID, KonJND} no
> worse than −0.10 on any single corpus.

| Criterion | V11-clean s3 result | Verdict |
|---|---|---|
| Step 1: A>>B on CID22 or AIC-3 decisively | CID22 ΔSROCC +0.0113 (under +0.015 decisive cut), PWRC +0.0082 (under +0.010 cut), Z-RMSE -0.027 (under -0.030 cut). AIC-3 is decisively B>>A (see step 2). **No decisive A>>B on either compression corpus.** | **FAIL** |
| Step 2: NOT decisively B>>A on other compression corpus | AIC-3 ΔSROCC -0.0240 (over -0.015 decisive-B cut), PWRC -0.0163 (over -0.010 cut), Z-RMSE +0.026 (over +0.020 cut). All three stats agree on decisive B>>A. | **FAIL** |
| Step 3: KADID regression ≥ -0.10 | -0.0120 | PASS |
| Step 3: TID regression ≥ -0.10 | +0.0028 | PASS |
| Step 3: **KonJND regression ≥ -0.10** | **-0.3453** | **FAIL (3.45× over cap)** |

**Overall gate verdict: 3-of-3 criteria FAIL.** Of the three Compression-trail
criteria (decisive win, no decisive loss on other compression corpus, ≤ -0.10 on
each of {KADID, TID, KonJND}), V11-clean fails all three. Steps 1 and 2 fail
because the CID22 +0.011 lift does NOT outweigh the AIC-3 -0.024 regression
within the Compression-trail's structural constraint. Step 3 fails on KonJND
by 3.45× the cap.

## Why the failure is structural

V11-clean was trained on the V11 multi-codec ssim2-anchored substrate (per the V11
methodology doc `benchmarks/v11_methodology_2026-05-20.md`). The 8,893-row anchor
parquet covers zenjpeg/webp/avif/jxl across 10 score bands via cvvdp→ssim2
conversion (with documented saturation issues at high q for zenwebp + zenjxl).

KonJND-1k is the only validation corpus that anchors on **PJND thresholds**, not
0..1 normalized MOS or DMOS. A bake trained against ssim2/cvvdp-derived score
targets will not, by construction, defend PJND threshold ordering unless the
training mix explicitly includes a PJND-anchored term — which V11-clean does not.

The 2026-05-18 V_24 Compression-trail ship handles KonJND via the `mix_cv25..cv75`
target column blends + KonJND-dense training subset (canonical-2026-05-18/train/
konjnd-dense.parquet). The V11-substrate recipe drops both.

The +0.011 CID22 SROCC and -0.024 AIC-3 SROCC pattern (lift on the multi-codec-
heavy CID22, regression on the single-codec JPEG-AIC-3) suggests the multi-codec
ssim2-anchor substrate biases the bake toward content-class generalization at the
expense of within-codec rank discrimination. Per `benchmarks/v11_methodology_
2026-05-20.md`'s recipe-limit findings, this trade-off appears intrinsic to the
ssim2-anchored-multi-codec substrate without a PJND term.

## Ship decision

**No ship.** V_24-per-sample-α s4 (`v_compression_persample_2026-05-18.bin`)
remains the shipped bake for `ZensimProfile::PreviewV0_5Compression`,
`PreviewV0_5CompressionV2` (V_24 + V10 calibration spline), and
`PreviewV0_5CompressionV3` (V_24 + V10 reallocated-spline).

V11 substrate work yielded:

- A multi-codec ssim2-anchored anchor parquet (8,893 rows × 311 cols).
- A documented set of cvvdp→ssim2 + ssim2-pivoted cross-codec equivalence pairs.
- Empirical confirmation that the V_24-per-sample-α head + V11-substrate +
  V_24-hparams recipe FAILS the Compression-trail gate on all three criteria
  against the current ship (not just the previously-reported KonJND collapse).
- A negative result documenting that PJND-anchor training data + AIC-3-class
  discrimination MUST be in the mix for any V11-substrate-derived bake to clear
  the Compression-trail gate.

No shippable bake, no new profile variant, no calibration spline refit. V_24 s4
holds the trail.

## Reproducibility

Worktree: `/home/lilith/work/zen/zensim--cross-codec-v8/`.

Commands run:
```
cargo build --release --bin bake_verdict -p zensim-validate
./target/release/bake_verdict --bake zensim/weights/v_compression_persample_2026-05-18.bin \
    --output /tmp/v_compression_v24_baseline_verdict.md
for s in 1 2 3 4 5; do
  ./target/release/bake_verdict \
      --bake /mnt/v/zen/zensim-eval/exp_v11_balanced_v2_clean_2026-05-20/cc4v11a_v2clean_s${s}.bin \
      --output /tmp/v11clean_s${s}_verdict.md
done
```

Verdict markdowns (ephemeral; numbers reproduced in this doc):
`/tmp/v_compression_v24_baseline_verdict.md`, `/tmp/v11clean_s{1..5}_verdict.md`.

Bakes: `/mnt/v/zen/zensim-eval/exp_v11_balanced_v2_clean_2026-05-20/
cc4v11a_v2clean_s{1..5}.bin` (preserved per CLAUDE.md "Never Delete Caches or
Generated Data").

Baseline bake: `zensim/weights/v_compression_persample_2026-05-18.bin`
(md5 `f09a9abdce00805000c1d112c2421b2d`, V_24-per-sample-α s4, the actual
shipped Compression bake per zensim/CLAUDE.md § Three-trail SOTA).

Refs: task #191. Prior V11 falsification: task #189, commits 9223706 + 84ce339,
`benchmarks/v11_methodology_2026-05-20.md`.
