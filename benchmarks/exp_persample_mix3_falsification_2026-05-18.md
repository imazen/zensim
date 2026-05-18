# EXP-PERSAMPLE-MIX3 — falsification (2026-05-18)

**Hypothesis (locked 2026-05-18, EXP-PERSAMPLE-MIX3)**: combine the
two strongest compression-trail directions from 2026-05-18 — the
**per-sample-α head** (V_24, current compression ship) and the
**3-way mix target** (`mix_cv30_iw40_sm30` = 0.3·cvvdp + 0.4·iwssim
+ 0.3·ssim2, all in log-norm score units 0..100) — and beat
V_24-per-sample-α s4 alone on the compression trail.

**Verdict**: FALSIFIED. The combined recipe loses both
{CID22, AIC-3} decisively to per-sample-α s4 alone (the current
compression ship), and the balanced trail fails on KADID + TID.

## 5-seed CI (full Mohammadi panel per corpus, after § A.9 bake_verdict)

Median seed by CID22 SROCC = **seed 1** (CID22 SROCC = 0.8549).

| Corpus | s1 | s2 | s3 | s4 | s5 | mean | median | range | σ |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **CID22** SROCC | 0.8549 | 0.8501 | 0.8403 | 0.8707 | 0.8564 | 0.8545 | 0.8549 | [0.840, 0.871] | 0.0110 |
| **KADID** SROCC | 0.9303 | 0.9306 | 0.9299 | 0.9289 | 0.9303 | 0.9300 | 0.9303 | [0.929, 0.931] | 0.0007 |
| **TID** SROCC   | 0.8784 | 0.8757 | 0.8753 | 0.8780 | 0.8752 | 0.8765 | 0.8757 | [0.875, 0.878] | 0.0016 |
| **KonJND** SROCC| 0.8944 | 0.8959 | 0.9014 | 0.8513 | 0.8828 | 0.8852 | 0.8944 | [0.851, 0.901] | 0.0201 |
| **AIC-3** SROCC | 0.8055 | 0.8056 | 0.8025 | 0.8048 | 0.8079 | 0.8053 | 0.8055 | [0.802, 0.808] | 0.0019 |
| **CID22** PWRC  | 0.9122 | 0.9110 | 0.9046 | 0.9221 | 0.9126 | 0.9125 | 0.9122 | [0.905, 0.922] | — |
| **AIC-3** PWRC  | 0.8788 | 0.8786 | 0.8770 | 0.8785 | 0.8808 | 0.8787 | 0.8786 | [0.877, 0.881] | — |
| **CID22** Z-RMSE| 0.518  | 0.530  | 0.543  | 0.492  | 0.516  | 0.520  | 0.518  | [0.49,0.54]    | — |
| **AIC-3** Z-RMSE| 0.580  | 0.580  | 0.581  | 0.580  | 0.577  | 0.580  | 0.580  | [0.58,0.58]    | — |

Seed variance is concentrated in **CID22** (σ=0.0110) and **KonJND**
(σ=0.0201). KADID / TID / AIC-3 are tight (σ < 0.002). Across-seed
mean CID22 = 0.8545 — below the current compression ship's 0.8641
(Δ ≈ −0.010), and the BEST seed CID22 of 0.8707 (s4) still doesn't
match the existing ship decisively without losing KonJND.

## Pack + drift verification

Median seed (s1) was packed via
`zenpredict repack --dtype i8 --zerobias 0.005 --compress --optimize`.

- Input: 261,316 bytes → packed: 53,824 bytes (20.6% of input)
- 18,175 of 64,000 weights zeroed by zerobias (28.4%)
- **CID22 SROCC drift**: orig 0.8549 → packed 0.8553 = **+0.0004**
  (within 0.0005 pack-quality threshold)

Packed bake retained for falsification record at
`/mnt/v/zen/zensim-eval/exp_persample_mix3_2026-05-18/persample_mix3_s1_h128_packed.bin`.

## Aggregate § A.9 verdicts (1000-bootstrap)

### vs Compression ship (`v_compression_persample_2026-05-18.bin`, V_24-per-sample-α s4)

Source report:
`/mnt/v/zen/zensim-eval/exp_persample_mix3_2026-05-18/compares_vs_compression/persample_mix3_s1_packed_vs_compression.md`.

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | DecScore | A.9 verdict |
|---|--:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8553 | 0.8641 | −19.614 | −40.977 | −0.000 | **B>>A** |
| KADIK10k | 10125 | 0.9304 | 0.9316 | −29.341 | −235.089 | −4.890 | **B>>A** |
| TID2013 | 3000 | 0.8783 | 0.8893 | −90.006 | −451.431 | −0.000 | **B>>A** |
| KonJND-1k (full) | 1008 | 0.8939 | 0.8080 | +40.087 | +118.222 | +33.406 | **A>>B** |
| AIC-3 CTC | 600 | 0.8057 | 0.8183 | −25.665 | −49.102 | −0.000 | **B>>A** |

**Compression-trail gate (per § "Trail definitions" in SOTA_TRAILS.md)**:

1. **Step 1 — decisive on ≥1 of {CID22, AIC-3}**: FAIL.
   CID22 B>>A (h_SROCC = −19.6), AIC-3 B>>A (h_SROCC = −25.7).
   Both compression-targeted corpora go to B.
2. **Step 2 — not B>>A on the other compression corpus**: vacuously
   PASS (step 1 already failed).
3. **Step 3 — synthetic mean Δ ≥ −0.10 on KADID/TID/KonJND**: PASS.
   (KADID Δ=−0.0012, TID Δ=−0.0110, KonJND Δ=+0.0859 — all
   within −0.10 tolerance.)

→ **COMPRESSION GATE: FAIL** (step 1).

### vs Balanced ship (`v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`, V_22-mix-LARGE+iwssim s3)

Source report:
`/mnt/v/zen/zensim-eval/exp_persample_mix3_2026-05-18/compares_vs_balanced/persample_mix3_s1_packed_vs_balanced.md`.

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | DecScore | A.9 verdict |
|---|--:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8553 | 0.8324 | +24.602 | +80.657 | +20.502 | **A>>B** |
| KADIK10k | 10125 | 0.9304 | 0.9677 | −86.885 | −750.455 | −0.000 | **B>>A** |
| TID2013 | 3000 | 0.8783 | 0.9729 | −54.371 | −292.908 | −0.000 | **B>>A** |
| KonJND-1k (full) | 1008 | 0.8939 | 0.8927 | +0.801  | +1.341  | +0.000 | tied |
| AIC-3 CTC | 600 | 0.8057 | 0.7845 | +20.800 | +43.507 | +17.333 | **A>>B** |

**Balanced-trail gate**:

1. **Step 1 — A>>B on CID22 decisively**: PASS (h_SROCC = +24.6,
   panel-decisive).
2. **Step 2 — no decisive B>>A on any of {KADID, TID, KonJND, AIC-3}**:
   FAIL on KADID (h_SROCC = −86.9) AND TID (h_SROCC = −54.4) — both
   are decisive losses by wide margin.

→ **BALANCED GATE: FAIL** (step 2; two-corpus loss).

## Final decision

→ **FALSIFY.** Neither trail gate passes. The combined recipe is
strictly dominated:

- On the compression trail it loses CID22 AND AIC-3 to V_24-per-
  sample-α s4 alone, while gaining only KonJND (which is not a
  compression-trail-relevant corpus).
- On the balanced trail it wins CID22 + AIC-3 vs V_22-mix-LARGE+
  iwssim but loses KADID + TID by wide margins, blocking the
  balanced gate's noise-strict step 2.

The packed bake at
`/mnt/v/zen/zensim-eval/exp_persample_mix3_2026-05-18/persample_mix3_s1_h128_packed.bin`
is retained as falsification record. **No ship rotation.**

## Mechanism analysis

Two mechanisms appear to drive the falsification.

**1. Adding ssim2 to the target degrades CID22.** The 3-way
`mix_cv30_iw40_sm30` target moves the supervision away from
cvvdp-dominant content (40 % iwssim + 30 % ssim2). The current
compression ship (per-sample-α s4) trained on `mix_cv40_iw60`
(cvvdp 0.4, iwssim 0.6, NO ssim2) and scored CID22 SROCC 0.8641 with
mean across seeds 1–5. Adding ssim2 to the target pulls toward
ssim2-shaped predictions; CID22 favors cvvdp + iwssim-shaped output.
Net result: median CID22 drops from 0.8641 → 0.8549 (Δ −0.0092).

This is consistent with the EX-MIX3 round-1 result documented at
`benchmarks/bake_compare_exmix3_vs_persample.md` (without per-
sample-α): mix3 ties CID22 (0.8642 vs 0.8641) but loses KADID +
TID + AIC-3 vs the persample ship. The current round-2 (with
per-sample-α) does not recover the CID22 win that the existing
per-sample-α ship has.

**2. KonJND gain is real but irrelevant to either gate.** Adding
ssim2 to the target (which correlates strongly with KonJND PJND
anchors per CID22 Table 4) lifts KonJND SROCC dramatically:
0.8944 vs 0.8080 (Δ +0.0859 across the bake's full surface, with
h_SROCC = +40.087 decisive). But:

- Compression trail doesn't gate on KonJND (step 3 is a noise
  tolerance not a win requirement).
- Balanced trail's KonJND ties (h_SROCC = +0.801, not decisive vs
  the existing balanced ship).

So the only place this recipe wins decisively is on a corpus that
neither trail rewards. This is exactly the EX-MIX3 round-1
finding repeating — see
`bake_compare_exmix3_vs_persample.md` aggregate row:
"KonJND-1k (full) | … | A>>B" with the existing per-sample-α ship.

**3. KADID + TID stay non-compression-shaped.** Adding 30 % ssim2
to the target weights ranking against ssim2's per-distortion
profile. KADID + TID are dominated by non-compression
distortions; their best metric is the trained-against-ssim2-only
balanced ship (V_22-mix-LARGE+iwssim at 0.9677 / 0.9729). The
3-way mix target can't reach that because cvvdp + iwssim only
contribute 70 % of supervision; the remaining 30 % ssim2 is
diluted by the iwssim and cvvdp contributions that drag KADID +
TID rankings toward compression-shaped output.

## What this rules out

- Adding ssim2 to the per-sample-α target column on the compression
  trail. Combining mix3 with per-sample-α does NOT compound the
  two compression-direction wins. Further mix-target work should
  cap ssim2 weight at ≤ 0.10 if attempted, or drop ssim2 entirely
  in favor of cvvdp + iwssim only (V_24-per-sample-α with
  `mix_cv40_iw60` remains the best known compression-trail recipe).
- This is also a falsification of the implicit "compression
  directions compose" assumption. Two independent compression
  wins (per-sample-α head + mix3 3-way target) do not
  multiplicatively beat each component alone — they trade off.

## What remains unexplored

- **Seed 4 alone has CID22 SROCC 0.8707**, which beats the current
  ship's median seed by ~0.007. Per the median-seed pick rule
  (avoid overfit to one seed), s4 is NOT the canonical ship
  candidate. A fresh 5-seed run with a different target column
  (e.g., `mix_cv50_iw50` or `mix_cv40_iw40_sm20`) might recover the
  CID22 win with less KADID/TID damage. Not pursued here.
- **α weight in mix target**: a sweep over the ssim2 contribution
  (0 / 0.05 / 0.10 / 0.20 / 0.30) would tell us whether the ssim2
  damage is dose-dependent. Not pursued here.

## Source data

All commits land on `main`:

- Bake (median seed 1, packed): `persample_mix3_s1_h128_packed.bin`
  (53,824 bytes, md5 `7f125de04923eb8ca190ad10ecfd32e7`)
- Bake (median seed 1, unpacked): `persample_mix3_s1_h128.bin`
  (261,316 bytes, md5 `9b34b873ed7d9bbb55ce5c71556f80bf`)
- 5-seed verdicts: `/mnt/v/zen/zensim-eval/exp_persample_mix3_2026-05-18/verdicts/persample_mix3_s{1..5}.md`
- vs Compression compare: `compares_vs_compression/persample_mix3_s1_packed_vs_compression.md`
- vs Balanced compare:    `compares_vs_balanced/persample_mix3_s1_packed_vs_balanced.md`
- Driver log: `/tmp/exp_persample_mix3_continuation_20260518T194740Z.log`
- Training data: `/mnt/v/zen/zensim-training/2026-05-18-mix3/{safesyn,kadid,tid,konjnd}.parquet`

## Training reproducibility

```
zensim_mlp_train
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-18-mix3/safesyn.parquet:1.0:0.0
  --group kadid:/mnt/v/zen/zensim-training/2026-05-18-mix3/kadid.parquet:0.3:1.0
  --group tid:/mnt/v/zen/zensim-training/2026-05-18-mix3/tid.parquet:0.3:1.0
  --group konjnd:/mnt/v/zen/zensim-training/2026-05-18-mix3/konjnd.parquet:0.02:1.0
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5
  --leaky-alpha 0.01 --val-policy min --early-stop-patience 0
  --max-features 372 --minibatch-size 256
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0
  --target-column mix_cv30_iw40_sm30 --target-scale 100.0 --out-dtype f32
  --per-sample-alpha-head
  --seed {1..5}
```

Bake on s1 packed at `2026-05-18T19:59:00Z` after eval pipeline (i8 +
zerobias 0.005 + lz4) confirmed +0.0004 SROCC drift vs unpacked.
