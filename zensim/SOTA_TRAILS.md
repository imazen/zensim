# Two-trail SOTA tracker

zensim ships TWO PreviewV0_5 variants in parallel, each defending a
different Pareto frontier. This doc is the source of truth for what
ships on each trail, what's been falsified against each trail's gate,
and the gate criteria themselves.

**Read this before training the next bake.** A new bake is shippable
on a trail iff it Pareto-beats that trail's current ship under that
trail's gate.

---

## Trail definitions

### Balanced trail (`PreviewV0_5Balanced` — historical alias `PreviewV0_5`)

**Audience.** Code that uses zensim as a general-purpose perceptual
metric across many distortion families — synthetic noise/blur,
geometric distortions, JND thresholds, compression artifacts.

**Gate** (formal):

| Corpus | Direction | Decisive rule |
|---|---|---|
| CID22  | A ≥ B   | § A.9 (n≥30 ∧ |h_SROCC|>1.96 ∧ |h_Z-RMSE|>1.96 ∧ PWRC_A>PWRC_B ∧ ≥4/6 panel stats favor A) |
| KADID  | A ≥ B   | not decisively B>>A on aggregate |
| TID    | A ≥ B   | not decisively B>>A on aggregate |
| KonJND | A ≥ B   | not decisively B>>A on aggregate |
| AIC-3  | A ≥ B   | not decisively B>>A on aggregate |

A ship requires **A>>B on ≥1 corpus and tied/A-favor on the other 4**.
Any single decisive B>>A on a corpus is a ship blocker.

### Compression trail (`PreviewV0_5Compression`)

**Audience.** Imageflow / commercial web compression pipelines where
the metric ranks compressed-image quality. CLAUDE.md: "Imageflow is
web-focused, not archival — commercial web compression targets
aggressive settings where every byte matters." CID22 is human MOS on
codec output; AIC-3 is human JND on near-PJND codec output. Both
score compression directly; the other 3 corpora score adjacent
non-compression behavior that's relevant but secondary.

**Gate** (formal):

1. **Decisive on at least one of {CID22, AIC-3}** per § A.9.
2. **Not decisively losing the other compression corpus** (CID22 or
   AIC-3, whichever wasn't the decisive winner).
3. **Mean SROCC regression on {KADID, TID, KonJND} no worse than
   −0.10 on any single corpus.** A −0.05 to −0.10 regression on
   synthetic / JND corpora is tolerated because they don't score
   compression directly.

A ship requires steps 1–3 ALL pass. Step 3 is the noise-tolerance
exception; without it the compression trail would collapse into the
balanced trail.

---

## Current ship per trail (2026-05-18)

| Trail | Bake | Bytes | Architecture | CID22 | AIC-3 | KADID | TID | KonJND |
|---|---|--:|---|---:|---:|---:|---:|---:|
| **Balanced** | V_22-mix-LARGE+iwssim s3 packed | 41,695 | 300→128→1 vanilla MLP | 0.8324 | 0.7845 | **0.9677** | **0.9729** | **0.8927** |
| **Compression** | V_24-per-sample-α s4 packed | 44,109 | 300→128→128(identity) + per-sample-α head | **0.8641** | **0.8183** | 0.9316 | 0.8893 | 0.8080 |

Balanced is ZNPR v3, i8 + zerobias + lz4 packed, no metadata
payload, standard `Predictor::predict` runtime.

Compression is ZNPR v3, i8 + zerobias + lz4 packed, carries
`zentrain.per_sample_alpha_head` metadata; runtime dispatch lives
in `zensim::metric::forward_one_bake` (per-sample-α dispatch landed
2026-05-18 — supersedes V_22-372feat s5 on the compression trail).

### Superseded compression ship

V_22-372feat s5 packed (51,153 bytes, md5
`3be4f781238dcb35f32c964cb218a8a4`) was the compression ship from
2026-05-18 until per-sample-α runtime dispatch landed later the
same day. It loses CID22 / AIC-3 / TID decisively to V_24-per-
sample-α s4 per § A.9 (1000-bootstrap, see § "V_24-per-sample-α s4
packed vs V_22-372feat s5" below). Kept at
`zensim/weights/v_compression_2026-05-18.bin` for reproducibility.

---

## Candidate matrix (every credible candidate evaluated against both gates)

| Candidate | Bake date | n_inputs | Runtime path | CID22 | AIC-3 | KADID | TID | KonJND | Balanced gate | Compression gate |
|---|---|--:|---|---:|---:|---:|---:|---:|---|---|
| V_22-mix-LARGE+iwssim s3 packed | 2026-05-18 | 300 | vanilla `predict` | 0.8324 | 0.7845 | 0.9677 | 0.9729 | 0.8927 | **SHIP** (current) | tied on CID22+AIC-3, no win |
| V_22-372feat s5 packed | 2026-05-18 | 372 | vanilla `predict` | 0.8580 | 0.8087 | 0.9319 | 0.8875 | 0.8125 | FAIL (B>>A on KADID/TID/KonJND) | **superseded** by V_24-per-sample-α s4 on 2026-05-18 (after runtime dispatch landed); decisive loss on CID22+AIC-3+TID. Kept in `weights/` for reproducibility. |
| V_22-372feat noLARGE s5 | 2026-05-18 | 372 | vanilla `predict` | 0.8425 | 0.8059 | 0.9311 | 0.8897 | 0.8371 | FAIL | promising — marginally smaller CID22/AIC-3 lift than s5+LARGE, retained as falsification record |
| V_24-per-sample-α s4 packed | 2026-05-18 | 300 | **per-sample-α head dispatch** (zensim::metric::forward_one_bake, 2026-05-18) | 0.8641 | 0.8183 | 0.9316 | 0.8893 | 0.8080 | FAIL (B>>A on KADID/TID/KonJND) | **SHIP** (current) — decisive A>>B vs 372feat on CID22+AIC-3+TID per § A.9 (1000-bootstrap); KADID promising; KonJND tied. KADID/TID/KonJND vs Balanced within −0.10 noise tolerance. |
| V_24-α=0.10 5-seed | 2026-05-18 | 300 | vanilla `predict` | 0.8686 | 0.7912 | 0.8996 | 0.8883 | 0.8306 | FAIL | FAIL (AIC-3 +0.004 not decisive, KADID/TID −0.07) |
| V_24-stdpool prod | 2026-05-18 | 300 | vanilla `predict` | 0.8376 | 0.7785 | 0.9167 | 0.8912 | 0.5414 | FAIL (KonJND catastrophic) | FAIL (no AIC-3 win; KonJND −0.35) |
| V_24-FT-gentle s4 packed | 2026-05-18 | 300 | custom head | 0.841 | 0.809 | 0.932 | 0.890 | 0.862 | FAIL (TID −0.04, KonJND −0.03) | promising but runtime-blocked |
| V_24-PS-konjnd010 | 2026-05-18 | 300 | custom head | 0.794 | 0.803 | 0.930 | 0.889 | **0.971** | FAIL (CID22 −0.04) | FAIL (CID22 −0.04 decisive) |
| V_24-hybrid NiN s4 | 2026-05-18 | 300 | custom head | 0.8657 | 0.8066 | 0.9304 | 0.8886 | 0.7913 | FAIL | runtime-blocked |
| V_22-IW v2 (calibrated) | 2026-05-16 | 372 | vanilla `predict` + feature_transforms | 0.8164 | 0.8071 | 0.9475 | 0.9617 | n/a | FAIL (CID22 −0.077) | tied on AIC-3 +0.023, but CID22 −0.077 (loses compression-trail gate step 2) |

**Runtime status (2026-05-18, late)**: the per-sample-α dispatch
landed in `zensim::metric::forward_one_bake`. Bakes carrying
`zentrain.per_sample_alpha_head` metadata are now scoreable through
the production runtime (the bake's final layer is a `n_hidden ×
n_hidden` identity passthrough; the runtime reads the post-LeakyReLU
hidden vector as `out`, parses the head payload, and mixes rank +
pool via the per-sample sigmoid gate). Same dispatch is in
`bake_verdict` (`score_row`) and `bake_compare` (`score_corpus`)
for parquet-driven validation.

Hybrid-head and finetune V_24 architectures (`zentrain.hybrid_head`,
`zentrain.pool_head_reducer`) remain runtime-blocked until their
respective dispatches land — they would have to be reimplemented
on top of the per-sample-α scaffold or as separate metadata paths.
None of them currently ship; per-sample-α was the only candidate
that beat 372feat decisively per § A.9.

---

## Per-corpus bake_compare verdicts (1000-bootstrap, § A.9)

### V_22-372feat s5 packed vs V_22-mix-LARGE+iwssim baseline (compression-trail ship)

Full report at `/tmp/two_trail_372feat_vs_baseline.md` (1000-bootstrap
A vs B per § A.9). Aggregate verdicts:

| Corpus | n | SROCC_A | SROCC_B | Z_A | Z_B | PWRC_A | PWRC_B | h_SROCC | h_Z | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8580 | 0.8324 | 0.520 | 0.559 | 0.9126 | 0.9006 | +29.571 | +80.453 | +24.643 | **A>>B** |
| AIC-3  | 600  | 0.8087 | 0.7845 | 0.577 | 0.606 | 0.8804 | 0.8630 | +20.574 | +40.963 | +17.145 | **A>>B** |
| KADID  | 10125 | 0.9319 | 0.9677 | 0.362 | 0.249 | 0.9601 | 0.9804 | -90.446 | -795.837 | (B>>A) | B>>A |
| TID    | 3000 | 0.8875 | 0.9729 | 0.436 | 0.236 | 0.9158 | 0.9832 | -54.017 | -303.890 | (B>>A) | B>>A |
| KonJND | 1008 | 0.8125 | 0.8927 | 0.498 | 0.376 | 0.8504 | 0.9178 | -38.736 | -118.530 | (B>>A) | B>>A |

Decision: **compression-trail SHIP**.

- Step 1: A>>B on CID22 AND AIC-3 (both decisively per § A.9). PASS.
- Step 2: not B>>A on the other compression corpus (CID22 ↔ AIC-3
  both decisively A>>B; neither is B>>A). PASS.
- Step 3: KADID −0.036, TID −0.085, KonJND −0.080 all within −0.10
  noise tolerance. PASS.

### V_22-mix-LARGE+iwssim baseline vs V_22-372feat s5 (balanced-trail defense)

Same data, opposite direction. The balanced ship beats 372feat
decisively on 3/5 corpora (KADID +0.036 decisive, TID +0.085
decisive, KonJND +0.080 decisive). 372feat would FAIL the
balanced-trail gate (which forbids decisive B>>A on any of
{KADID, TID, KonJND, AIC-3}). So the balanced trail keeps V_22-mix-
LARGE+iwssim; 372feat is incompatible with that gate by design.

### V_24-per-sample-α s4 packed vs baseline (compression-trail SHIP from 2026-05-18 — runtime dispatch landed)

Fresh 1000-bootstrap report at
`/tmp/two_trail_persample_vs_baseline.md`. Aggregate verdicts:

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8641 | 0.8324 | +32.369 | +94.139 | +26.974 | **A>>B** |
| AIC-3  | 600  | 0.8183 | 0.7845 | +24.868 | +49.602 | +20.724 | **A>>B** |
| KADID  | 10125 | 0.9316 | 0.9677 | -88.609 | -774.411 | (B>>A) | B>>A |
| TID    | 3000 | 0.8893 | 0.9729 | -53.470 | -303.898 | (B>>A) | B>>A |
| KonJND | 1008 | 0.8080 | 0.8927 | -40.936 | -121.668 | (B>>A) | B>>A |

Per the compression-trail gate this PASSES — and decisively beats
372feat in a head-to-head (see next sub-section). The bake's
`zentrain.per_sample_alpha_head` dispatch landed in
`zensim::metric::forward_one_bake` on 2026-05-18 (the runtime
detects the metadata payload, parses W_α / b_α / rank_w / rank_b /
reducer_w / reducer_b / p_norm as f32-LE, and mixes y_rank + y_pool
via the per-sample sigmoid gate). Bake_verdict and bake_compare
got the same dispatch in `score_row` / `score_corpus`. The packed
bake at `zensim/weights/v_compression_persample_2026-05-18.bin`
(44,109 bytes, md5 `f09a9abdce00805000c1d112c2421b2d`) IS the
current compression-trail ship.

Round-trip verification (packed vs unpacked seed4 on CID22): SROCC
0.8641 (packed) vs 0.8640 (unpacked) = 0.0001 drift, well under
the 0.0005 pack-quality threshold.

### V_24-per-sample-α vs V_22-372feat (compression-trail head-to-head)

Full report at `/tmp/two_trail_persample_vs_372feat.md` (1000-bootstrap).

| Corpus | n | SROCC_A | SROCC_B | h_SROCC | h_Z | DecScore | Decision |
|---|--:|---:|---:|---:|---:|---:|---|
| CID22  | 4292 | 0.8641 | 0.8580 | +26.374 | +104.817 | +21.978 | **A>>B** |
| AIC-3  | 600  | 0.8183 | 0.8087 | +52.598 | +110.536 | +43.832 | **A>>B** |
| TID    | 3000 | 0.8893 | 0.8875 | +48.655 | +224.779 | +32.437 | **A>>B** |
| KADID  | 10125 | 0.9316 | 0.9319 | -7.823 | -8.873 | +1.304 | promising |
| KonJND | 1008 | 0.8080 | 0.8125 | -5.255 | -9.529 | +0.000 | tied |

Per-sample-α decisively beats 372feat on 3/5 corpora; ties or
promising on the other 2. **Per § A.9 strict majority rule,
per-sample-α IS the compression-trail SOTA.** Confirmed 2026-05-18
post-runtime-dispatch with a fresh 1000-bootstrap rerun of
bake_compare against the packed bake; numbers above are stable
under the i8+zerobias+lz4 pack (round-trip CID22 SROCC drift
0.0001).

---

## Falsified hypotheses (closed against both trails)

These were tested and lost decisively. Re-opening requires NEW evidence.

- **V_24 full 3-way mix** (cv/iw/sm 0.33 each): CID22 +0.038 but
  KADID/TID −0.09. Decisively B>>A on 16/18 cells. *2026-05-18*
- **V_24-stdpool head NiN-off**: hypothesis was "NiN-on regularization
  caused KonJND collapse" — falsified. KonJND stayed at 0.52 with NiN
  off. *2026-05-18*
- **PJND-aware pair-weighting** (Gaussian boundary, gap anchor):
  CID22 +0.023 but KonJND collapsed −0.68 to −0.81. Lever real,
  pointed wrong direction. *2026-05-18*
- **V_20a IW + ext + transforms**: TID PWRC 0.9822 best ever seen but
  CID22 SROCC 0.4632 catastrophic. Wins on the "right metrics" don't
  rescue the SROCC-on-ssim2-trained-corpus bias. *2026-05-15*
- **V_20b Su 2023 contrastive pre-train**: Won KADID + TID (every
  metric), lost CID22 (every metric). FRIQUEE 2017 caveat:
  synth pre-train → authentic-distortion transfer fails. *2026-05-15*
- **dssim co-training (cycle 7)**: All 5 dssim-weighted variants
  regressed CID22 by 0.04–0.07. *2026-04*

---

## Process — when to ship to which trail

1. **Train + eval on bake_verdict** against all 5 corpora.
2. **Run bake_compare vs both trail ships** with 1000 bootstrap.
3. **Apply both gates.** Update the candidate matrix above with the
   result.
4. If passes ONE gate: ship to THAT trail. Update
   `PreviewV0_5Balanced` or `PreviewV0_5Compression` in
   `zensim/src/profile.rs`. **Don't bump the crate version** (per
   user 2026-05-18: "we don't want crate bumps every time we get a
   nice bake").
5. If passes BOTH gates: ship to both trails (rare — would require a
   strict Pareto improvement).
6. If passes NEITHER: add a row to the candidate matrix with the
   failure mode and move on.

---

## Why two trails, not one

The single-trail experiment over 2026-05-15 to 2026-05-18 mapped
the V_24 architectural frontier and confirmed a **structural
tradeoff**: bakes that win compression-corpora SROCC lose
synthetic-distortion SROCC and vice versa. No single bake within the
228-/300-/372-feature runtime can Pareto-dominate the balanced
ship — the feature space doesn't support it.

The compression trail unlocks the compression-specialist bake that
the balanced trail's noise-strict gate vetoes. CLAUDE.md established
the priority: "Imageflow and related work is web-focused, not
archival." Two trails make that explicit.

The balanced ship is preserved so non-compression callers
(saliency-aware crop, generic perceptual diff for non-codec
distortions) don't regress on their own benchmarks.

---

## See also

- `zensim/CLAUDE.md` — methodology, statistical rigor, the
  Mohammadi 2025 panel
- `PSYCHOVISUAL_LEARNINGS_FOR_ZENSIM.md` § A.9 — the decisive rule
- `benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md` —
  balanced ship methodology
- `/home/lilith/work/zen/zensim--372feat/benchmarks/v22_372feat_methodology_2026-05-18.md`
  — 372feat methodology + 5-seed CI
- `/mnt/v/zen/zensim-eval/v24_persample_alpha_2026-05-18/` —
  per-sample-α baseline + verdicts (compression-trail candidate
  blocked on runtime)
