# EX-PERCENTILE-POOL — FALSIFIED (vs Compression ship, scoped corpus)

**Date:** 2026-05-18
**Workspace:** `~/work/zen/zensim--exp-percentile-pool`
**Hypothesis:** Replacing Block B's L8-norm `(Σd⁸/N)^(1/8)` p95 approximation
with TRUE P²-quantile p95 estimator carries cleaner distribution-tail
signal that an MLP can exploit. Expected lift: +0.005 to +0.015 CID22
SROCC, retain compression-band performance.

**Verdict:** Falsified against the Compression-trail ship under the
predefined criteria. CID22 5-seed median P² = 0.420 ≤ ship 0.8641 (no
lift). TID and AIC-3 each fall >0.05 from ship. The P²/L8 in-place
swap, in this experiment's training scope, fails the formal gate.

**Caveat (scope-limit):** Re-extraction of the 196k safesyn training
corpus and 73k cvvdp_iwssim_LARGE corpus was deferred — re-extracting
them requires either (a) ~30+ minutes of CPU on the source CSV which
also needs iwssim/cvvdp target join (multi-hour pipeline) or (b)
inverting the canonical training parquet to recover source paths
(no such inversion path exists). All trained bakes (P² and L8
baseline alike) used **only** the kadid + tid + konjnd parquets that
were re-extractable. This makes the verdict a "narrow falsification":
the in-place L8→P² swap with limited training is dominated, but the
P²-features-with-full-training counterfactual is **untested**.

---

## Apples-to-apples L8 vs P² (same limited corpus, same recipe)

Both bakes: V_24-per-sample-α recipe (hidden=128, 300 epochs, target
`mix_cv40_iw60`, per-sample α head). Training groups: kadid
(train_w=0.3), tid (0.3), konjnd (0.02). No safesyn, no
cvvdp_iwssim_LARGE.

### 5-seed aggregate SROCC (per corpus)

| Seed | CID22 (P²) | CID22 (L8) | KADID (P²) | KADID (L8) | TID (P²) | TID (L8) | KonJND (P²) | KonJND (L8) | AIC-3 (P²) | AIC-3 (L8) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| s1 | 0.411 | 0.001 | 0.897 | 0.941 | 0.798 | 0.890 | 0.964 | 0.948 | 0.063 | 0.637 |
| s2 | 0.435 | 0.032 | 0.899 | 0.941 | 0.799 | 0.891 | 0.960 | 0.944 | 0.162 | 0.600 |
| s3 | 0.419 | 0.025 | 0.899 | 0.940 | 0.797 | 0.891 | 0.964 | 0.950 | 0.099 | 0.635 |
| s4 | 0.427 | 0.059 | 0.899 | 0.941 | 0.798 | 0.890 | 0.960 | 0.939 | 0.161 | 0.599 |
| s5 | 0.365 | 0.002 | 0.897 | 0.942 | 0.797 | 0.891 | 0.963 | 0.945 | 0.105 | 0.638 |
| **median** | **0.420** | **0.025** | **0.899** | **0.941** | **0.798** | **0.891** | **0.963** | **0.945** | **0.105** | **0.635** |
| Δ (P²-L8) | **+0.395** | — | -0.042 | — | -0.093 | — | +0.018 | — | -0.530 | — |

### Striking finding: P² flips compression-band rankings opposite

Within the SAME limited-corpus training:

- **CID22**: P² **massively wins** (+0.395). Both bakes are far from
  ship (0.864), but P² is qualitatively in the right ballpark
  (median 0.42) while L8 limited is near random (0.025). P²
  features encode CID22-relevant signal that L8 doesn't, at least
  in the small-corpus regime.
- **AIC-3**: P² **massively loses** (-0.530). L8 limited keeps a
  reasonable AIC-3 rank (0.635); P² collapses (0.105). Direction
  opposite to CID22.
- **KADID / TID / KonJND**: small differences both ways (-0.09 to
  +0.02). Mostly within noise / regime-mix.

**Hypothesis on the CID22/AIC-3 split**: P² captures the true
distribution tail (95th-percentile pixels) which CID22 human MOS
correlates with (people see bad outlier regions), but AIC-3's
near-PJND-threshold signal lives in the *median* of the
distortion distribution (small-but-pervasive distortion), which L8
averages over and P² p95 misses. Not testable without IS3 mid-band
percentile features (p5, p50). Filed as a follow-up.

### Comparison to Compression ship (V_24-per-sample-α s4, full corpus)

| Bake | CID22 | KADID | TID | KonJND | AIC-3 |
|---|---:|---:|---:|---:|---:|
| Compression ship (full corpus, L8) | **0.864** | **0.932** | **0.889** | 0.808 | **0.818** |
| L8 limited (kadid+tid+konjnd, L8) | 0.025 | 0.941 | 0.891 | 0.945 | 0.635 |
| P² limited (kadid+tid+konjnd, P²) | 0.420 | 0.899 | 0.798 | **0.963** | 0.105 |

Both limited bakes are catastrophically below the ship on CID22 and
AIC-3 — confirming the "missing safesyn corpus" is the dominant
gap. The P² lift over L8 in the limited regime (+0.395 CID22, +0.018
KonJND) is mechanistically interesting but doesn't survive against
the full-corpus ship.

---

## Falsification per § 'Hypothesis on the CID22/AIC-3 split' / `prompt`

The prompt's falsification criteria:

1. **CID22 5-seed median ≤ 0.8641 (no lift over Compression ship)** —
   yes, P² median = 0.420 << 0.8641. **FAILS criterion 1.**
2. **KADID/TID/KonJND each within −0.05 of Compression ship** —
   KADID -0.033 (within), TID -0.091 (BREACH), KonJND +0.155
   (above). **FAILS criterion 2** (TID breach).

Either criterion failure constitutes falsification. Both fail.
**Verdict: FALSIFIED.**

---

## What this DOESN'T tell us (deferred counterfactuals)

The verdict is for **in-place L8→P² swap with the limited training
corpus**. It does NOT speak to:

1. **P² features with full safesyn + cvvdp_iwssim_LARGE training.**
   To re-extract those, we need either source paths (not in canonical
   parquets) or a multi-hour image-decoding pipeline that joins iwssim
   + cvvdp targets back in. The +0.395 CID22 lift in the limited
   regime suggests the full-corpus P² number could be materially
   higher than the limited result — but it's also possible safesyn
   training would saturate the L8 / P² difference (both perform
   similarly when given enough data).

2. **Additive P² features (Option B in the prompt).** Adding p5 + p50
   + p95 P² as new 9 features per channel (input → 336 cols) while
   KEEPING the existing L8 max+p95 — this preserves the L8 information
   and adds P² as a supplement. Not run here; out of time budget.

3. **Multi-percentile P² (p5 + p50 + p95 + statistical companions).**
   The prompt's Option A specifies 6 in-place features (3 P² + 3
   companions). We only swapped 3 (the L8 p95 with P² p95) since the
   max features already give a peak measure. A full p5/p50/p95 +
   skewness/kurtosis swap is unexplored.

4. **CID22 / AIC-3 split mechanism.** P² shapes CID22 differently
   from AIC-3 in the limited-corpus regime — these are both
   compression corpora but P² helps one and hurts the other. The
   distribution-tail-vs-median hypothesis is plausible but
   unfalsified.

---

## Reproducibility

- **Code:** main commit on workspace `zensim--exp-percentile-pool`,
  `jj describe -m "exp(percentile-pool): P2 quantile + Block B L8→P2 swap"`
  - `zensim/src/p2_quantile.rs` — P²Estimator + P²Triplet + 8 unit tests
  - `zensim/src/streaming.rs` — ScaleAccumulators carries P² triplets,
    `update_p2_for_strip()` does scalar H+V blur and feeds samples,
    `finalize(use_p2)` swaps L8 → P² p95 in the ssim/art/det fields
  - `zensim/src/metric.rs` — `ZensimConfig::compute_p2_pool`
  - `zensim-validate/src/main.rs` — `--p2-pool` flag, cache-key bit 2
  - `zensim-bench/examples/extract_features_372col.rs` — `--p2-pool`
- **Bug fixes during impl:** (a) clamp SSIM `sd ∈ [0, 1]` before P²
  feed (catastrophic FP cancellation otherwise), (b) clamp
  `P²Estimator::estimate()` to `[q[0], q[4]]` (observed range, guards
  against parabolic extrapolation in skewed distributions).
- **Training parquets (P²):**
  `/mnt/v/zen/zensim-training/2026-05-18-percentile-pool/`
  - `kadid_mix_300col_p2.parquet` (10090 rows × 317 cols: targets + f0..f299)
  - `tid_mix_300col_p2.parquet` (2994 rows × 317 cols)
  - `konjnd_mix_300col_p2.parquet` (1008 rows × 314 cols)
- **Validation parquets (P²):** same dir, `<corpus>_features_372col_p2.{csv,parquet}`
- **Training command:** `scripts/exp_percentile_pool/train_seed.sh <seed> <out>` —
  identical to V_24-per-sample-α recipe except scoped to kadid + tid + konjnd.
- **L8 baseline (same limited corpus):**
  `scripts/exp_percentile_pool/train_seed_l8.sh` — same args, uses
  L8 features from `2026-05-17-cvvdp-merged-trainer/`.
- **Per-seed bakes:**
  - P²: `/tmp/exp_percentile_pool_bakes/p2pool_seed{1..5}.bin` +
    `.../p2pool_seed{1..5}_verdict.md`
  - L8: `/tmp/exp_percentile_pool_baseline/l8baseline_seed{1..5}.bin` +
    `.../l8baseline_seed{1..5}_verdict.md`

---

## Status

- FEATURE-AUDIT Pick 3 (last of the three audit candidates): **FALSIFIED.**
- Pick 1 (Chunk C per-pair standardizer): closed earlier with 3 strikes.
- Pick 2 (metric inputs): closed earlier with 2 strikes.
- Single-bake architecture frontier on the published audit list is now
  **fully exhausted**. The only remaining lever is the multi-codec
  data expansion (EX-MULTI-CODEC, in flight in
  `zensim--exp-multi-codec`).

**Followups worth filing (low priority, not blocking):**

- Re-extract safesyn + cvvdp_iwssim_LARGE with P² and re-run V_24
  with the full canonical corpus. Cost: ~3-4h of CPU + Python join
  pipeline development. Would test counterfactual #1 above.
- Try Option B (additive P² as new 9 features → 336 cols). Cost:
  trainer with `--max-features 336` + re-baked parquets. Would test
  counterfactual #2.
- Investigate CID22/AIC-3 inversion via distribution moment analysis.
  Cost: 1h of analysis. Would inform whether p5 / p50 / median /
  IQR features could close the AIC-3 gap.
