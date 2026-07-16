# The 5.8e6 feature explosion is unbounded SSIM `d`, not IW weighting or edge energy

**2026-07-16.** The real lever behind zensim's non-photo weakness, run to ground
by measurement after two wrong turns. Supersedes the mechanism half of
`benchmarks/iw_pooling_normalization_2026-07-15.md` and corrects
`docs/DATASET_HISTORY.md` §3.19's attribution.

Nothing is fixed. This is a characterization + a fix menu that needs the eval
panel and user sign-off (it changes metric behavior on every bake).

---

## 1. What is measured

Full scan of `bigcodec_hqdedup_traindigits_2026-07-02.parquet` (2,322,579 rows,
22 row groups), per-block max of every `f0..f371`, decoded against the true
pass-based layout (basic f0–155, peaks f156–227, masked f228–299, IW f300–371):

| block | max\|feature\| | worst column (decoded) |
|---|--:|---|
| basic | 29,009 | ssim_4th (unweighted) |
| peaks | 1.9 | (clamped) |
| masked | **5,797,029** | **f241 = masked_ssim_4th s0 ch2** |
| IW | **5,814,302** | **f313 = iw_ssim_4th s0 ch2** |

Top columns overall are **all `ssim_4th` / `ssim_2nd`**, in the masked and IW
blocks, concentrated in **channel 2 (XYB chroma), finest scale**:

```
f313  5,814,302  iw_ssim_4th     s0 ch2
f241  5,797,029  masked_ssim_4th s0 ch2
f314    525,471  iw_ssim_2nd     s0 ch2
f242    524,104  masked_ssim_2nd s0 ch2
f325    176,686  iw_ssim_4th     s1 ch1
```

So §3.19's number (5.8e6) is **real and reproduces** — it just names the wrong
features.

## 2. What §3.19 got wrong, and what my 2026-07-15 doc got wrong

- **§3.19 blamed `iw_art4`/`iw_det4`** — "unbounded edge energies". The edge
  features are NOT in the top; they measured 0.02–0.09 on the shipped extractor.
  The culprit is the **SSIM-map moments**, a different quantity.
- **My 2026-07-15 `mean_w` work** correctly proved the IW *weight* is a red
  herring (`mean_w` = 1.03–1.27×, not 15×) — but I initially concluded the
  explosion "does not reproduce". That was a **sampling error**: row group 0 does
  not contain the pathological rows, and a 4-neighbour blur is far too gentle to
  trigger real-artifact explosions. The full-parquet scan finds it. §3.19's
  "energy, not weight, is the primary driver" was right; I nearly false-falsified
  it. The `mean_w` finding stands (the weight fix would do nothing); the "no
  explosion" conclusion was wrong and is retracted here.

## 3. The mechanism (measured, not assumed)

The per-pixel SSIM difference is (`simd_ops.rs`, all **f32**):

```
num_m   = 1 − (mu1 − mu2)²                     // luminance term, NO C1
num_s   = 2·cov + C2                           // structure num,  C2 = 0.0009
denom_s = σ_src² + σ_dst² + C2
d       = (1 − num_m·num_s/denom_s) · mask     // .max(0) FLOOR, no upper cap
```

Two candidate unbounded paths, tested:

- **Denominator cancellation — FALSIFIED.** `denom_s = ssq − m1² − m2²` in f32
  risks catastrophic cancellation on large means, but a direct arithmetic test
  at XYB magnitudes 1…500 showed f32 floors `denom_s` at ~C2 (0.0009); the
  worst effect was **1.2×**, which *bounds* `d`, not explodes it.
- **Unbounded luminance term — the surviving cause.** The comment is explicit:
  *"There is no C1 — the luminance term uses `1 − (mu1−mu2)²` without a
  denominator."* So `num_m` has no lower bound: a large local mean difference
  makes it a large **negative** number, and `d = 1 − num_m·(…)` becomes a large
  **positive** number. `d` is floored at 0 but never capped above.

The **signature on the worst row confirms outlier amplification**, not a whole
flat region: at the max-f313 row, iw_ssim_mean = 9,234, iw_ssim_2nd = 525k,
iw_ssim_4th = 5.8M — ratios 4th/mean = **630×**, 2nd/mean = **57×**. A few
pixels with astronomically large `d`, amplified by the L4/L2 moment. (Even the
L1 mean, 9,234, is far above a bounded SSIM difference — so it is a bad *region*
plus extreme spikes, not a single pixel.)

**Why channel 2, why masked ≫ basic:**
- Channel 2 is the XYB blue-yellow chroma — the largest-magnitude channel, so
  `(mu1−mu2)²` reaches the largest values there.
- `masked` (5.8M) ≫ `basic` (29k) because the flatness mask is HIGH exactly on
  the flat/low-activity regions where a chroma artifact produces the biggest
  local-mean jump — the mask *multiplies* the worst pixels.

## 4. Why this is the non-photo lever specifically

High-contrast synthetic content (screenshots, documents, line-art, AI-gen) has
large flat chroma regions bordered by hard edges. A codec artifact on such an
edge produces a large local-mean difference in a region the flatness mask
weights heavily → the unbounded luminance term → a 5.8e6 feature. Photographic
content rarely has both the flat chroma AND the hard artifact edge, so it stays
bounded. That is the content-blindness the winsor guard currently *clamps*
rather than fixes — a band-aid over a feature that is genuinely wrong on the
content class we are weakest on.

## 5. Fix menu (NONE applied — each changes metric behavior; needs the panel)

1. **Cap `d` above.** `d.max(0).min(CAP)` — one line, mirrors the existing
   `.max(0)` floor. Bounds the whole `ssim_*` family at once. CAP from the
   healthy p99.9 (photographic ssim_4th ≈ 0.5, so a CAP around 2–4 is generous).
2. **Add a C1 to the luminance term.** `num_m = (2·mu1·mu2 + C1)/(mu1² + mu2² +
   C1)` — the standard SSIM luminance form the current code deliberately omits.
   Bounds `num_m` to ≤ 1 by construction. Closest to "correct SSIM".
3. **Per-image energy normalization** (§3.19's suggestion) — divide the map by a
   per-image scale before pooling. Heavier; changes rank structure more.
4. **Winsor guard** (shipped) — clamps the FINAL feature. Already in place, and
   it is why B ships despite this. It hides the defect from the model but leaves
   144 features carrying a clamped-outlier value on exactly the hard content.

**Recommendation to evaluate first:** option 2 (add C1) — it is the principled
fix, bounds the term by construction rather than by a magic CAP, and is a
localized change to one expression. Option 1 (cap) is the cheap fallback if the
retrain cost of 2 is not worth it. Either requires: re-extract the affected
corpora, refit, and run the full rank + dial panel (the change moves every
`ssim_*` feature, so every bake's numbers move). That is the retrain the
`mean_w` work correctly said the *weight* fix did not justify — but the *energy*
fix might, because this is a 5.8e6-vs-0.5 distortion on the non-photo axis, not
a 1.27× one.

## 6. Instruments landed

- `streaming.rs::tests::dump_ssim_moment_explosion` (`#[ignore]`) — dumps the
  biggest raw ScaleStats field per image; the tool that located this.
- `ScaleAccumulators::iw_a_sum` + `ScaleStats::iw_mean_w` (from the `mean_w`
  work, `iw-diagnostics` feature) — measures the weight factor, now known to be
  the *non*-lever.

## 7. One-line status for the next session

The non-photo lever is **unbounded per-pixel SSIM `d`** (no C1, no upper cap) →
5.8e6 in `ssim_4th`/`ssim_2nd`, XYB chroma. Weight-normalization (`mean_w`, Σw)
is NOT it. Fix = add C1 or cap `d`; both need a re-extract + full panel + sign-off.
