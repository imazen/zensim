# V0_2-style linear + new features + BVLS + (optional) input shaping — 2026-05-28

User pivot (2026-05-28): "look at what V0_2 trainer did and consider that
with the new features and data sets and shaping etc, and BVLS." V0_2
was 228 features, Nelder-Mead-fit, 127 non-zero weights, CID22 SROCC
0.8676. Today we have 372 features + per-feature shaping TSV + new
canonical-2026-05-21 dataset + BVLS solver. This is a clean experiment.

## Pipeline

`scripts/v_next/train_v02_bvls_shaped.py`. Pure linear: `y = w · σ(x) + b`,
where σ is per-feature shaping (Yeo-Johnson / WinsorP99 / SignedCbrt /
QuantileBins / Identity, sourced from the auto_transforms TSV). 372
features + bias = 373 BVLS unknowns. Sign-mask: 300 sign-safe (`w_i ≥ 0`),
72 free. Standardize on shaped features. Post-fit PCHIP spline on the
multiband anchor. Emits ZNPR v3 via `zenpredict-bake` JSON pipeline
(per CLAUDE.md mandate).

## Three runs

| variant | bake size | n_active | CID22 | KADID | TID | KonJND | AIC-3 | AIC-4 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| v47 strict-QAT (shipped) | 27,316 | n/a | **0.866** | **0.793** | **0.793** | 0.418 | 0.768 | 0.885 |
| **v02-bvls NO shaping** | **8,622** | 86 | 0.824 | 0.757 | 0.734 | **0.594** | 0.747 | 0.794 |
| **v02-bvls WITH shaping** | 19,440 | 80 | 0.828 | 0.667 | 0.709 | 0.072 | **0.846** | **0.897** |
| v47-linear h=8 strict | 15,800 | n/a | 0.632 | 0.825 | 0.815 | 0.322 | 0.591 | 0.628 |
| v47-linear h=8 keep-72 | 15,771 | n/a | 0.605 | 0.818 | 0.828 | 0.251 | 0.598 | 0.643 |
| v47-linear MVP-Python | (no bake) | 86 | 0.824 | 0.757 | 0.734 | n/a | 0.747 | n/a |

## Reading the result

The MVP-Python and v02-bvls-NO-shaping are **byte-for-byte identical**
on the four codec corpora (CID22, KADID, TID, AIC-3) — confirms the
Python emitter is correct and the Rust runtime reproduces the BVLS
forward pass exactly. Now we also have a real ZNPR v3 bake (8.6 KB)
and per-pair predictions on KonJND + AIC-4.

### NO shaping = balanced sibling

- Wins KonJND by **+0.176 vs v47** (0.594 vs 0.418). v47's MLP head
  cratered KonJND for f16 precision reasons (per `v0_qat_native_methodology`
  doc); the linear BVLS path preserves the fine-weight discrimination.
- −0.042 CID22, −0.036 KADID, −0.059 TID vs v47 (small).
- −0.091 AIC-4 (loses) but matches AIC-3 closely.
- Bake is **3.2× smaller** than v47 (8.6 KB vs 27.3 KB).
- Dial calibration via PCHIP spline lands clean: p5=-4.8, p95=87.4 → G1 = 1.00.

### WITH shaping = compression-corpus specialist

- Wins **AIC-3 by +0.078** (0.846 vs 0.768) and AIC-4 by +0.012 vs v47.
- CID22 tied with no-shaping (0.828 vs 0.824).
- LOSES KADID (-0.126), TID (-0.084), KonJND (-0.346) vs no-shaping.
- The shaping is calibrated on safesyn (codec-distortion noise) — it
  absorbs codec feature non-linearity but distributes badly on
  analytic-distortion features.

## Honest verdict

Both variants are interesting wins on different axes:

| | NO shaping | WITH shaping |
|---|---|---|
| Codec ranking | competitive | **best on compression specifically** |
| Analytic distortions | competitive | hurt |
| KonJND (visually-lossless calibration) | **+0.176 vs v47** | broken |
| Bake size | **8.6 KB** | 19.4 KB |
| Sparsity | 86/372 | 80/372 |
| Dial range | clean | clean |

**Neither replaces v47**, which still wins overall geomean3.
But these are strong candidates for **sibling profiles** in
`ZensimProfile`:

- A `Linear` variant (no-shaping) — small, balanced, KonJND-best.
- A `Compression` variant (shaping) — best AIC-3/AIC-4, weakest KADID/TID.

Or — for the "regression-test gating" use case the user requested
(image-engine-error detection), the no-shaping variant has the cleanest
profile: small, interpretable, KonJND-preserving.

## What surprised me

Shaping was supposed to help across the board (the screen showed
+Pearson lift on safesyn). It does help safesyn-trained features
(CID22 +0.004, AIC-3 +0.10, AIC-4 +0.10) — but it ALSO makes the
model SPECIFIC to the safesyn distortion distribution. KADID/TID's
analytic distortions distribute differently and the shaping
over-warps their feature space.

This is consistent with the V_20 input-shaping learnings (2026-05-15):
> "V_20 IS single-MLP closes CID22 B3 [30, 40) gap by +0.129 SROCC
> but costs −0.014 CID22 aggregate (B4–B8 each lose 0.02–0.06)."

Shaping is a B-band specialist mechanism, not a universal lift. The
v02-bvls confirms the same shape: **shaped models specialize to the
training data's distortion distribution**.

## Re-thinking the v47-linear ship

The original v47-linear shipped as h=8 strict-mode with CID22 0.632.
That now looks unimpressive vs the v02-bvls no-shaping at 0.824 with
1.8× smaller bake. The MLP wrapper at h=8 was actually limiting
expressivity vs pure BVLS — which makes sense because Adam's soft
penalty + projection bake mismatch ate 0.19 SROCC vs BVLS's bounded LS.

## Files

- Bakes:
  - `/mnt/v/output/zensim/bakes/v02_bvls_shaped_2026-05-28.bin` (19,440 B, with shaping)
  - `/mnt/v/output/zensim/bakes/v02_bvls_NO_shaping_2026-05-28.bin` (8,622 B, no shaping)
- Script: `scripts/v_next/train_v02_bvls_shaped.py`
- Verdicts: `/tmp/v02_bvls_*_verdict.md`

## Reproduction

```bash
python3 scripts/v_next/train_v02_bvls_shaped.py                  # with shaping
python3 scripts/v_next/train_v02_bvls_shaped.py --no-shaping \
    --out /mnt/v/output/zensim/bakes/v02_bvls_NO_shaping_2026-05-28.bin
./target/release/bake_verdict --bake <bake> --output <verdict.md>
```

Inputs: same as v47_linear (safesyn 196k, cid22_train 17.6k, kadid 10.1k,
tid 3k from canonical-2026-05-21/train + multiband_anchor_dial100). Sha256s
match the v47_linear manifest's pinned values.
