# Standard bake packing path — pack-then-calibrate (2026-05-27)

The canonical way to ship a zensim bake small. Established 2026-05-27 after
the V39-era workflow regressed on packing (V39 ships as a 257 KB raw-F32
binary; the i8+zerobias+lz4 pack step that made V0_5-era bakes ~54 KB stopped
being applied around 2026-05-25).

## The tool

`scripts/v_next/pack_and_calibrate.py <orig.bin> <out.bin> [flags]`

Standard invocation (recommended for every bake before ship):

```sh
python3 scripts/v_next/pack_and_calibrate.py IN.bin OUT.bin \
    --dtype f16 --zerobias-bulk 0.005 --neg-tail
```

Flags:
- `--dtype f16` — half precision (mantissa is plenty for weights ~O(1);
  near-lossless, unlike i8 which costs identity precision on the
  per-sample-α passthrough). `f32` to disable, `i8` only if you accept the
  identity-precision hit.
- `--zerobias-bulk TAU` — zero every `|w| < TAU` (default 0.005). lz4 then
  compresses the zeros hard.
- `--protect-last` — keep the final layer at f32 + τ=0. **Usually
  unnecessary** (the refit below recovers identity even with the last layer
  aggressively zerobiased), but available for arches where the refit can't
  re-anchor identity.
- `--neg-tail` — keep a single dial=0 bottom knot so PCHIP extrapolates
  NEGATIVE below the anchor's worst (corruption resolution), instead of
  flat-0.

## The load-bearing rule: QUANTIZE, *then* CALIBRATE

Zerobias / f16 / i8 preserve RANK (signs intact) but shift the network's raw
outputs. A spline fit on the **f32** network maps the **packed** network's
identity output to the wrong dial value → identity drops (e.g. 97.8 → 93.4).

**Fix: refit the output calibration spline on the PACKED network.** The
pipeline strips the old spline, packs (zerobias + dtype), then fits the spline
on the packed network's tanh-pin outputs over the multiband anchor and
re-injects it. SROCC is rank-invariant under the monotone spline; identity
re-anchors to its exact value.

This ordering makes plain GLOBAL zerobias safe — no per-layer surgery needed.

## Measured (v47-strict-recal-negtail, per-sample-α + tanh-pin + spline, 3 layers)

| path | size | identity | CID22 SROCC | blur ladder |
|---|--:|--:|--:|---|
| raw f32 (V39-era workflow) | 198 KB | 97.8 | 0.8547 | 0 above-id |
| f16 + global zb 0.005, **stale spline** (calibrate-then-quantize) | 50 KB | **93.4** ✗ | 0.8554 | 0 above-id |
| **f16 + global zb 0.005 + REFIT (pack-then-calibrate)** | **30 KB** | **97.5** ✓ | **0.8564** | 0 above-id, [−210, 97.5] |
| f16 + zb 0.005 + protect-L2 + refit | 30 KB | 97.5 | 0.8564 | identical (protect-L2 redundant) |

The 30 KB packed bake matches the 198 KB f32 on the full Mohammadi panel
(CID22 0.8564 / KADID 0.80 / TID 0.79 / KonJND 0.485 / AIC-3 0.771 /
AIC-4 0.894), G1 0.99, weighted goal 0.645 — at **6.6× smaller**, below even
the old 41–54 KB convention. Canonical packed bake:
`/mnt/v/output/zensim/bakes/v47_strict_recal_negtail_packed30k_2026-05-27.bin`
(29,995 bytes, md5 `4c6cfc67769132f01bc8cca81cc6d597`).

## Why not the surgical/per-layer machinery

`zenpredict-bake::apply_zero_bias_per_layer` + `bake_quant_stats` `l1_share`
saliency exist and CAN do per-layer / saliency-weighted pruning. But the
empirical finding is that **refit-on-packed makes global zerobias safe** — A
(global, 98.4% of L2 zeroed) and B (protect-L2) are bit-for-bit equivalent in
identity / SROCC / size. Reach for per-layer only if a future arch's refit
fails to re-anchor identity. Don't add complexity the refit already obviates.

## Standard for every ship

1. Train → f32 bake (trainer default; unchanged).
2. `pack_and_calibrate.py --dtype f16 --zerobias-bulk 0.005 --neg-tail`.
3. Verify: `bake_verdict` (panel + G1) + `eval_bake_blur_monotonicity.sh`
   (0 above-identity, identity ≈ exact) + identity score via
   `score_pair_with_bake --bake-post clamp --ref X --dist X`.
4. The packed bake is the ship artifact (committed to `zensim/weights/`).

Latent cleanup: the existing raw-F32 ships in `zensim/weights/` (V39 = 257 KB)
can be re-packed through this path to restore the small-weight convention
(SROCC-neutral by construction). Do that when rotating each profile's bake.
