# ZNPR sparse-I8 / zero-bias feasibility eval (2026-05-13)

**Question**: would a `WeightDtype::I8Sparse` plus training-side zero-biasing
shrink the V0_18 bake (93,064 bytes) materially?

**Method**: rebake V0_17 F32 → I8 with per-layer-max threshold τ, score on
full CID22 (n=4,292) via `dataset_metric_baseline`. Compressed sizes for the
i8 weight bytes only; full-bake adds 5,136 bytes overhead. Rebake at τ=0 is
bit-identical to V0_18 (ρ=1.0, max|Δ|=0 on 1,000 random inputs).

## Size vs SROCC sweep (CID22, n=4,292)

| τ     | i8 zeros | gzip-9 | zstd-22 | RLE u16,i8 | sparse u24,i8 | sparse varint | best full bake | CID22 SROCC | vs V0_18 0.8934 |
| ----- | -------- | ------ | ------- | ---------- | ------------- | ------------- | -------------- | ----------- | --------------- |
| 0.000 | 1.4%     | 84,044 | 83,902  | 260,244    | 346,992       | 173,498       | **89,038**     | 0.8934      | +0.0000         |
| 0.001 | 43.3%    | 63,768 | 59,560  | 149,532    | 199,376       |  99,690       | **64,696**     | 0.8934      | +0.0000         |
| 0.005 | 87.5%    | 20,190 | 18,597  |  32,952    |  43,936       |  21,970       | **23,733**     | 0.8933      | -0.0001         |
| 0.01  | 88.7%    | 18,629 | 17,415  |  29,790    |  39,720       |  19,862       | **22,551**     | 0.8933      | -0.0001         |
| 0.02  | 90.2%    | 16,535 | 15,249  |  25,800    |  34,400       |  17,205       | **20,385**     | 0.8931      | -0.0003         |
| 0.05  | 92.8%    | 12,992 | 12,029  |  19,086    |  25,448       |  12,732       | **17,165**     | 0.8920      | -0.0014         |

zstd-22 won every row. Per-band SROCC stays within ±0.005 at every τ ≤ 0.05.

## Best size at CID22 ≥ 0.890

- **τ=0.005, zstd-22 → 24 KB** (75% shrink). SROCC 0.8933, within 0.0001 of
  V0_18. Per-band shifts <0.001. **Recommended operating point.**
- τ=0.02, zstd-22 → 20 KB (78% shrink). SROCC 0.8931 (-0.0003).
- τ=0.05, zstd-22 → 17 KB (82% shrink). SROCC 0.8920 (-0.0014); outside the
  "within 0.001 of V0_18" envelope but ≥0.890.

The 50% / 46 KB user target is comfortably exceeded.

## Verdict

**No on `WeightDtype::I8Sparse`.** Sparse/RLE formats are 1.4-2.5× larger
than zstd at the same SROCC. New dtype variant pays format-extension cost
and loses the size race.

**Yes on a `flags` bit "weight section is zstd-compressed".** ~50 lines of
Rust plus `ruzstd`, no algorithm change, no SROCC regression, 75-80%
shrink. Format extension is one flag plus a decompress call at load time,
not a new wire encoding.

**Even better lever: shrink the architecture.** 88% of layer-0 can be
zeroed at no SROCC cost — the 228×384 layer is over-parameterized.
Retraining at 228×128 should deliver ~30 KB raw I8 with a stronger
training signal in the smaller search space. zstd and the architecture
lever stack: 228×128 + τ=0.005 + zstd lands around 10 KB.

## Structural findings that surprised me

1. **V0_18 layer-0 is only 1.0% true zeros, not 43%.** The user's 43%
   was measured against per-layer max. I8 quantization scales
   per-output-column; many columns have max ~100× smaller than the
   layer max, so weights that look tiny vs layer max are large vs their
   own column and quantize non-zero. Under per-column scaling only 0.3%
   of V0_17 weights round to 0 untouched.

2. **SROCC is robust to aggressive zero-biasing.** Zero out 88% of
   layer-0 weights, CID22 drops 0.0001. The 228→384 layer's behavior
   rides on its top ~12% largest-magnitude connections.

3. **zstd dominates structural sparse codings.** At 88% zero density,
   zstd-22's 18 KB beats sparse-varint (22 KB) and sparse-u24 (44 KB).
   zstd captures zeros + repeated small magnitudes + byte patterns
   together. A custom sparse format would need entropy coding to
   compete and become zstd.

4. **RLE-zeros at low density is catastrophic.** At τ=0 (1.4% zeros)
   the 3-bytes-per-run RLE triples the file size.

## Scratch on disk (uncommitted, /tmp/rle_eval/)

- `rebake_zerobias.py` — ZNPR v2 I8 rebaker (ρ=1.0 vs V0_18 at τ=0)
- `run_threshold_sweep.sh`, `sweep/v17_t*.bin`, `sweep/srocc_t*.log`
- `step5_combined_output.txt` — compression table source
