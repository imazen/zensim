# PU-XYB SIMD conversion bench — 2026-06-10

`cargo bench -p zensim --bench pu21_bench` (zenbench 0.1.8, interleaved
paired runs; NO -C target-cpu=native; 7950X, nice -n 19). Workload: 1080p
(2,073,600 px) absolute-luminance log-sweep 0.01..4000 nits with chroma
variation → positive PU-XYB planes. Branch: feat/hdr-pu-frontend (PR #44),
commit: see log for this file's introduction.

|  | mean ±mad | throughput | vs scalar |
|---|--:|--:|--:|
| simd_dispatch (incant v3/neon/wasm128/scalar, midp_precise exp2/log2) | 25.8 ±0.5 ms | 80.3 Mpx/s | baseline |
| scalar_powf | 85.5 ±0.6 ms | 24.3 Mpx/s | +229..233% (95% CI) |

= 12.4 ms/MP SIMD vs 41.2 ms/MP scalar — **3.3×**. Notes:
- v4 (AVX-512) tier intentionally absent: `X64V4Token` lacks `F32x8Convert`
  (the trait the generic transcendentals need) — same pattern as the f16
  forwarders; AVX-512 boxes dispatch the v3 (AVX2 8-wide) variant.
- `lowp` transcendentals were tried first and FAILED parity (max |Δ| 0.026
  on outputs in [0,2.5] — the ~1% error amplified through P6=596);
  `midp_precise` passes the 2e-3 band (`color::pu_simd_parity_tests`).
- Conversion is one O(N) pass before the multiscale pyramid; end-to-end
  compute_pu impact is correspondingly smaller. A log-domain LUT+lerp
  remains a candidate if the conversion ever shows up hot end-to-end.
