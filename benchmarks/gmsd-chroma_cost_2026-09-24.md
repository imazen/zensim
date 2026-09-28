# Revised C8 cost — 2026-09-24 (quarantined)

CONTENDED. Measured on the AVX2 tier (no AVX-512) of a household worker (node-3), not on the production AVX-512 tier; descriptive only. Both thread settings miss the inherited <=5% marginal-cost goal at1024².
These are descriptive on/off measurements, not evidence for adoption. No constants or scale choices were retuned.

| Threads | alpha(ns) | beta(ns/pixel) | Fit marginal at1024² | Raw median marginal at1024² | R² | Rounds at256/1024/2048/4096 |
|---:|---:|---:|---:|---:|---:|---|
| 1 | -2482160.7593486714 | 36.280039644661052 | 8.45592952164% | 8.99470849082% | 0.999816877862 | 8/8/8/2 |
| 8 | 63677.407824078051 | 22.462436058823393 | 10.0324590878% | 10.3869699435% | 0.999982907198 | 8/8/8/4 |

The wall budget curtailed4096² to2 ST and4 MT rounds. Drift warnings occur in both runs; MT reports28 gate waits/noisy rounds. Treat slopes/intercepts as descriptive, with especially weak large-image precision. A negative intercept is a fit artifact, not negative overhead.

| Threads | Size | OFF median(ns) | ON median(ns) |
|---:|---:|---:|---:|
| 1 | 256² | 22682557.5 | 24690449 |
| 1 | 1024² | 420535885.5 | 458361862.5 |
| 1 | 2048² | 1626646212 | 1770695832 |
| 1 | 4096² | 6435191077 | 7042646550 |
| 8 | 256² | 13607643.25 | 15030777 |
| 8 | 1024² | 235408373.5 | 259860170.5 |
| 8 | 2048² | 934934323 | 1028318703 |
| 8 | 4096² | 3781259616 | 4158352111.5 |

Owner: unchanged `zensim/benches/extract_paths_bench.rs`, arms `fold1322_rev4` and `fold1502_gmsbank` in the same interleaved groups. All sibling arms stay in the existing harness. Synthetic paired RGB8 inputs and the harness control the order/budget; raw samples are retained.
Worker2 shared heavy wrapper, portable Rust1.98.1, no target-cpu=native; Rev3/sqrt; requested8 rounds,180s cap per group. All settings are explicit arguments inside the neutral UTS namespace.

Binary SHA256 `e2f5b8cbe1190fd3d2be822e4ac87af7b3f9f3d5df065478912d89c48817cffb`.
`cost_build_v1.jsonl` proves both the ordinary zensim library and benchmark were compiled (`fresh:false`) from `c8/impl-src` before any parent ordinary-library build; v3 uses the same SHA. The invalid shared-target matrix involved the separate lib-test executable. `bench_to_final_feature_v2.patch` shows only documentation and a new flat-colour test differ from the final candidate; computational code is unchanged.

Raw root `/var/tmp/gmsd-chroma/c8/`:
| File | SHA256 |
|---|---|
| `cost_v3_mt1.zenbench` | `d4e6cd8ff2ff601a925f204425e87bf29cb8c783ddc429d9e6593d7a3ce28f89` |
| `cost_v3_mt8.zenbench` | `cb159daf11b952eb767dd15e0b002a7011d47d767dda3a3ecf6b5a4edaaccd22` |
| `cost_build_v1.jsonl` | `e8eb83667dae2d8c3457c9c0bb0ffd1a590648de721c6b013a72908ba4140b46` |
| `cost_build_v3.jsonl` | `7dfbf06b7ea9e5d18241bdb5236e0f38eba922913357e60e087199509101e9e4` |
| `logs/cost_v3.log` | `004bce9c6ec42d06f2989ff7168694d219490e75795e62065002c8da7d4fd771` |
| `bench_to_final_feature_v2.patch` | `b1cd9f3a0e785481771647d61cbe3ee2d65529955cffa11452dc7b6592718efd` |

Recompute: `python3 scripts/gmsbank/cost_fit.py /var/tmp/gmsd-chroma/c8/cost_v3_mt1.zenbench /var/tmp/gmsd-chroma/c8/cost_v3_mt8.zenbench`.
Actual output is preserved in `/var/tmp/gmsd-chroma/logs/c8_cost_fit_both_v3.log`. Costv1 failed preflight; costv2 used cleared environment/default sizes and was terminated. Neither contributes to this record.
