# PERF criterion-5 consolidated re-RUN — x86 tiers, both repos, this box (2026-08-27)

The one-table re-run the scorecard's PERF line left ❌: every formerly-NEON-only
kernel family's x86 dispatch, zenbench-measured on the dev box, **NO
`-C target-cpu=native`** (RUSTFLAGS unset, no cargo config target-cpu; verified
pre-run). Per-repo byte-identity gates are the standing evidence
(`tier_isolation`/`kernel_tiers` batteries + magetypes by-construction); this
table is the cross-repo measurement on one host.

- host: WSL2 AMD Ryzen 9 7950X (28c/59GiB visible), run-heavy (nice19+ionice3, 24G cap)
- jxl-encoder @ `204ca903`, zenavif @ `69944d3e`
- commands: `cargo bench -p jxl-encoder --bench tier_isolation --features _dev`;
  `cargo bench -p jxl-encoder-simd --bench kernel_tiers`;
  `cargo bench --bench tier_isolation --features _dev` + `--bench unpremul_tiers
  --features _dev` (zenavif)
- full logs: `~/tmp/perf_jxl_tier_isolation.log`, `~/tmp/perf_jxl_kernel_tiers.log`,
  `~/tmp/perf_avif_tier_isolation.log`, `~/tmp/perf_avif_unpremul_tiers.log`
  (zenbench summaries reproduced below — the committed record)

| repo | bench | cell | v3(avx2) | scalar | paired 95% CI (scalar vs v3) |
|---|---|---|---|---|---|
| jxl-encoder | tier_isolation | encode lossy d1.0 512² (VarDCT) | 53.2 ±4.6 ms | 128.7 ±4.6 ms | +134.9%..+151.6% |
| jxl-encoder | tier_isolation | encode lossless 512² (modular) | 181.4 ±2.9 ms | 197.7 ±4.5 ms | +4.3%..+16.5% |
| zenavif | tier_isolation | yuv420→rgb8 512×256 | 153.1 ±1.1 µs | 221.2 ±1.8 µs | (quick pass, CI per-arm) |
| zenavif | tier_isolation | yuv420→rgb8 1920×1080 | 2.05 ms | 3.28 ms | (quick pass, CI per-arm) |
| zenavif | unpremul_tiers | unpremultiply8 512px | 4.85 GiB/s | 1.20 GiB/s | +276.7%..+350.1% |
| zenavif | unpremul_tiers | unpremultiply8 1920px | 4.04 GiB/s | 1.23 GiB/s | +281.6%..+294.7% (v3 CV=36%) |

Readings:
- The jxl **mode split reproduces on x86** exactly as the 2026-07-28 NEON record
  (`neon_tier_isolation_2026-07-28.meta`, M4: 2.10×/1.18×) predicted: VarDCT
  ~2.4× from the AVX2 tier, modular ~1.1× (serial predictor+entropy bound).
  Same conclusion, third architecture: the coding MODE decides the SIMD payoff.
- zenavif unpremultiply8 confirms the parallel lane's 4.0×@1920px zenbench claim
  (`09494c6`) on an independent run (throughput ratio 3.3–4.0× across sizes).
- `jxl-encoder-simd kernel_tiers` is an **aarch64-only bench by design** (prints
  `[kernel_tiers] aarch64-only bench` and exits 0 with no groups on x86_64) —
  the x86 jxl evidence is `tier_isolation` above + the per-kernel dispatch-parity
  test batteries; noted so nobody reads its empty x86 run as a missing tier.
- zenavif `tier_isolation` ran zenbench quick-mode (0-round warning, per-arm CI
  only); directionally consistent with the paired-CI rows and the byte-identity
  gates. Anyone needing paired CIs for yuv420 specifically: re-run without
  quick mode.
