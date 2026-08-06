# Adam rsqrt-kernel precision tiers: cost + error (2026-08-05)

Companion to the tolgold-lane restructuring of the opt-in rsqrt Adam kernel
(`zensim-validate/src/adam_simd.rs`) after the 2026-08-05 user ruling:
*"full-precision fallbacks are acceptable only if runtime-optional via
generics — prefer archmage's precision-TIERED variants."*

## Design

- `RsqrtPrecisionTier` trait (zero-sized selector types, monomorphized into
  the kernel body via an `#[inline(always)]` generic + one thin `#[arcane]`
  wrapper per tier — no per-iteration dispatch cost).
- Tiers, built from what magetypes 0.9.28 ships for f64 (exactly two
  precision levels per op — the f32-only `_portable`/`_newton` families do
  not exist for f64, so the middle tier hand-rolls ONE Newton-Raphson step
  using magetypes' own step formulas verbatim: recip `r·(2−a·r)`, rsqrt
  `0.5·y·(3−a·y·y)`; magetypes/src/simd/impls/x86_v4.rs):

| tier | rsqrt / recip construction | step precision |
|---|---|---|
| `RsqrtFull` (default) | magetypes `rsqrt()` / `recip()` = VRSQRT14PD/VRCP14PD + 2 NR | ~52-53 bits |
| `RsqrtNr1` | `rsqrt_approx()`/`rcp_approx()` + 1 mirrored NR step | ~28 bits |
| `RsqrtEstimate` | raw `rsqrt_approx()` / `rcp_approx()` | 14 bits (2^-14 spec bound) |

- Runtime selection: `adam_update_rsqrt_v4_tiered(args, RsqrtPrecision::{Full,Nr1,Estimate})`;
  `adam_update_rsqrt_v4(args)` keeps its historical meaning (= Full).
  AVX-512-only — AVX2 has no f64 reciprocal estimate (magetypes f64x4
  `rsqrt()`/`recip()` are exact sqrt/div there), so tiering below v4 is a
  structural no-op. Non-AVX-512 hosts fall back to the exact scalar
  reference at every tier.
- The production Adam path (`adam_update`, vsqrtpd+vdivpd) is untouched and
  remains bit-identical to scalar; the tiers live on the opt-in rsqrt path.

## Measured per-tier error (Zen 4 / 7950X WSL2, fixture seed 0xC0DE, n=47,616, t=100)

Max/mean relative error in `w` vs the scalar reference — the w-relative
metric amplifies the update-step error by up to ~2e3× through cancellation
where `w - step` lands near zero (worst index i=35921, v≈2.4e-8). Gated in
`zensim-validate/tests/adam_simd_rsqrt_precision.rs` with derivations.

| tier | max_rel (measured) | mean_rel (measured) | gate |
|---|---|---|---|
| full | 1.117e-12 | 1.009e-16 | 1e-9 |
| nr1 | 1.665e-5 | 5.533e-10 | 1e-3 (+ must exceed 1e-9: catches silent re-refinement) |
| estimate | 1.116e-1 | 1.021e-5 | 1e0 |

The nr1 number equals the pre-repair kernel's four-cell A/B value
(`benchmarks/v1_golden_env_triage_2026-08-05.md` §A3) exactly — the tier
reproduces that expression tree by construction, turning the six-week
accidental degradation into a named, bounded, opt-in configuration.

## Measured tier cost (zenbench, same box)

`cargo bench -p zensim-validate --bench adam_bench`, `run-heavy --jobs 6`,
commit = this change; full output `~/tmp/tolgold/adam_bench.log` (summary
committed here per the benchmark-results rule).

`adam_w1_only_47616_params` group (the dominant per-step array), 4 rounds ×
30 calls, paired vs `scalar`:

| arm | mean ±mad | params/s | note |
|---|---|---|---|
| scalar | 91.1 ±2.0 µs | 522M | CV=49% |
| simd (production: vsqrtpd+vdivpd) | 36.5 ±0.7 µs | 1.30G | |
| simd_rsqrt_full (2+2 NR) | 40.5 ±5.8 µs | 1.17G | |
| simd_rsqrt_nr1 (1+1 NR) | 40.7 ±4.6 µs | 1.17G | CV=21%, drift flagged |
| simd_rsqrt_estimate (raw) | 36.6 ±7.2 µs | 1.30G | CV=56% |

run-heavy: rc=0, 548 s, peak-RSS 0.79 GiB, min-avail 27,547 MiB, peak-load
27.84. **Noise caveat:** the box was concurrently loaded by other agent
lanes (zenbench flagged 233 noisy rounds and CVs of 21-56%); the
cross-family ordering below is consistent and paired, but the
full-vs-nr1 delta is inside noise — treat per-arm means as ±10%-class.

## Reading

- On Zen 4 the PRODUCTION sqrt+div path (`adam_update`) remains
  fastest-or-tied — re-confirming the 2026-05-17 finding that motivated
  keeping the rsqrt path opt-in. No caller should switch on these numbers.
- `estimate` ties production simd (~36.5 µs); `full` and `nr1` sit ~11%
  behind both. The kernel at this size is loop/memory-bound, not
  FP-divider-bound, so removing NR steps buys little on this µarch — nr1 ≈
  full within noise. The tier mechanism's value here is OPTIONALITY (per
  the ruling) for µarchs where the FP-unit balance differs (Zen 5 /
  Sapphire Rapids), not a Zen 4 win.
- Precision-per-cost is therefore strictly ordered while cost is nearly
  flat: full (1.1e-12 max w-rel) ≥ nr1 (1.7e-5) ≥ estimate (1.1e-1) at
  ~equal Zen 4 runtime — which is exactly why `RsqrtPrecision::Full` is the
  default and the lower tiers are explicit opt-ins.
