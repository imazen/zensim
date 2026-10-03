# REV4VEC — decisions and open questions

Lane REV4VEC (SWE-2, 2026-10-03). Recorded before implementation; ASK
items need coordinator/user confirmation.

## V1 — Vectorized canon is the SAME formula, not a new approximation

The canonical scalar bodies in `det_math.rs` are byte-for-byte
replications of magetypes' generated `*_midp` vector formulas (verified
op-for-op against
`simd/generic/generated/transcendentals_f32x{8,16}.rs`, 0.9.28). On
tiers whose `mul_add` is a real FMA — x86 v3/v4/v4x and NEON — the
generated vector functions therefore produce, per lane, exactly the bits
the scalar canonical body produces. **Decision:** the vectorized canon
drivers call the generated `f32x16` midp/`cbrt` methods and the fused
opsin `mul_add` chains directly; no new polynomial, seed, or rounding is
introduced. This is what "the same per-element formulas lane-parallel"
means mechanically — not a re-derivation.

## V2 — Tier routing: fused tiers vectorize, scalar/wasm128 keep the scalar body

`mul_add` is unfused on the magetypes scalar and wasm128 backends, so
the vector form is NOT bit-identical to `f32::mul_add` there. The canon
drivers dispatch through `incant!` with:

- `v4x`, `v4`, `v3`, `neon` → one `#[magetypes]` generic body over
  `f32x16<T>` (native zmm on v4/v4x, 2×ymm FMA on v3, 4×`vfmaq` on
  NEON) + scalar-canon remainder.
- `wasm128` → the scalar canonical body for every pixel.
- `scalar` → the scalar canonical body for every pixel.

Wasm128 could in principle run `cbrt_midp` (FMA-free → lane-identical)
but the opsin `mul_add` mix is unfused there, so the WHOLE leaf must
stay scalar; no hybrid.

## V3 — One generic chunk width (16 lanes) for all fused tiers

A single `#[magetypes(v4x, v4, v3, neon)]` body over `GenericF32x16`
serves every fused tier: native 512-bit on v4/v4x, two fused ymm ops on
v3, four fused `vfmaq` on NEON. Per-lane arithmetic is identical by
construction (verified: x86-v3 `f32x16::mul_add` delegates to two
`_mm256_fmadd_ps`). Chosen over duplicating `OpsinChunk` per width —
one body to audit, one remainder rule.

## V4 — Remainders always run the scalar canonical body

Every vector driver processes `n / 16` full chunks then loops
`opsin_px_canon` / `pu_xyb_pixel_canon` (etc.) over the tail — the same
scalar body, bit-identical per element by construction (this matches the
canon drivers' existing convention; the production kernels' `cbrtf_fast`
tail is NOT used).

## V5 — Resolved: HDR transfer rows vectorized (PQ row + HLG primaries row)

`decode_pq_row_at_revision` (f32 PQ in) and
`decode_hlg_row_in_primaries_at_revision` both route Rev4 through
`#[magetypes]` f32x16 drivers now — the same tier routing as V2,
scalar/wasm128 stubs on the scalar canonical body. The PQ16 RGBA LUT
path stays untouched (LUT init is cold; per-pixel lookup is already
O(1)). Neither row is on the measured SDR workload (featpot bake is
rgb8) — implemented anyway per the task's "PQ/HLG transfer" scope, and
covered by the same per-tier bit-parity tests as the SDR leaves.

HLG specifics verified against the scalar body: `f32::clamp(v, 0.0,
1.0)` is a nested select (`v<0 → 0`, `v>1 → 1`, else v) — NOT
`max.min`, which would rewrite NaN→1 and −0→+0; the vector form uses
the equivalent nested `blend`. The `v ≤ 0.5` select is `simd_le` +
`blend`, `ys = Σ luma·e` stays UNFUSED mul/add (matching the scalar
body's unfused `a*b + c*d + e*f` — Rust never contracts to FMA), and
`pow_midp`/`exp_midp` are the generated forms whose op order the scalar
canon replicates.

## V6 — Resolved: `exact`/oracle arm untouched

`#[cfg(feature = "oracle")] mode.exact()` arms stay scalar f64 — they
are the ruler, not the measured path.

## V7 — NaN/special-input semantics are structural, not patched

The canonical opsin leaves kill non-finite inputs deterministically:
`mixed.max(0)` maps any NaN to +0 before `cbrt_midp`, so NaN pixels
produce identical finite outputs on every tier (verified by test, not
assumed). The midp gate additionally checks every compiled tier on NaN
*inputs* with both-NaN-equal output convention — the only spot where a
payload could legitimately differ — while NaN-vs-finite or
finite-bit divergence still fails. The leaf tests themselves are
strict `to_bits` and pass on NaN-bearing input batches.
