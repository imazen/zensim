# x86 SIMD coverage survey — jxl-encoder · zenavif · zenrav1e (2026-08-25)

Goal criterion 5 asks for x86 SIMD tiers "for every kernel that is NEON-only".
The user's premise was "jxl and avif are only arm optimized at this time". This
survey (read-only, source-verified 2026-08-25) tests that premise. **It is mostly
false for jxl and true for exactly one hot zenavif kernel.**

Method: grep each repo's `src/` for aarch64/NEON SIMD (`core::arch::aarch64`,
`neon`, `vld1`/`vld4`, `target_feature(enable="neon")`), then check each NEON
kernel for an x86 sibling (`core::arch::x86_64`, `avx2`/`avx512`, or a
`#[magetypes(...)]`/`incant!` dispatch that already emits x86). magetypes/archmage
generic bodies are multi-arch by construction and are NOT gaps.

## Result by repo

| repo | verdict | magetypes/archmage dep |
|---|---|---|
| **jxl-encoder** | NOT NEON-only — criterion effectively CLOSED | YES, `0.9.27` (current); already forwards an `avx512` feature |
| **zenrav1e** | NOT NEON-only — x86 is strictly AHEAD (has `ec`/`lrf`/`quantize` asm modules aarch64 lacks) | NO (only transitive in Cargo.lock) — don't add for this |
| **zenavif** | ONE genuine gap: `unpremultiply8` | YES but STALE `0.9.15` (jxl is on 0.9.27) |

### jxl-encoder — closed
43 hand-written NEON/AVX2 kernel pairs + 6 magetypes-consolidated bodies
(`gaborish5x5.rs:153`, `pixel_loss.rs:178`, `entropy.rs:215/452/583`,
`xyb.rs:398/559`) + 4 dual-arch parent-crate kernels
(`ac_strategy_search.rs`, `reconstruct.rs`). The 18 "unpaired" `*_neon` names are
internal helpers of a NEON kernel whose top-level fn already has an `_avx2` twin.

### zenrav1e — closed
`src/asm/x86/mod.rs` = {cdef,dist,ec,lrf,mc,predict,quantize,transform};
`src/asm/aarch64/mod.rs` = {cdef,dist,mc,predict,transform}. The only Rust-intrinsics
kernel (forward transform) is dual-arch (`asm/{aarch64,x86}/transform/forward.rs`,
shared `impl_1d_tx!` body). Nothing NEON-only.

## The work-list (ranked)

### 1. zenavif `unpremultiply8` — THE only real x86 arch gap, and it is hot
- `zenavif/src/simd/unpremul.rs:75` (NEON kernel); dispatch `:122` has only an
  `aarch64` arm → x86/wasm hit `unpremultiply8_scalar` (`:110`).
- Runs once per row on every **alpha-bearing** AVIF, buffered
  (`convert.rs:252`) AND streaming (`strip_convert.rs:359`). Per-pixel
  `min(255,(c*255+a/2)/a)`, `a==0` left untouched.
- The scalar path has an integer divide, so it provably cannot autovectorize —
  the x86 gap is real, not compiler-recoverable.
- FREE correctness oracle: exhaustive `(channel,alpha)` tests at `unpremul.rs:163`
  + `tests/unpremul8_exhaustive.rs`; bench `benches/unpremul_tiers.rs`.
- magetypes suitability GOOD for the math (widen/mul-add/f32-divide/min/select),
  but the RGBA deinterleave is arch-specific: NEON `vld4q_u8`/`vst4q_u8` is one
  instruction; **AVX2/AVX-512 need a `pshufb`/permute network** (or a strided
  load). Plan: RGBA↔planar transpose as a per-tier helper, math body shared.
- **Prerequisite:** bump zenavif magetypes/archmage `0.9.15 → 0.9.27`
  (`zenavif/Cargo.toml:30-31`). If the bump ripples, the lower-risk alternative is
  a hand-written `unpremultiply8_avx2` via `core::arch::x86_64` gated by an
  archmage `Desktop64` token — the exact dual-arch pattern
  `zenavif/src/yuv_convert_fast.rs` already uses (AVX2 `:21` / NEON `:367`).

### 2. (tier, not arch) jxl `xyb.rs:398` forward_xyb — add a `v4` tier
Only magetypes body still spelled `v3, neon, wasm128, scalar` with no `v4`
(`define(f32x8, f64x4)`); every other consolidated jxl body carries `v4`. Forward
XYB runs per-pixel on every encode. Caveat: the `f64x4` lane in the define list
needs an AVX-512 f64 mapping check before `_v4x`.

### 3. (tier, not arch) zenavif `yuv_convert.rs:113/495` incant sites — add `v4`
`yuv420/422_to_rgb8_inner` dispatch `v3, neon, wasm128, scalar`; adding `v4`/`_v4x`
(after the magetypes bump) is a straight tier extension.

## Out of scope for criterion 5 (reverse gaps — NEON missing, x86 present)
Recorded so they aren't re-discovered as "x86 gaps":
- zenavif strip path `yuv_convert.rs:753/826/899/930/961/992` (`yuv*_to_rgb8_strip`)
  dispatches `Desktop64`+`Wasm128` only → **aarch64** falls to scalar. This is the
  largest arch hole in zenavif but it is NEON work, on the live streaming decode
  path (`strip_convert.rs`, `codec.rs:4058`).
- jxl `special8x8.rs` (`identity_*`, `dct2x2_*`) + `fused_dct8.rs`
  (`fused_dct8_entropy`) + `dequant.rs` `dequant_8` are x86-only → NEON gap.

## Verdict for criterion 5
The criterion is far smaller than "jxl and avif are arm-only" implied: **jxl and
zenrav1e are already x86-covered.** The single production x86 deliverable is
zenavif `unpremultiply8` (headline), with two optional `v4`-tier extensions.
Sequence it after the model wave; gate it on the exhaustive test + a zenbench
A/B (no `-C target-cpu=native`).
