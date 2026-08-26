# x86 SIMD coverage survey — jxl-encoder · zenavif · zenrav1e (2026-08-25)

Goal criterion 5 asks for x86 SIMD tiers "for every kernel that is NEON-only".
The user's premise was "jxl and avif are only arm optimized at this time". This
survey (read-only, source-verified 2026-08-25) tests that premise. **It is mostly
false for jxl and true for exactly one hot zenavif kernel.**

> **UPDATE 2026-08-26 — the one mandatory gap is CLOSED.** Task #1 (zenavif
> `unpremultiply8` x86 tier) shipped in `zenavif` commit `b92880e` ("feat(simd):
> x86 AVX2 tier for unpremultiply8"): a `Desktop64`/AVX2 kernel
> (`src/simd/unpremul.rs:112`) with the dispatch (`:189`) now routing x86_64 to it
> instead of the scalar fallback. Verified 2026-08-26 on an x86_64 box: the
> **exhaustive `(channel, alpha)` oracle passes bit-exact** through
> `unpremultiply8_dispatch` (`exact_over_complete_domain`, `tail_lengths_exact`,
> `edge_alpha_semantics` all green — on x86 these exercise the AVX2 path). So the
> mandatory half of criterion 5 is DONE; only the two OPTIONAL `v4`/AVX-512 tier
> extensions below (tasks #2, #3 — speedups on AVX-512 hardware, not arch gaps)
> remain, each with the noted prerequisite (jxl `f64x4` AVX-512 mapping check;
> zenavif magetypes `0.9.15 → 0.9.27` bump). The module doc-comment header of
> `unpremul.rs` still reads "NEON unpremultiply" and predates the AVX2 body — a
> cosmetic staleness, not a functional gap.

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
| **zenavif** | ~~ONE genuine gap: `unpremultiply8`~~ → **CLOSED 2026-08-26 (`b92880e`, AVX2 tier, exhaustive oracle verified on x86)** | STALE `0.9.15` — but the AVX2 tier is hand-written via `core::arch::x86_64` + `Desktop64` (the lower-risk path the work-list named), so the magetypes bump was NOT needed for task #1 |

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

### 1. zenavif `unpremultiply8` — ✅ CLOSED 2026-08-26 (`b92880e`) — was THE only real x86 arch gap, and it is hot
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

## PERF RESULT — zenavif unpremultiply8 AVX2 (measured 2026-08-25)

Kernel committed: zenavif `b92880e3` (`src/simd/unpremul.rs` `unpremultiply8_avx2`),
correctness bit-identical to scalar over the complete (channel,alpha) domain.

zenbench `benches/unpremul_tiers.rs` (`--features _dev`, x86_64, Ryzen 9 7950X
class, NO `-C target-cpu=native`), v3(avx2) vs forced-scalar:

| row width | v3(avx2) | scalar | speedup | avx2 throughput |
|---|--:|--:|--:|--:|
| 1920 px | 1.5 ±0.7 µs | 5.0 ±0.1 µs | **~3.3×** (CI +160%..+333%) | 4.75 GiB/s |
| 512 px | 361 ±108 ns | 1309 ±174 ns | **~3.6×** (CI +216%..+321%) | 5.29 GiB/s |

The win is unambiguous (CI never crosses zero) and beats the NEON path's recorded
2.7×. **Caveat:** only 4 clean rounds — zenbench's resource gate discarded the
rest because the box was running a training wave (drift r=−0.80: later rounds
faster as it freed). A quiet-box re-run would tighten the MAD but cannot change
the direction. Criterion-5's one real gap is CLOSED (correctness + perf).
Optional follow-ups (not blocking): a uniform-alpha fast-path (scalar skips the
divide on a==255/0 pixels; the SIMD path divides unconditionally, same as NEON),
and the two `v4`-tier extensions (jxl forward_xyb, zenavif yuv inner).

## UPDATE 2026-08-26 — the jxl forward_xyb v4 tier is BLOCKED (not a NEON gap)
Followed up on the "add a v4 tier to jxl `forward_xyb`" item. Two facts settle it:
1. **It is NOT a NEON-only gap** — `forward_xyb_impl` already carries the `v3` (AVX2) tier
   (`X64V3Token: F64x4Backend` exists in magetypes 0.9.28, x86_v3.rs:867), so x86 IS covered. The
   criterion-5 requirement (NEON-only → x86) does not apply; adding `v4` would be an AVX-512 PERF
   upgrade, not a coverage fix.
2. **The v4 (AVX-512) tier is blocked on magetypes** — the kernel uses `f64x4` (cube-root Newton),
   and magetypes 0.9.28 still has **no `F64x4Backend for X64V4Token`** (impls exist only for X64V3 /
   Neon / Wasm128 / Scalar). So `#[magetypes(..., v4, ...)]` cannot compile the f64x4 lane for
   AVX-512 until magetypes adds that backend. This is the exact cap the xyb.rs:34 comment records.
**Verdict:** C5's one genuine NEON-only kernel (zenavif `unpremultiply8`) is shipped (`b92880e3`,
~3.3–3.6×, byte-identical). The v4 extensions are optional perf upgrades; the jxl one is
magetypes-gated (upstream feature), the zenavif yuv `v4` remains available (pure-f32) but is likewise
a perf upgrade, not a coverage gap. **Criterion 5 (x86 for every NEON-only kernel) is MET.**
