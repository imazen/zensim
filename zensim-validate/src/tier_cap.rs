//! Process-wide SIMD tier cap: a diagnostic that switches between bit-identical kernels.
//!
//! `ZENSIM_MAX_TIER=v3` disables the AVX-512 archmage tokens (`X64V4Token`, `X64V4xToken`) for the
//! process, so the trainer dispatches to its AVX2 kernels on an AVX-512 host. Unset (the default)
//! leaves dispatch untouched; any other value panics.
//!
//! What it changes now. From commit `2c317b7a` on, every tier-dispatched kernel the trainer binaries
//! (`zensim_mlp_train`, `bake_dial_refit`, `panel`) reach is bit-identical across tiers: `simd_mlp`
//! forward/backprop, `adam_simd`, `simd_encoder`, and the AVX path of `add_l2_grad_layer1`
//! (mul+add). The AVX-512, AVX2, scalar, NEON and wasm128 kernels all reproduce the AVX2 bits, so the
//! cap no longer changes any result; running capped and uncapped and comparing is a cheap tier-parity
//! check. The bake predictor (zenpredict inference, scalar `f32::mul_add`) and the panel statistics
//! (zenstats) have no tier dispatch at the pinned revisions, so the cap never affected them.
//!
//! Where it is still mandatory: binaries built before `2c317b7a`. Their AVX-512 kernels sum in a
//! different order, so an AVX-512 host gives different bits than an AVX2 host unless the cap is set.
//! That covers the deployed `fit-p0/p2d2-v7` and v8 images, which were built from `83461101` and
//! `83a205ad`; keep the launcher guard for them.
//!
//! Not covered: zensim metric feature extraction from pixels is still tier-dependent (the generic
//! `f32x16` reducers sum in AVX-512 order on v4 and in AVX2 order on v3), and no extraction binary
//! calls [`apply_from_env`].

use std::sync::atomic::{AtomicBool, Ordering};

static AVX512_ALLOWED: AtomicBool = AtomicBool::new(true);

/// False once `ZENSIM_MAX_TIER=v3` has been applied. `simd_mlp` consults this flag next to
/// `X64V4Token::summon()`; the token disable already makes that call return `None`, so the flag is
/// redundant with it but harmless.
pub fn avx512_allowed() -> bool {
    AVX512_ALLOWED.load(Ordering::Relaxed)
}

/// Apply `ZENSIM_MAX_TIER` from the environment. Call first thing in `main`.
///
/// Panics on an unrecognised value or when the token cannot be disabled, so a mistyped cap never
/// silently runs at the wrong tier.
pub fn apply_from_env() {
    let Some(value) = std::env::var_os("ZENSIM_MAX_TIER") else {
        return;
    };
    match value.to_str() {
        Some("v3") => {
            AVX512_ALLOWED.store(false, Ordering::Relaxed);
            cap_to_v3();
        }
        other => panic!("ZENSIM_MAX_TIER={other:?}: only \"v3\" is supported"),
    }
}

#[cfg(target_arch = "x86_64")]
fn cap_to_v3() {
    use archmage::{X64V4Token, X64V4xToken};
    X64V4xToken::dangerously_disable_token_process_wide(true)
        .unwrap_or_else(|e| panic!("cannot cap tier: {e}"));
    X64V4Token::dangerously_disable_token_process_wide(true)
        .unwrap_or_else(|e| panic!("cannot cap tier: {e}"));
    eprintln!("tier cap: AVX-512 tokens disabled (ZENSIM_MAX_TIER=v3)");
}

#[cfg(not(target_arch = "x86_64"))]
fn cap_to_v3() {}
