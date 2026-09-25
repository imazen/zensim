//! Process-wide SIMD tier cap for bit-identical results across CPUs.
//!
//! The MLP trainer, the bake predictor and the panel statistics dispatch on the CPU's SIMD tier
//! (`incant!` over `X64V4Token` / `X64V3Token`). The AVX-512 kernels sum in a different order than
//! the AVX2 ones, so the same run gives different bits on an AVX-512 host and an AVX2 host.
//! `ZENSIM_MAX_TIER=v3` disables the AVX-512 tokens for this process so an AVX-512 host reproduces the
//! AVX2 result exactly. Unset (the default) leaves dispatch untouched.

use std::sync::atomic::{AtomicBool, Ordering};

static AVX512_ALLOWED: AtomicBool = AtomicBool::new(true);

/// False once `ZENSIM_MAX_TIER=v3` has been applied. The hand-written `simd_mlp` kernels pick their
/// AVX-512 variant with `is_x86_feature_detected!` rather than an archmage token, so they consult
/// this flag as well.
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
