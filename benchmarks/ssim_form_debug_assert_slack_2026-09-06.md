# `ssim_form`'s boundedness `debug_assert!` is 8x too tight — a DEBUG build panics at revision 2

**Found 2026-09-06 by the zenmetrics GPU-copy lane**, which had to disable debug
assertions to run its parity suite and traced why. **Reported here, not fixed
here** — one line of arithmetic slack in another lane's owner is exactly the kind
of change that should be made by whoever owns the constant, with its own gate.

## The defect

`zensim/src/ssim_form.rs:427` checks the boundedness claim the bounded arms make:

```rust
debug_assert!(
    !form.bounds_dissim() || (-1e-5..=2.0 + 1e-5).contains(&d),
    "bounded SSIM form {form:?} produced d = {d} outside [0, 2]"
);
```

The comment says the tolerance "is for f32 rounding at the `num_s/denom_s = 1`
boundary, not for slack in the claim". **MEASURED: that rounding reaches 7.6e-5 on
real content — about 8x the 1e-5 allowed** — so at revision 2 a debug build panics
inside zensim on `cpu_gpu_diffmap_parity` and `pu_xyb_parity`.

## What it is NOT

* **Not caused by the GPU port.** It fires identically on the pre-port tree; the
  port only made a rev2 debug run happen for the first time.
* **Not a boundedness failure.** `d` is inside `[0, 2]`; it is the *assert's*
  tolerance that is too small, not the arm that is unbounded.
* **Not a release-build issue.** `debug_assert!` compiles out; every shipped byte
  is unaffected, and with assertions off both revisions read the identical
  failure set.

## Why it matters anyway

It makes `cargo test` (a debug profile by default) unusable for any rev2 path that
reaches the bounded arms through the PU-XYB route — which is the ONLY route that
reaches F4 at all (see below). A gate you have to disable to run is not a gate.

## A second fact from the same lane, which changes how F4 should be described

**F4 cannot be reached from 8-bit sRGB at all.** Over 29,700 `to_bits()` feature
values across 3 sizes x 5 fixture pairs x 3 regimes x {cold, warm-ref} + strip,
`Clamp` moves **0**. It is the **PU-XYB (HDR) route** that makes it live: 192
SSIM-derived values move there, with `ssim_max` going **5.4275 -> 1.0**.

That is consistent with — and sharper than — R6's "F4's pathology occurs on NONE
of the 217,756 rows", because every corpus in R6 is SDR. It means **an SDR fixture
is vacuous for F4 by construction**, so any future F4 test must state which route
it exercises or it is testing nothing.

## Suggested fix, for the owner to weigh

Derive the tolerance from the f32 epsilon at the `num_s/denom_s = 1` boundary
rather than picking a round number, and put the derivation in the message. The
measured 7.6e-5 is a lower bound on what real content needs, not the answer.
