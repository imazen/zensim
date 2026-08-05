# v1 golden byte-identity gate: environment-sensitivity triage (2026-08-05)

Recorded by the appendix-O (HF-NL) session while chasing its own CI gate; the
failure PREDATES every appendix-O commit and is an independent, older breakage.
Owner lane: the board-hygiene session's "rsqrt-investigation" (`.workongoing`).

## What fails

`zensim/tests/v1_golden_bytes.rs` (`v1_synthetic_fixture_matches_golden` +
`v1_real_fixture_matches_golden`) — the EXACT-`f64` byte-identity gate over the
v1-372 extraction, golden captured 2026-07-18 on the dev box (AMD Zen4, WSL
glibc 2.35). Failure shape everywhere: ~1e-10..1e-12-relative drift on
241-246 of 372 features **starting at f0** (i.e. upstream in the basic
pipeline, not one pool), with the DIVERGENT COUNT varying by platform
(ubuntu-latest 241, macos-latest 246) — the signature of vendor/libm-level
numeric differences, not a semantic change.

## Timeline facts (all verified against run logs, 2026-08-05)

| environment | state | evidence |
|---|---|---|
| CI windows-latest / windows-11-arm | red since ≥07-18, DIFFERENT tests (`corpus_slots_are_relative_or_declared_pinned`, should_panic classes) | run 29633223823 + 07-19 run |
| CI macos-latest + ubuntu-24.04-arm | golden red by ≥08-03; PASSED 07-19 (run for 7d32671b: only windows failed) | runs 30884946497/30891127917 fail on other tests pre-08-04, golden appears 08-04 14:32 run |
| **CI ubuntu-latest** | golden **green at head `926c71f7`** (run 30891127917, ubuntu+all-features SUCCESS) → **red at `05739a53`** (run 30919484848, 08-04 14:32) | binary-searched over all completed runs |
| local WSL (AMD Zen4, glibc 2.35, rustc 1.97.1) | PASSES at every commit incl. tip | this session |
| container glibc 2.36 (rust:1.97-bookworm) at tip | PASSES | `~/tmp/hfnl/ct_bookworm.log` |
| container Ubuntu 24.04 / glibc 2.39 / rustc 1.97.1 (runner-faithful) at tip, workspace tree AND fresh origin clone | PASSES | `ct_u2404.log`, `ct_fresh.log` |
| **Mac M4 Pro, macOS 26.5.2, rustc 1.97.1, native** | **FAILS at tip AND at `926c71f7`, `299ccc8c`, `c8e1c440` (08-02), `0b3d16b0` (07-27), `9f9cef56` (07-19)** — i.e. at commits where CI-macos was green | `~/ztriage/zensim` on the mac |

Also eliminated on the dev box: AVX-512-vs-AVX2 tier (V4x token disabled → all
372 bit-identical to golden), RAYON_NUM_THREADS ∈ {1,2,4}, workspace feature
unification, `--all-features`.

## Reading

1. The gate is **environment-fragile**: exact-`f64` equality holds only inside
   one (CPU vendor × libm version) class. My M4 (macOS 26 libm) fails at
   commits where CI's macos-15 runners passed ⇒ the gate cannot be
   bisected on a machine whose libm differs from the capture box's class.
2. The **ubuntu-latest flip is commit-driven**: green at `926c71f7`, red at
   `05739a53`, while every AMD-linux environment (glibc 2.35/2.36/2.39)
   passes BOTH endpoints. The flip therefore rides an interaction between a
   window change and the runner's (Intel) CPU. In-window suspects, in order:
   - **`aaf9b808` — Cargo.lock bump `archmage`/`magetypes` 0.9.26 → 0.9.28**
     (SIMD primitive crates under every kernel; a changed rsqrt/rcp-class
     approximation or horizontal-reduction shape is vendor-dependent by
     nature; `feature_v2.rs:3275` uses `.rsqrt()`);
   - `299ccc8c` — feature_v2.rs +367 / attribution.rs +189;
   - (de3482dd / ae852b1b / 92a23417 are validate-side, near-certainly not.)
3. Decisive next experiments (need an **Intel x86** box — no household AMD box
   can reproduce the ubuntu flip): at `926c71f7` vs `05739a53` run the golden;
   then at `05739a53` with the lock's archmage/magetypes pinned back to
   0.9.26 (`cargo update -p archmage --precise 0.9.26 -p magetypes --precise
   0.9.26`). If reverting the pin greens it, the bump is the breaker and the
   fix is (a) upstream archmage regression fix, or (b) re-capture goldens
   under a DOCUMENTED environment class + a tolerance-based cross-arch tier
   (the brief's zero-tolerance stance can hold per-class, not cross-vendor).

## Meta

Main CI has had **no successful run since 2026-07-16** (windows red since
≥07-18 on unrelated tests, everything else cancelled by the push train or red
per the table). "CI green" has been unverifiable for ~3 weeks; the golden
regressions above were invisible inside that. Worth a standing rule: a red
window platform must not be left to mask new red on the primary platforms.
