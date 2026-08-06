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

---

# ADDENDUM — the decisive experiments (board-hygiene / rsqrt-investigation lane, 2026-08-05)

Run after this doc landed, using CI itself as the failing environment
(probe PRs #57/#58/#59, closed unmerged) plus the log archaeology below.
Every §3 "decisive next experiment" was run; two of this doc's readings are
CORRECTED by the results.

## A1. The archmage/magetypes lock movement is EXONERATED for the golden

`aaf9b808` did move the LOCK 0.9.26 → 0.9.28 (archmage, archmage-macros,
magetypes — verified in its `Cargo.lock` hunks; the manifest stayed 0.9.23).
But the golden was already red at 0.9.26 on fixed-Intel hardware:
`Test (macos-15-intel)` fails BOTH golden tests with the SAME 241/372
signature at `bb5373a4` (08-03 05:01, the EARLIEST completed run containing
the golden), `ab6c8991`, `5a8adee7`, and `926c71f7` — all four with
`archmage = 0.9.26` in their committed lock. A dependency bump cannot
explain failures that predate it.

## A2. The "ubuntu flip window" was runner-CPU lottery, and the 07-19 green was vacuous

* Probe PRs #57 (05739a53 verbatim) and #58 (lock pinned back to 0.9.26)
  ran the golden 4× each, twice (16 draws total), logging `/proc/cpuinfo`:
  **every draw landed on AMD EPYC (7763 Zen3 / 9V74 Zen4) and every draw
  PASSED — both arms, both feature configs.** The ubuntu-latest fleet mixes
  vendors; a per-run verdict is a per-run CPU draw.
* The 08-04 ubuntu-latest red (run 30919484848): its printed drift values
  are **bit-identical, line for line, to macos-15-intel's in the same run**
  (first 20 divergent features compared exactly). The ubuntu red was an
  Intel-class draw exhibiting the same phenomenon as the Intel Macs —
  glibc vs Apple libm producing IDENTICAL drift also eliminates libm as
  the mechanism for THIS divergence (it is instruction-level,
  vendor-dependent — the estimate-instruction class is the standing
  candidate; the precise op is NOT yet pinned).
* `v1_golden_bytes.rs` landed `f247746f` (07-18 23:39). `7d32671b` —
  the "passed 07-19, only windows failed" run this doc leaned on — is
  07-18 18:18, i.e. **before the golden existed**. There is no pre-08-03
  Intel-green observation; between 07-19 and 08-03 05:01 the push train
  cancelled every run.

**Corrected reading: the golden has NEVER been observed green on an
Intel-identified machine.** It was born environment-classed at capture
(07-18, AMD Zen4 / WSL): AMD x86 passes everywhere it has been tried
(Zen3 EPYC, Zen4 EPYC, Zen4 dev box, glibc 2.35/2.36/2.39); Intel x86
(mac + the identified ubuntu draws) fails by ~1e-10 on 241/372; ARM
classes fail in their own patterns (mac-ARM 246/372; windows-11-arm fails
both goldens; the M4 fails back to 07-19 — consistent, since there was
never an ARM-green either). One windows-latest x64 run at 05739a53 passed
the golden on an unidentified-CPU draw (that fleet is mixed too).

## A3. The LOCAL rsqrt-kernel failure has a DIFFERENT root — and there the lock bump IS the breaker

`rsqrt_path_precision_vs_scalar` (the opt-in rsqrt Adam kernel's precision
gate, AVX-512-only) was measured in a four-cell A/B on the Zen4 dev box:

| kernel | @0.9.26 | @0.9.28 |
|---|---|---|
| pre-repair (`_approx` + 1 hand-NR) | **PASS 1.117e-12** | FAIL 1.6653e-5 |
| repaired `22e37ce3` (`rsqrt()`/`recip()`) | FAIL 1.6653e-5 | **PASS 1.117e-12** |

The two diagonals are BIT-identical because the expression trees coincide
(0.9.27's archmage `34f34b2` moved one Newton step across the API line:
`_approx` went refined→raw, `rsqrt()`/`recip()` went ~28-bit→full). So the
local failure began exactly at `aaf9b808` and is fixed by `22e37ce3` +
the `Cargo.toml` 0.9.28 minimum. **Joint determination: the local rsqrt
failure and the CI golden drift are DISTINCT roots under one theme —
exact-f64 expectations crossing a (CPU-vendor × libm × dependency-semantics)
class boundary.**

## A4. Windows red, itemized (the red-masking-red picture)

* `windows-latest` (x64) at 05739a53 fails exactly ONE test:
  `tests::corpus_slots_are_relative_or_declared_pinned` — a stale
  exemption: "hf_nearlossless is declared pinned-outside-features-root but
  its slot is relative now. Remove the stale exemption." (bake_verdict.rs
  slot audit; fires on Windows path semantics). The golden PASSED there.
* `windows-11-arm` at 05739a53 fails the two golden tests (the ARM class,
  per A2).
* Main CI: no successful run since 2026-07-16; nearly every 07-19..08-03
  run was cancelled by the push train's concurrency group, which is what
  made the golden's birth-state unobservable for two weeks.

## A5. Remedy options (USER decision per the never-relax rule — presented, not picked)

1. **Per-environment-class goldens** — capture one golden per
   (CPU-vendor × libm) class the matrix runs; exact-f64 holds within class.
2. **Tolerance-class gate cross-class** — exact on the capture class,
   bounded-relative elsewhere (observed drift ≤ ~1e-9): codifies that
   byte-identity never held cross-vendor.
3. **Pin the golden to the capture class** — run it only on
   AMD-x86-identified runners (or a self-hosted runner); other platforms get
   a visible SKIPPED-by-policy row, never a silent green.
4. **Make the pipeline vendor-invariant** — hunt the divergent op(s)
   (estimate-instruction class) and replace with IEEE-exact equivalents;
   the only option that makes one golden legitimately universal, at a perf
   and investigation cost.

Probe hygiene: PRs #57/#58/#59 closed unmerged, probe branches deleted.

---

# RESOLUTION — CLOSED BY POLICY (tolgold lane, 2026-08-05)

**USER RULING (verbatim): "the golden-gate policy is tiny tolerances, not
per-class exactness; and full-precision fallbacks are acceptable only if
runtime-optional via generics — prefer archmage's precision-TIERED
variants."** This is §A5 remedy 2 (tolerance-class gate) as a decision, not
an option list. Implemented in the same change that records this section.

## R1. Correction: the drift is ~6e-8 ABSOLUTE / up to 2.8e-4 relative — not "~1e-10 relative"

This doc's §"What fails" figure ("~1e-10..1e-12-relative drift") was the
panic printout's **abs-delta column misread as relative**. The full
372-feature cross-class drift was measured for the tolerance derivation
(Apple M4 native `capture_v1_golden` at main `7577dfa6` vs the AMD-Zen4
goldens; CI run 31048977926 logs confirm Intel x86 + Windows/Linux ARM
print bit-identical divergent values — ONE shared non-AMD result set):

| fixture | divergent | max abs delta | max relative |
|---|---|---|---|
| GOLDEN_SYNTHETIC | 246/372 | 2.09e-8 (f102) | 6.06e-7 (f89, scale 2.7e-2) |
| GOLDEN_REAL | 241/372 | 6.00e-8 (f62) | 2.76e-4 (f62 — its scale is only 2.2e-4) |

The drift is ABSOLUTE-shaped (≤6e-8 at every feature regardless of scale);
big relative numbers appear only on tiny-magnitude features.

## R2. The converted gate (zensim/tests/v1_golden_bytes.rs)

- `v1_*_fixture_matches_golden`: `|Δ| <= max(1e-6, 1e-5·scale)` per
  feature. ABS floor 1e-6 = 16.7× above measured max abs (6.00e-8);
  REL 1e-5 = 16.5× above measured max relative among features ≥1e-2 scale
  (6.06e-7), and equals the relative allowance v2 has always had. Full
  derivation lives on `assert_golden_close` in the test file.
- `v1_same_class_determinism_bitexact` (NEW): byte-exactness retained as a
  SAME-CLASS property — two computes on one machine must agree bit-for-bit
  on all 372 features, every runner class. Determinism was not lost in the
  conversion; only the false cross-vendor-portability claim was.

## R3. Companion: the rsqrt Adam kernel is now precision-TIERED

Per the same ruling's second clause, `22e37ce3`'s unconditional
full-precision repair was restructured: `adam_simd.rs` gained
`RsqrtPrecisionTier` (generic, monomorphized) with `RsqrtFull` (2 NR,
default — the only tier production would use), `RsqrtNr1` (1 NR, the
pre-repair expression tree as a NAMED tier), and `RsqrtEstimate` (raw
14-bit), runtime-selected via `adam_update_rsqrt_v4_tiered`. Measured
per-tier max w-relative error on the precision fixture (Zen 4): full
1.117e-12 (gate 1e-9), nr1 1.665e-5 (gate 1e-3), estimate 1.116e-1
(gate 1e0) — `tests/adam_simd_rsqrt_precision.rs` carries the
derivations; tier costs: `benchmarks/adam_rsqrt_tiers_2026-08-05.md`.

Status: **CLOSED-BY-POLICY.** The un-pinned vendor-divergent op hunt (§A5
remedy 4) remains NOT pursued — the tolerance gate makes it unnecessary for
CI health; reopen only if a product need for cross-vendor bit-identity
appears.
