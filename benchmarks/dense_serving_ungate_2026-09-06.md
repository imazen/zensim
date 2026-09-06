# The dense-serving gather was `#[cfg]`-gated; four shipped profiles were silently mis-scored in every build without `feature-regime-v2`

**Date:** 2026-09-06. **Lane:** `densegate`.
**Defect class:** silently-wrong output, no error. **Status:** FIXED + gated.

Prior art: [`PLAN_CRUFT_PURGE_2026-09-06.md`](../docs/PLAN_CRUFT_PURGE_2026-09-06.md)
(the contract), [`dense_bake_contract_2026-09-06.md`](dense_bake_contract_2026-09-06.md)
(the tool), `cb2f412d` (the flip that made four shipped bakes dense),
`e1324192` (CI hygiene lane 2, which found the WASM symptom).

---

## 1. What was wrong

`cb2f412d` made the shipped `A`, `B`, `BHdr` and `D` bakes **dense**: each
declares the feature ids it reads (`zentrain.feature_ids`) and its layer-0
width is the size of that read set, not 372. Serving one requires GATHERING
those ids out of the walk's identity-laid-out vector — `metric.rs`'s
`declared_layout` branch.

That branch, and the whole `feature_layout` module behind it, was
`#[cfg(feature = "feature-regime-v2")]`. `feature-regime-v2` is default-ON, so
a plain `cargo add zensim` was fine. **Any consumer that sets
`default-features = false` was not.** The width disagreement then fell through
to `prep_bake_input_f32`'s `n_inputs < features.len()` PREFIX branch: `B` reads
ids `f3..f369` and got served positions `0..94`. Plausible numbers, wrong
features, no error — the exact outcome the dense contract exists to forbid
(contract line 5: *"a mismatch is a loud, named refusal. Never a structural
zero, never a positional prefix, never a silent `Drop`."*).

The instrument that should have caught it — `every_shipped_profile_is_servable`
— lived in `feature_plan`, which is gated on the **same feature**. A census
that cannot run in the broken configuration is not a census. And it would not
have caught it anyway: it asserted `emitted >= declared_width`, a width, never
a value.

## 2. MEASURED — the mis-score

Instrument: `zensim/examples/serving_matrix.rs`, 6 profiles × 4 geometries ×
3 distortions, `to_bits()` printed so a diff is exact. Reference = the default
feature set; arm = **every default feature except `feature-regime-v2`** (so the
SIMD tier and rayon are held constant and the only variable is the gate).

| profile | cells | min \|Δ\| | max \|Δ\| | worst cell | default | without the gather |
|---|--:|--:|--:|---|--:|--:|
| **A** | 12 | 2.4921 | 7.6714 | 256×256 / checker_lsb | 94.114457 | **86.443061** |
| **B** *(the `codec_target()` default)* | 12 | 2.2580 | 34.6773 | 48×40 / blur3 | 48.171502 | **13.494210** |
| **BHdr** | 12 | 3.3364 | 13.5977 | 127×93 / checker_lsb | 95.117667 | **81.519927** |
| **D** | 12 | 4.5262 | **261.5804** | 576×96 / blur3 | 48.431759 | **−213.148613** |
| C / CHdr | 12 | — | — | — | 98.75 / 96.92 | `ModelForwardFailed` on **all 12** |

**Every one of the 48 (A, B, BHdr, D) cells was wrong.** `D` inverts sign on a
blur. `B` is what `ZensimProfile::codec_target()` returns, so this was the
default profile in a `default-features = false` build.

The WASM symptom that surfaced it (`e1324192`) was the mild end: profile `A` on
a single-LSB distortion, 86.30 against 93.15+.

C and CHdr **refused** on every cell — a loud, named error, so not this defect
class, but also not servable: they declare a 944-wide caller vector and only
the v2 walk emits 944.

### 2b. The measurement's own noise floor

Same instrument, default vs `--no-default-features --features
feature-regime-v2` — i.e. `avx512` + `threads` on against both off, the widest
SIMD-tier/threading swing a consumer can produce, with the gather live in both:
**max \|Δ\| = 1.048e−5** over 24 cells. That is the floor the pinned-score
gate's tolerance is derived from (§4).

### 2b-bis. Who was actually exposed

Not a hypothetical consumer. **Two in-workspace crates build `zensim` with
`default-features = false` and do not re-add `feature-regime-v2`:**

* **`zensim-regress` 0.4.0 — PUBLISHED** (`features = ["classification",
  "custom-profiles"]`). Its own internals use a `ZensimProfile::Custom`
  linear profile with no MLP bake, so its *own* checksums were never affected;
  but its documented usage is `Zensim::new(ZensimProfile::codec_target())`
  (`testing.rs:13`, `:846`), which is `B`, which is dense. A binary depending
  only on `zensim-regress` got zensim without the gather.
* **`zensim-experimental`** (unpublished; `["imgref", "threads",
  "custom-profiles"]`).

Cargo feature unification is what limits the blast radius: a binary that ALSO
depends on `zensim` with default features turns `feature-regime-v2` back on for
the whole graph, which is why the workspace's own test suites stayed green.
That is a coincidence of the dependency graph, not a property anyone chose.

### 2d. AMENDMENT (2026-09-06, same day) — the gate's own tolerance was derived on ONE architecture, and i686 CI caught it

The first push of this fix (`bac6cc81`) turned **one** CI job red:
`Test (i686-unknown-linux-gnu)`, `every_shipped_profile_scores_its_pinned_value`,
369 passed / 1 failed —

    PreviewV0_1/single-LSB: 98.378872789271 vs pinned 98.394762958421 (delta -0.015890)

Reproduced locally BIT-FOR-BIT (same value to the last digit), so the diagnosis
below is measurement, not inference.

**How to reproduce i686 on this box, since it is not obvious.** `cross` needs
Docker, which is not available here, and `~/.cargo/config.toml` sets
`runner = "qemu-i386-static"` for the target, which is not installed — so
`cargo test --target i686-unknown-linux-gnu` fails with a misleading `No such
file or directory`. Neither is needed: an i686 binary runs NATIVELY on this
x86-64 box (`/lib/ld-linux.so.2` is present), and archmage's x86-64 tokens are
`target_arch = "x86_64"`-gated, so a 32-bit build takes the scalar tier no
matter what the CPU supports. What IS needed is 32-bit link support
(`sudo apt-get install gcc-multilib libc6-dev-i386`), after which:

```sh
cargo test --target i686-unknown-linux-gnu -p zensim --lib --no-run
./target/i686-unknown-linux-gnu/debug/deps/zensim-<hash>          # run it directly
```

Turnaround is seconds instead of a 9-minute CI job. Same for wasm:
`RUSTFLAGS=-Ctarget-feature=+simd128 cargo build --target wasm32-wasip1 …` then
`wasmtime run --dir=. <…>.wasm`.

**It is not a mis-serve, and not this fix's doing.** `PreviewV0_1` carries **no
MLP bake at all** — `scoring_bake_bytes()` yields nothing for it — so no dense
layout, no gather, nothing this lane touched is on its path. All four DENSE
profiles (`A`, `B`, `BHdr`, `D`) passed on i686.

**What it is: my TOL was derived on a population one architecture wide.** The
original derivation used x86-64 builds only (AVX-512+threads vs neither, max
1.048e-5) and set `TOL = 1e-2`. MEASURED over **160 cells** (8 profiles × 4
geometries × 5 distortion strengths) on the four arms CI actually runs:

| arms | n | max \|Δ\| | cells > 1e-2 |
|---|--:|--:|--:|
| x86-64 AVX-512 vs AVX2 | 160 | **2.1411e-5** | 0 |
| x86-64 AVX-512 vs **i686 scalar** | 160 | **2.8221e-2** | 9 |
| x86-64 AVX-512 vs **wasm32 simd128** | 160 | **2.8221e-2** | 9 |
| **i686 scalar vs wasm32 simd128** | 160 | **0.0000e+00 — bit-identical** | 0 |

**Two arithmetic classes, not four.** Every backend with a FUSED multiply-add
(AVX-512, AVX2, and — by CI evidence, all three aarch64 jobs green — NEON)
agrees to 2.1e-5. magetypes' **scalar** and **wasm128** backends implement
`mul_add` as an unfused `a*b+c` — the same split `e1324192` measured as 1 ULP on
the ring tests — and are **bit-identical to each other on 160/160 cells**. This
reproduces, on scores, the structure the v1 golden policy already recorded for
features: every non-AMD class produces *one shared alternative result set*.

**Why 1 ULP of form becomes 1.6e-2 of score.** On a near-identical pair the
dissimilarity features sit at the **f32 cancellation floor** — measured for
`PreviewV0_1`/single-LSB: 193 of 228 features differ, the worst by **7.4×
relative** at absolute values of ~1e-6..1e-5 (`f143`: 1.806e-6 vs 1.512e-5),
and the raw distance moves **1.42 % relative** (3.16505e-2 → 3.20991e-2). Then
`score = 100 − 18·d^0.7` has slope `−12.6·d^{−0.3}` = **−35.5** at that `d`, so
1.42 % of a small distance is 1.6e-2 of score. `mean_offset` agrees to ~1e-9
relative, so the divergence is entirely in the features.

**The kernel is NOT the fix, and that is a measured prior decision, not a
dodge.** `e1324192` already tried fusing the kernel side; it works for the ring
tests but *"measurably shifts sigma_sq/sigma12 and newly failed
cross_platform::pixel_format_equivalence's Srgb16Rgba case"*, and the root cause
is upstream in magetypes. Nothing in this lane changes that judgement.

**The corrected bar.** The gate must separate two MEASURED populations:
cross-class noise **2.8221e-2** (160 cells, §2d) and the smallest mis-serve it
exists to catch, **2.258** points — the minimum over the 24 A/B/BHdr/D cells of
§2, whose distortion families (`truncate_lsb`, `blur3`, `checker_lsb`) are the
same ones the gate's own two cells use. Scoped honestly: that minimum is
measured on the serving-matrix population, not on the two census cells
themselves, because reproducing the mis-serve on those would mean re-gating the
code the fix removed. `TOL = 0.25` is their **geometric midpoint**
(`sqrt(2.8221e-2 · 2.258) = 0.2524`) — **8.86× above the noise, 9.03× below the
defect**: the maximally-separated choice, not one picked to make a build pass.
The same bar replaces the `1e-2` in `zensim-wasm-tests`. MEASURED why that one
passed: **its cell is well-conditioned** — wasm32 reads `93.150736250847`, the
x86-64 pin to all 12 printed digits, because 256×256 puts the pooled features
far from the cancellation floor. The bar is not set by that cell; it is set by
the class the backend belongs to, which reaches 2.8221e-2 on others.

**The tight complement is unchanged.** `scripts/serving_matrix.sh` requires
**bit-exact** agreement across feature sets WITHIN one architecture, and it
passes. Loose across classes, exact within one.

**Reusable facts:** (a) a cross-architecture score bar for this metric cannot be
tighter than ~3e-2 while magetypes' scalar/wasm128 `mul_add` stays unfused;
(b) i686 and wasm32-simd128 are ONE class — a fix or a pin validated on either
covers both; (c) tolerance derivations must state the POPULATION they were
measured on, because a same-architecture population silently understates this
one by three orders of magnitude.

### 2c. A second, independent "cannot be served": the packaged crate did not compile

`zensim/Cargo.toml`'s `include` is an ALLOWLIST. `cb2f412d` pointed A/B/BHdr/D
at four `*_byid_2026-09-06.bin` files and `c_sdr_purity944` / `c_hdr_l1t1944`
replaced `c_sdr_mlp944_corrmix`; the allowlist moved with neither. MEASURED
with `cargo package --list`: of seven weight files packaged, **exactly one**
(`v47_strict_qat_native_2026-05-27.bin`, and only via a doctest) was referenced
by the code, and **six** `include_bytes!` targets were absent. A published
0.3.0 would have failed to build for every consumer. Nothing in a workspace
checkout can notice, because the files are right there on disk.

## 3. The fix

**Option (a) from the brief — make id-routing available in every build — and
it is free**, because the dense-serving code never needed the v2 machinery.
`feature_layout` depends only on `feature_set_id`, `feature_defs` and
`mlp::Model`, none of which is v2-gated, and every shipped dense bake's
`walk_width()` (one past its highest declared id) is ≤ 372, which the legacy v1
buffered walk already emits. So the gather works against the walk that
`--no-default-features` already had.

1. **`feature_layout` is ungated**, as are `declared_feature_ids` and
   `ZENTRAIN_FEATURE_IDS_KEY`. The two gather sites in `metric.rs` — the
   single-forward path and the batched FD-gradient entry — lose their `#[cfg]`
   forks entirely. **There is no longer a legacy path to refuse from**: one
   code path, no fork, which is stronger than a refusal on a second path.
2. **`candidate-profiles = ["feature-regime-v2"]`.** `C` / `CHdr` cannot be
   served without the 944 walk, so the feature that ships them now requires the
   machinery that serves them. (`D` rides along; `D`'s reason to exist is the
   fold engine, behind the same feature — `profile_d_notax_2026-09-01.md`.)
   After this, **every profile a build can name, it can serve** — the census
   has zero refusals in every configuration.
3. **The servability census moved to `zensim/src/serving.rs`, always
   compiled.** The plan-shaped gates (revision agreement, id-space vs
   `from_block_profile`) stay in `feature_plan`; they import the ONE roster
   from `serving`, so there is still exactly one `#[cfg]`-dependent profile
   list.
4. **`Cargo.toml`'s `include` allowlist corrected** to the twelve files the
   code actually embeds, plus the pre-guard B ensemble and the manifest.

### Compile-time and size effect: +4.3 % of `.rlib` on a `--no-default-features` build, nothing on a default one

MEASURED as a same-parent A/B — a second `jj` workspace at `main@origin`
(`fc47b08e`) with its own `CARGO_TARGET_DIR`, so the numbers do not carry drift
from the concurrent score-path lane. Incremental rebuild of `zensim` alone
(`touch src/lib.rs`), best of 3 warm runs, and the `.rlib`:

| | before (`fc47b08e`) | after | delta |
|---|--:|--:|--:|
| `--no-default-features` `.rlib` | 3,138,964 B | 3,273,736 B | **+134,772 B (+4.29 %)** |
| `--no-default-features` rebuild | 2.60 s | 2.74 s | **+0.14 s (+5.4 %)** |
| default `.rlib` | 7,492,468 B | 7,496,008 B | **+3,540 B (+0.05 %)** |
| default rebuild | 5.85 s | 5.89 s | **+0.04 s (+0.7 %)** |

The default build barely moves, because `feature_layout` was already compiled
there — its +3,540 B is `serving.rs`'s roster plus the `metric.rs` hoist. The
`--no-default-features` cost is the real one and it is larger than "one 680-line
module" suggests: ungating `feature_layout` makes `feature_defs`'s registry
(`family_slots`, `full_width`, `REGISTERED_LAYOUT_WIDTHS`) reachable in a build
where nothing had referenced it, so its tables come with it.

**Read `.rlib` as an upper bound on shipped code, not as binary growth** — an
rlib carries crate metadata as well as object code. The number quoted is the one
that can be measured identically on both sides; a linked-binary comparison would
be smaller. **4.3 % of an rlib, and no measurable time, is the price of four
shipped profiles scoring correctly.** If it ever needs to come down, the lever
is narrowing what `feature_layout` imports from `feature_defs`, not re-gating
the gather.

## 4. Gates

| gate | where | what it proves |
|---|---|---|
| `serving::tests::every_shipped_profile_scores_its_pinned_value` | `zensim/src/serving.rs` | Every shipped profile's score on two fixed 64×64 pairs, pinned, **under every cargo feature permutation CI builds**. Tolerance **`0.25`** — the geometric midpoint of the two MEASURED populations it must separate: cross-arithmetic-class noise 2.8221e−2 (§2d) and the smallest real defect 2.258 (§2), so 8.86× above one and 9.03× below the other. Widen it only by re-deriving on a larger measured population — never to make a build pass. |
| `serving::tests::dense_bakes_resolve_to_a_dense_layout_and_the_gather_is_not_a_no_op` | same | The declaration is READ in this build, and — the negative control — the gathered vector genuinely DIFFERS from the positional prefix, so the gate can tell a served bake from a mis-served one. |
| `serving::tests::every_shipped_profile_is_servable` | same | Zero refusals, in every feature set (was v2-gated, i.e. blind exactly where it mattered). |
| `serving::tests::every_included_bake_is_packaged` | same | Every `include_bytes!("../weights/…")` is in `Cargo.toml`'s `include` allowlist. No filesystem — `include_str!` on both files. §2c is what it would have caught. |
| `serving::tests::the_serving_matrix_example_carries_the_same_roster` | same | The cross-build example's `#[cfg]`-dependent profile list and `shipped_profiles`'s reduce to the same `(gating feature, profile)` pairs. `#[cfg]`-independent by construction, because it compares SOURCE TEXT — a runtime comparison could not be. |
| `scripts/serving_matrix.sh` | repo | THE cross-build diff: 2 environments × 6 arms, every arm bit-identical to its environment's reference or a NAMED refusal. A third outcome is the failure. |
| `zensim-wasm-tests` | crate | Keeps `feature-regime-v2` (from `e1324192`) **and** now pins `ZensimProfile::A`'s score on the single-LSB distortion that first exposed this, at the same derived `0.25` (§2d — wasm32 is bit-identical to i686 scalar and carries the same 2.8e−2 floor). |

`scripts/serving_matrix.sh` result after the fix: **PASS**, all 8 arms, both
environments, **zero refusals**. Before the fix the v2-free arm differed on 48
of 72 rows.

Full gate pass on the landed tree: `cargo test --workspace --exclude
zensim-wasm-tests` **1,736 passed / 0 failed**, `cargo test -p
zensim-wasm-tests` (native) 16/16, `just clippy` (CI-exact, `-D warnings`)
clean, `cargo fmt --all --check` clean, `just api-doc-check` **current — zero
public-API delta**.

**Gotcha worth one line:** run the API check as `just api-doc-check`, never as
a bare `ZEN_API_DOC=check cargo test --manifest-path apidoc/Cargo.toml`. The
recipe sets `ZEN_API_DOC_TOOLCHAIN=nightly-2026-09-02`; without the pin a
newer nightly rewrites 11 `std::io::error::*` paths to `core::io::error::*` in
`zensim-regress.txt` and the check fails on a diff no code change produced
(the pin's own CI comment records the same measurement). Hit once in this lane.

### 4b. CI coverage, and why it was blind

The `Feature permutations` job runs `cargo test -p zensim $flags --lib` over 25
feature sets — so the pinned-score gate now runs in all of them. But **none of
those 25 cells named `deprecated-profiles` or `candidate-profiles`**, so `A`,
`C`, `CHdr` and `D` were only ever scored by the default-feature job, which was
on the correct side of the defect. Two cells added: `deprecated-profiles` (the
exact broken configuration — dense bakes, no `feature-regime-v2`) and
`deprecated-profiles,candidate-profiles`. `scripts/serving_matrix.sh` is a step
in the same job.

## 5. What this does NOT change

* **No shipped score moves.** The default build's numbers are untouched — this
  makes non-default builds agree with them, never the other way round.
* **No public API delta on a default build** (`cargo public-api`, §6 of the
  commit). `declared_feature_ids` and `ZENTRAIN_FEATURE_IDS_KEY` are
  `#[doc(hidden)]` and were already present under default features; they are
  now present under all of them, which is an ADDITION in the non-default
  configurations only.
* **C / CHdr are not densified.** That is the pending USER decision recorded in
  `dense_bake_contract_2026-09-06.md` §5 (0.866 / 0.311 zensim points), and
  this lane does not touch it.

## 6. The lesson, stated so it generalizes

**A gate that is `#[cfg]`-gated on the same feature as the code it protects is
not a gate.** Both halves of this defect have that shape: the gather was gated
on `feature-regime-v2` and so was the census that would have found it; the
packaging allowlist is only exercised by `cargo package`, which no workspace
build runs. The fix in both cases is to make the instrument reachable from the
configuration that can fail — a pinned score in the library's own unit tests
(runs everywhere), and an allowlist check that reads the manifest as a string
(needs no packaging step).
