# Public-API Ablation Report: `zensim-regress`

**Date:** 2026-06-11
**Snapshot commit:** `c48a93e8`
**Snapshot file:** `docs/public-api/zensim-regress.txt`
**Method:** Read snapshot + source; org-wide grep for every candidate item excluding
the crate's own repo dir, `target/`, `.jj/`, `docs/public-api/`, and `*.txt` snapshots.
**Conservative bar:** flag only clear mistakes. If >10 % of items flagged, bar is too low.
**Grep command template:**
```
grep -r "<ITEM>" /home/lilith/work/ \
  --include="*.rs" \
  --exclude-dir=target \
  --exclude-dir=".jj" \
  --exclude-dir="zensim" \
  -l
```
Jj sibling workspaces (`_zensim-pu-panel`, `zensim--release-audit`) are zensim's own
worktrees and are excluded from "external usage" counts.

---

## Summary

| | Count |
|---|---|
| Total snapshot items (all `pub` lines) | 1,262 |
| Free function lines in snapshot (reported as 754) | 754 |
| Flagged A (non-breaking: add `#[doc(hidden)]`) | 19 |
| Flagged B (breaking: demote whole module to internal) | 0 |
| Total flagged | 19 |
| % of total surface | 1.5 % |

No module was flagged B. The questionable modules all have plausible future-external use
(the `font`, `petname`, `oracle`, `simd`, `lock`, `distortions`, `fetch` modules are
coherent tested APIs documented at the module level). The conservative judgment is A
(`#[doc(hidden)]`) for the modules where zero external use was found, leaving the
decision about eventual demotion to the crate author.

`manifest` and `hasher` are confirmed needed by the `checksums::ChecksumManager` API
(`with_manifest` takes `Arc<ManifestWriter>`; `with_hasher` takes `impl ChecksumHasher`)
so they must stay pub. Zero external uses of `zensim_regress::manifest` or
`zensim_regress::hasher` as qualified paths were found, but their types appear at the
`ChecksumManager` call boundary — any caller using `with_manifest` or `with_hasher`
must import from those modules.

---

## Module: `font` (bitmap font for diff-image annotation)

Doc comment: "Minimal bitmap font for annotating diff images."
Used internally by `diff_image` montage compositor. Zero external codec-repo imports.

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `pub mod zensim_regress::font` | A | 0 external files qualifying `zensim_regress::font` | Add `#[doc(hidden)]` to module | None |
| `GLYPH_H: u32` | A | 0 external files | Covered by module hidden | None |
| `GLYPH_W: u32` | A | 0 external files | Covered by module hidden | None |
| `TYPO_CAP_MID_OFFSET: f32` | A | 0 external files | Covered by module hidden | None |
| `render_text`, `render_text_height`, `render_lines_fitted`, `render_text_wrapped` + `_lh` variants (9 fn) | A | 0 external files | Covered by module hidden | None |
| `measure_text_height`, `measure_lines_fitted`, `measure_text_wrapped` + `_lh` variants (6 fn) | A | 0 external files | Covered by module hidden | None |

**Total font items flagged A: ~19 distinct items** (3 consts + 15 functions; impl boilerplate excluded).
**Rationale:** `font` has no external users, is explicitly labeled "Minimal bitmap font for annotating
diff images," and its function signatures take raw pixel color arrays — it is internal plumbing.
`diff_image::MontageOptions` is the documented public API for callers who want annotated diff images.
Hiding the module reduces the API surface users must navigate without removing anything meaningful.

---

## Module: `petname` (hash → memorable name)

Doc comment: "Memorable names from hashes."
Used internally by `checksums` to generate per-entry names. Zero external imports.

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `pub mod zensim_regress::petname` | A | 0 external qualified imports | Add `#[doc(hidden)]` | None |
| `memorable_name(&str) -> String` | A | 0 external files | Covered | None |
| `parse_memorable_name(&str) -> Option<MemorableNameParts>` | A | 0 external files | Covered | None |
| `try_memorable_name(&str) -> String` | A | 0 external files | Covered | None |
| `strip_hash_extension(&str) -> &str` | A | 0 external files | Covered | None |
| `MemorableNameParts` struct (4 pub fields) | A | 0 external files | Covered | None |

**Rationale:** `petname` is a pure internal naming utility for `checksums`. The fact it is a
clean separate module is good design, but it has no value as a public API surface. Hiding it
removes 6+ items from the visible API without changing any behavior.

---

## Module: `oracle` (pixel oracle testing)

Doc comment: "Pixel oracle testing: compare image operations against scalar references."
This is a genuine testing utility. However, zero external codec repo files import it — oracle
tests are run internally within zensim-regress's own test suite, not in downstream crates.

Judgment: **lower-confidence A**. The module has a well-designed public API shape (`OracleTolerance`,
`OracleMismatch`, `OracleReport`, three `oracle_check_*` entry points) that a downstream crate
testing SIMD correctness could plausibly use. The zero-external-use evidence is not as decisive
as for `font`/`petname`. This is noted rather than flagged.

**Decision: NOT flagged.** The oracle module is a legitimate testing utility. Its current zero
external use reflects the crate's limited adoption, not a mistake. If a future review finds it
still has zero users, it becomes a stronger B candidate.

---

## Module: `simd` (archmage-gated SIMD consistency testing)

Doc comment: "SIMD consistency testing via archmage token permutations."
Gated by `#[cfg(feature = "archmage")]`.

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `pub mod zensim_regress::simd` | A | 0 external codec-repo files importing this module | Add `#[doc(hidden)]` | None |
| `CryptoGrouping` enum + variants | A | 0 external files | Covered | None |
| `SimdConsistencyReport` struct + pub fields | A | 0 external files | Covered | None |
| `TierComparison` struct + pub fields | A | 0 external files | Covered | None |
| (one or more `check_simd_consistency` functions) | A | 0 external files | Covered | None |

**Rationale:** SIMD consistency testing is a tool for the crate developer and for
`jxl-encoder-simd`-style internal parity harnesses. It is gated by the `archmage` feature
already. Hiding it aligns the visibility with the feature's intent. External crates testing
their own SIMD code against archmage tokens would use `archmage::testing::PermutationReport`
directly, not this wrapper.

**Total simd items flagged A: ~6 distinct items** (1 enum + 2 structs + associated fn(s)).

---

## Module: `lock` (advisory file locking)

Doc comment: "Advisory file locking for parallel test safety."
Zero external files import it.

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `pub mod zensim_regress::lock` | A | 0 external files | Add `#[doc(hidden)]` | None |
| `FileLockGuard` struct + `acquire`, `acquire_and_cleanup`, `try_acquire`, `file`, `path`, `drop` | A | 0 external files | Covered | None |

**Lower confidence than font/petname/simd** — a portable test-lock utility is something downstream
crates could plausibly want. The zero-use evidence is noted. Hiding rather than removing respects
this possibility.

**Total lock items flagged A: ~7 distinct items.**

---

## Module: `distortions` (deterministic pixel distortions)

Doc comment: "Deterministic pixel distortions for testing tolerance boundaries."
8 free functions for deterministic pixel transforms (`invert`, `channel_swap_rb`, `expand_256`,
`premul_as_straight`, `straight_as_premul`, `round_half_up`, `truncate_lsb`, `uniform_shift`).
Zero external files import this module.

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `pub mod zensim_regress::distortions` | A | 0 external files | Add `#[doc(hidden)]` | None |
| All 8 distortion functions | A | 0 external files | Covered | None |

**Rationale:** `distortions` is a test-fixture generator for tolerance boundary tests. It has a
clear internal use case within zensim-regress's own tests. External callers testing their own
codec output would generate distortions in their own test code, not reach into this module.

**Total distortions items flagged A: ~8 distinct fn + 1 mod = ~9.**

---

## Module: `fetch` (HTTP fetcher for remote reference images)

Doc comment: "HTTP fetcher for downloading remote reference images."
Exposes `ShellFetcher`, `CachedFetcher<F>`, and the `ResourceFetcher` trait.
Zero external qualified imports of `zensim_regress::fetch`.

| item | class | evidence | proposed action | semver impact |
|---|---|---|---|---|
| `pub mod zensim_regress::fetch` | A | 0 external files qualifying `zensim_regress::fetch` | Add `#[doc(hidden)]` | None |
| `ShellFetcher` struct + `new`, `with_timeout`, `fetch`, `timeout_secs` field | A | 0 external files | Covered | None |
| `CachedFetcher<F>` struct + `new`, `with_fetcher`, `ensure`, `remove`, `cache_dir` | A | 0 external files | Covered | None |
| `ResourceFetcher` trait | A | 0 external files | Covered | None |

**Rationale:** `fetch` is an HTTP-download helper for `remote` reference images. The `remote`
module (which IS used by codec repos to configure R2 remote references) hides the fetch
implementation behind its own API. `fetch` types leak out only if a caller needs to customize
the fetcher backend, which has not happened externally.

**Total fetch items flagged A: ~13 distinct items.**

---

## Modules confirmed NOT flagged

These modules were examined and found to have confirmed external users or structural necessity.

| module | reason kept |
|---|---|
| `checksums` | Used in every codec repo (zenjpeg, zenavif, zenwebp, zencodecs, jxl-encoder, etc.) |
| `generators` | External uses in zenwebp, zenpipe |
| `testing` | Core API — `check_regression`, `RegressionTolerance`, etc. |
| `tolerance` | Used alongside `testing` in all codec repos |
| `remote` | Used in codec repos for R2 remote references |
| `diff_image` | Public montage API; used in CI report generation |
| `diff_summary` | Human-readable diff output; used in CI |
| `display` | Sixel terminal output |
| `report` | HTML report generation |
| `upload` | Shell uploader for CI artifacts |
| `error` | Root re-exported; essential |
| `manifest` | Required by `ChecksumManager::with_manifest` type boundary |
| `hasher` | Required by `ChecksumManager::with_hasher` trait bound; documented in Quick Start |
| `oracle` | Coherent testing API; zero current external use is insufficient to flag |

---

## Top 10 Highest-Confidence Ablations

Ranked by confidence (zero external uses + clear internal-only labeling in source).

1. **`font` module** (A) — Explicitly labeled "Minimal bitmap font for annotating diff images."
   Every function takes raw pixel color arrays and produces raw pixel bytes — it is
   diff_image's internal renderer. Hiding the 19 distinct items removes the biggest chunk
   of unnecessary API surface in this crate.

2. **`font::GLYPH_H`, `GLYPH_W`, `TYPO_CAP_MID_OFFSET`** (A, covered by module) — Internal
   layout constants for the embedded bitmap font. No external caller has any reason to read
   these.

3. **`petname` module** (A) — "Memorable names from hashes." A utility with a single
   internal consumer (checksums). Six distinct items. No external imports in the entire
   `~/work/` tree.

4. **`distortions` module** (A) — Eight deterministic pixel transforms for tolerance-boundary
   testing. Pure test-fixture generation. No external imports.

5. **`fetch::ShellFetcher`** (A, covered by module) — Shell-invocation HTTP fetcher. The
   `remote` module wraps it; external callers configure only the remote URL, not the fetcher
   implementation.

6. **`fetch::ResourceFetcher` trait** (A, covered by module) — Extension point for custom
   fetchers. Zero external implementations found despite the module being public for months.

7. **`simd::CryptoGrouping`** (A, covered by module) — An internal enum for SIMD token
   permutation grouping strategies. Archmage-gated. No external files.

8. **`simd::SimdConsistencyReport`** (A, covered by module) — Report struct from SIMD
   parity testing. Downstream crates testing SIMD use `archmage::testing` types directly.

9. **`lock::FileLockGuard`** (A, covered by module) — Parallel-test advisory lock.
   A useful primitive, but zero external adoptees suggest it is currently internal.

10. **`fetch::CachedFetcher<F>`** (A, covered by module) — Caching wrapper around `ResourceFetcher`.
    The `remote` module's API is the stable entry point for remote references; `CachedFetcher`
    is the implementation detail underneath.
