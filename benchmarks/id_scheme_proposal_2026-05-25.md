# Artifact ID scheme — experiments, families, bakes, corpuses, parquets

**Status**: PROPOSAL, awaiting user approval before phase 1 migration.
**Date**: 2026-05-25.
**Scope**: zensim + zentrain + zenanalyze artifact namespaces only;
public `ZensimProfile` enum stays as-is.

## 1. Executive summary

Adopt **ZAR (Zen Artifact Reference)** — a 5-noun scheme rooted in
`fam/`, `exp/`, `bake/`, `corp/`, and `pq/` prefixes — backed by a
single `zen-artifacts.json` registry committed to the zensim repo.

The scheme separates four orthogonal axes that current names tangle
together: **methodology** (family), **training run** (experiment),
**produced bytes** (bake), and **training data** (corpus + parquets).
A bake is content-addressed by `sha256[:12]` but presented with a
human legible `bake/<family>/<experiment>/<variant>` alias. The
registry maps aliases ↔ hashes ↔ filesystem paths.

Public `ZensimProfile` variants stay decoupled: a profile's
`ProfileParams::bake_id: &'static str` field names the bake by
ZAR alias, never by enum-variant name. Renaming `PreviewV0_3 →
PreviewV0_6` is a search-and-replace on enum sites; no internal
artifact moves required.

The single most important user decision: **content-addressed hash
or human-legible alias as the primary key?** Recommendation:
**alias-primary, hash-secondary** (see § 7 Open questions).

## 2. Audit findings

Numbers measured 2026-05-25 against current main:

- **Experiments**: 197 methodology docs in `benchmarks/*.md`
  spanning at least 5 naming conventions: `V_18`, `V_20a`,
  `EXP-CROSS-CODEC-METRIC`, `cycle_9b`, `exp_v22_persample`. ~80
  are falsification docs for dead branches; ~30 represent
  shipped or candidate bakes; the rest are sub-experiments,
  diagnostics, or supporting analyses.
- **Bakes**: 27 active `weights/*.bin` files + 12 archived +
  2 `v11_candidates/` = **41 distinct bakes**. Filename patterns:
  `v0_18_zerobiased_lz4_2026-05-13.bin`,
  `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`,
  `v_tuner_v11_2026-05-24.bin`,
  `v_balanced_v3_per_codec_2026-05-20.bin`,
  `picker_zenjpeg_2026-05-19.bin` — five name shapes, no
  invariant. Some files mix architecture / methodology /
  parameters / packaging in one string; others use only
  family-version.
- **Corpus dirs**: 39 dirs under `/mnt/v/zen/zensim-training/`
  spanning `2026-05-07/` → `2026-05-21/`. 2 are explicitly
  canonical (`canonical-2026-05-18/`, `canonical-2026-05-21/`).
  The remaining 37 are per-experiment intermediates of which
  most are frozen but not labeled as such.
- **Parquets**: ~80 distinct parquets across canonical +
  per-experiment dirs. Ad-hoc names: `safesyn_4target_372col.parquet`,
  `cvvdp_iwssim_LARGE_372col.parquet`,
  `multicodec_cvvdp_300col_v2.parquet`. Schemas drift across
  experiments (300-col vs 372-col vs 343-col extended).
- **Public enum**: 23 `ZensimProfile` variants, of which **17
  are `PreviewV0_5*`**. Variant names embed methodology
  (`Balanced`, `Compression`, `Tuner`), iteration (`V2`, `V3`,
  `V4`, `V5`), packaging (`Calibrated`), and routing
  (`Ensemble`, `CrossCodec`). 8 are alias helpers
  (`balanced()`, `compression_v2()`, `tuner_v5()`,
  `codec_target()`). The enum has churned 4× in 5 days.
- **Patterns coexisting**: at least **7 distinct ID grammars**
  in current artifact names — `V_NN`, `V0_NN`, `EXP-NAME-VN`,
  `cycle_N_subname`, `exp_subject_verdict`, `v_role_vN`,
  date-tagged `name_YYYY-MM-DD`.

## 3. Proposed scheme

### 3.1 Family — recurring methodological theme

Stable across many experiments. Examples: "per-sample-alpha
head", "cross-codec equivalence loss", "PCHIP spline output
calibration", "input-shaping feature transforms".

**Grammar**: `fam/<kebab-name>`

- `fam/per-sample-alpha` — V_24 architectural family
- `fam/cross-codec-eq` — equivalence-pair loss family
- `fam/output-calibration-spline` — PCHIP post-network family
- `fam/input-shaping` — V_20 per-feature transforms family
- `fam/multi-bake-routed` — ensemble classifier routing family

Rules:
- Families are append-only. Never renamed.
- A family is registered when ≥ 2 experiments share its
  methodology. Single-experiment ideas live under the
  experiment ID alone.
- Family doc lives at `benchmarks/families/<name>.md` and
  describes the methodology + invariants + the set of
  experiments that ship it.

### 3.2 Experiment — one training run (or seed sweep)

A specific code-recipe + hyperparameter point + corpus join
that produces one or more bakes. Date matters but does not
encode identity.

**Grammar**: `exp/<family>/<slug>[-<iter>]` where:
- `<family>` references a registered family (or `misc` if
  none yet),
- `<slug>` is the experiment's distinguishing methodology
  axis,
- `<iter>` (optional) is a monotone integer for retries
  within the same hypothesis (NOT a date; iteration is
  semantic).

Examples:
- `exp/per-sample-alpha/v24-mix-cv40-iw60` — the V_24 ship
  recipe, 300-feat input
- `exp/per-sample-alpha/v24-mix-372feat` — same recipe,
  372-feat input
- `exp/cross-codec-eq/v6-multi-band-anchor` — V6 anchor
  pressure recipe (was EXP-CROSS-CODEC-V6)
- `exp/cross-codec-eq/v9-extended-range-spline` — V9 was
  EXP-CROSS-CODEC-V9
- `exp/multi-bake-routed/v05-corpus-classifier` — EXP-ENSEMBLE-V05
- `exp/misc/cycle9b-pair-boost` — one-off cycle experiment

Rules:
- An experiment has ONE recipe. Hyperparameter sweeps within
  the same recipe are bake-level variants, not new experiments.
  A 5-seed CI is bakes `s1` ... `s5` under the same experiment.
- Re-runs with a deliberately changed hyperparameter become a
  new iter: `exp/per-sample-alpha/v24-mix-cv40-iw60-2`. The
  iter is not a date — it survives calendar drift.
- Falsification IS data. A falsified experiment keeps its ID
  and a `status: falsified` flag in the registry.
- The methodology doc lives at
  `benchmarks/<family>/<slug>[-<iter>]_methodology.md`.

### 3.3 Bake — produced artifact bytes

A specific binary the runtime can load. Multiple bakes per
experiment (seeds, repacks, calibrations).

**Grammar**: `bake/<family>/<slug>/<variant>` where `<variant>`
encodes seed + packaging dimensions without bleeding into the
methodology. Suffixes are orthogonal axes joined by `.`:

- `s1`...`sN` — seed (mandatory)
- `f32` / `i8` / `i8z` (zerobiased i8) — dtype packaging
- `lz4` / `zstd` — compression
- `cal-v9` / `cal-v10` — calibration spline iteration
- `pcc-v11e` — per-codec calibration overlay
- `tanh20` / `tanh30` — runtime head tanh scale

Examples:
- `bake/per-sample-alpha/v24-mix-cv40-iw60/s3.i8z.lz4` —
  the Balanced ship (currently
  `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`)
- `bake/per-sample-alpha/v24-mix-372feat/s4.i8z.lz4` —
  Compression ship (currently `v_compression_2026-05-18.bin`)
- `bake/per-sample-alpha/v24-mix-372feat/s4.f32.cal-v10` —
  CompressionV3 (same bytes + V10 spline overlay)
- `bake/cross-codec-eq/v6-multi-band-anchor/s1.f32.tanh15` —
  was `v_tuner_v6_2026-05-19.bin`
- `bake/cross-codec-eq/v9-extended-range-spline/s1.f32.cal-v9.tanh20` —
  was `v_tuner_v9_2026-05-20.bin`
- `bake/cross-codec-eq/v11-multi-group-konjnd/s2.f32.tanh30` —
  was `v_tuner_v11_2026-05-24.bin`

**Content addressing**: each bake also has a stable
`sha256[:12]` (e.g. `bake@b703c9cfc7e1`) generated from the
final byte payload. The registry maps alias ↔ hash bijectively.
Tools accept either; canonical docs use the alias.

Rules:
- Variant suffixes are unordered conceptually but written in
  the canonical order: seed → dtype → compression → calibration
  → overlay. Registry validation enforces order.
- **Repack ≠ retrain**: an i8 → f32 repack with the same
  weights gets a new variant suffix on the same experiment,
  not a new experiment. The hash differs; the alias prefix
  matches.
- **Spline calibration overlay = new variant**. The underlying
  network bake is unchanged; the metadata addition is a
  `cal-vN` suffix. This matches today's V2/V3 reality where
  one network is shipped under multiple `ZensimProfile`
  variants via different splines.

### 3.4 Corpus — a canonical training/validation data set

A versioned, frozen collection of parquets representing the
canonical inputs as-of a date. Distinct from per-experiment
intermediates.

**Grammar**: `corp/<role>/<YYYY-MM-DD>` where `<role>` ∈
{`canonical`, `picker`, `anchor`, `exp-<slug>`}.

Examples:
- `corp/canonical/2026-05-21` — current trainer + val
  inputs (was `canonical-2026-05-21/`)
- `corp/canonical/2026-05-18` — superseded canonical (was
  `canonical-2026-05-18/`)
- `corp/anchor/v9-extended-range` — anchor parquets for the
  v9 spline fit
- `corp/anchor/v10-11-band` — V10 11-band anchor parquet
- `corp/exp-v22-372feat-fillv1` — short-lived experiment
  intermediate (most `2026-05-18-*/` dirs land here)

Rules:
- `corp/canonical/...` is the only role used as default
  trainer input. Adding a new target column or rebuilding
  bumps the date.
- `corp/exp-*` dirs are eligible for cleanup 14 days after
  the last experiment that referenced them ships or
  falsifies. The registry marks them `status: stale` after
  that.
- Tower mirror path is recorded in the registry, not encoded
  in the corpus ID.

### 3.5 Parquet — a single file in a corpus

**Grammar**: `pq/<corpus>/<purpose>/<facet>` parsed from the
file's logical role, not its filename. Examples:

- `pq/canonical/2026-05-21/train/safesyn`
- `pq/canonical/2026-05-21/val/cid22`
- `pq/canonical/2026-05-21/scores/cvvdp_imazen_v0_0_1`
- `pq/anchor/v10-11-band/anchors_11band`

Filenames on disk MAY drift; the parquet ID is logical. The
registry maps each `pq/...` to the concrete file path. New
parquets register as they're produced; legacy filenames are
mapped lazily.

## 4. Mapping table (high-traffic artifacts)

Currently shipping bakes (active `weights/*.bin`):

| Current filename | Proposed ZAR |
|---|---|
| `v0_18_2026-05-13.bin` | `bake/concat-3way/v18-tv-mix/s1.i8.f32mix` |
| `v0_18_zerobiased_lz4_2026-05-13.bin` | `bake/concat-3way/v18-tv-mix/s1.i8z.lz4` |
| `v0_20_is_calibrated_2026-05-15.bin` | `bake/input-shaping/v20-greedy98/s1.f32.cal-v18` |
| `v0_22_iw_v2_calibrated_2026-05-16.bin` | `bake/per-sample-alpha/v22-iw-v2/s1.f32.cal-v18` |
| `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` | `bake/per-sample-alpha/v24-mix-cv40-iw60/s3.i8z.lz4` |
| `v05_ensemble_classifier_2026-05-18.bin` | `bake/multi-bake-routed/v05-classifier/s1.i8z.lz4` |
| `v_compression_2026-05-18.bin` | `bake/per-sample-alpha/v24-mix-372feat/s5.i8z.lz4` |
| `v_compression_persample_2026-05-18.bin` | `bake/per-sample-alpha/v24-mix-372feat/s4.i8z.lz4` |
| `v_tuner_2026-05-18.bin` | `bake/per-sample-alpha/v24-tuner/s2.f32` |
| `v_cross_codec_2026-05-19.bin` | `bake/cross-codec-eq/v1-w1.0/s1.f32` |
| `v_tuner_v6_2026-05-19.bin` | `bake/cross-codec-eq/v6-multi-band-anchor/s1.f32.tanh15` |
| `v_balanced_v2_2026-05-20.bin` | `bake/per-sample-alpha/v24-mix-cv40-iw60/s3.i8z.lz4.cal-v9` |
| `v_balanced_v3_2026-05-20.bin` | `bake/per-sample-alpha/v24-mix-cv40-iw60/s3.i8z.lz4.cal-v10` |
| `v_balanced_v3_per_codec_2026-05-20.bin` | `bake/per-sample-alpha/v24-mix-cv40-iw60/s3.i8z.lz4.cal-v10.pcc-v11e` |
| `v_compression_v2_2026-05-20.bin` | `bake/per-sample-alpha/v24-mix-372feat/s4.i8z.lz4.cal-v9` |
| `v_compression_v3_2026-05-20.bin` | `bake/per-sample-alpha/v24-mix-372feat/s4.i8z.lz4.cal-v10` |
| `v_compression_v3_per_codec_2026-05-20.bin` | `bake/per-sample-alpha/v24-mix-372feat/s4.i8z.lz4.cal-v10.pcc-v11e` |
| `v_tuner_v4_per_codec_2026-05-20.bin` | `bake/cross-codec-eq/v10-reallocated-anchor/s1.f32.tanh20.cal-v10.pcc-v11e` |
| `v_tuner_v9_2026-05-20.bin` | `bake/cross-codec-eq/v9-extended-range-spline/s1.f32.tanh20.cal-v9` |
| `v_tuner_v10_2026-05-20.bin` | `bake/cross-codec-eq/v10-reallocated-anchor/s1.f32.tanh20.cal-v10` |
| `v_tuner_v11_2026-05-24.bin` | `bake/cross-codec-eq/v11-multi-group-konjnd/s2.f32.tanh30` |
| `v_cross_codec_v2_2026-05-20.bin` | `bake/cross-codec-eq/v11a-spline-rescue/s1.f32.cal-v9` |
| `picker_zen{jpeg,webp,jxl,avif}_2026-05-19.bin` | `bake/picker/v1-per-codec/<codec>.i8` |

Tail (archive + low-traffic): registered lazily as referenced.

## 5. Public-API ↔ internal bake mapping

`ZensimProfile` variants stay **opaque externally**, **registry-
backed internally**. Pattern:

```rust
pub(crate) struct ProfileParams {
    /// ZAR bake alias, e.g. "bake/per-sample-alpha/v24-mix-cv40-iw60/s3.i8z.lz4"
    pub bake_id: &'static str,
    /// Resolved at compile time by build.rs to a sha256[:12].
    pub bake_hash: &'static str,
    /// ...existing fields...
}
```

A `zensim-build` script reads `zen-artifacts.json` and emits a
Rust source `bake_registry_generated.rs` containing `const BAKE_<HASH>: &[u8] = include_bytes!(...)` for every referenced bake.
`ProfileParams::params()` calls the resolver:
`resolve_bake(self.bake_hash) -> &'static [u8]`. The enum
variant name NEVER appears in the bake filename or registry.

**Aliases as the public API surface**: `ZensimProfile::balanced()`,
`compression()`, `codec_target()` keep working; their backing
bake is one registry lookup away. Renaming `PreviewV0_5Balanced
→ PreviewV0_6` is a refactor on `profile.rs` only; the bake
file, the registry entry, the experiment doc, the corpus all
remain untouched.

The registry is **the durable mapping** that survives across
crate version bumps, profile renames, and bake rotations. Live
location: `~/work/zen/zensim/zen-artifacts.json` — committed,
human-readable, validated by `zensim-artifact-check` (new
binary).

## 6. Migration plan

**Phase 1 — registry-only (no file moves)** (1 session):
1. Add `~/work/zen/zensim/zen-artifacts.json` with current 41
   bakes + 5 active corpora + ~30 most-cited experiments.
2. Add `zensim-validate/src/bin/zar.rs` CLI for resolve /
   lint / list operations.
3. Land `families/` directory with stubs for the 5 families.
4. CHANGELOG entry.

**Phase 2 — adopt in new artifacts** (ongoing):
5. New methodology docs use `exp/<family>/<slug>` IDs in
   their headers + filenames.
6. New bakes get the variant-suffix filename:
   `bake-<family>-<slug>-<variant>.bin`. Legacy names continue
   working via the registry alias.
7. `profile.rs` adds the `bake_id` field; new profiles use it
   exclusively. Old `mlp_bytes: Option<fn() -> &[u8]>` stays
   for back-compat until phase 4.

**Phase 3 — high-traffic alias migration** (one PR per family):
8. Rename the 5-10 most-cited bakes on disk to their ZAR
   filenames. Old paths stay as symlinks for 30 days.
9. Update `profile.rs` `include_bytes!` paths.
10. Update active scripts/v_next/*.sh to use new paths.

**Phase 4 — archive low-value tail**:
11. After 60 days, mark unreferenced `corp/exp-*` entries
   `status: archive-eligible`. User reviews list. Move to
   `/mnt/tower/.../archive/<id>/` not delete.
12. Remove back-compat aliases the registry hasn't seen
   resolve in 90 days (subject to user approval per
   referenced item).

## 7. Open questions (user decides before Phase 1)

1. **Alias-primary or hash-primary IDs in tools?** I recommend
   **alias-primary**: docs cite
   `bake/per-sample-alpha/v24-mix-cv40-iw60/s3.i8z.lz4`,
   tools accept either. Hashes are the disambiguator. Pure
   content addressing is more rigorous but opaque to
   archeology. The proposed scheme stores both — the question
   is which to lead with in human-facing surfaces (docs,
   CLAUDE.md, methodology headers, commit messages).

2. **Should `ZensimProfile::PreviewV0_X` variants be locked
   forever as added, or are they free to be renamed?** Even
   with the ZAR decoupling, the public enum is a semver
   surface. I recommend **lock variants once 0.x.0 ships**;
   adding new variants OK, renaming/removing requires a
   crate major bump. The 17-variant proliferation is itself a
   debt — but cleanup is a separate task from this scheme.

3. **Per-bake methodology doc lives where — `benchmarks/` or
   `benchmarks/<family>/`?** I recommend nested
   `benchmarks/<family>/<slug>_methodology.md`. The flat
   197-doc benchmarks/ tree is unsearchable; nested is one
   `ls` away from the family. Migration: phase-2 lazy. Old
   docs stay where they are with hardlink shims if needed.

## 8. Anti-patterns this scheme blocks

| Anti-pattern (current example) | What ZAR enforces |
|---|---|
| `v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin` — six axes in one string | family / slug / variant separated; pieces individually addressable |
| `v_balanced_v3_per_codec_2026-05-20.bin` — methodology + iteration + overlay + date | `bake/<fam>/<slug>/<variant>` puts each axis at its proper level |
| `PreviewV0_5TunerV4Calibrated` + `PreviewV0_5BalancedV3Calibrated` — public API leaking calibration overlay | overlay is a `cal-vN` bake suffix; one bake serves both `BalancedV3` and `BalancedV3Calibrated` via different metadata |
| `2026-05-18-konjnd-dense/` — date-only corpus dir | `corp/anchor/konjnd-dense-v1` survives same-day rebuilds |
| `EXP-CROSS-CODEC-V9` vs `v_tuner_v9` vs `V_24` — three names for one substrate-family-iter | `fam/cross-codec-eq` + `exp/cross-codec-eq/v9-extended-range-spline` + `bake/.../s1.f32.tanh20.cal-v9` — one name per axis |
| Date-only methodology docs (`v0_18_methodology_2026-05-13.md`) | doc filename embeds `exp/<family>/<slug>` first, date second |

## 9. Worked example — the 2026-05-24 v11 ship

**Family**: `fam/cross-codec-eq`
**Experiment**: `exp/cross-codec-eq/v11-multi-group-konjnd`
- methodology doc:
  `benchmarks/cross-codec-eq/v11-multi-group-konjnd_methodology.md`
  (currently `v_tuner_v11_methodology_2026-05-24.md`)
- recipe: 5 training groups + tanh_output_head_scale=30 +
  konjnd-aggregation head w=0.05
**Bake** (median seed of 5-seed CI):
`bake/cross-codec-eq/v11-multi-group-konjnd/s2.f32.tanh30`
- file: `weights/v_tuner_v11_2026-05-24.bin` (legacy filename
  preserved via registry alias for now)
- sha256[:12]: `<hash>` (computed at registry init)
**Corpus**:
`corp/canonical/2026-05-21` (training input)
- parquets used: `pq/canonical/2026-05-21/train/safesyn`,
  `pq/canonical/2026-05-21/train/cid22-train-subset`,
  `pq/canonical/2026-05-21/train/kadid`,
  `pq/canonical/2026-05-21/train/tid`,
  `pq/canonical/2026-05-21/train/konjnd-dense`
**Public API**: `ZensimProfile::PreviewV0_5TunerV5` resolves via
`ProfileParams::bake_id =
"bake/cross-codec-eq/v11-multi-group-konjnd/s2.f32.tanh30"`.
`ZensimProfile::codec_target()` returns `PreviewV0_5TunerV5`.

Rotating to a future Tuner v12 ship: register the new
experiment + bake, swap `codec_target()`'s arm. The v11 entry
stays for back-compat. No files move; no methodology docs
rename; no corpus rebuilds.
