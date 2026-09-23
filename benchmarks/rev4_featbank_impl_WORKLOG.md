# Rev4 feature-bank implementation — worklog (2026-09-23)

## Codex takeover audit (2026-09-23 UTC; supersedes stale numbers below)

Lane `featbank-impl-fix` resumed Devin change `nwnuzsut` in this existing
workspace. `jj diff --from 1881409d --to @` showed only a moved-evidence
pointer, an env-gated C1/C3 diagnostic, and a tracked cargo-target symlink.
The symlink is untracked and ignored; the two targets and evidence already
reside under `/var/tmp/featbank-impl/`. No new workspace was created.

| Review correction | Handover state verified from source/evidence |
|---|---|
| 1 ST table | Untouched; old medians and fits contradicted `st1.zenbench`. Corrected below. |
| 2 bleed test | Untouched; textured/noisy fixture could fill the mask. |
| 3 ladder | Untouched; docstring promises emitted monotonicity, code uses hidden magnitude with 2% slack. |
| 4 encoded phase | Untouched; synthetic fixture only. |
| 5 corpus gate | Untouched; ignored tests returned on missing assets and filtered KADID by digit. |
| 6 C3 edges | Untouched; `TailEdges::build` used runtime `powf`. |
| 7 definition revision | Partial diagnostic logging added, no calibrated definitions or corpus numbers. |
| 8 API | Untouched; API snapshot mixed new variants with pre-existing `research::Dvifm*` rows. |
| 9 commit pin | Placeholder remains by coordinator ruling. |
| 10 footprint | Target/evidence moves done; one target symlink was tracked, worklog and manifest incomplete. |

The calibration was preregistered in
`benchmarks/rev4_featbank_impl_fix_prereg_2026-09-23.md` before reading the
diagnostic histograms (sha256
`ee2b9e2fbaabff338dae2e7b8696b8b74633ae339890b87627f652bfb6efd5c2`).
The old ST claims and verdicts later in this file are historical and invalid;
the following recomputation is authoritative.

### Correction 1 — ST raw recomputation

- UTC start/end: `2026-09-23T23:41:40Z` / `2026-09-23T23:41:40Z`; cwd:
  `/home/lilith/work/zen/zensim--rev4-featbank`; command:
  `python3 benchmarks/rev4_featbank_st_cost_recompute.py > /var/tmp/featbank-impl/st_recompute.txt`;
  exit `0`; output sha256 `43b7435ad2faa623ca96ee04f346bb1a97aa06f60b492b085f135a7fb7a50e83`.
  Input `st1.zenbench` sha256
  `8ee4e59b1c033a582aefa0bf14552818ef9ca437d777603120fc01ba0f5197ec`.
- Exact source lines: `MEDIAN 1024 fold944_full=63.02167 fold986_dvifm=101.87249
  gridblk=117.25398 ringbasis=134.00389 tailhist=184.55656
  arttype=105.53152 rev4_all=230.52894`; `FIT arttype
  alpha_ms=-0.601293 beta_ns_px=2.320346 r2=0.986581 fit_pct=2.906565
  raw_pct=5.805992`. The other fit lines are retained verbatim in the
  hashed output. C4 ST is **borderline: fitted +2.9% passes, raw +5.8% misses**.

Lane `rev4-featbank` (implementation), workspace `../zensim--rev4-featbank`,
parent `main@origin` `e6ce1565` ("docs: Rev4 feature bank plan (featbank design lane)").
Scratch: `/var/tmp/featbank-impl/rev4-gate/`. Design/qualification record:
`benchmarks/rev4_featbank_impl_2026-09-23.{md,json}`.

Scope per `FEATBANK_IMPL_brief.md`: implementation + qualification only. **No human
labels read, no models fit, no predictive claims.** All corpus reads are path columns
only (`ref_path`, `dst_path`); the KADID TSV's score column was dropped at ingest.

## Step 0 — workspace + manifest

```
jj git fetch && jj workspace add ../zensim--rev4-featbank -r main@origin
```

- `.workongoing` claimed; `.gitignore` already covers it.
- Binding docs read in full: `rev4/DEVIN_COMMON.md`, `rev4/FEATBANK_IMPL_brief.md`,
  `docs/DATA_SPLITS.md` (Ruling 2026-09-23: Rev4 feature-bank potential + LODO roles —
  permits imazen-26/CID22/KADID **TRAIN** paths for implementation gates; forbids
  label reads and potential claims), `docs/REV4_FEATURE_BANK_PLAN_2026-09-23.md`,
  `docs/FEATURE_V2_SPEC_2026-07-18.md`, `docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md`,
  `~/work/zen/CLAUDE.md`, `zensim/CLAUDE.md`.

## Step 1 — architecture mapping (no changes)

Mapped: `feature_defs.rs` (registry, `SignalDef`, `BlockDef`, `base_of`,
`REGISTERED_LAYOUT_WIDTHS`), `feature_set_id.rs` (`ComputeToken`, bitset,
string render/parse), `feature_plan.rs` (`ComputeSet`, `LayoutBlocks`,
`Plan::derive`, `populated_slots`), `feature_v2.rs` (streaming walk, materialized
walk, dense/gradient kernels, `V2NewFeatureToggles`, `FeatureRegime`), `research.rs`
(request → plan → extraction), extractor `extract_features_372col.rs` (`--full-*`
flags), `extract_paths_bench.rs` (zenbench arms, `rss_mode`).

## Step 2 — design note + registry scaffolding

- Wrote `benchmarks/rev4_featbank_impl_2026-09-23.md` with the pinned ID geometry:
  C1 gridblk 96 (f986–f1081), C2 ringbasis 72 (f1082–f1153), C3 tailhist 144
  (f1154–f1297), C4 arttype 24 (f1298–f1321); full width 1322.
- `feature_defs.rs`: `KernelId::Gridblk`, four `SignalDef` tables
  (`GRIDBLK_SIGNALS` 8×4×3, `RINGBASIS_SIGNALS` 6×4×3, `TAILHIST_SIGNALS` 12×4×3,
  `ARTTYPE_SIGNALS` 6×4), `BlockDef`s, `REGISTERED_LAYOUT_WIDTHS` += 1322,
  `REV4BANK_COMMIT` revision constant (**placeholder `"00000000"` — must be pinned
  to the landing commit; see FEATBANK_IMPL_DONE.md**).
- `feature_set_id.rs`: `ComputeToken::{Gridblk,Ringbasis,Tailhist,Arttype}`;
  `ComputeParts` widened u16→u32 (14+ tokens overflow u16; string wire format
  unchanged — verified `as_str`/`from_str` round-trip tests pass).
- `feature_plan.rs`: four compute flags + `LayoutBlocks` nested chain
  (each family requires the previous block's layout flag — positional widths);
  arttype `blur` inherits EWC's cross-scale gradient dependency in the fold mask.
- `V2NewFeatureToggles`: `rev4_gridblk/ringbasis/tailhist/arttype`, default OFF.

## Step 3 — kernels + accumulators (`feature_v2.rs`)

- Shared log-magnitude coordinate `u(v) = f64::to_bits(v) as f64` per plan §2 —
  C1 six signed bins (edges −4·C..+4·C in u-space), C2 six triangular hats
  (centres `REV4_HAT_CENTRES` const, `hat_memberships`-shaped eval — the initial
  bit-trick LUT indexed `u.to_bits() >> 52` out of domain; replaced with the
  spec-faithful partition eval), C3 32-bin true-log histogram edges
  (`TAIL_EDGES`).
- C1: `gridblk_strip_wide` `#[magetypes]`/`incant!` kernel — boundary excess
  `e = |Δ_dst| − |Δ_src|` normalized by local activity + `C_ACTIVITY`; stores
  signed `ẽ` into per-cell f32 planes (V/H) plus per-phase `Σ|ẽ|`, `Σẽ`
  accumulators; deterministic finalize rescan picks the argmax phase pair
  (V then H, lowest-index tie-break) and emits 6 bins + `on_mean` +
  `onoff_ratio` per (scale, channel). Periods Y `8>>s`, X/B `16>>s`; scale-3
  Y period 1 emits structural zeros but stays registered (brief's
  stated-definition option).
- C2: six triangular log-magnitude bins over the gradient kernel's existing
  `ring` term — pooled mass, nonnegative, `HigherIsWorse`.
- C3: per (scale, channel) four 32-bin histograms (SSIM-d, ART, DET, MSE)
  merged row-ordered across strips/threads; emits p95/p99 (lower edge of the
  first bin reaching `ceil(q·n)`) + exact max per map.
- C4: blur = `EDGE_WIDTH_CHANGE × HF_LOSS` on finished Y slots; noise = mean
  `hf_gain` where `activity ≤ C4_FLAT_ACT`; bleed = bounded chroma gradient
  excess outside a ±1-px dilated dst-luma-edge mask (luma-edge mask is a
  second `#[magetypes]` kernel, O(strip) memory).
- Emit: `finish_rev4_scale` shared by streaming + materialized engines;
  `FeatureRegime::Folded720Rev4`; extractor `--full-rev4` → 1322-wide request.

## Step 4 — parity bug found and fixed

`rev4_streaming_materialized_parity` failed on `onoff_ratio` only, cached-moments
arm only. Root cause: `box_blur_v_from_copy` uses a sliding-window running sum —
the same real row's activity carries different f32 rounding history depending on
which strip's window computed it. The cached-moments path gathered activity from
`act_full` (strip-0 history) while streaming computes it strip-locally. Fix: when
gridblk needs the wide activity window, run the full `run_blur_pass_strip` so
`scratch.activity` is strip-local bit-identical to streaming; dropped the
`act_full` gather. Verified: `abs_h` per-phase sums bit-identical across moments
modes after fix.

## Step 5 — unit tests (all in `feature_v2.rs::tests` unless noted)

All brief criteria — see design note §Results for the full list. Fixes during
bring-up:

- `hats()` LUT domain bug (above).
- Accumulator test: sequential f64 sums honestly drift ~1e-11 rel over 200k
  terms — bound set to 1e-10, not bitwise.
- Identity: C1/C2/C4 exact-0; C3 inherits the registered `d`-map fp residue
  (~1e-5–7e-5 ≪ 2e-3 bar) — asserted under the bar, justification in the note.
- Phase test: block-flatten removes steps off-lattice (inverts the profile);
  replaced with an alternating block-shift fixture that *adds* steps only at
  lattice boundaries — crop by 5 px → planted phase 3, argmax exact.
- Thread-invariance flake: `for_each_token_permutation` mutates process-wide
  dispatch — moved the tier matrix to isolated-process
  `tests/rev4_featbank_parity.rs` (repo rule: dispatch probes never run in the
  parallel lib-test process).

## Step 6 — corpus + SIMD gates

- Synthetic tier matrix (`rev4_featbank_parity.rs`): 8 geometries (sub-64,
  non-tight strides, past `H_TILE_WIDTH`) × 10 archmage token permutations →
  v4/v3/scalar. f0–f985 identical on/off within each tier; rev4 segments
  identical within-tier (0-ULP). Cross-tier drift measured: categorical
  bin-edge/argmax flips — era-2 contract, not asserted equal.
- Extractor corpus matrix (`extract_features_372col --full-rev4` vs
  `--full-986`, omni decode incl. zenavif AVIF; `--force-tier` flag added):
  CID22 64 + SafeSyn 64 + KADID TRAIN 16 pairs (paths only; KADID TSV filtered
  to TRAIN-source refs, score column dropped) × {serial, MT8} × {v4, v3,
  scalar} → **0 diffs / 851,904 cells**.
  Repair note (2026-09-23, post-record): the first forced-tier runs were
  written to mode-ambiguous filenames (`*-v3-*`, `*-scalar-*`), so the
  thread mode they carried could not be re-proven from disk. The four
  forced-tier combos were re-extracted to explicit thread-suffixed files
  (`*-v3t1`, `*-v3t8`, `*-scalart1`, `*-scalart8`) under
  `ZENSIM_FORMULA_REV=3 ZENSIM_ROOT_FORM=sqrt`; the on-disk matrix now
  independently proves all six modes — 18 comparisons, 851,904 cells,
  0 diffs (`python3 compare.py <986> <rev4>` per pair).
- In-tree `rev4_corpus_toggle_identity`: 250 pair-mode extractions, 0 diffs;
  19 SafeSyn AVIF pairs skipped (direct test decoder has no AVIF) — covered by
  the extractor matrix; limitation documented.
- CID22 canonical parquet: `--full-944` rows 0–499 vs
  `baseline-recovery/cid22-train944.parquet` → 472,000 cells, 0 bit diffs.
- `fold_engine_parity` (14), `feature_invariants` (10), `v1_golden_bytes` (5),
  `feature_set_id` (11): pass.

## Step 7 — research + registry

- `tests/research_engine_parity.rs` (6): bit-exact production/research at v1,
  944, 1322; `dropping_a_family_perturbs_only_its_own_slots` covers all four
  rev4 tokens (absolute IDs stay put, unpopulated = structural zero);
  thread-invariant.
- Registry/census lib tests (32): block bases, ID arithmetic round-trips at
  width 1322, servability census 0 refused, feature-set-id era tests.

## Step 8 — cost measurement

- ST: `taskset -c 8 RAYON_NUM_THREADS=1`, zenbench interleaved arms
  (`fold944_full`, `fold986_dvifm`, +each family, `fold1322_rev4`),
  sizes 256/1024/2048/4096², 20 rounds (8 at the two largest), raw at
  `/var/tmp/featbank-impl/evidence/st1.zenbench`
  (sha256 `8ee4e59b…`; pointer `benchmarks/rev4_cost_2026-09-23.pointer.md`). Contended box (loadavg ~10/32);
  40 noisy rounds + drift-correlated arms recorded, not hidden.
- Marginal `α+β·px` fits r²=1.0000: C1 13.1, C2 27.0, C3 75.3, C4 3.3,
  all-four 120.9 ns/px → **every family misses its registered budget**
  (C1 +23.4% vs ≤5%, C2 +49.5% vs ≤2%, C3 +130.3% vs ≤8%, C4 +5.9% vs ≤3%,
  all +205.5% vs ≤15% at 1024² vs `fold944_full`). Recorded plainly.
- RSS `/usr/bin/time -v` + heaptrack: rev4 adds ~32 B/px at 4096²
  (793.0 vs 266.6 MB; heaptrack 815.0 vs 275.9 MB) — C1's stored `ẽ` planes,
  O(image) vs the walk's O(strip) norm. Recorded as a structural regression.
- MT8: `RAYON_NUM_THREADS=8`, same interleaved protocol after a ~37 min
  lock queue; raw at
  `/var/tmp/featbank-impl/evidence/mt8.zenbench` (sha256
  `3e944e39…`), 49
  noisy rounds flagged. Marginal β drops ~2.6× vs ST (all-four 46.8
  ns/px) but every family still misses its budget (+29.8/+47.2/+112.8/
  +7.5 fitted, +4.3 raw median / +193.4 % at 1024²).

## Step 9 — CI gates

- `just clippy` clean (13 lints fixed: derivable `Default`, let-chains,
  `needless_range_loop`→zip/enumerate, `then_some`, `repeat_n`).
- `just lint-scripts` clean (626 scripts).
- `cargo fmt --all --check` clean (scoped fmt to workspace members).
- `cargo test -p zensim` release all-features: 566 lib + all integration
  tests pass; debug lib: 521 pass (debug_asserts hold).
- `just api-doc-check`: **base snapshot was stale at `e6ce156`** (pre-existing
  `research::Dvifm*` pub items missing from the committed snapshot — not this
  lane's code). Regenerated `docs/public-api/*`; supported-surface delta
  attributable to this lane is exactly the mandated vocabulary additions:
  `ComputeToken::{Gridblk,Ringbasis,Tailhist,Arttype}` +
  `FeatureRegime::Folded720Rev4`. Toggle fields land in the hidden surface.
  Check passes on the regenerated snapshot.

## Commands for the numbers above

Exact invocations, raw `.zenbench` rounds, RSS `.time` files and heaptrack
`.zst` recordings live in
`/var/tmp/featbank-impl/evidence/` (pointer
`benchmarks/rev4_cost_2026-09-23.pointer.md`); `.time`/rss.tsv stay in
`benchmarks/rev4-cost-2026-09-23/` and are cited
from `rev4_featbank_impl_2026-09-23.md` §Cost/§Peak memory.
