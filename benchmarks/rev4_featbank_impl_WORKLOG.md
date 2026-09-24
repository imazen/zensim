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

### Correction 10 — moved evidence and target audit

- The target directories were moved before takeover, without deletion:
  `/var/tmp/featbank-impl/target-zensim` is 10G (rounded `du -sh`),
  `target-zensim-bench` 1.5G, and moved evidence 63M. Workspace-local
  target symlinks point to these locations. The `zensim-bench/target`
  symlink was accidentally tracked in the handover change; it is now
  untracked and `/zensim-bench/target` is ignored in `.gitignore`.
- UTC start/end `2026-09-23T23:57:23Z` / `2026-09-23T23:57:23Z`;
  cwd `/var/tmp/featbank-impl/evidence/gate`; command
  `sha256sum -c SHA256SUMS`; exit `0`; 58 `: OK` lines in
  `/var/tmp/featbank-impl/footprint_sha_check.txt` (sha256
  `1af9185d37d39c51c3afc7ec948aa9b13face32f62ff8303e502836ae48b65ba`).
  The `SHA256SUMS` file itself is
  `b8c70ae45afbb148ea7db192073ee54a6f46fb25fbf0b5fb16bdc7842272a4ba`.
- All top-level evidence shas match
  `benchmarks/rev4_cost_2026-09-23.pointer.md`; no new `/mnt/v`
  outputs were written. The manifest records the move and corrected
  `CARGO_TARGET_DIR` under `/var/tmp/featbank-impl/`.
- UTC start/end `2026-09-23T23:39:40Z` / `2026-09-23T23:54:37Z`
  (queued for the shared lock, actual build 27.62 s); cwd this
  workspace; command `~/tmp/devin/heavy --mem 16G --jobs 8 -- env
  CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-zensim-bench cargo build
  --release --manifest-path zensim-bench/Cargo.toml --example
  extract_features_372col --features 'training zen-decode'`; exit `0`;
  `/var/tmp/featbank-impl/build_diag.log` sha256
  `668b5cdbdec277f66e066c7107c75cc69120e9f7cc62789654aaa5e27b9fc46c`.
  Exact result line: `Finished release profile [optimized] target(s) in 27.62s`.

### Correction 7 — preregistered TRAIN-only registry calibration

- First diagnostic attempt: UTC start/end `2026-09-23T23:55:08Z` /
  `2026-09-23T23:58:29Z`; cwd this workspace; command
  `~/tmp/devin/heavy --mem 16G --jobs 8 -- /var/tmp/featbank-impl/run_diag.sh`;
  exit `101`. The first input was a headerless path list, which the
  extractor refused: exact line `ref_path col` in preserved
  `calibration/cid22-64.diag.failed-1.log` (sha256
  `9a418b21ed7ff3c9ea8922c498068e61b809802c164ca738f9b4726ce6442aad`).
  The failed log and command ledger were moved to `*.failed-1*`, not
  deleted. The script was corrected to use the headered TRAIN TSVs.
- Successful diagnostic: UTC start/end `2026-09-23T23:59:56Z` /
  `2026-09-24T00:04:07Z`; cwd this workspace; same heavy-wrapper command
  with corrected script sha256
  `2c91f0c3e8d91500bb1773d35c7c5176ebbbf173ef793eb2e3b25f6785ad177e`;
  exit `0`; wrapper log sha256
  `daeef42b38c1ee2913952efb48575e613dba9350e391ee8dcf1a1c9e1b99b056`.
  The three exact per-extraction UTC intervals, exit codes and output/log
  shas are in `/var/tmp/featbank-impl/calibration/commands.tsv` (sha256
  `27988452a70ae8b3f78dd316d6ac147b1ff4c7564446f0a9fd140a972f922459`).
  The extractor binary sha256 was
  `84c68b088fc28415f2e21f836867c4410ead79418ad8fd06d1749fdfd84cee84`.
- Census: UTC start/end `2026-09-24T00:04:39Z` /
  `2026-09-24T00:04:39Z`; cwd this workspace; command
  `python3 benchmarks/rev4_featbank_registry_calibrate.py --diag
  /var/tmp/featbank-impl/calibration/{cid22-64,safesyn-64,kadid-train16}.diag.log
  > /var/tmp/featbank-impl/calibration_census.txt`; exit `0`;
  output sha256
  `9e719daa366ca448334c0d7de3815d00c0ed99ca7b174c05b57581684612deae`.
  Exact lines: `C3 safesyn rows=64 cells=768 old_p99_saturated=[8, 78,
  234, 14] candidate_2_saturation_upper_bound=[2, 0, 0, 0]`;
  `C3 cid22 rows=64 cells=768 old_p99_saturated=[0, 20, 94, 0]
  candidate_2_saturation_upper_bound=[0, 0, 0, 0]`;
  `C1 old_emitted_nonzero_cells=[1562, 1562, 1562, 1552, 1274, 0]
  total_cells=1728`; `C1 diag_lines=1562
  positive_on_grid_samples=62577517`. Quantile interval lines and all
  occupied bins are in the hashed output.
- A metadata-inspection command accidentally printed one **KADID TRAIN**
  `INPUTS.json` target field while locating its `role`. The value was not
  used for any selection, test or numeric claim; no held-out label was
  read. The corrected corpus test deserializes only corpus/reference/role
  fields, discarding target fields. This is a documented lane-rule
  deviation, not a hidden clean claim.

### Correction 8 — public API and pre-existing snapshot entries

- Exact supported additions are four variants of the non-exhaustive
  `ComputeToken` (`Gridblk`, `Ringbasis`, `Tailhist`, `Arttype`) and the
  non-exhaustive `FeatureRegime::Folded720Rev4`. They are listed in the
  design note and `CHANGELOG.md [Unreleased] → Added`. Four public but
  doc-hidden `V2NewFeatureToggles` fields are also listed there.
- UTC start/end `2026-09-24T00:08:41Z` / `2026-09-24T00:12:40Z`;
  cwd this workspace; command `~/tmp/devin/heavy --mem 16G --jobs 8 --
  env CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-apidoc just api-doc`;
  exit `0`; `/var/tmp/featbank-impl/api_refresh.log` sha256
  `efa7df6c472c75dce9e62490703eb55d29ba621927d2701b2c03227bf406db38`.
  Exact result line: `test public_api_surface_docs_are_current ... ok`;
  summary `1 passed; 0 failed`. Generated
  `docs/public-api/zensim.internal.txt` sha256
  `54fb62626dafb6346e775e2732d114ac14d99459b6f1e24813b8ece64e3fe417`.
  The pre-existing `research::Dvifm*` additions appear in their own
  commit named `docs: refresh public-api snapshot for pre-existing items`.

### Correction 2 — bleed negative-control repair

- First final-definition family run: UTC start/end
  `2026-09-24T00:07:14Z` / `2026-09-24T00:11:16Z`; cwd this workspace;
  command `~/tmp/devin/heavy --mem 16G --jobs 8 -- env
  CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-zensim
  ZENSIM_REV4_CORPUS_ROOT=/mnt/v/imazen-26-pristine/lilith cargo test
  -p zensim --release --all-features --lib rev4_ -- --include-ignored
  --nocapture`; exit `101`; log sha256
  `22087493b73b2b5675532a6ed72b6c0200fe847636086f6d784213c97bb4b7dc`.
  Exact summary: `10 passed; 1 failed`; failure was only
  `rev4_arttype_bleed_luma_vs_chroma`, with `X=0 B=0`. The original
  chroma pattern also made dst-Y edges, so it did not establish bleed.
- The replacement control uses opponent-colour stripes chosen for
  almost equal converted Y (<0.001 estimated from the source colour
  transform) outside a localized equal-RGB luma step. The test also
  measures the production dst-Y mask and asserts coverage <100%.
  Focused rerun: UTC start/end `2026-09-24T00:11:46Z` /
  `2026-09-24T00:15:20Z`; cwd this workspace; command
  `~/tmp/devin/heavy --mem 16G --jobs 8 -- env
  CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-zensim cargo test
  -p zensim --release --all-features --lib
  rev4_arttype_bleed_luma_vs_chroma -- --nocapture`; exit `0`;
  log sha256
  `ce15d4055245ae4f8c3957748cd1673526a1df61f62d7971879445c94c6ae041`.
  Exact line: `test feature_v2::tests::rev4_arttype_bleed_luma_vs_chroma ... ok`.
- Final extractor rebuild, including the revised definitions and manifest
  provenance fields: UTC start/end `2026-09-24T00:14:59Z` /
  `2026-09-24T00:16:55Z`; same cwd; command
  `~/tmp/devin/heavy --mem 16G --jobs 8 -- env
  CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-zensim-bench cargo build
  --release --manifest-path zensim-bench/Cargo.toml --example
  extract_features_372col --features 'training zen-decode'`; exit `0`;
  log sha256
  `f30e3756a79e5099ad23bf1815b16f8a5e14485404747841296321e12c9f9ed5`;
  binary sha256
  `18743b0391c0572c887c36c5aa8c9e787ead41532aecf659e94368b9f16c40e4`.
- Clean all-family final-definition run: UTC start/end
  `2026-09-24T00:15:46Z` / `2026-09-24T00:18:30Z`; cwd this
  workspace; command `~/tmp/devin/heavy --mem 16G --jobs 8 -- env
  CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-zensim
  ZENSIM_REV4_CORPUS_ROOT=/mnt/v/imazen-26-pristine/lilith cargo test
  -p zensim --release --all-features --lib rev4_ -- --include-ignored
  --nocapture`; exit `0`; log sha256
  `18a97b5fca0fb93613513da5c63a402a2cae5ccc7bd77ce790fbee3c5319cdfe`.
  Exact summary: `11 passed; 0 failed`; this includes every C1–C4
  family test and the mounted JPEG ladder.

### Correction 5 — explicit corpus opt-in, no silent skips

- UTC start/end `2026-09-24T00:16:28Z` /
  `2026-09-24T00:23:24Z` (includes shared-lock queue); cwd this
  workspace; command `~/tmp/devin/heavy --mem 16G --jobs 8 -- env
  CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-zensim just
  rev4-corpus-tests /mnt/v/imazen-26-pristine/lilith
  /mnt/v/output/zensim/dvifm-screen2c-2026-09-19/q1-human/INPUTS.json
  19`; exit `0`; `/var/tmp/featbank-impl/corpus_tests.log` sha256
  `4fe14ec609a1a95639dbdf49c209350a08e5ead7a2c95cbd8ffa39391d9a5919`.
  Exact lines: `rev4_synthetic_16_pair_identity ... ok`, mounted
  `rev4_gridblk_zenjpeg_ladder ... ok`, `cid22: 128 pairs checked ...
  0 skipped`, `safesyn: 90 pairs checked ... 19 skipped`, `kadid:
  32 pairs checked ... 0 skipped`, and `rev4 corpus toggle identity:
  250 pair-mode extractions, 19 pairs skipped`; final corpus test
  `1 passed; 0 failed`. KADID admission uses only `INPUTS.json` role
  metadata; the 19 skipped SafeSyn distortions are 11 AVIF and 8 JXL,
  with separate omni-extractor coverage. The original log reported only
  the aggregate; the re-review verified the extension census.

### Correction 7 — post-revision corpus gate and occupancy

- UTC start/end `2026-09-24T00:17:30Z` /
  `2026-09-24T00:25:34Z` (shared-lock queue included); cwd this
  workspace; command `~/tmp/devin/heavy --mem 16G --jobs 8 --
  python3 benchmarks/rev4_featbank_fix_identity_gate.py
  /var/tmp/featbank-impl/target-zensim-bench/release/examples/extract_features_372col`;
  exit `0`. Wrapper log `/var/tmp/featbank-impl/identity.log` sha256
  `b76649592594ad273eda99ea06aa136d2bf3e72d87444a270b16ca800b4f5021`;
  per-extraction command ledger `identity_revision/commands.jsonl` sha256
  `c4b32dbc131aade34d53d4ce6976d0ba19b6986bf98127e09ae22f62601c8afd`;
  summary `identity_revision/summary.json` sha256
  `eec8f2c18fd818d140b82acb7f6c139eae2d5ef66797cd9abf2612ee13644099`.
  Exact final line: `TOTAL comparisons=18 cells=851904 diffs=0`.
  Each comparison's line in the wrapper log says `diff=0`, with 64
  CID22, 64 SafeSyn and 16 KADID TRAIN rows for each of six modes.
  CSV manifests pin tier request, Rayon threads and binary sha256
  `18743b0391c0572c887c36c5aa8c9e787ead41532aecf659e94368b9f16c40e4`.
- UTC start/end `2026-09-24T00:25:59Z` /
  `2026-09-24T00:26:00Z`; same cwd; command
  `python3 benchmarks/rev4_featbank_registry_calibrate.py --post-dir
  /var/tmp/featbank-impl/identity_revision >
  /var/tmp/featbank-impl/post_census.txt`; exit `0`; output sha256
  `d7e279db864558142145ec7040b0d3f17411d58a959b2bfb4f0c18ca75d5e6c1`.
  Exact lines: `POST C1_all_nonzero=[1562, 1562, 1560, 1552, 1535, 1470]
  total_cells=1728`; revised C3 `C3_p99_saturated=[0, 0, 0, 0]`
  for both SafeSyn and CID22 (768 cells each). The script asserts
  ≤1% per map and asserts every C1 hat is nonzero on the 144-pair corpus.
- UTC start/end `2026-09-24T00:19:16Z` /
  `2026-09-24T00:27:56Z` (shared-lock queue included); same cwd;
  command `~/tmp/devin/heavy --mem 16G --jobs 8 -- env
  CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-zensim cargo test
  -p zensim --release --all-features --test rev4_featbank_parity
  --test research_engine_parity -- --nocapture`; exit `0`; log sha256
  `885a4279d32af237716ba3f8f81917790d1b9921f1009b388c19ce6ca7066a11`.
  Exact summaries: research parity `6 passed; 0 failed`; Rev4
  synthetic/tier parity `2 passed; 0 failed; 1 ignored` (the mounted
  corpus gate passed separately above).

### Final static checks

- `just lint-scripts`: UTC start/end `2026-09-24T00:25:35Z` /
  `2026-09-24T00:25:45Z`; cwd this workspace; exit `0`;
  `/var/tmp/featbank-impl/lint_final.log` sha256
  `72092547235a28d963bd77375081d464b488d5a38296652a099089ec8932f0da`;
  exact line `lint_scripts: 626 scripts checked, all runnable`.
- `cargo fmt --all -- --check`: UTC start/end
  `2026-09-24T00:27:15Z` / `2026-09-24T00:27:17Z`; same cwd; exit
  `0`; empty `/var/tmp/featbank-impl/fmt_check.log` sha256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- First final `just clippy`: UTC start/end `2026-09-24T00:27:23Z` /
  `2026-09-24T00:32:11Z` (shared-lock queue); command
  `~/tmp/devin/heavy --mem 16G --jobs 8 -- env
  CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-zensim just clippy`;
  exit `101`; log sha256
  `2e5167644881cdadcf670e6a71534f481afd6693972228c6cff63aa2ab967d71`.
  Two `clippy::neg_cmp_op_on_partial_ord` diagnostics were at
  `diag_bin01`/`diag_bin_abs` only. Replaced `!(v > 0.0)` with the
  equivalent explicit `v <= 0.0 || v.is_nan()`; these env-gated
  diagnostics were off in the corpus identity run.
- `just api-doc-check`: UTC start/end `2026-09-24T00:27:30Z` /
  `2026-09-24T00:32:16Z` (shared-lock queue); command
  `~/tmp/devin/heavy --mem 16G --jobs 8 -- env
  CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-apidoc just api-doc-check`;
  exit `0`; log sha256
  `ff181372095868387ed7886538c346d362acfd088c6f6b0dac181af8e4198acf`;
  exact lines `public_api_surface_docs_are_current ... ok` and
  `1 passed; 0 failed`.
- Final `cargo fmt --all -- --check`: UTC start/end
  `2026-09-24T00:33:06Z` / `2026-09-24T00:33:07Z`; same cwd; exit
  `0`; empty `/var/tmp/featbank-impl/fmt_check2.log` sha256
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- Final `just clippy`: UTC start/end `2026-09-24T00:33:13Z` /
  `2026-09-24T00:40:13Z` (shared-lock queue); same command as the
  first run; exit `0`; `/var/tmp/featbank-impl/clippy_final2.log`
  sha256 `e3d4a104064c0eb17cc978d1f3996d8c956348baa97ba63b5b88775cad2fc5c9`;
  exact result `Finished dev profile [optimized + debuginfo]
  target(s) in 5.17s` with no warnings.

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
  11 SafeSyn AVIF and 8 JXL pairs skipped (direct test decoder lacks both) — covered by
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

## HANDOFF — quota stop, 2026-09-24 UTC

The user stopped Codex lanes at 28.0% weekly usage. All ten review
corrections were completed before the stop in correction commit
`87db7d77319817a685ef4bf715180f7367d0df77`. The final revised
identity gate found 0/851,904 old-slot differences; all 11 family
tests, mounted corpus gate, Clippy, API snapshot, script lint and fmt
checks passed. The working copy was clean at the stop. The canonical
`FEATBANK_IMPL_DONE.md` has the CORRECTIONS table and the completion
marker is in the same rev4 directory. No command is in flight and no
work remains for this lane. Coordinator next step: short Opus review,
then pin `REV4BANK_COMMIT` at landing. No push was performed.

## Re-review corrections — 2026-09-24 UTC

User resumed the lane after quota stop. First 64 SafeSyn
TRAIN path rows (TSV sha256 `5a53976070a5e21b2bb7fe0d05f58b510b93e141dd9207dd3e2cd337fd15cd3b`)
give JPEG 34, WebP 11, AVIF 11, JXL 8; refs are PNG. No labels read.
Gate rejects other extensions or missing files and asserts 11/8.

The coordinator ruled CI has no TRAIN access. Plan §2.5 and the design
note now specify generated 16-pair CI and local `just rev4-corpus-tests`
for TRAIN. At `04:28:22Z`, the local `zensim-bench/target` symlink
was removed; its `/var/tmp/featbank-impl/target-zensim-bench` stays.
`cargo fmt --all -- --check` ran `04:30:41Z`–`04:30:42Z`, exit 0,
empty log sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Validation rerun (cwd this workspace; shared-lock queue included):

- `2026-09-24T04:30:13Z`–`04:38:18Z`, exit 0:
  `~/tmp/devin/heavy --mem 16G --jobs 8 -- env CARGO_TARGET_DIR=/var/tmp/featbank-impl/target-zensim just rev4-corpus-tests /mnt/v/imazen-26-pristine/lilith /mnt/v/output/zensim/dvifm-screen2c-2026-09-19/q1-human/INPUTS.json 19`;
  log `rereview_corpus.log` sha256
  `1d7a23d4efafcd6782dc2a5ea6a2eb6c1d2c479c55897861d2fc5490149919c3`.
  Exact counts: `safesyn: 90 pairs checked (serial + MT8), 19 skipped
  (AVIF=11, JXL=8)`; total `250 pair-mode extractions`; all three tests passed.
- `2026-09-24T04:30:53Z`–`04:38:21Z`, exit 0: same wrapper and
  `CARGO_TARGET_DIR`, `just clippy`; `rereview_clippy.log` sha256
  `98858c0246beaa1283df69d9c6e54c0388009fa2c38b7d1d3b5a245733bce4ec`;
  `Finished dev profile ... in 0.58s`.
