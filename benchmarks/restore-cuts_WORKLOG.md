# restore-cuts worklog

Lane `restore-cuts` (Claude Sonnet, execution lane). Workspace `../zensim--restore-cuts`, bookmark
`quarantine/claude/restore-cuts`, base `main@origin` `26494c8a`. No push.
Brief: `~/tmp/zensim-paper/rev4/RESTORE_CUTS_brief.md`. Rules: `DEVIN_COMMON.md`, `CODEX_NOTE.md`
(crates on main; r7900x/r5900xt/i265/mac off-limits), `SONNET_TAKEOVER_brief.md` tool discipline.

## 2026-09-24 ~19:00Z start

- Read the brief, COST_CUTS_AUDIT, DEVIN_COMMON, CODEX_NOTE, the C8 landing (`5cdcf70a`, `7b8e8a4f`) as the
  registration template, the gmsd A_dev patch and the zgeom/block5 research code.
- Findings that shaped the design (each verified in source):
  1. The v1 fused kernel is duplicated across tiers and lane widths (~24 accumulation sites), but the
     kernel already stores `mu1`, `mu2`, `sd` for the inner rows (`store_mu`, `store_sd`), so the eight
     per-pixel maps are re-derivable per band (the zgeom `v1_channel_maps_and_sums` recipe) without
     touching a kernel. mapdev and z1max are therefore one side pass (`feature_v2/restore_cuts.rs`).
  2. main's C7 (`dvifm.rs`) already computes the smooth CURVE visibility; the two-state GATE is on
     unlanded `943781e8`. Item 3's missing arm is the gate form.
  3. `audit.rs` accepts only 372/944/986 columns, so `--audit-jsonl` failed at 1322 on main (found by
     the partb lane). Widened to the registered widths (1322, 1502, 1562, 1790, 1820, ...).
- Base for builds: sibling repos re-archived from fetched mains (CODEX_NOTE crates-on-main rule); commit
  ids recorded below when the build is made.

### 2026-09-25 01:35Z lock note
The shared heavy lock has been held for >30 min by the potential lane MLP batches (8 replicates x many arms). Compile-only `cargo check -p zensim --lib` runs at `run-heavy --mem 6G --jobs 2` (cgroup-capped, nice/ionice, one at a time) OUTSIDE the lock so the first compile is not delayed by hours; every test, benchmark and extraction run stays under the lock.

## 2026-09-25 01:55Z — implementation checkpoint

- Commits: prereg `4ad33bc5`; implementation `2d7b33dd`; pin + API snapshots `429ff917`.
- Gates run so far (all `run-heavy --jobs 2`, cgroup-capped, outside the shared lock; the lock has been
  starved by back-to-back potential-lane MLP batches for >1 h, including two waiters of the partb lanes):
  - `cargo test -p zensim --lib --features training`: `470 passed; 0 failed; 7 ignored`
    (`/var/tmp/restore-cuts/logs/test_lib_full1.log`).
  - `cargo test -p zensim --features training --test restore_cuts_parity`: `2 passed; 0 failed`
    (`test_parity8.log`): f0..f1501 bit-identical with the families on/off on 3 SIMD tiers x 4 sizes,
    layout-width independence (mapdev@1562, z1max@1790, gmsnative@1820 = the full-width values), research
    owner equality, identity behaviour, MT8 == serial, strided == tight, same-tier repeatability; cross-tier
    drift MEASURED (test_parity7.log: non-SSIM worst 5.7e-3 relative on 1e-4-magnitude slots, SSIM-derived
    worst 6.8e-2), reported not bounded, per the rev4/C8 tier policy.
  - `cargo clippy -p zensim --all-targets --features training -- -D warnings`: clean.
  - api snapshots regenerated and `ZEN_API_DOC=check` passes.
  - NumPy mirror (`just restore-cuts-mirror` steps, run by hand): gmsnative max rel error `6.5e-16`,
    mapdev `5.3e-5`, z1max mse `3.7e-8`, art/det `2.1e-4`, SSIM-derived `2.1e-3` (float64 mirror vs f32
    kernel on 1e-4-magnitude maps); wrong-definition controls miss by `1.1e4x` tolerance (z1max block-mean)
    and `14.8` relative (gmsnative x16 stabilisers) and are rejected.
- Design finding recorded: the planner computes every family block a layout reaches
  (`a_wide_layout_computes_every_block_it_reaches`), so "family alone" is not a plan-level notion. The nested
  chain lets each family be requested at the narrowest layout that reaches it; its values do not depend on
  later families (asserted).

HEARTBEAT 2026-09-25T02:05:02Z restore-cuts extraction

### 2026-09-25 02:05Z gates and extraction start
- Clean-snapshot builds: baseline `main@origin` 26494c8a (binary sha256 `169b49c6...`), candidate `ec5b1821` (`4ea8f333...`); `build_meta.json`: 15 repositories, 20 local packages, 220 registry, 0 git sources, 0 outside the snapshot. Defect found on main: zensim-bench manifest had two `gmsd` keys (commit 9e62cbc4 dedupes; notice in ~/tmp/devin/NOTICE_zensim_bench_duplicate_gmsd_key.md).
- Measure-first gate (2,000 SafeSyn pairs, 8 threads, `gate_sample2.log`): baseline `--full-gmsbank` WALL 52.24 s (38.3 pairs/s, max RSS 750 MB); candidate `--full-gmsbank` (families off) WALL 51.04 s, CSV **byte-identical** to the baseline (both sha256 `dc3f98b1...`); candidate with the four families + `prefix` WALL 67.05 s (29.8 pairs/s, max RSS 888 MB), f0..f1501 `3004000 cells, 0 mismatches` vs the baseline.
- Full extraction started under the shared lock: `scripts/restore_cuts/run_all.sh` (one heavy call per set).
