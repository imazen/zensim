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
