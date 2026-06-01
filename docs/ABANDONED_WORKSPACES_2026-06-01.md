# Abandoned experiment workspaces — zensim (2026-06-01)

The 2026-06-01 cleanup wrapped up the long-running per-experiment jj workspaces
that had accumulated under `~/work/zen/zensim--*`. Each held one research line
that has since concluded (FALSIFIED) or landed/superseded on `main`. **All their
non-empty content is preserved on origin under the
`abandoned/principled-activity-2026-06-01/graveyard` octopus tag** (the shared
op-store graveyard) — every commit below is resurrectable by hash:
`git checkout <hash>` / `git cherry-pick <hash>`.

| Workspace | Lead commit | Verdict | Disposition |
|---|---|---|---|
| `zensim--exp-chunkc-pergroup` | `3245b7d4` exp(chunkc-pergroup): per-group standardizer | **FALSIFIED** — collapses CID22 (cross-corpus magnitude IS the signal). Also `c17447f5` multi-codec control (FLEET BLOCKED on cuda_dlsym_stub gap). | removed; preserved in graveyard tag |
| `zensim--exp-percentile-pool` | `81b5a36d` exp(percentile-pool): P² in-place Block B swap | **FALSIFIED** — limited training corpus loses to compression ship. | removed; preserved in graveyard tag |
| `zensim--cross-codec-v8` | `fa422097` investigate(v13-cvvdp-distill): cvvdp-as-teacher distillation (task #200) | **FALSIFIED** — fails on both linear + log-norm targets (emulating CVVDP output ≠ its CSF mechanism). | preserved in graveyard tag |
| `zensim--recover` | `c66cdb41` ship PreviewV0_5TunerV2 + `0d0aa5e5` SPEED-B K-batched aux losses | **SUPERSEDED** — TunerV2 falsified (infra landed on main); SPEED-B (K-batched aux / minibatch-32) landed on main. | preserved in graveyard tag |
| `zensim--v05-calibrate` | `72a1f174` affine-calibrate V0_5 to 0..100 + `fdd1b8f6` V0_5 identity short-circuit fix | **SUPERSEDED** — identity fix landed (apply_mlp_scoring + tests/v05_identity.rs); affine-calibration core shipped (v04_calibrate_mapping.rs). Extended V0_5 test harness preserved at `abandoned/principled-activity-2026-06-01/v05-calibration-tests`. | preserved in graveyard tag |

## Also retired
- **`zensim--prune-profiles`** (branch `zensim-prune-review`, commit `17f15444`) — obsolete: main already pruned `ZensimProfile` to `{A, PreviewV0_2, PreviewV0_3, Custom}` and moved the experimental profiles/weights into the dedicated `zensim-experimental` crate (the branch only deleted them).
- **`zensim--principled-activity`** — the standalone op-store; all content preserved under `abandoned/principled-activity-2026-06-01/*` tags (see `ABANDONED_EXPERIMENTS_principled-activity_2026-06-01.md`).
- **`zensim--productionize-v6`** — a stale 2026-05-19 clone (HEAD `693e901`) whose HEAD is an ancestor of current `origin/main`; zero unique work.

Detailed negative-result findings (per-commit) live in
`ABANDONED_EXPERIMENTS_principled-activity_2026-06-01.md`.
