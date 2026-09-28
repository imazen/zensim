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
| `zensim--cross-codec-v8` | `fa422097` investigate(v13-cvvdp-distill): cvvdp-as-teacher distillation (task #200) | **FALSIFIED** — fails on both linear + log-norm targets (emulating CVVDP output ≠ its CSF mechanism). | removed; preserved in graveyard tag |
| `zensim--recover` | `c66cdb41` ship PreviewV0_5TunerV2 + `0d0aa5e5` SPEED-B K-batched aux losses | **SUPERSEDED** — TunerV2 falsified (infra landed on main); SPEED-B (K-batched aux / minibatch-32) landed on main. | removed; preserved in graveyard tag |
| `zensim--v05-calibrate` | `72a1f174` affine-calibrate V0_5 to 0..100 + `fdd1b8f6` V0_5 identity short-circuit fix | **SUPERSEDED** — identity fix landed (apply_mlp_scoring + tests/v05_identity.rs); affine-calibration core shipped (v04_calibrate_mapping.rs). Extended V0_5 test harness preserved at `abandoned/principled-activity-2026-06-01/v05-calibration-tests`. | removed; preserved in graveyard tag |

## Also retired
- **`zensim--prune-profiles`** (branch `zensim-prune-review`, commit `17f15444`) — obsolete: main already pruned `ZensimProfile` to `{A, PreviewV0_2, PreviewV0_3, Custom}` and moved the experimental profiles/weights into the dedicated `zensim-experimental` crate (the branch only deleted them).
- **`zensim--principled-activity`** — the standalone op-store; all content preserved under `abandoned/principled-activity-2026-06-01/*` tags (see `ABANDONED_EXPERIMENTS_principled-activity_2026-06-01.md`).
- **`zensim--productionize-v6`** — a stale 2026-05-19 clone (HEAD `693e901`) whose HEAD is an ancestor of current `origin/main`; zero unique work.

Detailed negative-result findings (per-commit) live in
`ABANDONED_EXPERIMENTS_principled-activity_2026-06-01.md`.

## Mac checkout audit — 2026-09-26

The resumed Mac checkout had three uncommitted documentation edits in
`CHANGELOG.md`, `docs/CODEC_TARGET_METRIC.md`, and `zensim/src/profile.rs`.
Their complete patch ID is `4a4afd1bd05eb7a1725331f833bbde8bb915d20b`, exactly
matching `e833c8fef2191c6bdcfc55f8dfe8cfed5600920e`, which is an ancestor of
remote main. Reapplying them to September main would restore obsolete August
bake descriptions. No unique edits need landing. The snapshot before rebase
is `77521aeaed78aa9ebddf01b701f43c8706b2eae0`; the duplicate rebased change
`llnqrznsktlx` is retired after this verification.

The two other non-main heads are already published archive history:

- `2fe8bec9d2bac3d55f4b5ab8e24ceefd68373093`: the graveyard commit, reached
  by remote tag `abandoned/principled-activity-2026-06-01/graveyard`
  (annotated tag object `d3b8901b7ebdc5cf76609d93f5c108250c3e4221`).
- `e4ac1925ca06ee779828839e1c7e354d94e967fc`: the unused row-iterator XYB
  helpers, reached by remote tag `abandoned/principled-activity-2026-06-01/xyb-planar`
  (annotated tag object `99fcbe890f8fa166231e8347f847edf095be72c1`). Current
  `streaming.rs` reads `ImageSource::row_bytes` and handles strided images;
  those two proposed public wrappers still have no caller in main. They add
  no required behavior for the Margarine work and are not being imported.

GitHub's ref API matched both local annotated-tag objects; their peeled
commits match the hashes above. Excluding remote-main ancestry and tag
ancestry leaves only the duplicate documentation change and the active
Margarine x86 tail repair. There is one jj workspace (`default`) and no
checkout from this audit to remove. Archive tags are preserved unchanged.
