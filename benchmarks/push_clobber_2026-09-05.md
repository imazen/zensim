# The 2026-09-04 sideways-push clobber — what was dropped, what was re-landed, and the guard

**Lane:** `claude-recover` (sibling jj workspace `~/work/zen/zensim--recover`; the
primary checkout was held by `claude-ownerfix2` throughout and was never touched
except for read-only `git`).
**Incident window:** 2026-09-04 16:58:53 – 17:08:29 MDT.
**Detected by:** the coordinator's per-commit `git merge-base --is-ancestor <sha>
origin/main` sweep over the day's lane reports — nothing else failed.
**Guard landed by this record:** `scripts/safe_push.sh`.

---

## 1. What happened

`jj bookmark set main -r <rev> && jj git push --bookmark main` is a
**non-fast-forward push whenever `<rev>` does not descend from `main@origin`**, and jj
performs it with no prompt and no warning. The bookmark moves SIDEWAYS: every
commit reachable only from the old tip becomes unreachable from the new one. The
git objects survive — nothing points at them. The pushing lane's own `jj log`
looks correct, so the loss is invisible from inside the lane that caused it.

It happened **twice**, from the `git reflog show origin/main` walk (each entry
ancestor-checked against its predecessor):

| # | time (MDT) | old tip → new tip | jj op (point / push) |
|---|---|---|---|
| 1 | 16:58:53 | `af64c8d4` → `b36a5439` | `db7c8ca86b69` / `82a64afba73d` |
| 2 | 17:08:29 | `c6ec0bcc` → `5a42251e` | `0edf97e28a91` / `67de5e5730be` |

Every other one of the 59 `origin/main` moves that day was a clean fast-forward.

## 2. What was dropped

Move 1 dropped one commit. Move 2 dropped eight — everything pushed between
16:59:11 and 17:07:01, spanning **six lanes**:

| sha | subject | dropped by |
|---|---|---|
| `af64c8d4` | results(subsets): coverage does NOT explain seed spread | move 1 |
| `0d602d16` | feat(train): `--dry-run-sampling` — emit the coverage sidecar without training | move 2 |
| `d61df415` | fix(train): fail loud on the four STRATEGY knobs that were silently discarded | move 2 |
| `dfdc010f` | docs(subsets): record the STRATEGY knob-family defect | move 2 |
| `b63ba555` | docs(subsets): name the exact cause of the 17 unreplayed arms | move 2 |
| **`d3a948ca`** | **feat(G-ADDR): grade every fair-board cell through the owner — 47 NOT-SHIPPABLE** | **move 2** |
| `028ced9c` | docs(subsets): quantify WHY the pooled null was under-dispersed | move 2 |
| `c6ec0bcc` | ledger(rounds 77-78): G-ADDR re-pin + FAIR board + contract-tier coverage | move 2 |
| `1bf4806f` | (a re-landing duplicate of `af64c8d4`, itself dropped by move 2) | move 2 |

## 3. What was actually LOST — measured, not assumed

A dropped commit object is not the same as lost content: the subsets lane
re-landed most of its own work under new shas before anyone noticed. The honest
test is per-added-line, not per-sha. For each dropped commit, every non-blank
added line was checked for exact presence in the current `origin/main` blob of
the file it was added to:

| sha | added lines | not present in main | verdict |
|---|---|---|---|
| `dfdc010f` | 33 | **0** | fully re-landed |
| `b63ba555` | 6 | **0** | fully re-landed |
| `028ced9c` | 17 | **0** | fully re-landed |
| `af64c8d4` / `1bf4806f` | 134 | 8 | re-landed and **improved** — the residual 8 lines are the "seed 4004 is in 18 arms" mechanism paragraph, which `028ced9c` replaced on main with a stronger measured form ("132 distinct subsets over 51 …"). No loss. |
| `d61df415` | 150 | 15 | re-landed; the guard code and its `("--ema-decay", …)` table are byte-for-byte present at `mlp_train/mod.rs:1944`. The 15 residual lines are re-worded comment prose. No loss. |
| `0d602d16` | 140 | 98 | **superseded, not lost** — see §3.1 |
| `c6ec0bcc` | 85 | **85** | **half genuinely lost** — see §3.2 |
| **`d3a948ca`** | **498** | **482** | **GENUINELY LOST** — re-landed, §4 |

### 3.1 `0d602d16` (`--dry-run-sampling`) is superseded by `subset_sim`, and re-landing it would be a duplicate

The 98 absent lines are the whole `zensim_mlp_train --dry-run-sampling` flag and
its implementation. They are absent from `main`. They were **not** re-landed —
but `5a42251e` landed `zensim-validate/src/bin/subset_sim.rs`, which takes
`--group NAME:PATH:TRAIN_W:VAL_W[:withinref]` ("Mirrors `zensim_mlp_train
--group`"), replays through the same `mlp_train::sampling::simulate`, and adds
multi-seed simulation, `--verify-digest` against a real run's
`ZENSIM_SAMPLE_DIGEST=1` output, and JSON descriptor rows. It answers the same
question strictly better, from an explicit recipe as well as from a bake's
embedded repro.

**Determination: not re-landed, deliberately.** Restoring `--dry-run-sampling`
would put a second implementation of "replay this recipe's sampler without
training" in the tree, against this repo's NO DUPLICATE IMPLEMENTATIONS rule.
`d3a948ca`'s own commit body already points readers at `subset_sim` for the
existing-bake case. The owner is `subset_sim`; that is the whole of it.

### 3.2 `c6ec0bcc` — one ledger block superseded, one genuinely lost

`c6ec0bcc` appended two blocks to `benchmarks/balance_campaign_2026-08-28.md`:

- A **"ROUND 77 — afternoon"** summary (12 lines) of the G-ADDR re-pin to
  `peer_ssim2`. Main since grew its own fuller ROUND 77 (from `8a0c3af3`) plus
  ROUND 78 and 79, and re-landing a competing ROUND 77 heading would collide.
  Every measured number in it (`p5 10.26`, `−770.6`, `−55.35`, the 19 B-spline
  candidates, D's 96.12) is in `benchmarks/dial_addressability_gate_2026-09-04.md`
  on main — checked, 8 hits for `10.26`, 5 for `770.6`. **Superseded; the
  measurements were never at risk.**
- A **G-ADDR board-coverage round** (72 lines) — the `claude-gaddrboard` lane's
  own ledger row for `d3a948ca`. **Nothing on main carries it.** Re-landed as
  **ROUND 80** (78 and 79 are taken), with its `**Commit:**` line corrected from
  `d3a948ca` to the re-landed sha and a pointer to this record.

## 4. `d3a948ca` — the real loss, and the re-land

482 of its 498 added lines were absent from `main`, including all of
`scripts/cut_gaddr_negtail_probe.py` (the file did not exist on `main` at all).
The boards under `/mnt/v/output/zensim/reports/` had been generated **with** this
code, so the next regen from `main` would have silently produced boards with **no
NOT-SHIPPABLE badges** and no `--graft-gaddr` — plausible-looking output, wrong
content. A lane (`claude-replicate`) was holding its regen pending this recovery.

**Re-land: `jj duplicate d3a948ca -d main@origin`.** It applied with **no conflict**,
and the result's diff against `origin/main` is **byte-for-byte the original
commit's diff** — 6 files, +555/−23, the same 22 deletion lines
(`comm -23` of the two deletion sets is empty). Three main commits had touched
the same files after the merge-base (`5a42251e` `gauntlet.py`, `abfe13de`
`dial_addressability_gate_…md`, `92caf565` `CLAUDE.md`) and none of their content
was displaced; the fair-gauntlet lane's `e9457b05`/`8a0c3af3` were already in
`d3a948ca`'s own ancestry, so the expected `gauntlet.py`/`promote_fulleval.py`
conflict never arose.

### Sha map

| original | re-landed as | verified |
|---|---|---|
| `d3a948caecb45bbdb608ed6ca9fdb79a2397676b` | `2e5cdc8b237cf60b6f06a9e6a163532508a1783d` | `git merge-base --is-ancestor 2e5cdc8b origin/main` → **YES** |

Message: the original verbatim + one trailer,
`(re-landed after the 2026-09-05 sideways-push clobber; original d3a948ca)`.

### Verification actually run

- `ast.parse` on `gauntlet.py`, `promote_fulleval.py`, `cut_gaddr_negtail_probe.py` — OK
- `promote_fulleval.py --help` lists `--graft-gaddr` / `--self-test-graft-gaddr`
- `promote_fulleval.py --self-test-graft-gaddr` → **PASS (0 failures)**
- Fair board regenerated to a **scratch** path (`~/tmp/recover_board/`; the live
  boards were not overwritten): 8,280,322 B, 99 rows — 99 not the commit's 97
  because the D-id100 lane promoted three fullevals since.
- **Badges render: 46 rows carry `gaddr.cfail`**, per-row **C3 39, C4 39, C2 23,
  C6 2, C5 1** — exactly the commit's stated fair-board figure (its 47th, the C1
  fail, is the off-board cell). 99/99 rows carry a `gaddr` block; the one
  contract-clean cell reads as stated.
- `scripts/v_next/gauntlet_gates.sh` → **rc=0**, GATE 1 + GATE 2 PASS.

**One side effect, recorded rather than hidden:** `bandwise_dashboard.py`
*writes* its `--fairness-tsv` argument, so the scratch regen refreshed the live
`/mnt/v/output/zensim/reports/fairness_tiers_2026-09-04.tsv`
(322,478 → 323,958 B, 433 → 436 rows). The three extra rows are the D-id100
lane's new promotions. A refresh, not damage.

## 5. The guard — `scripts/safe_push.sh`

**fetch → assert `main@origin` is an ANCESTOR of the target → bookmark set → push →
verify the target landed.** On a sideways target it **exits 3, names every commit
the push would drop, and does not touch the bookmark.** There is no `--force`.

The gate and the diagnostic are one expression: `::<remote> ~ ::<target>` is the
set of commits reachable from the remote tip but not from the target, and it is
empty **iff** the remote tip is an ancestor of the target. (`<remote> ~ ::<target>`
is an equally correct test but names only the tip and hides the rest of the loss —
the first draft did that and was fixed after the retrospective control below
listed one commit instead of five.)

Two jj traps it handles, both previously learned the expensive way: a successful
push makes `@` immutable and jj creates a fresh empty `@` on top, so `-r @`
one command later targets the wrong commit (hence `-r <explicit sha>` internally
and a post-push verify); and a jj workspace has no `.git`, so read-only `git`
verification must run against the primary checkout.

### `--self-test` (4 cases, throwaway repo + bare remote, no network)

| case | asserts | result |
|---|---|---|
| 1 | a fast-forward target is ACCEPTED and provably lands | PASS |
| 2 | **a sideways target is REFUSED (rc=3) and the remote is provably UNMOVED** | PASS |
| 3 | the refusal NAMES the commit that would be dropped | PASS |
| 4 | after `jj rebase -d main@origin`, the push lands **both** lanes | PASS |

Case 2 is the negative control the whole script exists for. Cases 1 and 4 exist
so a guard that refuses everything cannot pass.

### Retrospective control on the real repo

Replaying move 2's target (`--dry-run -r 5a42251e`) against the live
`main@origin` reproduces the refusal and names all five commits the push would
have dropped — the diagnostic the clobbering lane needed and did not get.

### Adoption

`CLAUDE.md` now opens with a `PUSH ONLY THROUGH scripts/safe_push.sh` section
(bare `jj bookmark set` + `jj git push` is banned in this repo), and
`just lint-scripts` covers the new script (571 scripts, all runnable).

## 6. What this does NOT fix

- A lane that runs `jj git push` directly still bypasses the guard. The gate is a
  convention plus a tool, not a server-side hook. A `pre-push` hook or a branch
  protection rule requiring fast-forward would be the structural fix; neither is
  configured here, and configuring the remote is the user's call.
- The nine dropped commit objects are still unreferenced. They are reachable by
  sha and are quoted throughout this file; nothing depends on them.
