# Common lane rules (every DVIFM/zensim lane, 2026-09-20)

Repo `/home/lilith/work/zen/zensim` (jj-colocated). Several lanes run CONCURRENTLY in other panes. Therefore:

- **Every heavy command goes through `~/tmp/devin/heavy`**, which flocks a shared lock and then calls run-heavy:
  `~/tmp/devin/heavy --mem 16G --jobs 8 -- <cmd> 2>&1 | tee ~/tmp/devin/<lane>_<step>.log`.
  It blocks until no other lane is running a heavy job. Never call run-heavy or cargo/python directly for anything
  heavy, never bypass the lock, and while you hold it do only the one job.
- **Do light work while the lock is busy** (reading, design, writing code/tests, analysis). Never idle-poll.
- **`jj` only, small commits, DO NOT PUSH.** Never rewrite a commit you did not create. Other lanes commit too: if
  `jj status` shows files you did not touch, leave them alone and keep your commits to your own paths.
- Use your own `CARGO_TARGET_DIR=$HOME/tmp/devin/<lane>-target` so lanes do not clobber each other's fingerprints.
- Scratch only in `~/tmp/devin/`; never `/tmp`, `/run`, `/dev/shm`. Never delete caches or generated data.
- `cargo fmt --all -- --check`, `just clippy`, `just lint-scripts` clean before your final commit (run them under
  the lock). Never relax or `#[ignore]` a test; the 5 pre-existing `zensim-validate/tests/bake_surface.rs`
  failures stay as they are.
- No public API additions (`pub(crate)` / `#[cfg(test)]`). No GitHub writes. No household names, MACs or LAN IPs in
  any output. No `pgrep -f`. Touch no other repository.
- Refresh `/home/lilith/work/zen/zensim/.workongoing` with `<UTC ts> devin-<lane> <activity>` every ≤2 min; other
  lanes overwrite it — that is expected, ignore it.
- `/mnt/v` must keep ≥80 GB free; check before each heavy stage; your lane's outputs go under the directory your
  brief names and stay inside its stated cap.
- **Read these first:** `docs/PLAN_DVIFM_VERDICT_2026-09-20.md`, `docs/FITTED_CONSTANT_GUARDS_2026-09-19.md`,
  `docs/PLAN_JOINT_CORE_SET_2026-09-19.md`, `benchmarks/joint_core_v1_2026-09-20.md`, `CLAUDE.md`,
  `/home/lilith/.claude/CLAUDE.md`, `~/work/zen/CLAUDE.md`.
- **Report once** at the end to your brief's DONE file (or _BLOCKED.md), with measured numbers, a MISSING list
  first, and what you did not do. Append a progress line to your lane log at every step. A measured negative is a
  good outcome; an unearned pass is the worst one.
- Holdouts: never read CID22-B (one sealed read is spent by the X1 lane only), AIC-3/4, AIC2026, KonJND val,
  KonFiG test, KADID terminal refs, or any secret holdout.
