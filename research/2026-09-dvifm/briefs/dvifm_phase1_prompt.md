# Task: DVIFM block-visibility feature family in zensim — PHASE 1 (kernel, registration, gates). No training.

You are working in `/home/lilith/work/zen/zensim` (a jj-colocated git repo). A supervising session reviews your
work from artifacts, not from your summary. Work autonomously to the end of Phase 1; do not stop to ask.

## Your assignment

Read, in full, before writing code:
1. `/home/lilith/work/zen/zenpapers/docs/iqa-methods/dvifm-zensim-worker-brief.md` — the work order.
2. `/home/lilith/work/zen/zenpapers/docs/iqa-methods/dvifm-zensim-feature-design.md` — the spec.
3. `/home/lilith/work/zen/zenpapers/scripts/dvifm_block_visibility.py` — numpy reference; run it.
4. In zensim: `CLAUDE.md`, `SESSION-RESUME.md`, `docs/WAVE_PLAYBOOK.md`, `docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md`,
   `docs/FEATURE_SET_IDS.md` §7.4, and the csfw template commits `8eb8038e`, `4a08f77b`, `7bfd511d`, `3376baee`,
   `70e0fceb` (`git show <sha>`).
Also obey `/home/lilith/.claude/CLAUDE.md` and `/home/lilith/work/zen/CLAUDE.md` (machine-safety, jj, /tmp ban).

**Scope of this dispatch = the brief's "What to build" + "Gates before any screen" sections ONLY.**
Do NOT start the "Screen" section (no TRAIN extraction, no training, no block-stats cache fitting). That is a
separate later dispatch after the gates are reviewed. You MAY add the training-only block-stats side output
(behind the `training` feature) if it falls out naturally, but it is optional here.

## Order of work (land each as its own commit)

1. Verify the next free feature id really is f956 and how csfw (f944–f955) is registered; write down what you
   found in the gates doc. If the spec is wrong or ambiguous, write the problem to
   `~/tmp/devin/dvifm_BLOCKED.md` and stop — do not edit zenpapers.
2. Scalar reference kernel in Rust (f64-clean, obviously correct): normalisation constants min_Y/s_Y computed
   from zensim's own XYB transform over a dense sRGB grid (bake with a provenance comment + a test that
   recomputes them), binomial pyramid with reflect borders, Laplacian band and local-band option, 5×5 blocks,
   corner 3×3 extrema, φ_g, log-domain visibility, F1 + 5 F2 bins → 30 features.
3. Parity tests vs the Python reference (≤1e-6, pyramid planes included) on deterministic closed-form planes
   generated identically in both languages; commit only small expected-output fixtures (<30 KB total; if a
   fixture would exceed that, shrink the plane). Plus the exact identities and invariants listed in the brief.
4. Registration through the feature system (KernelId, ComputeToken in both the JSON registry and
   `ComputeToken::ALL`, FeatureDef entries, new set/era, `Plan::derive` serve + refuse), opt-in toggle OFF by
   default, modelled on `csfw_block`.
5. Streaming path: bit-identical to whole-plane at every strip size (test several strip sizes incl. ones that
   are not multiples of 5 or 16). One running accumulation order.
6. SIMD backend following the csfw generic-backend pattern (archmage/magetypes, `#[rite]` for nested helpers,
   `#[arcane]` only at the entry point; never the `wide` crate; `#![forbid(unsafe_code)]` stays). SIMD must be
   bit-identical to the scalar path or the difference must be bounded and tested — state which.
7. Byte-stability gate: every existing slot bit-identical with the toggle off AND on; shipped-bake golden
   gates unchanged.
8. Cost gate with the existing latency owners / zenbench (never criterion): toggle-off shows no measurable
   change; toggle-on added time and memory at 1024² and 2048², ≥30 paired interleaved rounds, pinned threads,
   dispersion reported, competing processes recorded. Memory via `/usr/bin/time -v` or heaptrack. Measured
   numbers only — never estimate or extrapolate across sizes.
9. `benchmarks/dvifm_block_gates_2026-09-19.md` modelled on `benchmarks/csf_tier1_gates_2026-07-28.md`, with the
   git commit and exact commands. State coverage as fractions and list what is MISSING first.
10. `just clippy`, `just lint-scripts`, feature-system tests, and `cargo test` for the touched crates all pass.

## Hard rules

- **Version control: `jj` only for writes** (`jj describe -m`, `jj new`). Never `git commit/reset/checkout/
  rebase/push`. Never `jj abandon`/`jj restore`/`jj op restore` on anything you did not create. The checkout has
  ~14 unpushed local commits from the supervising session under `@` — build on top of them, never rewrite them.
- **DO NOT PUSH.** No `jj git push`, no `scripts/safe_push.sh`. The supervisor reviews and pushes.
- **No public API change.** New items are `pub(crate)` or private. No new profile, default, shipped bake.
- **Never relax, `#[ignore]`, or weaken any existing test or threshold.** If an existing test fails, report it.
  Known pre-existing failures you must leave alone: 5 tests in `zensim-validate/tests/bake_surface.rs`, and one
  `prepare_steering` test mis-gated on `feature-regime-v2`.
- **Heavy commands** (cargo build/test/bench) run through
  `~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- <cmd> 2>&1 | tee ~/tmp/devin/<name>.log`, ONE at a time.
  Never pipe test/bench output through head/tail.
- **Never write to `/tmp`, `/run`, `/dev/shm`.** Scratch is `~/tmp/devin/`.
- **Coordination marker:** refresh `/home/lilith/work/zen/zensim/.workongoing` at least every 2 minutes of
  activity and before every long command, one line:
  `<UTC ISO timestamp> devin-dvifm <short activity>`. Leave it in place when you finish.
- **Do not touch any other repository** (zenpapers, zenmetrics, zenanalyze, …). Read-only outside zensim.
- No GitHub issues/PRs/comments. No files >30 KB and no images committed. No household names, MACs or LAN IPs
  in any file.
- Commit messages: `<type>: <short>` + what was done + what was found; only verified claims.
- Do not use `pgrep -f`/`pkill -f`.

## Reporting (the supervisor reads files, not chat)

- Append a line to `~/tmp/devin/dvifm_progress.log` at every step start/finish (`date -u` + step + status).
- When Phase 1 is complete OR you are genuinely blocked, write `~/tmp/devin/dvifm_PHASE1_DONE.md` containing:
  the list of jj change ids/commit shas, each gate's measured result (pass/fail with numbers), anything NOT
  done, and any spec ambiguity you resolved and how. Report once. An honest "gate X failed, here is the
  number" is a good outcome; a false "all pass" is the worst outcome.
