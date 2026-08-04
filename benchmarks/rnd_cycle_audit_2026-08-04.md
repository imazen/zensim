# R&D cycle-time audit — 2026-08-03/04 campaign

**Question asked:** why have R&D cycles gone glacial, and why does idle waiting
get re-charged instead of cached?

**Method:** wall-clock reconstructed from artifact mtimes (bakes, verdicts,
fullevals, lane logs) + `git log` commit times + lianli's own file mtimes;
token spend read from the Claude Code session transcripts
(`~/.claude/projects/-home-lilith-work-zen-zensim/**/*.jsonl`), which carry a
per-turn `usage` block with `cache_creation_input_tokens` /
`cache_read_input_tokens` / `ephemeral_5m_input_tokens`. Build cost measured by
running the builds, not estimated.

Scope: `2026-08-03T00:00Z → 2026-08-04T10:20Z` (34.3 h). All times UTC; the box
is `America/Denver` (MDT, UTC−6) and lianli renders mtimes in the same zone.

Reproduce: `~/tmp/cycle/{analyze_tokens.py,attribute.py,waves2.py}` (audit
scratch; the numbers below are the committed record).

---

## 1. Headline

| axis | measured |
|---|---|
| span audited | 34.3 h |
| registered waves / passes | 11 |
| whole-session idle (no agent turn anywhere, windows > 20 min) | **14.80 h** |
| ... of which compute was genuinely running | 8.03 h |
| ... of which **DEAD** (nothing running, or finished work unharvested) | **6.77 h (46 % of idle)** |
| total session token cost (Opus list price) | **$2,837.34** |
| ... wasted re-creating prompt cache that idle waiting expired | **$395.24 (13.9 %)** |
| total `cargo` wall-clock, all agents, all day | **23.0 min** |
| per-agent `target/` dirs on disk | 28 dirs, **113.6 GB** (95.1 GB zensim) |

Both of the user's symptoms are real and both are **orchestration-bound, not
compute-bound**. The single largest cause is the same for both: *an agent
asleep past the moment its work finished.* That costs wall-clock directly, and
costs tokens because the 5-minute prompt cache expires while it sleeps.

---

## 2. Causes ranked by hours lost

| # | cause | cost | bound by | fix |
|---|---|---|---|---|
| 1 | **Finished compute sat unharvested** — work complete, every agent asleep | **5.31 h** wall + most of the $395 | orchestration | `harvest_bakes.sh` + `await_artifacts.sh` + playbook |
| 2 | **Nothing queued at all** — no compute running *and* no agent awake | **1.46 h** wall | orchestration | playbook: one terminal condition, endgame in foreground |
| 3 | **Idle-lane imbalance** — local box idle while the remote lane held the critical path | **57.5 min** of a 28-core box (coherence wave) | orchestration | playbook: rebalance, don't statically split |
| 4 | **A post-bake hook that failed silently** — 9 auto-evals exited 2 on a missing corpus; 21 verdicts re-run by hand | **13.4 min** of recomputation, and it hid a 3 h 24 min lane | orchestration | `harvest_bakes.sh` fails loud |
| 5 | **Per-agent cold rebuilds** | **≤ 23.0 min** *total, all day* | compute | **no change** — see §5 |

Ranks 1–4 are the same defect wearing four hats: nothing in the loop is
*obligated to leave evidence when it stops.*

### The two events that dominate rank 1

**Wave 6 arm F — 125.6 min dead.** `~/tmp/wave6/process.log` records
`[proc] ALL SIX PROCESSED 2026-08-04T03:08:40Z`: every bake verdicted *and*
fullevaled, M3a computed, on disk. The results commit (`2817028f`) landed
`05:20:40Z`. Two hours and six minutes of finished, evaluated work sitting
untouched. Note what this is *not*: the harvest daemon worked perfectly. The
agent watching it did not wake.

**Coherence wave — 80.6 min dead.** lianli wrote its last bake
(`C_co2b_s1307`) at `19:11:30Z`. The first agent action after that was
`20:32:05Z`. Same shape: compute finished, nobody looking.

Between them: 206 min = 3.4 h, i.e. **65 % of rank 1** is two events.

### Idle-window attribution (the full ledger)

`DEAD` = time between the last campaign artifact write inside a no-agent window
and the next agent turn. Artifacts are campaign-scoped: `~/tmp` is shared with
unrelated projects, and an early cut of this table wrongly credited an
`fkdlocal_*.pth` training run as zensim compute.

| window (UTC) | total | compute | **DEAD** | last campaign artifact |
|---|---|---|---|---|
| 08-03 00:39:46 → 01:08:41 | 28.9 m | 0.0 | **28.9** | (nothing running) |
| 08-03 01:09:06 → 01:40:49 | 31.7 m | 0.0 | **31.7** | (nothing running) |
| 08-03 05:19:51 → 06:18:26 | 58.6 m | 58.2 | 0.4 | `C_em944_s99.bin.spec.json` ✅ |
| 08-03 06:19:39 → 06:59:02 | 39.4 m | 13.8 | **25.5** | `armC_s5.verdict.log` |
| 08-03 07:54:47 → 08:21:42 | 26.9 m | 0.0 | **26.9** | (nothing running) |
| 08-03 10:29:58 → 11:03:15 | 33.3 m | 21.9 | **11.4** | `armC_chain2.log` |
| 08-03 11:03:39 → 11:32:42 | 29.0 m | 8.3 | **20.7** | `armC_s512.log` |
| 08-03 12:07:03 → 14:26:57 | 139.9 m | 113.3 | **26.6** | `nt_local.log` |
| 08-03 14:51:44 → 18:41:46 | 230.0 m | 202.3 | **27.7** | `co_local.log` |
| 08-03 18:41:46 → 20:32:05 | 110.3 m | 29.7 | **80.6** | `C_co2b_s1307` (lianli) |
| 08-04 02:34:29 → 05:14:17 | 159.8 m | 34.2 | **125.6** | `wave6/process.log` |
| **totals** | **14.80 h** | **8.03 h** | **6.77 h** | |

The one well-timed window is `05:19:51 → 06:18:26`: **0.4 min** of dead time.
It is the existence proof that the target is reachable — the agent's wake
landed 24 s after the bake. Nothing structural distinguishes it from the
others; it got lucky.

---

## 3. Per-wave timeline

`reg` = pre-registration commit. `computeEnd` = last verdict / fulleval / bake /
lane-log write. `TAIL` = computeEnd → results commit (harvest + write-up).

| wave | reg | computeEnd | commit | elapsed | TAIL |
|---|---|---|---|---|---|
| W3 seed-scale (arm C) | 08:39Z | 11:57:31Z | 11:59Z | 199 m | 2 m |
| near-top arm (#50) | 12:04Z | 14:38:33Z | 14:39Z | 155 m | 1 m |
| coherence wave | 14:49Z | 21:20:08Z | 21:26Z | 397 m | 6 m |
| W4 seed-expansion | 21:35Z | 00:30:41Z | 00:32Z | 176 m | 2 m |
| W5 seed-ensemble | 00:47Z | 01:02:16Z | 01:08Z | 21 m | 6 m |
| W6 arm G (KonJND) | 01:19Z | 01:29:17Z | 01:33Z | 14 m | 4 m |
| **W6 arm F (distill)** | 01:19Z | 03:08:40Z | 05:20Z | 241 m | **132 m** |
| W7 arm H (KonJND leg) | 06:01Z | 07:43:10Z | 07:46Z | 105 m | 3 m |
| A8 balanced-selection | 06:32Z | 08:36:50Z | 08:51Z | 139 m | 15 m |
| packaging pass | 08:14Z | 08:36:50Z | 08:42Z | 28 m | 6 m |
| contrib appendix | 09:26Z | 09:54:34Z | 09:58Z | 32 m | 4 m |
| **total** | | | | **25.2 h** | **3.01 h** |

Read this table together with §2, not instead of it. Only W6-arm-F shows its
loss in the *tail*; the coherence wave's 80.6 min of dead time is **inside**
its 397 min elapsed (finish 19:11:30Z → harvest 20:34:40Z → verdicts
20:53Z–21:20Z → commit 21:26Z), so its tail reads a healthy 6 min. A tail-only
view understates the problem by ~2×.

---

## 4. Token: why waiting is re-charged instead of cached

The prompt cache entries in use are `ephemeral_5m` — a **5-minute TTL**
(`ephemeral_1h_input_tokens` is 0 on every turn today; the 1-hour tier is not
being used). Read is billed at 0.1× base, write at 1.25×. So a gap longer than
5 minutes converts the entire cached prefix from a 0.1× read into a 1.25× write
— a **12.5× multiplier**, and it applies to the whole prefix, which by late
session is 500–800 k tokens.

Turn-level totals, 3,779 assistant turns across 23 agents (deduped by
`requestId`):

| gap since that agent's previous turn | turns | cache_create |
|---|---|---|
| 0–300 s (cache still warm) | 3,618 | 14,820,640 |
| 300–600 s | 71 | 10,112,515 |
| 600–1800 s | 36 | 3,927,000 |
| 1800–3600 s | 14 | 1,817,702 |
| > 3600 s | 17 | 7,054,975 |

**138 turns — 3.7 % of all turns — carry 22,912,192 re-created tokens, 55.7 %
of every cache-write token spent today.**

| | tokens | Opus list |
|---|---|---|
| billed as cache **writes** (what happened) | 22,912,192 | **$429.60** |
| the same bytes as cache **reads** (had the loop stayed warm) | 22,912,192 | $34.37 |
| **waste** | | **$395.24 — 13.9 % of the $2,837.34 session** |

Two secondary observations:

- **The parking itself is cheap; the wake-up is expensive.** The 57 turns whose
  tool call *was* a wait (`Monitor`, `sleep`, `until`) cost 626 k cache-write +
  21.2 M cache-read — the polls read a warm cache. The money is spent on the
  first turn *after* a long silence. So "poll less often" is the wrong lever;
  "do not have long silences with an agent still attached" is the right one.
- **The worst single agent** ran 11.69 h wall with **7.92 h of >5 min idle** and
  spent 12,152,076 cache-write tokens — 29.5 % of the day's total cache
  creation, from one agent that mostly waited. Its costliest single turn:
  `08-03 14:28:08Z`, gap 141.2 min, **779,717 tokens re-created, 0 read**.

---

## 5. Build cache: hypothesis measured, and rejected

The prior suspicion was that per-agent `CARGO_TARGET_DIR`s force repeated cold
builds worth minutes × ~10 agents. **Measured, it is not a top cause.**

| measurement | result |
|---|---|
| cold build, `cargo build --release -p zensim-validate --bin bake_verdict`, empty target dir, `run-heavy --jobs 6` on a busy box | **72 s**, 221 crates, 677 MB |
| what the wave agents actually paid (`Finished ... in`, their own logs, `--jobs 24`) | 62–91 s |
| **total `cargo` wall-clock across every wave/tooling log, all day** | **91 builds, 1,380 s = 23.0 min** |
| second concurrent `cargo` on a *shared* target dir | **blocks 31.8 s** — `Blocking waiting for file lock on build directory` |

A shared `CARGO_TARGET_DIR` could save at most the cold-start portion of those
23 minutes — call it ~12 min — while imposing a measured 31.8 s serialization
on concurrent invocations. With 91 builds and 3–5 agents overlapping, the lock
plausibly costs more than the rebuild. There is no `sccache` installed and no
`RUSTC_WRAPPER`; installing one is a larger change than a 23-minute problem
justifies.

**Recommendation: keep per-agent target dirs. Do not add a shared one.** Use
the prebuilt-binary handoff that already exists — `sota944_verdict.sh`,
`run_full_eval.sh`, `wave6_distill_seed.sh` and `wave7_armH_seed.sh` all honour
`ZL_BV` / `ZL_TRAIN` / `CARGO_TARGET_DIR` env pointers, so an agent that only
*consumes* binaries never needs to build at all. The playbook now says so.

**The real build-side cost is disk, not time:** 28 target dirs, **113.6 GB**,
on a root filesystem at **95 % full (79 G free)**. Two are pathological — 60 GB
(`zensimissues-target`) and 18 GB (`zensimcons-target`). That is a cleanup
task, not a caching task.

---

## 6. The silent-hook failure (cause 4), in detail

The coherence wave's local lane *did* run a post-bake auto-eval. It failed
**nine times** with:

```
bake_verdict: MISSING corpus JPEG-AI SDR25 (HQ-zone human) at .../ext_sdr25.parquet
bake_verdict exited with exit status: 2
```

Non-fatal, logged into `~/tmp/sota944/co_local.log`, unread. The lane ran
`12,229 s` (3 h 24 min) and produced **zero usable verdicts**. All 21 were
re-run by hand starting `20:53Z` at a mean of 38.3 s each — **804 s = 13.4 min**
of pure recomputation, plus the delay it added to the harvest.

This is worse than having no hook, because "the driver auto-evals" was believed
true while it was false. `scripts/harvest_bakes.sh` therefore writes a
`.HARVEST_FAILED` marker **next to the bake**, appends to a failures file, and
exits 6 — three independent pieces of evidence, none of which requires reading
a log.

---

## 7. What was NOT the problem (measured)

- **Build caching** — §5. 23 min/day total; a shared dir would cost more.
- **The `Monitor` tool** — all 17 monitors started fine and return immediately;
  they do not block or bill while waiting. The failure is agent-side: an agent
  that stops after arming a monitor, or arms one on `tail -f` (which loses the
  file on rotation) instead of on a file that appears exactly once.
- **The pullers** — wave 4's puller pulled `C_co3a_s1361` **86 s** after lianli
  wrote it, and `C_co3a_s1409` **57 s** after. Remote-to-local transfer is not
  where time goes.
- **Poll frequency** — §4: polling reads a warm cache and is nearly free.

I could not reconstruct a defensible per-monitor "death count" from the
transcripts: a monitor's task-id appears in its own tool result as well as in
later events, so any regex count is ambiguous. Rather than publish a number I
cannot defend, §2 measures the *observable consequence* — dead time — which is
what actually costs.

---

## 8. Compute-bound vs orchestration-bound — what each fix buys

Sized so the two workstreams can be compared rather than confused. A sibling
agent is separately optimizing `bake_verdict` (35 s full panel) and the trainer
(~35 min/seed, validation-dominated); those are the inputs to the right-hand
column.

| pool | size today | which fix moves it |
|---|---|---|
| dead wall-clock (unharvested + unqueued) | **6.77 h** | orchestration — this audit |
| cache re-charge from idle | **$395.24 / 13.9 %** | orchestration — this audit |
| `bake_verdict` wall-clock (162 logged runs, mean 36.4 s) | **1.64 h** | perf (sibling) **and** orchestration (below) |
| trainer + lane job-hours (49 `run-heavy` jobs; overlapping lanes, so job-hours not wall-clock) | **13.33 h** | perf (sibling) |
| compute-bound idle wall-clock | **8.03 h** | perf (sibling) |
| `cargo` wall-clock, all agents | **23.0 min** | nothing — §5 |

Artifacts produced in the window: 106 bakes, 118 `full.json`, 114 `verdict.md`,
172 fullevals.

**The two workstreams compose, and one number belongs to both.** The 1.64 h of
`bake_verdict` was spent *serially, after* training finished — 21 coherence
verdicts run in a block at 20:53Z, wave-4's `verdict_daemon` after the lane. Run
inline via `harvest_bakes.sh`, that same work overlaps the *next* seed's
training and disappears from the critical path entirely. So harvest-inline buys
~1.6 h of compute-bound time on top of the 6.77 h of dead time — without making
`bake_verdict` one millisecond faster. Halving `bake_verdict` on top of that
saves ~0.8 h; the two are additive and neither substitutes for the other.

## 9. Fixes landed with this audit

| fix | addresses | evidence it addresses |
|---|---|---|
| `scripts/await_artifacts.sh` | rank 1, 2 | always writes `<hb>.done` (COMPLETE / TIMEOUT / SIGNAL + rc) — verified on all four exit paths; a hand-rolled loop leaves nothing |
| `scripts/harvest_bakes.sh` | rank 1, 4 | verdict+fulleval per bake as it lands (wave 6's uncommitted `process.sh`, generalized); fails loud — verified rc=6 + marker + failures file |
| `docs/WAVE_PLAYBOOK.md` | ranks 1–3 | the skeleton, with each anti-pattern carrying its measured cost from this audit |
| `CLAUDE.md` § *Latency + token discipline* | §4 | the $395.24 / 13.9 % rule, with the 22.9 M-token measurement behind it |

Deliberately **not** changed: the shared build cache (§5, measured and
rejected), the `Monitor` tool usage pattern (it is correct; the agent
discipline around it was not), and the target-dir disk bloat (flagged in §5 as
a cleanup task, out of scope for a cycle-time fix).
