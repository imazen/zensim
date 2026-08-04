# WAVE_PLAYBOOK — the skeleton every R&D wave uses

**Read this before launching any wave, arm, or sweep that runs detached compute.**

Every rule here is priced. The costs come from
[`benchmarks/rnd_cycle_audit_2026-08-04.md`](../benchmarks/rnd_cycle_audit_2026-08-04.md),
which measured the 2026-08-03/04 campaign: 34.3 h of wall-clock, 11 waves,
**14.80 h of whole-session idle of which 6.77 h was dead** — nothing computing,
or finished work nobody had looked at — and **$395.24 (13.9 % of a $2,837
session) burned re-creating prompt cache that idle waiting had expired.**

The compute was fine. The orchestration around it was the whole loss.

---

## The skeleton

```
1. PRE-REGISTER      arms, seeds, gates, decision rule -> commit BEFORE launching
2. BUILD ONCE        build the binaries; export ZL_* pointers for every consumer
3. LAUNCH DETACHED   setsid nohup <driver>            (one lane per executor)
4. HARVEST INLINE    setsid nohup scripts/harvest_bakes.sh --glob ... --count N
5. ONE TERMINAL      setsid nohup scripts/await_artifacts.sh --glob ... --count N
                     ... then Monitor exactly ONE file: <heartbeat>.done
6. SELECT            freeze_check --select <every fulleval> [--tsv]
7. ENDGAME FOREGROUND  tables, doc, gates — in the foreground, no waiting
8. PUSH + VERIFY     jj bookmark set main -r @ && jj git push --bookmark main
                     git merge-base --is-ancestor @ main@origin
9. CLEAN UP          workspace forget + rm -rf; drop your .workongoing line
```

Steps 4 and 5 are the ones that were missing. Step 4 makes a late wake-up
**free**; step 5 makes a dead waiter **visible**.

---

## Step 4 — harvest inline. Never batch the eval at the end.

```bash
setsid nohup scripts/harvest_bakes.sh \
    --glob '/mnt/v/output/zensim/bakes/sota944/bakes/C_w8_s*.bin' \
    --count 12 --regime 944 \
    --heartbeat ~/tmp/wave8/harvest --timeout 21600 >/dev/null 2>&1 &
```

Each bake is verdicted + fullevaled the moment it lands. By the time anyone
looks, the results exist. **This is what makes a missed wake-up cost zero.**

> **Measured:** wave 6 arm F did exactly this with an uncommitted
> `~/tmp/wave6/process.sh`. `ALL SIX PROCESSED 03:08:40Z`; results commit
> `05:20:40Z`. The daemon was perfect and the wave still lost **125.6 min** —
> because nothing woke the agent. Harvest-inline bounds the damage; it does not
> by itself remove it. You need step 5 too.

> **Measured:** the coherence wave's driver *had* an inline auto-eval that
> exited 2 on a missing corpus, **nine times**, into a log nobody read. 3 h
> 24 min of training produced zero verdicts; 21 were re-run by hand
> (804 s). `harvest_bakes.sh` writes `<bake>.HARVEST_FAILED`, appends to
> `<heartbeat>.failures`, and exits **6** — you cannot not notice.

Check a harvest is healthy without reading a log:

```bash
cat ~/tmp/wave8/harvest.status     # rewritten every poll — a stale mtime = dead
ls  /mnt/v/.../bakes/*.HARVEST_FAILED 2>/dev/null   # empty = clean
```

## Step 5 — one terminal condition, and it is a FILE

```bash
setsid nohup scripts/await_artifacts.sh \
    --glob '/mnt/v/output/zensim/reports/fulleval/C_w8_s*.fulleval.json' \
    --count 12 --label 'wave-8 fullevals' \
    --heartbeat ~/tmp/wave8/await --timeout 14400 >/dev/null 2>&1 &
```

Then arm **one** `Monitor` on `~/tmp/wave8/await.done` and go do other work.

`await_artifacts.sh` guarantees the sentinel is written on **every** exit path —
COMPLETE, TIMEOUT (rc 3), or SIGNAL (rc 5) — via an `EXIT` trap, and it sleeps
as `sleep & wait` so a signal is honoured immediately rather than after the
poll interval. SIGKILL cannot be trapped; that case shows up as a
`<heartbeat>.status` whose mtime stops advancing, which is still evidence.

**Watch the terminal artifact, not the last one.** Await the *fullevals*, not
the bakes — the bake landing is the middle of the pipeline, and an agent woken
by it still has 35 s × N of eval to run serially in the foreground.

---

## Anti-patterns, each with what it cost

| anti-pattern | measured cost | do this instead |
|---|---|---|
| Hand-rolled `while sleep` / `tail -f` waiter that exits without a trace | the day's two worst events: **125.6 min** + **80.6 min** dead | `await_artifacts.sh` — sentinel on every exit path |
| `Monitor` armed on `tail -f <log>` | loses the file on rotation/truncate, then waits forever | `Monitor` a file that appears **exactly once** (`<hb>.done`) |
| Batching verdicts+fullevals at the end of the wave | a late wake-up costs its own delay **plus** the whole eval chain | `harvest_bakes.sh` inline |
| A post-bake hook whose failure is non-fatal | **13.4 min** recompute + a 3 h 24 min lane silently voided | fail loud: marker file + failures file + nonzero exit |
| Agent parks and polls on short intervals | polls are cheap, but the **wake-up** after >5 min re-creates the whole prefix: **$395.24, 13.9 %** of session cost | arm one Monitor, then work on something else |
| Supervisor idle-waits on a delegated wave | its context ages while a subagent's does too — **two** prefixes re-charged per event | supervisor does independent work; artifacts are the channel |
| Static split of cells across lanes | local box idle **57.5 min** while lianli held the critical path (coherence wave) | rebalance: give the slower lane fewer cells, or pull work back |
| No work queued while an agent thinks | **1.46 h** of the day with nothing running *and* nobody awake | launch the next arm before writing up the last one |
| A second `cargo` on a shared `CARGO_TARGET_DIR` | **31.8 s** blocked on the build lock, per invocation | per-agent target dirs + `ZL_*` binary handoff (below) |

---

## Builds: do not build if you only consume

Measured: total `cargo` wall-clock across every agent, all day, was **23.0 min**
(91 invocations; a cold `bake_verdict` build is 72 s / 221 crates). A shared
target dir is **not** worth it — see the audit §5. Per-agent dirs stay.

But an agent that only *runs* binaries should not build at all. Every driver
honours env pointers:

```bash
export ZL_BV=/home/lilith/work/zen/zensim/target/release/bake_verdict
export ZL_TRAIN=/home/lilith/work/zen/zensim/target/release/zensim_mlp_train
# or, to reuse a whole tree:
export CARGO_TARGET_DIR=$HOME/tmp/zensimw8-target
```

`sota944_verdict.sh`, `run_full_eval.sh`, `harvest_bakes.sh`,
`wave6_distill_seed.sh` and `wave7_armH_seed.sh` all resolve binaries through
`ZL_*` → `CARGO_TARGET_DIR` → `$REPO_ROOT/target`.

Target dirs are cheap in time and **expensive in disk**: 28 dirs, 113.6 GB, on
a root filesystem at 95 %. Delete yours when the wave closes.

---

## Step 6 — SELECT on rank + dial + COHERENCE, with the owner

```bash
freeze_check --select /mnt/v/output/zensim/reports/fulleval/C_w8_s*.fulleval.json
```

The wave's decision is not "eyeball the scoreboard". It is a **registered
rule**, implemented in `freeze_check` (the bar/profile owner — do NOT write a
ranking script):

- **PRIMARY: profile floor count.** Coherence never overrides a bake that
  fails CID22 or the dial.
- **TIE-BREAK: `balanced_composite + 0.15 · M3a`.** Registered in the sota944
  campaign doc appendix E.4. 0.15 is the same weight class
  `balanced_composite` already gives csiq/live/band-tail.
- **Why M3a is in the rule at all:** the coherence study (n = 50,
  pre-registered) measured that **42.3 % of 944-class M3a variance is seed
  noise at fixed recipe** — `C_co3a` k = 6 spans 0.718–0.826. Coherence is a
  *selectable trajectory property*, so a k-seed wave that ignores it is
  leaving a free 0.1 of M3a on the table.
- **`sdr25` is NOT the primary.** It is a reported comparator column. It has
  decoupled from CID22 five times; that is exactly why the primary is the
  floor count.
- **Three M3a states, none of them zero.** `MEASURED` ranks normally;
  `NOT COMPUTABLE` (ensemble — the instrument loads one ZNPR) ranks in a
  separate section and is never penalized; `UNMEASURED` is listed but **not
  selectable**, and the tool prints the command to fix it.

M3a comes free with harvest (`run_full_eval.sh` computes it), so by the time
step 5 says COMPLETE the selection data already exists. If it does not,
harvest drops a `.NO_M3A` marker next to the bake and counts it in the
terminal line — read that before selecting.

---

## Endgame in the foreground

Once `<heartbeat>.done` says COMPLETE, everything left is bounded: read the
fullevals (already computed), build the tables, write the doc, run the gates,
commit, push, verify, clean up. **Do not arm another waiter for a bounded
endgame** — run it inline. The audit's `A8 balanced-selection` pass shows what
an unbounded endgame costs: a 15-minute tail on a pass whose compute was done.

Terminal checklist:

```bash
just lint-scripts                              # any script you added
jj bookmark set main -r @ && jj git push --bookmark main
git -C /home/lilith/work/zen/zensim merge-base --is-ancestor @ main@origin
jj workspace forget <name> && rm -rf <path>    # mandatory on merge
```

---

## The 30-second self-check before you go quiet

1. Is something computing **right now**? If not, why not — launch it.
2. Will the results be **fully evaluated** when they land, or will I have to
   run the eval myself after I wake?
3. Is there **exactly one** file whose appearance means "done", and is
   something watching it that leaves evidence if it dies?
4. Am I about to wait, or about to **work on something else**? Waiting attached
   is what costs $395/day.
