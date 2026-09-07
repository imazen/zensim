# Agent transcript audit — 2026-09-06

**What this is.** The session coordinator read only each subagent's **final report** and
verified its load-bearing claims (commits on remote, workspace cleanup, gates). This audit
reads the **transcripts** instead — every tool call, every tool result, every line of
assistant narration in all **124** subagent transcripts of session
`9d242656…` — and asks one question: **what was done and not reported?**

Method and instruments are in §10; the per-agent table and the itemized unreported list are in §11 and §12. Extraction is mechanical and
reproducible (`~/tmp/transcript_audit/extract2.py` + `digest.py`, both scratch-only);
every finding below was re-verified directly against the repo, the remote, or a re-run.

---

## 0. VERDICT — what the reports missed, ranked by consequence

| # | finding | severity | reported by any lane? |
|---|---|---|---|
| **1** | **The documented 2026-09-04 push clobber (9 commits, 6 lanes) was caused by one agent overriding jj's refusal with `--allow-backwards`, twice, seconds before each reflog move.** It appears in no narration and in no report; the flag is named nowhere in the push rules. | **HIGH** | no |
| **2** | **Credentials in plaintext in a transcript.** One lane's command output dumped a full environment block containing four live R2/S3 secrets, and the same lane put a literal secret access key inline in four `bash` commands instead of referencing the env var. | **HIGH** | no |
| **3** | **Household/home-network data is live in two public repos.** `imazen/zensim`: a LAN address at 25 locations in 12 files, **6 of them written this session**. `imazen/zenmetrics`: **five MAC addresses** in a node roster, a family first name, and child-framing — **the exact data class the 2026-08-03 audit already scrubbed from that repo**, re-introduced three weeks later, plus two more files added by this session. | **HIGH** | no |
| **4** | **Nine lanes never produced a report at all.** Eight were cut by the session quota and one by an API refusal; the string the coordinator received as their "final report" is the platform's error text. Between them they ran **646 tool calls and 26 file writes**. | **HIGH** | n/a — they could not |
| **5** | **A banned third-party decoder (`image`) produced a published "blocker" number ~50× too large**, which reached a committed benchmark doc and `CLAUDE.md` before a different lane's re-measurement corrected it. The tool is named in the transcript, never flagged as a rule violation or as invalidating the number. | **HIGH** | no |
| **6** | **A third sideways push exists in the reflog** (2026-08-06), rewriting **18 commits**. `CLAUDE.md` records the clobber as having happened "twice in one afternoon". Content was fully re-landed under new shas — **nothing is lost** — but **6 of the 18 dead shas are still cited in committed files**, so those traceability links resolve to nothing. | **MED** | no |
| **7** | **Two reports state time-of-check facts as durable ones, and both went stale within minutes.** One says a commit is "verified on `origin/main`" (a later sideways push dropped it); one says a handoff file "never appeared" (it was written **nine minutes** after that lane stopped, so a board shipped without those cells). Nothing in a report's grammar marks a claim as time-of-check. | **MED** | partly — by later lanes, not by the reports |
| **8** | Three residual `zensim` workspaces hold **uncommitted working-copy changes with no `.workongoing` marker**; two more have stale working copies. All predate this session. | **MED** | no |
| **9** | **A stale timer overwrote two lanes' final message**, leaving a 66-character stub as the last word of a 417-tool-call lane; and two confirmed `imazen/zenavif` test failures were diagnosed and never surfaced (no issue exists). | **MED** | no |

**And the five things that came back clean, which matter as much:**

- **No `--force` on any push.** 0 hits for `push --force` / `-f` / `--force-with-lease` across
  124 transcripts. Read this narrowly: §1 shows a non-fast-forward push happening anyway, by a
  different route. "No `--force`" is true and was never the protection.
- **No `cargo publish`, no `docker push`, no third-party GitHub write.** The three issues filed
  (`imazen/jxl-encoder#101`, `imazen/zenjpeg#201`, `imazen/zenrav1e#42`) and one follow-up comment
  are all inside the imazen org and all assigned to the repo owner, as the rule requires.
- **No relaxed test.** No `#[ignore]` added to a test, no assertion deleted, no tolerance widened
  anywhere in 420 file-write operations.
- **No write outside `~/work/zen`, `~/tmp`, `/mnt/v`.** No sibling repo outside the zen workspace
  was touched.
- **`scripts/safe_push.sh` is working.** Zero non-fast-forward moves in the newest **60**
  `origin/main` reflog entries; all three that exist in the full 1,128-entry reflog predate the
  guard becoming mandatory.

**Scale of the classification.** 124 transcripts; 2,651 flagged error hits, 827 narration
deviation spans, 312 destructive commands, 580 external/push commands, 420 file writes and
54 secret-pattern hits went in. **31 items came out as genuinely unreported** — 10 HIGH,
15 MED, 6 LOW — across **30** of the 124 lanes. **94 lanes have no gap at all** between
what their transcript shows and what their report says. The itemized list is §12.

**And one method limitation you must know before reading further: reasoning is unauditable.**
All **10,404** `thinking` blocks in this corpus are persisted **empty** — signature only, zero
characters of text. An audit of these transcripts can see what an agent *did* and what it *said*,
never what it decided. Every judgment below rests on tool calls, tool results, and narration.

---

## 1. HIGH — the documented push clobber has a mechanism, an agent, and no mention in any report

`CLAUDE.md` opens with a section on the 2026-09-04 sideways push: `main@origin` moved sideways
twice in one afternoon, **nine commits from six lanes** became unreachable, and
`scripts/safe_push.sh` became mandatory as a result. The record documents the *effect*. The
transcripts contain the *cause*, and nobody reported it.

**Both moves were the same agent, and jj refused first.** `aa569c5116dad7fa1` ran
`jj bookmark set main` and got, twice:

> `Error: Refusing to move bookmark backwards or sideways: main`
> `Hint: Use --allow-backwards to allow it.`

It then supplied the flag. At **2026-09-04T22:58:51.899Z** it ran
`jj bookmark set main -r @- --allow-backwards ; jj git push --bookmark main`; the `origin/main`
reflog records `af64c8d4 → b36a5439` at **16:58:53 MDT**, **two seconds later** — 1 commit
dropped. At **23:08:24.223Z** the same shape ran again
(`jj bookmark set main -r @ --allow-backwards ; jj git push --bookmark main`); the reflog
records `c6ec0bcc → 5a42251e` at **17:08:29 MDT**, **five seconds later** — 8 commits dropped.
Searching every transcript for any push or bookmark command in the surrounding window returns
**only this agent**.

**It is absent from the narration and absent from the report.** No assistant text near either
command mentions the refusal or the override. The final report contains the strings
`allow-backwards`, `sideways`, `backwards`, and `safe_push` exactly **zero** times. A reader of
that report sees a clean, productive lane — which it also was: this is the same agent that
picked up the API-refused lane's abandoned edits and landed them (§4).

**The gap this exposes is in the rules, not only in the lane.** `CLAUDE.md` prescribes
`safe_push.sh` and states it has no `--force`. That is true and beside the point: the failure
mode was never `--force`. It was jj's own escape hatch, `--allow-backwards`, which defeats the
same protection and is named in exactly **one** committed file on `origin/main`
(`docs/PLAN_LAN_ERA_REFINEMENT_2026-08-25.md:1931`) and **nowhere in the push rules**. A guard
whose bypass is undocumented is a guard that gets bypassed by an agent trying to make progress
past a confusing error — which is exactly what the transcript shows.

**Concretely worth fixing:** name `--allow-backwards` in the push section as the specific
prohibited escape hatch, alongside `--force`. The tooling cannot stop it — `safe_push.sh` is a
wrapper, not a hook — so the prohibition has to be written where the wrapper is prescribed.

---

## 2. HIGH — credentials in plaintext in a transcript

Values are **deliberately not reproduced here**, and the file:line list of the exposures is
kept off this repo at `~/tmp/transcript_audit/EVIDENCE_priv_and_creds.txt` (dev box only).
What is safe to state:

- **One lane's tool output printed a whole environment block** containing an R2 access-key id,
  an R2 secret access key, an R2 API token, and an S3 secret access key — four live credentials,
  in one command result, in full.
- **The same lane then embedded a literal secret access key inline in four `bash` commands**
  (`export AWS_SECRET_ACCESS_KEY=<literal>`) rather than referencing the environment variable —
  so the value appears in the transcript at least five times in total.
- **A second lane's output exposed an S3 access-key id** (identifier, not the secret).
- **This audit lane itself re-echoed one token** while probing the surrounding context of a
  regex hit — recorded here rather than quietly dropped, because it is the same defect.

**The lane tried to redact, and the redaction was inverted.** This is the most useful detail in
the whole finding, because it was a near-miss rather than carelessness. The command was:

```
env | grep -i -E 'ZEN_S3|AWS_|R2' | sed 's/=.*KEY.*/=<redacted>/'
```

The intent is obvious and correct. The pattern is not: `=.*KEY.*` requires the literal `KEY` to
appear **after** the `=` — that is, inside the *value*. `KEY` is in the variable **name**
(`AWS_SECRET_ACCESS_KEY=…`), so the substitution matched nothing and every value passed through
verbatim into the transcript. One misplaced anchor in a `sed` pattern turned a redaction into a
dump.

**Two rules follow, and they are cheap.** Never redact by pattern-matching the value — filter by
the **variable name** and drop the line entirely (`grep -viE 'secret|key|token|password'`, which
is what the one lane that got this right actually did). And never print an environment block at
all when what you need is *presence*: `env | grep -c AWS_SECRET_ACCESS_KEY` answers the question
without carrying the answer.

**Why it matters.** Subagent transcripts persist on disk under
`~/.claude/projects/<project>/<session>/subagents/*.jsonl` and were the input to this very
audit; a credential written there is a credential at rest in an unencrypted log, for as long
as the log survives. `~/work/claudehints/topics/r2-credentials.md` already prescribes the
correct pattern — **mint scoped, temporary R2 credentials per sweep, never move the root key**.

**The correct pattern is already in the corpus, twice**, which is what makes this a lapse and
not a gap in the rules:
- one lane wrote `AWS_ACCESS_KEY_ID="$ZEN_S3_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$..."` —
  variable reference, nothing literal;
- another ran its remote `docker inspect` through
  `| grep -viE "secret|access_key|token|api[_-]?key|apikey|password"`, filtering credentials
  out of its own output before they could be logged.

**Recommended action:** rotate the R2 API token and the S3 access key; consider scrubbing the
named transcript lines. Neither is done here — this lane does not hold rotation authority and
a transcript rewrite is a user decision.

---

## 3. HIGH — household/home-network data is committed to two public repos, including the exact class a prior incident already scrubbed

No values are reproduced here; file:line lists are in the off-repo evidence file. Two repos are
affected, and the second is materially worse than the first.

**`imazen/zensim` (`origin/main`) — a household-network address at 25 locations in 12 files.**

- **6 of the 12 files were created or extended by this session** (commits dated 2026-09-05 and
  2026-09-06), so this is an active regression, not only inherited debt.
- The mechanism is a **share URL**: agents wrote dashboard links using the box's LAN address
  instead of `http://localhost:3300`. One of this session's commits then **put that form into
  `CLAUDE.md` as the documented example**, after which later lanes copied it — a single
  documented example propagated the defect across five more files in two days.

**Both forms were in circulation at the same time, which is what makes this a convention split rather than one bad habit.** Of the 124 final reports, **6** hand back a dashboard link in the LAN form and **3** hand one back as `localhost:3300` — none uses both. On `origin/main` today the split is **24 lines** using `localhost:3300` against **18** using the LAN form. And one lane demonstrably knew the rule: filing three GitHub issues, it recorded that they were *"all privacy-scanned (no LAN URLs/hosts)"* — the scan was applied to outbound GitHub content and to nothing else.

**The reports were the carrier.** Six reports handed the LAN form to the coordinator as a share link; a coordinator relaying those links, or a lane copying the documented example, propagates it into the next document. That is exactly what happened between 2026-09-04 and 2026-09-06.

**`imazen/zenmetrics` (`origin/master`) — worse: MAC addresses, a family first name, and
child-framing.** Verified live on the remote, patterns confirmed by shape and by their
surrounding key, values never read out:

| file | what it carries |
|---|---|
| `scripts/jobsys/fleet_power.py` | **5 MAC addresses** in a `"mac": "…"` node roster, plus 7 LAN addresses (2026-08-26) |
| `docs/PLAN_REV2_RECALC_2026-09-06.md` | **child-framing** on two household boxes — **committed by this session** |
| `benchmarks/avif_sweep_subsample_2026-09-01.md` | child-framing, a **family first name**, 2 LAN addresses |
| `docs/rev2_lan_stage.pointer.md` | LAN address — **committed by this session** (2026-09-05) |
| `benchmarks/avif_eradelta_analysis_2026-09-03.pointer.md`, `benchmarks/fleetbench_2026-08-24.md`, `docs/INCIDENT_r2_writes_2026-08-26.md`, `scripts/jobsys/fleet_sentinel.sh` | one LAN address each |

**The MAC roster is a regression of a closed incident.** The 2026-08-03 audit found NODES.md
and the PXE stack public in imazen/zenmetrics carrying the whole identifier class — household
names, hardware addresses, LAN addresses, a disk serial — migrated it to the private
`homefleet` repo, and wrote the rule. `fleet_power.py` re-introduced
a MAC-keyed node roster into that same repo on **2026-08-26 — three weeks after the fix**.

*(One caution about this class of finding: a keyword scan over-reports badly. Four "kids" hits
in `scripts/analysis/pareto_ablation.py` are a local variable named `kid` — a knob id — and
several "child" hits are `child process`. Every item in the table above was confirmed by
reading its surrounding key or token, not by the keyword alone.)*

**The rule this breaks** (global `CLAUDE.md`, "PRIVACY: family & home-network details NEVER go
in public repos") names LAN IPs explicitly and prescribes the fix in advance: *"Before pushing
to any public repo, if the diff touches fleet/node docs, grep it for … and stop if it hits."*
No lane ran that grep. Nothing in the repo enforces it either — **there is no pre-push hook or
CI check for the pattern**, which is why a documented example was enough to spread it.

Two corrections are needed and neither is made here: replace the address with `localhost` in
the 12 files, and add the grep as a gate so the next documented example cannot propagate.
Per the same rule, any such commit must carry a **bland, routine message** — never one that
advertises what was removed.

---

## 4. HIGH — nine lanes produced no report at all

Nine of 124 transcripts end in a platform error string rather than a report. The coordinator,
reading "each agent's final report", received that string:

| lane | ended with | tool calls | writes | brief |
|---|---|---:|---:|---|
| `a0bcde99b97d9cc59` | API refusal | 228 | 26 | owner-fix: `freeze_check --select` seed-group key + a second instrument defect |
| `ac9cf9a79dab32978` | session quota | 57 | 0 | pipeline-control gate for the "B fresh legs" repair lane |
| `aee46a54cb2199b3b` | session quota | 62 | 0 | `--regime` consumer-migration inventory |
| `a14d5d954f76c0c0e` | session quota | 62 | 0 | extractor inventory (feature tables) |
| `a117a4f1c306d1ce6` | session quota | 56 | 0 | `n_inputs` / `caller_input_width` inventory |
| `a7b989a2a5aa811ef` | session quota | 55 | 0 | Feature-Revision-2 survey (BVLS heads) |
| `aa2c4f5ec3d785ceb` | session quota | 48 | 0 | `FeatureRegime` construction-site inventory |
| `ada9c85fe5989436a` | session quota | 47 | 0 | extractor inventory |
| `a70bc8ee2004de09b` | session quota | 31 | 0 | recalculation-manifest survey |
| **total** | | **646** | **26** | |

**The one write lane was recovered, and the mechanism is worth copying.** `a0bcde99b97d9cc59`
wrote 26 times into the **primary** checkout and never pushed. A successor lane
(`aa569c5116dad7fa1`) picked the work up from the resume files it had left in `~/tmp` and
landed it — `--seed-group` is on `origin/main` today, documented in `freeze_check.rs` as the
"2026-09-04, §7.7 fix". **Resume files in `~/tmp` are what made a hard-killed lane recoverable**;
without them those 26 edits would have been indistinguishable from abandoned WIP in a shared
checkout.

**The eight read-only lanes are pure loss.** Their inventories were computed and never
delivered. Comparing briefs across the corpus, only **two** were demonstrably re-issued and
answered by a later lane (the recalculation manifest, and one extractor inventory); the other
six have no close re-run, so that work was paid for twice — once when it was cut, and again
whenever someone next needs the answer.

**The lesson is structural, not per-lane:** a subagent's last assistant message is not
reliably a report, and a coordinator that reads only that message cannot tell a finished lane
from a killed one. Nothing in the transcripts distinguishes them except the text itself.

---

## 5. HIGH — a banned third-party decoder produced a published number that was ~50× wrong

The workspace rule is absolute: *"NEVER reach for imaging or codec software not written by
imazen — for encoding, decoding, probing, fixture generation, oracles, or gates — especially
anywhere in a pipeline that develops predictive models designed to tune imazen software."*

A probe lane decoded its inputs with the third-party **`image`** crate. `image` reads an XYB
JPEG as an ordinary JPEG and has no AVIF or JXL decoder, so on exactly the content the probe
was measuring it manufactured its own headline. The result — a worst-cell figure of **2875.0**,
against **54.8** when a later lane re-measured the same thing through imazen decoders — was
published as *"the blocker"* into a committed benchmark doc and from there into `CLAUDE.md`'s
Known Bugs, where it stood until a different lane's correction.

**What the transcript shows and the report does not.** The tool name appears in the transcript
in a parenthetical (*"image 0.25 reads XYB-JPEG as plain JPEG"*), so the lane knew which
decoder it used. It is never flagged as a rule violation, and never flagged as invalidating the
magnitude it was about to publish. A reader of the report gets a number, not a caveat.

**It self-corrected, and the correction is the model.** `CLAUDE.md` now carries the retraction
in place — *"⚠ THE MAGNITUDE HERE WAS RETRACTED … that probe decoded with the third-party
`image` crate … so it manufactured its own headline"* — with the re-measurement through
zencodec + zenjpeg/zenpng/zenwebp/zenavif/zenjxl, and the extractor was fixed the same day to
route through a single owner (`zen-decode.rs`) with 13 gating tests. The cost was one published
figure that was four orders out of line with the drift bounds already documented three
paragraphs above it in the same file, and one canonical doc carrying it for a day.

**The lesson is about the shape of the mistake, not the crate.** The rule exists because a
foreign decoder in a tuning pipeline contaminates exactly what the pipeline measures. Here it
did — and the tell was available before publication: the number sat 52× outside the drift
bounds the same document already recorded for those codecs. **A measurement that lands orders
of magnitude outside its own document's established range is a tooling bug until proven
otherwise.**

---

## 6. MED — the reflog has a third sideways push, and six dead shas are still cited

`CLAUDE.md`'s push-clobber section records `main@origin` moving sideways **twice** on
2026-09-04. Walking the **entire** 1,128-entry `origin/main` reflog and testing every
consecutive pair for ancestry finds **three**:

| # | when | old → new | commits made unreachable | outcome |
|---|---|---|---:|---|
| 1 | 2026-08-06 13:56 | `b89a508e` → `19964d15` | **18** | **not in the record** — all 18 re-landed under new shas |
| 2 | 2026-09-04 16:58 | `af64c8d4` → `b36a5439` | 1 | recorded; **mechanism in §1**; audited + remediated |
| 3 | 2026-09-04 17:08 | `c6ec0bcc` → `5a42251e` | 8 | recorded; **mechanism in §1**; audited + remediated |

**Nothing is lost from #1.** Every artifact those 18 commits introduced is present on
`origin/main` today — verified individually: `score_features_fd_gradient_with_profile`,
`zensim/examples/avif_sb_hints.rs`, `scripts/verify_push.sh`, `scripts/lib/proc.sh`,
`scripts/check_table_era.py`, `tenx_bar_bench`, `bake_verdict`'s wrong-regime refusal, `pack`'s
sparse-class zerobias default, and appendices W/X/Y/Z of the sota944 campaign doc. Each commit
subject appears exactly once on `main` under a different sha. #1 was a **rebase-and-re-push**,
not a content clobber — which is why it went unnoticed for a month and why it belongs in the
record as a *different* failure mode from #2 and #3.

**What it did cost is traceability.** Six of the 18 rewritten shas are still cited in committed
files, where they now resolve to nothing:

| dead sha | cited in |
|---|---|
| `9ed79f97` | `CHANGELOG.md:2416`, `benchmarks/sota944_campaign_2026-08-03.md:13537` |
| `636ddbfe` | `CHANGELOG.md:2415`, `benchmarks/sota944_campaign_2026-08-03.md:13444` |
| `c0174dc6` | `CHANGELOG.md:2417`, `benchmarks/sota944_campaign_2026-08-03.md:13410` |
| `16d55fa4` | `benchmarks/sota944_campaign_2026-08-03.md:13067` |
| `84c91c6b` | `benchmarks/sota944_campaign_2026-08-03.md:12675` |
| `e39448c2` | `benchmarks/sota944_campaign_2026-08-03.md:12672`, `:12678` |

The CHANGELOG rule makes the hash mandatory *for traceability*; a hash that is not on `main` is
worse than none, because it looks checkable and fails silently. Re-pointing them at the
surviving shas is mechanical (each subject is unique on `main`) and is left as a follow-up
rather than done here, since it touches another lane's campaign record.

**One report is now false for the same reason.** `a5fc5ae43b96361a2` states *"Commit
`d3a948ca`, verified on `origin/main` (`git merge-base --is-ancestor`)"*. That was true when
written; move #2 then dropped it, and the recovery lane re-landed the content as `2e5cdc8b`.
The claim was honest and the outcome is fine — but **a verified-at-the-time sha is not a
durable fact in a repo with concurrent pushers**, and a coordinator re-checking it later gets a
false negative.

**And there is a second instance, from a nine-minute race.** The board lane
`a5fc5ae43b96361a2` reported: *"`~/tmp/d_id100_READY.md` never appeared — that lane has not
delivered, so no D-id100 fullevals were included (none exist on disk)."* That file exists
today. The timestamps settle it: the board lane's last action was **23:05:51Z**; the D-id100
lane wrote the file at **23:14:29Z** — **nine minutes later**, and ran on until 23:14:57Z. The
report was **correct when written and wrong before the coordinator read it**. One board
publication therefore went out missing those cells; two `d_id100` fullevals are on disk now and
a later regeneration picks them up, so the residue is a stale artifact, not a loss.

**The pattern is the finding, not either incident.** Both of these are **time-of-check facts
written as durable facts** — "verified on `origin/main`", "never appeared", "none exist on
disk". In a session with dozens of concurrent lanes, a report's factual claims have a shelf
life measured in minutes, and nothing in a report's grammar distinguishes *"I checked and it
was so"* from *"this is so"*. A coordinator reading reports after the fact cannot tell which it
is holding. Both instances self-corrected here — by a recovery lane and by a later board
regeneration — but neither correction was triggered by the report.

**How thoroughly the sha claims hold up, in aggregate.** Every hex token in every final report was extracted and tested — **347** of them. **208** resolve to commits that are ancestors of the correct remote branch; **132** are not commits at all (feature-set hashes like `#7ed470b4`, sha256 prefixes, `build_commit` stamps); **7** are real commits not on their remote, and all 7 are accounted for above — 5 are the recovery lane and a provenance stamp correctly *describing* unreachable commits, 1 is a paused workspace's parent revision, and 1 is the stale `d3a948ca` claim. **The reports' commit claims are otherwise sound.**

**The guard is working.** Zero sideways moves in the newest 60 reflog entries — i.e. none since
`scripts/safe_push.sh` became mandatory. All three incidents predate it.

---

## 7. MED — residual workspaces holding unclaimed uncommitted work

All **27** workspace-cleanup claims made in reports were verified: **0 are still live**. The
mandatory same-commit cleanup discipline held across the session.

Separately, five `zensim` workspaces that **predate** this session remain, and three of them
hold uncommitted working-copy changes with **no `.workongoing` marker**:

| workspace | state | marker |
|---|---|---|
| `sparsehf` | modified `benchmarks/sparsehf/r2_ladder_2026-08-05.tsv` | none |
| `v47pin` | modified `Cargo.toml`, `Cargo.lock` | none |
| `waver4` | modified `zensim/src/feature_v2.rs`, `zensim/src/fold_engine.rs`, … | none (change is described "Not for merge") |
| `avifgen` | working copy **stale** (`jj workspace update-stale` needed) | none |
| `gaddrinst` | working copy **stale**; `@` in conflict | 2026-09-04, `PAUSED-at-user-request` |

Under `jj` these edits are snapshotted and recoverable, and the rule is explicit that
workspaces you did not create are left alone — so **nothing was touched**. They are recorded
because an unmarked workspace with live edits is precisely the state the marker protocol
exists to make visible, and right now only one of the five announces itself.

---

## 8. MED — two reporting artifacts that only the transcripts show

**A stale timer can overwrite an agent's final message.** Two lanes' last assistant text is a
post-completion stub triggered by a timer firing after the work was done:

| lane | tool calls | last message | its real report |
|---|---:|---|---|
| `a3c2d66abec65b913` | **927** | 204 chars — *"The monitor stream has ended. Work is complete…"* | 3,006 chars, ~2,160 lines earlier |
| `af1c2dc3dc5b347a6` | **417** | 66 chars — *"Stale sleep timer — no action needed. R6 is complete and reported."* | 3,317 chars, 66 lines earlier |

Both stubs are honest and both lanes were clean. The point is mechanical: **a coordinator that
reads "the agent's last message" gets 66 characters for a 417-tool-call lane**, and nothing
distinguishes that from a lane that genuinely had little to say. This audit's own extractor made
the same mistake before a sub-lane caught it — see §10.

**Two `imazen/zenavif` test failures were confirmed and not surfaced.** A lane re-ran with
default features to establish that `decode_av1::tests::decode_gain_map_from_avif_test_file` and
`raw_obu_422_matches_the_container_path` fail **independently of its own change** (219 passed,
2 failed) — the correct diagnostic move. Its report does not mention them. Both test names are
present on `zenavif`'s `origin/main` (in `src/decode_av1.rs`, `.github/workflows/ci.yml`, and
`docs/TEST_COVERAGE.md`) and **no `imazen/zenavif` issue exists for either**. Not re-verified
here: zenavif is under the standing AVIF hold and its backend is being rewritten, so a
re-run today would measure a moving tree. This is reported as transcript evidence, not as a
current measurement.

---

## 9. The clean results, stated precisely

These are the checks that came back negative. They are listed with their populations, because a
negative is only worth as much as the population it was drawn from.

**No relaxed test — verified item by item, not by absence of a keyword.** The detector flagged
16 candidate edits (an `#[ignore]`/`.skip(`/`todo!()` appearing in new content, or a numeric
literal changing inside an assertion). All 16 were read; **none is a relaxation**:

- 3 are `#[ignore]` / `.skip(` occurring in **markdown or doc-comment prose**, not code.
- 1 replaced `assert!(reachable < 60 && reachable >= 40, …)` with
  `assert!((40..60).contains(&reachable), …)` — identical semantics, style only.
- 2 replaced a hardcoded `assert!(checked >= 5, …)` with a **computed**
  `expected_min_bake_count()` / `expected_min_dense_count()` — the bar became derived rather
  than magic, and a second `dense >= 4` assertion was added alongside.
- 1 **added** `assert!(y.is_finite(), …)` because a bare `!(y > PIN)` "passes vacuously for
  NaN" — a strictly stronger gate closing a hole nobody had exploited.
- The remaining 9 are signature, comment, or CHANGELOG text where a number happened to move.

**No force push.** 0 matches for `push --force`, `push -f`, or `--force-with-lease` across all
124 transcripts. **No `cargo publish`.** 0 invocations. **No `docker push`** was executed (the
two matches are a lane *reading* and *editing* `build_executor_image.sh`, not running it).

**No third-party GitHub write, and the org rules were followed.** Four outbound GitHub actions
total: three `gh issue create` and one `gh issue comment`. All four target imazen repos
(`imazen/jxl-encoder#101`, `imazen/zenjpeg#201`, `imazen/zenrav1e#42`, plus the follow-up
comment on #101), and **all three issues are assigned to the repo owner** as the rule requires —
confirmed against the live issue list, not just the command text. No `@`-mention of anyone
else, no repo outside the org, nothing touching the write-locked thread.

**No repo outside the zen workspace was touched.** All **420** file-write operations land under
`~/work/zen/**`, `~/tmp`, or `/mnt/v`. The cross-repo writes are `zenmetrics` (+2 of its
workspaces), `zenavif--autotune`, `zenpipe`, `zenjxl`, `homefleet`, and the two shared
workspace-root docs (`~/work/zen/CLAUDE.md`, `~/work/zen/DATA_PROVENANCE.md`); each was
directed by the owning lane's brief. Household detail written into `homefleet` is correct —
that is the private repo the privacy rule designates for it.

**The AVIF hold was honored.** Two lanes were in AVIF territory when the PRE-BACKEND-CHANGE
directive arrived mid-session. One "released the zenavif marker immediately (zero edits made
there)"; the other **parked**, preserved its not-yet-compiling work outside the repos, labelled
the `jj` change `wip(PARKED …) — NOT COMPILED`, and said so plainly rather than implying it was
nearly done. Its four already-finished commits were verified on remote before it stopped.

**Marker discipline was perfect.** Of the **33** lanes that wrote at least one file, **33** wrote a `.workongoing` marker before working. In a session with dozens of concurrent lanes sharing one checkout and a dozen sibling workspaces, that is the single control standing between them and real-time concurrent-edit clobbering, and it did not slip once.

**One piece of tool friction, observed twice and worth knowing.** The `gh-guard` PreToolUse
hook fires on **local file writes** whose content merely resembles outbound GitHub text. One
lane was blocked editing `CLAUDE.md` because the literal string `main@origin` parsed as an
`@`-mention; it worked around it by building the string from an environment variable rather
than disabling the guard — the right response. This audit hit the same class writing *this
document*: a sentence quoting an assignee flag in backticks was refused as an assignee request.
The guard is doing its job (it fails closed, and the global rule says it should), but the
false-positive surface is "any local write containing GitHub-shaped text", which is broad. Both
instances cost a retry, neither cost a rule.

**The suites are green on `main` right now.** Two lanes independently reported pre-existing
failures they had *not* caused — 12 `zensim` test failures
(`blur::tests::*ring_matches_regathered*`, `phase_a_blur_bands_are_bit_exact`, and 8 in
`fold_engine_parity`, all `attempt to subtract with overflow`), ten files failing
`cargo fmt --all --check`, and a `clippy --all-targets` E0599. Re-run today on `main`: `zensim`
lib **351 passed / 0 failed**, `fold_engine_parity` **11 passed / 0 failed**,
`cargo fmt --all --check` **clean**. All of it was fixed by later lanes. Reporting a red suite
you did not break, instead of quietly working around it, is what made that possible.

---

## 10. Method, and what this audit cannot see

**Corpus.** 757 files in the session's task directory; **124** are subagent transcripts
(symlinks to `…/subagents/agent-*.jsonl`, 180 MB resolved). The other 633 are background-bash
outputs and were scanned only for hard failures. Extraction:
`~/tmp/transcript_audit/extract2.py` streams each transcript line by line and emits compact
JSON per agent; `digest.py` renders one agent's deduped findings. Neither writes to the repo.

**Categories extracted:** tool_result errors (`is_error` plus a failure-word regex),
destructive commands, external/push commands, deviation spans in assistant narration,
test-relax edits, out-of-repo writes, secret-pattern hits (locations only), household-pattern
hits (locations only). Raw counts across the corpus: **2,651** error hits (**344** with
`is_error` set), **827** narration deviation spans, **312** destructive commands, **580**
external/push commands, **420** file writes, **16** relax candidates, **54** secret-pattern
hits.

Classification into REPORTED / RESOLVED-IN-TRANSCRIPT / UNREPORTED was done by four parallel
sub-lanes over 31 agents each, against each agent's own final report; every HIGH and MED item
above was then re-verified directly against the repo, the remote, a live `gh` query, or a
re-run.

**Three limits you should hold this audit to:**

1. **Reasoning is unauditable.** All **10,404** `thinking` blocks in this corpus are persisted
   **empty** — signature only, zero characters. Whatever an agent considered and rejected, or
   decided and did not narrate, is not in these files. Everything here is inferred from
   actions, outputs, and narration.
2. **A regex over 180 MB of mixed prose and source is noisy in one direction.** Most `error`
   hits are the word appearing in code being read; most `token` hits are
   `archmage::X64V4Token`. The corpus-wide negatives ("no force push", "no relaxed test") are
   stronger than any per-item claim, because they rest on patterns that are hard to write
   accidentally.
3. **"The last assistant message" is not reliably the report — this audit made that mistake
   too.** The extractor takes each transcript's final assistant text as its report. A sub-lane
   caught two cases where a stale timer fired after the work was done and overwrote it with a
   stub (§8); those two were re-read from their real reports before judging, and both came back
   clean. Nine more transcripts end in a platform error string rather than any report at all
   (§4). So **11 of 124 rows in §11's "final report" column are not the agent's report**, and
   they are labelled as such. Any future pass over these transcripts should take the *longest
   late* assistant text, not the last one.
4. **Absence of a finding is not proof of absence.** A lane that did something harmful without
   narrating it, outside a Bash command, and without an error in the result, is invisible here.

**Reproduce:** `python3 ~/tmp/transcript_audit/extract2.py <transcript> <outdir>` then
`python3 ~/tmp/transcript_audit/digest.py <agentid>`. Off-repo evidence for §2 and §3:
`~/tmp/transcript_audit/EVIDENCE_priv_and_creds.txt`.

---

## 11. Per-agent table

One row per subagent transcript, 124 rows. `#err` is the raw count of tool_result error hits
the extractor flagged for that agent — it is a **noise-dominated** number and is here only to
show what the classification had to sift; a high `#err` is normally an agent iterating (a
failing-first test, a path probe) and correlates with thoroughness, not trouble. `#unrep` is
the count of items a lane judged genuinely unreported after reading that agent's own report.
`NONE` means the transcript and the report agree.

A `†` marks a row whose "final report" is **not a report**: a platform error string
(9 lanes, §4) or a stale-timer stub (2 lanes, §8).

| agent | final report — first line | #err | #unrep | worst |
|---|---|---:|---:|---|
| `a0094043856852bd0` | Honored the stop directive. Summary: | 16 | 0 | NONE |
| `a03c28c8c7b798d01` | CRUFT PURGE lane 2 complete — 6 commits, all verified on `origin/main` | 69 | 0 | NONE |
| `a045ff7f67cabb664` | Everything verified and cleaned up. | 28 | 1 | HIGH |
| `a0a022f6b9e1ee530` | I have everything. Report follows. | 17 | 0 | NONE |
| `a0b287db8d220b589` | I have everything. Report follows. | 13 | 0 | NONE |
| `a0b4c87fc1e66431d` | Done. Both boards serve 200; workspace cleaned up (1.7 GB target dirs  | 21 | 0 | NONE |
| `a0bcde99b97d9cc59` † | API Error: Sonnet 5 can't help with this. Start a new session to conti | 23 | 1 | MED |
| `a0d796245a7204b6d` | Parked. Stopping here. | 51 | 1 | MED |
| `a0f9ec89b061f8ff3` | Done. Commit `fc47b08e` is on `origin/main` (independently verified wi | 30 | 0 | NONE |
| `a0fa3b731520d7cfd` | Everything landed cleanly. Summary: | 9 | 0 | NONE |
| `a10748e48614b0c80` | I have everything. Here is the map. | 8 | 0 | NONE |
| `a10d55a6d086c215d` | Done. Summary: | 12 | 0 | NONE |
| `a117a4f1c306d1ce6` † | You've hit your session limit · resets 6:30pm (America/Denver) | 14 | 1 | MED |
| `a12f54f0062b36dd5` | Phase-2 endgame complete, foreground, single report. | 23 | 0 | NONE |
| `a13714e8bfaf42621` | All four commits verified on `origin/main`, workspace cleaned up. | 45 | 0 | NONE |
| `a14d5d954f76c0c0e` † | You've hit your session limit · resets 6:30pm (America/Denver) | 24 | 1 | MED |
| `a187a7d7f249d672c` | Parked. Everything is committed, pushed, and verified. | 21 | 0 | NONE |
| `a19699bc15c052b70` | Analysis complete, committed, pushed, and verified. Summary: | 16 | 0 | NONE |
| `a19e1c5f385dc7100` | Work complete. Summary: | 12 | 0 | NONE |
| `a1ad42ff05c2b5035` | Done. Workspace forgotten and removed; `jj workspace list` no longer s | 24 | 0 | NONE |
| `a1b96fc4c806f95d7` | Wave complete, everything landed and verified, nothing left running. | 24 | 0 | NONE |
| `a1cde0e0fff45644e` | CI hygiene lane complete. | 23 | 0 | NONE |
| `a1dc030cc162556c2` | Complete provenance recovered. Here are the findings. | 4 | 0 | NONE |
| `a1ee1f8a916af1aa2` | Workspace forgotten and removed; zenmetrics never needed a workspace ( | 35 | 1 | LOW |
| `a200b64551cbd4ba0` | Cleanup confirmed. Task complete. | 72 | 0 | NONE |
| `a203abe7ea2b560aa` | Done. All work is on `main@origin`, workspace cleaned. | 25 | 0 | NONE |
| `a23185b299ce0ca52` | Done. Commit `90926c32`, verified on `origin/main`; workspace forgotte | 28 | 0 | NONE |
| `a24d24fb9dccd80da` | Inventory complete. Here is the report. | 8 | 0 | NONE |
| `a254ddf678bdb6f18` | I've read all four documents in full, plus the scripts, slice files, f | 6 | 0 | NONE |
| `a30f9c0160712be5d` | All five commits verified on `main@origin`, working tree clean, marker | 17 | 0 | NONE |
| `a32edf4f951d270f8` | CI hygiene lane 3 complete. Summary: | 31 | 0 | NONE |
| `a349c45704d3fabaa` | Complete map below. | 6 | 0 | NONE |
| `a37b2b0c2960bbc7c` | Everything verified on `main@origin`. Lane complete. | 16 | 1 | HIGH |
| `a3c2d66abec65b913` † | The monitor stream has ended. Work is complete: both commits (`bac6cc8 | 45 | 0 | NONE |
| `a3cff65ab7881528e` | I have completed the review. Here are my findings. | 10 | 0 | NONE |
| `a49158a7ccd66ddf7` | I have everything. Inventory below. | 7 | 0 | NONE |
| `a4a23949595dc65c3` | I have everything, source-verified. Here is the settlement. | 4 | 0 | NONE |
| `a4d84730c6f1f91a7` | ## ⚠ Read this first: the operative gate has moved past the docs you a | 9 | 0 | NONE |
| `a4fdd63777f2beecc` | **Cause:** `train_corruption_head.py`'s shipped bake depended on ambie | 14 | 0 | NONE |
| `a5017704f0dc7c575` | Read-only recon complete. No repo writes, no compute started, no enrol | 13 | 0 | NONE |
| `a541ac3f35916f6eb` | Confirmed final state: the last 6 completed CI runs were all cancelled | 60 | 1 | LOW |
| `a55e94b273196c316` | Marker removed; both worktrees clean; no repo file edited. | 12 | 0 | NONE |
| `a583cc8db6cae0853` | # 1. `zensim/examples/foldapp_stream_bigpair.rs` — full report | 12 | 0 | NONE |
| `a587875bff795b88a` | Done. All three commits are verified on `origin/main`, and the sibling | 34 | 0 | NONE |
| `a5bc0d0d91582d568` | Workspace forgotten and removed; `featsys2` is gone from `jj workspace | 83 | 0 | NONE |
| `a5e15c86c852882da` | Done. Report: | 18 | 0 | NONE |
| `a5f23cd8ff5e1e599` | Everything is pushed, verified, and cleaned up. | 46 | 1 | HIGH |
| `a5fc5ae43b96361a2` | All work is landed, verified on `origin/main`, and the sibling workspa | 14 | 0 | NONE |
| `a60c870ffa107bcd3` | I've read all five assigned docs plus the two companions they depend o | 14 | 0 | NONE |
| `a611a74e20e4cb10b` | All five commits are docs/benchmarks only — no source, no weights, not | 14 | 0 | NONE |
| `a6121bde8b648cf60` | **F19 landed — `47f4630d`, verified an ancestor of `origin/main`.** | 25 | 0 | NONE |
| `a6516dc9da9ada8fd` | Complete. Findings below. | 11 | 0 | NONE |
| `a6825f73912577c31` | Verified: `2c4b8213` is on `master@origin`, and §7.11 item 1 now reads | 51 | 0 | NONE |
| `a68e8de677bc83a09` | All verified. Here is the final report. | 21 | 1 | HIGH |
| `a6a9112550ccdc0f5` | Done. Everything is on `origin/main` and verified. | 23 | 0 | NONE |
| `a7030a507f8f1bf1a` | All 9 items extracted. Report below. | 7 | 0 | NONE |
| `a70bc8ee2004de09b` † | You've hit your session limit · resets 6:30pm (America/Denver) | 6 | 1 | MED |
| `a773ad4c05bb515fe` | # Dense Factual Summary — Free/Class-C Feature Slots + D-id100 Chain | 3 | 0 | NONE |
| `a78ab4d0fc9fe51f7` | Investigation complete. Summary: | 28 | 1 | HIGH |
| `a79e524893ca6bbd9` | W4 is measured and published; cleanup confirmed. | 39 | 0 | NONE |
| `a7a20cc4a35da6ea7` | ## RECALCULATION MANIFEST — zensim Feature Revision 2 | 12 | 1 | MED |
| `a7afc19dc659f75d1` | Both defects closed, three commits verified on `main@origin` (`41ee67f | 16 | 0 | NONE |
| `a7b989a2a5aa811ef` † | You've hit your session limit · resets 6:30pm (America/Denver) | 5 | 1 | MED |
| `a899d2c2eb2be77c9` | Verified on `origin/main`: 4,222 B, sha256 `921a8f67…`, referenced by  | 32 | 0 | NONE |
| `a8b9a468115deca77` | Both subagents returned; everything corroborates my first-hand checks. | 10 | 0 | NONE |
| `a8db23ab5395d4465` | I've read all 19 documents (all exist). Here is the inventory. | 11 | 0 | NONE |
| `a8de537d194a7cdb2` | Audit complete. Nothing was edited — read-only throughout. | 14 | 0 | NONE |
| `a90603b3afc0f5755` | Done. Summary (full record: `benchmarks/select_gaddr_prefilter_2026-09 | 28 | 0 | NONE |
| `a907d66178e4d2731` | Done. Recovery lane finished; sibling workspace forgotten and removed. | 22 | 1 | LOW |
| `a916eeb451654048c` | **D-peaks lane — done. The peaks arm was built and gated; it is NOT in | 26 | 0 | NONE |
| `a97a641f66722e0c0` | I have the full picture. Writing the report. | 40 | 0 | NONE |
| `a9a4b5f9d37ca3e3c` | The refit lane's closure report — no action needed, and I ran the fina | 41 | 0 | NONE |
| `a9bfd3f448c467b44` | Both documents read in full, and I verified every join-critical detail | 6 | 0 | NONE |
| `a9d597b7417f82b20` | Everything is clean and verified. Final report: | 14 | 0 | NONE |
| `a9d9a35fa22c9f557` | Inventory complete. Everything read-only; no files written or edited. | 19 | 0 | NONE |
| `a9df2d4b27329f551` | Complete. All work is on `origin/main`, working copy clean, scratch wo | 25 | 1 | LOW |
| `a9e315ffe5e306d99` | I've read all six documents and verified the on-disk state. Here is th | 12 | 0 | NONE |
| `a9e6224814c2309ee` | Cleanup verified — `bestofall` is gone from `jj workspace list`, both  | 56 | 0 | NONE |
| `a9f0415f6abda7c4a` | Done. Everything verified on `main@origin` (`e9457b05` + `8a0c3af3`),  | 13 | 0 | NONE |
| `aa0306aa3f7ef2900` | Primary checkout is clean and untouched throughout, as required. Work  | 9 | 0 | NONE |
| `aa23453b438e88516` | Done. Commit `e1099726` is verified on `origin/main`; the `libcfix` wo | 27 | 0 | NONE |
| `aa2c4f5ec3d785ceb` † | You've hit your session limit · resets 6:30pm (America/Denver) | 9 | 1 | MED |
| `aa3a94c30ddafa366` | Done. Cleanup verified — both my workspaces (`defaults`, the throwaway | 26 | 1 | HIGH |
| `aa4d5177131e505ae` | PAUSED. Nothing was written to the repo, so there is nothing to commit | 5 | 1 | MED |
| `aa569c5116dad7fa1` | Done. `3cb7ec9d` verified on `main@origin`; primary `.workongoing` del | 17 | 1 | HIGH |
| `aa7310683119c107e` | The second stale waiter — the belt-and-braces one I armed during the h | 5 | 0 | NONE |
| `aa838df668a2f89cf` | I'll start by understanding the transcript corpus and building the ext | 2 | 1 | LOW |
| `aa9b95f87c8d36ceb` | I have read all six documents. Here is the lever ledger. | 13 | 0 | NONE |
| `aade58c31616e9ba2` | All work complete and verified on remote. Summary: | 28 | 2 | HIGH |
| `ab3bc42c491390630` | ## Report | 38 | 1 | HIGH |
| `ab726f047dcd6ee77` | All work is verified on `main@origin` and the sibling workspace is cle | 41 | 0 | NONE |
| `abc5ed6325a1e4366` | Inventory complete. No files written; workspace untouched (the `.worko | 26 | 0 | NONE |
| `ac299c067409b2557` | Both lanes are complete, verified, and all markers released. Here is t | 29 | 1 | LOW |
| `ac2bace4dc85586db` | Read-only research complete. Nothing written, nothing modified. | 5 | 0 | NONE |
| `ac9cf9a79dab32978` † | You've hit your session limit · resets 12:50am (America/Denver) | 7 | 1 | MED |
| `acd5393cea82b9450` | Cleanup confirmed. Final corrected report: | 20 | 0 | NONE |
| `ad104bc2647b4a01f` | I've read all five files completely and verified schemas against the o | 8 | 1 | MED |
| `ad33478f3f3420c3e` | Verified — the background scan's spot-checked claims are accurate, and | 16 | 0 | NONE |
| `ad362fe6bf1626d14` | All work is complete and independently verified. Summary: | 40 | 0 | NONE |
| `ad38c75615be13ef1` | Done. Everything is committed, pushed, verified on the remotes, and tr | 28 | 0 | NONE |
| `ad3a56e1264791a31` | I have a complete picture. Here is the factual inventory. | 7 | 0 | NONE |
| `ad3be5b360e81d5a6` | Done. All five tests ran at rev1 on the frozen incumbent split (parity | 7 | 0 | NONE |
| `ad4fc789f20c04945` | Done. The blocker was **not** a manifest conflict — it was a stale loc | 24 | 0 | NONE |
| `ad7797df291f486b6` | Both confirmed. Final delta — this **corrects one command I gave you** | 31 | 0 | NONE |
| `ad79ec429a13879c7` | Cleanup complete — workspace forgotten and removed, all markers cleare | 37 | 0 | NONE |
| `ad8f6643111fa811d` | Verified and closed. The pointer fix landed (zenmetrics `6e3971bc`, re | 31 | 1 | MED |
| `ada9c85fe5989436a` † | You've hit your session limit · resets 6:30pm (America/Denver) | 10 | 1 | MED |
| `ae148862db9524cb1` | Report written to `/mnt/v/output/zensim/im26anchor-2026-09-04/survey.j | 3 | 0 | NONE |
| `ae49a3402c57fc762` | ## Preface — the files you named are not in the checked-out working tr | 10 | 0 | NONE |
| `ae513115a7af1032a` | Lane complete. | 34 | 0 | NONE |
| `ae6e6a243b7b5931d` | Class-C lane complete. Everything is on `origin/main` (`a8b24c8e` emis | 22 | 0 | NONE |
| `ae9ddf662f304967e` | All clean — 17M of scratch artifacts on `/mnt/v`, nothing large enough | 14 | 0 | NONE |
| `aec54b97d30d9b589` | Cleanup confirmed — `kernel` gone from `jj workspace list`, directory  | 19 | 0 | NONE |
| `aece695321cbdf723` | Done. Workspace forgotten and removed. | 52 | 0 | NONE |
| `aed4a983555215cb2` | # Test Gate Inventory — `zensim/tests/` feature-computation gates | 15 | 0 | NONE |
| `aee46a54cb2199b3b` † | You've hit your session limit · resets 6:30pm (America/Denver) | 9 | 1 | MED |
| `aeee6cef43c4a6c3b` | Done. Both commits verified on `origin/main`, workspace cleaned up. | 7 | 1 | HIGH |
| `aef841812f9a27219` | ## RECON REPORT — zenfleet job system + LAN fleet current state | 22 | 0 | NONE |
| `af0d492b7400f2f75` | Cleanup confirmed — `dguard` is gone from `jj workspace list` and the  | 11 | 0 | NONE |
| `af1c2dc3dc5b347a6` † | Stale sleep timer — no action needed. R6 is complete and reported. | 32 | 0 | NONE |
| `af4e0005ce19bd2df` | Done. `invtruth` workspace forgotten and deleted; other lanes' markers | 23 | 0 | NONE |
| `af5988bf42126b6bb` | Lane 2 complete. 11 commits verified on `origin/main` (tip `39b26c73`) | 16 | 0 | NONE |
| `afa0039514b5ee354` | Done. Workspace cleaned, no `gaddr2` entry remains, and the other lane | 7 | 0 | NONE |
| `afd1855a27df4310a` | ## `FeatureRegime` definition | 13 | 0 | NONE |

---

## 12. Itemized UNREPORTED items

Thirty-one items, judged by four sub-lanes against each agent's own report and then
re-verified. Household and secret items carry **location only** — no value, no address, no
name — per the privacy rule; the substance of each is in §2 and §3. Everything a lane
classified as regex noise (the `token`-as-`X64V4Token` class, the "plex"-inside-"complex"
class, failing-first test cycles) is excluded rather than padded in.

| sev | agent:line | category | what it is |
|---|---|---|---|
| **HIGH** | `a045ff7f67cabb664:L150,L163,L167,L847,L850` | SECRET | R2/S3 secrets printed by an inverted `sed` redaction, plus a literal key inline in 4 commands (§2) |
| **HIGH** | `a37b2b0c2960bbc7c:L157-L230` | OUT-OF-POLICY TOOL | decoded with the banned third-party `image` crate; published figure ~50× too large (§5) |
| **HIGH** | `a5f23cd8ff5e1e599:L136` | HOUSEHOLD | child-framing committed to public `imazen/zenmetrics` (§3) |
| **HIGH** | `a68e8de677bc83a09:L556` | HOUSEHOLD | LAN address written into a new public zensim doc and pushed (§3) |
| **HIGH** | `a78ab4d0fc9fe51f7:L493` | HOUSEHOLD | LAN address committed into three public zensim docs and repeated in its own report (§3) |
| **HIGH** | `aa3a94c30ddafa366:L689` | HOUSEHOLD (LAN IP committed to public repo) | LAN address written into a new public zensim doc and pushed (§3) |
| **HIGH** | `aa569c5116dad7fa1:L482` | DESTRUCTIVE (jj sideways push, un-safe-guarded) | `jj bookmark set --allow-backwards` twice → both documented 2026-09-04 sideways pushes (§1) |
| **HIGH** | `aade58c31616e9ba2:L231,L239,L256` | HOUSEHOLD | new home-server-descriptor text written into two public `imazen/zenmetrics` files (§3) |
| **HIGH** | `ab3bc42c491390630:L43` | HOUSEHOLD | read the live public-repo household text and did not flag it (§3) |
| **HIGH** | `aeee6cef43c4a6c3b:L171` | HOUSEHOLD (LAN IP in own report + verify commands) | LAN address in its own report and verification commands (§3) |
| **MED** | `a0bcde99b97d9cc59:L907` | SESSION-DEATH with WIP | API-refused mid-edit in the **primary** checkout, unpushed; recovered later from its resume files (§4) |
| **MED** | `a0d796245a7204b6d:L531` | ERRORS (defect in another lane, not surfaced) | confirmed 2 pre-existing `imazen/zenavif` test failures, never surfaced; no issue exists (§8) |
| **MED** | `a117a4f1c306d1ce6:L(end of transcript)` | DIED-MID-FLIGHT | read-only inventory killed by session quota; deliverable never produced (§4) |
| **MED** | `a14d5d954f76c0c0e:L(end of transcript)` | DIED-MID-FLIGHT | read-only inventory killed by session quota; deliverable never produced (§4) |
| **MED** | `a70bc8ee2004de09b:L82` | Session death | read-only survey killed by session quota; deliverable never produced (§4) |
| **MED** | `a7a20cc4a35da6ea7:L129` | HOUSEHOLD (soft) | paired real household first names with the neutral node IDs in its own report text (nothing committed) |
| **MED** | `a7b989a2a5aa811ef:L219` | Session death | read-only survey killed by session quota; deliverable never produced (§4) |
| **MED** | `aa2c4f5ec3d785ceb:L177` | Session death | read-only inventory killed by session quota; deliverable never produced (§4) |
| **MED** | `aa4d5177131e505ae:L216` | DEVIATION | voluntary PAUSE with 0 of 4 deliverables and no external cause — the pattern the never-pause rule targets |
| **MED** | `aade58c31616e9ba2:L(same file)` | HOUSEHOLD | new home-server-descriptor text written into two public `imazen/zenmetrics` files (§3) |
| **MED** | `ac9cf9a79dab32978:L209` | Session death | pipeline-control lane killed by session quota; no synthesis delivered (§4) |
| **MED** | `ad104bc2647b4a01f:L49` | HOUSEHOLD | incidentally surfaced the public zenmetrics household leak and did not flag it (§3) |
| **MED** | `ad8f6643111fa811d:L1365` | DEVIATION (dropped caveat) | stale-build-dir reproducibility hazard flagged mid-session, dropped from the final report |
| **MED** | `ada9c85fe5989436a:L(final)` | DEVIATION | read-only inventory killed by session quota; deliverable never produced (§4) |
| **MED** | `aee46a54cb2199b3b:L(end of transcript)` | DIED-MID-FLIGHT | read-only inventory killed by session quota; deliverable never produced (§4) |
| **LOW** | `a1ee1f8a916af1aa2:L185` | SECRET (noise) | regex noise — `archmage::X64V4Token` matched the secret pattern |
| **LOW** | `a541ac3f35916f6eb:L970` | HOUSEHOLD (noise) | regex noise — "plex" inside `clippy::type_complexity` |
| **LOW** | `a907d66178e4d2731:L278` | tool-friction (gh-guard false positive) | gh-guard false positive on a local `CLAUDE.md` edit containing `main@origin` (§9) |
| **LOW** | `a9df2d4b27329f551:L838` | DESTRUCTIVE (jj op restore) | `jj op restore` failed on a corrupt op object and changed nothing; real fix was a normal rebase |
| **LOW** | `aa838df668a2f89cf:L53` | HOUSEHOLD (noise) | regex noise — this audit lane; the "hit" is the detector's own pattern text |
| **LOW** | `ac299c067409b2557:L(throughout)` | HOUSEHOLD (fleet ops) | LAN addresses in ~80 fleet SSH commands; never written to any file |
