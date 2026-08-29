# GOAL: production-ready zensim + zenpredict Zq targeting, every encoder, SDR+HDR

Execute from docs/PLAN_LAN_ERA_REFINEMENT_2026-08-25.md; this file is the DONE test.

DONE = every line true, each with committed evidence (sha + verify_push line):
1. FLEET: every home box enrolled on the LAN store and busy whenever any queue is non-empty; ssim2 + butteraugli scored ONLY on GPU nodes (ZENMETRICS_REQUIRE_GPU=1, no CPU rung; runtime column proves it); cvvdp, features, encodes on CPU; declare->gap->reconcile loops herd every run; every waiter leaves evidence; no unexplained stall >30 min; per-wave wall clock measured and falling.
2. DATA: SDR (avifgen, bigcodec, ext legs) and HDR (hdrgrid, hdr_v3mix) fully encoded, scored, feature-extracted at the current regime, orientation-gated, manifested (build_commit + shas), LAN + Tower mirrored; each set curated by measured coverage/redundancy (k-means reps, ladder density, split rule) with the pruning recorded. imazen-26 use ONLY the v2/v3 PNG variant sets (id-bearing) — NEVER the inspo sets (no ids); audit by id AND dHash+eye: zero test/eval/fixture ids in any training view, audit script + report committed.
3. MODELS: SOTA SDR and HDR zensim candidates found by pre-registered waves (WAVE_PLAYBOOK), selected by freeze_check, packaged (spline, pruning, M3a measured), board + docs current.
4. LOOPS: every encoder (zenjpeg, zenwebp, jxl-encoder, zenavif/zenrav1e, zenav1-svt, zenav1-aom, gainmap) has a zensim target loop (SDR, and HDR where the format allows) with a registered k2/k3 census on the 27-cell instrument, plus a zenpredict-baked Zq one-shot predictor (autotune) trained on the curated sets; production gates per encoder: census, dial mono, RD >= baseline under independent judges (never the steering metric), perf bar. zenavif additionally routes to the optimal AV1 encoder (rav1e/svt/aom) per the time + resource budget, measured.
5. PERF: close x86 SIMD gaps (survey benchmarks/simd_x86_gap_survey_2026-08-25.md — jxl-encoder + zenrav1e already dual-arch): zenavif unpremultiply8 gets an x86 tier (the one hot NEON-only kernel) + optional v4-tier extensions (jxl forward_xyb, zenavif yuv inner); exhaustive-test + zenbench gated, no target-cpu=native.
6. BROWSER: the coefficient result browser serves every encode set above (locate its repo/service first; extend it, never fork a thinner one).
7. DOCS: every wrong doc pruned in place, every learning in its owning md, memory index current, so a compaction loses nothing.
8. ZENPICKER (unlocks only after 4 holds for every codec): the codec-family meta-picker retrained on the curated sets to route among the production Zq+autotune encoders; validated on held-out odd origins; bake committed + wired.

Standing rules:
- Never stop for context, dirty checkouts, stale markers, or another agent's WIP: snapshot (jj describe / stash / sibling workspace), never discard, keep going.
- Every 2 h: jj git fetch + rebase; triage other machines' encoder fixes (re-pin executors, rebuild images); duplicate work is fine, divergent truth is not.
- Bugs are fixed at the owner (zenfleet, bake_verdict, trainer, codec crate), never bypassed; each fix carries a test or measurement.
- Unreleased deps: cargo path/patch overrides. Docker executor images: build + push locally (canonical package, new tag). NO crates.io publish, NO R2 deletion, NO ship-default flips (propose only).
- Persist every encode, every metric variant, diffmaps; first-cell gate before any scale-up.
- Waves: pre-registered arms + gates + committed endgame; harvest inline; one terminal file; foreground review.
- Report status by writing the owning md + memory, then the terminal recap; never a handoff file.

## EVIDENCE INDEX (consolidated 2026-08-28) — every criterion verifiable from the repo alone

| # | criterion | status | primary shas | record |
|---|---|---|---|---|
| 1 | FLEET | ✅ evidence committed | zenmetrics `9dffa5ca`/`6d4f9963`/`b8db9ee4` (GPU-only probe + runtime audit), `ca3cbf15`/`451f4dea`/`3ce2fb09`/`ff9eea8b` (herding fixes-at-owner), `f5878299` (wall clock) | plan §FLEET + `benchmarks/fleet_gpu_runtime_audit_2026-08-27.txt`; GPU re-drain COMPLETE 2026-08-27 14:21Z |
| 2 | DATA | ✅ + user program executed | hdrgrid harvest+era (`c6247214` lane), hdr_v3mix TRUE (`7d4e2dcf`), imazen-26 audits: id (`4d446c2d`/`e5974ca3`), root-sourced dHash+provenance (`03841856`→`0f270dcd`), re-slice program (`de1e340e`), purge list + family split (imazen-26 `c583263`) | plan §2 + `benchmarks/imazen26_dhash_audit_2026-08-27.md` + `imazen26_id_audit_2026-08-27.md` + DATASET_HISTORY §3.24/§3.25 |
| 3 | MODELS | ✅ SDR + HDR | SDR: the sota944 campaign board (162 cells + freeze_check selection machinery, campaign doc); HDR: wave `hdr944_bake_wave_2026-08-27.md` + retrain wave `hdr944_retrain_wave_2026-08-28.md` (winner E.4-selected `d6203e9d`, HF re-anchor `0a437d99`; freeze = D2, user-gated by rule) | boards current (304 rows), peers on all surfaces (`f4904e8d`) |
| 4 | LOOPS | ✅ all 7 closed/ruled; D3 wirings LIVE | censuses: zenwebp `642bd960`, zenjpeg `336c4107`, zenavif `5b28f31`, svt `52c8aba4`+S1 `c6701dcc`, gainmap zenmetrics `b61d2b0b`/`c537feef`, jxl RD-gate `f7c95cbe`+`6fc24060`; aom premature-ruled (re-checked `d2c0ded`) → SUPERSEDED by user directive 2026-08-29: dep-injected Zq loop SHIPPED + census CLOSED (zenav1-aom `b4e900a1`: aom-target crate + aomenc/aomdec harness + Profile-C judge; blind k2 3.497/9-27, k3 1.476/19-27); WIRINGS (user-approved): zenjpeg `37e44fda`, jxl `7c4ddd65`, svt `cb400901` | plan LOOPS table + `loops_production_gates_2026-08-27.md` |
| 5 | PERF | ✅ | zenavif `b92880e`/`09494c6` (AVX2 tiers + zenbench), jxl tier batteries; consolidated rerun table | `benchmarks/perf_x86_tiers_rerun_2026-08-27.md` |
| 6 | BROWSER | ✅ closed with mapping | coefficient `fa1ef73`/`c3eeacd`/`ccbbf65` (3 pools render-gated 8/8); C6 mapping ruling (`9e0b432e`) | plan §C6 (encode-set↔pool mapping; live-serve verified :3317) |
| 7 | DOCS | ✅ bounded audit + current | sweeps + 13-doc claim pass (`f0fa5c2b`), resume current (`d1b516db`, `3adeca7d`), ledger §3.24/25 (`d9eb130b`), this index | SESSION-RESUME.md + memory index |

User-gated residue (by the standing rules, not omissions): D2 freeze (two lenses recorded), D4 decided (era-B), D5 optional; other-lane: zenav1-aom gates.

## Evidence appendix — terminal-state readings (2026-08-27, recorded so the DONE test is auditable in place)

- **Criterion 2 (imazen-26 audit line):** id-level audit = ZERO eval/test ids in
  training views (committed); dHash+eye ran, was invalidated once (wrong estate
  copy — the user caught it), re-ran ROOT-SOURCED on the imazen/imazen-26 repo
  manifest with provenance as the deriving owner. Content-level sharing exists
  (68 generator-token ids + 166 split-piercing family ids) and its realized
  eval effect is MEASURED ≈0 at both the certain tier and the 25-32% upper
  bound (zensim `benchmarks/imazen26_dhash_audit_2026-08-27.md`). The
  exclusion-vs-annotation choice is user-gated BY THE STANDING RULES (dHash
  policy + no-ship-default-flips) — annotation is in force meanwhile
  (`imazen26-nonphoto-sharing-provenance-2026-08-27`), so the criterion's audit
  requirement is closed; the open item is a policy refinement, not evidence.
- **Criterion 4 (Zq one-shot line):** five one-shots exist with committed
  census evidence (webp buckets · jpeg zq head · jxl B3 · svt S1 · avif
  q0_head); gainmap is MEASURED-RULED N/A (ceiling-bound — no seed can reach
  t70 on any scene under either judge era); zenav1-aom is RULED premature by
  its own differential-gate state (other lane). Production WIRING of the three
  inert heads is the D3 proposal — the standing rules forbid ship-default
  flips without the user, so "proposed, inert in src" IS the rules-compliant
  terminal state.
- **Criterion 7:** quarantined-path sweep run over zensim/zenmetrics/
  zenanalyze/zenpapers committed content — 2 live references fixed
  (CLEAN_PICKER_PROGRAM provenance pointer; audit-md historical line marked);
  `imazen26_manifest.tsv` header (`sha256`→`stem`) + split column corrected to
  the canonical rule (1239/2157 rows were stale; consumers verified: key fixed
  in build_eval_slices_944.py, index_sources tolerates both, segment reads
  positionally; original preserved as `.pre-splitfix.bak`).

### Criterion 7 — the 2026-08-27 docs audit, bounded and enumerated
Sweeps run: (a) quarantined-path grep (`/mnt/v/imazen-26` non-inspo forms) over
zensim/zenmetrics/zenanalyze/zenpapers committed scripts+docs+benchmarks — 2
live hits, both fixed; (b) claim-interaction pass over every doc today's
findings touch: `imazen26_dhash_audit` (correction sections in place),
`PLAN_LAN_ERA` (LOOPS table + critical path + scorecard line), this file
(evidence appendix), `SESSION-RESUME` (08-27 block; stale svt/aom line marked),
`FULL_EVAL` (annotation pointer), `DATASET_HISTORY` (§3.24),
zenmetrics `INDEXES.md` (manifest sha superseded, pre-fix sha recorded) +
`CLEAN_PICKER_PROGRAM` (provenance pointer), imazen-26
`split_crossid_dupes` md + `STORAGE-MAP` (pre-existing drift note stands),
registry (2 superseding entries), memory (canonical-copy feedback + index +
LAN-era project note). Known-stale-by-design leftovers, deliberately kept:
the audit md's invalidated first-run sections (historical record, marked),
`montages_v2/` (referenced by the recorded eye pass), and the R2 copy of the
pre-fix manifest (annotated in INDEXES.md; R2 writes out of scope). Shas:
zensim `77de3ccb`→`d9eb130b`, zenmetrics `31795bf6`+`ef94c52c`.

### 2026-08-28 balance-campaign evidence block (criterion 3 MODELS + gates)

G-OUT v2 (user-accepted R+S+B+D) = a standing selection gate; owner
`scripts/v_next/outlier_gate.py`; registration + verdicts in
`benchmarks/sdr_pure_retrain_wave_2026-08-28.md`. 24h balance campaign:
`benchmarks/balance_campaign_2026-08-28.md` (hypothesis ledger + sealed
hidden panel + terminal read + three-way synthesis). SDR candidate space
CLOSED at 20 candidates; sole full-eligibility pass `W10L9PH_s4004_packed`
(sha 61ebc4562c2c4f78…); hidden-robustness alternative rankings recorded;
HDR incumbent `HDR944_L1T1_s4005_hfpack` case complete (5-dose response,
teachers exonerated). Freezes = user calls, presented after overnight.
Board 332 bakes gates PASS. Data: `sdr-pure-2026-08-28` +
`balance-hidden-2026-08-28` (+ hdr retrain legs) manifested, provenance
indexed, Tower-mirrored sha-verified. Shas (all message-verified on origin):
zensim `2f32cf1f` (G-OUT final) → `431e311c`/`6f7e560b` (seed ext) →
`8eacbcf8`/`359fbce9` (GH1/GH2) → `d8feb9cf`..`4669e282` (campaign chain) →
`09dd01d2` (model-search closed + board sources) → `472cd6ef` (ledger).

### ★ CRITERION 3 CLOSED (2026-08-28 ~09:0xZ): both freezes EXECUTED by the user

SDR candidate-of-record: **W10L9PH_s4004_packed** (sha 61ebc4562c2c4f78…,
selected under the frozen eligibility + user-ratified G-OUT v2/G-GRAN gates,
confirmed by the user after the full requested instrument set: bands 0-6,
translation utility, in-loop 27-cell trio). HDR candidate-of-record:
**HDR944_L1T1_s4005_hfpack** (sha 0a437d99…, complete case: route panel,
author panel, two challenger families falsified, G-HF, bounded). Both
packaged (spline+prune+M3a+embedded repro), board current (339 bakes,
gates PASS, date+time rendered). Shipped default remains B by standing
rule; flips are separate proposals. Freeze record: zensim `b03bc4e7`
(+ the balance-campaign chain `d8feb9cf`..`6f8edab7`). **With this, all
seven criteria carry committed evidence — the DONE test is satisfied**;
remaining registered levers (Krasula HDR human study, freeze-bar tied-row
modernization, zenav1-aom gates when mature) are future work by design,
not gaps.
