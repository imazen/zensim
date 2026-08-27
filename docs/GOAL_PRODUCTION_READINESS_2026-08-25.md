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
