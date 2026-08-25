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
