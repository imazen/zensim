# SOTA-944 CAMPAIGN (user directive 2026-08-01)

Directive, verbatim: "complete all backfills, fix all logged issues and read gh
issues and md files - don't pause until you achieve all goals and a new sota
model." This doc is the durable program spec — supervisors and agents execute
FROM HERE. Mandatory pre-reads for every phase: `docs/TOP_MODELS_COOKBOOK.md`,
`benchmarks/profile_b_methodology_2026-07-12.md`, CLAUDE.md ("★ 924 PARQUETS",
"★ E-M CAMPAIGN", RECURRING PRIORITIES), the zenpapers freeze plan §9b,
`benchmarks/linear924_phase1_2026-08-01.md` (incl. CORRECTION block).

## P1 — 944 backfill across ALL corpora (task #9; the validity gate)

Re-extract at `FeatureRegime::Folded720Append2` (944; append2 ON — its constants
are PROVISIONAL per `benchmarks/append2_bandvis_gates_2026-07-27.md`; record the
extractor commit in every manifest so a later constants change is detectable):
1. The 11 ext legs at `/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/`
   → sibling `ext944-canonical-<date>/` (same rows, same target columns CARRIED
   — incl. the cvvdp/iwssim/mix columns where present; never claim absent
   without checking the fill4metrics/mm6 sidecars per RECURRING).
2. `kadis-924-2026-07-27/` pair (700k + negrich) → 944.
3. bigcodec 21 split views (keyed encode_sha) → 944.
4. Both eval grids (dial 4817, corruption 2016) → 944col.
**Gate G-BF1 (per file): f0..f923 BITWISE-IDENTICAL to the 924 parquet for every
row** (byte-stability is proven at both tips — any mismatch = STOP, diagnose).
G-BF2: row counts + key columns identical; zstd parquet (never snappy).
G-BF3: triple-mirror (local + s3://zentrain + Tower) + `_MANIFEST.json` with
build_commit + shas + DATA_PROVENANCE.md pointer. Compute: local first (legs +
grids + kadis are local-feasible); bigcodec 5.7M via zenfleet/household nodes if
local ETA > ~6h (NEVER hand-rolled fleets).

## P2 — logged-issue wave (after P1's heavy extraction; one heavy stream)

From `gh issue list` 2026-08-01: FIX in code: **#48** (max_pixels safe default +
cooperative cancellation via `enough`), **#50** (top-cliff 100→95.7: diagnose —
dial-top spline/anchor territory; fix or document mechanism with measurement),
**#17** (gamut clamp masking), **#14** (zensim-regress ad-hoc diff CLI).
PREP-TO-GATE: **#46** (version the zenpredict dep; the PUBLISH chain
zenpredict-v3 → zensim 0.3.0 stays USER-GATED — never publish without explicit
go + README verification). CLOSE-WITH-EVIDENCE (comment + close): #26 (KonJND
anchor landed), #22 (superseded by freeze plan), others only if genuinely done.
LEAVE: #54 (post-freeze by design), #25, #33, #38 (HDR program), #41, #6 unless
trivially closable. Every fix: test + CHANGELOG + push + verify.

## P3 — the SOTA-944 model campaign (tasks #10 + SOTA; blocked on P1)

Arms, all fit/evaluated on the UNIFIED 944 regime:
1. **Additive class (user: THE key)** — ADD156 lineage extended to the
   spatializable 944 subset (basic-156 + mean-pooled append slots); monotone
   per-feature transforms; solver = linear-projection/Rust gram path (the ONLY
   additive emitter; zensim_mlp_train has no linear mode). Steering: abs fold.
2. **B-recipe replay at 944** — BVLS multi-head + ssim2-derived per-corpus
   anchor targets + cid22_train(ssim2-anchored) + kadid + tid (+hdr_v3mix
   equivalent when HDR-944 exists), inclusive-winsor, extend-top dial.
3. **E-M MLP recipe at 944** — fold-class + WT40 + DET/ART mask + kw0.15 +
   `--coarse-decay 1e-5` + QAT opt-in, k seeds → sdr25/best_val selection.
4. bigcodec discipline: w≤0.5 for MLP (saturated at 1.0), EXCLUDED for
   linear/additive (pitfall list) — cite, don't re-measure.
**SOTA bar (frozen):** on the unified 944 eval — CID22 > 0.8924 (EM4) with
KonJND ≥ 0.43, nonphoto ≥ 0.90-class, HF-NL not below B's 0.614, dial mono
≥93%/tied ≤5%, M3a ≥ 0.85 (attribution map), G-RANGE clean, embedded repro —
i.e. beat the best of EVERY era on its own strength, no axis sacrificed
silently. Report the full scorecard vs winner_dial/Ebothg/EM4/B (era-tagged).
Then: gauntlet + 2/3-shot loop panel re-run incl. the additive candidate
(best-measured steering never entered a loop study), freeze_check.

## P4 — records

Dashboard regen + artifact republish; CLAUDE.md/cookbook updates; memory;
honest losses listed; ship swaps stay user-gated (as do all publishes).

## Standing rules for every agent in this campaign

jj + `.workongoing` append-protocol; own workspaces only; push+verify
(merge-base) every commit; run-heavy caps; one heavy job at a time; ~/tmp only;
pre-register before measuring; never relax gates; honest nulls stand; the
supervisor independently re-derives headline numbers before accepting.
