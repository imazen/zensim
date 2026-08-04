# zensim

Workspace with three crates: `zensim` (library), `zensim-regress` (regression testing binary), `zensim-validate` (validation binary).

**Feature-gap map (read before feature work):**
`~/work/zen/zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` — the
2026-07-26 audit of the 720-feature set (v1-vs-v2 iw/masked naming trap,
ranked weaknesses, fast-CPU candidates with evidence, regime-inversion
finding, don't-build list). The f720+ append block
(`benchmarks/v2_append_block_2026-07-26.md`, `FeatureRegime::Folded720Append`)
implements its A1-A5/A9 candidates.

## Known Bugs

_(none open)_

### Resolved

- **Validate-side output-spline upper extrapolation diverged from the product
  runtime — RESOLVED 2026-07-04 (`5d4978db`).** `output_calibration_spline::
  apply` extrapolated linearly UNCAPPED above the top knot while
  `zensim/src/metric.rs` caps at ≤100; the file's "bit-exact" claim was false
  above the top knot and produced dial-p95 artifacts of 300-500 on linear
  bakes. Now capped for parity (bottom stays uncapped — neg-tail corruption
  resolution). `parse_round_trip_minimal` had enshrined the divergent value
  (110); expectation corrected to 100 per the product contract.

- **konjnd-agg 2-layer w1 gradient "bug" — RESOLVED 2026-05-27 as a
  malformed test, NOT a gradient error.** The
  `konjnd_aggregation_2layer_w1_gradient_matches_finite_difference` test
  reported ~2–3% (and at one point 48%) relative error vs the analytical
  gradient. The gradients are **correct**; two compounding test defects
  produced the spurious failure: (1) the forward computes in **f32**
  (`dot_bias` casts f64→f32), so a central difference is floor-limited —
  at ε=1e-6 the rounding noise in `(f₊−f₋)` (~1e-7) swamps the signal
  (~2·ε·grad), giving O(1) relative error; (2) a **pure relative gate**
  is unbounded as the true gradient → 0 (near-zero entries like
  `gw1[4]≈-9e-4` have abs diff ~6e-6 = a correct gradient). Fixed by
  ε=1e-2 + the standard `|num−ana| < atol + rtol·max(|·|)` gradcheck
  criterion. The earlier "α≈1 drops error 48%→10.5%" observation was
  itself an artifact of the f32 floor manifesting differently across α
  regimes — debunked by a new train-core test
  (`per_sample_alpha_head::tests::backprop_heads_dl_dh_matches_finite_difference`)
  that FD-checks the head/encoder gradient directly (L=y, dl_dy=1) and
  passes cleanly. Shipped bakes were never affected. Commit: see
  `fix(#35)`.

## Canonical training data + indexes (added 2026-05-20)

**The canonical index for all ML data lives at `~/work/zen/DATA_PROVENANCE.md`.**

Quick paths:
- Trainer input: `/mnt/v/zen/zensim-training/canonical-2026-05-21/` (local) + `s3://zentrain/canonical-2026-05-21/` (R2) + `/mnt/tower/output/zensim-archive-2026-05-20/` (Tower)
- Per-row truth: `_MANIFEST.json` in each canonical/picker training dir
- Master inventory: `~/work/zen/_ml-inventory-2026-05-20/00-MASTER-SYNTHESIS.md` (7-part forensic inventory of repos + parquets + datasets, 2026-05-20)
- Worktree audit: `~/work/zen/_ml-inventory-2026-05-20/01-zensim.md`

The 2026-05-20 byte-equivalence audit (`10-canonical-build-audit.md`) confirmed current zensim main produces features bit-equivalent to all 13 canonical-2026-05-21 parquets (sub-ULP precision). No build drift; trustworthy as-is. The `cvvdp_iwssim_LARGE_372col.parquet` (73,300 rows, 85.5 MB, sha256: 14c205332701b5ff6f2842a8d60f8ac1282f8be3d5cd89c11700e1e4b864a20f) lives at `canonical-2026-05-21/features/` — extracted 2026-05-20 to fill the f300..f371 IW-pool gap.

## ★ THE 924-FEATURE PARQUETS (folded+append STREAMING regime) — the current-era datasets (2026-07-27/28)

**Every canonical dataset now exists at 924 features** (regime `Folded720Append`, zensim
`0b3d16b0` C5 streaming-only, `ZENSIM_AB_MODE=foldapp`, `codec_target` profile, RAW unpadded
slices). Layout: f0..f155 folded v1-basic, f156..f371 STRUCTURAL ZEROS, f372..f719 v2-348,
f720..f923 append-204. **REGIME PURITY: never column-mix 924 rows with 720/v1 parquets**
(padded-width divergence + zeroed pools). All triple-mirrored (local + R2 + Tower, sha-manifested);
full provenance in `~/work/zen/DATA_PROVENANCE.md`. Train on THESE for all new work:

| dataset | rows | local path |
|---|---|---|
| 11 local legs (cid22val/aic3/aic4/csiq/live/kadid/tid/safesyn/cid22t201/konjnd/sdr25) | 149,195 | `/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/` |
| bigcodec fleet table `tbig_924_full.parquet` (keyed encode_sha) | 5,742,660 | `/mnt/v/output/zensim/tbig-924-2026-07-27/` |
| bigcodec 21 split views (7 picker datasets × train/validate/test, match_rate 1.0000) | 5,742,660 | `/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/bigcodec/<dataset>/<split>_924.parquet` |
| `kadis700k_924.parquet` (7 byte-carried metric targets, split on source_id) | 699,999 | `/mnt/v/zen/zensim-training/kadis-924-2026-07-27/` |
| `kadis_negrich_924.parquet` (severe, score_zensim_gpu<0 — corruption-head negatives) | 167,034 | same dir |
| eval instruments: `corruption_grid_924col` + `dial_grid_924col` | 2,016 + 4,817 | `/mnt/v/output/zensim/v2-eval-924-2026-07-27/` |

Eval slices: FULL_EVAL's 924-era imazen26/nonphoto point at the **canonical bigcodec 924 TEST
views** (see `docs/FULL_EVAL.md` "924-era eval slices"); the NN-matched `ext_*_720` tables are
720-legacy, never rebuilt. First instruments on this data: P12 residual-boost ranking + P11
decorrelated-auto (both arms + the empirical S-class map) — `benchmarks/p12_*`/`p11_*_2026-07-27*`.

**Coming later: additional HDR features.** The 924 set is SDR-only by design; a future wave
appends HDR-specific features (the append-only discipline holds — new slots will EXTEND, never
renumber, and HDR rows will be their own regime/datasets, never column-mixed into these).

## ★ THE E-M CAMPAIGN (924/v3 era) — findings + recipes + the steering pivot (2026-07-28/29)

Full evidence: `benchmarks/coherent089_seeded_frontier_2026-07-27.md` (E-M1..E-M9).
Commits `555b1a48`..`aa5576f4`. Bakes: `/mnt/v/output/zensim/bakes/coherent-089/{,em2/}`.

### Rank results (924 regime = coherent by construction)
- **fold924 kw0.5, n=6: CID22 0.8825±0.0025**; **no-KADIS n=6: 0.8816±0.0016 + KonJND 0.398**
  (v3 features carry near-threshold natively; KADIS ROLE-REVERSED at 924 — suppresses KonJND,
  rescues CSIQ; crash zone kw(0.15,0.5]).
- **ATTRIBUTION (E-M6b, width discriminator): most of the CID22 lift is the DATA** — the
  924-era bigcodec slice lifts 720-width to 0.8837; v3-marginal ≈ +0.001. tbig_924_200k is
  the one non-row-identical slice (no join key) and drives the coarse-scale shift too.
- **Seed selection is VALIDATED**: sdr25 (bake_verdict corpus, never trained, not a gate)
  predicts CID22 at SROCC +0.752 over 35 bakes, rejects every collapsed seed. Best selected:
  `EM4_mask2_kw0.15_s42` CID22 0.8924 + KonJND 0.4286 (`coherent924_selected` on the gauntlet).
  **⇒ SUPERSEDED 2026-08-04 as the ship rule** (still true as a fact about sdr25): selection
  is now **rank + dial + COHERENCE**, run by the OWNER —
  **`freeze_check --select <every fulleval>`**. PRIMARY = profile floor count; TIE-BREAK =
  `balanced_composite + 0.15·M3a`; `sdr25` is a reported comparator only, **never the
  primary** (it has decoupled from CID22 five times). Rationale: the coherence study measured
  **42.3% of 944-class M3a variance is seed noise at fixed recipe** (`C_co3a` k=6 spans
  0.718-0.826), so M3a is a *selectable trajectory property* and a k-seed wave that ignores it
  leaves ~0.1 M3a on the table. Three M3a states, **none of them zero**: MEASURED ranks;
  NOT COMPUTABLE (ensemble — the instrument loads one ZNPR) ranks separately and is never
  penalized; UNMEASURED is listed but **not selectable**. M3a comes free with
  `harvest_bakes.sh` (27 cells, **66 s/bake** measured); a missing one drops a `.NO_M3A`
  marker. Registered: campaign appendix E.4; workflow: `docs/WAVE_PLAYBOOK.md` step 6.
- Corruption ordering breaks at 924 in ALL arms (0.03-0.17 vs 0.214@720); occlusion blamed
  DET/ART_DEV2 but **masked RETRAIN does not recover (occlusion≠ablation)** — distributional;
  mitigate with the corruption HEAD (negrich_924), not the dial.

### The coherence resolution (THE steering pivot — build task #67)
- Mechanism: 924-era data → ~89% basic gradient mass on COARSE-scale MSE → the
  mass-blended per-pixel fold degrades to 1/8-res → M3 0.10-0.25 while **M2 ≈ 0.99**.
- Weight-decay remedies FAIL: coupled L2 is neutralized by Adam (AdamW insight); decoupled
  `--coarse-decay` bites but hits data-preference equilibrium (s3 ~50% across 100× rates).
  **Keeper: `--coarse-decay 1e-5` = KonJND +0.15, CSIQ +0.07, ~free.** Scale-mass = symptom.
- **PROVEN architecture (E-M9): per-block gradient attribution M2 = 0.999-1.000 at EVERY
  block size 16-128px** on the pathological bakes, while the signal-fold M3 inverts at 128px.
  **Ship design: per-pixel attribution DENSITY (true per-feature integrands; mean-pooled
  exact — covers the dominant MSE slot; p2/p4 value-weighted) + summed-area table → O(1)
  arbitrary-rectangle queries** for variable codec partitions (AV1 4-128px, JXL var-DCT,
  HEVC CTU). The old signal fold becomes visualization-only. Perf bar: build ≤ current
  diffmap build (same planes, different weighting, +O(N) SAT). jxl's loop is
  global-iterative (iters+1 full-frame compares) → full-map-per-iter is its contract;
  region-incremental re-query is for per-block-probe codecs (zenjpeg) — SAT re-query is
  already O(1), local re-COMPUTE bounded by blur footprint is the later optimization.
  **Order (user): get it coherent, THEN optimize.**
- **C1 SHIPPED (2026-07-29): `Zensim::compute_attribution_density` + `AttributionResult`
  (SAT, `query_rect` O(1)) with exact basic integrands** (p2/p4 removal-consistent 1/p;
  hf = signed clamped-ratio first-order — slots 10-12 are ratio-pooled, NOT means).
  M3a > M3 in 8/8 gate cells; healthy-720@128 = 0.895; but the ≥0.85 gate is NOT met:
  ATTRDIAG proves |s|-mass ≠ rank-variance — EM2's 98.4%-mass basic block true-ranks ΔS
  at only 0.33-0.54 (append-blind ceiling 0.43-0.68, NEGATIVE at 128 where the 0.5%-mass
  append block alone carries the signal). Distance = append fold + exact non-additive v2
  integrands, both measured. Raw v2map ADD into the score-unit density is unit-broken
  (swamps it) — use weights ×1/(w·h). Perf 2.4-3.4× (C2 target ≤1.1×).
  `benchmarks/attribution_map_c1_2026-07-29.md`.
- **C2a SHIPPED (2026-07-29): `compute_attribution_density_full` — exact integrands for
  ALL v2+append slots** (production-kernel pass A, 1e-9 feature parity; FD direction tests
  caught an edge-width sign bug pre-landing). Gate 5/8: **EM2 4/4 ≥0.85 — the 128px
  inversion is CURED (−0.36 fold → +0.99, at the M2 ceiling; the 0.5%-mass append block
  was the whole coarse signal)**; K720 1/4 (0.69/0.53/0.38/0.90) — miss isolated to v2
  density approximation at fine blocks (v2attr-vs-true-lin 0.61/0.40/0.18/0.89; basic holds
  0.89-0.98) → C2b lever = blur-bleed spreading + perf (full 125-138ms vs fold 11ms).
- **C2b SHIPPED (2026-07-29): bleed-spread hypothesis MEASURED-FALSIFIED + perf −30%.**
  The `I−K` adjoint is structurally wrong for residual signals (zero net mass); the 50/50
  split REGRESSED all 8 cells; window-only spread (shipped) is neutral — K720's fine-block
  gap is the finite-removal floor, not allocation. Gate unchanged 5/8 (EM2 4/4 holds).
  Perf: single-sweep + channel/row-band rayon → full 95-98ms, basic ~38ms vs fold 11-12ms
  (8.3×/3.3×; ≤1.1× structurally needs fusion into a shared 924 compare — levers + estimates
  in the C2b doc section). `blur::box_spread_sum_preserving` = the exact-sum spread primitive.
- **C3a SHIPPED (2026-07-29): the FUSED compare** — `Zensim::compute_with_ref_score_and_
  attribution` = score + steering map from ONE pipeline (v1/372 class; score BIT-identical
  to the fold path, gated; standalone paths untouched, 8-cell identical). 576²: score+map
  **14.8ms total** (the old standalone map alone was 36.8). Marginal-map bar (≤1.1× fold's
  marginal) missed at floor 5.7×@576/2.2×@1152 — the fold folds in-kernel with no pooled
  scalars; next levers ranked in the C3a doc (ref-side hf cache → in-kernel mean slots →
  stale-scalar single-pass). 924 fusion needs extractor-side hooks (that session's domain).
- **C3b (jxl loop A/B, `f195c8c0` in jxl-encoder): attr steering did NOT beat the fold at
  target-hitting** — med |achieved−target| v47A 0.807 vs fold 0.594 (and BASELINE 0.244
  beats both model maps there); shippedB 1.507 vs 0.982; 1W/7-8L per-cell, equal median
  iterations. Probe shows the tile signal is STRONG (full ratio range at 8px) and the fused
  scalar tracks decode BETTER than the fold arm — the loss is the allocation×redistribution×
  controller interaction, NOT the map. So: **M3a coherence is proven; loop VALUE is not** —
  the next lever for closed-loop wins is the controller/redistribution design, not map
  fidelity. `attr-stale ≈ attr` (0.589/1.355) ⇒ the stale-scalar single-pass ≤1.1× perf
  endpoint is semantically viable when a positive product case exists. Fused adds
  +8.2ms/compare @576² (matches C3a's marginal). Caveats: n=9 cells/bake, t=92 clamps
  saturate, t=75 controller-overshoot-dominated. Tables:
  `jxl-encoder/benchmarks/zensim_attr_ab_2026-07-29.md` (+2 TSVs; medians re-derived
  independently from the TSVs — exact match). Bonus: C3b found+bisected the pre-existing
  `from_linear_planar` sub-64 panic (missing reflect-pad; ≤63px planar refs died in the
  mean-offset pass since the entry point landed) — FIXED this commit with a
  fails-without-fix regression test (`m1_sub64_planar_precompute_scores_and_matches_
  interleaved`, planar-vs-interleaved pad-path agreement ≤1e-4).
- **#69 loop-steering study (jxl `d17cf7ce`; plan `5f7d16a3`, gates frozen): H3 MAGNITUDE
  steering is the one loop rule with value** — passes G1+G2 on v47A (t70 |err| 0.31 vs
  baseline 1.87 at bytes ratio 0.99; supervisor re-derived medians from the TSVs — exact)
  and is the only arm beating baseline on nonphoto on BOTH bakes; staleness free (G4) so
  the single-pass ≤1.1× endpoint stays viable for H3. The ratio-normalized family (C3b
  attr / H1-signed / H2-ctrl) NEVER beats the plain damped controller, and on shippedB-
  linear ALL arms fail G1. Mechanism: score-unit steps skip the normalization that erased
  the map's magnitude information. ZENSIM_H3_GAIN unswept (registered default 10.0) —
  a gain sweep is future work, not claimed. Hazard noted: unknown JXL_ZENSIM_MODEL_MAP
  values fall through to baseline silently (caught in-run by a control-arm mismatch).
  `jxl-encoder/benchmarks/zensim_attr_loop69_2026-07-29.md`.
- **★ APPEND2 COVERAGE FIX (2026-08-04, `299ccc8c`) — EVERY 944-era M3a MEASURED BEFORE THIS
  COMMIT IS TOO LOW.** `compute_attribution_density_full` sliced `s[720..min(len,924)]`, so on
  a **944** bake the whole append2/BANDVIS block never reached the density. Determination from
  the feature definitions (not the slice): `BANDVIS_GAIN/LOSS` (8 slots) are **class E** —
  plain means of a per-pixel `bounded_excess` indicator, the exact form v2 `HF_GAIN/HF_LOSS`
  already carry — so they were **real dropped coverage**; `LUMA_MEAN_REF` (reference-only) and
  `HL_BIN1/2` (HDR-gated on a structurally-SDR route) are **correctly zero**, now by an
  explicit named decision instead of an unreached bound. Measured shift on the 32-bake 944
  population: **M3a rises materially** (registered sample: +0.0236/+0.0823/+0.0263/+0.1045/
  −0.0039; 3 of 5 change tier); M3 unchanged (the legacy fold is untouched);
  372/720/924 unaffected by construction. Guarded by
  `attribution_covers_expected_slots_per_width` (probes every width, so a regime bump cannot
  silently drop a block again) + a plane-sum identity vs the production 944 features (8-9
  digits) + per-slot FD direction. Superseded numbers are registered in
  `benchmarks/eval_annotations.json`. Record: `benchmarks/attribution_append2_e1_2026-08-04.md`,
  registration: campaign appendix E.

### Trainer/eval capabilities added this campaign (all on main)
- **MANDATORY embedded repro**: every new bake carries `zentrain.repro` (inputs w/ sha256 +
  rows, seed, argv, trainer HEAD, `best_val`) via `zenpredict_bake::append_metadata_utf8`
  (section-splice, byte/score-identity gated). Embed failure = exit 4. bake_verdict emits
  `repro` (embedded > .spec.json > null+warning); dashboard badges it.
- `--coarse-l2-mult` / `--coarse-decay` (decoupled, post-Adam-step; the coupled form is a
  no-op under Adam — do not use it expecting effect). `ZENSIM_DECAY_DEBUG=1` telemetry.
- `bake_verdict`: sdr25 corpus; bands + per-codec dial curves + gates + `model` block +
  bootstrap `srocc_ci` + signed SROCC + `frac_negative` + `train_eq_val` in `--full-json`;
  `product_composite` = THE ranking composite (dashboard reads, never re-derives).
- `diffmap_block_coherence`: n_in==924 via the CANONICAL folded-append streaming extractor
  (extended path would inject untrained-weight noise into the structural-zero block);
  `ZENSIM_GRAD_MASS=1` gradient-mass diagnostic (region/v2-slot/basic-scale/top-idx);
  tie-correct midrank Spearman; dropped-mass printed for every layout.
- `run_full_eval.sh`: `ZENSIM_M3_REUSE=1` (schema re-emits carry M3 — a fulleval re-run is
  a cheap rescore over stored parquets; only re-measure when bake/parquets/fixtures change).

### Gotchas (bled for; do not re-learn)
- pq.write_table defaults SNAPPY → the Rust parquet reader has no snap; write eval grids zstd.
- f32-vs-f64 + clip-saturation traps in parquet key-joins (multiset join on
  `(source_id, f32(clip(s/100)))`).
- The JS dashboard template is a RAW Python string: `\'` becomes a literal backslash-quote
  and kills the whole <script> (blank page). Gate regen on `node --check` + the DOM-shim
  render harness (both in the pipeline now).
- pkill/pgrep -f self-match the invoking shell (locally AND over ssh); pgrep name-match
  truncates comm to 15 chars. Kill by PID; use pgrep -x with the truncated name.
- Observe-before-load on fleet nodes (node-2 7× incident = a live zensim-720 backfill
  worker, not a slow box).
- Trainer-bin globals must be set BEFORE the training call — the best_val relocation
  silently moved the regularizer setup post-training (caught by the decay-debug counter).

## LATENCY + TOKEN DISCIPLINE — idle waiting is re-charged, not cached (2026-08-04)

**MEASURED, `benchmarks/rnd_cycle_audit_2026-08-04.md`.** Over the 2026-08-03/04
campaign (34.3 h, 11 waves): **14.80 h of whole-session idle, 6.77 h of it dead**
(nothing computing, or finished work sitting unharvested), and **$395.24 —
13.9 % of the $2,837.34 session — spent re-creating prompt cache that idle
waiting had expired.** The mechanism: cache entries are `ephemeral_5m`, read at
0.1× and written at 1.25×, so **any gap over 5 minutes converts the whole
prefix from a 0.1× read to a 1.25× write — a 12.5× multiplier** on 500–800 k
tokens. 138 turns (3.7 % of all turns) carried 22.9 M re-created tokens =
**55.7 % of every cache-write token spent that day**. One agent idled 7.92 h of
its 11.69 h span and alone burned 12.15 M cache-write tokens; its worst single
turn re-created **779,717 tokens after a 141-minute gap**.

Rules, all load-bearing:

- **Do NOT park on short-interval polls.** Polling is not what costs — the polls
  read a warm cache (57 wait-turns = 626 k write vs 21.2 M read). The **wake-up
  after a long silence** is what costs. So the fix is never "poll less often";
  it is "do not be idle-attached at all".
- **Arm ONE terminal condition, then go do other work.** `Monitor` a file that
  appears exactly once — `scripts/await_artifacts.sh --heartbeat X` writes
  `X.done` on **every** exit path (COMPLETE / TIMEOUT / SIGNAL + rc). Never
  `Monitor` a `tail -f` (loses the file on rotation) and never hand-roll a
  `while sleep` waiter: the two worst events of the day, **125.6 min** and
  **80.6 min** of dead wall-clock, were both a bespoke waiter that stopped
  without leaving evidence.
- **Supervisors must not idle-wait on delegated work.** A supervisor watching a
  subagent ages *two* prefixes, so one event re-charges both. Do independent
  work; let artifacts be the channel.
- **Make a late wake-up free.** Harvest on completion — `scripts/harvest_bakes.sh`
  verdicts + fullevals each bake as it lands, so results are already on disk
  when anyone next looks. A post-bake hook MUST fail loud: the coherence wave's
  hook exited 2 nine times into an unread log and silently voided a 3 h 24 min
  lane (21 verdicts re-run by hand, 804 s).
- **Batch status into one report.** Do not emit a turn per artifact; one report
  per terminal event.

Wave skeleton + the priced anti-pattern table: [`docs/WAVE_PLAYBOOK.md`](docs/WAVE_PLAYBOOK.md).

**Not a cause, measured and rejected:** per-agent `CARGO_TARGET_DIR` cold
rebuilds. Total `cargo` wall-clock across every agent all day was **23.0 min**
(91 builds; cold `bake_verdict` = 72 s / 221 crates), while a second concurrent
`cargo` on a *shared* target dir **blocks 31.8 s** on the build lock. Keep
per-agent target dirs; agents that only consume binaries should build nothing
and use the `ZL_BV` / `ZL_TRAIN` / `CARGO_TARGET_DIR` pointers the drivers
already honour. The real target-dir cost is disk — 28 dirs, 113.6 GB, root at
95 % — so delete yours when a wave closes.

## ⇒ POST-COMPACT / NEW SESSION: read [`SESSION-RESUME.md`](SESSION-RESUME.md) FIRST

Then return here. `SESSION-RESUME.md` is the canonical entry point —
it points at every other doc + lists the current critical-path
tasks. Reading order on resume:

1. [`SESSION-RESUME.md`](SESSION-RESUME.md) — current state, ~1 min
2. [`docs/TOP_MODELS_COOKBOOK.md`](docs/TOP_MODELS_COOKBOOK.md) — **the validated
   science + exact reproduction of the top models + the new-model loop + the
   pitfall list.** THE entry point for any model work (2026-07-18).
3. [`docs/MODEL_SELECTION_SCORECARD.md`](docs/MODEL_SELECTION_SCORECARD.md) —
   the five-gate exam (RANK/DIAL/STEER/RD/TARGET) every ship candidate takes.
4. This doc (`CLAUDE.md`) — methodology + workflow + gotchas. **NOTE: the V0_x /
   PreviewV0_5-era historical sections were excised 2026-07-19 to
   [`docs/HISTORY-2026-05-v0x-era.md`](docs/HISTORY-2026-05-v0x-era.md); the
   cookbook supersedes them for current state.**
5. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — **the traceability
   spine**: how a number chains back to bytes (verdict → bake sha → manifest →
   input shas → trainer commit), which gate enforces each link, and the honest
   list of gaps. Read before making or citing any measurement.
6. [`RESEARCH.md`](RESEARCH.md) — corpus map + workflow recipes
7. [`benchmarks/INDEX.md`](benchmarks/INDEX.md) — find prior experiments
8. Run `TaskList` and work on the lowest unblocked task

(CONTEXT-HANDOFF.md is DELETED — handoff files are banned; durable facts live in
the docs above. The IQA literature corpus is `~/work/zen/zenpapers` — search it
before designing features or metrics.)

## RECURRING PRIORITIES + ASSETS — do not re-forget (consolidated 2026-07-16)

The user has had to repeat these across the last week. They are load-bearing;
re-search + honor them EVERY session. This section exists because they kept
getting lost (wrong dashboard picked twice, HF parquets forgotten, bigcodec
metrics assumed absent, negative-value + diffmap-coherence requirements dropped).

### The product = a CONSISTENT DIAL (not just a ranker)

Users type a target zensim; the codec tunes to hit it, using the diffmap to close
the loop. Every metric decision serves this:
- **Monotone in codec quality** (so target-hitting converges) + **bounded [0,100]**.
- **NEGATIVE zensim values MUST work** — inputs worse than the worst codec output
  score BELOW 0 (do NOT clamp at 0; the lower spline extrapolation + profile
  `extrapolate_score` carry it). Negative-tail training data:
  `canonical-2026-07-15/train/kadis_negrich.parquet` (negative-rich).
- **The diffmap MUST match the scalar** — `DiffmapResult.diffmap()` must reflect
  the SAME model as `.score()`, so the per-block "where to adjust" signal drives
  the closed loop. Currently INCOHERENT (diffmap uses per-scale SSIM weights,
  scalar uses the 372-feat model) — this is the #1 closed-loop blocker.

### Evaluation north stars (priority order)

- **ssim2 is the best north star for NON-PHOTO content** (user directive).
  `imazen26` (real-codec ssim2) + `nonphoto` (non-photo ssim2) are FIRST-CLASS
  gates in bake_verdict (G-IM26, G-NP). Eval every ship-grade bake on them.
- **CID22** = gold human-MOS holdout (validation only). CID22 trades are user-gated.
- **HF near-lossless is the metric's WEAK ZONE** — high-fidelity / near-lossless
  (B8/B9, q75-100) is where compression product decisions live AND where every
  learning metric is weakest. Always eval AND train it.

### Data assets that keep getting forgotten

- **HF near-lossless parquets**: `canonical-2026-07-15/train/hf_nearlossless_{train,val}.parquet`
  (900 + 300 rows × 372 feat; targets human_score + ssim2_gpu). The `hf_nearlossless`
  corpus in bake_verdict. INCLUDE in training + eval.
- **Negative-rich data**: `canonical-2026-07-15/train/kadis_negrich.parquet` — the
  negative-dial-tail training corpus.
- **bigcodec's cvvdp/iwssim ARE backfilled** (the depth-iter
  `bigcodec_train_120k_stride.parquet` is ssim2-only, but the metrics exist
  elsewhere). CONCRETE (audited 2026-07-16): bigcodec cells WITH `score_cvvdp` +
  `score_iwssim` (99.6%/100% non-null, feature-space, f0..f371) live at
  **`/mnt/v/output/zensim-multicodec-probe/bigcodec_mm6_traindigits_2026-07-02.parquet`**
  (1.56M rows; 6 codecs + jxl-hqfill, NO avif — "mm6"). Authoritative per-encode
  metric sidecar (all 6 codecs, 4.18M rows, key=`encoded_filename`/`encode_sha`):
  `/mnt/v/datasets/fill4-6codec-2026-07-01/fill4metrics_sidecar_patched_2026-07-02.parquet`
  + JXL near-lossless top-up `hqfill_7metric_sidecar_2026-07-02.parquet`. The
  120k-stride file dropped `encoded_filename`, so the fallback fingerprint to
  rejoin it is `(ref_basename, round(human_score,9), round(f0..f2,9))` (per
  `scripts/v_next/linear_projections_2026-07-03.py:1064`). Column names differ:
  mm6 uses `score_cvvdp`/`score_iwssim`, canonical uses `cvvdp_score`/`iwssim`.
- **CVVDP/IW-SSIM mix training targets**: safesyn/kadid/tid/cid22_train carry
  `iwssim`, `cvvdp_score`, and `mix_cv{25,50,75}_iw{75,50,25}` (all positive-
  direction, [0,100]). Use them (`train_minmax --synth-target`, or the trainer's
  `--target-column`) to recover CID22 from the ssim2-shaping bias. bigcodec/kadis
  need the sidecar join first. NOTE: pure-cvvdp-SCALAR target is a known dead-end
  ([[feedback_cvvdp_scalar_target_dead_end]]); a MIX / IW-SSIM is the ask.
- **High-quality-zone HUMAN data (untapped, local)**: JPEG-AI-SDR25
  (`/mnt/v/datasets/jpeg-ai-sdr25/`, 95k triplets, q75-100), AIC-3 raw triplets
  (`/mnt/v/datasets/aic3-btc-ptc/`, 420k), AIC-HDR2025 (`/mnt/v/datasets/aic-hdr2025/`).

### THE dashboard (the "pretty one" the user means)

`scripts/v_next/bandwise_dashboard.py` — the every-graph dashboard. THIS is the
combined dashboard; EXTEND it, don't rebuild a thinner one. Three modes:
- `--bakes label:path.bin,...` — compare bakes directly (shipped B auto-prepended;
  ssim2/cvvdp/butteraugli refs auto-added).
- `--from-search /mnt/v/output/zensim/reports/blend/blend_results_r7_2026-07-15.json`
  — the blend-candidate view.
- **`--fulleval-dir /mnt/v/output/zensim/reports/fulleval` — the INTERACTIVE
  summer-gauntlet** (2026-07-26). Reads pre-computed per-bake `*.fulleval.json`
  (schema + fixtures: `scripts/v_next/make_stub_fulleval.py`; ordering from
  `best_per_day.json`) and emits ONE self-contained, **OFFLINE** HTML
  (`--out …/summer_gauntlet.html`) via `scripts/v_next/gauntlet.py`: bake-toggle
  checkboxes (stable per-bake color; all/none/top-6), a **sortable scoreboard**
  (click any header — CID22/KonJND/dial-mono/M3/corruption/composite/…), a
  cross-corpus SROCC heatmap, the CID22-vs-{nonphoto,KonJND} trade map, and the
  **correlation SCATTER MATRIX** — predicted vs each reference (MOS/JND/ssim2/
  butteraugli/cvvdp), one clean faceted scatter per (bake × corpus) with OLS fit +
  canonical SROCC/PLCC. Hand-rolled inline SVG+JS (no CDN/plotly — opens offline),
  theme-aware (light/dark), dataviz-validated palette. **Stats are NEVER
  hand-rolled**: SROCC/PLCC come from the fulleval JSON's `scatter` block (eval
  agent → canonical `panel`) or, if omitted, `scripts/lib/zen_stats.panel` at build.
  Plus (2026-08-01) the **JXL loop-targeting panel**: `2shot/3shot ±2` scoreboard
  columns (emit-best, mapped bakes only via `gauntlet.LOOP_BAKE_MAP`) + its own
  section table (all loop models incl. emit-last, the outer arms and ssim2, which
  are not bakes), fed by the jxl-encoder sweep summary via `--loop-targeting`
  (default = the committed
  `~/work/zen/jxl-encoder/benchmarks/zensim_loop_23shot_summary_2026-08-01.json`;
  section omitted with a loud note when absent). Counts/medians are READ from that
  JSON, never re-derived (the jxl-encoder analyze script is the owner).
  Plus (2026-08-03) the **cross-bake per-band SROCC table** under the Mohammadi
  panel's 10-band bars: rows = visible bakes (same order + color), columns = the
  bands that carry pairs (n=0 bands are dropped — CID22's B0/B1 are structurally
  empty), header shows each band's n, cells with n&lt;30 render parenthesized +
  dimmed, and a JS-computed **band-profile line** names the leader in the highest
  and lowest populated band (on CID22 the two ends have DIFFERENT leaders — the
  top band B9 is a real discriminator while the low bands rest on a few dozen
  pairs). Read down a column, never across one: band SROCC is range-restricted.
  All values are read from `rank.<corpus>.bands[]`; nothing is recomputed.
  Plus (2026-08-04) **ENSEMBLE rows**: a fulleval JSON stamped `model.kind:"ensemble"`
  by `scripts/promote_fulleval.py --members` (the generalized promoter — it publishes
  ANY verdict, single-bake or ensemble, onto the board and recomputes NOTHING — every
  stat block is asserted byte-identical to the source verdict, whose path+sha256 land
  in `source_verdict`; `--strip-per-pair` implements the registered board-size rule,
  `--graft-into` folds a `*_corrjoint` verdict's corruption_head under the plain name)
  renders an `ens×k` marker wherever the bake is named, and its Model-details card
  leads with a warning that the architecture/repro shown is the ANCHOR member only;
  `m3_coherence`/`m3a_coherence` are **null** because the coherence instrument loads
  one ZNPR, and null renders as an em-dash (NOT MEASURED — never shaded or displayed
  as a measured zero). Wave-5's six arms are promoted by `scripts/wave5_ensemble.sh
  promote`, which reuses the same frozen membership arrays it scored with.
  Plus (2026-08-04) **FULL-GRID COVERAGE + curation + zoom/pan**: every sota944
  campaign verdict cell is on the board (162 bakes; `scripts/promote_sota944_board.py`
  — a caller of the generalized `promote_fulleval.py` — with a COVERAGE GATE: every
  non-excluded verdict must map to a board file; mapping index
  `fulleval/_sota944_board_map.tsv`; excluded = LOO/REPROCHK/XBUILDCHK/GATE/recheck/
  SMOKE instrument duplicates; `*_corrjoint` fold under the plain name as sha-gated
  corruption_head grafts). Presentation is two-tier so the grid doesn't drown a fresh
  reader: `gauntlet.py CURATED_BOARD` (29 names — era flagships + every arm-candidate/
  named leader + the ensembles) is the default-visible set AND the only set with
  embedded per-pair scatter (registered size rule; grid-interior cells keep every
  scalar stat, their full per-pair stays in the source verdict, recorded via
  `source_verdict` + `per_pair_stripped`); family toggles (arm A/B/C-seeds,
  coherence/W4, near-top, distilled, ensembles, era bridge, pre-944 era) + a
  collapsible per-bake picker + 'curated' preset replace the flat 160-chip bar; the
  scoreboard still lists every cell (dimmed = hidden, click to toggle).
  `model.feature_transforms` embeds are capped at the 48 chips the card can show
  (`n_feature_transforms` keeps the true count).
  Plus (2026-08-04, dashboard-rebuild session) **ECharts SEMANTIC zoom + real
  sortability + regime truth**: the five heavyweight panels (scatter-matrix cells,
  per-codec dial curves, 10-band bars, cross-corpus heatmap, trade maps) are **Apache
  ECharts 5.6.0** (canvas renderer) — dataZoom rescales the AXES and re-plots while
  marks/strokes/labels stay constant size; trade-map labels `labelLayout.hideOverlap`
  (hidden at 1x, REAPPEAR zooming in); dial tooltips show p25/p50/p75 at the hovered
  q; heatmap has a calculable visualMap; band bars show negative bands (old view
  clamped to 0); double-click = reset. The predecessor `makeZoomable` viewBox zoom
  (geometric — overlaps stayed overlapped at every level) is DELETED. **Vendoring:
  the bundle is >30 KB so it is NEVER in git** — bytes at
  `/mnt/v/zen/vendor/echarts/echarts-<ver>.min.js`, described + sha256-pinned by the
  committed `scripts/v_next/vendor/echarts.pointer.md`; `build_html` verifies the
  sha and fails LOUD with download instructions (env `ZEN_ECHARTS_JS` overrides the
  path); the bundle rides its own `<script id=vendor-echarts>` ahead of the app
  script. Charts ink from `THEME_VARS` — ONE Python dict generating both the CSS
  custom properties and `DATA.chartThemes` (light+dark) — and rebuild on
  prefers-color-scheme flips AND on the artifact viewer's `data-theme` attribute
  (MutationObserver, typeof-guarded). echarts.init is guarded on a real canvas 2d
  context + lazy on first viewport intersection, so the DOM-shim harness (no canvas)
  still renders the page. Same session fixed the **scoreboard sort regression**
  (th.onclick called renderTable() — which RETURNS a detached wrapper — instead of
  mountTable(); sorted tables were built and thrown away since 62404415) and made
  EVERY stat table sortable (`makeSortable`: Mohammadi, band, gates, loop). The
  scoreboard/chips **regime** now shows the model's TRUE input width from `n_inputs`
  (372/720/924/944-class) — the stored campaign flag string reads "720" cosmetically
  on all 166 board JSONs. 166-bake board = ~10.3 MB (cap 12 MB), gates PASS.
  Plus (2026-08-04, board-integrity pass) **the annotations-registry + dominance +
  block-usage layer**: `benchmarks/eval_annotations.json` = ONE committed
  machine-readable registry of invalidated/annotated/absent-not-failed numbers
  (schema in its `_schema` header; append-only), consumed by `freeze_check
  --annotations` (absent axes print `— (absent)` DISTINCT from measured fails;
  n/8 keeps the registered absent=not-passed rule AND the n/m-measured form is
  stated; TSV +n_measured/absent/annotations/blocks/dominated_by) and by the
  board (⚠ badges + hover reasons, caption line, `DATA.annRegistry`).
  `dominated_by` marks = strict same-class Pareto (`scripts/sota944_dominance.py`
  rule strict-pareto-2026-08-04 + `promote_fulleval.py --mark-dominated`;
  17 trims, all 944-single) — dominated cells render dimmed + default-off behind
  a "dominated" chip, files NEVER deleted. Scoreboard gained **HF-NL/ref**
  (hfnlproxy per-ref mean; the era incumbents were FILLED by
  `derive_hfnlproxy_372.py` — exact-key join, identity-gated vs the 944 slice —
  and grafted sha-gated via `promote_fulleval.py --graft-rank`; era per-ref
  0.64-0.83 ≫ the 944 frontier's 0.13-0.42). **Feature-block-usage filter**:
  `bake_block_profile` (via `zenpredict::Model`; no new wire code) →
  `block_profile` injected into all fullevals (`--set-block-profile`, sha-gated);
  "uses f156-371" chip + per-family used/total on the Model-details card —
  f156-371 were ZEROED by the folded regimes (slots preserved per the
  append-only discipline, not removed); 944 MLPs zero them exactly (216/216),
  B uses 49/216. §8's falsified pool-wide census was corrected in place
  (campaign doc CORRECTIONS + BOARD-INTEGRITY PASS sections).
  **Regen gates (MANDATORY, run on every emitted HTML):**
  `scripts/v_next/gauntlet_gates.sh <html>` = `node --check` on EVERY extracted
  `<script>` block + the DOM-shim render harness (`gauntlet_render_check.js`) —
  committed 2026-08-01 (previously ad-hoc; the raw-Python-string `\'` escape class
  blanked the page once, e7f929ca). The harness now ALSO dispatches real header
  clicks and asserts the ATTACHED tables reorder (the sort-regression test), checks
  ECharts mounts + built options + both chart themes, and SSR-renders one option per
  panel kind through the real echarts (svg SSR, no canvas) so a malformed option
  fails the gate instead of blanking the page.
The first two modes' plots: per-bake scatter+trend, grouped 10-band SROCC bars,
calibration curve, residual, candlestick, SROCC heatmap, 2-panel Pareto trade
(CID22 vs nonphoto / KonJND), composite ranking bar, per-codec dial plots +
dial-mono %, full Mohammadi stat panel (incl. low-tail/high-tail SROCC), 10-band
table, honesty/provenance panels. Run from `scripts/v_next/` (imports `blend_lib`/
`gauntlet` from cwd). `bake_report.py` adds the 2×4 8-corpus scatter grid with 4PL
fit + PWRC (reports/); `bake_verdict --html` is the single-bake Rust report.

> **Historical (May-2026 V0_x / PreviewV0_5 era):** the training goals + three-trail SOTA + shipping/experiment-rigor policies, 2026-05-1x eval mandates, V_20/V39 learnings, canonical-2026-05-18 corpus archaeology, the interactive-site spec (since shipped: `site/compare.html`), V_X experiment workflow, V0_1-era weight status, and V0_7 e1 fill were moved verbatim to [`docs/HISTORY-2026-05-v0x-era.md`](docs/HISTORY-2026-05-v0x-era.md) on 2026-07-19. Current guidance: [`docs/TOP_MODELS_COOKBOOK.md`](docs/TOP_MODELS_COOKBOOK.md).

## TWO-PANEL EVAL MANDATORY — rank + dial, every ship-grade bake (added 2026-05-29)

**`bake_verdict` runs BOTH panels natively (Rust) on every invocation** —
the DIAL panel is built in, so any time you compute a bake's SROCCs you
also get its dial metrics. Never accept a rank-only verdict:

1. **RANK panel** (`bake_verdict`) — full Mohammadi 2025 stats on the 6
   canonical val parquets. Held-out corpora are CID22 + AIC-3 + AIC-4
   (+ KonJND semi); **KADID/TID are 100% train==val pair-overlap** so
   their numbers reward memorization, not skill — treat as integrity
   guards, not ranking signal.
2. **DIAL panel** (`qsweep_eval` on the densified multi-codec grid) —
   monotonicity + tied-rate + per-q dial span across codec configs
   (G1 dynamic range, G3 monotonicity ≥93% / tied ≤5%, G4 reach). The
   grid is densified where dial precision matters: **q0 + step-1
   q90→q100 + JND-zone (q70→90 step2) + JXL-in-butteraugli-distance**,
   4 codec families, 372 features.

A bake can win the rank panel and be a broken dial (V0_5 Balanced:
panel-best by meanG3, 60% tied / collapses to 0 above q50). A bake can
pass a coarse dial and fail near-lossless step-1 (Cell5: 0.8% tied on
the 16-q grid → 13.1% on the densified grid). **Single-panel verdicts
are a regression — do not accept them.**

**Stored feature sets live on R2** (`s3://zentrain/eval-grids/`:
`dial_grid_372col_2026-05-29.parquet`,
`corruption_grid_372col_2026-05-28.parquet`). `bake_verdict` reads the
dial grid directly (default path or `--dial-grid` / `ZENSIM_DIAL_GRID`)
and forwards the bake over the stored 372-feature vectors —
**rescore any model with no re-encode/re-extract.** If the grid isn't
local, `bake_verdict` emits a loud SKIPPED note; fetch it once with
`aws s3 cp s3://zentrain/eval-grids/...`. Full spec + gates + refresh:
`docs/EVAL_PANEL_REQUIREMENT.md`. Pointer:
`benchmarks/eval_grids_2026-05-29.pointer.md`.

## JSON pipeline mandate for ZNPR v3 bakes (2026-05-15)

**Ad-hoc Python emitters for ZNPR v3 wire format are BANNED.** All
new bake-side serialization goes through the
`zenpredict-bake <input.json> <output.bin>` CLI (binary at
`~/work/zen/zenanalyze/target/release/zenpredict-bake` after a
`cargo build --release -p zenpredict-bake`).

The JSON format is documented in `zenpredict-bake/src/json.rs`:
`BakeRequestJson` with fields `schema_hash, flags, scaler_mean,
scaler_scale, layers[], feature_bounds[], metadata[],
output_specs[], sparse_overrides[]`. Per-bake metadata entries
declare `key: String, type: utf8/bytes/numeric, value: ...`.

Use `scripts/v_next/v0_20b/bake_znpr_v3.py` as a template — emits
JSON, shells to `zenpredict-bake`, exits.

**Why**: the wire format is small but easy to get wrong (alignment,
section ordering, header layout). zenpredict-bake is the canonical
serializer; trusting it keeps wire-format invariants in one place.
Ad-hoc emitters drift, get out of sync with v3.x extensions, and
ship wrong-shape bakes that load but score garbage.

## CID22 is VALIDATION-ONLY (added 2026-05-15)

**CID22 human MOS is sacred validation across the entire zensim
project. NEVER use CID22 human MOS as a training target.** This rule
is load-bearing — every documented contamination cleanup
(2026-05-12 perceptual-overlap purge, 2026-05-14 dHash audits) exists
to defend this gate.

### What "validation only" means in practice

- **NO** `--group cid22:...` argument in any `zensim_mlp_train`
  invocation that loads CID22 human MCOS as the `human_score`
  column. CID22 human MOS appears only at the END of an experiment
  via `dataset_metric_baseline --cid22 /mnt/v/dataset/cid22/...`.
- **NO** "CID22-train-fold" or "CID22-train-subset" carved out of
  the validation set for fine-tuning a head. The 49-reference
  held-out set is the WHOLE CID22 (4,292 pairs across the 49 refs).
  There is no "training-fold half" to peel off.
- **NO** indirect leakage: training-source perceptual-near-duplicates
  of CID22 references count as contamination too. The
  `check_holdout_overlap` audit (dHash d≤10 + user-eye verification
  per the 2026-05-14 revert) is mandatory before any new training
  corpus lands.

### What IS permitted

- CID22 ssim2 or CVVDP metric scores on the **training-only subset
  of the broader CID22 image library** (i.e., images that exist in
  the CID22 source pool but are NOT part of the 49-reference
  validation set + their distorted pairs). The training-only subset
  must be extracted from a different source than the validation set
  on disk — typically the unfiltered CID22 image library at the
  upstream source, NOT `/mnt/v/dataset/cid22/CID22_validation_set/`.
- Metric-anchored training signal on that training-only subset uses
  ssim2 (fast-ssim2 / GPU ssim2) or CVVDP as the target column —
  never human MOS.
- Whoever extracts the training-only-subset metric-anchored CSV
  MUST document the cut clearly (`_MANIFEST.md` entry: "CID22
  training-only subset, ssim2-anchored, N pairs, source images
  NOT in the 49-ref validation set, verified by basename diff").

### What's currently extracted

`/mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.csv`
is **validation only** (4292 pairs from the 49-ref held-out set,
`human_score` = MCOS / 100). It exists for end-of-experiment full-
panel evaluation, NOT training input. The file's `_MANIFEST.md`
spells this out.

The historical V_18/V_19/V_20a/V_20b training pipelines have NEVER
included CID22 as a `--group` to the trainer — confirmed by
inspecting every methodology doc at `benchmarks/v0_1*_methodology*.md`
and `benchmarks/v0_19_REVERTED_2026-05-14.md`. The training command
loads `safesyn + kadid + tid + konjnd` only.

### Why this rule is absolute

CID22 (Sneyers / Ben Baruch / Vaxman 2023, JPEG WG1 `wg1m99012`)
is the only large human-MOS dataset that exercises **codec-output
distortions** specifically (KADID + TID are ~95 % non-compression
synthetic distortions). It is the **single gold-standard
generalization holdout** for compression-targeted metrics. If we
train on any part of its human-MOS labels — even a "train fold"
carved from the same 49 references — we lose the only honest
generalization check we have.

Past CID22-contamination incidents (V0_8 perceptual-near-duplicate
leak, V0_19 indirect KADID-overlap inflation) cost the recovery
cycle weeks of wasted training. The "no CID22 human MOS as training
target" rule prevents the next such incident. Re-read this section
whenever drafting a new training corpus or fine-tune fold.

## ZNPR v2 PROHIBITED (added 2026-05-15)

**Producing ZNPR v2 bakes is BANNED. Period.** Every new bake MUST
be v3 (header byte 4 = `0x03`). Tools producing v2 are bugs that
need fixing on contact — not "legacy support" or "compatibility
shims."

### Why

The current zensim runtime loads v3 bakes only. The 2026-05-15
falsification re-evaluation exposed ~150 pre-existing v2 bakes
across `benchmarks/rust_*`, `benchmarks/h*x*`, and
a scratch bakes dir (artifact was under /tmp — wiped; re-derive if needed) that are **structurally unevaluable** by
the current runtime — every recovery-cycle falsified hypothesis
(cycles 7–14) is locked behind this wire-format gap. Producing
more v2 makes the gap worse and creates "ghost bakes" that look
like data but can't be re-tested.

### How to comply

- **Bake-emitting code** uses `zenpredict::bake(&BakeRequest{...})`
  (the v3 path). NEVER call `zenpredict::bake::bake_v2`.
- **Read the bake's header byte 4** as a smoke test in any tool
  that produces a bake: assert it's `0x03` before writing the file.
- **Function names + docs** that say "v2" but emit v3 are
  misleading — rename + correct comments on contact (e.g. zensim's
  `bake_two_layer_znpr_v2` was renamed to `bake_two_layer_znpr_v3`
  on 2026-05-15; the function had been emitting v3 internally for
  weeks).
- **Tests that lock in v2** (`assert_eq!(version, 2)`) are wrong —
  fix them to assert v3.

### Audit list (as of 2026-05-15)

Existing `bake_v2` callers in this repo:

- `zensim-train-core/src/mlp.rs` — REMOVE v2 path; only emit v3.
- `zensim-bench/examples/quant_compare.rs` — same.
- `zenpredict::bake::bake_v2` is still EXPORTED from the sibling
  `zenanalyze/zenpredict` crate, but it MUST NOT be imported into
  zensim crates. If you see `use zenpredict::bake::{..., bake_v2}`,
  fix the import to `bake` only.

### Re-bake old v2 bakes when possible

If a falsification's bake is v2 and the hypothesis is worth
re-testing: **retrain** under the current trainer (which emits v3
through `bake()`). Don't write a v2→v3 upgrade tool — the right
fix is "retrain, evaluate on full Mohammadi panel" per the
principled experiment workflow. Bakes are cheap; ghost data isn't.

## NO DUPLICATE IMPLEMENTATIONS — one owner per task, extend it or don't do it (2026-07-15, user directive)

**Every task below has exactly ONE canonical implementation. Re-implementing
any of them — in Python, in a second Rust site, in a script, anywhere — is
PROHIBITED. Not discouraged: prohibited.** If the owner can't do what you
need, **extend the owner**. That is always the move.

| Task | THE owner | Never |
|---|---|---|
| Load a feature parquet | `zensim-validate/src/parquet_loader.rs` (`load_parquet`) | `pq.read_table` in a script that then trains/evals |
| IQA stats (SROCC/PLCC/KROCC/OR/PWRC/Z-RMSE) | **`zenstats`** (`zenmetrics/crates/zenstats/src/panel.rs`), reached via `zensim_validate::panel` (an 82-line `pub use` re-export — the shim is NOT the owner) or the `panel` bin; Python shells it or uses `scripts/lib/zen_stats.py` | `scipy.stats.spearmanr`, a hand-rolled `_srocc`, any private stat math |
| Per-reference SROCC | `bake_verdict` (first-class panel row since 2026-07-15) | reducing `--per-pair-output` in a script |
| Train a model / bake | `zensim-validate/src/bin/zensim_mlp_train.rs` | a torch MLP in a script |
| Evaluate a bake | `bake_verdict` (rank + dial panels) | ad-hoc scoring loops |
| Edit bake bytes (spline / winsor / gate) | `bake_dial_refit` | numpy PCHIP, `struct.pack` |
| Serialize / inspect / repack a bake | `zenpredict` CLI (`bake`/`inspect`/`repack`) | any other ZNPR emitter |
| Build a canonical corpus parquet | `scripts/canonical_corpus/` + `join_safety` | a bespoke join in a probe script |
| Train/val/test split | `zenmetrics/scripts/picker/origin_split.py` (`split_of()`) | a seeded shuffle (per-rendition → scale leakage) |

**Python is not banned — DUPLICATION is.** Python is correct where it IS the
owner: canonical-corpus building, plotting/HTML dashboards, R2 sync. It is
prohibited where a Rust owner exists. The test is not "what language" but
"does this already have an owner". A second **Rust** site is just as much a
duplicate as a Python one — zensim currently carries ~10 private Rust copies of
`spearman` across probe binaries plus a separate impl in
`zensim-train-core/src/stats.rs`.

### The ONE exception: a gated mirror

A second implementation is legitimate **only** when it exists for a measured
engineering reason AND a test holds it bit-exact against the owner. Two real
ones, both keep-don't-delete:

- `zenpicker-train/src/picker_eval.rs` — `pwrc_sa_st_auc_lowmem` (O(n²)→O(1)
  memory), gated by `pwrc_lowmem_matches_canonical_exactly`.
- `zensim-validate/src/panel.rs` — `compute_light_panel_subsampled`, which
  fixed a 307 GB OOM.

"Mirror" means: the owner is still the source of truth, the mirror exists to
solve a *specific measured* problem the owner can't, and a test fails the build
the moment they diverge. Without that test it is not a mirror, it is a fork
with a good story.

### Why the old narrow rule failed

A rule already said "Do NOT hand-roll srocc/plcc/krocc/pwrc/z_rmse in Python"
(the 14-fork consolidation, `benchmarks/iqa_stats_consolidation_2026-05-26.md`).
On 2026-07-15 an audit of `scripts/v_next/` found **30 of 134 scripts
hand-rolling IQA stats anyway**, 69 loading parquet in Python, 11 running
parallel torch trainers, 33 editing bake bytes after that work was migrated to
`bake_dial_refit`. The rule named one *symptom* (srocc in Python) instead of
the *principle* (one owner per task), so every new script re-derived the
forbidden thing under a slightly different name. This section states the
principle. It covers tasks, not function names, and it covers Rust too.

### What duplication actually costs — measured, not theoretical

- **It hides capability gaps.** `blend_lib.py` grew a within-ref RankNet term
  in Python because nobody checked whether the Rust trainer could do it. It
  couldn't — `zensim_mlp_train` drew every pair uniformly across a group
  (cross-image). The gap sat invisible behind a working Python script until
  2026-07-15. Extending the owner surfaced it in an hour and fixed it for
  every future recipe; the duplicate would have hidden it forever.
- **It diverges silently, and you find out months later.** `bake_verdict` once
  had its own inline copy of every stat. When `panel.rs`'s OR + PWRC were
  rewritten to the paper-correct ITU-T P.1401 / Mohammadi SA-ST forms
  (`83e7ff70`), the copy wasn't. **Every bake_verdict output before that fix
  reported the wrong OR + PWRC** while the `panel` binary reported correct ones
  on the same fixture. Nothing failed; the numbers were just wrong. That is the
  characteristic damage: not a crash, a quietly wrong number in a shipped
  report. (The 2026-05-26 consolidation found the same shape three more times —
  PWRC argument order off by ~0.2, an OR definition off by 0.375, and one
  script whose "pwrc" was Spearman-as-Pearson and not PWRC at all.)
- **It re-pays the same debugging.** `blend_lib._load` OOM'd on a 5.3 GB
  parquet (one `read_table`, ~2x peak). `parquet_loader.rs` had never had that
  bug. The duplicate bought a fresh copy of a solved problem.
- **Extraction is not migration.** The 2026-05-26 consolidation succeeded
  architecturally — `zenstats` shipped, both siblings consume it, the parity
  gate passes at ~5e-11. It still failed behaviorally, because the old call
  sites were never migrated and new ones kept appearing. In zenanalyze,
  `load_features_raw` adoption went from 7-of-25 to ~15-of-35: the lib exists,
  the forks kept coming. **Landing the owner is half the job; deleting the
  callers is the other half, and it is the half that gets skipped.**

### NEVER hardcode a sibling-worktree path in a committed script

**A worktree is ephemeral; the repo is not.** Writing
`/home/lilith/work/zen/zensim--my-experiment/target/release/foo` into a script
you commit guarantees that script dies the moment the worktree is cleaned up —
which the mandatory cleanup rule says MUST happen. The worktree rule and the
script outlive each other badly: cleanup works, and silently leaves fossils.

MEASURED 2026-07-15: **25 of 130 scripts** in `scripts/v_next/` pointed into
`zensim--cross-codec-metric`, `--cross-codec-v7/v8/v9`, `--v10`,
`--v10-human-eval`, `--eval-accel`, `--picker-train`, `--exp-tuner-v2`,
`--cli-per-codec-calibration`. Every one had been unrunnable for weeks. **None
of them needed to be**: a worktree is a *copy of this repo*, so every one of
those binaries exists here — the fix was `zensim--whatever` → `zensim`, one
sed, zero deletions.

- **Reference the main repo**, or better, a repo-relative path / an env var
  with a repo-relative default (`ZM_BIN`, `SCORE_BIN`).
- **`just lint-scripts`** fails on a dead worktree ref or a binary with no
  source. Run it before committing a script that shells out.
- A missing artifact whose **source still exists** is just unbuilt — that is a
  `cargo build`, not a fossil. The linter distinguishes these; only 2 of the
  original 25 were genuinely dead (their target had no source anywhere).

Related failure from the same audit: `metric_compare_report.py` had not
**parsed** since a bulk sed (`731cf0eb`) inserted an unescaped
`<meta charset="utf-8">` into a Python string literal. A bulk edit across 293
files broke one and nothing caught it, because nothing ever asked "does this
still run?". `just lint-scripts` asks.

### The rule in practice

1. **Before writing a script that loads/trains/evals/bakes: check the table.**
   If an owner exists, use it. If it can't do the thing, go to 2.
2. **Extend the owner.** Add the flag/mode to the Rust binary, with a test.
   Prove you didn't break existing callers (build the binary at the parent
   commit, run an identical recipe, diff the bake bytes — that's how the
   `:withinref` change proved byte-identity, md5 `346c5a6d…` from both).
3. **A probe/experiment is not an exemption.** "It's just an experiment" is
   how all 134 scripts started. If the experiment needs a capability, the
   capability belongs in the owner; the experiment is then three lines of
   shell.
4. **Delete on sight.** A duplicate found is a duplicate removed, same commit
   if it's dead, next commit if something still calls it. Do not "queue for
   removal" — queueing IS the bug (ML Data Pipeline Discipline §6).

Companion rule in `~/work/zen/zenanalyze/CLAUDE.md` — the other trainer-owning
repo (4,347-line `zentrain/tools/train_hybrid.py`, `zenpicker_train.rs`, ~15
Python torch trainers). zenmetrics deliberately does NOT carry this rule: it is
the *supplier*, not the patient — it owns `zenstats` (which both siblings
consume), has zero MLP trainers, and already enforces single-source rules of
its own (`origin_split.py` hard-errors rather than allow a leaky fallback —
that hard-error is the enforcement pattern to copy).

Audit of record: `benchmarks/duplication_audit_2026-07-15.md`. Prior art:
`benchmarks/iqa_stats_consolidation_2026-05-26.md`,
`benchmarks/cross_repo_duplication_audit_2026-05-26.md`.

## Canonical bake / eval / training tool inventory (added 2026-05-17)

**When you need to do X, use this tool — don't write a new one.**

### Primary `zenpredict` CLI (bake / inspect / repack)
**`zenpredict` binary** at
`/home/lilith/work/zen/zenanalyze/zenpredict-bake/src/bin/zenpredict.rs`.
Build with `cargo build --release --bin zenpredict -p zenpredict-bake`.

The single canonical CLI with three subcommands:

```sh
# Convert BakeRequestJson to ZNPR v3 bin
zenpredict bake <input.json> <output.bin>

# Inspect a ZNPR v3 bake's structure + metadata + weight stats
zenpredict inspect <bake.bin>

# Re-bake an existing v3 with different dtype/compression
zenpredict repack <input.bin> <output.bin> \
    [--dtype f32|f16|i8] [--zerobias <tau>] [--compress] [--optimize]
```

`repack` preserves `feature_transforms`, `output_specs`,
`discrete_sets`, `sparse_overrides`, and all metadata entries.
Verified 2026-05-17 on V_22-IW v2 PreviewV0_5 (200,984 → 14,065 bytes,
7.0% of input, CID22 SROCC delta 0.0003).

The legacy `zenpredict-bake` and `zenpredict-inspect` binaries still
ship but are thin shims that call the same `cli::run_*` functions —
they're deprecated-in-favor-of subcommands. Per zenanalyze CLAUDE.md,
binaries are not part of the semver surface; future passes may remove
the legacy aliases.

**DO NOT USE** `zensim-bench/examples/quant_compare.rs` — it drops
metadata, causing catastrophic SROCC collapse (0.88 → 0.53 on the mix
champion). It is a diagnostic-only weight-magnitude reporter; for any
actual rebake, use `zenpredict repack`.

The JSON pipeline is still mandated for any new bake-producing tool
(per "JSON pipeline mandate" section below). See template at
`zensim/scripts/v_next/v0_20b/bake_znpr_v3.py`.

### STANDARD bake packing — QAT-native (2026-05-27)

**The trainer emits the packed + calibrated bake NATIVELY — no Python
post-step.** Use `--qat-fine-tune-epochs N` (recipe field `qat_fine_tune_epochs`)
+ `--out-dtype f16`: the last N epochs train quantization-aware (f16+zerobias
straight-through estimator), the post-training dial spline is fit on the
PROJECTED+QUANTIZED (shipped) net, and the 2-layer bake stores f16
(encoder) + compressed. One `zensim_mlp_train --manifest v47_strict_qat.toml`
pass → a ~27 KB bake, identity 97.7 (exact), 0 above-identity, correct dial.
VERIFIED 2026-05-27 (`benchmarks/qat_fine_tune_2026-05-27.md`): CID22 0.8657
(> the non-QAT recal 0.8564), Z-RMSE 0.512.

**Load-bearing rule (BOTH paths): fit the dial spline on the SHIPPED net —
projected (encoder≥0, rank_w≤0, α≡1) AND quantized.** Fitting on the
un-projected/f32 net inverts the pred↔target correlation → the spline picks
the wrong direction (blur scored UP to 2184) or identity drops (97.8→93.4).
QUANTIZE-then-CALIBRATE, on the projected net.

**QAT trade (intrinsic, not tau-tunable):** QAT improves CID22 + Z-RMSE but
regresses KonJND (0.485→0.418 — f16 removes the fine-weight precision PJND
discrimination needs; both fail G5's 0.70 floor regardless). So QAT is kept
OPT-IN (`qat_fine_tune_epochs` default 0): the codec-dial ship recipe opts in
(CID22 + native packing win); an HF/PJND-focused bake stays non-QAT.

**Non-QAT fallback** (existing f32 bakes, or HF-focused bakes that can't take
the KonJND trade): `bake_dial_refit pack --in IN.bin --out OUT.bin --neg-tail`
(defaults `--dtype f16 --zerobias-bulk 0.005`) — pack-then-calibrate as a
post-step (strip → zerobias+dtype → refit spline on packed → re-inject).
_(Was `scripts/v_next/pack_and_calibrate.py` — DELETED 2026-07-29 after the
Rust port reproduced the shipped `v47_strict_recal_negtail_packed30k` artifact
BYTE-IDENTICALLY, sha256 `302c9154…`, triple-matched vs a fresh Python run.)_

#### `bake_dial_refit pack` (non-QAT post-step) details

**Load-bearing rule: QUANTIZE, then CALIBRATE.** zerobias/f16/i8 preserve
RANK (signs intact) but SHIFT the network's raw outputs, so a spline fit on
the f32 net maps the PACKED net's identity output to the wrong dial value →
identity drops (97.8 → 93.4 observed). `bake_dial_refit pack` refits the
output spline ON THE PACKED network (strip → zerobias+dtype → refit spline on
packed → re-inject), which re-anchors identity exactly. SROCC is rank-invariant
under the monotone spline. This makes plain GLOBAL zerobias safe — `repack`'s
naive global `--zerobias` (calibrate-then-quantize order) drops identity; do
NOT use it for a spline-bearing bake.

Result on the per-sample-α arch: f32 198 KB → **30 KB**, identity 97.5 (exact),
CID22 0.8564 (≈ f32), 0 above-identity — 6.6× smaller, below the old 41–54 KB
convention, zero quality cost. Per-layer (`--protect-last`) is available but
USUALLY UNNECESSARY (refit recovers identity even with the last layer 98%
zerobiased). Full method + numbers: `benchmarks/standard_bake_packing_2026-05-27.md`.

The V39-era workflow regressed on packing (V39 ships raw F32 257 KB); re-pack
existing `zensim/weights/` F32 bakes through this path when rotating each
profile (SROCC-neutral by construction).

#### DEAD-COLUMN PRUNING — automatic, on by default (2026-08-04)

`pack` now drops layer-0 inputs that **cannot** change a prediction, in the
same pass as zerobias + dtype + spline refit. Order is **zerobias → PRUNE →
quantize → spline** (zerobias is what creates most dead columns; the spline
still lands last on the final packed net, so QUANTIZE-then-CALIBRATE holds).

**The caller's feature width never changes.** A pruned bake declares
`FeatureTransform::Drop` (zenpredict, landed 2026-08-04) on the dead raw lines,
so it still takes 944 features and internally forwards 667. Consequently
`Model::n_inputs()` (667) ≠ `Model::caller_input_width()` (944) — **size every
feature vector by `caller_input_width()`.** Mis-sizing fails loud
(`FeatureLenMismatch`); it never scores a prefix.

**Three classes of "dead", only two prunable** — the whole correctness story,
enforced in `zensim-validate/src/prune.rs` + `tests/prune_classes.rs`:

| class | test | prunable |
|---|---|---|
| 1 weight-dead | `W0[k,:]` exactly zero | **yes — BIT-identical** |
| 2 transform-forced-constant | the bake's OWN transform pins input `k` (winsor family, `lo >= hi`) | **yes** — contribution folded into `b0`; exact in real arithmetic, not bit-identical (the fold reorders one f32 sum) |
| 3 inert on a corpus | `bake_contrib` says mean\|Δ\|≈0 but the weight is live and no transform pins it | **NO** — the corpus merely never exercised it |

Class 3 is the trap: it is indistinguishable from class 1 in any corpus report
and is *not* mathematically dead. `prune::plan()` takes **no corpus statistic
as input**, which makes class 3 structurally unreachable rather than merely
discouraged. Class 2 is refused outright on an i8 layer 0 (removing a nonzero
row can move the per-output max-abs quantization scale).

**Identity gate runs on every pack.** Pre- vs post-prune scores over the anchor
corpus must be bit-identical when only class 1 fired, else within
`--prune-identity-tol` (default 1e-4). Fails loud and refuses to write.

Flags: `--no-prune` (off; restores byte-exact reproduction of pre-2026-08-04
bakes), `--no-prune-constants` (class 1 only ⇒ bit-identical for every input
including NaN), `--prune-identity-tol`.

MEASURED on the three sota944 ship candidates
(`benchmarks/dead_column_pruning_2026-08-04.md`): **944 → 667 layer-0 inputs,
all 277 class 1, identity gate bit-identical on 2035 anchor rows, verdicts
byte-identical.** File size barely moves (−382 B, 0.2%) — LZ4 was already
squeezing the zero rows — so **the win is inference and decompressed
footprint, not bytes**: 29.3% fewer layer-0 rows, a zenbench-measured
**−25.4% forward time** (71.6 → 53.4 ms / 256 rows, 95% CI [−29.6%, −19.1%];
4-round result on a busy box), and −73,128 B resident. The
`bake_contrib` "73 KB = 44% of the packed encoder" figure was a *decompressed*
measurement; do not quote it as a file-size saving.

**`--no-prune` is required to reproduce a historical bake byte-for-byte** —
verified: `pack --no-prune` on `C_em944_s31_dial.bin` reproduces the shipped
`C_em944_s31_packed.bin` sha256 `5870046d…` exactly.

### Bake evaluation (per-bake instant verdict from parquet sidecars)
**`bake_verdict` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/bake_verdict.rs`.
Build with `cargo build --release --bin bake_verdict -p zensim-validate`.

```sh
./target/release/bake_verdict --bake <bake.bin> \
    [--corpora cid22,kadid,tid,konjnd,aic3] \
    [--features-root /mnt/v/zen/zensim-training/2026-05-15-full-features] \
    [--output verdict.md]
```
Loads pre-extracted 372-feature parquets per validation corpus + bake
bytes, scores MLP via `Predictor::predict_transformed`, emits full
Mohammadi panel (SROCC + PLCC + KROCC + OR + PWRC + Z-RMSE) aggregate +
10-band per corpus. **~3.5 sec for all 5 corpora.** Replaces the older
`dataset_metric_baseline` which re-decodes images (~15-20 min per bake).

**Freeze-decision surface (2026-07-31):** `bake_verdict --corruption-head
<head.bin>` adds the companion corruption-head joint report (the shipping
design's corruption owner — the 924 dial's own ordering is broken by design;
dial-alone numbers kept for honesty; `corruption_head` block in
`--full-json`). **`freeze_check`** (same crate) turns one bake's fulleval
JSON into the freeze-bar PASS/FAIL table:
`freeze_check --fulleval <bake.fulleval.json> [--bar csiq=X --bar live=X]` —
externally-owned rows (UPIQ/Korshunov/perf/LOO/corruption-ORDERING) print as
explicit ATTACH rows, exit 1 on any FAIL; it computes NO stats, only compares
what the owning tools produced. Bars = zenpapers final-metric plan §5; owner
map = `benchmarks/decision_surface_audit_2026-07-31.md`. The Korshunov row's
owner (audit gap 3) is now **`scripts/external_reads/run_external_reads.py`**
— the committed seven-domain external-read runner (UPIQ hdr-dmean / SI-HDR /
HDR-VDC / AVT / CHUG / Rousselot / BANDVIS+CSFW LOO): `--from-stored`
rescores the stored feature tables in ~11 s and gate-checks the recorded
numbers (Korshunov 0.9346, Narwaria 0.7688, AVT pooled 0.7742, …);
`--scorer bake:<final.bin>` is the Phase-4 final-bake mode; as-run
provenance copies live in `scripts/external_reads/asrun/`. See its README.

### IQA statistical panel on arbitrary (predicted, target) pairs
**`panel` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/panel.rs`.
Build with `cargo build --release -p zensim-validate --bin panel`.

THE canonical entry point for the full Mohammadi 2025 panel (SROCC +
PLCC + KROCC + OR + PWRC + Z-RMSE + per-sample Z-RMSE + 4-param
logistic) on an arbitrary table — the NON-bake case. (For a bake on a
canonical corpus use `bake_verdict`.) Reads TSV or Parquet with columns
`predicted`, `target`, optional `sigma`, optional `band`:

```sh
panel --input scores.tsv [--json]            # aggregate panel
panel --input eval.parquet --json            # + per-band when `band` present
panel --batch jobs.tsv --stats srocc         # N (x,y) pairs -> N rows, ONE process
```

Wraps `zensim_validate::panel::{compute_panel, z_rmse_per_sample,
rescale_logistic}` directly — zero new stat math. Verified equivalent
to scipy to <= 1e-9 by `scripts/verify_panel_parity.py` +
`tests/panel_parity.rs`. Python pipelines that can't shell directly use
the thin `scripts/lib/zen_stats.py` shim (`from scripts.lib.zen_stats
import panel`). **Do NOT hand-roll srocc/plcc/krocc/pwrc/z_rmse in
Python** — that re-creates the 14-fork divergence this consolidates
(see `benchmarks/iqa_stats_consolidation_2026-05-26.md`).

**Batch mode (2026-07-31, audit gap 4):** `panel --batch <FILE|->`
takes a manifest of many (x, y) vector pairs — explicit rows, or
`#def`'d base vectors + index-set resamples (the paired-bootstrap
shape; the caller keeps the RNG) — and emits one TSV stat row per pair
in ONE process, so a 10k-resample bootstrap is a single invocation.
`--stats srocc` fast path; full mode adds `srocc_signed` (pre-abs
midrank) + `plcc_raw` (un-rescaled Pearson). Python:
`zen_stats.panel_batch` / `panel_batch_indexed`. Gate:
`scripts/verify_panel_batch_parity.py` (<=1e-12 vs scipy midrank incl.
tie-heavy; determinism) + `tests/panel_parity.rs --ignored`.
**scipy-in-a-bootstrap-loop is the banned pattern this replaces** —
`scripts/hdr/upiq_panel.py` is the migrated exemplar (byte-identical
recorded outputs, 3× faster).

### Bake training (MLP supervised learning)
**`zensim_mlp_train` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/zensim_mlp_train.rs`.
Supports `--group <name>:<path.parquet>:train_w:val_w` (auto-detects
.csv vs .parquet by extension), `--target-column NAME`,
`--feature-set extended_iw|extended|standard`, full PWRC+NiN flags,
auto-transforms via `--auto-transforms <screen.tsv>`.

### Eval matrix comparison (N bakes side-by-side)
**`scripts/cvvdp_matrix_compare.sh`** at
`/home/lilith/work/zen/zensim/scripts/cvvdp_matrix_compare.sh`. Runs
bake_verdict on every `*.bin` in a dir + emits a per-corpus
SROCC/Z-RMSE/PWRC table for ship-decision review.

### Bake dial / spline refit + tail gate (bake_dial_refit)
**`bake_dial_refit` binary** at
`/home/lilith/work/zen/zensim/zensim-validate/src/bin/bake_dial_refit.rs`.
Build with `cargo build --release -p zensim-validate --bin bake_dial_refit`.

THE canonical Rust home for editing a ZNPR bake's output-calibration
spline / feature-winsor guard — replaces the `scripts/v_next/*.py` that
hand-edited bake bytes in numpy (2026-07-05 migration). Reuses the shared
serializer (`zenpredict_bake::bake`), spline eval (`output_calibration_spline`),
and stats (`zenstats::panel`) — never re-serializes or re-implements PCHIP.
Subcommands:

```sh
# extend the spline TOP by the training-fitted concave saturation (THE
# shipped-B producer; reproduces b_sdr_linear_cid80_dense_dial BYTE-IDENTICALLY)
bake_dial_refit extend-top --in <winsor.bin> --out <out.bin> \
    --anchor <multiband_anchor.parquet> --target-col target_score
# whole-spline refit to a shared anchor (percentile-edge fit_spline_knots)
bake_dial_refit shared-anchor --in <bake> --out <out> --anchor <parquet> \
    --target-col <col> [--target-scale 100]
# prepend a (floor_raw, 0.0) bottom knot   (BYTE-IDENTICAL to Python)
bake_dial_refit bottom-extend --in <bake> --out <out> --floor-raw 0.0
# add 372 winsor_p99 guards from a fit corpus (functionally identical)
bake_dial_refit add-winsor --in <raw.bin> --out <out.bin> --fit-corpus <parquet>
# G-RANGE tail gate (below/above-knot raw-pred fraction) + Z-RMSE/OR/SROCC,
# NO PWRC (OOM-safe). The 3rd eval panel SROCC is blind to.
bake_dial_refit gate --bake <bin> --corpus <parquet> [--ref-col human_score]
# STANDARD non-QAT packing: per-layer zerobias + DEAD-COLUMN PRUNING + dtype,
# spline refit ON THE PACKED net. Pruning is ON by default (944 -> 667 on the
# sota944 bakes, bit-identical); --no-prune restores the pre-2026-08-04
# BYTE-IDENTICAL reproduction of pack_and_calibrate.py + the shipped packed30k
bake_dial_refit pack --in <f32.bin> --out <out.bin> [--neg-tail] \
    [--dtype f16] [--zerobias-bulk 0.005] [--protect-last] \
    [--no-prune] [--no-prune-constants] [--prune-identity-tol 1e-4]
# drop one metadata entry, rest verbatim (BYTE-IDENTICAL to the deleted
# strip_spline_metadata.py on both MLP and linear fixtures)
bake_dial_refit strip --in <bake> --out <out> [--key zentrain.output_calibration_spline]
# lasso-CD fit on a FROZEN gram npz + f16 pack + anchor spline + bake — THE
# shipped-BHdr producer, pure Rust (task #68; lasso w/bias/mu/sd f64 BIT-EXACT
# vs the Python fit via --parity-fit; whole file sha 7d7f2123… BYTE-IDENTICAL;
# reproduce_bhdr.sh now runs zero Python between fit and bake)
bake_dial_refit fit-lasso --gram <grams/hdr_v3mix.npz> --space shaped \
    --target human_score --lam 0.0003 --anchor <val/anchor.npz> \
    --transforms-tsv <screen.tsv> --out <bake.bin> [--parity-fit <fits/*.npz>] \
    [--tau 0] [--expect-sha256 <hex>]
```

Method + measured byte-parity: `benchmarks/bake_refit_rust_migration_2026-07-05.md`
(+ `benchmarks/pack_rust_migration_2026-07-29.md` for `pack`,
`benchmarks/key_bake_repro_verification_2026-07-29.md` for `fit-lasso`).
`fit-lasso` support modules: `zensim_validate::gram_lasso` (bit-exact
MixGram+lasso port, single-rounding f64→f16, CPython float-repr) and
`zensim_validate::npz` (minimal stored+deflate npz reader via zenflate).

### Affine calibration of an existing bake
**`affine_calibrate` binary** at `zensim-validate/src/bin/affine_calibrate.rs`.
Applies `y' = α + β·y` (`W' = β·W`, `b' = β·b + α`) to a ZNPR **v2 or v3** bake
— v2 and v3 share the layer-table layout for the first 96 header bytes, and
v3.1's reserved fields are zero for the F32 bakes we calibrate.

_(Corrected 2026-07-15: this section previously read "Missing v3 equivalent for
the affine op — build a v3 affine tool when needed". That was false; the Rust
port has existed since 2026-06-18. A stale "missing tool" claim is worse than
no claim — it tells the next session to rebuild something that ships, and it
excused `scripts/v_next/affine_calibrate_bake.py` as filling a gap that was
never open. Per the no-duplication rule, that script is a duplicate.)_

Output-**spline** / dial refits are NOT affine — those live in
`bake_dial_refit` above.

### Per-corpus baseline metric extraction
**Missing v3 equivalent.** Older `score_unified_with_bake.py` was
v2-only (DEPRECATED, refuses). Use `zenmetrics batch` (from
`/home/lilith/work/zen/zenmetrics/target/release/zenmetrics`) for
metric scoring on (ref, dist) pairs, then merge into per-corpus
parquet sidecars analogous to T11.7 safesyn CVVDP backfill.

### CVVDP scoring
**`zenmetrics`** at
`/home/lilith/work/zen/zenmetrics/target/release/zenmetrics`.
Build with `cargo build --release --bin zenmetrics --features 'gpu-cvvdp,gpu-cuda' -p zenmetrics-cli`.

```sh
zenmetrics batch --metric cvvdp --gpu-runtime cuda \
    --pairs <pairs.tsv> --output <scores.tsv>
```
Pairs TSV must have `ref_path` + `dist_path` columns. Note: rejects
16-bit RGB and 8-bit RGBA inputs (decoder widening pending). For TID's
`.BMP` images, convert to PNG first (see T11.10b notes).

### Migration tools
- **`zenanalyze/zentrain/tools/migrate_znpr_v2_to_v3.py`** — converts
  an old v2 bake to v3. Use this exactly once per archived bake; the
  trainer + bake_verdict + zenmetrics all produce v3 natively now.

### Deprecated / DO NOT USE
- `zensim-bench/examples/quant_compare.rs` — drops metadata, catastrophic SROCC loss.
- `dataset_metric_baseline` (zensim-bench example) — slow (15-20 min)
  AND silently drops KADID rows on image-decode failures. Use
  `bake_verdict` instead.
- **DELETED 2026-07-15** (superseded; git history preserves them —
  "kept for provenance" was redundant with version control, and a
  deprecated file left in tree is a file the next session copies):
  `dense_dial_refit_b.py` → `bake_dial_refit extend-top`,
  `bhdr_bottom_extend.py` → `bottom-extend`, `winsorize_bake.py` →
  `add-winsor`, `w11_webp_ood_refit_2026-07-05.py` (falsified campaign).
  The first three were proven **byte-identical** to their Rust
  replacements before deletion. Already deleted earlier:
  `affine_calibrate_znpr_v2.py`, `score_unified_with_bake.py`,
  `soft_iso_smooth.py` — this list claimed they were "deprecated but
  present" long after they were gone.
- **DELETED 2026-07-29**: `pack_and_calibrate.py` → `bake_dial_refit pack`
  — proven byte-identical THREE ways (fresh Python run == Rust ==
  the shipped `v47_strict_recal_negtail_packed30k_2026-05-27.bin`,
  sha256 `302c9154…`; `benchmarks/pack_rust_migration_2026-07-29.md`).
  Also deleted same day: `bake_outlier_gate.py` → `bake_dial_refit gate`
  (its one importer `xmetric_consensus.py` now shells the canonical
  `predict_features_with_bake` forward + `zen_stats.srocc` — smoke-verified
  on a kadis-gpu slice); `shared_anchor_refit.py` → `shared-anchor`
  (the claimed `hdr_anchor_dense_refit.py` importer was STALE — it imports
  `linear_projections`, the mention was docstring-only);
  `strip_spline_metadata.py` → `bake_dial_refit strip` (byte-identical on
  the v47 MLP `7c65814e…` AND shipped-B `5ec68b1f…`; live caller
  `recal_v47_dial.py` migrated); `bake_to_znpr.py` (DEAD: emitted banned
  v2, trainer gone); `affine_calibrate_bake.py` (duplicate of the Rust
  `affine_calibrate` bin per the affine section above).
- `hdr_anchor_dense_refit.py` is PARTIALLY migrated: its base whole-spline
  refit is `bake_dial_refit shared-anchor`; only the 28-bin densify + Q-Q
  top-end knots remain as experiment logic. Its bake primitives live in the
  Rust bin — don't resurrect the numpy PCHIP/serialize code.

## zenpredict crate dependency policy (added 2026-05-15)

**Use path or git refs to the local `zenanalyze/zenpredict` repo,
NEVER the published crates.io version.** zenpredict 0.1.0 on
crates.io is v2-only; v3 lives unpublished on the local sibling.
Pinning the published version would silently ship a runtime that
can't load any current bake.

### Default: path ref (sibling worktrees)

In the zensim workspace `Cargo.toml`:

```toml
[workspace.dependencies]
zenpredict = { path = "../zenanalyze/zenpredict" }
zenpredict-bake = { path = "../zenanalyze/zenpredict-bake" }
```

This works when the user's machine has both repos checked out as
siblings under `~/work/zen/` — which is the standard layout for
zen-org work. Path is preferred because it makes cross-repo edits
inspectable in `cargo build` output and avoids stale lockfiles.

### Fallback: git ref (CI, fresh clones)

For CI or environments without the sibling worktree, use git refs
pinned to a specific commit:

```toml
zenpredict = { git = "https://github.com/imazen/zenanalyze", rev = "<commit-sha>" }
zenpredict-bake = { git = "https://github.com/imazen/zenanalyze", rev = "<commit-sha>" }
```

Update the `rev` deliberately when a v3 feature lands that zensim
needs. Do NOT use a branch ref (`branch = "main"`) — that causes
silent breakage when zenanalyze's main moves.

### Audit

When adding a new zen-internal dependency (zencodec, zenresize,
etc.), check the workspace `Cargo.toml` for the right pattern. If
a sibling exists under `~/work/zen/`, use path. Never copy a
published-crate version from crates.io into a workspace dep.

## Shell scripting gotchas (added 2026-05-15)

### Bash readonly variables: GROUPS, PIPESTATUS, EUID, UID, ...

Assigning to these in a bash script may silently fail to take effect
— the result `$VAR` resolves to the builtin value, not yours. The
trap that bit a Phase 3 retrain script today:

```bash
GROUPS="--group safesyn:... --group kadid:..."   # silently overridden
zensim_mlp_train $GROUPS ...                     # bash sees $GROUPS = "1000"
# error: unexpected argument '1000' found
```

`$GROUPS` is bash's primary-group ID (e.g., `1000` on most Linux
boxes). Reading from it gives the readonly builtin; writing to it
in `bash` works in interactive sessions but is unreliable in scripts
(depends on `set -u`, shell mode, etc.).

**Avoid these names in scripts**: `BASH`, `BASHOPTS`, `BASHPID`,
`BASH_*`, `COMP_*`, `DIRSTACK`, `EUID`, `FUNCNAME`, `GROUPS`,
`HISTCMD`, `HOSTNAME`, `HOSTTYPE`, `LINENO`, `MACHTYPE`, `OSTYPE`,
`PIPESTATUS`, `PPID`, `RANDOM`, `SECONDS`, `SHELLOPTS`, `UID`.
Pick descriptive prefixed names instead (`DSET_GROUPS`, `TRAIN_GROUPS`,
`PIPE_STATUS`).

When debugging a script that produces unexpected positional args:

```bash
# This trick reveals readonly-builtin collisions:
GROUPS="hello"; echo "[$GROUPS]"   # might print "[1000]" not "[hello]"
```

If you see "unexpected argument 'NNNN' found" from a CLI tool and
NNNN is a small integer (often 1000, 65534, 0), suspect a readonly
collision before suspecting the CLI.

### `set -u` masks the readonly collision

With `set -u` on, writing to a readonly variable produces no error;
the read silently uses the readonly value. Without `set -u`, the
same script may still appear to work in some shells. Make the
diagnostic explicit by renaming.

## Release Process

`zensim` and `zensim-regress` are released **independently** with **separate semver**. A bump to zensim does not require a bump to zensim-regress, and vice versa. Tag format:

- `zensim-v0.2.0` for the zensim library crate
- `zensim-regress-v0.1.1` for the regression testing crate

`zensim-validate` is internal tooling — not published.

### Before any release

1. Run `cargo semver-checks` against the previous published version:
   ```bash
   cargo semver-checks --manifest-path zensim/Cargo.toml
   cargo semver-checks --manifest-path zensim-regress/Cargo.toml
   ```
   Fix any semver violations before bumping. If the API change is intentional, bump the appropriate semver component (minor for additions, major for breaking changes).

2. Run the full test suite: `cargo test --workspace`

3. Run clippy clean: `cargo clippy --workspace --all-targets`

4. Verify README.md is accurate — ask user to confirm before publishing.

### Release steps (per crate)

1. Bump version in `<crate>/Cargo.toml`
2. Run `cargo update -w` to update workspace lockfile
3. Run `cargo semver-checks --manifest-path <crate>/Cargo.toml`
4. Commit: `release: <crate> v<version>`
5. Tag: `git tag <crate>-v<version>`
6. Push tag: `git push origin <crate>-v<version>`
7. Publish: `cargo publish --manifest-path <crate>/Cargo.toml`

Never publish without a matching pushed tag. Never tag without passing semver-checks.

## Weight Training & Dataset Contamination

### dHash threshold (2026-05-14, after revert)

`check_holdout_overlap` uses dHash-64. The literature thresholds:

| Hamming distance | Label | Use for contamination? |
|---|---|---|
| d = 0      | identical (bit-perfect)                 | yes |
| d ≤ 5      | near-identical (recompression / resize) | yes |
| d ≤ 10     | "very likely the same image"            | **yes, but require user-eye verification** |
| d ≤ 16     | "possibly the same image" (screening)   | **NO** — too many false positives in our content domain |

**The d ≤ 16 default in `check_holdout_overlap.rs` is a screening
threshold for HUMAN review, NOT an automatic contamination cutoff.**
A 2026-05-14 cleanup based on d ≤ 16 produced a 149-basename blocklist
that user review proved was mostly false positives (UI screenshots
matching by flat-region dHash; "blue sky" overlap mistaken for content
overlap). The cleanup was REVERTED — see
`benchmarks/dhash_threshold_revert_2026-05-14.md`.

**Ship policy for any future contamination claim**:
1. Run `check_holdout_overlap --threshold 10`.
2. Build side-by-side montages for every flagged pair.
3. Get user sign-off entry by entry before adding to any blocklist.
4. Never auto-quarantine based on dHash alone.

### Safe synthetic dataset (V0_18 ship corpus)

- File: `/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv`
  (218,089 pairs; sha256
  `659982b3ce8d26184eca835a85f8d66c8550d945d659559c499b5670cd5d8589`).
- R2 mirror (2026-05-22):
  `s3://zentrain/synthetic-v2/training_safe_synthetic.csv`.
  Encoded distortion **bitstreams** under `/mnt/v/input/zensim/images/`
  (`<ref>/<codec_dir>/q<X>.{jpg,webp,avif,jxl}`) are mirrored to
  `s3://codec-corpus/synthetic-v2/<ref>/<codec_dir>/q<X>.<ext>` —
  **bitstreams only (~38 GiB, 729,703 objects, R2-verified 2026-06-22)**.
  The earlier claim of a `q<X>.png` mirror was WRONG: R2 never held the
  decoded PNGs.
- **⚠ 2026-06-22: the `q<X>.png` decode-cache was DELETED** (~402 GiB;
  `images/` went 440.80 GiB → bitstream-only ~38 GiB; freed because both
  `/mnt/v` and `/` were ~94-99% full). Each `q<X>.png` was a lossless
  decode of the adjacent `q<X>.<bitstream>` that `extract_features_372col`
  consumed via the CSV `decoded_path`. **So the CSV `decoded_path` PNGs no
  longer exist — re-extraction must DECODE the bitstream first** (unified
  zencodec API; reference: `zensim-bench/examples/verify_bitstream_decode.rs`,
  `--features verify-decode[,verify-avif,verify-jxl,verify-webp]`). The
  canonical feature parquets are unaffected (frozen + mirrored R2/Tower);
  only the regenerable cache is gone. Decoder-drift caveat measured
  2026-06-22: zencodec re-decode is **byte-exact** for the May-gen
  `zenjpeg-420-e1` run, but March-gen JPEG runs drift (zenjpeg decoder
  evolved: max_abs ≤ 5; XYB ≤ 42) and JXL differs (zencodec uses
  `zenjxl-decoder`; the generator used `jxl-oxide`) — so re-decoded pixels
  will NOT byte-match the canonical parquets for those codecs. If exactness
  matters, re-extract ALL corpora through one decoder rather than mixing.
- Tower mirror:
  `/mnt/tower/output/zensim-archive-2026-05-20/synthetic-v2-{tables,images}/`.
- Created from `training_concordant.csv` minus all 49 CID22 validation
  image sources.
- 475 CID22-contaminated pairs removed.
- Always use this CSV for V_X training; never `training_with_dssim.csv`
  or `training_concordant.csv`.
- Feature cache: `training_safe_synthetic.csv.features.*.bin`.
- Also valid: the 2026-05-12 post-CID22-purge variant at
  144,791 rows (artifact was under /tmp — wiped; re-derive if needed),
  produced after Phase-1 CID22 d ≤ 16 purge — **also** at the
  loose threshold; subject to the same false-positive caveat.

### Dataset contamination rules (2026-05-14, post-revert)

- **CID22**: 49 validation images. Original 2026-05-12 Phase-1 purge
  removed 361 sources at d ≤ 16 from CID22 refs. Those flags were at
  the loose threshold and need re-audit at d ≤ 10. CID22 ↔ KADID and
  CID22 ↔ TID cross-corpus audits at d ≤ 10 BOTH find **zero matches**
  — CID22 is perceptually disjoint from both holdouts.
- **KADIK10k**: Uses I01-I81 reference images. At d ≤ 10, **6 training
  sources** match KADID refs (4 `gmessages_*` variants near I18,
  `e7a01ec14bcca684_769x513.png` near I18 d=7, `2232979_512sq.png`
  near I25 d=10). Several of those are flat / UI / screen-content
  images where dHash is unreliable (large zero blocks dHash to zero
  regardless of content). User review pending in
  `/mnt/v/output/zensim/contamination_review_2026-05-14/d10_kadid_matches/`.
- **TID2013**: 25 reference images. At d ≤ 10, **1 training source**
  matches TID I12 (`b5cd470348ef0609_769x513.png` d=10). User review
  pending.
- **The file-name "no overlap" check is insufficient**. Hex-hashed
  training source names don't collide with KADID's I01..I81 or TID's
  I01..I25 namespace, but content can still overlap. Use
  `check_holdout_overlap` (dHash-64 at d ≤ 10) PLUS user review of
  side-by-side montages before declaring contamination.
- **Synthetic training sources**: Hex-hashed tiles from CLIC 2025 +
  CID22 collections.
- **dssim co-training is FALSIFIED** (cycle-7 verdict, commit
  `4ed499e`): all 5 dssim-weighted variants regressed CID22 by 0.04–0.07
  vs V0_16 baseline. Don't retry without a fundamentally different
  mechanism. The identified next lever for B0/B1 SROCC is direct
  JPEG-AI training-corpus acquisition (not started).
- **AIC-3 / AIC-4 are HOLDOUT-ONLY**. Never train on them.

### Contamination guard status

The `scrub_csv_or_die` runtime guard (in
`zensim-validate/src/contamination_guard.rs`) is still present but
its 149-basename embedded blocklist is **stale and over-aggressive**
(loose-threshold false positives). Don't rely on the guard's
embedded blocklist; regenerate it at d ≤ 10 + user verification
before reactivating as a ship gate.

### Available human datasets for training/evaluation
Three independent human datasets: **KADIK10k** (10,125 pairs), **CID22** (4,292 pairs), **TID2013** (3,000 pairs).
- Train on synthetic + 1-2 human sets, validate on remaining holdout(s)
- Use `--also type:path` and `--dataset-weights name:weight` flags
- Human datasets should be weighted to exceed synthetic (e.g., 1.0:2.0)

## KADIS-700k dataset (zensim 2026-06-30; GPU-metrics 2026-07-01)

700,000 distorted-image cells — 140k KADIS pristine references × 1 `dist_type_1` × 5 severity
levels, each with its 372-D zensim feature vector. **The zensim score and the 372-D `feat_*`
vectors are produced by THIS crate** (`Zensim::compute_extended_features`, `with-iw` regime);
the pure-CPU path (no GPU dep) is what made the cheap-fleet zensim sweep reliable, and the
GPU-metrics variant additionally runs zensim's GPU backend as `score_zensim_gpu`. Two canonical
variants (same 700k cells, same `source_id` split key):

- **★ GPU-metrics canonical (2026-07-01) — current, richest.**
  `s3://zentrain/kadis-700k-gpu/canonical/kadis700k_canonical_gpu_2026-07-01.parquet`
  (700k×387, ~936 MB zstd, 0 nulls; sha256 `c9a6fd56…`). **7 perceptual scores** —
  `score_{zensim,ssim2,butteraugli_max,butteraugli_pnorm3,iwssim,dssim}_gpu` + `score_cvvdp_cpu_imazen_v0_1_0`
  — plus `distorted_url` (a persisted distorted PNG per cell → rescore-from-links), on top of the
  372-D `feat_*` + shared keys. Sidecars `s3://zentrain/kadis-700k-gpu/{omni,zensim_features,pairs}/`
  + `distorted/<chunk>/*.png`.
- **zensim-only canonical (2026-06-30) — earlier variant.**
  `s3://zentrain/kadis-700k/canonical/kadis700k_canonical_2026-06-30.parquet` (700k×380, ~906 MB
  zstd, 0 nulls; sha256 `b57e4b3f…`). `score_zensim` + `feat_0..feat_371`. Sidecars
  `s3://zentrain/kadis-700k/{omni,zensim_features,source_features}/` (350 each).
- **Shared keys (both):** `source_id` (stable split key 0..139999 — split on this, never on row,
  for leak-free train/val/test), `source_filename`, `dist_type`, `dist_name`, `severity_level`,
  `dist_param` (signed for types 7/18/25 → U-shaped scores by design).
- **Mirrors:** `/mnt/v/datasets/kadis700k/canonical/`, `/mnt/tower/output/kadis700k/canonical/`.
- **Full README + schema:** `s3://zentrain/kadis-700k-gpu/README.md` + `s3://zentrain/kadis-700k/README.md`
  (and `~/work/kadis-distort/docs/DATASET.md`).
- **Credit:** reference images + distortion design © VQA Group, Universität Konstanz (Lin, Hosu,
  Saupe) — KADID-10k / KADIS-700k, https://database.mmsp-kn.de/kadid-10k-database.html ("freely
  available to the research community"). Cite KADID-10k (QoMEX 2019) + DeepFL-IQA (arXiv:2001.08113).
