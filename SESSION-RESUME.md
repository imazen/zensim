# SESSION-RESUME — read this first after every compact

**Last updated: 2026-09-05 (the floor-dense ladder instrument; see the ⚡ 2026-09-05 block below).** The current era is the
**SOTA-944 model campaign** — pre-registered, five seed/lever waves + two
ensemble waves, all appended in place in the one authority doc:
[`benchmarks/sota944_campaign_2026-08-03.md`](benchmarks/sota944_campaign_2026-08-03.md).
Everything before it (372-era, 720/924-era) is historical context, era-tagged —
never compare numbers across eras without the doc's era-bridge notes.

## ⚡ 2026-09-06 — the dial's C5/C6 can be made STRUCTURAL, and output polarity had no owner

Read [`benchmarks/best_of_all_2026-09-06.md`](benchmarks/best_of_all_2026-09-06.md);
plan + deviations [`docs/PLAN_BEST_OF_ALL_2026-09-06.md`](docs/PLAN_BEST_OF_ALL_2026-09-06.md);
ledger [`docs/DATASET_HISTORY.md`](docs/DATASET_HISTORY.md) §3.56.

- **`--nonneg-distance` makes C5 and C6 structural.** `raw(x) = pin − g(x)` with
  `g ≥ 0` and `g(0⃗) = 0` bit-exactly ⇒ `raw(0⃗)` is the argmax over the whole
  input space, by construction, in the SHIPPED wire format with **zero** runtime
  change. MEASURED on the 228-slot recipe against its own control: **C6
  1,642 → 0 WHILE `tied` goes 0.0017 → 0.0000** (the gate record's C2 ⊻ C6
  either/or is DISSOLVED, not traded), C5 38 → 0, grid max **exactly 100.000**.
  Cost at that seed: CID22 −0.0088, C1 0.94868 → 0.93040 against a 0.93 bar.
  **It does NOT buy A7r** — 5 of 5 still fail and the per-codec floors move in
  BOTH directions.
- **Output polarity had one owner at 1 of 8 loss sites**, which is why the
  fastclass2 campaign's best CID22 *ordering* (`|−0.8921|`) arrived negative and
  could not be splined. Reproduced at raw SROCC −0.9970/−0.9986 in a unit test;
  **4 of 5 of those tests FAIL at `0c6307a7`**. Nothing shipped moved — five
  rank-only bake sha256s reproduce byte-for-byte.
- Four flags stopped being silent no-ops (`--minibatch-size>1`/NiN with an
  absolute term; `--n-hidden-layers>=2`/`--skip-connection` off the α-head path),
  and `--leaky-alpha` stopped being a train/serve divergence.
- **Two chain traps:** `shared-anchor` asserts a SINGLE-LAYER linear bake (an MLP
  must concatenate anchors up front), and a **DENSIFIED bake cannot be graded by
  the pinned probes** (they are 372-wide; a densified 228-caller bake reads
  C3/C4/C5/C6 all NOT MEASURED). Score the PACKED bake.

## ⚡ 2026-09-05 — the FLOOR-DENSE ladder instrument: the shipped dial fails a jpeg floor it used to pass

Read [`benchmarks/ladder_instrument_2026-09-05.md`](benchmarks/ladder_instrument_2026-09-05.md);
plan (pre-registered) [`docs/PLAN_LADDER_INSTRUMENT_2026-09-05.md`](docs/PLAN_LADDER_INSTRUMENT_2026-09-05.md).
Grade on it: `ZL_ERA=ladder scripts/dialgate_arms.sh score <label> <bake> 372`
(944: add `ZL_GRID=<…/dial_grid_944col_ladder.parquet>` + `ZL_ROOT=<a 944 root>` and
pass regime `944`).

- **Every prior dial grid could not answer A7r's question for jpeg.** `zenjpeg` emits
  **ONE bitstream for every q in 0..10**, so q 0/5/10 is one setting sampled three
  times and the mentor's jpeg bar was a vacuous `0.0000`.
- The rebuilt grid is floor-dense and dedups **by encode hash** (`avif-svt` is 36.4 %
  duplicate settings vs `avif-rav1e`'s 3.0 % — no fixed step could serve both), with
  **five ladders incl. two AVIF backends** at 39 refs each. Registered append-only at
  both 372 (sha `4c3874a7…`) and 944 (`0e8e5fb7…`) with identical `peer_ssim2` bars.
- **⛔ Shipped `ZensimProfile::D` FAILS A7r here — on `jpeg`, by one ladder** (0.5128
  vs 0.5385) — plus A1 and A3. Profile B fails **all five** codecs. **Nothing was
  installed;** `zensim/weights/` untouched.
  **⚠ NAME THE RULE (clarified 2026-09-06).** That jpeg failure is under
  `--floor-rule distinct`. The OPERATIVE default is now **`resolvable` @ margin
  0.5** (`bake_verdict` calls `operative_floor_rule()` when `--floor-rule` is
  omitted), and under it shipped D passes **5 of 5**. Both readings are true of
  different rules; an A7r number without its rule is not a number. Grade both —
  `distinct` is what every published board cell used.
- **MEASURED: re-anchoring cannot fix it.** All 19 failing jpeg ladders are RAW
  (pre-spline) inversions, raw-vs-dial verdicts agree **39/39**, and the two shipped
  D bakes (same weights, different spline) have **identical A7r on all five codecs**.
  **The lever is the WEIGHTS.**
- Side-products: `imazen/jxl-encoder#101` (SizeHeader rounds odd dims UP to even at
  distance **>= 10.0** exactly), the `zenav1-svt` pin at `2d75a105f` (**measured
  1.498x**, 9/9 byte-identical — not the "2x" it was described as), and a fixed
  silent fallback where `dialgate_arms.sh` ignored `ZL_GRID` for non-372 regimes.

## Current true state (2026-08-04)

## ⚡ 2026-08-27 — production-readiness close-out: all executable lanes closed or ruled; the board holds the open decisions

**The DONE test** (`docs/GOAL_PRODUCTION_READINESS_2026-08-25.md`) now carries a
terminal-state **evidence appendix**; the plan doc's LOOPS table + critical path
are current. Decision Board artifact (D1–D5) = the live user-decision surface.

**LOOPS — all 7 lines closed or ruled**: jxl ✅ (+RD full pass vs independent
judges `f7c95cbe`+`6fc24060`); zenwebp ✅ (buckets = the one-shot); zenjpeg ✅
(inert zq head +47.9%/+45.9% — wiring = D3); zenavif(+rav1e) ✅ (q0_head);
**zenav1-svt ✅ NEW** — `svtav1-target` crate, HDR census (blind k2 17.64 → S1
seeds {t70:22, t80:13, t88:5} k2 3.306/k3 1.513, svt `c6701dcc`); **gainmap ✅
census CLOSED, CEILING-BOUND** (no config crosses t70 on any scene; Ultra HDR =
low/mid-fidelity; multi-channel falsified + ultrahdr encode bug fixed
`971ad8d4`); zenav1-aom premature-ruled (its own gates mid-flight, re-checked
08-27 at `d2c0ded`).

**MODELS/HDR**: HDR-944 wave complete — winner `HDR944_L1T1_s4005` (packaged,
sha `d6203e9d`), runner-up T2_s4004; freeze = D2. Gauntlet now shows per-bake
**training dates + recipes** (`6d25dfee`).

**imazen-26 audit — root-sourced (the user caught the wrong estate copy)**:
canonical = the `imazen/imazen-26` repo manifest + png-v3; `o_NNNN` = the
4-digit id; `/mnt/v/imazen-26*` = quarantined inspo, NEVER use (memory:
`feedback_imazen26_canonical_copy`). **Provenance is the deriving owner**
(`imazen-26/scripts/derive_sharing_provenance.py`), dHash the verifier.
Sharing: 68 generator-token ids + 166 split-piercing family ids; **realized
eval inflation MEASURED ≈0** at both tiers (upper bound excludes 25–32% of
rows: median Δ +0.0043, max 0.0143, nonphoto deltas positive). Exclusion = D1,
low-stakes; annotation `imazen26-nonphoto-sharing-provenance-2026-08-27` in
force. `imazen26_manifest.tsv` header + split column corrected (`77de3ccb`).
Record: `benchmarks/imazen26_dhash_audit_2026-08-27.md`.

**RESOLVED 2026-08-28**: D1 EXECUTED (family re-slice) · D2 BOTH FROZEN — SDR candidate-of-record W10L9PH_s4004_packed (61ebc456), HDR HDR944_L1T1_s4005_hfpack (0a437d99); shipped default stays B · D3 wired (zenjpeg/jxl/svt) · D4 era-B. Balance campaign: `benchmarks/balance_campaign_2026-08-28.md`.
wiring (zenjpeg zq / jxl B3 / svt qp_start) · D4 judge-era unification · D5
history rewrite. Other-lane: zenav1-aom differential gates. Shas of record
today: zensim `6d25dfee`→`77de3ccb`, imazen-26 `946cb61`+`a7bea19`, zenmetrics
`31795bf6`, svt `c6701dcc`.

## ⚡ 2026-08-26 — LAN-store day (updated 17:15Z): HDR chain CLOSED same-day, loops complete, 3 browser pools

**Store**: fleet writes ONLY the LAN SeaweedFS; avifgen-enc blobs migrated
(542,483 verified). R2 receives zero writes (deletion user-gated).

**Same-day HDR chain (C2+C3)**: hdrfeat944 fleet run (~8 h, auto-paused at
3.8-6.7× rescore tax) → per-arm 944 feature writebacks → `hdrgrid944-leg`
(orientation-gated, LAN+Tower) → linear floor 0.7609 → **registered wave-1:
G-W1/W2 PASS, best `wave1_h64_s3` val 0.8130** (372-route best 0.8163; MLP
closes the folded front-end's −0.050 linear gap to −0.0033). Remaining C3-HDR:
HDR fulleval instrumentation + the HDR-append feature wave.

**Loops (C4) COMPLETE for turnkey codecs**: zenjpeg Zq seed SHIPPED (−13.5%
encodes, 0 regressions); jxl Zq FAILED-as-registered (5.6%<15%; nonphoto −51%
= future lever, head env-gated OFF); **gainmap loop shipped**
(`ultrahdr-rs/target_quality.rs`, 9/9); zenwebp no-headroom; svt/aom premature
(no encode API) — *superseded 08-27: svt loop+census+S1 CLOSED, gainmap census CLOSED; see below*.

**Browser (C6)**: three pools render-gated — default SDR-07-01, `?set=hdrgrid`,
`?set=avifgen`.

**Fleet (C1)**: diffmap's 46,680 gm failures root-caused (GPU image lacked
hdr-gainmap) → image `exec-gpu-812a00d9` → ZERO post-swap failures; ledger
readers tolerate in-flight chunks; reconcile --auto-pause self-heals. avifgen
DATA closed (scores+features+encodes, mirrored, manifested).

**Infra lessons banked (memory)**: git-object corruption repair (`--refetch`),
jj detached-HEAD orphan trap, per-regime batched writers, OOM = run-heavy
always.

**Open user decisions**: r7900x power (hard-off, 2×WoL failed); zentrain 3-tier
plan (`zenmetrics docs/R2_ZENTRAIN_TRIAGE_2026-08-26.md`); board size cap
(14.5MB vs 12MB); lilith (WSL, RTX 5070) power.

---

- **★ 2026-08-05: Profile `C` SHIPPED (user-gated)** — the wave-11
  battery-selected `W10L9_s4003_packed` (k=8-confirmed corrected-mix recipe,
  appendix K.R) is `ZensimProfile::C` (`zensim-c`), weight
  `zensim/weights/c_sdr_mlp944_corrmix_2026-08-05.bin` (sha `1a2c8d52…`,
  first PRUNED shipped bake, caller 944 / internal 667). **`B` remains the
  default** — C-vs-B is a stated trade (C: CID22/LIVE/CSIQ/nonphoto/dial-mono/
  M3a + corruption head; B: KonJND + HF-NL). No crates.io publish (separately
  gated). Repro + provenance: `docs/PROFILE_C_REPRODUCTION_2026-08-05.md`;
  distribution `s3://zentrain/profiles/C-2026-08-05/` + Tower.
- **Regime = 944** (folded+append+append2). Canonical data roots + grids are
  resolved by `bake_verdict --regime 944` itself (test-pinned; see entry
  point 1). REGIME PURITY is absolute: never column-mix 944 rows with
  720/372 parquets.
- **Campaign standing result:** the frozen 5-row bar (CID22 > 0.8924, KonJND
  ≥ 0.43, nonphoto ≥ 0.90, HF-NL-proxy, dial) has been cleared row-by-row but
  **never by one artifact**. Registered levers (seed scale n=23, near-top
  anchor, coherence, wave-4 combos) were honest nulls; the stabilized
  single-model ceiling is **`C_co3a_s1301` CID22 0.89067** (KonJND 0.405,
  nonphoto 0.905, HF-NL +0.251, dial 95.9%/0%).
- **Seed ENSEMBLES moved both blockers:** wave 5 `W5_E1_k2` CID22 **0.89425**
  (first CID22-bar pass in 64+ draws; paired bootstrap P(Δ>0)=0.968), wave 6
  `W6_GE2_trio` KonJND **0.4543** (the KonJND blocker broken; binding row is
  now CID22). Ensembles are evaluation functions, not shippable bakes — M3a
  not computable for them; distillation (wave-6 arm F) is the ship route and
  is **in flight** as of this writing.
- **Freeze decision = the USER'S, pending:** stabilized ~0.891 with better
  secondary axes vs. the unstable 924-era 0.8924 peak (EM4 — which fails the
  campaign's own HF-NL row; see the doc's Corrections section). Freeze bars +
  owner map: `freeze_check` (zensim-validate) +
  `benchmarks/decision_surface_audit_2026-07-31.md`.
- **G-RANGE on 944 MLPs** (2026-08-04 addendum): the gate tool now evaluates
  every bake class, and it surfaced that no 944 MLP candidate carries an
  output spline — dial packaging (`bake_dial_refit add-spline`) is required
  before that bar row can be judged on a freeze candidate.
- **Selection frame = the BALANCED profile (user-directed, 2026-08-04):** the
  user lowered the bar to surface candidates balanced across bands, datasets
  and uses — registered as `freeze_check --profile balanced-2026-08-04`
  (campaign doc AMENDMENT 8 + its RESULTS section: floors, composite, classes,
  full pass matrix, frontier, trade cards; §1 stays the freeze bar). Headline:
  **0/172 board cells pass all 8 floors** — classic-IQA breadth (CSIQ/LIVE ≥
  0.83) is the 944 era's binding balance axis; wave-7's `H_co3abpg_s2507` is
  the frontier-top single (kon 0.459 ∧ breadth ∧ nonphoto ∧ M3a 0.866 GOLD,
  missing only CID22 by 0.0045), and the packaging pass showed all raw-unit
  dial-mono numbers are unit-flattered (no packaged cell holds ≥93% in dial
  units — see the doc's unit caveat).

## ⇒ 2026-08-25: LAN era + the refinement plan (read after this file)

- **Fleet + storage are LAN-local** (user directive 2026-08-08; store =
  SeaweedFS on tower, `ZEN_STORE` defaults to LAN since 08-10, buckets keep
  their R2 names, R2 = cold/user-gated rundown). The operator cheat-sheet,
  the discrepancy list (production enroll script still R2-pinned; `fleet
  status` reads 0 on LAN — use `pool_progress.py`) and the program order live in
  [`docs/PLAN_LAN_ERA_REFINEMENT_2026-08-25.md`](docs/PLAN_LAN_ERA_REFINEMENT_2026-08-25.md).
  **The DONE test for the whole program** (user directive 2026-08-25, <4k chars):
  [`docs/GOAL_PRODUCTION_READINESS_2026-08-25.md`](docs/GOAL_PRODUCTION_READINESS_2026-08-25.md).
- **Wave-12 (appendix AD, 2026-08-21) is pre-registered, data-gate OPEN, and
  was never launched** — no `W12_s*` bake exists. It is the first compute to
  start (plan §4 B1; amend AD.7 by measurement — the box is 60 GiB again).
- **HDR phase-2 corpus (appendix S)**: encode drained 99.888 % on 08-07, score
  waves declared on R2 and never harvested, orientation gate PENDING
  (`/mnt/v/output/hdrgrid-2026-08-06/_MANIFEST.json`). Plan §4 B4.
- **jxl loop has NO secant controller** (power law exp 1.0 / clamp 2.0, k2
  18/27 · k3 24/27); the secant/bracket/per-tile arms are plan §5. **AV1
  steering** moves to the λ-side (rdmult) channel across zenrav1e /
  zenav1-svt / zenav1-aom behind one harness — plan §6.

## ⇒ 2026-08-25 GOAL RUN progress (docs/GOAL_PRODUCTION_READINESS_2026-08-25.md)

Per-criterion state (committed shas; `git -C <repo> merge-base --is-ancestor <sha> origin/<branch>` to verify):
- **C3 MODELS (SDR): CONVERGED.** Wave-12 (avif944 leg) run to completion — Profile C stands 8/8;
  best full-weight seed 7/8 (F1 CID22). The registered half-weight (w=0.5522) follow-up found the
  first 8/8 wave-12 seed (`W12hw_s4203`: CID22 0.8881>C, AVIF-dial 0.9673, M3a 0.8836) — a LATERAL
  trade with C (worse CSIQ/KonJND/nonphoto), G-AC3 not met, C still selected. avif944 adopted as a
  standing mix leg @ w≈0.55. zensim `AD.R`/`AD.R.1` (`bf15ed9e`,`1ad91786`). HDR models: NOT started
  (needs the HDR corpus, criterion 2).
- **C4 LOOPS: all 4 main image codecs OWN their SDR target loop (per-codec-ownership directive).**
  jxl `vardct/zensim_loop.rs` (+`JXL_ZENSIM_SECANT`, `1ed4ee72`); zenavif `target_quality.rs`
  (bracketed secant/bisection + `q0_head` zenpredict seed — EXEMPLAR); zenwebp `encoder/zensim_target.rs`
  (VERIFIED 2026-08-26: real one-pair secant + anchor seed + per-segment diffmap overrides, most advanced);
  **zenjpeg NEW `target_quality.rs` (`277b1efb` on origin/main) — `search_target` + `encode_with_target`,
  injected-scorer (zensim deps zenjpeg ⇒ cycle ⇒ MUST inject), 9 unit tests + a REAL-CODEC production
  gate (`79935f20`, tests/target_quality_real.rs, feat=zencodec: encode→decode→fast_ssim2, MEASURED
  6/6 target convergence, k2 1/6 k3 2/6 — a Zq seed would cut iters).** Dep-cycle finding
  + per-codec table: plan "CRITERION-4 STATUS". **Zq AUTOTUNE — my `ee7ab1f3` fit RETRACTED as LEAKAGE (distorted features); the VALID non-leaky formulation (source-features + zq_norm) ALREADY EXISTS/prototyped in `picker_zenjpeg_A_sourcefeat_v3` (2026-05-27, no q-leakage per its .toml); production follow-on = dense sweep + q_start head + current-ref source-feature extraction + wire, benchmarks/zq_autotune_zenjpeg_2026-08-26.md):**
  a feature+target→q ridge on each codec's bigcodec-924 view cuts q-prediction error 59-78% vs a
  target-only anchor on held-out TEST (zenavif 76%/±10q 96%, zenwebp 78%/93%, zenjpeg 64%/74%, zenjxl
  59%/74%) — criterion-4's zenpredict-baked Zq one-shot predictor, the MODEL. FOLLOW-ON (mechanical,
  feature-gated per [[feedback_no_zenpredict_in_codecs]]): MLP via zensim_mlp_train → zenpredict-bake →
  wire behind each codec's auto-tune feature (cheap 8-feature q0_head design for inference; zenjpeg CAN
  dep zenanalyze, no cycle). Other TODO: gainmap loop (HDR, needs HDR-zensim); production gates
  (census/dial-mono/RD-under-independent-judge/perf); svt/aom loops (cloned; but they're low-level AV1 algorithm ports with no turnkey encode-at-quality API yet — loop is premature, not blocked on cloning);
  Program-D per-encoder λ-side steering. `zensim-target` (`7e17945e`) is the shared-algo reference.
- **C5 PERF (x86 SIMD): CLOSED.** Survey: jxl+zenrav1e already dual-arch; only zenavif `unpremultiply8`
  was NEON-only → AVX2 tier shipped (bit-identical + ~3.3-3.6×). zenavif `b92880e3`, zensim `9afa10f8`,
  `benchmarks/simd_x86_gap_survey_2026-08-25.md`.
- **C2 DATA: HDR corpus scoring LIVE (2026-08-26).** imazen-26 ID audit CLEAN (`78b60142`). HDR:
  98,805 hdrgrid encode blobs salvaged R2→local; the score jobs were ALREADY DECLARED on R2 (resume,
  not re-declare) — `hdrgrid-sf-gpu` (ssim2-gpu+iwssim-gpu, hdr:true) was 0/0, now RUNNING. First-cell
  GATE PASSED (693-pair chunk, 0 errors); scaled to **2 GPU boxes**: .27 (GTX 1060 6GB, r7900x≡lianli ssh-aliased) on **sf ssim2/iwssim** +
  **node-2/i134 (RTX 3070 8GB, .148) on sf2 butteraugli** — parallel, different metrics, no lease
  contention; sequencers via `lan_gpu_sequence.sh`. 3 CPU boxes (i265+r5900xt+tower) on `sf-cpu`
  features. sf/sf2-small DONE (r5900xt's 2GB DID do butteraugli-small — 687/687). NOTE: ssim2-gpu
  failed 15/453 HDR images on sf-huge (3.3% tail, investigate). Producing (GPU ~10 chunks + CPU 1263 in ~20 min). Recipe + resume:
  plan "HDR CAMPAIGN — EXECUTION LOG". TODO: drain → reassign 6GB boxes to `-huge` → `sf2`/butteraugli →
  `writeback_scores.py` → `_MANIFEST`+orientation gate+Tower mirror → HDR model wave; dHash+eye follow-up; curation.
- **C7 DOCS:** jxl comment + zenmetrics SeaweedFS doc-truth fixed; plan/survey/campaign records current.
- **C1 FLEET: 4 boxes BUSY (2026-08-26).** A1 enroll LAN fix landed; A3 cred distribution DONE (R2 creds
  pushed to r7900x/lianli/r5900xt/i265 — authorized). ssim2+butteraugli GPU-only (`ZEN_REQUIRE_GPU=1`, proven on the ONE GTX 1060 6GB .27; the
  GTX 1050 2GB is too small→CPU); cvvdp/zensim/features on CPU (i265). Drain/stall monitor armed
  (`/home/lilith/tmp/hdr-fleet-monitor.sh`). Note: the vast GPU-score launchers are cloud-only; LAN
  scoring uses the direct-manifest docker worker (recipe in plan). **C6 BROWSER:** located, not extended.
  **C8 ZENPICKER:** blocked on C4 autotune/gates.

Next: HDR GPU buckets drain → reassign to `-huge` + run `sf2`/butteraugli + harvest; then gainmap loop /
zenpredict Zq autotune. Repo gotchas: zenmetrics=`master`; verify pushes with `origin/<branch>` (git) not
`main@origin` (jj); after `bookmark set main -r @`+push verify `@-` (push auto-advances @ to an empty child).

## THE three entry points (a newcomer starts here)

1. **Evaluate any bake, correctly, with one command:**
   `bake_verdict --bake X.bin --regime 944` — resolves the ext944 features
   root, 944 dial/corruption grids, kadis-944 per-pair source, and the frozen
   12-corpus campaign list; a bare run cannot silently omit a corpus. Add
   `--fulleval out.json` for the schema-complete dashboard JSON, or run
   `scripts/run_full_eval.sh <bake> <name> 944` to also measure M3/M3a.
   (`scripts/sota944_verdict.sh` is the campaign's thin wrapper over the same
   preset.)
2. **See every model compared:** the summer-gauntlet board at
   `/mnt/v/output/zensim/reports/fulleval/summer_gauntlet.html` (rebuild:
   `scripts/v_next/bandwise_dashboard.py --fulleval-dir …/fulleval`; every
   regen must pass `scripts/v_next/gauntlet_gates.sh <html>`).
3. **Understand the science:** the campaign doc above (bar, arms, corrections,
   ensemble waves, addenda) + [`docs/TOP_MODELS_COOKBOOK.md`](docs/TOP_MODELS_COOKBOOK.md)
   (the 372-era roster + pitfall list — era banner at top).

## Reading order on resume

1. **This file** (~1 min)
2. **[`benchmarks/sota944_campaign_2026-08-03.md`](benchmarks/sota944_campaign_2026-08-03.md)** —
   the era authority: frozen bar, every wave's results, corrections, addenda.
3. [`docs/TOP_MODELS_COOKBOOK.md`](docs/TOP_MODELS_COOKBOOK.md) — validated
   science + exact reproduction of the (372-era) top models + the pitfall list.
4. [`docs/MODEL_SELECTION_SCORECARD.md`](docs/MODEL_SELECTION_SCORECARD.md) —
   the five-gate exam (RANK/DIAL/STEER/RD/TARGET) every ship candidate takes.
5. `CLAUDE.md` — rules + methodology (★924-parquets, ★E-M campaign, the
   NO-DUPLICATE-IMPLEMENTATIONS owner table, tool inventory).
6. [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) — how a number chains
   back to bytes; [`docs/DATA_SPLITS.md`](docs/DATA_SPLITS.md) +
   [`docs/DATASET_HISTORY.md`](docs/DATASET_HISTORY.md) — corpus law.
7. [`benchmarks/INDEX.md`](benchmarks/INDEX.md) → prior experiments; run
   `TaskList` for open work.

## Doc pointers (updated 2026-08-04)

- Campaign era: `benchmarks/sota944_campaign_2026-08-03.md` (authority) ·
  plan `docs/PLAN_SOTA944_CAMPAIGN_2026-08-01.md` · B lineage
  `benchmarks/profile_b_methodology_2026-07-12.md` · era bridge + backfill
  `benchmarks/backfill944_2026-08-01.md` + `backfill944_bigcodec_2026-08-02.md`.
- Shipped defaults: `ZensimProfile::B` (SDR) + BHdr; the 372-era candidates
  and their scorecards live in the cookbook; swaps remain user-gated.
- External reads / HDR domains: `scripts/external_reads/README.md`
  (seven-domain runner, `--from-stored` rescores in ~11 s).
- Eval-panel law: `docs/EVAL_PANEL_REQUIREMENT.md` (rank+dial two-panel
  mandate) · `docs/FULL_EVAL.md` (fulleval schema + 924/944-era eval slices).
- Historical: `docs/HISTORY-2026-05-v0x-era.md` (V0_x era) ·
  `benchmarks/best_per_day_summer_2026.md` (per-day 372-era champions) ·
  `zenanalyze/everything.md` is frozen-HISTORICAL (its own banner).

(CONTEXT-HANDOFF files are banned; durable facts live in the docs above. The
IQA literature corpus is `~/work/zen/zenpapers` — search it before designing
features or metrics.)

**⚡ 2026-08-28 addendum — user decisions EXECUTED:** family-aware purity
program done (slices family-pure + 280-row board rescore-graft; SDR
instrument clean; HDR instrument: hdr_v3mix carries 7/9 census scenes —
annotated, freeze gates clean; purge list committed). D3 wirings LIVE:
zenjpeg `seeded_for_image` (`37e44fda`), jxl `s4_eps` B3 default
(`7c4ddd65`), svt `TargetOptions::seeded` (`cb400901`). D4 = stay era-B
(decided). D2 freeze re-proposed post-purity. zensim `de1e340e`. jj lesson
banked: verify pushes by MESSAGE, never bare sha
([[feedback_jj_sideways_push_clobber]] addendum).

**⚡ 2026-08-28 later — HDR retrain wave + peers-everywhere:** D2 evolved:
retrain wave ran (user hold call, premise later corrected — candidates were
census-clean all along; hdr_v3mix overlap = BHdr the judge). L0 (HF-anchored
re-pack) fixed the user's HF-addressability requirement (d1.0→d0.1 jxl band
reachable, p50 93.9 vs old 81); 6 HF-weighted retrains all pass G-HF+G-EXT
but flip sihdr; **E.4 and the GATE scorecard DISAGREE** (L0: floors/sel_comp;
t2_s4003: g1 1.0/weighted_goal/dial reach) — recommendation withdrawn, both
lenses recorded, D2 open. **Peers COMPLETE: ssim2/butter/cvvdp/iwssim × all
12 board rank axes** (stored refmetrics + identity-gated lineage sidecar
join + fresh scoring: cvvdp CPU local, GPU trio on wsl 5070 via the baked
cuda13 container; LIVE via PNG mirror; /mnt/v NOT shared dev↔wsl —
sharecheck-verified, data ships with jobs). Dial/corruption-grid peer curves
scoring in flight from persisted pixels. Board 304 rows; 'sprint bests'
preset; training-date column. Wave md: hdr944_retrain_wave_2026-08-28.md.
