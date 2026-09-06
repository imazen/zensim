# `benchmarks/` — INDEX

> **⚠ BANNER 2026-07-18:** this index STOPS at 2026-05-15 — its 'ship a new bake' reading order is obsolete (points at v0_18 methodology). For 2026-06/07 work (five-gate scorecard, RD probe, diffmap coherence, additive-vs-MLP correction, HDR steer screen) start from `docs/TOP_MODELS_COOKBOOK.md` or `ls benchmarks/*2026-0[67]*.md`.

> **★ RETROSPECTIVE 2026-07-26:** [`best_per_day_summer_2026.md`](best_per_day_summer_2026.md) — the **best model per calendar day** (2026-05-01 → 07-25), with verified bake paths, recipes, headline metrics, and the summer champions (best CID22 = winner_dial 0.894; best KonJND = cl_tfm 0.761; best HF-NL/dial = Ebothg_scr0.5_dial; shipped B/A/BHdr). Machine-readable twin: `/mnt/v/output/zensim/reports/best_per_day.json`. The one-stop map of every model-experiment day this summer.

> **★ 2026-09-05 — DEFAULT PROPOSALS:** [`default_proposals_2026-09-05.md`](default_proposals_2026-09-05.md)
> — the SDR + HDR default answer under the 2026-09-05 G-ADDR ruling (`resolvable` floor window
> operative, `A1`-`A6` report-only). **SDR: keep Profile D** (the only scorer passing both tiers on
> the FLOOR-DENSE ladder; all 97 re-graded board cells fail `A7r`). **HDR: keep `BHdr`** (UPIQ 0.7536,
> above ssim2-PU and every HDR944 arm; G-ADDR on HDR is NOT MEASURED). Grading changes + the
> reversibility proof: [`dial_addressability_gate_2026-09-04.md`](dial_addressability_gate_2026-09-04.md) §17.

> **★★ R6b — THE F17 ARM IS `SaturatingExcess`, AND BOUNDING LIFTS LIVE BY +0.214 2026-09-06:**
> [`feature_rev2_2026-09-05.md`](feature_rev2_2026-09-05.md) §11 — R6 §9 reported an unbounded
> feature that is NOT F4 and did not fix it. Registered as **F17** and decided here: five arms of
> `contrast_inc` = `max(0, var_dst/var_src - 1)` from ONE binary over **216,756 rows**, fitted at
> 2 slices x 2 solvers = 20 bakes, graded pre-registered
> ([`../docs/PLAN_FEATURE_REV2_2026-09-05.md`](../docs/PLAN_FEATURE_REV2_2026-09-05.md) §11, pushed
> at `e09f6e9a` before a table existed). **Its twelve slots are the TOP TWELVE of all 372 by maximum**
> (36,465.7) **and the thirteenth is 1.972** — a partition, x105,127 the gold holdout's own p99.9 —
> while its two siblings max at exactly 1.000000, because their numerators their own denominators
> bound and the gain member's does not. Unlike F4 it fires on five distortion corpora and the training
> leg. **`REV2_HFGAIN = SaturatingExcess` (`g/(g+1)`)**, the only arm passing H3+H4+H5: `bexcess`
> **263,195 order inversions** (it reads the MAGNITUDE, so it replaces the statistic instead of
> bounding it — the plan predicted the opposite and the measurement corrected it), `cap` **67,224 new
> ties** (F4's `Clamp` analogue, free there and not here), `log1p` unbounded at **10.504**. **LIVE
> 0.7357 -> 0.9500 (+0.214)**, TID +0.033, KADID +0.021, CID22 +0.0027..+0.0090 CI-excluding — and
> **KonJND regresses -0.013..-0.080 on EVERY bounded arm**, so that cost is a property of bounding,
> not of the arm. `FormulaRevision::Rev2` now batches THREE eras (`v1ssimcap` + `freecomp` +
> `v1hfgain`); F17's twelve slots are the same at **every** pool state, unlike F4's 132-vs-36.
> **`SHIPPED_REVISION` stays `Rev1`.** Two things the fleet needs: a rev2 flip served to an
> un-refitted Profile D costs |SROCC| <= 6e-5 and <=0.64 % of pairs past the 0.5-pt dial bar (~4
> orders below an era shift), and **a FOURTH hand-copy lives in `zenmetrics`
> (`zensim-gpu/src/pipeline.rs:1305-1310`)** — land it there first or pin the oracle to the CPU walk.

> **★★ R6 — THE F4 ARM IS `Clamp`, AND F4 NEVER FIRES ON REAL PIXELS 2026-09-05:**
> [`f4_arm_decision_2026-09-05.md`](f4_arm_decision_2026-09-05.md) — four arms of the per-pixel SSIM
> luminance term extracted from ONE binary over **217,756 rows** (7 human corpora + the full
> 196,086-row safesyn leg), fitted through the shipped Profile-D recipe at 3 slices x 2 solvers = 24
> bakes, graded pre-registered ([`../docs/PLAN_FEATURE_REV2_2026-09-05.md`](../docs/PLAN_FEATURE_REV2_2026-09-05.md)
> §7, pushed at `090d55d7` before a table existed). **`clamp` is a pathology DETECTOR** — it differs
> from the shipped form only where `(mu1-mu2)^2 > 1` — and it **fires on NOTHING**: 0 cells moved, no
> slot above `|f| > 2`, against the 5,814,302 on record (which belongs to the bigcodec sweep, no local
> pixels). So `clamp` is BIT-IDENTICAL to rev1 through features -> Gram -> solve -> spline -> ZNPR bytes
> (all six bakes sha-for-sha) and its rank delta is exactly 0. `c1` wins a 2-of-3 CI-excluding majority
> in **one of six** variants (+0.00016 CID22) while moving **29.4 M healthy cells** (worst 0.771 vs a
> 1e-4 bar); `lorentz` moves 24.0 M. Rule 4 selects `clamp` — the prediction the plan stated in advance.
> **`SHIPPED_REVISION` stays `Rev1`.** Two corrections found on the way: F4's blast radius keys on
> **pool state, not width** (`ext944`/`ext924` zero `f156..371`, the 2026-09-05 pools-live ladder grid is
> 98.7 % nonzero, so F4 reaches 36 slots on one and 132 on the other), and the **"winsor already clamps
> it" mitigation covers Profile B ONLY** — Profile D, the SDR default, carries no `feature_transforms`
> and no `feature_bounds` at all. **⛔ And the unbounded feature that DOES fire is `contrast_inc`, not
> F4**: 36,465.7 on safesyn, 3,598 on LIVE, 122 of 779 LIVE rows above 100 — `hf_energy_gain =
> max(0, hf_dst_L2/hf_src_L2 - 1)`, unbounded by exactly F4's flat-source mechanism, with no registered
> defect. Reported, not fixed.

Methodology docs, falsification logs, sweep outputs, perf
benchmarks, and bake binaries. 76 markdown files as of 2026-05-16,
organized by theme + chronology. Each entry is one line:
file path + one-line summary.

If you arrived here cold, start with [RESEARCH.md](../RESEARCH.md)
first — this index is a deep dive for follow-up.

> **★ D-VS-SSIM2 INVERSION CENSUS 2026-09-05:** [`d_inversions_2026-09-05.md`](d_inversions_2026-09-05.md) — the board's `d_id100_negrich@did100lane` dial block is on the CANONICAL `..._quarantined_v2.parquet` grid, so none of its 24 counted inversions can sit on one of the 9 dropped w11/GPU-odd-dim ladders or the 33 dropped pre-fix JXL cells (structurally absent from the file). Full codec×zone census, D vs `peer_ssim2`, on BOTH the board grid and the new floor-dense ladder instrument (cross-validated Python port of `bake_verdict`'s classifier, 0 mismatches vs D's own `--full-json` on every counter, both grids — needed because peer mode has no per-ladder JSON at all). **Board grid: D inverts at-or-above ssim2 in 11/12 codec×zone cells.** **New instrument: the picture flips at the floor** — ssim2 itself inverts MORE than D at q<50 on 5 of 5 codecs (the mentor's own RD curve is 67-78% genuinely non-monotonic there, per the floor-resolution doc). Coincidence (same ladder-zone inverted by both): 46% (board grid) → **74%** (new instrument) — mostly shared encoder non-monotonicity, not D-specific; one recurring exception (`090d19695a8b43c2_512sq`) found across both grids. Top-10 worst material inversions by magnitude: **9 of 10 independently confirmed by ssim2 at the same step**, usually with a larger drop, at flat bytes — visual page (full-frame + native 1:1 crop per step) at `http://192.168.50.44:3300/zensim/ladder-2026-09-05/inversions/index.html`. Also flagged (worked around at the time) a `zensim-bench` build break — **FIXED same day, §6 addendum**: not the `zenjxl` pin itself (already `^0.4.0` on `origin/main` six days earlier; the reporting checkout was just stale), but a since-stale git-rev stopgap in `zensim-bench/Cargo.toml`'s own patch table, retired back to a plain path patch once both sides agreed on `0.4.0`.

> **★ IS THE LADDER FLOOR REAL OR NOISE? 2026-09-05:** [`ladder_floor_resolution_2026-09-05.md`](ladder_floor_resolution_2026-09-05.md) — report-only follow-up to the ladder instrument below. Classifies every one of the mentor's own bottom-triplet A7r failures: jpeg **14/18 (78%)** and `avif-rav1e` **12/18 (67%)** are GENUINE inversions (median 1.29/1.43 ssim2 points at +0.6%/+0.8% bytes), not ties — the floor is mostly real, not noise. But shipped D's specific one-ladder jpeg miss IS a boundary artifact: re-grading `A7r` on a window of steps the mentor can actually resolve (≥0.5 pt apart) puts D at **5/5 codecs**, jpeg exact parity with the mentor; a window forced to fixed separation (+2/+5 ssim2 pts) also cures jpeg but **newly fails `avif-rav1e`** (0.9487 vs mentor 0.9744) — a gap the noisy pinned rule was hiding. A1's 99.99996372-vs-100.0 gap is traced to 9 cells, all `avif-rav1e` @ q=99.9, verified pixel-identical to reference with an exactly all-zero 372-feature vector — a deterministic spline-at-identity constant, not an extraction artifact (repeat run bit-identical to every digit). `minus_f162` (which fixed jxl's A7r on the OLDER canonical/preC/postC instrument) does **not** generalize to this denser grid. Nothing installed, no registry write.

> **★★ THE FLOOR-DENSE LADDER INSTRUMENT 2026-09-05:** [`ladder_instrument_2026-09-05.md`](ladder_instrument_2026-09-05.md) — **every dial grid before this one asked A7r's question of jpeg and could not answer it.** MEASURED: `zenjpeg` emits **ONE bitstream for every q in 0..10** (identical bytes AND identical ssim2, on every reference), so the old grids' bottom three jpeg steps — q 0/5/10 — are one setting sampled three times, and the mentor's jpeg bar was a **vacuous 0.0000** that anything passes. Rebuilt floor-dense (66 q steps, 0..30 step 1) over the 39 canonical references with saturated steps removed **by encode hash** — never a per-codec step table, because `avif-svt` is **36.4 %** duplicate settings against `avif-rav1e`'s **3.0 %** on the same axis — and **five** ladders including **two AVIF backends** at 39 refs each (the canonical grid has avif 35 / jpeg 22 / jxl 33 / webp 16). **The result: shipped Profile D, a clean A7r pass on every older grid, FAILS on jpeg by one ladder** (0.5128 vs 0.5385), plus A1 (99.99996372 vs ssim2's exact 100.0) and A3; Profile B fails **all five** codecs with a POSITIVE `dial_min` on each. **Nothing installed.** And the fix cannot be calibration: all 19 failing jpeg ladders are inversions in the RAW pre-spline model (raw-vs-dial verdicts agree **39/39**), and the two shipped D bakes — same weights, different spline — have **identical A7r on all five codecs** while `avif-rav1e` `dial_min` moves −13.49 → −59.81. **The lever is the WEIGHTS.** Side-products: `imazen/jxl-encoder#101` (SizeHeader rounded UP to even at butteraugli distance **>= 10.0 exactly** — 9.9 is fine — so 513x769 declares 514x770; encoder-side, read from the codestream's own header; pre-existing, and diagnosable with **no re-encode** only because this run persisted encoded bytes), the `zenav1-svt` pin at `2d75a105f` (**MEASURED 1.498x**, 9/9 cells byte-identical — *not* the "2x" it was described as), and two process lessons that are invisible in output data: never edit a shell script while bash is executing it, and `nohup` inside a backgrounded shell gets killed with its task's process group (use `setsid`).

> **★★ PROFILE D — THE GATING TAX, AND A W4 EXCEPTION THAT WASN'T 2026-09-01/02:** [`profile_d_notax_2026-09-01.md`](profile_d_notax_2026-09-01.md) — removes the **gating tax** (`feature-regime-v2` is now a default feature; `cargo semver-checks` clean, zero public API delta, new parity gate) and the one real **tier duplication** in the touched kernels (raw moments: 10 hand-duplicated per-tier sites → 2 generic helpers + 1 scalar pair; `dense_block_kernel` is ERA-LOCKED and untouched). No feature slots minted. **The measurement is the interesting half.** The first 162-cell both-tier sweep was CORRUPT and the doc says so in §4.3: a `zenbench` `compare` group given too little `max_wall_time` for its size **degenerates to a spuriously near-zero mean for every arm at once** — caught by cross-checking `fast_ssim2`, an arm whose cost cannot depend on zensim's thread count, reading 1.58 ms against 690–724 ms in adjacent rows. Three 2304² cells had **0-of-9, 1-of-9 and 5-of-9** valid starts. Independently, this lane's own `nice`-d concurrent builds contaminated the same tier — `nice` lowers priority, it does not isolate from a `taskset`-pinned sweep — swinging `fast_ssim2` **128.9–633.6 ms inside one cell**. **⇒ `min()` over process starts is only safe against noise that is one-directional; a harness that can report LOW defeats it, and `min()` will select the corrupted reading as "the best one".** Now a standing warning in `CLAUDE.md`. **§4.4 is the clean re-measurement**: per-size wall budget (8/15/**60**), a collection-time plausibility filter, an idle box, n=9 valid starts × 18 cells × both tiers, **0 corrupt reads and 0 retries in 54 invocations** — and **every cell passes the 1.25× W4 bar at 1T and 8T**, worst ratio 1.189×. **The a4bkon lane's recorded 1.44–1.46× FAIL at 1152²/8T does NOT reproduce** (1.143× `v4x`, 1.026× `v3`), and the contamination cuts both ways: this lane's own spurious 2.037× at 1152²/1T resolves to 1.079×. Recorded as a dated **APPENDIX C ADDENDUM** in the ssim2-bar doc — that verdict is NOT edited in place, and "nothing passes W1–W7" still stands (W1/KonJND remains the binding failure).

> **★★ PROFILE D SHIPS + THE PUBLISHED-CRATE SPEED CHECK 2026-09-01:** [`profile_d_and_published_speed_2026-09-01.md`](profile_d_and_published_speed_2026-09-01.md) — **closes the W7 gap the ssim2 bar and HYBRID docs both flagged** ("today's best 156-class candidate needs `custom-profiles`... no profile slot, no `from_block_profile`"): `ZensimProfile::D` now ships (behind default-on `candidate-profiles`) carrying ADD156 arm-A (spline-top-extended, rank-exact), and `ComputeSet::from_block_profile` (internal) derives its compute set automatically. Measures BOTH the default-build and `feature-regime-v2` forms rather than assuming either, and separately answers the user's "did we regress vs the crates.io-published profile A, I remember a 10" — a same-process, ASLR-controlled bench of the published `PreviewV0_2`/`A` against current main, `B`, `D`, and `fast-ssim2` as an opponent row, across 576²/1152²/2304² × 1/8/16T.

> **★★ THE ssim2 BAR 2026-08-31:** [`ssim2_replacement_bar_2026-08-31.md`](ssim2_replacement_bar_2026-08-31.md) — **the first head-to-head against SSIMULACRA2 in the project's history**, as ONE registered exam (W1–W7) with every threshold derived from a measured noise floor. Answers the user's "have we made real progress" per mission axis. Headlines: **SPEED WON** — zensim is faster at every size and thread count (576²/1T: fast-ssim2 21.7 ms vs the 944 walk 18.3 / the 372 fold 9.4 / the basic fold 7.4; 6.5× at 8T), and nobody had ever measured it. **RANK a TIE** — reference-clustered paired bootstrap on CID22 (49 refs, B=10,000): both 944 flagships tie ssim2 pooled *and* within-image, **shipped B is measurably WORSE within-image (−0.0079, CI excludes 0)**, ADD156 worse on both; CSIQ is the one strict win (+0.047). **DIAL split** — `W10L9P_s4005` beats ssim2 (mono 0.9947 vs 0.9930, 6 % vs 14 % of near-lossless ladders inverting) while B and ADD156 are the only arms that END ladders backwards. **HDR** — shipped `BHdr` beats ssim2's integrated-PU path 0.7536 vs 0.7044; the frozen HDR candidate-of-record loses to both. Plus the gates-and-goals audit (every gate measures us against ourselves; `peer_ssim2` scores 0.8979 on our own `balanced_composite`, above every model, by arithmetic) and the 10-row premise ledger incl. the traced 944 cost story. New instruments: `bake_verdict --dial-peer-scores`, `panel --per-group`, the `fast_ssim2` bench arm.

> **★★ THE HYBRID + the SPEED CLASS BAR 2026-09-01:** [`hybrid_candidate_2026-09-01.md`](hybrid_candidate_2026-09-01.md) — the ssim2 exam's **W4 amended twice** (both user directives): APPENDIX B adds the **8-thread** count, then **AMENDMENT B2** replaces the opponent bar with a **CLASS bar — ≤ 1.25× the 156-walk class at BOTH 1 and 8 T**, `fast-ssim2` kept as context. The 1.25 is derived: below ~1.10× is inside the bar arm's own ASLR spread, and the measured classes are 372 → 1.55–1.85×, 944 → 2.06–2.68×, so it cuts in the gap. **MEASURED 1 T** (per-start ratio, 5 ASLR starts, 6 arms interleaved in ONE process): bar **6.90 / 25.70 / 107.20 ms**, `zensim_B` 1.55–1.85×, `flagship_944off` 2.06–2.16×, `q7b_944pools` 2.42–2.58×, `hybrid_944pools` 2.57–2.68×, `fast_ssim2` 2.78–3.40×. **No 944 model can pass** — the gap is the extraction — so both flagships, `Q7b` and every 944 hybrid become **teachers/upper bounds**. The ensemble's SECOND forward costs **1–6 % of one compare**. **PART I:** a convex blend `w·W10L9PH + (1−w)·Q7b` **PASSES W1, which NEITHER parent does**, over a measured window **w ∈ [0.76, 0.86]** bounded below by LIVE and above by KonJND; **KonJND is SUPER-ADDITIVE** (parents 0.5006 / 0.5118, blend **0.5390**, ssim2 0.5272); the blend cuts Q7b's deepest q≥85 backwards step **91.3 → 30.4** and removes the flagship's tied rate. **W2 still fails, structurally** — the named near-lossless win survives to w ≈ 0.4–0.6 and W1 starts at 0.76, so the feasible regions are **disjoint**. **PART II — distillation into the 156 set:** the control did not reproduce `ADD156`, and every explanation was tested — clip +0.007, λ +0.06 and plateaus, solver a trade, **fit substrate NOTHING**, and **the TRAINING LEG +0.057, which reproduces the incumbent exactly** (196k canonical safesyn → CID22 0.8643 / KonJND 0.5406 at 31 coefficients). Over nine λ the teacher target beats the human one on CID22 (8/9, median +0.017) and CSIQ (+0.022) and **costs KonJND monotonically (−0.008 → −0.115)**. **The leg is ~7× the teacher and free of the KonJND cost** — the priced ask for the era-2/radius-4 wave. `SADD_BIGLEG` (156-class, 4,117 B) **ties `ADD156` and gains no clause** — its one clear edge is **KonJND 0.5432, above `peer_ssim2`'s 0.5272**; on the regime-matched grid both fail W3 (a first draft's W3-gain claim was a cross-grid comparison and is corrected in the doc). New/fixed instruments: `bake_verdict --ensemble-weights`, `bake_dial_refit predict --ensemble-weights` **plus a pruned-bake width bug** (it was prefixing a 944-row to 667 columns), `gram --max-feat`, `fit-lasso --anchor-prefix`, the four amended-W4 bench arms, and **LIVE + AIC-4 added to the paired bootstrap** (the exclusion was ROOT-scoped: 372 root max |Δ| 1.12, both 944 roots 0.0).

> **★ HF HUMAN DATA 2026-09-01:** [`hfhuman_2026-09-01.md`](hfhuman_2026-09-01.md) — **the JPEG-AIC study family, ingested as eval axes.** AIC-3 BTC/IPTC, AIC-4/PTC and JPEG-AI-SDR25 are **ONE study on TEN source images**, and three board axes already read it (`aic3`, `aic4`, and `sdr25` = `bake_verdict`'s selection comparator). **Registered split rule `jpeg-aic-family-holdout-2026-09-01`: HOLDOUT-ONLY family-wide, membership by CONTENT — no training leg exists or should be built** (training on any member contaminates all three axes at once and buys ≤900 rows on 10 images; the alternative partition is priced and rejected). What it DOES give is the exam's missing instrument: a non-circular human near-lossless axis at **n = 515,250 forced choices** (vs `hfnl_cid22band`'s 1,425). **VERDICT** (native reading): ssim2 acc 0.7302 against a majority-oracle **ceiling of 0.7346**; Q7b/W10L9P/ADD156/W10L9PH all **TIE**, shipped **B −0.0025 [−0.0037, −0.0014] BEHIND** — the CID22 verdict again, on 88× the judgments. **AND the corpus supplies its own robustness test**: BTC stimuli are *boosted* (2× magnified, ~1.8× amplified — measured), so `btc_displayed` and `btc_native` serve the IDENTICAL responses and differ only in the pixels — **14 of 36 verdicts FLIP**. What survives: **nobody strictly beats ssim2 under both readings** (zero WIN→WIN cells); **shipped B is BEHIND under both**; **Q7b never loses** on the pooled or cross-codec axis under either; **the 944 pair is the most reading-sensitive arm** (cross-codec −0.0020 → −0.0276). The one strict win — W10L9PH *and* W10L9P on the JPEG-AI leg, +0.0012 [+0.0002,+0.0025], two seeds — **reverses to a significant loss on the other reading**, and is reported as arm-dependent rather than claimed. Also: **`aic3`'s target is a DESIGN value** (`−0.25 × level` exactly on 600/600 rows, 121 interpolated) and **`sdr25` is a 50-row sub-slice of `aic4` with a different reconstruction**. Overlap audit clean (min d=12 over 1,221 refs, eye-verified). New statistic `zensim_validate::pairwise` + `panel --pairwise`. Artifacts: [`hfhuman_2026-09-01.pointer.md`](hfhuman_2026-09-01.pointer.md).

> **★ FAILURE PROFILES 2026-08-31:** [`failure_profiles_2026-08-31.md`](failure_profiles_2026-08-31.md) — **where each model HURTS**, not how it ranks. The statistic→production-situation mapping; the board-wide ladder-inversion measurement (`dial.zones`: inversions by codec × quality zone × content class, plus the worst ladders by reference image name, 322 of 379 cells); and the board's new *Failure profile* panel. Headlines: `q>=85` is the failing band (**189 of 322 models** carry a ladder that ends backwards there); avif is the worst codec and webp nearly clean; shipped **B** ranks **20.8 %** of `hf_nearlossless` references backwards where ADD156 ranks 0 %, and two board cells invert ~86 % of references while publishing positive pooled SROCCs. Artifacts: [`failure_profiles_2026-08-31.pointer.md`](failure_profiles_2026-08-31.pointer.md).

## Headline reference cards

These are the docs to read first for a given purpose:

| Purpose | Doc |
|---|---|
| **Canonical codec-target metric** (2026-05-24) | [`tuner_v10_cross_codec_baseline_2026-05-24.md`](tuner_v10_cross_codec_baseline_2026-05-24.md) + [`../docs/CODEC_TARGET_METRIC.md`](../docs/CODEC_TARGET_METRIC.md) |
| **Tuner v11 in-flight retrain** (task #6) | [`v_tuner_v11_methodology_2026-05-24.md`](v_tuner_v11_methodology_2026-05-24.md) |
| **Most recent full-stat panel comparison across all bakes** | [`v0_20_all_bakes_stat_comparison_2026-05-15.md`](v0_20_all_bakes_stat_comparison_2026-05-15.md) |
| **Currently-shipped bake methodology** | [`v0_18_methodology_2026-05-13.md`](v0_18_methodology_2026-05-13.md) + [`v0_18_ship_reference_card_2026-05-14.md`](v0_18_ship_reference_card_2026-05-14.md) |
| **Runtime cost of extended + IW features** | [`extended_iw_runtime_perf_optimized_2026-05-15.md`](extended_iw_runtime_perf_optimized_2026-05-15.md) |
| **Falsification re-evaluation triage + results (today's methodology shift)** | [`falsification_reeval_triage_2026-05-15.md`](falsification_reeval_triage_2026-05-15.md), [`falsification_reeval_results_2026-05-15.md`](falsification_reeval_results_2026-05-15.md) |
| **Mohammadi 2025 stat-panel reproduction proof** | [`mohammadi_2025_verification_2026-05-14.md`](mohammadi_2025_verification_2026-05-14.md) |

## V_X bake methodology + ship decisions (chronological)

### Pre-V_18 (early experiments)

- [`v04_3norm_baseline_2026-05-01.md`](v04_3norm_baseline_2026-05-01.md) — V_4 butter-3norm baseline.
- [`v04_3norm_compat_2026-05-01.md`](v04_3norm_compat_2026-05-01.md), [`v04_ssim2_holdout_baseline_2026-05-01.md`](v04_ssim2_holdout_baseline_2026-05-01.md), [`v04_ssim2_holdout_calibration_2026-05-01.md`](v04_ssim2_holdout_calibration_2026-05-01.md), [`v04_ssim2_holdout_compat_2026-05-01.md`](v04_ssim2_holdout_compat_2026-05-01.md), [`v04_to_ssim2_anchors_2026-05-01.md`](v04_to_ssim2_anchors_2026-05-01.md), [`v04_calibrate_mapping_2026-05-01.md`](v04_calibrate_mapping_2026-05-01.md) — V_4 ssim2-target compatibility audit.
- [`profile_compat_v02_v04_2026-05-01.md`](profile_compat_v02_v04_2026-05-01.md) — V_2 (linear) vs V_4 (MLP) per-pair compatibility.
- [`baseline_metrics_with_konjnd_2026-05-01.md`](baseline_metrics_with_konjnd_2026-05-01.md) — KonJND-1k anchor alongside KADID/TID/CID22 for the first time.
- [`v0_6_eval_2026-05-11.md`](v0_6_eval_2026-05-11.md) — V_6 eval.

### V_18 ship lineage (2026-05-12 → 2026-05-14)

- [`v0_18_methodology_2026-05-13.md`](v0_18_methodology_2026-05-13.md) — **The shipped V_18 methodology doc.** Read this for the canonical recipe (3-way concat of base + cycle-14 s1 + cycle-14 s42, h=128, α/β calibration).
- [`v0_18_ship_reference_card_2026-05-14.md`](v0_18_ship_reference_card_2026-05-14.md) — Quick reference: V_18 ship's per-corpus + per-band numbers.
- [`v0_18_zerobiased_lz4_10band_2026-05-14.md`](v0_18_zerobiased_lz4_10band_2026-05-14.md) — V_18 zerobiased + LZ4 ship-form 10-band SROCC reproducing the F32 numbers.
- [`v0_18_repro_and_cross_corpus_analysis_REVERTED_2026-05-14.md`](v0_18_repro_and_cross_corpus_analysis_REVERTED_2026-05-14.md) — REVERTED: cross-corpus contamination audit reverted (loose d≤16 dHash false-positives).
- [`v0_18_1_full218k_noship_2026-05-14.md`](v0_18_1_full218k_noship_2026-05-14.md) — V_18.1: full 218k retrain. **No-ship** (−0.011 CID22).

### V_19 (KADID/TID-purge — reverted)

- [`v0_19_methodology_initial_failure_REVERTED_2026-05-14.md`](v0_19_methodology_initial_failure_REVERTED_2026-05-14.md) — V_19 retrain on KADID/TID-purged synth. **REVERTED** (CID22 −0.015).
- [`v0_19_REVERTED_2026-05-14.md`](v0_19_REVERTED_2026-05-14.md) — V_19 final REVERTED note.
- [`v0_19_smaller_retrain_2026-05-13.md`](v0_19_smaller_retrain_2026-05-13.md), [`v0_19_10band_2026-05-14.md`](v0_19_10band_2026-05-14.md) — V_19 component evals.

### V_20 cycle (input-shaping, IW-SSIM exploration, 2026-05-14 → 2026-05-15)

- [`v0_20_v0_21_design_2026-05-14.md`](v0_20_v0_21_design_2026-05-14.md) — V_20/V_21 design doc.
- [`v0_20_input_shaping_methodology_2026-05-15.md`](v0_20_input_shaping_methodology_2026-05-15.md) — **V_20 IS methodology**. 98 transforms (winsor_p99 / signed_cbrt / signed_sqrt / quantile_bins / log1p).
- [`v0_20_three_directions_summary_2026-05-15.md`](v0_20_three_directions_summary_2026-05-15.md) — D1 (concat) / D2 (ensemble) / D3 (tighter transforms) summary.
- [`v0_20_d2_ensemble_v18+is_2026-05-15.md`](v0_20_d2_ensemble_v18+is_2026-05-15.md) — **D2: V_18 ship + V_20 IS multi-bake ensemble** (live as `PreviewV0_4`).
- [`v0_20_4_runtime_mix_sweep_2026-05-15.md`](v0_20_4_runtime_mix_sweep_2026-05-15.md) — Runtime mix α sweep for the D2 ensemble.
- [`v0_20_l0_norms_2026-05-15.md`](v0_20_l0_norms_2026-05-15.md) — **GD-selection analysis**. Confirms IW + masked features ARE selected by gradient descent (Pearson correlation < 0.85 between weight maps).
- [`v0_20_extended_falsification_2026-05-15.md`](v0_20_extended_falsification_2026-05-15.md) — 300-feat extended bake. Falsified on CID22 SROCC (but per the new methodology, that gate is suspect — see `falsification_reeval_*`).
- [`v0_20_low_n_band_analysis_2026-05-15.md`](v0_20_low_n_band_analysis_2026-05-15.md) — Low-n band ceiling analysis. Empirical SROCC max per cell when n<100.

### V_20a IW-SSIM weighted pooling (Wang & Li 2011)

- [`v0_20a_iwssim_design_2026-05-14.md`](v0_20a_iwssim_design_2026-05-14.md) — Design doc for IW pooling.
- [`v0_20a_iw_perf_2026-05-14.md`](v0_20a_iw_perf_2026-05-14.md) — Per-pair compute cost of the IW pool.
- [`v0_20a_smoke_methodology_2026-05-14.md`](v0_20a_smoke_methodology_2026-05-14.md), [`v0_20a_sweep_methodology_2026-05-14.md`](v0_20a_sweep_methodology_2026-05-14.md) — Sweep methodology.
- [`v0_20a_path_a_falsification_2026-05-14.md`](v0_20a_path_a_falsification_2026-05-14.md) — **V_20a falsification doc** (CID22 catastrophic, TID/KADID wins — see "Methodology shift" in [`falsification_reeval_results_2026-05-15.md`](falsification_reeval_results_2026-05-15.md) for re-eval).
- [`v0_20a_ship_form_verification_2026-05-14.md`](v0_20a_ship_form_verification_2026-05-14.md) — Hypothetical ship form (not landed).

### V_20b distortion-manifold pre-training (Su 2023)

- [`v0_20b_distortion_manifold_design_2026-05-15.md`](v0_20b_distortion_manifold_design_2026-05-15.md) — Design doc. Falsified for CID22 transfer; KADID + TID win.

### V_20d JND-anchored output calibration (queued)

- [`v0_20d_jnd_calibration_design_2026-05-14.md`](v0_20d_jnd_calibration_design_2026-05-14.md) — Design doc. Not yet trained.

### Falsification re-evaluation (2026-05-15 methodology shift)

- [`falsification_reeval_triage_2026-05-15.md`](falsification_reeval_triage_2026-05-15.md) — **Catalogue of every SROCC-gated falsification**, classified by re-eval priority.
- [`falsification_reeval_results_2026-05-15.md`](falsification_reeval_results_2026-05-15.md) — **Full-panel re-eval of 8 falsified bakes**. All 8 confirmed against original SROCC-only verdict — but with the caveat that all bakes were trained against ssim2 targets, so the gate is structurally rigged toward V_18.

### Cycle outcomes (recovery cycle, 2026-05-11 → 2026-05-13)

- [`cycle_summary_2026-05-11.md`](cycle_summary_2026-05-11.md) — Overall cycle summary.
- [`cycle_6_finals_2026-05-12.md`](cycle_6_finals_2026-05-12.md) — Cycle 6 finals.
- [`cycle_7_dssim_outcomes_2026-05-12.md`](cycle_7_dssim_outcomes_2026-05-12.md) — Cycle 7 dssim co-training **falsified** (V_24/V_27 variants).
- [`cycle_8_konjnd_pareto_outcomes_2026-05-13.md`](cycle_8_konjnd_pareto_outcomes_2026-05-13.md) — Cycle 8 KonJND Pareto.
- [`cycle_9_lowq_boost_outcomes_2026-05-13.md`](cycle_9_lowq_boost_outcomes_2026-05-13.md) — Cycle 9 low-q row-weight boost **falsified**.
- [`cycle_9b_pair_boost_outcomes_2026-05-13.md`](cycle_9b_pair_boost_outcomes_2026-05-13.md) — Cycle 9b pair-resampling boost **falsified**.
- [`cycle_10_kadid_tid_outcomes_2026-05-13.md`](cycle_10_kadid_tid_outcomes_2026-05-13.md) — Cycle 10 KADID/TID hyperparam sweep.
- [`cycle_12_midq_boost_outcomes_2026-05-13.md`](cycle_12_midq_boost_outcomes_2026-05-13.md) — Cycle 12 mid-q boost ("first positive signal", falsified at multi-seed).
- [`cycle_14_per_band_tv_outcomes_2026-05-13.md`](cycle_14_per_band_tv_outcomes_2026-05-13.md) — Cycle 14 per-band TV (foundation of V_18 ship).
- [`pareto_2026-05-11.md`](pareto_2026-05-11.md) — Pareto front analysis.
- [`tv_smoothness_sweep_2026-05-10.md`](tv_smoothness_sweep_2026-05-10.md) — TV smoothness sweep across weights.

### Specific recovery-cycle variants

- [`v0_24_dssim_cotrain_v1_result_2026-05-12.md`](v0_24_dssim_cotrain_v1_result_2026-05-12.md), [`v0_24_v2_dssim_cotrain_2026-05-12.md`](v0_24_v2_dssim_cotrain_2026-05-12.md) — V_24 v1/v2 dssim co-training results.
- [`v0_26_konjnd_aligned_2026-05-12.md`](v0_26_konjnd_aligned_2026-05-12.md), [`v0_27_konjnd_dssim01_2026-05-12.md`](v0_27_konjnd_dssim01_2026-05-12.md) — V_26/V_27 KonJND + dssim variants.

## Champions + corpus comparison

- [`champion_2026-05-10.md`](champion_2026-05-10.md), [`champion_candidate_2026-05-10.md`](champion_candidate_2026-05-10.md) — Champion register entries.

### Per-corpus full-panel reports

- [`aic3_zensim_vs_baselines_2026-05-12.md`](aic3_zensim_vs_baselines_2026-05-12.md) — AIC-3 zensim vs baselines.
- [`aic4_zensim_vs_paper_metrics_2026-05-12.md`](aic4_zensim_vs_paper_metrics_2026-05-12.md) — AIC-4 zensim vs paper metrics.
- [`aic_combined_per_codec_2026-05-12.md`](aic_combined_per_codec_2026-05-12.md), [`aic_per_codec_v0_16_2026-05-12.md`](aic_per_codec_v0_16_2026-05-12.md) — Per-codec AIC.
- [`cid22_full_v0_16_vs_ssim2_2026-05-12.md`](cid22_full_v0_16_vs_ssim2_2026-05-12.md), [`cid22_per_codec_v0_16_2026-05-12.md`](cid22_per_codec_v0_16_2026-05-12.md) — V_16 CID22 reports.

### Statistical methodology

- [`mohammadi_2025_verification_2026-05-14.md`](mohammadi_2025_verification_2026-05-14.md) — Our logistic rescale reproduces Mohammadi 2025 anchor Z-RMSE values.

## Performance + optimization

- [`extended_iw_runtime_perf_2026-05-15.md`](extended_iw_runtime_perf_2026-05-15.md) — Original 4-permutation runtime cost benchmark.
- [`extended_iw_runtime_perf_optimized_2026-05-15.md`](extended_iw_runtime_perf_optimized_2026-05-15.md) — **Post-optimization**: combined Extended+IW dropped from +25 % → +12 % at 1024² via fused 2-mask SIMD + V-blur-only on H-blurred sigma buffers + IW-only mu1 bug fix.
- [`iw_perf_hotspots_2026-05-15.md`](iw_perf_hotspots_2026-05-15.md) — Profiling hotspots that drove the optimization.

## Paper-faithful spike

- [`iw_pyramid_spike_methodology_2026-05-15.md`](iw_pyramid_spike_methodology_2026-05-15.md) — Wang & Li 2011 steerable-pyramid weight estimator design.
- [`iw_pyramid_ab_results_2026-05-15.md`](iw_pyramid_ab_results_2026-05-15.md) — A/B vs spatial-variance weights. Pearson 0.838 — weights ARE different.

## Wire format / bake binary

- [`reorder_lz4_zstd_eval_2026-05-13.md`](reorder_lz4_zstd_eval_2026-05-13.md) — HU L2-reorder + LZ4 ship-form 5.2× compression.
- [`reorder_rle_eval_2026-05-13.md`](reorder_rle_eval_2026-05-13.md) — RLE sweep (dead end).
- [`v0_17_quantization_review_2026-05-13.md`](v0_17_quantization_review_2026-05-13.md) — F32 / F16 / I8 quantization review.
- [`zenpredict_compression_eval_2026-05-13.md`](zenpredict_compression_eval_2026-05-13.md) — zenpredict-side compression eval.
- [`zenpredict_rle_zerobias_eval_2026-05-13.md`](zenpredict_rle_zerobias_eval_2026-05-13.md) — zerobias + RLE eval.

## Data integrity audits

- [`coefficient_blocklist_patch_2026-05-12.md`](coefficient_blocklist_patch_2026-05-12.md) — coefficient blocklist patch.
- [`holdout_overlap_audit_2026-05-11.md`](holdout_overlap_audit_2026-05-11.md) — Holdout overlap audit (initial).
- [`dhash_threshold_revert_2026-05-14.md`](dhash_threshold_revert_2026-05-14.md) — **dHash d≤16 cleanup REVERTED** (false positives; d≤10 + user-eye verification is the new standard).

## Sweep verification

- [`jsmlp_codec_sweep_verify_2026-05-12.md`](jsmlp_codec_sweep_verify_2026-05-12.md) — JS MLP codec sweep verification.

## Sweep budgeting (cost before you launch)

- [`hdr_sweep_budget_2026-08-05.md`](hdr_sweep_budget_2026-08-05.md) + [`.tsv`](hdr_sweep_budget_2026-08-05.tsv) —
  multi-codec **HDR** sweep budget: measured `alpha + beta*pixels` per (codec, preset, quality)
  for zenav1-svt (p6/p10/p13 + HdrFork), zenjxl (e7/e1), jpeg+gainmap (ultrahdr), and zenavif
  (speed 4/10) on real 10-bit PQ sources across a 64x64 -> 7.08 MP ladder, plus per-pair GPU/CPU
  metric cost. **Headline: encode is minutes, scoring is hours** — the 76-source corpus encodes in
  6.9 CPU-h (~3-5 min of distributed wall) but takes 11.5-16.1 GPU-h to score, so sweep wall-clock
  is the metric queue and the QUALITY presets are effectively free. Weekend N = 283 (all-GPU
  metrics) to ~800 (cvvdp-on-CPU + ssim2 only).

## Reading order suggestions

**If you want to ship a new bake**:
1. [`v0_18_methodology_2026-05-13.md`](v0_18_methodology_2026-05-13.md) — canonical reference recipe
2. [`v0_18_ship_reference_card_2026-05-14.md`](v0_18_ship_reference_card_2026-05-14.md) — gate numbers
3. [`v0_20_input_shaping_methodology_2026-05-15.md`](v0_20_input_shaping_methodology_2026-05-15.md) — `--auto-transforms` default
4. [CLAUDE.md "Principled experiment workflow"](../CLAUDE.md) — 10-step protocol

**If you want to understand a falsification**:
1. [`falsification_reeval_triage_2026-05-15.md`](falsification_reeval_triage_2026-05-15.md) — catalogue
2. [`falsification_reeval_results_2026-05-15.md`](falsification_reeval_results_2026-05-15.md) — verdict reproduction
3. [CLAUDE.md "SROCC-only verdicts BANNED"](../CLAUDE.md) — why these were SROCC-gated and what replaces SROCC

**If you want to optimize runtime**:
1. [`extended_iw_runtime_perf_optimized_2026-05-15.md`](extended_iw_runtime_perf_optimized_2026-05-15.md) — latest numbers + cost breakdown
2. [`iw_perf_hotspots_2026-05-15.md`](iw_perf_hotspots_2026-05-15.md) — where the time goes
3. CLAUDE.md "Performance Optimization" — archmage SIMD patterns

**If you want to make EXTRACTION faster** (read this BEFORE re-profiling any kernel):
1. [`v2_block_cost_2026-08-31.md`](v2_block_cost_2026-08-31.md) — **what the v2-348 + append-204
   block actually costs, at the tier that ships.** `dense`'s `POOL_SIMD` path is `v4x`-only and
   valgrind masks AVX-512 out of CPUID, so callgrind cannot profile production; this is the wall-clock
   re-profile the predecessor asked for, via seven new `fold_timing` phases. **Every v2 feature kernel
   is flat in ns/px across a 16× pixel range; the plane passes degrade 1.8–3.7×**, so the block is
   64 % plane traffic at 2304² and 32 % at 576² — its composition inverts with size. `dense` is
   **13.5 % of the block on `v4x`**, not the 22–26 % the `v3` Ir profile shows. Production : consumption
   of the six shared planes = **1.84 : 1**. Includes four falsified levers (rayon band split,
   `STRIP_ROWS`, a bit-identical row-major V blur, bounds checks) and a measured era-2 retarget.
2. [`feature_cost_frontier_2026-08-31.md`](feature_cost_frontier_2026-08-31.md) — which FAMILIES a model
   class needs, and what each class's walk costs (the model-class Pareto front).
3. [`extraction_perf_and_buffered_removal_2026-08-30.md`](extraction_perf_and_buffered_removal_2026-08-30.md)
   — the callgrind family split and the levers rejected with reasons. Its Ir shares are the `v3` tier;
   read 1 above before quoting any of them as a wall share.
4. [`fold_footprint_2026-08-31.md`](fold_footprint_2026-08-31.md) — the working-set closed form and the
   per-thread L3 budget. [`fold_mt_scaling_2026-08-31.md`](fold_mt_scaling_2026-08-31.md) — thread scaling.
5. [`era2_perf_break_2026-08-31.md`](era2_perf_break_2026-08-31.md) — the byte-moving break. Its dense
   framing predates 1; the plane pipeline is the bigger prize.

**If you want to make TRAINING faster** (read this before re-profiling the trainer):
1. [`trainer_perf_2026-08-04.md`](trainer_perf_2026-08-04.md) — fresh profile: Adam 44 % (divider-bound,
   irreducible at fixed math), the trainer is RAM-bound not FLOP-bound, half of every lane's 11.88 GB was
   dead after standardization (fixed — see 2 below), `--minibatch-size 32` = 3.63× but changes the bake
   (gated + measured).
   Also: lianli is a same-ISA-tier drop-in node, tower is AVX2-only; zenfleet training not worth it
   below ~60–80 runs/wave.
2. [`trainer_mem_release_2026-08-04.md`](trainer_mem_release_2026-08-04.md) — the dead copy, REMOVED:
   each group is one flat buffer that standardization takes and transforms in place, so a lane holds
   one copy of the features, not two. Full-data (11-group, 779k-row) bit-identity gate. Includes the
   measured failure of the obvious fix — freeing per-row `Vec`s moved full-recipe peak RSS only
   ~0.4 GB, because scattered ~7.5 KB chunks freed out of interleaved glibc arenas never return to
   the OS. Gate memory claims on the real process, not an allocator probe.
3. [`trainer_perf_2026-08-04.tsv`](trainer_perf_2026-08-04.tsv) — raw counters, K-scan, gate output

**If you want to investigate the IW path**:
1. [`v0_20a_path_a_falsification_2026-05-14.md`](v0_20a_path_a_falsification_2026-05-14.md) — original V_20a falsification (SROCC-gated)
2. [`v0_20_l0_norms_2026-05-15.md`](v0_20_l0_norms_2026-05-15.md) — GD-selection confirmation across 4 bakes
3. [`iw_pyramid_spike_methodology_2026-05-15.md`](iw_pyramid_spike_methodology_2026-05-15.md) — paper-faithful pyramid path (untested for ship)
4. [`iw_pyramid_ab_results_2026-05-15.md`](iw_pyramid_ab_results_2026-05-15.md) — pyramid vs spatial-variance A/B

**Era-2 break — rank preservation (the flip gate)**:
[`era2_rank_preservation_2026-08-31.md`](era2_rank_preservation_2026-08-31.md) — the registered §21.1 bar executed across the roster on old-vs-new features: 7 arms × 6 models × 9 corpora. Tiling at the production width passes 5 of 6 and is bit-identical on 8 of 9 corpora *by construction* (the corpora are narrower than the tile); radius 4 passes 2 of 6 and dominates the combined verdict; the era-2 dense kernel is not wired and so is not measurable. Also closes bar clause 3 (the dial gates) for BOTH tiling and radius 4.

**Era-2 break — the user's decision surface**:
[`era2_drop_redefine_table_2026-08-31.md`](era2_drop_redefine_table_2026-08-31.md) — items E (drops) + F (redefinitions) as one table: what each buys, its per-model rank cost, and what it takes to ship. Backed by [`era2_perf_break_2026-08-31.md`](era2_perf_break_2026-08-31.md) and [`blur_radius_locality_branches_2026-08-31.md`](blur_radius_locality_branches_2026-08-31.md).
[`era2_blast_radius_2026-08-31.md`](era2_blast_radius_2026-08-31.md) — the era-2 flip's third prerequisite: what re-pins, what re-extracts, what retrains, and the registered (unlaunched) wave.
[`era2_fast_profile_subset_2026-08-31.md`](era2_fast_profile_subset_2026-08-31.md) — "what subset should a high-speed model be limited to?": the measured cost curve for five compute sets at 1/8/16T, what each gives up, and the recommended pair (`156` for a fast profile, `944peaks` free for 944 MLPs).

**Board currency (2026-08-31)**:
[`board_regen_2026-08-31.md`](board_regen_2026-08-31.md) — the KonJND JPEG-504 ruler reaches the gauntlet: the 17 diluted-ruler cells re-scored through the owners (all 378 board cells now read n=504), the ordering changes it causes (the KonJND leader was a ruler artefact; composite top-10 and the `--select` winner unchanged), badge verification end-to-end, curation of the W-LIN 7b candidate, and three pre-existing board-integrity defects found in passing (276 stale composites, peer rows leading the default sort on an unnormalised scale, one cell that is not valid strict JSON).

**Board fairness (2026-09-04)**:
[`fair_gauntlet_2026-09-04.md`](fair_gauntlet_2026-09-04.md) — the FAIR gauntlet: eight corrections applied (best-of-k seed groups, the `predict --ensemble` units defect, four era/circularity items that were registered but badged nothing, the free-40 precision skew, G-ADDR, the composite's own 37% ssim2-circularity, train==val), a seven-criterion fairness filter with per-row tiers (42 VERIFIED-FAIR / 55 FAIR-NOTED / 336 LEGACY), what moves at the top once best-of-k becomes a k-mean, and three defects found in passing (a registry field-scope collision, a Python/Rust mirror drift, and a render-gate whose last third never gated). Audit TSV: [`fairness_tiers_2026-09-04.pointer.md`](fairness_tiers_2026-09-04.pointer.md). Exam transcription: `ssim2_exam_scorecard_2026-08-31.json`.

**Dial addressability — the D lineage (2026-09-04)**:
[`d_id100_2026-09-04.md`](d_id100_2026-09-04.md) — `D-id100`: the registered-not-run §14.6(ii) build, executed. The shipped Profile D chain reproduced byte-exactly end to end (closing its manifest's stated provenance gap); the identity pin measured to be a **spline** property, not a fit one (eight real re-fits at 0.1–20 % identity data mass move the identity dial +0.0055 and cost −0.0125 CID22 — the identity Gram's `S`/`s`/`q` are exactly zero); 21 identity anchor rows give **CONTRACT 6/6 + REGRESSION 4/9** with byte-identical weights and zero pair-order flips, and the anchor's own unclamped `ssim2_gpu` column takes it to **7/9**. Includes a fixed owner defect (`fit_spline_knots`'s neg-tail dedup deleted genuinely-negative knots) and measured impossibility proofs for the two axes that remain. Artifacts: [`d_id100_bakes_2026-09-04.pointer.md`](d_id100_bakes_2026-09-04.pointer.md).

**FASTCLASS2 — the fast class's blocker is the FLOOR, not the rank (2026-09-05)**:

[`fastclass2_campaign_2026-09-05.md`](fastclass2_campaign_2026-09-05.md) +
registration [`../docs/PLAN_FASTCLASS2_2026-09-05.md`](../docs/PLAN_FASTCLASS2_2026-09-05.md)
— the SET x WIDTH x HEAD sweep on the 156-plus-cheap class, and the results it
produced **before its own arms landed**. **Gate G4 closed the era and RAISED the
bar**: the 944 leaders re-read on `ext944-era2r4-2026-09-01/foldapp2_views` (their
own compute, the fast class's era — NOT the pools root, because their `f156..371`
weights are untrained init on structurally-zero inputs, 13.5-15.8 %/12.2-12.9 %/
8.1-8.3 % of L2 mass with **0.00 %** on append/append2) read composite
**0.8636/0.8626**, CID22 **0.8877/0.8908**, KonJND **0.4783/0.4782**, against the
incumbent `FC_D3`'s 0.8645 / 0.8863 / 0.4322 — **past the composite bar, inside the
CI on CID22, −0.046 on KonJND alone.** So the rank problem is one axis wide and the
SHIP problem is a different quantity: **A7r floor representability**, which the fast
class fails 5/5 codecs, which survives SEVEN recipe variants (uniform pairing,
either within-ref ladder, both, high-q-boost, KADIS, class-C — not one of 35 codec
cells clears the mentor), whose failures are **100 % ordering inversions and 0 %
clamps**, and which **no dial chain can touch**. Isolated at fixed class + layout +
anchor chain: 156 -> 228 costs **three of five codecs** and 0.0315 of monotonicity.
**Identity localised to FOUR slots** (`LUMA_MEAN_REF` f926/931/936/941 — max |v|
0.688 vs <=4.8e-3 for everything else in the 265 set, 0.45-0.64 % of layer-0 mass;
new 261-slice) and **closed** by the id100 anchor chain, contract 5/6 -> **6/6**
with CID22 bit-unchanged. **Servability MEASURED** (`zensim/examples/serve_custom_bake.rs`,
new): a 372-layout bake that reads the peaks block SERVES; every 944-declared bake
is REFUSED — so only S156/S228 can ship, and only at the v1-372 layout, which is
why a servable 372 lane was added and runs first. Side findings: `Zensim::compute`
short-circuits byte-identical input to `(100,0,zeros)` **before the model**, so a C5
failure is never a claim that `zensim(x,x) != 100`; and two silent-no-op defects
fixed — **`--coarse-decay` was discarded on the per-sample-alpha head** (the other
rider of the same `apply_post_adam_penalties` had been guarded since it landed) and
the 372 lane's first draft named the **un-normalised** targets (two groups at ~100x
the others' scale, nothing would have crashed; caught in pre-flight, 15 fits
unspent). Five feature sets + one root
registered, owner-hash-verified. **RESULTS (57 cells, k=3 each):** the answer is
**YES on both layouts** — the selected `S372_S228_H128_p` reads composite
**0.8732** / CID22 **0.8896** / KonJND **0.4999** against era-closed leader bars
of 0.8626 / 0.8877 / 0.4782 (rulers verified identical: 504 KonJND refs, 4,292
CID22 pairs everywhere), and `freeze_check --select --seed-group --min-k 2` picks
it independently at **8/8 floors on every seed**. **Capacity is not a lever**
(H32 ≈ H128 inside every spread at 30-47 % fewer bytes) and **the COMPUTE CEILING
is BELOW the restricted sets** — `SORACLE`, the same recipe free to read all 944
coordinates, is the lowest non-degenerate cell at 0.8581 / KonJND 0.4191, which
falsifies the campaign's own registered "the KonJND gap is a compute gap"
hypothesis. The α head is an **inverted ranker** (raw CID22 −0.8921, a better
ordering in magnitude than the plain path's +0.8863) so its pack could not fit a
monotone spline, and `--monotonicity-reg` stays UNMEASURED behind that
prerequisite. **Ship: PROPOSE, not install** — C5 (identity dial 90.9368), C6
(1,642 of 9,593 cells above identity, worse than shipped B's 6.01 %) and A7r
(5/5 codecs) all fail, every one a weights property. **W4 NOT MEASURED**: box at
load 72-79 from other lanes; `scripts/fastclass2_w4_deferred.sh` finishes it and
refuses a busy box.

**Push clobber + the push guard (2026-09-05)**:
[`push_clobber_2026-09-05.md`](push_clobber_2026-09-05.md) — `origin/main` moved SIDEWAYS twice on 2026-09-04 (jj ops `db7c8ca86b69`, `0edf97e28a91`), dropping nine commits from six lanes with no error and no warning; the per-added-line audit separating the seven that were re-landed or superseded from the one that was genuinely lost (`d3a948ca`, the G-ADDR board coverage — 482 of 498 added lines absent, `cut_gaddr_negtail_probe.py` absent entirely, and the boards on `/mnt/v` already generated with it), the re-land (`2e5cdc8b`, diff byte-for-byte identical to the original), and the owner guard `scripts/safe_push.sh` (fetch → ancestor-assert → set → push → verify; refuses sideways with rc=3 and names every commit it would drop; 4-case self-test incl. the negative control, plus a retrospective control that replays the real clobber).

**D + the free set (2026-09-05)**:
[`d_free_id100_2026-09-05.md`](d_free_id100_2026-09-05.md) — the user's question *"did the +free get tried for D-id100-negrich or just the base 156?"* answered by building it: twelve arms (four coordinate slices × three dial variants) on the one leg that carries the free slots, with a **matched-leg 156-only control** because the D lane's own 372 Gram cannot carry a free slot at all. All four `-id100-negrich` arms reach **CONTRACT 6/6 + REGRESSION 7/9**, failing only the D lane's structural A7/A9 pair, on a different leg and a different feature width. The free set's gain reproduces the free-features lane's λ=0.3 table to 4 dp **with CIs it did not have** (CID22 +0.0405 [+0.0357, +0.0456], KonJND +0.0813, hfnlproxy +0.0707; against SDR25 −0.0547, AIC-4 −0.0232). Headline: **the ZERO-marginal-compute half carries it** — a peaks-only slice gets 97 % of the CID22 gain, 95 % of KonJND, 96 %/98 % of the `hfnl_cid22band` pooled/per-ref gain, and beats the full free set on 8 of 12 corpora, with no exposure to the free-40 route-parity skew. Three findings in passing: **at 944 width the identity feature vector is NOT the zero vector** (286 of 944 slots, varying by image — the gate's own note constant says otherwise); the free-features lane's published λ=0.3 bakes carry a **pre-`abfe13de` spline** whose negative tail was deleted (dial floor −113.5 vs −207.6; rank untouched); and a `zensim_speed_bar` reading that made a strict superset look 3.6 % faster was **rejected** after the owner's own 15-start instrument read 1.0020. New G-ADDR registry rows (append-only) for the 944-POOLS grid + two new probes. Artifacts: [`d_free_id100_bakes_2026-09-05.pointer.md`](d_free_id100_bakes_2026-09-05.pointer.md).

**Profile D — the ship flip (2026-09-05)**:

[`d_peaks_372_postC_2026-09-05.md`](d_peaks_372_postC_2026-09-05.md) — the registered peaks follow-up, BUILT and GATED and **NOT INSTALLED**, plus the runtime-era 372 ruler it needed. `Dpeaks372_id100negrich` is the §7 command with one flag changed (`--slice-file <0..227>`): 372-wide so `Zensim::compute()` serves it natively, 26 coefficients = 20 basic + **6 peaks** (largest column norm in the bake is a PEAKS one), CID22 **0.87126 vs shipped D's 0.86333 = +0.00798 [+0.00476, +0.01116]** paired-bootstrap, KADID +0.01225, CONTRACT **6/6** — and it does not ship because G-ADDR **A8 goes PASS → FAIL** (negative-tail p1 −167.7 against the incumbent's −212.1 and the ssim2 bar's −187.1), which is the floor reduction the user's hard dial rule names; TID −0.01853 and AIC-3 −0.00770 also lose with CIs excluding zero. The ruler work: a `v1postc` root at HEAD (**the `v1cur`→`v1postc` rank shift is ≤ 6.8e-4 SROCC across five shipped bakes** — two orders below the previous 372 era step) with a positional `human_score` alignment GATE in the packer; **the canonical dial grid is UNREBUILDABLE** (its own build list points at the decode cache deleted 2026-06-22, 0 of 2,560 paths exist) yet the surviving 2026-07-27 pixels are the RIGHT instrument because the registered `peer_ssim2` pins were measured on THEM — so the canonical instrument has always paired ssim2 truth from one pixel set with features from another; the identity probe is era-INVARIANT (a HEAD rebuild is byte-identical); the negative-tail probe is recovered 2,000/2,000 from R2. Two defects fixed: **`extract_features_372col` sorts its output by `ref_basename`**, so positional re-attachment silently scrambled the first grid build (shipped D's ladder mono 0.9847 → 0.5611, jpeg q0 reading 75.6 on an ssim2 = −8.03 image), and a manifest-declared `feature_set_id` resolved to an EMPTY slot set because the lookup used the full id against class-form registry keys. New: `fit-lasso --feature-set-id` (checked against the emitted bytes, refuses rather than writes a wrong stamp) and `ZL_ERA=preC|postC|canonical` on `dialgate_arms.sh`.

[`d_ship_flip_2026-09-05.md`](d_ship_flip_2026-09-05.md) — the first sanctioned ship-default flip of the campaign: `ZensimProfile::D`'s bake becomes `d_sdr_add156_id100_negrich_dial_2026-09-05.bin`. The forward pass is **byte-identical** to the era-1 bake (both strip to weight-sha `330d8c09…`), so rank is bit-identical on 11 of 14 corpora and within a monotone remap's tie residue on the other three (CID22 **0.863380** either way) while the dial goes CONTRACT 5/6 → **6/6** and REGRESSION 2/9 → **7/9** — identity 96.1157 → **100.0000**, reach 108.25 → 156.55, negtail floor −100 → −213.1, 0 cells above identity in both eras. **The `-peaks-` half of the user's request did NOT ship, and the doc says why with a measurement for each blocker**: (1) that arm is 944-declared-width and `Zensim::compute()` cannot feed it — installed and run, it fails `ModelForwardFailed: "bake declares more input features than the caller supplied"` on every non-identical pair, the same documented limitation `profile_c_tests::compute_on_non_identical_pair_fails_loud` already pins for `C`; (2) its CID22 on its own native root is 0.8465 against shipped D's 0.8634. Two findings in passing: **the shipped runtime is one extraction era AHEAD of both v1-372 eval roots** (option C landed `56bbcda2` at 15:43 on 2026-08-30; the default root was built at 13:21 — re-extracting CSIQ at HEAD with the same tool on the same input moves **120,804 of 135,096 basic cells, max |Δ| 4.54**), which also makes `CLAUDE.md`'s "Not flipped" claim stale, corrected in place; and **the frozen 372 safesyn Gram — ADD156's own 196k leg — already carries the peaks block fully populated** (72/72 diag-nonzero), so the peaks idea is a 372-width refit on the right leg, not a fleet wave.

[`d_peaks_lambda_sweep_2026-09-05.md`](d_peaks_lambda_sweep_2026-09-05.md) — the registered λ sweep on the peaks slice, executed: 6 λ (5e-4..1.6e-2), plus a new owner capability `bake_dial_refit fit-lasso --anchor-weight <N>` (row duplication in the spline anchor — `fit_spline_knots` bins by percentile/median, so there is no numeric-weight mechanism to extend; 4 unit tests, byte-identical default) exercised as a negative-tail lever (×2/×4 weight on the anchor's own 147 `ssim2_gpu<0` rows) at the top-2 λ by CID22, CONFIRMED spline-only by a strip+cmp control. **MID-TASK CORRECTION** (a garbled first ruling naming "−5" was superseded, verbatim: *"the number is −50, NOT −5 … the tail is judged PER CODEC"*) replaced A7/A8 with four checks graded PER CODEC FAMILY on the grid (jpeg/webp/avif/jxl) and per KADIS distortion family on the negtail probe (which carries no codec info at all — a join recovers `dist_type`), with families ssim2 itself never reaches −50 exempted. **Result: 0 of 11 arms — including shipped D itself — pass check (1)** (blocked by one thin family, `mean_shift` n=8, where no arm's floor comes within 35 points of ssim2's); check (4) (grid, avif+webp) is passed by exactly 3 arms (`lam5em4`, `lam1em3_w4`, `lam2em3_w4`) and **failed by shipped D** (webp short by 1.9). A D-relative reading (reported as context, not the rule) shows the `--anchor-weight` lever regresses EVERY negtail distortion family relative to D (24/24) while it narrowly helps grid webp/avif — a universal floor cost for a targeted gain. Under this literal reading **no arm satisfies the full corrected ship rule**; `lam5em4` is the most defensible single candidate (zero grid regressions vs D, fewest negtail regressions, positive CID22 CI) if one must be named. **Nothing ships** (lane scope): `ZensimProfile::D`/`zensim/weights/` untouched.

**G-ADDR negative tail — FLOOR REPRESENTABILITY, per codec (2026-09-05)**:

[`dial_addressability_gate_2026-09-04.md`](dial_addressability_gate_2026-09-04.md) **§16** — the USER RULING (three forms in one day; the third operative: *"i care that the lowest configurable settings per codec are representable, not that negative fifty is in that specifically"*) that retires `A7`/`A8`/`A9` and leaves **no dial value as a bar anywhere in the tier**. `A7r` asks, per codec on the canonical dial grid, what fraction of `(image_id, codec)` ladders have their **K=3 lowest configurable settings REPRESENTED** — strictly ordered across those steps and into the next one up, and off the dial's clamp unless that ladder is its sole holder — with the **bar being the mentor's own fraction on the same cells**; `A8r` (the negtail probe) becomes **report-only** because that instrument carries no codec identity at all, and `A9r` is dropped into the report block. MEASURED bars: avif **1.0000**, jxl **0.9697**, webp **1.0000**, jpeg **0.0000** — and jpeg's zero is the **encoder saturating** (22/22 ladders emit byte-identical output at their three lowest settings, max |Δ| exactly 0.0), which is what makes exemption a measurement rather than an exception. **16 scorers graded on the runtime-era instruments give the install answer: shipped Profile D is the only bake that is both `A7r`-PASS on every codec and CONTRACT-PASS**, beating the mentor on jxl; all 11 λ-sweep arms and D-peaks fail on jxl, and D-peaks is 44 points *deeper* than shipped D at the probe's p1 while being *worse* at resolving jxl's floor — the clearest possible statement of why depth was the wrong bar. Board re-graded and re-grafted with the contract-driven NOT SHIPPABLE badge asserted unchanged (47 measured / 46 on board, 0 contract rows changed); `--gaddr-tail-pins retired` reproduces every pre-ruling number. Owner for the board pass: [`scripts/gaddr_board_regrade.sh`](../scripts/gaddr_board_regrade.sh).

[`d_peaks_jxl_floor_2026-09-05.md`](d_peaks_jxl_floor_2026-09-05.md) — per-ladder classification of the jxl `A7r` failures for `lam1em3` and `Dpeaks`/`lam2em3`: both fail on the IDENTICAL 4 of jxl's 33 ladders, all 8 (4 ladders × 2 arms) are `INVERSION`s (zero `TIE`, zero `CLAMP`, cross-checked exactly against `bake_verdict`'s own `codec_floor` counts), and in every case the pre-spline RAW model output is already inverted at the same step pair — so **the lever is in the fit (weights), not the spline**: a monotone calibration spline cannot introduce or fix a rank inversion, which is also why ROUND 90's `--anchor-weight` spline-only lever was structurally incapable of reaching this bug (its own control proved the CD lasso's `w`/`bias` never reads an anchor row). Shipped D (`ADD156`, no peaks block) passes all 4 ladders cleanly; `lam1em3` (38 coeffs) and `Dpeaks` (26 coeffs) are different fits at different λ yet invert on the same 4 references, implicating the peaks block (`f162-164, f211-212, f224`) rather than the λ choice. Two fixes REGISTERED, neither run: an isotonic/monotone shape term in `fit-lasso`'s solver, or row-level up-weighting of the jxl-floor rows in the training GRAM itself (not the spline anchor). Nothing installed.

[`d_peaks_slot_ablation_2026-09-05.md`](d_peaks_slot_ablation_2026-09-05.md) — isolates the jxl inversion to ONE feature: per-feature decomposition (`raw_pred(hi)−raw_pred(lo) = Σ coef_k·(x_k(hi)−x_k(lo))`, exact for the single-layer identity-activation bakes, cross-checked against ROUND 93's own published raw deltas on all 8 cases) finds `f162`'s own contribution is 2× the entire net inversion on every failing (ladder × arm) case, every other active feature nets in the CORRECT direction, and its coefficient sign AGREES with its own 33-ladder majority trend — the defect is a LOCAL non-monotonic bump at the failing step, not a sign error. 8 single-slot LOO refits (one per lam1em3's active peaks slot) + 1 leave-all-suspects refit, all λ=1e-3, id100+negrich chain control-verified byte-identical to `lam1em3` except `--slice-file`: **only dropping f162 (alone or with the other suspects) reaches jxl `A7r` 1.0000**; the other 7 leave the inversion completely unchanged. **Neither ships** — both f162-dropping arms trade `A7r` for a brand-new `A4` (robust floor, dial p5) failure that shipped D and the untouched peaks arm both pass, a clean either/or with no arm failing both. CID22 gains cleanly for all 9 arms by paired bootstrap (+0.0085 to +0.0098 vs D, CI excludes zero) but the ship rule needs both axes clear, and **zero of the 9 arms satisfy it**. Nothing installed.

[`d_peaks_jxl_ladders_2026-09-05.md`](d_peaks_jxl_ladders_2026-09-05.md) — presentation + board pass over the two prior entries: a visual page (`zensim-bench/examples/ladder_tile_gen.rs` — `zenpng`+`zenresize` only, no ImageMagick in the decode/resize path) showing all 4 failing jxl ladders, reference + bottom-4-q + one mid step, full-frame downscale paired with a native 1:1 detail crop (origin stated), scorer table with the inverting pair highlighted — matches the two source docs' published numbers exactly. Promotes `lam1em3` and `minus_f162` onto the summer-gauntlet board as `Dpeaks_lam1em3` / `Dpeaks_lam1em3_minus_f162` (new `gauntlet.family_of` branch, "D-peaks candidates"), scored on the postC 372 root with the full 12-corpus set + M3/M3a coherence + G-ADDR grafted (the graft was NOT a no-op — it normalizes the inline-embedded block's key set and adds `dial_gaddr_source` provenance). Both regenerated boards pass every gate (fair 9.78 MB, under the 12 MB cap; all-rows 22.02 MB, over cap per pre-existing documented policy). Two findings, both reported rather than silently worked around: the rows are fair-board **FAIR-NOTED / k=ungroupable** (not the brief's guessed "k=1 UNREPLICATED" — a deterministic lasso fit's repro carries no seed field at all), and `peer_ssim2` cannot appear in ANY regen of the fair-only board (fails `a_repro` structurally — it is a reference metric with `repro: null`, not a trained bake) so the task's 4-way compare URL is delivered against the all-rows board instead, both compare URLs harness-verified clean (no banner, exact rows, fragment order).
[`corruption_head_d_2026-09-05.md`](corruption_head_d_2026-09-05.md) — Profile D's companion corruption head, built at the RUNTIME (post-option-C) extraction era and wired into `bake_verdict`. D's corruption weakness is **intrinsic, not an era artifact**: re-extracting the 2,016 persisted gate PNGs at HEAD moves **73.7 %** of basic cells (max |Δ| 4.35) and D's `pass_q20` reads 26.9 % → 26.8 %. Five blockers cleared first — the corruption pixels are deleted by design (regenerable, the generator is deterministic in `(ref_id, seed, params)`); the 2026-07-24 `sources.tsv` points into the **quarantined imazen-26 inspo tree** (154/174 paths dead); that build **died `rc=1` at 141/174**; `--corruption-head` takes a **BAKE** while every 2026-07-24 head is `.json`, so **no 372 head had ever been through the gate**; and that head reads `f228..371`, which D's walk does not compute. Since `Off` and `Peaks` cost the SAME, `f0..227` is FREE for D, and within that free set peaks are worth +1.0 point of detection for nothing (`d228` **85.9 %** / 0.31 % severe FP / 0.00 % anchor FP, as BAKED, vs `d156`'s 84.9 % / 0.41 %). **But masked/IW is worth +4.8 points on top** — bake-vs-bake on identical held-out rows the 2026-07-24 head reads **90.7 %** at the same severe FP — so that head's ablation conclusion that the signal needs the mask/iw/peak block is **SUPPORTED**, and D's companion is a trade (85.9 % free, or 90.7 % by forcing `V1PoolsMode::Full`). A separate defect surfaced on the way: `train_corruption_head.py` has always reported a `CalibratedClassifierCV` while persisting a different `LogisticRegression`+isotonic model — quote the BAKE (`d228` ladder FP 11.22 % as baked vs 15.83 % as reported). **D goes 26.8 % → 91.4 % `pass_q20`** under the registered `min(perceptual, gate)`. **But the gate cannot select a head**: the no-codec-negatives ablation WINS it (99.1 %) by being trigger-happy while being the worst head on honest content. On the 2026-09-05 ladder (9,593 honest current-era imazen codec cells) `d228` fires on 11.2 % — and **entirely at HIGH quality**: 0.0 % below q50, **53.7 % at q95-100** (avif-rav1e 97.2 %), 1,134 of 1,139 flagged cells at q ≥ 80, because a corruption confined to an 8×8 square is *also* nearly identical to its reference. A `dial < 90` guard reads 64.0 % gate at 0.74 % honest FP (`dial < 80`: 47.9 % at 0.00 %) — measured as a PROPOSAL, implemented nowhere. Cost: zero extra extraction, forward **≤ 2.5 µs/compare**; the zenbench arm cannot resolve it and its 8T/576² cell is discarded as degenerate. Nothing installed in `zensim/weights/`; no public API changed.

**Inversion attribution — a backwards rung is the ENCODER's when BOTH references agree (2026-09-05)**:

[`inversion_truth_2026-09-05.md`](inversion_truth_2026-09-05.md) — the USER DIRECTIVE (*"for inversions, we should choose say ssim2 and butter and only flag true inversions where they agree, and we can then file or update tracking issues on codecs for when they are nonmonotonic"*) implemented at ONE owner, `dial_addressability::encoder_inversion`, shared by the G-ADDR contract's `mono`/C1 input and `bake_verdict`'s ladder-inversion census so the gate and the census cannot drift. **The rule:** a material backwards rung leaves the dial's count only where `Δssim2 ≤ −0.5` pt AND `Δbutteraugli-pnorm3 ≥ +0.05` distance. **The butteraugli margin is NOT a noise margin — there is none**: a from-scratch re-run of the instrument's jpeg leg reproduces 2,574/2,574 cells at `max |Δ| = 0` on BOTH butteraugli variants (extending §8.0's gate, which checked only bytes + ssim2), so it is derived by equivalence to ssim2's own materiality — the **p85 of |Δ| on FORWARD pairs whose Δssim2 ∈ [0.45,0.55], rounded UP to the next 0.05** (pnorm3 0.0481 → **0.05**; max 0.2189 → **0.25**), rounding up being the conservative direction because a larger margin excuses FEWER rungs. **pnorm3 is PRIMARY on measurement**: 94.30 % direction agreement with ssim2 over 9,411 pairs against `max`'s 75.27 %, and `peer_butteraugli_max` fails C1 (0.9286) under both readings. **"BOTH, not EITHER" is load-bearing** — of 105 ssim2-alone material inversions butteraugli corroborates only 47, and on D's ten worst **ssim2 alone confirmed 9/10 while both references confirm 5/10** (on two, ssim2 reads a 7–10 pt loss while butteraugli says the higher setting is BETTER). **Per-codec encoder-confirmed non-monotonicity** on the 2026-09-05 ladder instrument (bake-independent, `bake_verdict --encoder-inversion-census`): `avif-rav1e` **20**/2,457 (14 refs, 13 also costing bytes), `jpeg` **5**/1,950 (3 refs, all costing bytes), `avif-svt` **1**/1,599, **`jxl` 0 and `webp` 0**. Issues filed: `imazen/zenjpeg#201`, `imazen/zenrav1e#42`, `imazen/zenav1-svt#19`. **Dial-attributed re-grade** (single → agree): D **0.99310 → 0.99470** (15 rungs), D-prev 0.99420 → 0.99540, A 0.98030 → 0.98120, B 0.97760 → 0.97870, `peer_ssim2` **0.98880 → 0.99160** (26) — D's dial-attributed inversion rate is **0.53 % against the mentor's 0.84 %**. **The board provably cannot move and did not** (badge count identical, gates PASS): `mono_agree ≥ mono_single` always, `mono` gates exactly one row (C1, a `≥` bar), and all 130 board G-ADDR cells already read C1 PASS; the two-reference reading is moreover NOT MEASURABLE on the canonical-372 / 944-POOLS grids, whose only butteraugli is the `max` variant (identified at median rel err 0.0029 over 4,105 cells) with pnorm3 unrecoverable without a decoder-era confound. `--inversion-truth single` is **byte-identical to the pre-ruling binary** (0 JSON differences), and every pre-2026-09-05 count is scoped in `eval_annotations.json` as `inversion-counts-single-reference-pre-2026-09-05`. Gate doc **§18**.


**Do we have bugs in feature calculations? — the audit, the invariant probes, and the servability census (2026-09-05)**:

[`../docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`](../docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md) — the user's question answered with measurements: a committed inventory of **27 distinct feature-calculation defects** (8 FIXED, 4 OPEN-data, **1 OPEN-live**, 3 OPEN-plumbing, 11 BEHAVIOUR-not-bug) plus 9 invariant probes and a servability census. **The single live arithmetic defect is F4** — v1's SSIM per-pixel `d` has a `.max(0)` floor and no upper cap and its `num_m` carries no `C1`, so `f313` reaches **5,814,302** against a photographic p99.9 of 0.48; open by decision (the winsor guard clamps the symptom), not by oversight. **The engines themselves are clean**: bit-identical across 5 repeats and rayon pools 1/28 (12 geometries), **33 of 33 engine-parity comparisons BIT-EXACT** (buffered v1-372 ↔ fold v1-only ↔ fold944-full ↔ both product engines), cross-tier v4x-vs-v3 max abs **3.48e-8** with **0 cells over the golden tolerance policy** and the whole v2-era block `f372..943` bit-exact, **0 NaN / 0 Inf** across 72 vectors on five pathological input families, and no width-class effect at tight / non-tight / odd / sub-64 / past-`H_TILE_WIDTH`. **Four new findings.** (1) `V1FreeExtras` is **silently inert unless `append_block` is also declared** — `append_block` does double duty (layout *and* compute), the raw-moment slots live at `f720+`, so a `v1_only` walk asking for `RawMoments` without it emits 720 wide with a populated count identical to `Off` (228 vs 228), no error; now gated. (2) The **372 identity all-zero vector is FABRICATED and has never been measured** — both product entries short-circuit `source == distorted` and synthesise `(100, 0, zeros)`; computed on the same pixels the v1 block populates **144 of 372** slots (max 1.12e-3), and at 944 **286 of 944** (independently reproducing §3.36 on a different population), resolving into exactly three classes: 15 reference-only (`GRAD_SRC_MEAN`, `LUMA_MEAN_REF` — correct), 12 `PJND_FRAGILITY` (a formula artifact, 0.395 full-walk / exactly 1.0 v1-only), 259 fp residue ≤ 1.12e-3. (3) **Correction**: `FEATURE_SET_IDS.md` §1 failure #9 ("the v1-372 `f0..155` is NOT the 944 fold's `f0..155`, 156 of 156 slots differ") is an **era artifact of two stored instruments**, not a code claim — in one process at one commit they are **bit-exact on 372 of 372 slots at 11 geometries and both tiers**. (4) The **identity score cliff**: `Zensim::compute` returns exactly **100.000000000** for `ref == dist` but **96.2296** when ONE byte in ONE channel of ONE pixel of 90,000 changes — a 3.77-point step at zero distortion that is the mechanism behind G-ADDR's "266 of 4,424 cells score above a perfect copy". **SERVABILITY CENSUS** (§4A, the architecture lane's phase-gate baseline): **400 of 433 board bakes, 3 of 11 shipped bakes, 11 of 14 registered feature sets and 2 of 10 selectable shipped profiles are REFUSED by `Zensim::compute`**, every one with the identical `ModelForwardFailed { "bake declares more input features than the caller supplied" }` — one cause (the product entry emits a 372 layout; `caller_input_width() > 372` is refused), four symptoms. **`ZensimProfile::C` and `CHdr` are unservable while `candidate-profiles` is default-on and C's bake ships to crates.io**, and the identity short-circuit hides it: both still return `100.000000` on a ref-vs-ref smoke test. Every SERVED case is also **served-but-MISMATCHED**, for one reason with no new mechanism — the runtime is self-consistent (bit-exact, measured) while the 372 roots are one extraction era behind (F3b). **Monotonicity, with a control**: on two ladders whose own MSE control is monotone 12/12, **40 and 55 slots are persistently, amplitude-really non-monotone**, mostly correctly so (62 violating series contain an exact 0.0 beside non-zero values — the signature of a rectified one-sided feature like the `GLOBAL_CGAIN`/`GLOBAL_CLOSS` pair); a third ladder (repeated box blur) was **DISCARDED because its own control fails on 12/12 images** after it produced 176 false violations. Instrument `zensim/examples/feature_invariant_probe.rs` (10 modes incl. `ZEN_FIP_CAP_V3` tier capping, which refuses to run rather than mislabel a native pass as capped); gates `zensim/tests/feature_invariants.rs` (10 tests, all passing, full suite green 27/27 binaries); five entries registered in `benchmarks/eval_annotations.json`; artifacts + `_MANIFEST.json` at `/mnt/v/output/zensim/feature-audit-2026-09-05/`.

**Corruption-head theories — gating, model form, misses, family subtraction (2026-09-06)**:

[`corruption_head_theories_2026-09-06.md`](corruption_head_theories_2026-09-06.md) — four user questions, pre-registered at [`../docs/PLAN_CORRHEAD_THEORIES_2026-09-06.md`](../docs/PLAN_CORRHEAD_THEORIES_2026-09-06.md) and pushed before any result existed; the incumbent `d228` split read VERBATIM and parity-gated against its own `metrics.json`, `f0..227` for every arm, rev1 throughout. **The head should be nonlinear, and that answer dissolves three other questions.** On identical features/split/calibration, pAUC over ladder-FP ∈ [0,5 %] goes **54.20 (logistic) → 97.73 (`mlp64_32`) → 98.11 (`HistGradientBoosting`)**; at T = 0.9 `hgb` reads **98.90 % detection at 1.23 % honest FP and 2.38 % near-lossless (q ≥ 95) FP** against the incumbent's 86.01 / 11.37 / **50.00**. Train ≈ test for every arm (hgb 99.2 / 98.5), the ordering holds on the single-source gate grid that removes the content degree of freedom entirely (672 `gb82_dog` triples: DEPLOY `pass_q20` **99.85 %** vs logistic 91.37 % vs D's dial alone 26.79 %), and the incumbent's ladder FP is 0.0 % below q50 but 50.0 % at q ≥ 95 *on the same images* — which a content discriminant cannot be. **Dial gating is real but is a crutch**: a two-parameter soft gate fit by train MLE converges to *no gate* (G = 110 on a grid to 160, output identical to ungated to six figures); the hard `dial < 90` mask is genuinely the best low-FP policy for the LINEAR head (51.7 % detection at 0.5 % FP vs 43.5 %, disjoint CIs — so the pre-registered "conditional beats hand-set" criterion FAILS as written) yet costs `hgb` **33 pAUC points** (98.11 → 64.93) because it caps detection at the dial's own reach; the dial as an input feature is worth +3.5 pAUC, as a *conditioning* variable (per-band heads) +19.1, and ~0 once the head can bend its own boundary. **Two corrections to [`corruption_head_d_2026-09-05.md`](corruption_head_d_2026-09-05.md) §4.3**: the near-lossless FP is NOT a "tiny local break looks near-lossless" confusion — small regions are the EASIEST (`sq8` 95.0 % recall vs `whole` **66.9 %**), and 64.4 % of flagged honest cells (66.1 % at q ≥ 95) have **`chroma_boundary`** as their nearest positive, i.e. the head confuses near-lossless chroma with a chroma-plane break; and it is NOT "a separability limit of the feature set" — the same 228 features separate them at 2.38 % FP under `hgb`. **What gets missed** is whole-image low-amplitude edits (`edge_duplicate_top_row` **17.2 %**, `edge_shift_interior1px` 34.5 %, `channel_zero_b` 38.7 % at a dial of 91), with severity running BACKWARDS (op20 90.4 % > op100 82.9 %) — and `hgb` lifts the worst family to 82.8 % and the worst region to 94.5 %, so the miss profile was the linear boundary, not the catalogue. **Family subtraction works, second-order**: over all 44 leave-one-out refits only **`channel_zero_b`** cuts honest FP on its own with a paired-bootstrap CI excluding zero (**−1.69 pt [−2.88, −0.61]**, while raising matched-FP detection **+6.36 pt [+5.58, +7.16]**); the greedy top-8 (`channel_zero_b`, `block_garbage`, `channel_swap_rb`, `edge_border_top_k4`, `channel_swap_gb`, …) reads **−4.30 pt FP [−6.45, −2.42] and +12.88 pt detection [+11.47, +14.29]** — four of the top five drivers chromatic, matching the mechanism above. At a FIXED threshold the whole effect is invisible (removing positives shifts the class prior, raising FP *and* detection together), which is why every comparison here is at matched FP. Family-grouped heads (73.65 pAUC) and per-dial-band heads (73.27) land at the same place, far under `hgb` — added capacity is the lever, the conditioning axis barely matters. Nothing wired in, no bake replaced, no ZNPR emitted (a nonlinear head has no wire format: `emit_znpr` writes one identity layer from `coef_`, and the owner now refuses `--bake-out` for the new forms rather than baking one wrong). **Addendum (§11, same day): the BLAS/OpenMP thread-count nondeterminism found on the way (§9) is FIXED at the owner** — `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS`/`VECLIB_MAXIMUM_THREADS`/`NUMEXPR_NUM_THREADS`/`BLIS_NUM_THREADS` forced to `"1"` before `numpy` import plus a `threadpoolctl.threadpool_limits(1)` pin, proven by `scripts/v_next/corrhead_determinism_gate.py`: the exact `d228` recipe at ambient 1/4/8/28 threads now yields byte-identical `corruption_head_d228.bin` / `..._w944.bin` / `metrics.json` / weights `.json` (sha256 `6f97b653…`, which is exactly the historical "1T" value). It does **not**, and structurally cannot, reproduce the historical 28-thread shipped bake (`da411c8c…`) byte-for-byte — the shipped bake was **NOT replaced**; registered delta on the canonical `gb82_dog` held-out gate grid, scored via the actual baked bytes (`predict_features_with_bake`): detection at T=0.9 **83.929 % → 84.077 % (Δ +0.149 pt)**, FP on `q10`/`q20` matched anchors **unchanged at 0.000 %**. A synthetic smoke test found no thread-order sensitivity in `HistGradientBoostingClassifier` either, before or after the fix, at the scale tested. Ledger `docs/DATASET_HISTORY.md` §3.48 (Ledger ROUND 100); CLAUDE.md Known Bugs updated in place.
