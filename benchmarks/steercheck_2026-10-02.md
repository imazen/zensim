# STEERCHECK — recheck of the spatial-steering claims; steerability of the featpot candidate sets

Registered 2026-10-02 (Mountain Time, afternoon), before any fit. Lane `steercheck` (jj workspace, bookmark
`quarantine/claude/steercheck`, never pushed from the lane). Brief: `~/tmp/zensim-paper/rev4/STEERCHECK_brief.md`.
Results are appended below this registration in the same file once the runs finish.

> **Correction (2026-10-04 03:40 MT).** The steering pass counts in this record (broad 96 / owner 12) come from the COSTSET
> strict-route human-only bakes (H128, 8,000 human pairs, no SafeSyn/CID22/coverage legs). Those bakes were later found
> non-monotone under per-block JPEG upgrades (`neighsteer_2026-10-04.md`, correction banner and §7), so pass counts measured with
> them do not rank the feature sets as an adopted-recipe model would. With featpot cv16:cf98 bakes (densified, stamped Rev4, uniform
> three-seed ensembles): broad by_v2fy 92/96 and v2 + basic 91/96 (COSTSET bakes: 86 and 87-88); owner 12/12 each
> (`jpegsteer_2026-10-04.md` banner). Cost and rank numbers here are unaffected. Conclusions that rest on a few broad passes between
> sets (e.g. a set "losing" 6-14 passes) should be re-measured on adopted-recipe bakes before use.

## Registration (Part B)

Recipe: `benchmarks/steercheck_2026-10-02.recipe.json` (committed with this text, before any extraction or fit).

* **Owner.** The existing feature-screen owner (`scripts/run_full_eval.sh --stage feature-screen`,
  `scripts/lib/feature_screen_ceiling.py`) at the checkout this lane starts from (`main` a36fdf01). No second evaluator.
* **Why the strict v2 route and not the 09-13 recipe form.** The 09-13 steerable screen ran the v1 fit/dev/test recipe. Commit
  `a9a2cea0` (2026-09-13) retired that form: the owner now refuses v1 recipes and any `test` segment, and the later DATA_SPLITS
  clarification allows a published-TEST read only for a frozen candidate. A subset screen is not a frozen candidate, so this
  lane does not reopen the v1 form. It uses the v2 train/eval-only recipe with the SAME hash-pinned admitted segments that
  `benchmarks/minimal_top_2026-09-13.recipe.json` already used: 8,000 train pairs (human), 3,125 eval pairs (the pairs the
  09-13 screen called its test partition), and the 4-case eval spatial manifest. In this route the trainer receives train
  tables only, runs no early stopping and selects no checkpoint on eval, so eval does not steer the fit. The eval pairs were
  nevertheless used to choose subsets in the 09-13 study and in the featpot design work: development evidence, not a fresh
  holdout.
* **Tasks: human only.** No admitted train/eval segment exists for codec-proxy or corruption. The 09-13 codec/corruption tables
  are inner splits of training-origin data labelled `test`, which `DATA_SPLITS.md` says is not authorization to reuse. Building
  new admitted segments is a new data admission, which the brief excludes. Codec and corruption errors are therefore NOT
  measured here. This is a gap against the brief, stated again in the results.
* **Subsets** (explicit canonical Rev3 ID lists, no reordering): `basic228` (0-227), `basic156` (0-155), `v2basic` (0-155 and
  372-719, 504 columns), `full944` (0-943), `v2only` (372-719), `basic_masked` (0-155 and 228-299), `basic_iw` (0-155 and
  300-371), `v2basic_masked` (0-155, 228-299, 372-719), `v2basic_nobandfrag` (v2basic minus the 12 BANDING slots
  `372+87s+29c+27` and the 12 PJND_FRAGILITY slots `372+87s+29c+21`, s=scale 0-3, c=channel 0-2; 480 columns), `local120` and
  `y60` (the two spatial controls, ID lists copied from the 09-13 steerable and minimal-top recipes).
  "masked" is f228-299 and "iw" is f300-371, as the design log's group definitions have it.
* **Recipe.** As the 09-13 screen: H128, 32 epochs, 8,192 pairs per epoch, seeds 5101/5103/5107, fresh Rev3 extraction of the
  pinned pairs by the owner's `prepare` stage (its `features.csv` is compared with the minimal-top extraction as an identity check).
* **Spatial gate.** (1) The owner's audit: eval-source cases at block 32, M2 >= .99 and M3f >= .70, an UNSUPPORTED result is
  not a pass. (2) The broad panel of the 09-13 steerable study (`PIXEL_REGISTER.json` of the 2026-09-08 max-attribution work:
  24 pairs from eight training origins, three JXL distances, blocks 8/16/32/64 = 96 cases per candidate), run through the
  current `diffmap_block_coherence` with the uniform three-seed ensemble of each candidate, same gates. Those pairs are
  TRAIN-role content, so this is a development diagnostic of steering behaviour, not a held-out claim.
* **Reported per subset:** human eval MAE and SROCC (three-seed median and range), spatial passes on both panels, worst M2 and
  M3f, count and IDs of unsupported refinement terms, plus the Part A support table that says which ID ranges can be steered at all.
* **Untouched:** CID22 gold, AIC, every sealed set. No fit, selection or calibration uses eval rows.

## Results (2026-10-02, Mountain Time evening; build = main a36fdf01 + the owner `source_diff` fallback)

Everything below is development evidence on Rev3 tables. Not qualified; nothing here selects a shipping model.

### Part A — claims rechecked against current source and measurement

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | masked/IW (f228-371) have no spatial support | **CONFIRMED** | `candidate_map_sensitivities` lists every f228-371 ID unsupported; broad panel: all 96 cases of `basic_masked`, `basic_iw`, `v2basic_masked`, `full944` are UNSUPPORTED (72 IDs per masked-or-IW block, 144 for full944). |
| 1 | per-ID support, measured | **MEASURED** | Probe `id_support_probe.rs` (unit sensitivity on one ID, |density| mass through the buffered and the fused-944 entries, Rev3, 192x160 test pair): 629 of 944 IDs give mass; both entries agree on every ID. Zero mass: f155-371 except the L8 slots below (f156-227 slots 3-5 of each six DO have mass; the 36 hard-max slots 0-2 have none in the density but are served by the finite max-removal refinement, `bind_max_removals`), all 12 PJND_FRAGILITY slots (reference-only), and in append/append2 the reference-only and HDR-gated slots plus further append cells (exact list in `probe_support.json`; not analysed per signal). **All other v2 signals: 12 of 12 cells each have mass, including BANDING, BLOCKINESS, RINGING, EDGE_WIDTH_CHANGE.** A handful of basic cells (f10-11, f23-24, ... f142) read zero on this one probe image; that is a measured zero on one pair, not a design statement. f944+ (csfw, dvifm, research families) have no integrand at all: the slice stops at 944. |
| 1 | the prepared API's range | **CHANGED vs the 09-13 doc** | `BakeScorer::prepare_steering` (the product surface) refuses ANY bake that reads an ID >= 228, v2 included: "steering session currently supports basic/peak feature IDs below 228". v2/append/append2 steering exists only on the older `compute_with_ref_and_attribution` + `Fused944Session` path that `diffmap_block_coherence` and the owner's audit use. The 09-13 steerable-subset results used the prepared path, which is why its subsets were all < 228. |
| 2 | v2 spatial prediction changed since 09-13 | **REFUTED (no change)** | Retained audit `human-basic_v2_coarse-h128-full-s5101--jxl_2010`: re-run with the current `diffmap_block_coherence`: M2 0.9967874231032124 and M3f 0.07491455912508543, base score, M3a, 56 interventions and the 36-ID density-unsupported list all identical to the 09-13 file. Attribution/diffmap commits since 09-13 (09-14 finite moments/maps, 09-25 gating, 09-26 featcanon) did not move this case. |
| 2 | "1/18" vs "18/18 fail at 619" | **RECONCILED** | Different runs and different bars. The ceiling study's 619 layout (`fine_y_v2_half`), scored with M3f >= 0.9, passes 1 of 18 human checks (the 3/1/1/3 for 514/476/619/800 reproduces exactly at M3f >= 0.9); at today's M3f >= 0.70 the same stored files give 5/5/4/6 of 18. The native-extraction 619 in `scales944/native` fails 18 of 18 at either bar, and that is the "18/18" in the plan doc. Neither number is a count of the same thing: pooled over 3 seeds x 6 non-swap cases, different extraction, different thresholds. |
| 3 | F4: SSIM `d` unbounded, f313 = 5.8e6 | **CHANGED** | Rev2/3/4 select `SsimLumaForm::Clamp` (d in [0,2]); the 5.8e6 is a Rev1-era value from the bigcodec sweep, which fired on none of 217,756 real rows. Measured on read-only tables (`f4f15.py`): max of f313 = 0.95 (Rev3 human fit), 1.02 (Rev3 corruption fit), 1.22 (Rev4 human), 0.69 (Rev4 safesyn, 141,054 rows); the largest value anywhere in f228-371 is 2.69 (Rev3 corruption) and 1.67 (Rev4). No column exceeds 10. Rev4 additionally bounds negative tail bins (`c3negfold`). |
| 4 | F15: PJND_FRAGILITY constant 1.0 | **CONFIRMED only for a v1-only walk** | On the full-walk tables it is not constant: Rev3 human 0.0-0.934 (58 distinct values per slot), codec/corruption 0.0-1.0; Rev4 0.064-1.0 (55-2,312 distinct per slot). The source (`DEFECT_F15`) now says the constant 1.0 belongs to a v1-only walk (zeroed accumulators); the identity-pair value is a reference property, not a defect. The slot is reference-only, so it has no steering integrand either way. |
| 5 | Rev4 bakes cannot be served or steered | **CONFIRMED, reason stated** | `refuse_rev4_served` fires on every `Zensim`, `BakeScorer`, HDR, diffmap, attribution and corruption-head entry. Reason: Rev4 is canonical only on `research::extract`; the served paths still run SIMD-tier-dispatched leaves (`color::linear_to_pu_xyb_planar_into`, the edge-only `blur::fused_blur_h_mu` route in `streaming`, and `attribution::attr_pass_b_*`), so a Rev4 score or map would depend on the CPU tier, and `BakeScorer::ensemble` also refuses a Rev4-declared bake or any bake in a Rev4 process. To serve or steer a Rev4 bake those three leaf families must be made canonical (the featcanon arithmetic) and the refusal removed with parity gates; whether the attribution integrands also need changes for Rev4 arithmetic was not examined. Until then the featpot v2c bakes can be scored only through `research::extract` + `predict_features_with_bake`, with no map. |

### Part B — steerability of the candidate sets

Human task only; eval = 3,125 pairs (development evidence; the pairs were already used to choose subsets), three seeds
(5101/5103/5107), H128, 32 epochs, train-only fits. "Owner spatial" = the owner's audit, 4 eval-source cases x 3 seeds = 12 checks at
block 32. "Broad" = 96 cases per candidate (24 TRAIN-role pairs x blocks 8/16/32/64), uniform three-seed ensemble, M2 >= .99 and M3f >= .70,
UNSUPPORTED is not a pass. Codec and corruption errors: NOT MEASURED (see registration). Per-seed values: `owner_results.json`;
broad per-case rows: `~/tmp/zensim-paper/rev4/steercheck-artifacts/broad_RESULT.json` (over 30 KB, not committed); summary `broad_summary.json`.

| Subset | cols | eval MAE (median, range of 3) | eval SROCC (median) | owner spatial (of 12) | broad pass / fail / unsupp. (of 96) | worst M2 / M3f (broad) | unsupported IDs |
|---|---:|---|---:|---|---|---|---:|
| v2basic | 504 | 8.201 (see json) | 0.9314 | 2 pass, 10 fail | 40 / 56 / 0 | 0.804 / -0.371 | 0 |
| v2basic_masked | 576 | 8.223 | 0.9301 | 12 UNSUPPORTED | 0 / 0 / 96 | 0.821 / -0.231 | 72 |
| v2only | 348 | 8.262 | 0.9324 | 4 pass, 8 fail | 24 / 72 / 0 | 0.909 / -0.345 | 0 |
| full944 (R0) | 944 | 8.272 | 0.9266 | 12 UNSUPPORTED | 0 / 0 / 96 | 0.839 / -0.279 | 144 |
| v2basic_nobandfrag | 480 | 8.476 | 0.9290 | 3 pass, 9 fail | 27 / 69 / 0 | 0.941 / -0.161 | 0 |
| basic156 | 156 | 8.641 | 0.9258 | 12 pass | 83 / 13 / 0 | 0.762 / 0.625 | 0 |
| basic228 (core) | 228 | 8.993 | 0.9254 | 11 pass, 1 fail | 86 / 10 / 0 | 0.930 / 0.483 | 0 |
| local120 | 120 | 9.104 | 0.9214 | 11 pass, 1 fail | 78 / 18 / 0 | 0.615 / 0.631 | 0 |
| basic_masked | 228 | 9.107 | 0.9242 | 12 UNSUPPORTED | 0 / 0 / 96 | 0.895 / 0.476 | 72 |
| y60 | 60 | 9.124 | 0.9237 | 12 pass | 92 / 4 / 0 | 0.972 / 0.817 | 0 |
| basic_iw | 228 | 9.210 | 0.9218 | 12 UNSUPPORTED | 0 / 0 / 96 | 0.657 / 0.371 | 72 |

Reading, stated plainly:

* **v2 + basic is the best human-rank set here (MAE 8.20, SROCC 0.9314) and fails steering.** 56 of 96 broad cases fail and 10 of 12 owner
  cases fail, with worst M3f -0.37: its rectangle predictor can rank repairs in the wrong order. M2 stays high, so the local linearization
  is fine and the failure is in the feature-to-rectangle prediction (as the 09-13 record said). Every v2 signal has integrands, so this is
  not an UNSUPPORTED artefact: it is a measured spatial failure of supported terms.
* **R0 (944), and every set containing masked or IW, cannot be steered at all** (UNSUPPORTED in all 96 cases). Adding masked or IW to
  basic did not buy rank either (MAE 9.11 / 9.21 vs 8.64 for basic156).
* **Removing BANDING and PJND_FRAGILITY did not improve steering** (27/96 vs 40/96 passes, with three seeds) and costs 0.27 MAE.
* **The steerable sets with the best broad passes are the narrow ones: y60 92/96, basic228 86/96, basic156 83/96, local120 78/96.** Their
  eval MAE is 0.4-0.9 above v2basic. y60 is the only candidate whose worst M3f (0.817) clears the bar everywhere it matters; its four failures
  are M2 or M3f misses at block 64.
* The MAE differences between subsets are 3-seed medians on one eval set that was already used for selection; seed ranges overlap for
  several adjacent rows. The ordering of adjacent subsets is not established; the gap between the v2 sets (about 8.2) and basic228 /
  local120 / y60 (9.0-9.1) is larger than the spread.
* This is Rev3 data. The featpot candidates in the design log are Rev4 (v2c bank); Rev4 v2 numerics differ, and no Rev4 bake can be served
  or steered (Part A #5), so the spatial result for the actual v2c bakes remains unmeasured.

### Reproduction
Tools built at main a36fdf01 into `/var/tmp/steercheck/target`, extractor into `target-bench` with `--features training,zen-decode`. Owner stages:
`scripts/run_full_eval.sh --stage feature-screen benchmarks/steercheck_2026-10-02.recipe.json /var/tmp/steercheck/screen --ceiling-stage prepare|fit|audit|report`
(the fresh `features.csv` is byte-identical to the minimal-top extraction, sha256 28ec0ff7...); broad panel `benchmarks/steercheck_2026-10-02/run_broad.py`
with `ZENSIM_FORMULA_REV=3` and NOT the prepared-steering env (that path refuses IDs >= 228). Not done: codec and corruption tasks, per-seed broad
panels, Rev4 spatial, any v2-pool restriction (the E11 spatial track needs the Part A table above).
Pre-existing failing test, not touched: `scripts/tests/test_feature_screen_splits.py::test_targeted_capacity_controls_do_not_expand_other_layouts` (KeyError `epochs`).
