# STEERCHECK — recheck of the spatial-steering claims; steerability of the featpot candidate sets

Registered 2026-10-02 (Mountain Time, afternoon), before any fit. Lane `steercheck` (jj workspace, bookmark
`quarantine/claude/steercheck`, never pushed from the lane). Brief: `~/tmp/zensim-paper/rev4/STEERCHECK_brief.md`.
Results are appended below this registration in the same file once the runs finish.

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
