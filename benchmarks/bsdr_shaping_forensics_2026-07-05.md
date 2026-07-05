# How B SDR skipped the shapers — forensic timeline

Read-only forensics on `/home/lilith/work/zen/zensim` (git/jj colocated repo),
HEAD at time of investigation: `3131ec39` ("fix(profile-B): winsorize B SDR
via the EXISTING winsor_p99 transform op — not a new primitive"). All claims
below are cited to commit SHA or `file:line`; nothing is inferred without a
citation.

## (a) Root cause, one sentence

`ZensimProfile::B`'s shipped bake (`ens-Pline-cid80`) is a convex blend of
two **raw-feature-space** linear heads because the probe's ensemble
machinery structurally only blends raw-space fits (mixing spaces is
mathematically undefined for a linear blend), and — independently — every
**shaped**-space fit tried for every SDR-relevant data mix lost head-to-head
against its raw sibling on the probe's own train-legal selection axes, so
shaped SDR candidates never survived past the `fit` stage to be eligible for
anything, ensemble or otherwise; `BHdr` only carries the winsor/quantile/cbrt/
yeo-johnson shaping stack because its shaped fit *won* its selection axis
(UPIQ) for the unrelated reason of pure-HDR feature statistics — and at no
point in either lineage did anyone gate on the *raw output range* of the
candidate, so B's tail (raw min **−1131** on the 147k-row validation set)
was invisible to every SROCC-based check until a downstream dial-qualification
investigation stumbled onto it a day after ship.

## (b) Cited timeline

| Timestamp (local) | Commit | What happened |
|---|---|---|
| 2026-07-03 14:45:02 | `87b3ee25` | Lands `scripts/v_next/linear_projections_2026-07-03.py` (`gram`/`fit`/`finalize`). The `gram` phase accumulates moments in **both** `raw` and `shaped` feature spaces for every group (`cmd_gram`, script:130,152), and `cmd_fit` (script:333-365) solves **every mix × family in both spaces** (`for space in ("raw", "shaped")`, script:341) — this is where shaped SDR fits are actually computed, not skipped. Round-1/2 panel: `benchmarks/linear_projections_2026-07-03.md:131-212`. |
| 2026-07-03 17:23:25 | `ee850a95` | Same commit does **two** relevant things: (1) adds `cmd_ensemble` (script:563-698) with `HEAD_POOL` restricted to raw-space `.npz` fits only (script:479-490) and the explicit "RAW feature space only" design comment (script:475-478); (2) adds `cmd_residual` (script:747-837), whose comment at script:763-767 says *"the sparse HDR-fit head extrapolates wildly on OOD bigcodec rows (raw residuals reached +158 on valdigits...)"* about fit key `hdrmix-lasso0.002-raw` — **the exact same fit that becomes the 80%-weight "cid" component of `ens-Pline-cid80`** (`HEAD_POOL[0] = ("cid", "hdrmix-lasso0.002-raw", 0.0)`, script:480; `BASE_KEY = "hdrmix-lasso0.002-raw"`, script:744) — and mitigates it with an output clamp (`clamp_lo, clamp_hi = pa_anchor.min(), pa_anchor.max()`, script:770) for the residual-stack corpus, never applied to the ensemble path. The cid80/cid85 ensemble points themselves came from an ad-hoc "frontier probe" run interactively *after* seeing the pre-registered P-line panel (`benchmarks/linear_projections_2026-07-03.md:442-445`), not from the tuple literally committed in `cmd_ensemble` (`for a in (0.3, 0.5, 0.7)`, script:677 — unchanged since this commit, confirmed via `git log -p -S"Pline-cid"`). |
| 2026-07-03 19:55:32 | `f4b22cae` | MLP-vs-linear campaign closes in linear's favor ("bench: w8 closed"); does not touch the shaping question. |
| 2026-07-04 12:35:58 | `e96ee8f7` | "Campaign close-out sweep" — creates `benchmarks/provenance_best_results_2026-07-04.md`, declares the SDR pick (`lp_ens-Pline-cid80-anchored-f16`, 823 B, sha `7b326ac56a05c240`) and HDR pick (`lp_hdr-lasso0.001-shaped-anchored2-f16`, 11,684 B, sha `373eac56e7a07d6d`) as the campaign's winners (provenance doc:11-19). No raw-range figure appears anywhere in this doc at this point. |
| 2026-07-04 12:41:04 | `fe8b00aa` | **Ships** `ZensimProfile::B` and `ZensimProfile::BHdr`. `zensim/src/profile.rs` (original diff): the `linear_bake_bhdr_shaped()` doc comment explicitly states *"carries `zentrain.feature_transforms` metadata so the runtime dispatches `predict_transformed`"*; the sibling `linear_bake_b_cid80()` doc comment says nothing about transforms in either direction. Adds smoke test `profile_b_tests::profile_b_loads_scores_and_holds_identity` asserting `s.is_finite() && s < ident && s > -50.0` — but the fixture is 64×64 **synthetic per-pixel hash noise** with a uniform every-5th-byte darkening (profile.rs, added by this commit), not real downscaled screen content. This is 5 minutes after `e96ee8f7`. |
| 2026-07-04 12:46:33 | `e3438a71` | Adds entry-path routing between `B` and `BHdr`; not related to the tail. |
| 2026-07-04 23:38:02 | `d7da8ffd` | "Profile-B knob qualification" — G2 section reports *"Spline: 18 knots, raw-x [−1.974, 1.138] (a real input range...)"* (provenance doc:136-137). This range is computed from the **2,000-row curated anchor set** (`multiband_anchor_dial100.parquet`) used for spline-knot fitting, not from `bigcodec_val` (147k rows) where the pathological rows actually live — so even this first "range" observation could not have caught the tail. Also flags *"sole blocker = 3 webp low-ceiling ladders"* — first hint of trouble, ~11 h post-ship. |
| 2026-07-05 00:08:52 | `2e655672` | "webp trio = B-specific OOD blindness" — observes raw B scores around −81 on three named webp renditions; **misdiagnosed at this point** as content-class OOD blindness, proposes "corpus refit" or "knob bottom-floor" as fixes. |
| 2026-07-05 00:11:22 | `64e432fb` | Refines the read against butteraugli; confirms the magnitude pathology is real on all three named images, still not yet traced to a specific feature. |
| 2026-07-05 00:41:38 | `ae4209a8` | **"bench(w11)"** (the commit this task's own directory `w11-webp-ood` is named for) — separates a *second*, unrelated bug: 9/115 dial-grid ladders (8/24 webp) carry bit-constant garbage in masked/IW-pool features from a `zensim-gpu` odd-dimension extractor defect. Declares shipped B "sane+ordered" on fresh pixels for the three named images and falsifies "corpus refit" as a fix. At this point the investigation could plausibly have stopped here (data-corruption bug found and quarantined) — it didn't. |
| 2026-07-05 01:15:40 | `06deaa0e` | **"f155 tail forensics"** — separates the real residual bug from the GPU-corruption bug: feature `f155` has a genuine heavy tail on tiny (36–384 px) screen-content renditions. Concrete examples in `/mnt/v/output/zensim-multicodec-probe/w11-webp-ood/f155_offenders.json`: `o_8301.png.scale54x96` → f155 = 14,532.53 vs fit-corpus p99.9 = 0.479; corresponding raw B scores in `tail_inventory.json` reach **−938.52** on that same row. Reaches the WINSORIZE verdict (provenance doc:213-223, as first written by this commit). |
| 2026-07-05 01:50:14 | `3131ec39` (HEAD) | **The fix.** Re-bakes B SDR with 372 `winsor_p99` `zentrain.feature_transforms` entries (fit-corpus [p0.1, p99.9] per feature), keeping weights/scaler/spline unchanged. New bake `b_sdr_linear_cid80_winsor_2026-07-05.bin` (12,891 B, sha `b92b0b7a…`). Measured via `bake_verdict` (which auto-dispatches `predict_transformed`): raw min **−1131 → −1.86** (commit message; profile.rs:755-756 says −1131→−1.86 as well), CID22 0.8732→0.8763 (commit msg) / 0.8762 (profile.rs:756, same measurement rounded slightly differently across the two write-ups), KonJND 0.5434→0.5474/0.5470. Adds regression guard `both_b_bakes_carry_winsorizing_transforms` (`zensim/src/profile.rs:1127`) that fails loud if either B bake ever loses its `feature_transforms` metadata again. |

## (c) Code-level reason the ensemble pool excluded shaped fits

`scripts/v_next/linear_projections_2026-07-03.py:475-478`:

```python
# Diverse head pool for convex ensembles. RAW feature space only — a convex
# blend of raw-space linear heads collapses to a SINGLE 372->1 linear layer
# (v_k = w_k/sd_k in raw space; scaler folds away), so the ensemble bakes as
# one tiny layer. Shaped heads are excluded (incompatible input transform).
```

This is a genuine mathematical constraint, not an arbitrary exclusion: raw-space
standardization (`x' = (x-μ)/σ`) is affine, so a convex blend of affine-standardized
linear heads is itself one affine map and folds into a single `372→1` layer
(`_load_head_rawspace`, script:545-554, does exactly this fold: `v = w/sd`,
`c = bias - mu@v`). Shaped-space transforms (`winsor_p99`, `quantile_bins`,
`signed_cbrt`, `yeo_johnson` — defined in `scripts/v_next/train_v02_bvls_shaped.py:134-176`,
reused verbatim per script:62-70) are **nonlinear per-feature maps**; a blend of
two heads that each depend on a *different* nonlinear pre-transform of the same
raw feature cannot be re-expressed as one linear layer over the raw features.
`HEAD_POOL` (script:479-490) accordingly lists eight heads, every one of them a
`-raw` fit key (`hdrmix-lasso0.002-raw`, `canonhdr15-bvls-raw`,
`canonhdr15-lasso0.0005-raw`, `canon-ridge1e-05-raw`, `canon-bvls-raw`,
`hdr-lasso0.001-raw`, `pjnd-ridge0.001-raw`, `hdrmix300-lasso0.002-raw`) — there
is no shaped key anywhere in the pool, and no code path in `cmd_ensemble`
(script:563-698) that could combine one.

**Separately and independently confirmed:** shaped SDR fits were not merely
excluded from ensembling — they were *fit and compared to raw on the
train-legal selection axes* by `cmd_fit`, and lost. Direct evidence from
`/mnt/v/output/zensim-multicodec-probe/linear-probe/fit.log` (round 1, `canon`
mix — the ancestor mix of the `kad`/`cbv` ensemble-pool heads and the
"2026-05-28 recipe"):

```
canon-ridge1e-05-raw     bigval=+0.8870  hdrval=+0.8741  hdrmixval=+0.8985  guard=0.0822
canon-bvls-raw           bigval=+0.8489  hdrval=+0.8484  hdrmixval=+0.8725  guard=0.1655
canon-lasso0.001-raw     bigval=+0.8843  hdrval=+0.8554  hdrmixval=+0.8797  guard=0.0960
canon-ridge1e-05-shaped  bigval=+0.8289  hdrval=+0.8332  hdrmixval=+0.8509  guard=0.0592
canon-bvls-shaped        bigval=+0.8258  hdrval=+0.8431  hdrmixval=+0.8655  guard=0.0556
canon-lasso0.001-shaped  bigval=+0.8401  hdrval=+0.8089  hdrmixval=+0.8239  guard=0.0841
```

Raw beats shaped on every axis for every family. The same pattern holds for
`canonhdr15` and `big` in `/mnt/v/output/zensim-multicodec-probe/linear-probe/fits/table.json`
(e.g. canonhdr15-bvls bigval 0.8452 raw vs 0.8224 shaped; big-bvls 0.8947 vs
0.8796). This is exactly the basis for the campaign doc's documented negative
result: *"Input shaping (Yeo-Johnson TSV) loses on every axis for SDR mixes
(confirms 2026-05-28) — EXCEPT pure-HDR"* (`benchmarks/linear_projections_2026-07-03.md:307-311`),
repeated in the top-level falsification list of `benchmarks/provenance_best_results_2026-07-04.md:113`
("input shaping outside pure-HDR"). Because shaped SDR fits lost at the `fit`
stage on the selection axes (bigcodec_val, hdr_val, hdr_valmix, konjnd_guard —
`SEL_AXES`, script:492), they were never promoted to `finalize` (baking +
full `bake_verdict` panel), and so never became `HEAD_POOL` candidates in the
first place — the raw-only ensembling rule and the raw-fits-won-anyway result
are two independent, mutually reinforcing reasons B ended up raw, not one.

## (d) The eval gap that let the tail through

No stage of the selection or verification pipeline ever computed a raw-output
range, percentile, or outlier check on a *candidate* bake before shipping:

- `val_metrics()` (script:283-290) — the function every fit and every
  ensemble candidate is scored by — computes only `spearmanr(pred, y)` per
  validation set. No min/max/percentile of `pred` is ever captured or
  returned.
- `cmd_fit` (script:333-365) prints/logs only `bigval`, `hdrval`, `hdrmixval`,
  `konjnd_guard` per candidate (script:360-362) — again, all rank statistics.
- `cmd_ensemble` (script:563-698) selects purely on the four `SEL_AXES` (rank
  correlations, script:492, 597-603) plus corner-normalized scalarizations
  (script:613-618) — no output-range term anywhere in any scalarization.
- `bake_candidate` (script:396-450) fits the PCHIP dial spline against the
  anchor set and bakes — it computes `raw_a` (script:411) only to derive
  spline knots, never reports its range as a gate.
- The campaign doc's full panel tables (`benchmarks/linear_projections_2026-07-03.md:131-149,189-212,447-464`)
  and the provenance doc's candidate table (`benchmarks/provenance_best_results_2026-07-04.md:11-19`)
  report exclusively SROCC-family numbers (CID22/KADID/TID/KonJND/AIC-3/AIC-4/UPIQ,
  plus `G1`/`mono`/`goal`) — never a raw min/max column.

Crucially, the pathological rows were **not** absent from the validation data used
to select and verify the candidate — `bigcodec_val` (147,067 rows, one of the
five `SEL_AXES`) is the exact set on which the −1131 raw output was later
measured (`3131ec39` commit message; provenance doc:194 — "Real-input fragility
(bigcodec_val heavy tails f155/f52/f216, 0.95% below raw −2, min −938) is
documented in §w11"). SROCC over 147k rows is essentially insensitive to a
sub-1%-of-rows tail of extreme-but-still-correctly-ordered outliers — the
metric that gated every decision in this campaign structurally cannot see the
failure mode that shipped. This is a gap in the *harness*, not a missed data
source.

The one place a raw-range number *was* reported pre-tail-discovery
(`benchmarks/provenance_best_results_2026-07-04.md:136-137`, from `d7da8ffd`,
2026-07-04 23:38, ~11 h after ship) — *"Spline: 18 knots, raw-x [−1.974, 1.138]
(a real input range...)"* — was computed from the 2,000-row curated
`multiband_anchor_dial100.parquet` anchor set (`bake_candidate`, script:409-411),
which does not contain the tiny-screen-content renditions where `f155`
degenerates; so even this one range observation could not have caught the bug.

The one place a raw-output *safety mechanism* did exist pre-ship is
`zensim/tests` via `profile_b_tests::profile_b_loads_scores_and_holds_identity`
(added in `fe8b00aa`, same commit as the ship): it asserts
`s.is_finite() && s < ident && s > -50.0` on a distorted synthetic image. Had
this fixture exercised real downscaled screen content it would have caught
the bug outright (−1131 ≪ −50). It did not, because the fixture is
per-pixel hash noise at 64×64 with a uniform darkening distortion — nothing
like the "scale-pyramid degeneracy at ~50-100px" (`benchmarks/provenance_best_results_2026-07-04.md:220`)
that triggers `f155`'s real tail. A bound check existed; its test-fixture
coverage did not overlap the failure domain.

One layer deeper: `PROFILE_B`'s `ProfileParams` (`zensim/src/profile.rs:778-796`)
sets `soft_clamp_score: false` and `extrapolate_score: true` — identical to
`PROFILE_B_HDR` (`profile.rs:802-820`) — so neither the score-level soft clamp
nor a bounded-extrapolation switch was engaged for either profile at ship
time; the only thing that differed between the two profiles' exposure to OOD
input was the presence or absence of `feature_transforms` metadata on the
bake itself.

**The knowledge that raw-space linear heads can extrapolate wildly on OOD rows
already existed in this exact codebase, one commit before ensembling.**
`cmd_residual` (`ee850a95`, same commit as `cmd_ensemble`), script:763-767:

```python
# Clamp the base pred to the anchor-observed domain (the dial spline's
# trusted raw range). Without this, the sparse HDR-fit head extrapolates
# wildly on OOD bigcodec rows (raw residuals reached +158 on valdigits and
# the outliers squashed the OLS slope to 0.17). The clamp is exactly
# replicable at runtime (two constants, recorded in the manifest).
```

This clamps the *output scalar* of fit key `hdrmix-lasso0.002-raw` to the
anchor-observed min/max before using it as a residual-stack base predictor.
That fit key is **the same head that supplies 80% of `ens-Pline-cid80`'s
weight** (`HEAD_POOL[0] = ("cid", "hdrmix-lasso0.002-raw", 0.0)`, script:480;
`BASE_KEY = "hdrmix-lasso0.002-raw"`, script:744 — confirmed identical string
via `git blame -L 763,771`, both attributed to `ee850a95`). The OOD-extrapolation
failure mode of this exact head was identified and mitigated once, for the
residual-stack corpus — and never cross-applied to the ensemble/`Profile::B`
path, which carries the same head unclamped straight through
`_load_head_rawspace` (script:545-554) into the shipped bake.

## (e) Reasoned tradeoff or unexamined default?

Both, at different layers:

- **The raw-only ensembling restriction is a reasoned, correctly-justified
  engineering decision.** It is explicit in code comments (script:475-478),
  mathematically sound (nonlinear per-feature shaping cannot fold into a
  linear blend), and its consequence (shaping loses on SDR selection axes
  anyway) is independently measured and documented as a negative result
  (`benchmarks/linear_projections_2026-07-03.md:307-311`,
  `benchmarks/provenance_best_results_2026-07-04.md:113`). Nobody skipped
  evaluating shaped SDR fits — they were fit, logged, and lost fairly.

- **BHdr's shaping is an accident of which axis its shaped fit happened to
  win**, not a deliberate "HDR needs robustness" decision. The `gram`/`fit`
  phases apply the identical raw+shaped sweep to every mix, SDR and HDR
  alike (script:130,152,341), reusing transform code verbatim from an
  unrelated, five-weeks-earlier experiment (`train_v02_bvls_shaped.py`,
  landed `5da05be1`, 2026-05-28; imported at script:62-70). The shaped HDR
  fit (`hdr-lasso0.001-shaped`) simply won UPIQ decisively (0.7313 vs 0.6488
  raw sibling, `benchmarks/linear_projections_2026-07-03.md:143-146,309`)
  while every shaped SDR fit lost its own axis. Tail-safety was never the
  reason BHdr kept its transforms — general-purpose robustness against
  extreme inputs was a side-effect nobody named until 2026-07-05.

- **The downstream consequence — that whichever profile ends up raw has zero
  tail protection — was an unexamined blind spot**, not a considered
  tradeoff, right up until it was accidentally rediscovered by an unrelated
  investigation (a webp dial-quality anomaly hunt) a day after ship. No commit,
  doc section, or test between `87b3ee25` and `fe8b00aa` frames "does the
  raw-space pick need its own OOD guard, since it doesn't get one from
  shaping" as a question to answer. The one place in the whole campaign
  where that exact question WAS asked and answered — `cmd_residual`'s
  output clamp on the very same underlying head, in the very same commit
  as the ensembling code — answers it for a sibling artifact and was never
  connected to the artifact that shipped as `Profile::B`. That is the
  sharpest, most specific evidence that this was a genuine miss rather than
  an examined-and-accepted risk: the mitigating idea existed in the
  session's own working memory, in the same file, and simply wasn't
  threaded through to the ensemble/ship path.

## Files referenced

- `scripts/v_next/linear_projections_2026-07-03.py` (script line numbers
  above refer to the file as of HEAD `3131ec39`; the file has not changed
  since `f4b22cae`, confirmed via `git log --follow`)
- `scripts/v_next/train_v02_bvls_shaped.py` (shaping transform source, landed `5da05be1`)
- `benchmarks/linear_projections_2026-07-03.md`
- `benchmarks/provenance_best_results_2026-07-04.md`
- `zensim/src/profile.rs` (current `linear_bake_b_cid80`/`linear_bake_bhdr_shaped`/`PROFILE_B`/`PROFILE_B_HDR` at lines 747-820; regression test at 1127)
- `/mnt/v/output/zensim-multicodec-probe/linear-probe/fit.log`, `fits/table.json` (raw-vs-shaped selection-axis evidence, not committed to git)
- `/mnt/v/output/zensim-multicodec-probe/w11-webp-ood/f155_offenders.json`, `tail_inventory.json` (concrete tiny-screen-content tail examples, not committed to git)
