# Correction: the "additive/linear basic-156" campaign was measuring MLPs (2026-07-18)

## The error

The 2026-07-18 "final dial metric" campaign
(`benchmarks/final_metric_experiments_2026-07-18.md`,
`docs/FINAL_DIAL_METRIC_DESIGN_2026-07-18.md`) repeatedly labelled its
`--max-features 156` bakes "**additive / linear basic-156**" and drew its central
conclusion from them: *"the additive basic core beats the full model at CID22
0.8978, at no diffmap or quality cost — go additive."*

**Every one of those bakes is actually a `156→128→1` LeakyReLU MLP**, verified with
`zenpredict inspect`:

| bake | n_inputs | n_layers | layer activations | additive? |
|---|--|--|---|---|
| `L1_linear_mf156` ("additive core", CID22 0.8978) | 156 | 2 | leakyrelu → identity | **✗ MLP** |
| `L1_linear_mf372` | 372 | 2 | leakyrelu → identity | ✗ MLP |
| `cl_linear` | 372 | 2 | leakyrelu → identity | ✗ MLP |
| `Eboth_basic156` | 156 | 2 | leakyrelu → identity | ✗ MLP |
| `Ebothg_hfgain_winsor` (the promoted "winner") | 156 | 2 | leakyrelu → identity | ✗ MLP |
| `b_sdr_linear_*` (Profile B) | 372 | 1 | identity | ✓ additive |
| `ADD156_*` (this correction) | 372 | 1 | identity | ✓ additive |

### Root cause

`zensim_mlp_train` **has no linear mode.** `--n-hidden-layers 0` is stored in the
hyperparameter struct but **never consumed by `zensim-train-core`** (grep: zero
references in `zensim-train-core/src/*.rs`), so the trainer always emits its
default `n_hidden=128` single-hidden-layer MLP. "basic-156" restricts *which*
features enter (`--max-features 156` → f0..155 only), NOT the linearity of the
combination. So "basic-input MLP" was conflated with "additive linear model" for
the whole campaign. The only genuinely-additive bakes in the program are the B
family (from the Python linear-projection solver, `n_layers=1` identity).

### The compromised sidecar field

`emit_bake_metrics.py` encoded the same fallacy: `diffmap_basic_fraction` returned
`1.0` for **any** bake with `n_inputs ≤ 156`, i.e. it reported the basic-input MLP
winner as "additive / exact-gradient-capable". Fixed 2026-07-18 — the sidecar now
records two DISTINCT properties:
- `additive` — all layers identity (exact-gradient diffmap). MLP → `false`.
- `basic_input_only` — reads only f0..155 (diffmap confined to the spatializable block).
- `diffmap_basic_fraction` — |w|/scale mass on f0..155, **defined only for additive bakes**.

`build_bake_index.py` and `bandwise_dashboard.py` updated to match.

## What a GENUINELY additive basic-156 actually scores

`scripts/v_next/additive_basic156_probe.py` builds the first real additive basic-156:
it imports the linear-projection owner (`MixGram` / `fit_spline_knots` /
`bake_candidate` — no new solver), slices the standardized Gram to its leading
`[:156,:156]` block (which IS the `w[156:]=0`-constrained least squares), solves
ridge/bvls/lasso in raw + winsor-"shaped" spaces across two training mixes, pads
weights back to 372 (zeros on the non-basic block), and bakes one identity layer.
All 12 verify `additive=True`, `diffmap_basic_fraction=1.0` (100% weight mass on
f0..155), evaluated on the full panel (bake_verdict, 10 corpora incl. LIVE/CSIQ/PIPAL).

**Results (CID22 held-out SROCC, dial-monotonicity on the quarantined_v2 grid):**

| additive basic-156 candidate | CID22 | dial | note |
|---|--|--|--|
| `safesyn_only_raw_lasso` | **0.8634** | 0.985 | best additive-156 so far |
| `safesyn_only_raw_ridge` | 0.8458 | 0.978 | |
| `safesyn_only_raw_bvls` | 0.8421 | 0.980 | |
| `cidmix_shaped_lasso` | 0.8563 | 0.973 | best of the cid-mix set |
| `cidmix_shaped_ridge` | 0.8292 | 0.957 | |
| `cidmix_shaped_bvls` | 0.8218 | 0.939 | |
| `cidmix_raw_ridge` | 0.8161 | 0.978 | |
| `cidmix_raw_bvls` | 0.8129 | 0.976 | |
| `cidmix_raw_lasso` | 0.8104 | 0.987 | |

(Reference points: additive-372 **B** 0.8764; basic-156 **MLP** winner 0.8939;
mislabeled `L1_linear_mf156` MLP 0.8978.)

### Secondary finding — less-is-more training mix for additive

`safesyn_only` (drop `cid22_train` + `kadid` from the mix) gives the BEST additive
CID22, ~+0.05 over the cid-heavy mix. This matches the standing note that the
ssim2-anchored `cid22_train` and analytic-distortion `kadid` groups pull a *linear*
model away from real CID22 rank (`cid22tr anti-correlates with real CID22` for
linear; `bigcodec mass poisons linear CID22`). The MLP absorbs those groups without
harm; the additive model cannot.

## The three claims — corrected

| design-doc claim | verdict | reality |
|---|---|---|
| "additive basic core beats full at CID22 0.8978" | **FALSE** | that was an MLP; real additive-156 ≈ 0.863 |
| "peak/max features f156–371 add nothing to CID22" | **FALSE** | additive-372 B (0.876) > additive-156 (0.863): they add ~0.013–0.02 |
| "additive is free — no quality cost vs the MLP" | **FALSE** | additive costs ~0.03–0.04 CID22 vs the MLP (0.894); the LeakyReLU does real ranking work |
| "additive → smooth dial via spline" | **TRUE** | additive + spline dial-mono 0.973–0.987; additive was never the dial problem |
| "additive → exact-gradient diffmap 0.987" | **TRUE** | derived from the additive formula `s_k=w_k/scale_k` (`diffmap_coherence_2026-07-16.md`); applies to any real additive model, NOT the MLP |

## The real closed-loop tradeoff (to surface to the user, not pre-decide)

There is a genuine tradeoff the campaign hid by comparing MLPs to MLPs:

- **Exact-gradient diffmap** (the closed-loop ideal) requires an **additive** scalar.
  Best additive: B (372-input, CID22 0.876) or additive basic-156 (CID22 0.863,
  diffmap confined to the spatializable f0..155 block). Smooth dial either way.
- **Best rank** is the **MLP** (CID22 0.894) — but its diffmap is input-dependent,
  needs a per-image backward pass, and its coherence has **never been measured**
  (the 0.987 number is additive-only). So promoting the MLP as the "closed-loop
  winner" was doubly unsupported.

Net: **additive costs ~0.03 CID22 to buy an exact, fixed, spatializable diffmap.**
Whether that trade is worth it is the user's call; it is not the settled "go
additive at no cost" the design doc asserted.

## Also landed this session

- **LIVE-R2** integrated as a held-out FR corpus (Sheikh 2006; 29 refs × 779
  distortions, JPEG+JPEG2000+blur+noise+fastfading; realigned DMOS + per-sample σ).
  Winner-MLP SROCC **0.9600**, additive B/ADD156 ≈ 0.95 (shaped). Builder:
  `build_fr_corpus_pairs.py live`. Also CSIQ (0.958) + PIPAL (0.624, GAN axis).
- All 24 corr-lq sidecars + 2 shipped-weight sidecars re-emitted with corrected
  `additive`/`basic_input_only` and the new LIVE/CSIQ/PIPAL panel columns.
