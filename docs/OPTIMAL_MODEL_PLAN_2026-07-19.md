# Optimal global model + feature validation + diffmap completion — plan & state (2026-07-19)

THE forward plan after the feature-v2 program's lab-scale phase. Also the
**post-compaction resume point** for that work (see §D). Everything measured in
this session is preserved in the docs cited here — this file is the index + the
next steps, not a re-derivation.

## Why this exists (the methodological pivot)

A feature's worth is **marginal and conditional on the best model** — never its
isolated correlation, and never its behaviour in a seed-broken lab recipe. Every
feature-v2 evaluation so far violated that: the A/B + ablations ran on the lab
recipe (single seed, **epoch-0 checkpoints**, ±0.10 seed swings — CID22/LIVE
flipped sign between seeds; `benchmarks/v2_trainability_ab_2026-07-19.md`). So we
do NOT yet know which v2 features are load-bearing.

**The fix, in order:** (1) build the *optimal global model* on the full 720
feature space under a production recipe that trains past epoch 0; (2) validate
each feature as a **marginal component** (leave-one-out / `s_k` sensitivity) at
that optimum, full Mohammadi panel, multi-seed; (3) complete the runtime diffmap
**only** for the load-bearing survivors. Building diffmap machinery for features
the optimal model doesn't use is premature optimization of the sacred-pixel path.

The coherence work de-risks this: deprecating v1's non-spatializable block cost
~0 on compression and the spatializable subset carried the signal (ext-lumacoh =
100% spatializable at ~0 compression cost), so the **optimal *spatializable*
model ≈ the unconstrained optimum** — feature-validation and diffmap-inclusion
will very likely converge on the same set.

## A. The fleet extraction (get 720 features on the production corpus)

**Feature space:** append-only 720 = frozen v1-372 ++ v2-348 (`f372..f719`).
Extractor exists: `zensim/examples/v2_ab_extract.rs` (single-pass, both blocks on
the same decoded pixels — the join was eliminated after `(ref,human_score)` proved
non-unique for kadid/aic3).

**The fleet already emits 720 — no new JobKind.** As of 2026-07-19 the zenmetrics
executor (`zenmetrics-cli/src/jobexec.rs` → `metrics/zensim.rs::score_with_features_v2ab`)
routes every `metric=zensim`/`zensim-gpu` request to the CPU path that computes
v1 (`compute_zensim_with_config`, extended+iw) AND v2 (`Zensim::compute_v2_features`)
on the same reflect-padded pixels and concatenates → a 720-length `feature` row
alongside the metric row. The GPU zensim kernel was deliberately disabled (it only
implements v1). `cpu-metrics` (a default feature) pulls in `zensim/feature-regime-v2`
automatically. `JobKind::Feature{regime}` exists in the type system but is NOT
implemented in the executor — use `ScoreFile`/`Metric` with `metric=zensim-gpu`.

**⚠ BLOCKER (must fix before any Hetzner/CPU run):** the CPU executor image
`ghcr.io/imazen/zenfleet-worker:exec` is STALE — last pushed 2026-07-19T19:34Z,
*before* the V2Ab commit `428b4aeb` (21:13Z). A run against it today silently
produces **372**-feature rows, no error. Rebuild + push first:
```
cd ~/work/zen/zenmetrics
cargo build --release -p zenmetrics-cli --no-default-features \
  --features sweep,png,jpeg,webp,avif,jxl,cpu-metrics
PUSH=1 bash scripts/jobsys/build_executor_image.sh
# verify: docker run --entrypoint zenmetrics <img> jobexec --help
```
(The GPU image `:exec-gpu-zensimv2-6b3619a621` IS current + smoke-tested; but this
is CPU-bound work — Hetzner CPU per project memory, not vast GPU.)

**Declare → launch → monitor → teardown** (`zenmetrics/docs/RUNNING_JOBS.md`):
```
target/release/zenfleet-ctl declare --spec <spec.json> --out <manifest.json>
bash scripts/jobsys/launch_fleet.sh <N_JOBS> <HETZNER_X86> 0 0 0   # Hetzner-only
bash scripts/jobsys/fleet watch <RUN>
target/release/zenfleet-ctl catalog --manifest <manifest.json> --ledger <ledger.parquet>
bash scripts/jobsys/teardown_fleet.sh <RUN>
```
Declare tooling for the bigcodec pairs shape already exists:
`scripts/jobsys/build_scorefile_from_pairs.py <pairs.parquet> <tar_uri> <run_id>`
(reads `image_path/codec/dist_tar/dist_member`; its `METRICS` list already
includes `zensim-gpu`).

## B. Corpus + targets (the project already settled this — DATASET_HISTORY §5)

**Training mass:** `bigcodec_hqdedup_traindigits_2026-07-02.parquet` (2.32M, REAL
multi-codec zenjpeg/webp/png/jxl±lossless; `/mnt/v/output/canonical-picker-2026-07-01-zensimA/`,
dist via `encodes/`+`variant_tar_r2_url` — NOT the 404-trap `variant_r2_url`) +
safesyn (196k, foundational — needs bitstream decode, PNG cache deleted) +
cid22_train-201 (17.6k, **ssim2-anchored NOT MCOS** — legal, disjoint from the
49-ref holdout) + KADID/TID (13k, integrity-guard weight only). Target = ssim2
across groups (MLP absorbs bigcodec; a LINEAR fit it would poison — MLP-only mass).

**Held-out (T0, never train):** CID22-49 (gold MCOS) + AIC-3 CTC (600 JND) + AIC-4
(300 JND) + KonJND-1k val + JPEG-AI-SDR25 (95k, HQ-zone) + imazen-26 `nonphoto`
gate. Splits: `docs/DATA_SPLITS.md`.

**⚠ Do NOT add AIC-3 raw triplets (420k BTC-PTC-24)** without verifying ref-
disjointness — they likely PAIR WITH the AIC-3 CTC 10-ref holdout (Testolina 2025
same study family; `docs/AIC_DATASETS_2026-05-12.md`: "Pairs with AIC-3 CTC
images"). Memory's "untapped pairwise training data" note LACKS this caveat.

## C. Diffmap — core landed, completion method known

**Landed (commit `ce45a1ff`, on origin/main):** `compute_v2_diffmap_channel_scale`
(`feature_v2.rs`) — per-pixel diffmap `Σ_k w_k·M_k(x,y)` for one channel-scale,
**test-gated** by the block-pool identity `mean(map)==Σ features` (rel_err 1.2e-8;
`v2_diffmap_block_pool_matches_features`). Spatializes 12 simple-mean families
(bounded-SSIM d, art, det, mse, hf×3, pjnd-core, gms, blockiness, ringing,
banding). Reuses the real strip/halo/blur machinery, so it's exact by construction.

**Excluded families ARE computable generally** (not "no way"): the closed-loop
diffmap is `∂score/∂(distorted pixel) = Σ_k s_k·∂f_k/∂pixel`, and *every* feature
is differentiable, so every feature has a per-pixel gradient map. Block-pooling
that gradient = the M2 linearization term, which we measured at **M2≈1.0** — so a
gradient diffmap hits the coherence ceiling by construction, for all families.
- **masked×4, iw×4** (ref-based weights): per-pixel map = `w_i·v_i/Σw` — clean,
  one extra pass for `Σw`. Easiest.
- **soft-peak×3** (distortion-dependent saliency): + product-rule term.
- **dev2/dev4**: per-pixel deviation `∝(d_i−d̄)·∂d_i` (needs whole-image mean first).
- **transducer bank lo/hi**: identical to core, trivial.
- **edge_width**: genuinely cross-scale (couples two scales).
- **pjnd_fragility**: reference-only → `∂/∂distorted=0` → correctly contributes 0
  to a *steering* diffmap (not a gap, a correct zero).
Two routes: analytic per-family (bounded, do it for the survivors), or reverse-mode
autodiff through the pipeline once (fully general, covers future features too).

**Follow-on (deferred until §step-3):** wire `compute_v2_diffmap_channel_scale`
into the streaming `compute_with_diffmap` for combined bakes (the runtime map a
codec consumes) — and add the load-bearing excluded families' gradient maps. The
coherence NUMBER is already anchored (deployed M3 for v1 = 0.5415 ≈ its 53%
spatializable-mass proxy → ext-lumacoh's 100% ⟹ deployed M3≈1.0). See
`benchmarks/v2_trainability_ab_2026-07-19.md` "M3 cross-validation".

## D. Execution order + resume manifest

1. **[BLOCKER]** rebuild+push `zenfleet-worker:exec` (CPU) with V2Ab (§A). Verify
   `jobexec` help + one smoke cell returns a length-**720** feature row.
2. Build the training pairs spec from bigcodec (`build_scorefile_from_pairs.py`) +
   a **fresh run-id/ledger** (content-address gotcha: a stale ledger silently skips
   previously-scored cells, leaving 372-only rows — bigcodec's original zensim scores
   went through old chunk-mode, likely clean, but VERIFY don't assume).
3. Smoke ONE Hetzner box, derive the real per-box rate (do NOT size off the
   workstation projection ~112-169 ms/pair; measure). Then scale.
4. Assemble the 720 training parquets + join targets; add safesyn (decode pass) +
   cid22_train-201.
5. Train the optimal model (`zensim_mlp_train`, production recipe, multi-seed,
   held-out val groups REQUIRED). This is the ceiling + the feature-importance oracle.
6. **Feature validation**: LOO / `s_k` marginal contribution at the optimum, full
   panel, multi-seed → load-bearing set / redundant set / spatialize set.
7. Complete the diffmap for the load-bearing survivors (§C follow-on).

**Risks (ranked):** stale CPU image → silent 372; content-address skip → silent
no-op on re-scored cells; v2 is scalar (perf bottleneck — "run now vs wait for a
SIMD pass" is a conscious call); AIC-3 triplet contamination; `variant_r2_url` 404.

**Doc index for the science (compaction-safe):**
- A/B + decision + coherence + ablation + M3 cross-val: `benchmarks/v2_trainability_ab_2026-07-19.md`
- Timing + sqrt-already-SIMD: `benchmarks/v2_extraction_timing_2026-07-19.md`
- v2 feature spec (families, bounds, layout): `docs/FEATURE_V2_SPEC_2026-07-18.md`
- Append-only numbering directive: memory `feedback_feature_numbering_append_only`
- Program state: memory `project_feature_v2_program`
- This plan (fleet + optimal model + diffmap completion): THIS FILE
