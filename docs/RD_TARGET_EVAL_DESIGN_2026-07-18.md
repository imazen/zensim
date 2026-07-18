# Eval design: how helpful + efficient is a diffmap model for hitting a target zensim at optimal RD? (2026-07-18)

User ask (2026-07-18): *"make a worktree of jxl that uses diffmaps — and a worktree of zenjpeg —
and see if we can figure out a way to eval how helpful — and efficient — a diffmap model is for
quickly getting a codec to hit optimal RD for a target zensim value."*

## What already exists (recon 2026-07-18 — do not rebuild)

- **jxl-encoder** (worktree `~/work/zen/jxl-encoder--zensim-diffmap-rd`): three working per-block
  diffmap-driven QF-refinement loops (`butteraugli_refine_quant_field`, `zensim_refine_quant_field`
  with AC-strategy split, the unified `PerceptualBackend` path), opt-in features
  `zensim-loop`/`ssim2-loop`; iterations are IN-PROCESS (no bitstream round-trip; entropy-codes
  only the winner) — structurally cheap. The #38 harness `examples/zensim_diffmap_rd.rs`
  (2026-05-27) already encodes a corpus per driver + emits a manifest for an INDEPENDENT judge
  panel (`zen-metrics`: butteraugli+dssim+ssim2) — the circularity guard. Historical result:
  zensim-driven loops beat butteraugli/cvvdp loops by 1–5% bytes-at-equal-quality on independent
  judges (`benchmarks/diffmap_comparison_results_2026-05-27.md`).
  Worktree additions (2026-07-18): `JXL_ZENSIM_RD_PROFILE=a|b|latest` (backend + legacy loop),
  `JXL_ZENSIM_DIFFMAP_SIGNALS=all` (edge/mse/hf signals in the steering map). Defaults unchanged.
- **zenjpeg** (worktree `~/work/zen/zenjpeg--zensim-diffmap-rd`): `Quality::Zq/ZqExplicit` closed
  loop — starting-q from a calibrated table, ≤2 correction passes, each pass a FULL re-encode +
  real-JPEG decode + zensim measure; per-block diffmap-driven AQ via `ScalingController`
  (block-diffmap p25/p75 → multiplicative AQ scale, clamp [0.40,1.80]) + first-class
  `BlockArtifactBound` (per-8×8 diffmap ceiling); `Quality::ZqPicker` = ONE-SHOT (108-feature
  distilled picker, no measurement) — the Eval-C incumbent. `EncodeMetrics` already reports
  `passes_used/achieved_score/bytes/targets_met`. Feature `target-zq`.
  Worktree addition: `ZENJPEG_ZQ_PROFILE=a|b|latest` override.
- **zensim**: coherence instrumentation + the driver menu (see
  `benchmarks/mlp_diffmap_coherence_2026-07-18.md`): deployable steer-maps —
  ADD156+ModelSensitivity(abs) 0.849 · winner+ModelSensitivity(signed) 0.759 · winner+shipped
  0.746 · B+shipped **0.243** (today's pairing, worst) · SSE anti-correlated on texture.
- Gaps confirmed by recon: no efficiency instrumentation anywhere; Eval B (closed-loop
  convergence) + Eval C (one-shot residual) specified in `FINAL_DIAL_METRIC_DESIGN` §4/§6 but
  never built; the 2026-07-16 RD probe was judge-circular-aware but n=50, mozjpeg-only, q15/q30
  only (fails the q5–q60 density rule for anything source-informing).

## The two questions, made measurable

**HELPFUL (RD value of the diffmap):** at equal *achieved* target score, how many bytes does
per-block diffmap steering save vs the same codec's global-knob-only search?

- Per (image, codec): build the **no-diffmap baseline frontier** — dense global-quality sweep
  (zenjpeg: jpegli-q ladder; jxl: distance ladder), score each rung with the FIXED comparison
  scalar (zensim B for cross-driver comparability) → monotone (bytes ↔ score) curve.
- Per (image, codec, target T, driver): run the driver's targeting loop → final bytes + achieved
  score. **helpfulness = 1 − bytes_driver / bytes_baseline(achieved_score)** (interpolate the
  baseline frontier at the driver's achieved score — never compare at nominal T).
- **Anti-gaming judge panel (mandatory):** score every final encode with independent judges
  (ssim2 + butteraugli via `zenmetrics`, optionally cvvdp) — a driver that saves bytes by
  gaming its own scalar shows up as a judge regression at equal claimed score. Home-turf cells
  are labeled (the #38 convention).

**EFFICIENT (cost to converge):** what does hitting |achieved − T| ≤ ε cost?

- `n_encoder_passes` (zenjpeg: `passes_used`; jxl: refinement iters + seed encodes),
- `n_metric_evals` (scalar scores + diffmap computes; MLP gradient s_k recomputes counted),
- wall time, split encode-vs-metric (instrument both loops; the repos have NO committed
  steady-state numbers — the 250 ms/iter@1MP zensim-CPU figure is a risk threshold, not a
  measurement),
- **one-shot residual (Eval C):** ZqPicker / jxl `zensim_targets` table: |achieved − T| with
  ZERO measurement passes — the floor every iterative driver must beat to justify its cost.

## Matrix

| axis | values |
|---|---|
| codec | zenjpeg (Zq loop), jxl-encoder (QF loop) |
| driver | none (global-only) · SSE-analog · butteraugli-map (jxl native) · zensim-B+shipped map (incumbent) · winner-MLP+signed model map · ADD156+abs model map |
| target T | {25, 40, 55, 70, 80, 90} — q5–q60-band-heavy per the sweep-density rule |
| content | gb82 (photo) + gb82-sc (screen/nonphoto) + KonJND-src sample; ≥12 images/class for the probe, full-corpus only after the mechanism is confirmed |
| ε | |achieved − T| ≤ 2.0 (report the residual distribution too, not just pass/fail) |

Instrumentation lands as TSV rows (one per cell): image, class, codec, driver, T, bytes,
achieved_self, achieved_B, judges…, passes, metric_evals, ms_encode, ms_metric. Stats via
`zensim_validate::panel` / `scripts/lib/zen_stats.py` (no hand-rolled stats). Output:
`/mnt/v/output/zensim/rd-target-eval-2026-07/` + committed summary in `benchmarks/`.

## Phasing

1. **Smoke (now):** 3 images × 2 targets × {butteraugli, zensim-B} on jxl via the #38 example +
   distance search; zenjpeg `ZqExplicit` with/without `block_artifact`. Proves plumbing +
   captures first per-iteration timings.
2. **Driver wiring:** jxl backend consumes ModelSensitivity maps (winner signed / ADD156 abs —
   needs the bake-file mount in the backend, mirroring the coherence example); zenjpeg
   ScalingController fed by the same maps (its diffmap source is the zensim call inside the Zq
   loop — profile/map choice via the env overrides).
3. **Probe matrix:** the full table above at probe-n; verdict on helpful+efficient per driver.
4. **Only if the probe shows real byte savings:** scale corpus + densify targets per the
   source-informing sweep discipline before any constant lands in source.

## Verdict criteria (pre-registered)

- A diffmap driver is **helpful** if median bytes-saved > 2% at equal achieved-B-score with NO
  independent-judge regression beyond noise (paired bootstrap) — the 2026-05-27 result suggests
  1–5% is on the table.
- A driver is **efficient** if it converges within ≤2 correction passes (zenjpeg) / ≤4 QF iters
  (jxl) AND its metric wall-time ≤ encode wall-time per pass (otherwise the metric dominates the
  encoder and "quickly" fails).
- The MLP-vs-additive steer question (coherence 0.759 vs 0.849, rank 0.894 vs 0.863) is settled
  HERE, in bytes and passes — not by another coherence proxy.
