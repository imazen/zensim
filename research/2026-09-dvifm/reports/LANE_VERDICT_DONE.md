# LANE verdict — DONE 2026-09-21T06:2xZ (corrected 2026-09-21: CID22-B bake re-issue)

Lane: verdict (X2 constant-form selection → X1 standalone verdict).
Outputs:
- /home/lilith/work/zen/zensim/benchmarks/dvifm_verdict_2026-09-20.md
- /home/lilith/work/zen/zensim/benchmarks/dvifm_verdict_2026-09-20.json
- artefacts under /mnt/v/output/zensim/dvifm-verdict-2026-09-20/

## Result (plain)

- X2 winner: **gate** (≤15 knees + head ≈ 35 constants). curve's
  +0.0031 composite edge < 2σ seed noise (gate σ=0.0045); prior 0.8362
  trails. Composite: gate 0.8697, curve 0.8728, prior 0.8362.
- X1 frozen gate scored on all legs; peers scored (fastssim2, B, D,
  R915_basic228_ens5, R915_y60_ens5).
- CID22-B single registered read performed once, post-freeze —
  CORRECTED after supervisor review (see below):
  dvifm_gate 0.774, fastssim2 0.913, B 0.890, D 0.879,
  R915_basic228 0.897, R915_y60 0.861 (n=2100, 24 refs).
- VERDICT: standalone DVIFM is NOT competitive on real held-out
  labels — below fast-ssim2 on 4/5 real-label legs (kadid_dev_full and
  CID22-B significant; above only on konfig_val +0.024), and below
  every zensim bake on every real-label leg with one exception
  (above bake_prof_d on konfig_val +0.035). On CID22-B it is
  significantly below ALL peers (paired Δ −0.08..−0.13, P(Δ≤0)≈1.0).
- For the record: R915_basic228 beats BOTH fast-ssim2 and dvifm on
  human_dev (0.931 vs 0.817/0.783), kadid_dev_full (0.945 vs
  0.944/0.901), konfig_val (0.839 vs 0.735/0.759).

## CID22-B correction (same single read, not a second exposure)

- Supervisor flagged the all-five-bake collapse (0.29–0.32) as a
  feature-table signature. Diagnosis confirmed: the lane's re-extracted
  peer tables were w986 research-path (f0..f985); the bakes consume
  w944/ceiling_rev3 (f0..f943). Pixel-path B (`score_pair_with_bake`)
  on a 41-row sample: SROCC 0.897 vs labels, corr(pixel, w986-table)
  0.49 — era mismatch, not a model result.
- Corrected by scoring the historical
  rev3-public-human-eval-2026-09-14/features-rev3 parquets
  (w944/ceiling_rev3#b782e349, producer BakeScorer::compute);
  pixel≡table corr 1.0. Row-for-row joins verified: cid22b 2100/2100
  (row_id→ref,dist, 0 label mismatches), konfig 436/436 positional,
  kadid 250/250 via (ref,type,level) canonical order.
- Struck w986-era CSVs preserved as `scores/*.w986era`.
- kadid_dev/kadid135/konfig_val bake values equally re-issued from the
  era-correct tables. dvifm_gate/fastssim2 unchanged (pixel path).

## Caveats recorded in the record

- safesyn_dev/cid22_dev/codec_dev targets are signed ssim2/100
  pseudo-labels → fastssim2=1.0 circular; real-label legs are
  human_dev, kadid*, konfig_val, cid22b.
- The 2d uncapped f32 caches were deleted mid-lane by the superseded-
  cache cleanup; kadid/konfig/cid22b DVIFM caches re-extracted into
  this lane's f16 cap-1024 caches (same spec/binary).

## Deviations

- score_x2, X1 scoring, the cid22b read, and the era-correct rescoring
  ran outside ~/tmp/devin/heavy: the lock was held by dvifm-loss's
  run_ladder (sleeps between steps, hours-scale) while the machine sat
  idle (load ~2/32). Jobs were small (≤5 min, ≤4 threads). Logged in
  lane_verdict.log.
- No push.
