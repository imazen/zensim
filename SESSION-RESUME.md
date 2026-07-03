# SESSION-RESUME — read this first after every compact

**Last updated:** 2026-07-02 (splits registry + iteration protocol + PLAN_BEAT_A
locked; Profile-A byte-reproducible; Hetzner-first infra live)

## Reading order on resume

1. **`docs/PLAN_BEAT_A.md`** — the plan-of-record for beating Profile::A as a
   knob AND as an RD-loop metric (three bets, pre-registered gates, in-flight
   state). This is what the current work is FOR.
2. **`docs/DATA_SPLITS.md`** — the canonical train/val/test registry (LSD
   origin rule, KADIS mod-10, holdout tiers T0-T3, per-dataset conventions vs
   the literature, instrument overlap caveats). THIS FILE WINS over older docs.
3. **`docs/ITERATION_PROTOCOL.md`** — how to run experiments (measured cost
   model, screen→confirm pipeline, box lifecycle, mandatory parquet
   validation, anti-patterns).
4. `CLAUDE.md` — standing methodology (two-panel eval, SROCC-only ban, CID22
   validation-only, shipping policy).
5. `benchmarks/multicodec_profile_probe_2026-06-30.md` — the full evidence
   trail of the 2026-06-30 → 07-02 campaign (760+ lines: sign-artifact
   discovery, byte-identical A reproduction, #40 bisect, v48-v51 results,
   HQ-zone instrument, dHash audits).
6. `TaskList` — work on the lowest unblocked task.

## Ground truths that must not be re-derived (all verified)

- **Profile::A = v47-strict-QAT-native** (27,316 B, sha `d0ef7a30…`),
  **byte-reproducible**: trainer @ `e9442678` (or current main ≥ the #40 fix
  `9af7b789`) + `v47_strict_qat.toml`. Training is DETERMINISTIC. The
  `trainer_commit` + input-sha gates in the manifest loader enforce this.
- **The #40 rank_w init flip was the trainer-drift bug** (fixed, gated to
  h=1); every pre-fix v48 result is invalid evidence.
- **cid22_train human_score = ssim2_gpu/100 exactly** (NOT MCOS); CID22 human
  MOS has never trained. The 49-ref CID22 set is sacred T0.
- **Held-out val groups in selection work** (v51: 0 collapses, CID22 seed-sd
  0.009 vs 0.10 before). Mandatory for new recipes (DATA_SPLITS §5).
- **KADIS safety-grid oracle ceiling = 0.980** (cvvdp's own step-inversions);
  a bake at 0.99+ is over-constrained, target band 0.96-0.98.
- **ssim2 was tuned on CID22-201 + TID + KADID + KonFiG** (README, read
  2026-07-02) — never scoreboard vs ssim2 on KADID/TID; CID22-49 is fair.
- **The 85-100 zone's binding constraint is the ssim2 LABEL** (cvvdp-agreement
  0.48 there, measured by the HQ instrument) → Bet 1 in PLAN_BEAT_A.
- Corpora: canonical picker 2026-06-27 = **5,742,660 rows** (multi-metric
  backfill in flight); KADIS-700k-gpu = 700k cells × 7 metrics; AIC-3 raw
  triplets (420k) + SDR25 (95k) = untapped human pairwise data, local.
- imazen-26 origins vs CID22-49: dHash-CLEAN at d≤10 (min d=12).

## Infra (2026-07-02)

- **Hetzner-first: ALL minutes-scale work on ephemeral ccx63 boxes** —
  `scripts/hetzner/hz.sh` (provision/bootstrap/push-eval/push-manifests/run/
  status/pull/**retire**/restore). SSH identity `~/.ssh/zen-arm-dev`. Boxes
  are snapshot+deleted after results are pulled (user rule); base snapshot
  `zen-train-1-1782989687`. Scoped R2 temp creds only (never root keys).
- **`scripts/v_next/validate_parquet.py`** runs on every parquet before it
  trains anything (wired into box bootstrap + runcells preflight). Its first
  run caught real negative kadis targets.
- **`scripts/v_next/make_manifest.py`** — the ONLY way to generate manifests
  (string-surgery banned; it broke two waves).
- Workstation = orchestration + seconds-scale bake_verdict + commits only.
- One persistent quiet Monitor per session for terminal events; state lives in
  files (status.tsv, logs, benchmarks/), not conversation context.

## In-flight at last update (2026-07-02 ~22:30Z — check before starting new work)

**Program of record: v53 replicate-then-surpass ssim2** (PLAN_BEAT_A amendment
2026-07-02). avif fill DESCOPED by user (runs paused on R2, resumable).

- **Train box zen-train-1 (159.69.55.206, ccx63 $1.80/hr)** is running
  **wave-2 under systemd unit `wave2`**: v52_s47/s63 + v53_s17/s7/s31/s47/s63
  at PAR=7 (~2 waves × ~11 min). Check: `hz.sh status 159.69.55.206` or
  `ssh -i ~/.ssh/zen-arm-dev root@IP 'systemctl status wave2; cat /data/out/status.tsv'`.
  When done: `hz.sh pull`, plus `rsync root@IP:/data/derived/v5*.bin` (bake
  files land in /data/derived via the path symlink — manifest [bake].file
  points at the probe dir). THEN `hz.sh retire zen-train-1` (MANDATORY).
  NOTE the box is ~2× oversized for 4-cell grids — restore ccx53/43 next time
  unless running ≥6 cells (user cost question 2026-07-02).
- **v52 wave-1 DONE + pulled** (hetzner-out/ + v52_s*.bin in the probe dir):
  CID22 SROCC s17 0.8512 / s7 0.8275 / s31 0.7150 — all below A's 0.8657,
  seed spread is BACK on the deduped corpus (the 22% knob-no-op dups in v51's
  corpus likely acted as regularization — investigate before concluding).
  Dial healthy: mono 0.9786 (target band), 0 dead zones. Full verdicts:
  /tmp/v52_s*.verdict.md (regenerate: bake_verdict --bake <bin> — seconds).
- **v53 = v52 + konfig group** (KonFiG-IQA ingested: 1,090-row parquet,
  JND-design-grid targets; loader `--corpus konfig` in extract_features_372col).
  Gates: CID22-49 ≥ 0.8854 (ssim2's own number), SDR25 pooled ≥ 0.958 (A is
  at 0.904; scoreboard via sdr25_eval_pairs.tsv + zenmetrics batch), KonJND ≥
  0.4185, dial/safety as v52.
- **SDR25 T0 anchor BUILT** (sdr25_jnd_reconstructed_2026-07-02.parquet +
  scripts/v_next/reconstruct_sdr25_jnd.py). KonFiG recovered + Tower-mirrored.
- **Multimetric Bet-1 input BUILT**: bigcodec_mm6_traindigits (1.57M rows,
  cvvdp/butter/dssim/iwssim joined; avif absent by descope). hqfill 7-metric
  sidecar aggregated (62,258 cells).
- **Infra rules learned today (all committed)**: hz.sh run uses the REPO
  runcells (scp copies go stale); restored boxes need known_hosts scrub +
  numpy/scipy + path-map symlinks (all in bootstrap now); pinned inputs
  RSYNC from workstation (on-box rebuilds diverge); per-input manifest
  contracts (target_range/validate_kind/allow_dup_rate) — safesyn is
  [-8,1.001], kadis [-0.5,1.001]; **use systemd-run on boxes** (nohup-over-ssh
  dies with the session); vastai destroy needs -y (returns rc=0 on abort).
- Concurrent-session etiquette: check zenmetrics .workongoing before
  committing there (claude-pngfix was active most of today).

## STRATEGY ablation campaign (2026-07-02 late — LIVE)

- **Strategy suite implemented + reference-tested** (commit `78ec8e61`):
  EMA / hard-pair / stratified-bands / σ-MSE(ZENSIM_SIGMA_MSE=1) / GroupDRO /
  ListMLE / triplet-NLL. Tests in `mlp_train/strategy.rs` (failures = "IMPL
  BUG (not strategy)"). All-active smoke deterministic. KonFiG raw triplets
  decoded per-BoostType (flicker=MORE-DISTORTED, non-flicker=CLOSER; 85.6%
  design agreement; 541,895 responses in
  `konfig_triplet_responses_2026-07-02.tsv`; EXP_II interior-pivot deferred).
- **18-cell ablation (9 variants × 2 seeds, v53 base)** running on BOTH:
  (a) zen-train-1 ccx63 (systemd unit `ablation`, PAR=8);
  (b) 9× cx53 fleet `sfb-abv1-*` (~€0.05/hr ea) — `strategy_fleet.sh
  status abv1 /tmp/ab_cells.txt`, results at
  `s3://zentrain/strategy-results/abv1/`, **reap with `strategy_fleet.sh
  reap abv1`** when scored. Same manifests both machines ⇒ per-variant
  cross-machine determinism check for free.
- Scoring on completion: bake_verdict per bin (local, seconds) → variant table
  (CID22/KonJND/dials vs ab_base) → pick winners for v54. v53 5-seed spread
  was CID22 0.638-0.837 (!) — EMA-vs-base is the money read.
- Wave-2 results already scored: v53≈v52≈below A on CID22 at high seed
  variance; KonFiG-as-group inconclusive at this spread.
- NOT yet in this wave: σ-MSE ablation (needs an mm6 group in the recipe).

## Campaign CLOSED OUT (2026-07-03 ~01:10Z — check FIRST on resume)

- **`w3_hponly51_s7` = CID22 0.8767 — FIRST cell above Profile-A (0.8657)**,
  dial G1 1.00 / mono 0.975. Recipe: v51 base (pre-dedup corpus) +
  `hard_pair_frac=0.5, hard_pair_max_delta=0.05`, trainer `78ec8e61`.
  KonJND gate NOT yet passed on that seed (0.318 < A's 0.4185).
- **Strategy campaign verdicts** (benchmarks/strategy_ablation_2026-07-02.md):
  hardpair = single best lever (+0.13, dial-repairing, 5 seeds no collapse);
  t1dro = tightest + best KonJND; hpdro rejected (non-additive collapse);
  listmle craters KonJND as configured; ema/strat washes; triplet mild
  (+ flat-dial trap at one seed — G1=0.00 masked by mono 0.996; always read
  BOTH dial numbers). Corpus effect: v51 dups act as regularization (+0.02
  over dedup corpus).
- **GRADUATION DONE — t1dro51 is the winner and PASSES the SDR25 gate**:
  SDR25 pooled 0.955-0.969 across 5 seeds (ssim2 0.958, A 0.904; within-image
  1.0000 ×5); CID22 0.858±0.010 (two seeds > A), zero collapses; dials pass.
  Open gates: CID22-49 ≥0.8854 (nothing passes it, incl. A) and KonJND ≥
  0.4185 (mixed 0.31-0.44). Full record:
  benchmarks/strategy_ablation_2026-07-02.md. Recipe of record: v51 base +
  ema=0.9 hard_pair=0.5@0.05 strat=10 dro_eta=0.5, trainer 78ec8e61
  (manifests w3_t1dro51_s*).
- **Box RETIRED** (server deleted; NEW restore base snapshot
  `zen-train-1-1783047322` — newest, post-wave-4; supersedes 1783040838/1782989687). cx fleet reaped. Zero
  cloud burn. Bakes: w3_t1dro51_s{17,7,31,47,63}.bin in the probe dir.
- **Wave-4 done** (mining band δ = the CID22↔KonJND dial: 0.03 rank-leaning /
  0.08 threshold-leaning; frac stays 0.5). **kagg falsification RETRACTED —
  scale mismatch** (agg loss compares pinned RAW output (~0.23-wide band)
  against raw PJND 22-70; valid retest needs dial-output comparison — design
  change queued). triplet-on-stack: replicate-before-trusting.
- Next-campaign levers (KonJND + CID22-49 gates): kagg-on-dial-output
  redesign; konjnd-scoped hard-pair mining (per-group frac plumbing); EXP_II
  interior-pivot triplet model; within-ladder listwise; dup-aware sampling.
- Infra: autoretire is billing-aware (retires at min>=45 of the paid hour)
  with pidfile discipline; NEVER pattern-kill watchers/monitors.
- **Profile B (HDR) state 2026-07-03:** v1 = w5_hdrmix_s17 (hdr_val 0.969
  stable across 3 seeds; SDR parity at s17/s7, s31 pays −0.07 CID22 —
  held-out-val selection applies). UPIQ shows the u8-shell feature ceiling
  (~0.65 vs cvvdp 0.758/iwssim-HDR 0.808). **v3 = PU-linear features**
  (Zensim::compute_pu_linear_extended_features + zenmetrics
  --hdr-features-pu-linear, both landed+tested): re-extraction fleets
  v3june/v3hq + UPIQ ran overnight → on completion: merge_v3_shards.py →
  build_hdr_train_parquets.py (point at v3 datagen dirs) → retrain w5 →
  UPIQ panel + consistency harness (PU-linear variant) + HDR dial grid.
  v2 (anchored shell) OBSOLETE. REAP the hdrsf-* fleets when done (no
  autoretire on those). Docs: PLAN_HDR.md + benchmarks/strategy_ablation
  §Profile B; memory: project-profile-b-hdr.
- cx fleet: REAPED (0 boxes). Wave-1 cross-machine: 16/18 byte-identical;
  2 divergent = one AVX-512-tier box (ISA reduction order, documented).
- Next levers if KonJND gate blocks: hardpair delta/frac sweep; t1dro51
  (better KonJND family); hard-pair mining ON konjnd-dense group
  specifically; triplet weight sweep (its KonJND was neutral-positive).
## Older state

Everything before 2026-06-30 (v47 ship history 2026-05-27, #33 Approach-B,
#35 resolution, V39-defect era) lives in this file's git history and in
`benchmarks/INDEX.md` + CLAUDE.md. Do not act on pre-2026-07 claims without
re-verifying — several were corrected this cycle (see the probe doc's
CORRECTION PASS sections).
