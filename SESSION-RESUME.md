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

## In-flight at last update (2026-07-02 eve — check before starting new work)

- **zen-train-1 RESTORED** (159.69.55.206) running the v52 grid chain
  (`box_v52_chain6.log` in the probe dir): v51box_s17 determinism check +
  v52_s17/s7/s31 (dedup+hqfill corpus). Retire (`hz.sh retire`) after pull.
  Data path is RSYNC-PINNED-INPUTS (on-box rebuild diverged from manifest
  pins — bca4f6bd); bootstrap deps gate numpy/scipy; restore scrubs host keys.
- **avif 4-metric fill fleet**: 8 vast runs `fill-avif-b0..b7-2026-07-02`
  (image `exec-gpu-hqfillA-d5a142e0e166` — plain `exec-gpu` tag is GONE from
  ghcr; fleet.env default stale). b0 verified writing blobs. Merge sidecar
  when done → extend mm6 with avif.
- **Built today**: `bigcodec_hqdedup_{train,val}digits` (deduped+hqfill, v52
  inputs); `bigcodec_mm6_traindigits` (1.57M rows, 4 metrics joined — Bet-1
  input); `hqfill_7metric_sidecar`; SDR25 T0 anchor (A FAILS the SDR25≥ssim2
  gate: pooled 0.904 vs 0.958 — the target to beat).
- **Pending in zenmetrics** (blocked on `claude-pngfix` marker):
  build_scorefile_from_pairs.py + launcher overrides commit; re-push a stable
  exec-gpu tag or repoint fleet.env.
- Backfill: fill4 sidecar landed (patched variant is canonical; avif in
  flight); hqfill was already 7-metric-scored. Remaining prereqs: test-digit
  HQ/dial grid rebuild; within-ref pairwise eval in bake_verdict.

## Older state

Everything before 2026-06-30 (v47 ship history 2026-05-27, #33 Approach-B,
#35 resolution, V39-defect era) lives in this file's git history and in
`benchmarks/INDEX.md` + CLAUDE.md. Do not act on pre-2026-07 claims without
re-verifying — several were corrected this cycle (see the probe doc's
CORRECTION PASS sections).
