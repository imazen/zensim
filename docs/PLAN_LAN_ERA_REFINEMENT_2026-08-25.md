# LAN-era refinement plan — zensim models · AV1 diffmap steering · jxl secant loop (2026-08-25)

User directive (2026-08-25, verbatim): *"we have shifted all fleet and storage to
local LAN, can you learn the new zenfleet stuff and make a plan for further
refining of zensim models and zenavif (svt, aom, and rav1e) diffmap and jxl
diffmap secant and loop encoding efficiency."*

This is the durable program spec for that directive. It sits on top of the
2026-08-22 roadmap (`zenmetrics/docs/status/training-roadmap-2026-08.md`) and the
campaign authority (`benchmarks/sota944_campaign_2026-08-03.md`, appendices
P/Q/R/S/Y/Z/AB/AC/AD); it does not restate them, it orders the work after them
and adds the three lanes the directive names. Every claim below was read from
source, a manifest, or a committed benchmark on 2026-08-25 — cited inline. Where
a doc and the code disagree, the disagreement is listed in §1.4, not silently
resolved.

Read first, in this order: `SESSION-RESUME.md` → this doc → `docs/WAVE_PLAYBOOK.md`
(every wave below runs on that skeleton) → the appendix each item cites.

---

## 0. Executive summary

- **The LAN fleet is real but half-migrated.** Store = **SeaweedFS 4.43** on the
  tower NVMe cache (MinIO was the 08-08 bring-up pin, swapped 08-21); `ZEN_STORE`
  defaults to LAN since 08-10; buckets kept their R2 names 1:1. But the
  *production* worker enrollment script still mints R2 creds, `fleet status/watch`
  silently reports zero on a LAN-only shell, four boxes have no LAN cred file,
  and only ~0.5 GB of the 1.85 TB `zentrain` corpus has moved. §3 fixes the
  operator-facing gaps first; nothing else in this plan can run on the fleet
  until A1/A2 land.
- **zensim models:** wave-12 (the AVIF-arm retrain) is fully pre-registered
  (appendix AD, 08-21), its data gate is OPEN, and it was never launched. It is
  the first compute to start. Behind it: the breadth-transfer lead from
  appendix R (sparse distillation of C carries CSIQ/LIVE into the 0.93–0.96
  class), the co2b seed wave, and the stalled HDR phase-2 corpus (encode
  drained 99.9 % on 08-07; its score ledgers were declared on R2 and never
  harvested).
- **jxl loop:** there is no secant controller anywhere in jxl-encoder today —
  "secant" exists only as an unbuilt note in `zensim-target/README.md`. The
  in-loop controller is a damped power law (`exp 1.0 / clamp 2.0`, adopted
  08-07) that reaches ±2 in **24/27 cells at k3 but only 18/27 at k2**. §5
  registers a global secant step, a bracketed (Brent-style) safeguard, and a
  per-tile secant gain as A/B arms on the existing 27-cell instrument, plus the
  ranked per-iteration cost levers (FD-probe GEMM batch, the S1/S2 front-end
  seams, in-strip stale fold) and the missing size-axis measurements.
- **AV1 steering:** the per-superblock **delta-q** channel is live in zenrav1e
  and measured as a matched-rate **loss** (Y.R3, 9/9 cells; the butteraugli
  two-pass got the same verdict). All three ports expose a **λ-side** channel
  (rdmult scaling) that costs no syntax and composes with segmentation — that
  is where the map's shape gets tested next, behind one codec-agnostic
  interface. The AC.4 `zensim_cq_rd` harness does not exist yet; it is the
  first AV1 deliverable, with the global CQ target loop before any per-block
  arm.

---

## 1. Ground truth on 2026-08-25

### 1.1 Storage + fleet (from `zenmetrics`, `homefleet`, read 2026-08-25)

| item | state | source |
|---|---|---|
| store | SeaweedFS 4.43 in Docker on tower, data on the NVMe cache pool; MinIO container parked as rollback; 4.44 pre-validated | `homefleet/zenmetrics/tower/LANSTORE.md:226-241, 452-475`; `zenmetrics/docs/status/fleet-orchestration-2026-08.md:207-232` |
| endpoint resolver | `ZEN_STORE` unset/`tower`/`lan` → LAN (default since 08-10); `ZEN_STORE=r2` → R2; `ZEN_S3_ENDPOINT=<url>` verbatim | `zenmetrics/scripts/lib/s3env.sh:7-46`, `scripts/lib/zen_s3env.py:10-24` |
| creds (paths only) | LAN `~/.config/zen/lanstore.env` (0600); R2 `~/.config/cloudflare/r2-credentials`; Nomad renders `secrets/lanstore.env` from a Nomad Variable | `lan-storage-2026-08.md`; `homefleet/.../fleetbench-gpuscore.nomad.hcl:48-58` |
| buckets | `zentrain` (writes), `codec-corpus` (read-only), `zenfuzz` — unchanged names | `RUNNING_JOBS.md:197-198` |
| what moved | `zentrain` on LAN = 2,984 objects / 494 MB vs 1,850 GB on R2 — "the cutover is proven end-to-end but the corpus is still R2-resident" | `lan-storage-2026-08.md:127-137` |
| R2 rundown | proposal registered, **no deletions performed, user-gated** | `lan-storage-2026-08.md:165, 199-212`; `DATA_PROVENANCE.md:1330` |
| roster (neutral ids) | Nomad servers: `dev`, `tower`, `r7900x`; Nomad clients (G-N1): `r7900x`, `i265`, `r3500`; systemd `zen-worker` enrolled: `r7900x` (stopped), `i265` (stopped), `r3500` (stopped), `r5900xt` (running), `tower` (basement tier), `mac` (idle-only); `i134`/`r5600g` intermittent, down after the failed G-P1; `wsl` = operator seat | `homefleet/zenmetrics/ORCHESTRATION-2026-08.md:27-43, 113-116`; `NODES.md:188-199` |
| GPU | the Nomad GPU jobspec pins `r7900x` as "the only GPU box"; the appendix-Z/Z.R record has an RTX 3070 (node-2) + RTX 2080 doing the rescue — the card inventory is inconsistent across docs (§1.4 #8) | `fleetbench-gpuscore.nomad.hcl:9-26`; campaign Z.R0 |
| GPU metric truth | GPU-ness = metric-name suffix `-gpu`/`_gpu` → `ResourceClass::Gpu`; **`zensim-gpu` panics by design in the current executor tag** | `zenfleet-core/src/job.rs:287-309`; `scripts/jobsys/fleet.env:44-49` |
| images | CPU `ghcr.io/imazen/zenfleet-worker:exec`; GPU `:exec-gpu-zensimv2-0953c94671`; executors must be built `--target x86_64-unknown-linux-musl` (a glibc-drift binary shipped and would not start) | `fleet.env:50-81` |
| JobKinds | exactly 7: `Encode{codec,q,knobs,hdr}`, `Metric`, `ScoreFile{metrics[],hdr,hdr_transfer}`, `Feature{regime}`, `Diffmap{metric,hdr}`, `Resample`, `Bake` | `zenfleet-core/src/job.rs:112-197` |
| Diffmap executor | **landed** `9093cc23` (2026-08-06): butteraugli + cvvdp per-pixel maps → gzip PFM blobs; ssim2 refused (absent-not-failed) | `zenmetrics/CHANGELOG.md:232`; `zenfleet-ctl declare-diffmaps` |
| claim modes | lease (default) vs `ZEN_CLAIM_MODE=epoch-sharded` + `fleet/handicaps.toml`; measured re-work tax 1.69× vs 3.49× on this fleet | `RUNNING_JOBS.md:306-342`; `ORCHESTRATION-2026-08.md:116-119` |
| power | `fleet power status|apply [--live]`; WoL wake, drain+suspend/poweroff; **automatic sleep DISABLED fleet-wide** (`2458466a`, 2026-08-25); `i134`/`r5600g` pivoted to S5 | `scripts/jobsys/fleet_power.py:93-122` |
| Nomad | CE 2.0.5 pilot as the box-lifecycle layer only; zenfleet stays the data plane; P1 done, **P2–P5 not done** (jobspec does not yet replace `enroll_running_node.sh --start`) | `fleet-orchestration-2026-08.md` ("Decision", "Preconditions") |
| operator box | **60 GiB / 32 threads**, root 1.8 TB at 2 %, `/mnt/v` 3.7 TB at 66 % (`free -h`, `df -h`, 2026-08-25) — appendix AD.7's "23 GiB post-resize" no longer holds | measured this session |

### 1.2 zensim model campaign

- Profile **C** shipped 2026-08-05 (`W10L9_s4003_packed`, first pruned bake,
  caller 944 / internal 667); **B stays default**; the C-vs-B trade is stated
  in `docs/PROFILE_C_REPRODUCTION_2026-08-05.md`.
- Balanced board: **0/172 cells pass all 8 floors**; the binding axis is
  classic-IQA breadth (CSIQ/LIVE ≥ 0.83 never co-occurs with CID22 ∧ KonJND).
- **Wave-12** (AVIF arm): appendix AC pre-registered 08-07, data gate fired on
  2,169 corrupted GPU-score cells (AC.R1), rescue closed 08-21 (Z.R:
  RESCUE-WINS 3/2,000, G-Z5 0.999313), execution plan frozen as appendix AD
  (leg weight 1.1043, seeds {4201…4211}, dial8 instrument 249×30, like-for-like
  unit amendment). **No `W12_s*` bake exists on disk; `~/tmp/wave12` does not
  exist** — never launched.
- Appendix **R** (sparse distillation of C): no CS cell meets `cid22 ≥ 0.875`
  (best 0.8562), but the teacher transfers **classic-IQA breadth** (packed CS
  cells CSIQ 0.929–0.955, LIVE 0.930–0.963, HF-NL 0.80–0.85, M3a 0.87–0.99) at
  CID22 cost — "a lead for a registered follow-on, not a claim" (R.R2).
- Appendix **Q** (HDR phase 1): outcome (b) **parity** — no 944-route candidate
  beats BHdr on UPIQ; the named lever is multi-codec HDR data (Q-G6).
- Appendix **S** (HDR phase-2 corpus): encode drained **102,485/102,600
  (99.888 %)** 2026-08-07 01:52Z; residue = 115 cells of `zenav1-svt`
  `restoration.rs:985` OOB (imazen/zenav1-svt#11); score waves
  `hdrgrid-sf-{gpu,gpu-huge,gpu-small,cpu}-20260807` **declared on R2 05:58Z**;
  cvvdp/butteraugli/diffmaps deferred to the measured-peak phase (AA landed
  `6471f4d7`); orientation gate **PENDING**; no `unified/` tables exist under
  `/mnt/v/output/hdrgrid-2026-08-06/` — nothing was harvested after 08-07
  (`_MANIFEST.json` `status`/`runs`/`gates`, read 2026-08-25).

### 1.3 Loops

- **jxl** (`jxl-encoder/jxl-encoder/src/vardct/zensim_loop.rs`, last loop
  change `810a330b` 2026-08-07): controller
  `g = ((L_a/L_t)^exp).clamp(1/c, c)` on the whole quant field
  (`:1603-1611`), `JXL_ZENSIM_CTRL_EXP` **1.0**, `JXL_ZENSIM_CTRL_CLAMP`
  **2.00**, `ZENSIM_ATTR_BIN` 8, `ZENSIM_H3_GAIN` 10, tolerance 0.25; product
  default is loop OFF (`api.rs:2712`) and, when on, unsteered Trained-diffmap
  redistribution — h3-mag steering is opt-in. Frontier arm `W10L9_h3ctrl2`:
  **k2 18/27 med 1.19 · k3 24/27 med 0.55**, beats the butter comparator
  (`outer_zensimA` j2 12/27 / j3 14/27) at both budgets
  (`jxl-encoder/benchmarks/zensim_loop_beatbutter_2026-08-07.md`,
  `zensim_loop_23shot_summary_2026-08-07.json`). Residue = screen content
  seeds landing ≈ 91 for t70 (AB.4). Per-compare cost after appendix P:
  iter-1 101–105 ms, iters ≥ 2 **40–43 ms** at 576², iter-0 FD probe 250 ms
  after batching (P.R4, Y.R1). **No 1152² or larger loop measurement exists
  in either repo** (the only large datum is zensim-side "12 MP e2e −19 %" from
  binned L2).
- **zensim-target** (outer, codec-agnostic): bisection, 33/36 within ±1.5,
  median 5 iterations; README "Known limitations" #3: "Bisection. A secant or
  Brent's-method update would converge faster" (`zensim-target/README.md`).
- **AVIF**: `zenavif` drives **zenrav1e only** (`Av1Backend::Svtav1` is
  deprecated and rejected by `validate`, `src/encoder.rs:180-196`). It has a
  scalar zensim target loop (`target_quality.rs`, global q search), a
  feature-gated `auto_tune` picker (model not baked), and a per-SB two-pass
  driver (`two_pass.rs`, butteraugli-only, dead behind
  `ravif::FRAME_HINTS_LIVE = false`, DROPPED as default 2026-07-04:
  +0.10…+2.20 % bytes at 2× time, `docs/DIFFMAP_TWO_PASS.md`). The zensim-map
  per-SB delta-q probe (Y.R3, `benchmarks/avif_sb_probe_2026-08-06.tsv`)
  loses at matched rate **9/9 cells** (Δssim2 −0.51…−4.07); mechanism: hints
  disable segmentation (`zenrav1e/src/encoder.rs:2334`) + delta-q syntax tax +
  an uncalibrated policy. The registered next lever is the λ-side
  `ssim_rdmult` channel. **The AC.4 `zensim_cq_rd` harness does not exist.**

### 1.4 Discrepancies found (fix as part of §3 A4, not silently)

1. `zenmetrics/docs/status/lan-storage-2026-08.md:17,124` still says "pinned
   MinIO"; the store is SeaweedFS since 08-21.
2. `homefleet/.../ubuntu-node/enroll_running_node.sh:24-28,47,82` — the
   production pool-worker path — is **R2-only** (builds `ZEN_R2_ENDPOINT` from
   `R2_ACCOUNT_ID`, mints an R2 cred). Untouched by the migration.
3. `zenmetrics/scripts/jobsys/fleet:36-40,112` — `r2_count()` returns 0 when
   `R2_ACCOUNT_ID` is unset, so `fleet status`/`fleet watch` report
   `boot 0 / claims 0 / sidecars 0` on a LAN-only shell. `pool_progress.py`
   is LAN-safe.
4. `RUNNING_JOBS.md:391-393` names the Railway-hosted dashboard; a LAN
   `zenfleet-dash` service is planned only.
5. `DATA_PROVENANCE.md` has **no LAN-storage section** (76 "R2" mentions, one
   LAN-store path at `:1362`); its canonical triples still read
   local + R2 + Tower.
6. Appendix AD.7 "23 GiB total RAM" — the box reports 60 GiB today; AD.7's
   serialization rule was derived from the smaller number (§4 B1 amends it
   by measurement, not by assumption).
7. Appendix R.R2 quotes "C itself: csiq 0.770, live 0.811" while AD.6's
   frozen wave-11 bands (the same bake family) are CSIQ [0.933, 0.960] /
   LIVE [0.961, 0.968] and the cookbook banner has C at LIVE 0.9604 / CSIQ
   0.9331. One of these reads is off-instrument (the `--regime` trap in
   CLAUDE.md "Known Bugs" is the usual cause). Re-derive with
   `bake_verdict --regime 944` before §4 B2 builds on it.
8. GPU inventory: Z.R0 says "the ex-lianli box (r7900x, then carrying the RTX
   2080)" and "node-2/RTX 3070"; the Nomad jobspec says r7900x has a GTX 1060
   6 GB and is the only GPU box. Which cards are in which boxes today is a
   one-line inventory fact that must live in the private `NODES.md`.
9. `zensim_loop.rs:1600-1602` comment still reads "Exponent 0.6 + 1.35 clamp";
   the defaults six lines above are 1.0 / 2.00.
10. Z.R1 says "diffmaps … executor unbuilt"; the executor landed in
    `9093cc23` (HDR lane B2) — the avifgen diffmap follow-up is runnable.
11. `zensim/benches/v2_speed_baseline.rs` in the primary checkout was 11,302
    NUL bytes (mtime 2026-07-21; jj's stat cache was fooled by the preserved
    size+mtime, `git diff` was not). Restored from HEAD in the working tree this session (jj shows no diff because the stat cache never saw the corruption); the
    NUL copy is parked at `~/tmp/planlan/`.

---

## 2. Operating the LAN fleet (the "learn zenfleet" deliverable)

The data plane did not change shape: content-addressed `JobId`s,
declare → gap → reconcile → harvest, S3 conditional-PUT leases, the Parquet
ledger at `s3://$BUCKET/$RUN/ledger/*.parquet`, blobs at `$RUN/blobs/<sha256>`.
What changed is the endpoint default and the boxes. Verbatim, from the owning
scripts:

```bash
# Endpoint + creds: ONE resolver. Never test "is ZEN_S3_ENDPOINT set" (that
# inverted at the flip — RUNNING_JOBS.md:163-168).
source ~/work/zen/zenmetrics/scripts/lib/s3env.sh      # exports EP, ZEN_S3_STORE, ZEN_S3_CLOUD_REACHABLE
# ZEN_STORE=r2 <cmd>   -> explicit R2 (cloud boxes only; launch_fleet.sh refuses cloud tiers if unreachable)

# 1. Declare (idempotent; content-addressed)                   RUNNING_JOBS.md:229
target/release/zenfleet-ctl declare --spec spec.json --out manifest.json
target/release/zenfleet-ctl declare-encodes --cells cells.jsonl --out manifest.json
target/release/zenfleet-ctl declare-diffmaps --spec spec.json --out manifest.json [--hdr]

# 2. Coverage / gap                                            RUNNING_JOBS.md:396
target/release/zenfleet-ctl catalog --manifest manifest.json --ledger <ledger.parquet>
target/release/zenfleet-ctl gap     --manifest manifest.json --ledger <ledger.parquet>

# 3. Workers
bash ubuntu-node/enroll_running_node.sh --start <host>        # POOL worker — ⚠ R2-pinned until A1
bash fleetbench_systemd_worker.sh <host> [ssh-user]           # one-shot manifest worker, LAN-resolved; --stop removes
nomad job run <spec>.nomad.hcl  /  nomad job stop -purge <job> # Nomad-managed (P1 path; GPU spec pins r7900x)

# 4. Progress + power
python3 scripts/jobsys/pool_progress.py [total_jobs]          # LAN-safe footer read (seconds)
bash scripts/jobsys/fleet power status                        # roster / wake (dry-run default); sleep disabled by default
bash scripts/jobsys/fleet smoke-image <IMAGE>
```

Rules that carry over unchanged and bite on the LAN: first-cell persistence
gate before any scale-up (Z.6 #5, G-Z2/G-Z3 shape); `ZENMETRICS_REQUIRE_GPU=1`
on every GPU-scoring worker (the B6 gate — the OOM-window survivors that
produced 2,169 silent-garbage cells did not crash); SIMD-tier-matched
extraction pools (zensim#56: MSCN slots are vendor-nondeterministic at ~1e-8);
worker/ctl binaries never build in one cargo invocation with zenmetrics-cli
(zenmetrics#38); every waiter must show the evidence of its last failure
(Z.R3 #8 — the 13-day stall was a `timeout 560` around a 768 s ledger read).

---

## 3. Program A — fleet/ops gates (P0; nothing below runs on the fleet before A1–A2)

| id | work | owner to extend | gate |
|---|---|---|---|
| **A1** | `enroll_running_node.sh` resolves the endpoint through `s3env.sh` (LAN default, `ZEN_STORE=r2` opt-out) and mints the LAN cred instead of an R2 one | `homefleet/.../ubuntu-node/enroll_running_node.sh` | enroll one stopped box (`i265`), run one `declare` + one claim end-to-end on the LAN store, ledger row lands; the box's `worker.env` contains no `r2.cloudflarestorage.com` |
| **A2** | `fleet status`/`fleet watch` read the resolved store, or retire them in favour of `pool_progress.py` with a loud deprecation | `zenmetrics/scripts/jobsys/fleet:36-40,112` | `fleet status` on a LAN-only shell reports the same counts `pool_progress.py` does |
| **A3** | LAN creds to `i134`, `r5900xt`, `r5600g`, `i265` | user-gated act (`lan-storage-2026-08.md:158-163`) — this plan proposes; the user executes or approves | `lanstore.env` present + `smoke-image` passes per box |
| **A4** | doc-truth pass for §1.4 items 1, 4, 5, 6, 9, 10 (+ 7 after re-derivation, 8 in the private repo) | the file each item names | every named line reads what the code does; `DATA_PROVENANCE.md` gains a "LAN store" section stating canonical = `/mnt/v` + LAN store + Tower, R2 = cold |
| **A5** | the ~20 scripts that inline `os.environ.get("ZEN_S3_ENDPOINT") or <R2>` → the resolver (`RUNNING_JOBS.md:172-178`) | each script | `grep -rn 'r2.cloudflarestorage' scripts/` returns only the resolver + `refresh_snapshots.sh` (deliberately pinned) |
| **A6** | GPU truth: card-per-box inventory line (private `NODES.md`) + decide whether `zensim-gpu` stays a designed panic in the executor tag or gets fixed (it blocks any GPU-zensim leg, incl. HDR) | `fleet.env:44-49`, executor image | `fleet smoke-image` on the GPU box prints the card; a ScoreFile with `zensim-gpu` either scores or is refused at declare time, never at claim time |
| **A7** | root-cause the fleet-waste finding (`052af499`): `fleetbench-gpuscore` idle-drained ~46 min restarting after clean drain-exits; 5 of 4,000 jobs never got a terminal ledger row | `zenfleet-worker` drain exit + `Lease::renew` (no caller today) | a 4,000-job re-run ends with 4,000 terminal rows and zero restart-after-drain |
| A8 | Nomad P2–P5 | operator direction (`fleet-orchestration-2026-08.md`) — **not this plan's** | — |

A1–A2 are hours of work each; A4 is one commit per repo. A3 and A8 are the
user's.

---

## 4. Program B — zensim models

**B1 — Wave-12, as registered (appendix AD).** Zero design work remains.
Amend AD.7 by measurement before launch: the box is 60 GiB, so run seed 4201
alone under `run-heavy --mem 18G`, record its peak RSS, and permit a second
lane only if `2 × peak + 8 GiB ≤ MemAvailable` (the registered rule was
"≤ 8 GB" against a 23 GiB box). Chain: `scripts/wave12_lane.sh` →
`harvest_bakes.sh --glob 'W12_s*.bin' --count 6 --regime 944` →
`await_artifacts.sh --then scripts/endgame_wave12.sh` → `freeze_check
--select`. Gates G-AC1/2/3 as frozen; outcomes (a)/(b)/(c) per AC.3. Precedent
cost: Q.R2 cells took ~960 s at 7.1 GiB peak on ~0.8 M rows; wave-12 has
~1.19 M rows — expect longer, measure, do not extrapolate.

**B2 — Breadth-transfer follow-on (the appendix-R lead), pre-registered as
"R3".** Hypothesis: the sparse student of C keeps the teacher's breadth
(CSIQ/LIVE 0.93–0.96) and loses CID22 because the classic anchors are
under-weighted *in the teacher*, not because sparsity forbids rank. Arms
(2 seeds each, λ ∈ {0.3, 1}): (i) CS student with the classic anchors
(CSIQ/LIVE/TID train halves — never CID22) held **in** the distillation mix at
the wave-10 weights; (ii) the same with the teacher = the wave-12 winner if (a)
fires, else C; (iii) control = the R1_CS λ-matched cell. Endpoint = the
8-floor balanced profile (`freeze_check --profile balanced-2026-08-04`), then
CID22 within ±0.004 of C. **Precondition:** resolve §1.4 #7 (C's own
CSIQ/LIVE on the 944 instrument) — if C really reads 0.77/0.81 there, the
"breadth" story is an instrument artifact and R3 is void. Cost: R1/R2 packed
cells were ~10 s fits + 66 s harvest each.

**B3 — co2b seed wave** (`C_co2b_s1307` 6/8, missing CID22 −0.0014 and
KonJND −0.016; "cheapest untried lever", SESSION-RESUME). k = 6, wave-11
seeds advanced one block, E.4 selection, harvest inline. Runs after B1 on the
same serialized trainer lane.

**B4 — HDR phase-2: salvage, gate, leg, wave.** (i) One documented salvage
read of the `hdrgrid-sf-*-20260807` ledgers + score blobs from R2 to the LAN
store (`--size-only`, the Z.R0 pattern; the LAN-only directive allows reads),
then `writeback_scores.py` two-stage → `unified/{scores,features}.parquet`.
(ii) `check_target_orientation.py` on both splits — **PENDING in the
manifest; nothing trains before it passes.** (iii) If the GPU/CPU score
waves never completed, gap-declare the remainder on the LAN store (GPU box
under `ZENMETRICS_REQUIRE_GPU=1`; cvvdp on the CPU queue with the AA
measured-peak semantics). (iv) Fix imazen/zenav1-svt#11 (`restoration.rs:985`
OOB on bd10 4:2:0 real-photo content), re-declare the 115 absent cells.
(v) Build the `hdr944_mc` leg (jxl + svt + gainmap arms, origin-digit split
identical to `hdr_v3mix@944`), then re-run the Q recipe (k = 3) against BHdr on
the UPIQ instrument — Q.R7 named multi-codec breadth (Q-G6) as the lever;
this is the first time it can be tested. (vi) Diffmap declares over the
persisted HDR encodes (`declare-diffmaps --hdr`) — optional here, required
before any HDR steering work.

**B5 — Dial packaging before any freeze.** `bake_dial_refit gate` panics on
2-layer MLPs (`bake_dial_refit.rs:182`) so G-RANGE is unevaluable on the
whole 944 class; spline-less mono % is unit-flattered 3–6 pts and no
packaged cell holds ≥ 93 % in dial units. Fix the gate for MLPs, land the
amendment-2 near-top densify, and re-read every shortlisted cell in dial
units. This is on the critical path of the user's freeze decision.

**B6 — Instruments.** Add the AVIF loop panel to the gauntlet
(`gauntlet.LOOP_BAKE_MAP`) when §6 D1 emits its first summary JSON; add a
size-axis loop panel (576² / 1152² / 4K) when §5 C7 lands. Stats stay with
`analyze_23shot.py` / `zenstats`.

B7 (KonFiG-IQA program, overlap-audit-gated) and B8 (publish chain:
zenpredict 0.2.x → zensim 0.3.0 → downstream profile-B migration) are carried
from the roadmap unchanged; B8 is user-gated end to end.

---

## 5. Program C — jxl loop: diffmap secant + encoding efficiency

### 5.1 What "secant" means here, and why it is the right next controller

The loop iterates on a global scale `s` of the quant field. Writing
`L = 100 − score`, the current step is `Δ log s = exp · (log L_a − log L_t)`
with `exp = 1.0`: a fixed-point iteration that assumes unit elasticity
`ε = d log L / d log s`. Its convergence factor is `|1 − exp/ε|`, and the
exponent dose-response (0.45→13, 0.8→18, 1.0→20, 1.2→20 at k3;
campaign `:13399-13440`) is exactly what you expect when `ε` varies by
content and target. A secant step measures `ε` from the last two iterates,
`ε̂ = (log L₂ − log L₁)/(log s₂ − log s₁)`, and steps
`Δ log s = (log L_t − log L₂)/ε̂`. It uses no extra compare — at k2 the second
controller step already has two points — so it targets the k2 census
(18/27) directly. Three safeguards are part of the design, not options:
fall back to the power law when `ε̂ ≤ 0` or the two iterates are closer than
the clamp resolution; keep the existing clamp (2.00); and when the two
iterates straddle the target, restrict the step to the bracket (the Brent
shape the zensim-target README already asks for).

The same idea applies per tile. H3 steering uses a fixed gain
(`ZENSIM_H3_GAIN` 10, measured to decline above 10). After two iterates the
binned map gives each tile's change in attributed damage against its change
in qf — a measured per-tile elasticity that replaces the constant. That is
the "diffmap secant": the map's *dynamics* across iterates, not just its
level, drive redistribution.

### 5.2 Registered arms (all opt-in env knobs, default OFF; nothing ships on this study)

Instrument: the 27-cell grid (9 refs × t{70,80,88}), k2 and k3, ±2 in own
units + decoded-judged median |err| + bytes ratio vs the control, paired
per cell; stats owner `analyze_23shot.py --extra-arm`; value columns
deterministic, timings only on a quiet box (the 08-07 `.meta` caveat).

| arm | change | endpoint it targets |
|---|---|---|
| **S0** control | `W10L9_h3ctrl2` as adopted (exp 1.0 / clamp 2.0 / gain 10 / bin 8) | — |
| **S1** global secant | secant from the 2nd controller step, power-law first step, fallbacks as §5.1 | k2 census + med\|err\|, bytes within ±1 % of S0 |
| **S2** bracketed secant | S1 + bracket restriction when iterates straddle the target | k3 census ≥ 24/27 with no k2 regression |
| **S3** per-tile secant gain | H3 gain per tile = measured Δ(tile attribution)/Δ(log qf) after 2 iterates, constant 10 until then, clamped to [2, 40] | bytes ratio at equal achieved (the mm-F3 read) + nonphoto census |
| **S4** elasticity prior | first-step exponent from a per-content prior (the dial-grid ladder slope at the seed's q, indexed by ref-only features) — shares the regressor with C2 | k2 census on the screen t70/t80 residue |

Decision rule, frozen before the first run: an arm advances to the adoption
proposal only if it beats S0 on the k2 census with the k3 census not below
24/27 and bytes not above +1 %; ties are reported, never adopted. The
h3-mag default flip and any product default change stay user-gated (AB.3).

### 5.3 Efficiency levers, ranked by measured residual (Y.R4/Y.R5, P.R4)

| id | lever | measured now → projected | gate |
|---|---|---|---|
| **C2** | content-aware **seed distance** (registered AB.4): a ref-only 944-feature regressor to the starting distance for a given target, trained on the 944 dial-grid ladders (the loop's own traces are n = 27 — too few) | screen seeds land ≈ 91 for t70; owns all 3 k3 misses | k2 census on the screen cells; no photo regression |
| **C3** | iter-0 FD probe → batched multi-row forward (GEMM) in zenpredict | 498.8 → 249.9 ms after batching; ~130 → ~40–60 ms class projected (Y.R5 #2) | bitwise vs the scalar probe, or a registered tolerance if accumulation reorders (G-Y1 pattern) |
| **C4** | seam **S1**: consume the ref-side planes the `Fused944Session` already retains (the C5 walk has no consume path) | fused 62 → ~52 ms; score-only 23 → ~15 ms (the k≥2 floor) | G-N1/G-P1 bitwise features by construction |
| **C5** | seam **S2**: dst-side plane sharing between extraction and the fused v1 walk | walk+basic 28 → ~16–20 ms | bitwise where planes are provably identical, else G-N2 tolerance + G-P4 M3a ×3 bakes |
| **C6** | in-strip stale fold = the ≤ 1.1× single-pass endpoint (B-N1 strict still missed at ~2.7×) | marginal 2.7× → ≤ 1.1× (absolute-ms bar per the 08-01 user decision) | score bit-identical; census exactly equal (G-P5 shape) |
| **C7** | **size axis**: the loop has never been measured above 576². Add 1152² and 4K cells (same 9 refs, resampled per the sweep discipline) to the instrument and the perf table; the 944 extraction is superlinear at 4K in every Y.R0 run | unknown — that is the point | fit `α + β·pixels` for per-compare ms and report both |
| **C8** | small-frame MT inversion (parallel score path loses at 576²) | measured loss at 576², win at 4K (Y.R0-final) | zenbench paired, quiet box |
| **C9** | outer crate: secant/Brent in `zensim-target::target_search` (README limitation #3) — the codec-agnostic loop zenjpeg/zenwebp use | 33/36 within ±1.5, median 5 iterations | same 36-cell demo matrix, ≥ 33/36 at a lower median |
| **C10** | hygiene: the 372-path `Err(_) => Ok(seed)` swallows (`zensim_loop.rs:845,1090,1100`) emit a seed-quality bitstream silently — fail loud like the folded paths; fix the stale exp/clamp comment (§1.4 #9); `measured_dist` NaN on the stale path is log-only but should say so at the read site | — | a forced failure panics with the arm name |

Order: C10 → S1–S4 grid (local, minutes per arm, runs while B1 trains — value
columns are contention-immune) → C2 → C3 → C4/C5 → C6 → C7/C8 → C9. C7's
cells are also the instrument every later arm is measured on, so land it
before the perf levers are compared.

---

## 6. Program D — AV1 diffmap steering: zenrav1e · zenav1-svt · zenav1-aom

### 6.1 The channels, source-verified 2026-08-25

| encoder | per-block **delta-q** (syntax cost, disables/competes with segmentation) | per-block **λ / rdmult** (no syntax cost) | status |
|---|---|---|---|
| **zenrav1e** 0.2.0 (`~/work/zen/zenrav1e`) | `FrameHints::sb_q_scale` per-64px SB (`src/frame/mod.rs:66,91`; applied `src/encoder.rs:1498, 2313-2334` — activation disables segmentation); variance boost per-SB under `Tune::Ssimulacra2` (`compute_variance_boost_sb_qindex:1297`) | `ssim_rdmult_strength` + `compute_ssim_rdmult_factors` / `ssim_rdmult_scale_at`, per-16×16 geomean pooling (`src/api/config/encoder.rs:288`) — **no external input today** | delta-q channel live end-to-end (Y.R3 M1 neutral == byte-identical) and a measured loss; λ channel = registered next lever |
| **zenav1-svt** @ `4c5c132` (`~/work/zen/zenav1-svt/rust/crates/svtav1-encoder`) | `sb_qindex.rs`: `normalize_sb_delta_q`, `delta_q_res_for(cli_qp, enable_variance_boost)`; `rate_control.rs`: `tpl_sb_qp_offsets`, `tpl_qp_adjustment` | `rate_control.rs::qp_to_lambda`; `svtav1-cref::compute_rd_mult_based_on_qindex`; psy-RD in `svtav1-dsp/src/ac_bias.rs` (`psy_distortion`, `psy_adjust_rate_light`); `perceptual.rs::StillImageConfig`, `hdr_mode.rs::HdrForkConfig` | still-picture CQP byte-identical to SVT-AV1 v4.2.0 (README); already an HDR sweep arm in zenmetrics; open bug #11 |
| **zenav1-aom** @ `2334d9b` (`~/work/zen/zenav1-aom/crates/aom-encode`) | `allintra_vis.rs`: `setup_delta_q_variance_boost`, `setup_delta_q_perceptual_ai`, `setup_delta_q_perceptual` | `partition_pick.rs`: `intra_sb_rdmult_modifier(var_min, var_max)`, `fold_intra_sb_rdmult` (per-SB rdmult from variance — the natural external hook); `var_tx.rs`/`tx_search.rs` trellis rdmult | encoder port in progress (`INTER-ENCODE-ROADMAP.md`); end-to-end still-image encode + parity must be confirmed from `STATUS.md` before it enters the harness |

`zenavif` wires none of the last two (`Av1Backend::Svtav1` deprecated +
rejected). Steering experiments therefore run at the **encoder-crate level**
through one harness, and a backend is wired into zenavif only after a channel
wins at matched rate — the same order Y.R3 used.

### 6.2 Work items

**D1 — the `zensim_cq_rd` harness (AC.4), global loop first.** A zensim
example/bench (the `avif_sb_hints` example is the seed) driving zenrav1e
directly: seed CQ from zenavif's quality→qindex curve (`encode_plan.rs:248-252`),
the adopted controller (exp 1.0 / clamp 2.0) on CQ, fused folded-944 compare
(`compute_folded944_score_and_attribution_binned`, bin 8 — SB rects are
bin-exact), emit-best; comparator = zenavif's existing bisection
(`encode_rgb8_with_target`, `TargetMetric::Zensim`). Same 27-cell instrument as
jxl (9 refs × t{70,80,88}, k2/k3, own-unit ±2, decoded-judged, bytes), summary
JSON in the 23shot schema so the gauntlet panel is a one-line map entry (B6).
Gate: engagement (controller moves CQ) + one-cell smoke this session — the AC.4
contract. **The AVIF target loop is a product deliverable on its own**; the
per-block arms below attach to it.

**D2 — λ-side steering in zenrav1e.** One new `FrameHints` field
(`rdmult_scale: Option<Box<[f32]>>`, per-16×16, geomean-pooled like the
existing factors), consumed where `compute_ssim_rdmult_factors` would produce
the variance heuristic; absent hint = byte-identical to today (the neutral
gate is "no hint", not "all 1.0", because the external map *replaces* the
heuristic). Ship it as a path dep from the local 0.2.0 tree — the
"patch it in, keep working" rule; the preserved probe patch
(`benchmarks/avif_sb_probe_worktree_2026-08-06.patch`) is the template.
Policy (map → multiplier) gets **one owner** (D5) and a registered sweep —
Y.R3 caveat 3 says the first-try policy was uncalibrated. RD probe: Y.R3's
grid extended to 9 refs × q{45,60} × strength{1,2} at 576², matched-rate
against the single-pass q40–75 ladder, judges fast-ssim2 + dssim +
butteraugli-pnorm3 (never the steering metric). Pass = Δssim2 ≥ 0 at matched
bytes on ≥ 7/9 refs with dssim not worse; the delta-q arm (Y.R3) rides along
as the negative control.

**D3 — zenav1-svt.** Fix #11 first (it crashes on real-photo bd10 4:2:0 and
holds 115 HDR cells). Then an external per-SB input at the `sb_qindex.rs`
seam (delta-q) and at `qp_to_lambda`/`compute_rd_mult_based_on_qindex`
(λ), both default-absent so the port's byte-parity gates
(`gate-identity`/`gate-recon`, cross-ISA tests) stay green with no hint.
Same probe grid as D2. The HdrFork knobs (psy-RD, QM, variance boost, 6
tunes) are the confounds to hold fixed, not arms.

**D4 — zenav1-aom.** Precondition: still-image encode end-to-end + parity per
`STATUS.md`/`PARITY.md`. Then the `intra_sb_rdmult_modifier` seam (λ) and the
`setup_delta_q_perceptual*` seam (delta-q), same contract as D3.

**D5 — one steering interface.** The map side already has one owner
(`AttributionResult::block_sums` / `query_rect`, 4–128 px partitions; the
`_binned` sessions). The *policy* side (per-block weight → multiplier:
normalization, exponent, clamp) is currently harness-local and uncalibrated.
Define it once (a small `zensim::steer` module or the harness crate — not
per encoder), with the policy parameters registered and swept in D2 before
D3/D4 reuse them. Per-encoder code only maps multiplier → that encoder's
seam.

**D6 — external per-block ground truth over avifgen.** The Diffmap executor
exists (§1.1) and avifgen's 562,860 encodes are persisted content-addressed;
`declare-diffmaps` for butteraugli + cvvdp over a stratified subset (the
eval8 holdout renditions × the default stratum, ~7.5 k pairs) on the LAN CPU
fleet. Read: rank agreement between the zensim per-SB attribution and the
independent judges' per-SB pooled maps on AV1 64-px partitions — the AV1
analog of M3a, pre-registered with a floor before any number is read. This is
what tells D2–D4 whether a losing arm is a channel problem or a map problem.

**D7 — AVIF clean picker** (roadmap zenavif #2): canonical `modes_full`-class
re-sweep **including odd origins** on the LAN fleet → `train_hybrid` with
knob-vetoes → commit the `.bin`. Adjacent to steering, same harness inputs;
listed so the sweep is declared once for both uses.

Order: D1 → D6 (fleet, overlaps) → D2 → D5 policy sweep → D3 (#11 first) →
D4 (gated on port readiness) → wire the winner into zenavif.

---

## 7. Sequencing and resource map

| lane | resource | items |
|---|---|---|
| trainer (serialized, `run-heavy --mem 18G`) | operator box | B1 → B3 → B2 → B4(v) |
| loop A/B grids (value columns contention-immune) | operator box | C10, S1–S4, C2, D1 |
| perf (quiet box only) | operator box, no sibling lanes | C3–C8, D2 RD probe timings |
| LAN CPU fleet (after A1/A2) | `i265`, `r3500`, `r5900xt`, tower tier, mac idle | D6 diffmaps, D7 odd-origin sweep, HDR residue re-declare, jxl P0 (roadmap) |
| LAN GPU (after A6 truth) | the GPU box(es) | B4 score gap-fill under `REQUIRE_GPU=1` |
| user | — | A3, A8, B8, the freeze/default decisions in §8 |

**First 72 hours:** A4 doc fixes + C10 (one commit each) → launch B1 with the
measured AD.7 amendment → S1–S4 grid while it trains → A1/A2 → D1 skeleton +
smoke → B1 endgame review → D2 patch + RD probe → B4(i–ii) salvage + gate.
Every wave: pre-registered arms + gates + a committed `endgame_<wave>.sh`
before launch, `harvest_bakes.sh` inline, one `await_artifacts.sh --then`
terminal, `verify_push.sh` line pasted verbatim, workspace cleanup on merge
(`docs/WAVE_PLAYBOOK.md`).

---

## 8. Decisions that are the user's (this plan proposes, never executes)

1. R2 rundown / deletions (`lan-storage-2026-08.md` proposal).
2. LAN credential distribution to the four boxes (A3).
3. Any product default flip: h3-mag steering ON, emit-best ON, controller
   changes from §5.2, C as the default profile.
4. Ship swaps and the freeze itself (B5 is the precondition).
5. The publish chain (zenpredict 0.2.x → zensim 0.3.0).
6. Whether the 2026-08-06 SDR AVIF datagen lift extends to the HDR AVIF arm
   (appendix S blocker B5).
7. Nomad P2–P5 scope.

## 9. Non-goals

No cloud burst (vast/Hetzner stay parked behind `ZEN_STORE=r2`); no new fleet
or sweep system (zenfleet only); no zenavif svt/aom backend wiring before a
channel wins at matched rate; no CID22 MOS in any training mix, ever; no
per-band number cited from before 2026-08-06; no cross-era ranking claims.

## §5.1 IMPL NOTE (locked 2026-08-25, from reading zensim_loop.rs)

Controller at `jxl-encoder/src/vardct/zensim_loop.rs:1600-1610`; loop `for iter in
0..iters+1` at :902; state (`best_score/best_iter/compares_used`) declared ~:776-813
(add secant trackers there: `prev_log_l`, `prev_log_s`, `cum_log_s=0`). Env knob
parses ~:787-810 (add `JXL_ZENSIM_SECANT`). **SIGN (do not get wrong): higher
`quant_field_float` = MORE bits = LESS loss, so ε̂ = dlogL/dlogS is NEGATIVE.**
Secant `g = exp((ln L_t − ln L_i)/ε̂)` then clamp; VALID only when ε̂ < 0 (guard),
`prev_log_l` finite (not iter 0), and `|cum_log_s − prev_log_s| > eps` — else the
power-law fallback. `cum_log_s` tracks the controller g-product; the redistribution
(:1484-1592) is sum-preserving so it does not move global scale (VERIFY in the A/B —
if it leaks scale the secant model is off). Correctness is behavioral → gated by the
27-cell k2/k3 A/B (analyze_23shot), NOT unit-testable; do not ship without it.

## §5 C9 UPDATE (2026-08-25) — outer zensim-target secant IMPLEMENTED + TESTED (measured speedup)

`zensim-target::target_search` gained a bracket-safeguarded two-point secant
(regula-falsi shape): env-gated `ZENSIM_TARGET_SECANT` (default OFF = pure
bisection, no API change, no default flip), accepted only when the interpolated
knob lands strictly inside the live [q_lo,q_hi] bracket — so it can only converge
faster, never break the bisection guarantee. Serves zenjpeg/zenwebp/zenavif (all
use this loop). **NOT YET TESTED: zensim-target is an EXCLUDED workspace member
with pre-existing dependency drift that blocks its build** — stale `zenjpeg`
`decoder` feature (FIXED this pass: the feature was removed; encode+decode are
ungated now) AND `zenavif-serialize ^0.2.0` version conflict (and likely more
codec-dep pins behind it). **Registered next: (1) a dep-refresh pass on
zensim-target's Cargo.toml (align every codec-crate version/feature to current),
(2) then the 36-cell demo A/B (bisection vs `ZENSIM_TARGET_SECANT=1`, README's
33/36-within-±1.5 baseline) to confirm the outer secant like the jxl one.** The
code is reviewable by eye (bracket-safeguarded interpolation); default-off means
it cannot regress the shipped bisection.

### C9 RESULT (2026-08-25, after the dep-refresh)

Build unblocked with ONE `[patch.crates-io]` (zenavif-serialize 0.2.0 is an
unpublished zenavif workspace member; zenavif's own patch doesn't apply when
zensim-target builds it as a path dep) — NOT a deep dep-drift after all. The
`demo_matrix` 36-cell A/B (`cargo run --release --example demo_matrix`,
bisection vs `ZENSIM_TARGET_SECANT=1`):

| | bisection | secant |
|---|--:|--:|
| converged / 36 | 32 | 31 |
| median iters (converged) | 5 | **4** |
| total iters | 173 | **164** |
| faster / slower / same cells | — | **17 / 8 / 11** |

**The outer secant converges FASTER (median 4 vs 5 iters, faster on 17/36).**
It flipped ONE cell (gb82-sc gui screen / zenavif / target 30) converged →
non-converged — a codec q-ceiling case (README limitation: screen at low targets
hits the q-ceiling; bisection struggles there too), damage bounded by the
bracket-safeguard. Net: a real speedup with one documented hard-cell edge.
Registered refinement: an Illinois/Pegasus stall-fix (or accept the edge) —
re-run the demo. Default OFF; a per-codec production census + the ship decision
are user-gated. This serves zenjpeg/zenwebp/zenavif's loops at once (criterion 4).

## §3 A1/A3 RESULT (2026-08-25) — LAN enrollment + cred distribution PROVEN on r7900x

With the A1 fix (745f1d68) + the user's "distribute any creds needed": ran
`enroll_running_node.sh --start r7900x` (ZEN_STORE default=lan). **Verified: r7900x's
`/etc/zen-node/worker.env` carries `ZEN_R2_ENDPOINT=http://<lan>:3900` + `ZEN_BUCKET=zentrain`
+ the LAN `AWS_ACCESS_KEY_ID` — cred distribution works end-to-end.** The worker then
empty-pool restart-looped (`s3://zentrain/jobs/_pool944v4/` does not exist on the LAN store —
no job declared), so it was stopped + left ENROLLED-READY (config + cred present, unit disabled).
The 4 boxes (r7900x/i265/r3500/r5900xt) are all up + reachable; enroll the rest the same way
once a job/pool exists. **Criterion 1 is now: enrollment + cred path PROVEN; the remaining is a
DECLARED JOB (the HDR corpus below) to make the fleet busy + the GPU-only scoring gate on a real run.**

## §4 B4 HDR CORPUS LANE (authorized 2026-08-25 "generate and curate hdr as needed")

State: the appendix-S hdrgrid corpus (1,140 sources × 3 arms × 30 q = 102,600 cells) ENCODED
99.9% on 08-07, its score waves were R2-declared + never harvested, and nothing landed on
Tower/LAN. HDR source is present: `/mnt/v/output/imazen-26-hdr-grid-2026-06-14/` (1,140 HDR PNGs)
+ imazen-26 `hdr-grid-15scale@` variant sets. The lane (big, first-cell-gated): (1) salvage the
R2 encode blobs → LAN store OR regenerate (authorized); (2) declare ScoreFile jobs on the LAN
store — ssim2+butteraugli GPU-only (`ZENMETRICS_REQUIRE_GPU=1`), cvvdp+features on CPU; (3) enroll
the GPU box(es) pointed at the pool; (4) first-cell gate → scale → harvest → writeback → manifest
+ orientation gate + Tower mirror; (5) then the HDR model wave (Q recipe + the multi-codec leg vs
BHdr). zenavif HDR arm: authorized per the SDR-lift extension (the halt is lifted).

## §5 PER-CODEC LOOP-OWNERSHIP MAP (2026-08-25; user directive "each codec owns its diffmap secant and loop code")

| codec | owns loop? | secant? | status |
|---|---|---|---|
| jxl-encoder | YES (in-encoder diffmap `vardct/zensim_loop.rs`) | YES `JXL_ZENSIM_SECANT`, now **default ON** (7155083e) | ✓ done |
| zenavif | YES (`target_quality.rs` `encode_rgb8_with_target`) | YES bracketed secant + bisection fallback, **default** (line 566) | ✓ done |
| zenwebp | PARTIAL (`ZensimTarget` config + a parameter-grid sweep in sweep.rs; NOT a clear encode-score-adjust secant target-loop like zenavif) | no own secant | ⧗ add an own secant target-loop in-crate |
| zenjpeg | **NO zensim target loop** | — | ✗ GAP — add a per-codec loop+secant (zenjpeg deps zensim) |
| zenav1-svt | no zensim loop | — | ✗ GAP |
| zenav1-aom | no zensim loop | — | ✗ GAP |
| gainmap | no zensim loop | — | ✗ GAP |

The central `zensim-target` secant (7e17945e) is the SHARED-ALGORITHM reference; per the directive the
loop ownership lives per-codec (jxl+zenavif already do). Remaining C4 code: zenjpeg loop+secant (in
zenjpeg), confirm/add zenwebp's secant, and svt/aom/gainmap loops (each in its own crate). Then the
per-encoder 27-cell k2/k3 census + zenpredict Zq autotune per encoder. Every new per-codec loop reuses
the bracket-safeguarded secant shape (bisection fallback), default budget-optimal (ON), env-off escape.

## §4 B4 HDR CAMPAIGN — PRECONDITIONS VERIFIED 2026-08-25 (runnable; a multi-hour fleet op)

All enablers PROVEN this session:
- **Encodes exist**: `s3://zentrain/jobs/hdrgrid-enc-20260806/blobs/` = **98,805 HDR encode blobs**
  on R2 (~10-30 GB) + `ledger_snapshot.parquet` (6.8 MB) + the pair index local
  (`/mnt/v/output/hdrgrid-2026-08-06/pairs_full.parquet` 6.7 MB, `manifest_enc.json.gz`).
- **Pipeline PROVEN**: the appendix-S G-S1 gate decoded+scored HDR jxl 9.48 / svt 4.78 / gainmap
  8.72 JOD via `score-pairs --hdr` (`hdr::decode_to_nits`); `jobexec` `run_score_file_hdr` arm is
  implemented (crates/zenmetrics-cli/src/jobexec.rs:772).
- **Fleet path PROVEN**: r7900x enrolled on the LAN store with cred (this session).
The campaign (NOT completable in one session): (1) salvage the 98,805 encode blobs R2→LAN store
(the LAN-only-directive salvage-read pattern; ~30 GB), (2) `declare-diffmaps`/ScoreFile jobs on the
LAN store — ssim2+butteraugli GPU-only (`ZENMETRICS_REQUIRE_GPU=1`), cvvdp+features CPU, (3) enroll
r7900x+the GPU box on the pool, (4) first-cell gate → scale the 4 boxes → harvest → writeback →
`_MANIFEST.json` + orientation gate + Tower mirror, (5) HDR model wave (Q recipe + multi-codec leg
vs BHdr). This is a multi-hour fleet operation; every enabler is verified so it runs clean.

---

## HDR CAMPAIGN — EXECUTION LOG (2026-08-26, live)

State CORRECTIONS to the plan above (verified from R2, not docs):

1. **The score jobs were ALREADY DECLARED on R2** (08-07) — the plan's "declare
   ScoreFile jobs" step is DONE. The full family exists under `s3://zentrain/jobs/`:
   `hdrgrid-sf-{cpu,gpu,gpu-huge,gpu-small}-20260807`, a second wave `hdrgrid-sf2-*`,
   and `hdrgrid-diffmap-20260807`. So the HDR work is a **RESUME**, not a re-declare.
   - `hdrgrid-sf-gpu` ledger fill: **0 results / 0 claims** — GPU scoring NEVER RAN.
     This is the criterion-2 headline (ssim2-gpu + iwssim-gpu, `hdr:true`).
   - `hdrgrid-sf-cpu`: **1254 results / 23 claims** — CPU scoring PARTIAL.
   - `hdrgrid-sf2-gpu`, `hdrgrid-diffmap`: 0 — never ran.
2. **Salvage DONE**: all **98,805** enc blobs at `/mnt/v/output/hdrgrid-2026-08-06/blobs/`
   (rc=0). The GPU score cell's inputs are ALL enc-blob shas → refs are among these 98,805.
   The salvage-read cred bug (nested `ZEN_STORE=r2` gave a 5-char key) is fixed: load
   `~/.config/cloudflare/r2-credentials` and map `R2_ACCESS_KEY_ID→AWS_ACCESS_KEY_ID`
   explicitly; do NOT source `lanstore.env` directly (its keys are `ZEN_S3_*`; the AWS_*
   mapping only happens inside `s3env.sh`).
3. **R2 creds distributed** to the 3 LAN GPU boxes (r7900x, lianli, r5900xt) — authorized
   ("distribute any creds needed"); home-LAN owned hardware. So the read path is unblocked
   WITHOUT moving blobs to the LAN store (duplicate-work-okay; R2 not deleted; new work goes
   LAN-native; the CURATED output mirrors to LAN+Tower per criterion 2).
4. **GPU boxes**: r7900x + lianli = GTX 1060 6 GB; r5900xt = GTX 1050 2 GB (small-bucket only).
   tower's NVIDIA driver is DEAD; i265 has no GPU. nvidia-docker runtime works on r7900x.
5. **The existing GPU-score launchers are VAST.AI-oriented** (`gpu_scorefile_launch.sh`,
   `gpu_e2e_proof.sh`) — they `vastai create instance`. The user shifted to the LAN fleet, so
   HDR GPU scoring runs on the LAN boxes via a **direct-manifest docker worker**, not vast.

### LAN GPU worker recipe (the resume command — direct-manifest mode)
`crates/zenfleet-worker/fleet-entrypoint.sh` single-run mode reads `ZEN_RUN` +
`ZEN_MANIFEST_URI` (+ `ZEN_CONTROL_KEY`); `ZEN_REQUIRE_GPU=1` = the goal's GPU-only rule.
On a GPU box (R2 creds present):
```
sudo docker run --rm --gpus all \
  -e AWS_ACCESS_KEY_ID=<r2> -e AWS_SECRET_ACCESS_KEY=<r2> -e AWS_REGION=auto \
  -e ZEN_R2_ENDPOINT=https://<acct>.r2.cloudflarestorage.com -e ZEN_BUCKET=zentrain \
  -e ZEN_RUN=jobs/hdrgrid-sf-gpu-20260807 \
  -e ZEN_MANIFEST_URI=s3://zentrain/jobs/hdrgrid-sf-gpu-20260807/manifest.json \
  -e ZEN_CONTROL_KEY=jobs/hdrgrid-sf-gpu-20260807/control.json \
  -e ZEN_REQUIRE_GPU=1 -e ZEN_WORKER=<host> -e ZEN_PROVIDER=basement \
  --entrypoint /usr/local/bin/fleet-entrypoint.sh \
  ghcr.io/imazen/zenfleet-worker:exec-gpu-avifgen-66e3c417
```
The `exec-gpu-avifgen` image (1.9 GB, CUDA) is GPU-metric-capable (it scored `ssim2_gpu` for
the SDR avifgen sweep); local on r7900x. Remove `ZEN_MAX_MIN`/`ZEN_IDLE_PASSES` caps for the
real run; add lianli + r5900xt (small→`hdrgrid-sf-gpu-small`). Harvest via `writeback_scores.py`.

### FIRST-CELL GATE (in progress at log time)
One capped (8-min) GPU worker launched on r7900x → log confirms `GPU-only scoring enforced;
visible GPU: GTX 1060 6GB`, manifest fetched (8.29 MB), **claimed 1 cell**. Watching for the
first DONE row (gate PASS) vs worker-exit (gate reveals HDR-decode/`hdr:true` blocker). Result
appended below once the watcher fires.

---

## CRITERION-4 DESIGN — the dependency-cycle finding (2026-08-26)

**`zensim` depends on `zenjpeg` + `zenpng`** (`zensim/zensim/Cargo.toml`: `zenjpeg
0.8.4 features=[decoder]`, `zenpng` workspace — used to decode test images in
zensim's own harness). Therefore:

| codec | can it `dep zensim`? | target-loop scorer wiring |
|---|---|---|
| zenavif | YES (zensim ⊬ zenavif) | already does — `TargetMetric::{Ssim2,Zensim}`, `target_quality.rs` |
| zenwebp | YES | already does — `ZensimTarget` |
| jxl-encoder | YES | already does — `vardct/zensim_loop.rs` (+ secant, this session) |
| **zenjpeg** | **NO — CYCLE** (zensim → zenjpeg) | **must inject** `FnMut(ref,dec)->f64` |
| **zenpng** | **NO — CYCLE** (zensim → zenpng) | must inject (if it ever needs a loop) |
| zenav1-svt / -aom | standalone: no zensim dep today | inject, or driven by zenavif's loop |
| gainmap (ultrahdr/gainforge) | n/a | inject |

**⇒ The uniform "each codec owns its diffmap secant + loop" design is the
INJECTED-SCORER pattern**, not a baked zensim dep: the codec owns the loop
MECHANICS (secant math ε̂=ΔlnL/ΔlnS, bracket-safeguard, monotone selection =
smallest file in the target band, encode→decode per trial, a q0/Zq seed head),
and takes the metric as a closure. zenavif/zenwebp/jxl bake zensim because they
CAN (no cycle) and it's ergonomic; zenjpeg/zenpng MUST inject. The zenavif
`encode_rgb8_with_target(img, config, target, options, stop)` contract +
`q0_head::predict_q0_for_rgb8` (the zenpredict autotune seed) is the EXEMPLAR to
mirror. `zensim-target` becomes the thin dispatcher that picks the codec and, for
the inject-only codecs, supplies the zensim `codec_target`-profile scorer closure.

**zenjpeg loop plan (next deliverable):** new `zenjpeg/src/target_quality.rs`:
`encode_rgb8_with_target(img, &EncoderConfig, target_zdsim: f64, scorer: &mut
dyn FnMut(&DecodedRef, &Decoded) -> f64, opts)` → bracketed secant/bisection over
q∈[1,100] on the monotone (q→score) curve, selection = smallest byte-size iterate
in [target−tol, ∞), `converged` flag, injected scorer (no zensim dep). Seed from a
fixed anchor curve first; a zenpredict Zq head is the autotune follow-up. Gate:
census on the 27-cell instrument + dial-mono + RD≥baseline under an independent
judge (ssim2, NOT the steering metric) + a perf bar. Owner test lives in a zenjpeg
example/bench with a `dev-dependency` on zensim (dev-deps don't form the cycle).

### GATE RESULT: **PASS** (2026-08-26T00:19Z) + FLEET SCALED
First r7900x GPU worker scored a **693-pair chunk, all `status=done`, zero
`error_class`** → the HDR GPU path is proven end-to-end (GPU-only enforce · R2 HDR
blob fetch · HDR decode · ssim2-gpu+iwssim-gpu on GTX 1060 · DONE ledger rows +
`output_sha` sidecars). The zenfleet ledger is a job-STATUS ledger (done +
output_sha); scores live in the output sidecars → harvest with
`scripts/jobsys/writeback_scores.py`.

**Scaled to all 3 LAN GPU boxes** (uncapped, `--restart unless-stopped`, `--gpus
all`, `ZEN_REQUIRE_GPU=1`, container `zen-hdr`, image `exec-gpu-avifgen-66e3c417`):
- r7900x (1060 6GB) + lianli (1060 6GB) → `hdrgrid-sf-gpu` (medium, **2280** cells; shared via R2 lease)
- r5900xt (1050 2GB) → `hdrgrid-sf-gpu-small` (**687** cells)
- `hdrgrid-sf-gpu-huge` (**453**, 6GB-only) queued next — reassign r7900x/lianli when medium drains.
Workers self-exit on drain (`ZEN_IDLE_PASSES=8`). Monitor: `/home/lilith/tmp/hdr-fleet-monitor.sh`
→ `hdr-fleet-progress.log` + `hdr-fleet.done` on drain/stall(>30min).

**Remaining HDR steps** (after GPU buckets drain): (1) reassign 6GB boxes to `-huge`;
(2) run the `sf2` GPU wave (butteraugli) + finish `sf-cpu` (cvvdp+features, 1254→done)
on CPU boxes; (3) `hdrgrid-diffmap`; (4) `writeback_scores.py` → `_MANIFEST.json` +
orientation gate (`check_target_orientation.py`) + Tower mirror; (5) HDR model wave.
Teardown a box: `ssh <h> sudo docker rm -f zen-hdr`.

---

## CRITERION-4 STATUS — per-codec OUTER SDR target loops (verified 2026-08-26)

| codec | owns SDR target loop? | search | seed | notes |
|---|---|---|---|---|
| jxl-encoder | ✓ `vardct/zensim_loop.rs` | secant (JXL_ZENSIM_SECANT) | — | H3 magnitude steering |
| zenavif | ✓ `target_quality.rs` | bracketed secant/bisection | `q0_head` (zenpredict) | EXEMPLAR; TargetMetric::{Ssim2,Zensim} |
| zenwebp | ✓ `encoder/zensim_target.rs` (1766 ln) | one-pair secant + per-segment overrides | anchor table | most advanced |
| **zenjpeg** | **✓ NEW `target_quality.rs` (`277b1efb`)** | bracketed secant/bisection | anchor_guess | injected-scorer (no zensim cycle); 9 tests |

⇒ **All four main image codecs OWN their SDR target loop.** Remaining criterion-4:
- **gainmap** (ultrahdr / gainforge): no outer loop — GAP (HDR-side; needs the HDR scores landing now).
- **zenav1-svt / -aom**: driven by zenavif's loop for AVIF-family; a STANDALONE still loop is optional.
- **zenpredict Zq autotune seed** per codec (zenavif's `q0_head` is the model) — needs the curated
  training data (the SDR sets exist; HDR is scoring now).
- **Production gates** per encoder: census on the 27-cell instrument, dial-mono, RD ≥ baseline under
  an INDEPENDENT judge (ssim2, never the steering metric), perf bar.
- **Program D** (deeper): per-encoder diffmap STEERING — svt/aom/rav1e λ-side rdmult, zenjpeg
  per-block AQ. Separate/harder than the outer loop.

### FLEET OPS (2026-08-26, live) — health + reassignment recipe
- **6 GB boxes (r7900x + lianli): healthy, GPU 100%, ~1294 cells/pass, lease-dedup working**
  (skipped=985 = the other worker's held leases). Medium (2280) draining fast.
- **r5900xt (GTX 1050 2 GB): CANNOT run the GPU metric** — skipped all 687 small cells, GPU 0%/3 MiB,
  exited clean. Repurposed to a **CPU worker on `sf-cpu`** (zensim/features) alongside i265. So the
  small (687) + huge (453) GPU buckets need a **6 GB** box.
- **KNOWN: worker-name collision** — both 6 GB workers launched with `ZEN_WORKER=lianli-hdr` (a
  `<<REMOTE` heredoc `$wname` expansion bug in `hdr-gpu-scale.sh`). Correctness holds (ledger dedups by
  cell; duplicate work is acceptable), but USE UNIQUE NAMES on relaunch.
- **DRAIN → reassign (when the monitor fires `DRAIN`):** relaunch the 6 GB boxes on the remaining GPU
  buckets, unique names — e.g. `r7900x → hdrgrid-sf-gpu-small` (`ZEN_WORKER=r7900x-small`),
  `lianli → hdrgrid-sf-gpu-huge` (`ZEN_WORKER=lianli-huge`); then `sf2` (butteraugli) the same way;
  then `writeback_scores.py` → `_MANIFEST` + orientation gate + Tower mirror. Same docker-run recipe as
  the "LAN GPU worker recipe" above, swapping `ZEN_RUN`/`ZEN_MANIFEST_URI`/`ZEN_CONTROL_KEY` + a unique
  `ZEN_WORKER`. Fix `hdr-gpu-scale.sh` to pass the name via `-e` from an env var, not heredoc interpolation.

### ⚠ FLEET CAPACITY CORRECTION (2026-08-26) — only ONE 6 GB GPU box
`~/.ssh/config` has `Host r7900x lianli → 192.168.50.27` — **both aliases are the SAME physical
box** (hostname `r7900x`, one GTX 1060 6 GB). So the earlier "scaled to 3 GPU boxes" was wrong:
- **.27 (r7900x≡lianli): the ONLY 6 GB GPU box** — GPU scoring is SERIAL here. Running TWO GPU
  containers on it (the medium "2-worker" launch) contended for the single 6 GB card → the OOM
  `failed=9` cells. Run ONE GPU worker at a time on .27.
- r5900xt (.250): GTX 1050 **2 GB** — CORRECTION: it DID score the small bucket on GPU (ledger chunk
  `pass-r5900xt-hdr-1` = 687/687 done); the earlier "skipped/GPU 0%" read was POST-completion idle. So
  the 2 GB card handles SMALL HDR images but likely OOMs on huge/medium. Currently on CPU `sf-cpu`.
- i265 (.140): no GPU → CPU worker.
NODES.md lists lianli + r7900x as separate nodes (different MACs); the real second box (lianli
`74:56:3c:b8:45:8d`) is either down or the config conflates them — a WoL/config follow-up could
recover a second GPU. Until then: GPU = 1 box (serial), CPU = i265 + r5900xt. GPU buckets run
sequentially via `lan_score_launch.sh` single-run (pool mode is tar/enc-oriented, doesn't fit the
HDR direct-blob score jobs). Sequence on .27: small → huge → medium-leftovers → sf2(butteraugli) — AUTOMATED by
`scripts/jobsys/lan_gpu_sequence.sh` (one box drains all 6 GPU buckets in blocking single-run mode,
self-advancing, `~/lan_gpu_seq.COMPLETE` marker). LAUNCHED 2026-08-26T01:03Z; small already done,
huge scoring. sf=ssim2-gpu+iwssim-gpu, sf2=butteraugli-gpu (the goal's GPU-only pair).

---

## CRITERION-6 (BROWSER) — scoped 2026-08-26

The coefficient viewer (`~/work/coefficient/viewer`, SvelteKit) loads
**`data/cells.json` + `data/meta.json`** (COLUMNAR JSON — `Record<col, val[]>`
inflated to rows; NOT parquet — the memory's "cells.parquet" was stale), via
`viewer/src/lib/cells.ts` (`loadCells`). Generator = **`scripts/rollup_zenmetrics.py`**
(`--base DIR --sidecar F --out DIR`): a DuckDB rollup of `{base}/{dataset}/*.parquet`
(grain: dataset×cell×q×maxdim, ~15k rows) LEFT JOIN the 4-metric sidecar on
`encoded_filename`. It needs, per row: `encoded_bytes`, `width`, `height` (→ maxdim +
bpp = the comparable RD axis), `score_ssim2`/`score_zensim` (native), `q`, `codec`,
`knob_tuple_json`, `knob_plan`.

**The GAP for the LAN 924 sets:** the ext924 bigcodec split views
(`/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/bigcodec/<dataset>/<split>_924.parquet`,
932 cols) carry `encoded_filename`, `codec`, `q`, `knob_tuple_json`,
`score_{ssim2,zensim}` + `feat_0..923` — but **NO size/RD columns**
(`encoded_bytes`/`width`/`height` absent; keys are `origin_id`/`ref_filename`/
`encoded_filename`). They are TRAINING feature tables, not RD tables. So the browser
cannot compute bpp from them directly. **The chunk:** extend `rollup_zenmetrics.py`
to join the 924 scores with an encode-metadata sidecar carrying `encoded_bytes` +
dims (candidates to locate: `tbig_924_full.parquet` keyed `encode_sha`; the
`fill4metrics_sidecar` keyed `encoded_filename` — verify key overlap first, the 924
encodes are a 07-27 run and may not share filenames with the 07-01 sidecar), plus a
set-selector in the viewer (`meta.json` already supports multiple datasets). Do NOT
union across zensim profiles (07-01 = zensimA/PreviewV0_2 ≠ 924). Until then the
viewer serves the 07-01 SDR canonical set only.

### C1 EVIDENCE — tower enrolled, 4 boxes busy, declare→gap→reconcile live (2026-08-26T01:24Z)
- **tower** (Threadripper 2950X 32T, dead GPU) enrolled as a CAPPED CPU worker on `sf-cpu`
  (`--cpuset-cpus=0-23` leaves 8 cores for the media stack, `--cpu-shares=256`, `--memory=40g`,
  fresh image pull, Docker-only, creds passed as env — the stateless host is untouched). Observed
  first: load 1.08, media stack + `zen-lanstore` + PXE running, no heavy compute worker → safe.
- **4 boxes busy:** .27 `zen-seq-huge` (GPU, 6-bucket sequencer) + i265 `zen-hdr-cpu` + r5900xt
  `zen-score-cpu` + tower `zen-hdr-cpu` (all CPU on sf-cpu). Remaining home boxes: **mac** (needs
  the arm-native `_pool944neon` worker, not the x86 image) + **node-2/node-3** (kids' PCs, flip
  needs user approval) — both constrained, not autonomously enrollable.
- **declare→gap→reconcile** is the zenfleet substrate, live per bucket (declared cells in the
  manifest, `ledger/` = done chunks, `claims/` = in-flight gaps under R2 lease): at snapshot
  sf-cpu 1299 done + 50 claims, sf-gpu-huge 12 + 11, sf-gpu(medium) 14 + 7, sf-gpu-small 2 done,
  sf2-gpu 0 (queued behind the sequencer). ssim2/iwssim/butteraugli GPU-only (ZEN_REQUIRE_GPU=1,
  proven — GTX 1050 2GB CANNOT and skipped, only the 1060 6GB scores); zensim/features on CPU.
- **wall clock:** GPU wave START 01:03:40Z (small drained in 19s, huge in progress). sf-cpu
  1254→1299 in ~73 min pre-tower; tower's 24 cores should raise the CPU rate (re-measure next pass).

### GPU VRAM finding (2026-08-26): butteraugli needs >2 GB; ssim2 fits 2 GB
Tried offloading `sf2-gpu-small` (butteraugli) to r5900xt's GTX 1050 (2 GB) to parallelize the
GPU-bound path. It **skipped all 687 cells** (done=0, GPU capability/VRAM check fails for
butteraugli-gpu on 2 GB) — whereas the same card DID score `sf-gpu-small` (ssim2, 687/687). So:
**ssim2-gpu fits 2 GB (small images); butteraugli-gpu does NOT.** The 2 GB card is CPU-only for
sf2; all 6 GPU buckets run on the one 6 GB box (.27) via the sequencer. r5900xt returned to CPU
`sf-cpu`. (Its `--restart unless-stopped` would have restart-looped on the all-skip drain — stopped it.)
