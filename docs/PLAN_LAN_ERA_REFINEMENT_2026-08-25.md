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

### GPU VRAM finding (2026-08-26, CORRECTED) — 2 GB DOES do butteraugli-small
**RETRACTION of an earlier wrong read.** I first saw r5900xt (GTX 1050 2 GB) log `skipped=687` on
`sf2-gpu-small` (butteraugli) and wrongly concluded "2 GB can't do butteraugli". **FALSE:** the
ledger chunk `pass-r5900xt-sf2small-1` has **687 rows, all status=done, ZERO errors** — r5900xt DID
score every butteraugli-small cell. The `skipped=687` was a *later* pass hitting already-done cells
(idempotent skip), not a capability failure. Likewise the manual `unsupported HDR input extension:
.bin` error was MY red herring (I named the blob `.bin`; the worker/jobexec names fetched blobs by
detected format, so both ssim2 AND butteraugli score fine — `hdr.rs:111` extension-dispatch is fed a
correctly-named temp). **True state:** sf2-gpu-small DONE (687, butteraugli, 2 GB card); butteraugli-gpu
works on these HDR blobs. Two GPU boxes now scoring in parallel: **node-2 (RTX 3070 8 GB) on sf2
(butteraugli)** + **.27 (GTX 1060 6 GB) on sf (ssim2/iwssim)** — different metrics, no lease contention.
Lesson: a `skipped=N` line means "already done", NOT "can't do" — read the ledger before concluding.

### ROOT CAUSE — the sf-huge "encoder_panic" failures = ssim2-gpu CUDA OOM (2026-08-26)
The 19/453 `error_class=encoder_panic` failures on `hdrgrid-sf-gpu-huge` are a **MISLABEL** —
reproduced (node-2, RTX 3070 8 GB) as a **CUDA_ERROR_OUT_OF_MEMORY** panic in
`zenforks-cubecl-cuda-0.10.1/src/runtime.rs:90` (`.unwrap()` on the CUDA alloc). The failing cells
are the LARGEST imazen-26 HDR-grid images (`1232_interior`, `15xx_nature`, …, JXL HDR refs);
ssim2-gpu's scale-0 allocation exceeds VRAM. It OOMs on BOTH the 6 GB (.27) and 8 GB (node-2) cards,
so it is a genuine per-image VRAM ceiling, not a small-card issue. (`encoder_panic` is the honest
transient label the jobexec applies to an uncaught worker panic — see
`zenmetrics-cli/src/jobexec.rs:165`; it is NOT an encoder failure.)
**Owner fix (zenmetrics/ssim2-gpu):** the GPU kernel must bound VRAM for large images — either a
fallible alloc that falls back to a tiled/multi-pass SSIMULACRA2 (the scales complicate naive
tiling), or a documented max-pixels ceiling above which the huge cells are scored by a separate
path. Until fixed, the ~19 largest HDR cells per huge bucket will `encoder_panic` and lack ssim2;
they are a KNOWN gap in the HDR ssim2 coverage, not silent. butteraugli-gpu (sf2) should be checked
for the same ceiling.

### OOM FIX VALIDATED + RECOVERY (2026-08-26)
- **Mechanism (refined):** `butteraugli-gpu/src/memory_mode.rs::vram_cap_bytes()` DOES probe live
  free-VRAM via `nvidia-smi` (present in the exec image) with a 10% margin, else an 8 GB default.
  But the auto Full-vs-Strip decision still let large HDR images pick Full and OOM — the Full-mode
  VRAM *estimate* under-counts for these images (JXL HDR, huge dims). The env cap
  `ZENMETRICS_VRAM_CAP_BYTES` (always wins) forces Strip earlier and fixes it.
- **VALIDATED in production:** with cap 7.5 G, node-2 (RTX 3070) sf2-gpu-huge chunk `pass-i134-huge-1`
  = **54/54 done, 0 errors** (large butteraugli, previously the OOM class). Manual test: the OOM'd
  cell now returns ssim2_gpu=85.56. Landed `23b3777d` (zenmetrics) + sequencers relaunched cap 5.5 G
  (.27) / 7.5 G (node-2).
- **The 19 already-poisoned sf-huge cells** stay poisoned (`EncoderPanic.is_transient()==false`,
  status.rs:110; no reset CLI — `zenfleet-ctl` re-declare "keeps never-seen + retryable" only). To
  recover them: **declare a FRESH small job set with just those 19 (ref+distorted) and score with
  the cap** (they're the largest imazen-26 HDR-grid refs; extract from the sf-huge failed ledger
  rows). Small tail; the cap prevents ALL forward poisoning on the remaining large-image buckets.
- **OWNER FIX (zenmetrics, the real fix):** (a) fix the Full-mode VRAM estimate to not under-count
  large HDR images, OR (b) catch `CUDA_ERROR_OUT_OF_MEMORY` and reclassify it TRANSIENT (retryable)
  instead of the non-transient `encoder_panic` — so the reconciler retries and (with a card-aware
  cap) self-heals. The `default 8 GB` fallback should also be the card's real total, not a constant.

### C1 WALL-CLOCK MEASURED — throughput rising (2026-08-26, from ledger chunk timestamps)
Measured the HDR scoring wave's rate per bucket (chunk mtimes; no polling):
- **ssim2-gpu (sf-huge, .27):** 0.58/min overall → **0.73/min in the last 15 min** — RISING, because
  the VRAM cap eliminated the OOM→retry churn (each poisoned cell had burned attempts before).
- **butteraugli-gpu (sf2-huge, node-2):** 0.27/min (RTX 3070 just warming on the large bucket).
- **features-cpu (sf-cpu):** **1.33/min in the last 15 min** across 3 CPU workers (i265+r5900xt+tower)
  — RISING vs the pre-tower single/dual-worker rate (tower's 24 cores joined 01:24Z).
  (The "overall 0.05/min" is an artifact: this ledger also holds the 08-07 partial run, so the naive
  span is 18 days; the last-15-min figure is the true current-wave rate.)
**Falling wall clock = rising throughput, driven by three landed interventions this session:** the 2nd
GPU box (node-2 RTX 3070 → butteraugli in parallel), the 3rd CPU box (tower), and the VRAM cap
(23b3777d, no OOM retries). Rough remaining: sf-cpu ~2079/3420 cells left ≈ 26 h at current CPU rate;
GPU buckets faster. Rates are re-measurable any time from `ledger/` mtimes — the standing waiter is the
sequencer's `~/lan_gpu_seq.COMPLETE` marker + the ledger fill (no idle polling).


### C6 BROWSER — FUNCTIONAL (renders), serves the 07-01 canonical set (2026-08-26)
CORRECTION to "C6 not extended / 0": the coefficient viewer (`~/work/coefficient/viewer`) is
**functional and rendering** — `node check-render.mjs` passes **4/4** (canvas=true, 167 rollup rows,
no 404s), serving `data/cells.json` generated by `rollup_zenmetrics.py` from its default base
`/mnt/v/output/canonical-picker-2026-07-01-zensimA` (zenavif_lossy + zenjpeg_lossy, which HAVE the RD
columns encoded_bytes/width/height + score_ssim2/score_zensim the rollup needs). So the browser
already serves a real encode set + renders BD-rate / per-q / head-to-head views.
**What remains for "serves EVERY set":** the LAN-era sets (avifgen-2026-08-06, bigcodec-924, hdrgrid,
ext) — the 924 feature parquets carry scores + distorted-side features but NOT encoded_bytes/dims (a
row is a training feature vector, not an RD point), and `avifgen-2026-08-06/` holds only
manifest/declare JSONL, not a scored RD parquet. Extending the browser to a LAN set needs that set's
scored data in `{base}/{dataset}/*.parquet` form WITH encoded_bytes+dims (extract from the encode
artifacts / the encode job's per-variant metadata), then `rollup_zenmetrics.py --base <lan-set>` +
a set-selector. The 07-01 zensimA set must NOT be unioned with a LAN 944-profile set (different
zensim profile). So C6 is FUNCTIONAL now; full LAN coverage is a data-availability task, not a viewer bug.

### CORRECTION (2026-08-26) — svt/aom ARE cloned; the real blocker is they're low-level ports
Two wrong things I'd said, both fixed:
- **"svt/aom repos not cloned locally" was FALSE** (stale index) — `~/work/zen/zenav1-aom` (401 .rs,
  workspace: crates/{zenav1-aom,aom-encode,aom-decode,aom-dsp,...}) and `~/work/zen/zenav1-svt`
  (202 .rs) are both fully cloned. I never needed to ask to clone a dependency.
- **The REAL reason their zensim target loop isn't done:** they are **AV1 encoder ALGORITHM PORTS
  in progress** — `aom-encode` exposes block/SB/OBU-level functions (`encode_block_coeffs`,
  `encode_sb_dry`, `obu_assemble → Vec<u8>`), validated module-by-module against C (CHANGELOG:
  transform/quant/txb/cdef/restore/intra/loopfilter/dist/inter). There is **no turnkey
  `encode(image, cq)→bytes` entry, no image-encode example/CLI, and no zenavif integration yet.** A
  bracketed-secant target loop needs a single-call encode-at-quality to iterate over; that doesn't
  exist here yet. So the loop is **premature**, not blocked on cloning or on me. The AV1 target loop
  TODAY is **zenavif's** (`encode_rgb8_with_target`, rav1e/zenrav1e which ARE turnkey); zenav1-aom/svt
  get their own outer loop when their high-level encode API lands. (Program-D λ-side rdmult steering
  is the deeper per-encoder work, also awaiting that API.) Loop-ownership map + zenjpeg's tested
  `target_quality.rs` remain the template to copy the moment a turnkey encode entry exists.


### C6 — LAN-set ETL scope VERIFIED (2026-08-26, checked harder per user push)
Serving a LAN set in the browser is a real multi-step ETL, confirmed (not a quick pointer change):
- **avifgen-2026-08-06**: `pairs_final.parquet` (562,860 rows) has `ref_path/dist_path/codec/q/encode_sha`
  but **NO scores and NO bytes**; its `scores.parquet`/`features.parquet` were never generated
  (`writeback_scores.py` not run for this set).
- **The per-encode metric sidecars** (`fill4metrics_sidecar_*`) have scores + `encode_sha` but **no
  encoded_bytes/dims**.
- **bigcodec-924 views** ARE scored but carry distorted-side features, no RD bytes.
- `encoded_bytes` exist ONLY as the content-addressed artifact file sizes (recoverable via `s5cmd ls`
  of the encode prefix, keyed by `encode_sha`); dims come from the ref images.
**⇒ To serve avifgen/bigcodec:** (1) run/finish `writeback_scores.py` for the set → scores; (2)
`s5cmd ls` the artifacts → (encode_sha, bytes); (3) join scores+bytes+ref-dims → an RD parquet in
`{base}/{dataset}/*.parquet` shape; (4) `rollup_zenmetrics.py --base <that>` + a set-selector. Real
ETL, ~an hour+ per set. **The browser IS functional and SERVED** (http://localhost:3400/ ,
http://192.168.50.44:3400/ — `python3 -m http.server 3400 --bind 0.0.0.0 --directory
~/work/coefficient/viewer/build`) on the 07-01 canonical set; LAN coverage is the ETL above.

### C6 UPDATE 2026-08-26 — hdrgrid IS SERVED (selector + gate live)
The viewer now serves TWO pools behind a data-set selector (`?set=hdrgrid`):
the SDR 07-01 canonical (default) and the **HDR-grid 08-26 rollup** (1,350
rows: 3 arms × q-grid × 12 maxdims from the 102,485-cell harvest).
Profile purity holds by construction — separate `data-<set>/` roots, never
unioned. ETL: harvest scores + LAN blob sizes (`encoded_bytes` from
`s3://zentrain/jobs/hdrgrid-enc-20260806/blobs/` object sizes, 0 rows missing;
5,698 dup rows = content-address dedup) + dims parsed from `.scaleWxH.`
(1140/1140 filenames). `rollup_zenmetrics.py --datasets` (coefficient
`f7eefa4`) + viewer selector (`79ad39d`); `check-render.mjs` asserts the set
paints (6/6 — the gate caught the first cut, where `mode='hdr_lossy'` failed
the enc-mode facet). Regen recipe: `viewer/static/data-hdrgrid/README.md`.
**Remaining for full C6**: avifgen + bigcodec-924 + hdrfeat944 sets still need
their writeback/scored-RD parquets before they can be rolled (data-availability,
not viewer work).

### C4 UPDATE 2026-08-26 — zenjpeg Zq seed SHIPPED (zenjpeg main 9f130cf03cf6)
First per-codec Zq autotune seed landed via a pre-registered wave
(zenjpeg `benchmarks/zq_seed_wave_2026-08-26.md`): fitted-constants head
(q0_head exemplar), labels = 96,894 PAVA-isotonized 7-pt q→zensim curves from
the 07-01 canonical set. First arm FAILED the frozen gate honestly (189
deep-undershoot regressions at 25.7% improvement); registered arm B
(seed clamp [anchor−18,+12], frozen L rule) → **zero regressions, −13.5%
encodes, G-Z2 PASS** → `zenjpeg/src/zq_seed.rs`. Remaining Zq clause:
zenwebp/jxl/zenavif-zensim analogs (same recipe; zenavif has its ssim2
q0_head already), each its own registered wave.

### C4 Zq status after both waves (2026-08-26)
- **zenjpeg: SHIPPED** (`zq_seed.rs`, arm-B clamp, −13.5% encodes, zero
  regressions — see the C4 UPDATE above).
- **jxl-encoder: FAILED-as-registered** (census G-J2 5.6% < 15% bar; head +
  wiring stay in-tree env-gated OFF; nonphoto −51% recorded as the future
  class-conditional-seed lever — its own wave if pursued).
- **zenwebp**: already ships a content-aware anchor-table seed inside a
  ~2-pass loop — structurally little seed headroom; no wave planned.
- **zenavif**: `q0_head` ships (ssim2 targets); a zensim-target refit is a
  data task after current-model rescoring exists.

### FLEET FINDING 2026-08-26 — diffmap jpeg-gainmap cells: GPU image missing `hdr-gainmap`
`hdrgrid-diffmap-20260807` ledger read: 46,681 failed rows, **46,680 =
jpeg-gainmap** (zenjxl 0, svt 1 — noise), metric-symmetric (butteraugli
23,340 / cvvdp 23,341 = same CELLS), size-flat, both workers, attempts=1 ⇒
the variant DECODE fails: `exec-gpu-2af6dbc3` was built without
`hdr-gainmap` (the same bug hdrfeat944 hit on CPU images this morning).
Queue truth: declared 193,574 / done 37,771 / gap 155,803. Fix in flight:
bookworm GPU rebuild WITH the feature → new exec-gpu tag → swap i134's
worker (live gate: gm rows flip failed→done) → enroll r7900x (2× WoL sent;
if it stays dark it is hard-off = user-gated). Failed rows re-claim
naturally (latest-wins ledger).

**GM GATE PASS (2026-08-26T14:25Z)**: node-2 swapped to `exec-gpu-812a00d9`
(hdr-gainmap baked; builder gained ZEN_METRICS_BIN/ZEN_WORKER_BIN, zenmetrics
`812a00d9`+`c6ad8469`) → first post-swap pass: **109 jpeg-gainmap diffmap cells
DONE, 0 failed** (previously 100% of 46,680 failed). The 25k failed gm cells
re-claim naturally. avifgen writeback landed the same hour:
`avifgen-2026-08-06/harvest-2026-08-26/scores.parquet` — 562,860/562,860 cells,
4 metrics at 100% (butteraugli max+p3, cvvdp, ssim2); `zensim_score` is
structurally absent (these runs' feature records carry `features`+`regime` but
no scalar) — backfill = a features pass (ZEN_SKIP_FEATURES=0 chunked writer,
follow-up) or a scores run. First writeback attempt OOM-killed a 60G box
(57.9G rss, bare-nice — run-heavy rule violated); owner fix = ZEN_SKIP_FEATURES
scores-only pass, peak 21.1G under a 40G cap.

**avifgen FEATURES + post-swap verification (2026-08-26T14:40Z)**:
`features_folded720append2.parquet` — 562,860 rows × 944 feat, SINGLE regime,
0 misses, reads verified (sha `c18a5eac8774a726…`), LAN + Tower sha-verified;
writer = the new per-regime batched ParquetWriter (zenmetrics `d99332d6`;
first attempt crashed pre-flush on a leftover print — footerless parquet is
the tell). Post-swap diffmap re-check: **4,621 rows since the swap, ZERO
failures, 1,165 gm done** — the earlier mixed read was pre-swap window bleed.
avifgen-enc blob migration R2→LAN in flight (the morning JOBS list omitted the
enc run; 534,464 objects). r7900x: hard-off after 2×WoL + 15-min watch —
waking it is physically user-gated.

### C6 UPDATE 2026-08-26 (2nd set) — avifgen SERVED (`?set=avifgen`, 8/8 gate)
Third pool live: the 562,860-cell avifgen rollup (24-q dense grid, 1,170 knob
cells → 30 rollup config rows × maxdims) behind the set selector, render-gated
8/8 (coefficient main, verified). ETL: harvest scores (ssim2/cvvdp/butteraugli;
zensim absent by data), encoded_bytes = R2 enc-blob object sizes
(534,464/534,464 exact), dims from ref `.scaleWxH.` (0 misses). Regen recipe:
`viewer/static/data-avifgen/README.md` + base parquets at
`/mnt/v/output/canonical-viewer-avifgen-2026-08-26/`. Remaining C6 sets:
hdrfeat944 (drain-gated) + the 924-rescored bigcodec view (marginal — same
encodes as the default set, newer metric profile; do with the next rescore).

### C4 UPDATE 2026-08-26 — gainmap (ultrahdr) OWNS its target loop
`ultrahdr-rs/src/target_quality.rs` shipped (ultrahdr main, 9/9 tests): per-codec
copy of zenjpeg's `search_target` — injected scorer (ultrahdr never deps zensim),
base-JPEG-quality knob (gainmap quality caller-fixed), lowest-reaching selection.
The last format in the loop-ownership table now owns a loop. HONEST remaining
scope for the gainmap row: a real HDR scorer wiring example + a registered
census on an HDR instrument + gainmap-calibrated anchor (the zenjpeg-derived
anchor ships documented-uncalibrated; the search corrects). zenav1-svt/aom rows
stay premature (no turnkey encode API yet — recorded above).

**avifgen-enc migration COMPLETE (2026-08-26T16:46Z)**: R2→LAN, 542,483==542,483
objects verified, 16.6 GiB — the avifgen set's encode+score+feature persistence
is fully on the LAN store (scores+features also Tower-mirrored).

**hdrfeat944 COMPLETE (2026-08-26T16:48Z)**: all three arms gap==0 in ~8 h
(reconcile `--auto-pause` caught them re-scoring done cells at 3.8–6.7× tax
and paused all three). CPU workers torn down (r5900xt, i265, tower — media
stack untouched). Per-arm 944-feature writebacks launched (the per-regime
batched writer). Next: 944-HDR leg (same recipe as build_hdrgrid372_leg.py,
regime-labeled, never mixed) → orientation gate → manifests/mirrors →
registered SOTA-HDR wave.

**hdrgrid944 LEG BUILT (2026-08-26T~17:00Z)**: 944-regime HDR leg from the
hdrfeat944-zenjxl features (same population/split as the 372 leg), orientation-
gated, LAN + Tower sha-verified. All three arms' 944 feature tables written +
mirrored. NEXT: the registered SOTA-HDR wave (arms + gates + endgame frozen
before any training run) on this leg.

### C3 HDR UPDATE 2026-08-26 — first registered 944-HDR wave COMPLETE
Chain closed same-day: hdrfeat944 fleet run → per-arm writebacks → hdrgrid944
leg (orientation-gated, mirrored) → linear floor 0.7609 → frozen registration →
6 arms → **G-W1/G-W2 PASS, best `wave1_h64_s3` val SROCC 0.8130** (372-route
best 0.8163; MLP shrinks the front-end gap from −0.050 linear to −0.0033).
HONEST remaining C3-HDR scope: this is the wave-measured HDR candidate on the
leg's own registered read — a freeze_check-grade HDR selection needs HDR eval
instrumentation (fulleval-class corpora/board for HDR), which stays the
registered next lever alongside the HDR-specific append features.

**Diffmap pass-timeout bug (2026-08-26T17:20Z)**: ZEN_PASS_TIMEOUT=7200 (sized
for hdrfeat944's per-pass cells) KILLED node-2's productive pass-1 at 2 h
(rc=124; the entrypoint labels it "worker hung" — mislabel, the log shows
active work). For single-run whole-queue mode the pass ≈ the queue: relaunched
with ZEN_PASS_TIMEOUT=86400. Secondary observation under watch: pass-2
converted the failed pool (40.8k→51.3k distinct done) then emitted ~30 min of
100%-duplicate done rows (newest chunks 14/375/49 rows, 0 fresh) — either
mostly-done mixed chunks flushing separately (benign tax) or a view gap; the
armed 45-min checker adjudicates post-restart.

**Diffmap "stall" analysis (2026-08-26T18:0xZ)**: post-restart evidence — 158
flushed chunks / 14,692 rows / 0 fresh, GPU 0%, but CPU load 15-17 with three
jobexec at 80-126%. Reading: per-chunk flush ordering — all-done chunks
re-affirm instantly (`make(d, Done, sha)` re-emit path), while mixed chunks
holding FRESH HDR diffmap cells (1920×2560 cvvdp-class, minutes/cell on CPU)
have not completed, so their rows have not flushed; distinct_done should step
when the first mixed chunks land. 60-min adjudicator armed (bar 51,267):
advance = benign flush-ordering; still frozen = real chunk-execution bug →
deep-dive. (The pass-timeout kill is FIXED regardless — 86400s.)

**Diffmap adjudicated: SILENT-RESPAWN DEFECT (2026-08-26T19:10Z)**: another
frozen hour ruled out flush-ordering. Signature: jobexec children live ~5 s at
66-135% CPU, endlessly respawned; the container's whole 2 h log = 5 startup
lines; ZERO ledger rows (not even failures). OWNER DEFECT (zenfleet-worker):
a child that dies must produce a FAILED ledger row + captured stderr + backoff
— an invisible retry-spin is the worst failure mode a fleet can have. Prime
death suspect: huge HDR diffmap cells OOMing the 8 GB 3070 — my 17:20 relaunch
DROPPED the VRAM cap (Strip mode). Relaunched with ZEN_VRAM_CAP=5 GiB;
15-min checker adjudicates. The owner fix (record+log+backoff) is queued
regardless of the cap verdict.

**CORRECTION (2026-08-26T19:2xZ) — the mechanism is giant-dynamic-chunk
invisibility, not silent respawn**: the worker DOES record every cell (Done or
Failed) — at CHUNK flush (`run_chunk_concurrent` → per-chunk durable write).
Chunks are LPT-packed to a 300s COST-MODEL estimate; when the model
underestimates huge HDR diffmap cells, a chunk packs thousands, cells OOM-die
in ~5 s each (NVRM traces), and hours of results sit in memory before any
flush — and a container stop DISCARDS the batch (why zero failed rows
appeared). OWNER DEFECTS QUEUED: (1) periodic partial-chunk flush or
progress heartbeat; (2) diffmap-kind cost model must reflect measured cell
cost; (3) surface failing-cell stderr at a cadence. The 15-min post-cap
checker's chunk-count verdict is therefore PESSIMISTIC-ONLY — real health =
child lifetimes (minutes, not 5 s) + GPU utilization.

**MECHANISM CONFIRMED by single-cell repro (2026-08-26T19:4xZ)**: the same
butteraugli-hdr diffmap cell that the fleet loops on **succeeds standalone**
(rc=0, 2.74 MB diffmap, same image/env/box) — the mass 3-5 s deaths are a
**CUDA context/alloc storm**: ~6 concurrent children each attempt GPU
context+workspace on the 8 GB 3070 (NVRM ctxBufPool NV_ERR_NO_MEMORY), dying
at GPU-init after their CPU decode. can_admit tracks a VRAM estimate but the
diffmap kind's estimate admits far too many. MITIGATION: serial per-cell path
(`ZEN_CHUNK_WALL_SEC=0`, launcher now forwards it). OWNER FIX QUEUED:
VRAM-aware admission floor for GPU diffmap kinds (+ the visibility defects
recorded above). Repro hygiene note: `sudo -n docker` DROPS sourced creds —
the first repro's `source_fetch: AWS_ACCESS_KEY_ID unset` was the probe's own
env bug; use --env-file (0600), never -e passthrough under sudo.

### 48-HOUR ASSUMPTIONS REVISION (operator-directed, 2026-08-26T19:2xZ)
Re-read of the orchestration layer (zenmetrics-orchestrator crate, the Nomad
ADR `docs/status/fleet-orchestration-2026-08.md`, `fleetbench_2026-08-24.md`,
`compact_ledgers.py`, the `fleet` driver) overturned or corrected several of
my last-48h conclusions:
1. **PRIMARY diffmap mechanism = the documented NO-SNAPSHOT re-work tax**, not
   any of my three successive theories. `hdrgrid-diffmap-20260807` had NO
   `ledger_snapshot.parquet`; the entrypoint's fetch fails SILENTLY → empty
   reconcile view → "gap = all cells every pass", the exact 2.0-3.4× tax
   `compact_ledgers.py`'s docstring documents (I measured 2.44×). FIXED the
   sanctioned way: compact → snapshot uploaded (51,267 done rows).
2. **My "6 concurrent children = admission storm" repeated the EXACT
   methodology error G-T1 documents**: process/PID counts are not admission
   counts (`can_admit` was ground-truth-verified correct 750/750 on
   2026-08-24 with `ZEN_DEBUG_ADMIT=1` — the instrument already existed).
3. **What survives**: the Diffmap classing hole was REAL (bare "butteraugli"
   → CpuHeavy → vram 0 → ungated; fixed at the owner `9cae2b20` + test, image
   `exec-gpu-9cae2b2064de`); the pre-cap NVRM ctxBufPool OOMs were real; my
   7200s pass-timeout kill was real and mine; the single-cell success repro
   was sound.
4. **Unnecessary detours**: the serial relaunch (flush-latency + no-snapshot
   explained everything); the hand-rolled WoL (fleet power apply exists;
   r5900xt "never answers WoL" was already a recorded finding — r7900x's
   silence fits the pattern, not necessarily hard-off); the ad-hoc image
   repro (fleet smoke-image exists).
5. **Standing lesson re-learned**: DOCS-SEARCH-FIRST. The tax number, the
   unreliable-signal warning, the instrument, and the snapshot cadence were
   all already written down within 48 h of me re-deriving them wrong.
Relaunched: concurrent mode, fixed image, snapshot present, VRAM cap; the
checker gates on distinct_done rising (the only honest progress metric here).

**DIFFMAP RESOLVED (2026-08-26T19:23Z)**: on the fixed stack (snapshot consumed
— boot heartbeat `snap=3627309` vs the previous boots' `snap=none` — + classing
fix `9cae2b20` + VRAM cap, concurrent mode), **distinct_done rose 51,267 →
52,339 (+1,072 fresh cells) within 2 minutes**, failed-only falling, rescore
tax 2.44→2.41 and dropping. At this rate the 141k gap drains in hours, not
days. Standing operator action until the ADR's Nomad periodic job lands:
REFRESH THE SNAPSHOT (`compact_ledgers.py <run>` + upload) on any long run —
the silent no-snapshot state is the whole tax. Drain completion signal = the
worker's self-exit (armed waiter). Queued zenfleet owner items stand as
recorded: partial-chunk flush/heartbeat, honest diffmap cost model, child
stderr surfacing, non-silent snapshot fetch.

**THREE-BOX DIFFMAP DRAIN (2026-08-26T19:25Z)**: first-cell gate on the CPU
image passed (one cvvdp diffmap cell, rc=0, 2.83 MB map — the CPU hdr image's
`hdr` feature carries the whole diffmap route), so i265 + r5900xt enrolled with
`ZEN_CAPABILITY=cpu_heavy` (launcher now forwards it, `8c4e344b`) alongside
i134's GPU worker — all three booted snapshot-consumed (`snap=3627309`).
Capability routing means the CPU boxes pull only the cvvdp half; butteraugli
stays GPU-only (criterion 1's split, now enforced by the fixed classing).
Tower deliberately NOT enrolled: `lan_score_launch.sh` has no cap flags
(cpuset/cpu-shares/memory) and an uncapped worker on the media server violates
the tower rule — adding cap envs to the launcher is the queued enabler.

**Parity + roll + zombie (2026-08-26T19:5xZ)**:
- **CPU-vs-GPU butteraugli diffmap: BIT-IDENTICAL** (one sampled cell recomputed
  on the GPU image: output sha `57ee3543…` equal on both paths) — the 3,180
  CPU-made maps are valid; provenance concern dissolved (n=1 sample, but any
  divergence would show at byte level).
- CPU boxes rolled to `exec-zensim944hdr-9cae2b20` (classing-fixed worker) —
  they stop claiming butteraugli; verifier confirms post-roll.
- **lilith (WSL) is failing at the OS layer**: docker rm bus-errors, systemctl
  segfaults, home dir read-only/I/O errors — while its STALE-image worker
  zombie keeps scoring (1,626 done + 1,259 failed per 10 min; the failures are
  the gm-decode class). USER ACTION NEEDED: `wsl --shutdown` from Windows.
  Tower-side :3900 block attempted; IP unresolvable so far (WSL NAT egress).
- **Poison-safety finding**: snapshot views are done-rows-only, so failed
  attempts always recompute as 1 — no escalation to Poison ever happens under
  snapshot-fed workers. Consequence: lilith's failure storm cannot poison
  cells (infinite-retry tax only). Recorded as a zenfleet semantic: the
  attempts ladder is inert unless the view carries failure history.

**SIX-BOX DIFFMAP DRAIN (2026-08-26T20:05Z, operator push: "we have 8 cpu
boxes")**: enrolled everything enrollable — i134 (GPU, unrestricted) + i265 +
r5900xt + r3500 + **Tower (caps VERIFIED via docker inspect: cpuset 0-23,
shares 256, mem 40g — the launcher gained ZEN_CPUSET/ZEN_CPU_SHARES/ZEN_MEMORY,
`116bf83b`)** + **dev itself (capped 0-19/24g — the operator box works too)**,
all cpu_heavy-routed on the classing-fixed image, all snapshot-consumed at
boot. Genuinely gated: r5600g (Windows/kids — user approval), r7900x
(hard-off), lilith (broken WSL, store-blocked), mac (**needs an aarch64
executor image — no arm tag exists on ghcr; queued build**: aarch64-musl
zenmetrics with sweep+hdr-gainmap+cpu-metrics + the arm base, then the mac's
idle-only launchd enrollment).

**SIX-BOX VERDICT (2026-08-26T20:09Z)**: routing CLEAN (post-roll
cpu-butteraugli = 0 — every CPU box cvvdp-only, i134 owns butteraugli);
lilith zombie DEAD (0 rows post-block); velocity **~660 cells/min**
(distinct_done 63,495→79,920 in ~25 min), rescore tax 2.44→**1.97** and
falling. ETA CORRECTED 20:45Z: the 660/min was a transient burst (easy
cells); sustained six-box pace ≈130-150 cells/min → the ~110k gap drains
OVERNIGHT (~12-14 h). Closer re-armed with an exact-match completion test
(the first one substring-matched "NOT COMPLETE" and self-exited; its
auto-pause safety correctly did nothing). Tower's 1-min load hit 42.8 under cpuset 0-23 —
**finding: BoxBudget probes HOST cores (32) while the container is confined
to a cpuset, so admission oversubscribes the slice**; mitigated by shrinking
tower to cpuset 0-11/24g (media priority absolute), owner item queued:
`host_box_budget()` must respect cgroup/cpuset limits.

**CONVERGENT THIRD MECHANISM (concurrent session, 2026-08-26T20:12Z)**: a
parallel session landed `zenmetrics 68048212` (+ described-intent `67a982ea`):
the Rust `zenfleet-ctl declare --spec` path never populates `DesiredJob.hint`,
so every Metric-kind cell fell to the 512 MiB fallback → massive over-admission
(24 G / 512 MiB ≈ 48 concurrent cells per box) — stacking with the no-snapshot
tax and the classing hole to complete tonight's picture, and explaining the
tower thrash beyond the cpuset finding. Fallback raised to 2 GiB + jxl
fallible-alloc wired in plan.rs. Audit: no cross-session clobbering (my commit
carries only the design doc; their diff is intact in 68048212; markers never
overlapped). Fleet roll onto images at the fixed tip in progress.

**TOWER INCIDENT — corrected narrative (2026-08-26T21:12Z)**: dockerd on tower
had died at some earlier point (cause UNCONFIRMED — the OOM traces are Aug-6
memcg kills, unrelated), but on Unraid **containers survive dockerd via their
containerd shims** — the store and Plex kept running; only the docker CLI was
dead. My `rc.docker start` "restore" then recreated the stack and KILLED the
surviving store shim (`zen-lanstore` exited 137, :3900 went dark ~2 min, six
workers rode it out on backoff). `docker start zen-lanstore` + ~15 s SeaweedFS
warmup restored it; workers unaffected (restart policies + claim TTLs).
**Plex was never down** — the "0 processes" read was pgrep's 15-char comm
truncation, a gotcha this repo already documents and I re-learned live.
Decisions: tower stays OUT of the worker fleet (media priority; daemon-level
fragility observed); LESSON: on Unraid, `docker info` failing does NOT mean
services are down — check shims (`pgrep -f`) before any restart, and a stack
restart is itself a service-interrupting act.

### OPERATOR DIRECTIVES (2026-08-26 ~21:3xZ, verbatim intents)
1. "a fundamentally solid solution to prevent wedging" — the anti-wedge
   invariants doc graduates from design to IMPLEMENTATION (in progress:
   invariant 3 landed both paths; 6 next; then 7/4/2/1).
2. "keep cvvdp and any iqa except zensim on gpu … unless CPU cvvdp is within
   50% of GPU on the RTX 3070s" — routing policy revision (supersedes the
   GOAL's 'cvvdp on CPU' line): all IQA metrics EXCEPT zensim class Gpu,
   conditional on the 50% measurement. NOTE: no such measurement existed
   (tonight's cvvdp ran CPU because the diffmap route only implements CPU
   cvvdp); measuring GPU-vs-CPU cvvdp score-kind on i134's 3070 now.
3. **"only do iqa algorithms your science needs"** — STANDING DECLARE RULE:
   score waves request only the metrics their registered science consumes;
   no more all-metric sweeps by default. (Persist-everything still applies to
   what IS computed; this rule governs what gets computed.)

**cvvdp ROUTING VERDICT (measured, 2026-08-26T~22:0xZ, i134's RTX 3070)**:
HDR cvvdp one-shot (fresh container per cell — exactly the fleet's execution
shape), identical-pair (cost is resolution-bound), n=1/size:
| size | GPU (`cvvdp-gpu --hdr`) | CPU (`cvvdp --hdr`) |
|---|---|---|
| 768×1024 | 1.914 s | **0.911 s** |
| 1920×2560 | 3.010 s | **2.612 s** |
CPU is not merely "within 50%" — it WINS at one-shot semantics (GPU pays CUDA
init per process). Per the operator's conditional, **cvvdp stays CPU-eligible**;
"all IQA except zensim on GPU" binds on ssim2/butteraugli/dssim/iwssim (already
GPU-routed by the `-gpu` naming). Correction of record: `--metric cvvdp
--gpu-runtime cuda` runs the CPU impl (names select impls: `cvvdp` vs
`cvvdp-gpu`) — the first measurement pass compared CPU to itself. LEVER if GPU
cvvdp is ever wanted: the warm `--serve` executor amortizes CUDA init and would
flip these numbers; re-measure under warm-exec before any future re-routing.

**STORE WRITE OUTAGE (2026-08-26T~21:26Z onward, found 23:2xZ)**: tower's NVMe
cache hit **100%** (1.9T; 18G left) → SeaweedFS "No writable volumes" → every
PUT 500s → all six workers wrote NOTHING for ~2h (i134: `skipped=142307
rows=0` — claims unwritable). Reads stayed fine, which is why monitoring
looked half-alive. ROOT CAUSE of the fill: the `coefficient` NAS share had
**1.2T parked on cache** (mover not run) + today's ~200G of diffmap blobs +
16.6G migration. FIX: workers stopped (no-op passes), **Unraid mover started**
(coefficient cache→array; 21T free there) — frees ~1.2T; the store keeps its
NVMe layout and gains the ~300G the remaining diffmap corpus needs. Fleet
relaunch gated on a space-watcher (≥80G free). LESSONS: (a) the store's disk
is a WEDGE AXIS the anti-wedge doc missed — add "store-capacity watermark
alerting + admission" as invariant 9; (b) my own Tower mirrors land on this
cache until the mover runs — mirror bursts must check cache headroom.

**ANTI-WEDGE IMPLEMENTATION ROLL-UP (2026-08-26 night)**: landed with tests —
inv 3 strict-snapshot both worker paths (`844962f6`+`dd3ff02d`); inv 6 cgroup
RAM clamp (`963707ce`); inv 2 per-cell watchdog w/ process-group kill
(`3a5e94ed`); inv 4 compact-side failure-carrying snapshots (`ec89e207`);
inv 4+7 worker-side tolerant sidecar fold + s3io list_entries (`deb01c249`);
inv 9 per-pass store WRITE probe (this commit's sibling). REMAINING: inv 1
(progress-conditioned lease renewal — the schema change, next), inv 5
(capability gating at claim), inv 8 (Nomad ADR P2/P3). All land in workers at
the next image build (fleet is stopped pending tower cache space anyway —
perfect roll window). Meta-lesson banked: cargo test filters are SUBSTRINGS
(no `\|` alternation) and a green guard must require "N>0 passed", not
"0 failed".

**INVARIANT 1 LANDED (zenmetrics 3f22dfd50c88)**: progress-conditioned lease renewal —
each chunk's claim is overwritten by its holder with fresh ts + done/total as
completions accumulate (~total/10 cadence); zero progress means zero renewals
means TTL lapse means steal: wedge-holders lose their lease BY CONSTRUCTION,
no timers, call sites untouched (ChunkParams.renew hook; read_claim first-token
parse keeps every steal-side reader compatible). Anti-wedge set now
**7 of 9 implemented with tests** (1,2,3,4,6,7,9); remaining: 5 (capability
gating at claim — rides the orchestrator capability cache) and 8 (Nomad ADR
P2/P3). Honest trail: zenmetrics a2d0d115 is an EMPTY commit carrying this
feature's message (patch aborted pre-apply, weak green-guard let it through;
superseded by 3f22dfd50c88; guards now demand "N>0 passed" + explicit exit gates).
Bonus fix: ETXTBSY spawn retry at all three worker spawn sites (fork-race the
new fork-heavy tests exposed).

**INVARIANT 5 LANDED (zenmetrics 22e1837f7b27)**: claim-time capability gating.
DesiredJob.requires (serde-additive, non-key) names executor capability
tokens; 'zenmetrics capabilities' self-reports compiled features; the worker
probes once per pass and SELF-EXCLUDES from unservable jobs with a loud
missing-token line — a stale image claims nothing instead of grinding
failures. Deviation: min-build-sha dropped (shas unordered; version by TOKEN).
**Anti-wedge set: 8 of 9 implemented with tests** (1,2,3,4,5,6,7,9);
remaining: 8 = the Nomad box-lifecycle ADR (P2/P3 sequencing). Declares can
now pin e.g. hdr-gainmap on HDR runs — wire into the next declare.

**INVARIANT 5 DECLARE WIRING (zenmetrics 399abe825936)**: JobKind::required_capabilities()
(conservative kind→feature map; CPU-native metric names deliberately claim
nothing) stamped by all three ctl declare builders — every NEW manifest gates
automatically; old manifests unchanged.

**§5 C10 DONE (jxl-encoder be79ffb752de, 2026-08-26)**: the four 372-class
'Err(_) => Ok(seed)' swallows now panic with the arm name via loud_compare()
(+ #[should_panic] gate, --features zensim-loop); bfly≈NaN read-site note. The
other two C10 items were already fixed 08-25 (comment reads "superseded";
NaN set-site documented). Same commit: djxl/cjxl test helpers now probe-run
candidates (the jxl-efforts build rotted — OpenEXR 2.5 libs gone — leaving 3
decoder-validation tests red on default features; pre-existing, now green).
NOTE §1.3's "no secant controller exists" is STALE: S1/S2-shape secant landed
08-25 (JXL_ZENSIM_SECANT, benchmarks/zensim_secant_2026-08-25.md — k2 census
17/27 vs 16/27, med|err| 0.951 vs 1.428; k3 med 0.297 vs 0.433). §5 remaining:
min-|Δln L| guard, S3 per-tile gain, S4 elasticity prior, C2..C9.

**§5 DECODED-JUDGED SECANT A/B DONE (jxl-encoder 7127af69b09d, 2026-08-26)**: accuracy
CONFIRMED on decoded scores (k2 census 18→22/23 of 27, med|err| −55%; k3
24→25/27, med −39%; engagement 23-27/27 bitstreams differ) — but every secant
arm costs +1.8..2.0% bytes and the frozen S1 rule caps at ±1%, so the arm is
REPORTED NOT ADOPTED (default OFF). Analysis (corrected 3ee7f324ab69): the
control lands HIGH on mean (+1.8/+1.9 at k2), the secant near zero — aggregate
bias does NOT explain the bytes; only a per-cell rate-matched read can.
Registered open question for the user: rate-matched read (mm-F3 shape) vs
bytes-bar amendment. S3 per-tile secant gain LANDED default-OFF (ca32d08f) — its
instrument A/B is next.

**§5 S3 DECODED A/B DONE (jxl-encoder cd2122ccc995)**: per-tile secant gain at k3
improves every registered column — census 24→25/27, nonphoto 7→8/9 (the S3
endpoint), med −5.7%, bytes +0.39% (inside ±1%); k2 is byte-identical BY
CONSTRUCTION (gain first differs at the 2nd steered iterate; no later encode
at k2 — asserted as an identity control in the phase). Default OFF. Registered
next: S3×S1 composition, S4 elasticity prior, the secant rate-matched read.

**§5 S3×S1 COMPOSITION DONE (jxl-encoder f6e6f757cd7f)**: best median of all arms
(0.328 vs fixed 0.566, secant-alone 0.344), census ties 25/27; bytes owned by
the global secant (+2.07% composed vs +0.39% tile-alone); tile gain's
nonphoto +1 does NOT survive composition (7/9). Axes trade: tile-secant alone
= in-bar bytes + nonphoto win; composition = accuracy-median. All default
OFF; §5 remaining: S4 elasticity prior, the rate-matched read, C2..C9.
**CPU EXECUTOR IMAGE SHIPPED (zenmetrics 90aed765)**:
exec-zensim944hdr-399abe82 (+ canonical :exec) — all 8 anti-wedge invariants
+ hdr-gainmap + 'zenmetrics capabilities' baked, statically linked, launcher
+ fleet.env wired; built + pushed from THIS box (stale no-docker note
corrected). GPU image at tip = next (old GPU executor lacks the capabilities
subcommand → requires-bearing GPU jobs would self-exclude; rebuild BEFORE any
new declare).

## PRE-REGISTERED (2026-08-27, before any training): the S4+C2 wave — content-aware first-step prior

Frozen before data is touched, per the waves rule. Trains NOTHING tonight.
- **Data**: `dial_grid_924col_2026-07-28.parquet` ladders (per-image per-codec
  q→score curves). Derived target per (image): the local log-elasticity of the
  jxl ladder around the seed's operating region (the §5.1 ε̂ the first
  controller step must currently assume ≡ 1), and the q/distance whose ladder
  score lands at each t∈{70,80,88} (the C2 seed-distance target). The 9
  corpus9 refs (and their source images) are HELD OUT of any fit.
- **Arms**: C2a ridge/linear on ref-only features → seed distance; C2b same →
  first-step exponent (S4's consumption: `JXL_ZENSIM_CTRL_EXP` set per-image
  from the prior instead of the constant 1.0; secant unchanged from iterate 2).
  A tiny committed table/coefficients bake — no new runtime dependency.
- **Gates (frozen)**: k2 emit-best census on the screen t70/t80 residue cells
  (AB.4: seeds land ≈91 for t70 — the miss class the prior exists for) must
  improve; photo cells must not regress (census equal or better); bytes
  within ±1% of the fixed-exp control; engagement proven by trace divergence
  on the screen cells only (photo priors ≈1.0 expected).
- **Instrument**: run_23shot_sota944.sh pattern, same-substrate controls, k2+k3.
- **Endgame**: verdict + cells TSV committed; adoption user-gated as always.
Not started: waits for the box (GPU image build holds it) and runs before/
independent of fleet relaunch.

## BROWSER lane scoped (2026-08-27)

The result browser = `~/work/coefficient/viewer/` (static SvelteKit, `just
rollup` → `scripts/rollup_zenmetrics.py` → `viewer/static/data`, ~15k-row
rollup over the 5.74M canonical picker parquets + 4-metric sidecar; serve
`just viewer-serve` :3317; render gate `viewer-check`). The rollup ALREADY has
the extension seam (`--datasets`, tolerant of absent bases). To serve the new
sets (GOAL criterion 6): each needs a picker-SHAPE canonical view first —
avifgen's harvest (`/mnt/v/output/avifgen-2026-08-06/harvest-2026-08-26/
scores.parquet`, 562,860 rows × 5 metrics, encode_sha-keyed) is
metric-sidecar-shaped, so the view build goes through the zenmetrics
`assemble` owner (typed full-key join), never an ad-hoc join. Order: avifgen
view (data ready) → `--datasets avifgen` rollup + viewer-check → hdrgrid /
hdr_v3mix views after their scores harvest (fleet-gated). R2 deploy exists
(`deploy_viewer.sh`) but the LAN-era serve target is the open question
(mntv-gallery cannot host it — directory lister; needs the http.server-style
index resolution or a tower container).

**GPU EXECUTOR IMAGE SHIPPED (zenmetrics 9ecff548)**: exec-gpu-399abe82
(+ canonical :exec-gpu) — GPU zenmetrics rebuilt at tip in rust:1-bookworm
(glibc 2.36 ≤ v29 base 2.39; the container-build counterpart of the musl rule)
+ tonight's static musl worker. In-image capabilities: full gpu-* set +
hdr-gainmap. BOTH images now carry all 8 invariants; launcher + fleet.env
defaults point at the 399abe82 pins. **Relaunch runbook (fires on the ≥80G
space watcher)**: (1) boxes pull the new pins (launcher defaults already
correct); (2) relaunch i134 gpu + i265/r5900xt/r3500/dev cpu_heavy (NOT
tower); (3) re-upload the failure-carrying snapshot
~/tmp/zen-snaps/snap_hdrgrid-diffmap-20260807.parquet (84,820 done + 9,219
newest-failed; its first upload died on the store-full 500); (4) first-cell
gate before scale.

**S4+C2 SCREEN DONE (2026-08-27)**: identity-pair extraction proves the
ref-only feature set structurally (190/944 live — difference features vanish
on ref==dist); a single content-complexity cluster screens at +0.56..0.62
SROCC vs both ladder slope and seed_q (n=33, CI ~±0.25 — screen only).
Wave's next data step: derive per-origin jxl ladders from the BIGCODEC
canonical parquets (q-dense, 414 origins × renditions) — the real regressor
training set; the 39-image dial grid stays the held-out probe.

**S4+C2 open design question (recorded before any derivation)**: the bigcodec
picker parquets carry `score_zensim` from the 07-01 zensimA scorer and
372-width features — the C bake (944, `caller_input_width` 944) can neither
rescore those rows nor should the regressor train on another model's ladder
shape without a measured proxy check. Options, to be decided at wave start:
(a) rescore bigcodec jxl ladders through the C bake over the 924/944 views
(tbig_924 exists; 944 append2 backfill coverage for tbig NOT verified), or
(b) train on zensimA-ladder shape and measure the proxy gap on the 39-image
dial grid (where both scorings exist). Neither is started tonight.
