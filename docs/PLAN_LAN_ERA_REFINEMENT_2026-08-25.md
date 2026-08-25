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
