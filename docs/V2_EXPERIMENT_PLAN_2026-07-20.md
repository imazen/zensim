# Feature-v2 → optimal-model experiment ladder (2026-07-20)

Step-by-step plan for EVERY remaining experiment in the feature-v2 program:
datasets, evals, gates, and order. Companion to
`docs/OPTIMAL_MODEL_PLAN_2026-07-19.md` (fleet mechanics + methodology); this
file is the executable ladder. Every experiment is pre-registered (hypothesis +
kill band written BEFORE unblinding) per `docs/ITERATION_PROTOCOL.md`.

**Feature space:** append-only 720 = frozen v1-372 (f0..371) ++ v2-348
(f372..719). Deprecation = column masking (width-constant), never renumbering.
New features (E6/E7) append at **f720+**.

**Division of labor (2026-07-20):** the zenmetrics session owns the FLEET
backfill (its `.workongoing`: "launch 40x cx43 full backfill" — in flight).
This session owns the LOCAL backfill leg, the experiment ladder, and all
zensim-side training/eval work.

---

## Datasets

### ⇒ LIVE FILL STATUS (2026-07-22) — the fleet is filling T-big multi-codec NOW

A **3-worker zenfleet** (`dev` + `tower-unraid` + `node2`, bucket `zentrain`, job-pool
`s3://zentrain/jobs/_pool/runlist.tsv`, 54 tar-boxes) is grinding the bigcodec multi-codec
720-feature backfill "to 100%" (concurrent zenmetrics session; DO NOT disrupt). Codecs in
the pool: **zenjxl-lossy/VarDCT (24 boxes), zenjxl-modular (10), zenwebp (9), zenavif SDR
(8), zenpng (2), +1** — the codecs the native `v2_ab_extract` couldn't decode, now done via
the docker `jobexec` path (fetches R2 tars, decodes all codecs, emits 720). Source runs:
`canonical/2026-06-27/zenjpeg_lossy/encodes` + `mandfix*` per-codec runs. This is T-big.
- Note: **zenavif SDR is being feature-extracted** here — that is scoring EXISTING encodes,
  distinct from the halted avif-**HDR datagen** ([[feedback_zenavif_in_flux_no_datagen]] is
  about generating new avif-HDR, not extracting features on existing SDR encodes).
- Already at 720 (done earlier): safesyn-JPEG (111k), T-cid201, KADID, TID, and the full
  held-out set (CID22-49, AIC-3, AIC-4, KonJND-val, SDR25, CSIQ, LIVE).
- **CONSOLIDATED 2026-07-22:** all 11 local-leg datasets (149,195 rows) promoted to
  `/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22/` + R2
  (`s3://zentrain/ext720-canonical-2026-07-22/`) + Tower, with unified `_MANIFEST.json`
  (per-corpus sha256/build_commit/role/target semantics) — the E1 "convert + manifest +
  index" gate is CLOSED for the local legs. Pointer:
  `benchmarks/ext720_canonical_2026-07-22.pointer.md`; index: `~/work/zen/DATA_PROVENANCE.md`.
  Fleet write-backs (T-big, T-safe, bigcodec valdigits — asked of zenmetrics 2026-07-22)
  get indexed there when they land; bigcodec **testdigits pair list is not yet materialized**
  (needs the 2026-07-02 dedup pipeline rerun on {7,9}-digit origins).
- The tower's enrollment + the 2.33× extractor perf win (ref-pyramid reuse, cached moments,
  SIMD pools) landed 2026-07-20…22.

### Training (T)

| id | corpus | pairs | codecs / distortions | target | 720 status |
|---|---|--:|---|---|---|
| T-big | bigcodec_hqdedup (canonical-picker-2026-07-01-zensimA) | 2,322,579 | REAL zenjpeg/webp/png/jxl ±lossless | ssim2 [0,1] | **TRAINING CORPUS BUILT 2026-07-23 → `tbig_720_TRAIN_NN.parquet` (2,133,071 rows @720, ssim2 target).** Fleet extracted the WHOLE canonical corpus (5,742,660 rows @720 = 100.0000% of cells). The target join (bigcodec `human_score`=ssim2, 2,307 refs) onto the 720 features: exact-hash hit only 65.2% (decoder drift), **NN-variant join (`join_eval_nn_720.py`, thresh 0.03) recovered to 91.8%** — 189,508 too-far dropped+flagged (never fabricated), 0 no-ref, nn_dist median 1.7e-08. v2 block `f372..f719` verified 100% nonzero. Durability triple: local + `s3://zentrain/tbig-720-2026-07-22/` + Tower (`benchmarks/tbig_720_train_nn_2026-07-23.pointer.md`, `_MANIFEST_TRAIN_NN.json`). Supersedes the 65.2% exact `tbig_720_TRAIN.parquet` (→ `_superseded/`). Note: hqdedup's knob-no-op-dedup selection not reproduced (no variant key) — this is the full pre-dedup traindigits; dedup-equivalent selection is an E2 follow-up if wanted. |
| T-safe | safesyn full | 196,086 (−avif ≈ ~160k) | mozjpeg/zenjpeg/zenjxl/zenwebp (+avif EXCLUDED: zenavif-in-flux + mm6 precedent) | gpu_ssim2/100 | fleet or local-multicodec; bitstream decode (PNG cache deleted; decoder-drift caveat — decode ALL through one decoder) |
| T-cid201 | cid22_train 201-ref subset | 17,611 | mixed real codecs | ssim2-anchored (NOT MCOS — legal) | **DONE** (`ext_cid22_train201`, backfill 2026-07-20; verified 0-overlap w/ 49-ref holdout) |
| T-kadid / T-tid | KADID-10k / TID2013 | 10,125 / 3,000 | analytic (guard weight only) | DMOS/MOS | **DONE** (`ext_kadid/ext_tid`, 2026-07-19) |
| T-konjnd | KonJND-1k train, JPEG half | ~10k | JPEG (BPG half: no decoder — documented gap) | **SKIPPED** (backfill 2026-07-20): no local parquet carries a ref/dist path + the 20-sample selection discriminator; target is a CVVDP+IW-SSIM blend (zenmetrics territory). Fully-local ssim2-anchored alt `konjnd_full_scored.csv` (50,400 rows) exists if wanted — NOT auto-substituted. |
| T-negrich (opt) | kadis_negrich | subset | analytic negatives | **SKIPPED** — own manifest says "selection rule unrecorded" (un-recoverable, not an R2 pull) |
| T-hfnl (opt) | hf_nearlossless train | 900 | JXL near-lossless | **SKIPPED** — pixels were in a wiped `/tmp` scratchpath; 0 rows have a persisted dist bitstream; `.jxl` refs not decodable by the extractor anyway |

### Held-out (H — NEVER trained; the verdict set)

| id | corpus | pairs | axis | 720 status |
|---|---|--:|---|---|
| H-cid22 | CID22-49 val (gold MCOS) | 4,292 | compression, human | **DONE** |
| H-aic3 | AIC-3 CTC (JND) | 600 | compression, human | **DONE** |
| H-aic4 | AIC-4 sample (JND) | 300 | compression, human | **DONE** (`ext_aic4`, 2026-07-20) |
| H-konjnd | KonJND-1k val, JPEG half | 504 | near-threshold (raw mean-PJND target) | **DONE** (`ext_konjnd_jpeg_val`, 2026-07-20) |
| H-sdr25 | JPEG-AI-SDR25 | **50** (NOT 95k) | HQ zone, JPEG-AI only | **DONE** (`ext_sdr25`, 2026-07-20). CORRECTION: the "95k" were triplet-comparison *responses*, not pairs — they collapse via ordered-probit into 50 scoreable JPEG-AI pairs (byte-identical to the pre-existing `sdr25_eval_pairs.tsv`). A HQ-zone gate at n=50 is thin — treat as directional, not a hard gate. |
| H-csiq / H-live | CSIQ / LIVE-R2 | 866 / 779 | general-FR (context, not gate) | **DONE** |
| H-nonphoto | non-photo ssim2 gate (bake_verdict G-NP) | 10,000 | non-photo | **CLOSED 2026-07-23 — 9,244/10,000 (92.44%), coverage-complete.** After the zensim-720 T-big backfill DRAINED across all 54 fleet pool runs, `scripts/v_next/merge_fleet_720.py` assembled every run into a per-run 720 shard (`/mnt/v/zen/zensim-training/720shards/`, ~5.7M rows) across dev+node+tower, then `scripts/v_next/join_eval_nn_shards_720.py` drift-robust NN-joined (thresh 0.03, output = fleet variant's own self-consistent f0..f719 + eval ssim2 `human_score` + nn_dist). **`no-origin=0` (128/128 origins / 2.79M variants) PROVES the 7.6% gap is drift/absent-cells, not coverage** — too-far tail median 0.074, only 26% mild-drift; 92.4% is the honest ceiling, threshold stays 0.03. Never fabricated (too-far dropped+flagged). Output `ext_nonphoto_720_nn_full.parquet` + `_MANIFEST.json` in `/mnt/v/output/zensim/v2-eval-720-2026-07-23/`. Supersedes the 2026-07-22 partial (26%). |
| H-imazen26 | real-codec ssim2 gate (bake_verdict G-IM26) | 10,025 | real-codec, non-photo content | **CLOSED 2026-07-23 — 9,265/10,025 (92.42%), coverage-complete**, same method/run as H-nonphoto (`no-origin=0`, 74/74 origins / 1.63M variants). Same drift ceiling. Output `ext_imazen26_720_nn_full.parquet`. Supersedes the 2026-07-22 partial (22.9%). |
| eval: dial grid | G-DIAL (mono/reach) | 4,817 cells | JPEG/WebP/JXL/AVIF q-sweep | **BACKFILLED 2026-07-22 — 4817/4817 (100%) matched.** Re-encoded the exact `build_qsweep_expanded.py` grid via `zenmetrics sweep` (CPU encode+decode-back only; GPU zensim SCORING is now fully disabled so the original `--metric zensim-gpu` recipe can't literally rerun) + `zensim/examples/v2_ab_extract` (CPU 720) + joined back to the original identity. 536/4817 rows flagged as cross-backend (GPU-original vs CPU-backfill) drift, concentrated exactly on the already-documented w11 webp/jpeg + JXL near-lossless contaminations (cross-validates the pipeline). Two zenjxl encode-limitation findings surfaced (distance=0 now rejected; odd-dim images decode back +1px) — reported, not fixed (out of scope). Tool: `scripts/v_next/backfill_dial_grid_720.py`. Details: `benchmarks/v2_eval_720_backfill_2026-07-22.md`. |
| eval: corruption grid | corruption gate | — | corruptions | **BACKFILLED 2026-07-22 — 2016/2016 (100%) matched.** Pixels already existed (`/mnt/v/output/zensim/corruption_gate/`) — pure re-extraction via `zensim/examples/v2_ab_extract`, no re-encode. Near-ULP verify vs the original (both CPU): L2 median 2.0e-8, 0 flagged. Tool: `scripts/v_next/backfill_corruption_grid_720.py`. |

**Bans in force:** CID22-49 human MOS never trains. AIC-3 raw triplets (420k)
banned pending ref-disjointness vs the CTC 10-ref holdout. AIC-4 holdout-only.

### ⇒ 720 GAP AUDIT (2026-07-22) — 2/4 CLOSED, 2/4 PARTIAL same day; history below for context

**UPDATE (2026-07-23): ALL 4 gates now CLOSED.** dial_grid (4817/4817) +
corruption_grid (2016/2016) were closed 2026-07-22 by re-encode/re-extract,
100% matched and verified. **nonphoto + imazen26 closed 2026-07-23 at the
drift ceiling: 9,244/10,000 (92.44%) + 9,265/10,025 (92.42%).** The zensim-720
T-big backfill DRAINED across all 54 fleet pool runs; `merge_fleet_720.py`
assembled the complete 54-shard 720 index (~5.7M rows) across dev+node+tower,
and `join_eval_nn_shards_720.py` drift-robust NN-joined the gates against it.
Critically, `no-origin=0` for both — every eval origin is present (128/128,
74/74 origins) — so the remaining 7.6% is the 0.03 drift threshold / genuinely
-absent+contaminated cells, NOT a coverage gap. This SUPERSEDES the 2026-07-22
partial (26%/22.9%), which was limited by a partial blob set, not by drift. The
full-index run proves further fleet extraction would not raise the match rate.
Method, coverage proof, and too-far distribution:
`/mnt/v/output/zensim/v2-eval-720-2026-07-23/_MANIFEST.json`. History below:
`benchmarks/v2_eval_720_backfill_2026-07-22.md` +
`/mnt/v/output/zensim/v2-eval-720-2026-07-22/_MANIFEST.json`. Tools:
`scripts/v_next/backfill_dial_grid_720.py`,
`scripts/v_next/backfill_corruption_grid_720.py`,
`scripts/v_next/fleet_blob_fetch_720.py`. Two notable corrections to the
plan as originally written below: (1) `join_eval_720.py` (the tool this plan
names) assumed the fleet exposed a clean `(ref, f0..f719)` parquet —it does
not; the fleet's `ledger/*.parquet` is job-tracking metadata only (q=-1,
knob_tuple_json="scorefile" for score_file jobs) and blobs are per-pool
JSONL batches of ~5-12 variant records, not one-blob-one-JSON-object, so the
actual join needed a 2-stage ledger-scan + blob-fetch pipeline
(`fleet_blob_fetch_720.py`), not a direct parquet join. (2) the original
GPU-based dial-grid recipe (`--metric zensim-gpu`) can no longer run at all —
GPU zensim SCORING was fully disabled 2026-07-19 — so "regenerate" meant a
CPU-only re-encode + re-extract, not a literal rerun.

**Original audit (2026-07-22, pre-backfill) — kept for the reasoning trail:**

**Answer: NO — the eval-side gaps are NOT queued. The fleet pool is 100% bigcodec
TRAINING encodes** (54 tar-boxes: zenjxl-lossy/modular, zenwebp, zenavif-SDR, zenpng);
zero eval sets in it. Full status:

- **DONE @720 (11 corpora):** train — safesyn-JPEG, cid201, KADID, TID; held-out —
  CID22-49, AIC-3, AIC-4, KonJND-val, SDR25, CSIQ, LIVE.
- **IN PROGRESS (fleet, live):** T-big multi-codec (bigcodec) — the training mass.
- **LACKS 720 AND UNQUEUED — 4 validation instruments** (needed to run the panel on the
  720 model): **nonphoto** (G-NP) + **imazen26** (G-IM26) — extractable, pairs need
  reconstruction, ~20k pairs total, local job; **dial grid** (G-DIAL) + **corruption
  grid** — pixels not persisted, need re-encode+re-extract (or accept 372-only on those
  two gates).
- **Optional/undecided:** safesyn multi-codec (webp/jxl, ~50k) — bigcodec already covers
  real multi-codec, so likely redundant; not queued.

**Close-the-gap plan (REVISED 2026-07-22 after tracing the cells):**

**nonphoto + imazen26 are NOT a separate corpus — they are content-filtered subsets of
the bigcodec (canonical-picker) validate/test cells the fleet is filling NOW.** Verified:
their `ref_basename` (`o_NNNN.png.scaleWxH`) is identical to canonical-picker
`test.parquet`/`validate.parquet` `ref_filename` (100% overlap on the sampled refs). So
their 720 features are a BYPRODUCT of the running fleet — no re-extraction needed. The eval
parquets carry only `(ref_basename, human_score=ssim2, f0..f371)`, and the fleet 720 output
(`s3://zentrain/jobs/bf-*/blobs/`) carries `f0..f719` for the same cells, so the backfill is
a **feature-space JOIN**: match each eval row to the fleet row on `ref_basename` + exact
`f0..f371` (the 372 block uniquely fingerprints the cell), append `f372..f719`. Tool:
`scripts/v_next/join_eval_720.py` (written 2026-07-22). Runs when the fleet completes
(currently grinding); reports match-rate and flags any unmatched row (never fabricates).
**[Superseded — see UPDATE above: the fleet has no clean `(ref, f0..f719)` parquet to join
against; the actual tool ended up being `fleet_blob_fetch_720.py`'s ledger-scan +
JSONL-blob-fetch + inline-fingerprint-match pipeline.]**

**dial + corruption grids are the ONLY genuinely-new work.** They are a custom multi-codec
q-sweep (q0 + step-1 q90→100 + fractional near-lossless + JND zone + jxl-in-butter-distance
over JPEG/WebP/JXL/AVIF) — DIFFERENT q points than bigcodec, so NOT fleet cells, and pixels
were never persisted (parquet is feature-vectors only). Backfilling 720 requires
**re-encoding the exact documented sweep** on the source refs (`image_id`) + extracting 720
— a sweep job (encode+score+extract), the fleet's `Dockerfile.sweep` domain. Options:
(a) add the two grid sweeps to the fleet (zenmetrics session owns the sweep image), or
(b) run an independent sweep leg. Until decided, G-DIAL/corruption run on the 372 block only.
**[Superseded — see UPDATE above: ran as an independent local sweep leg via
`zenmetrics sweep` (encode/decode-back only) + `zensim/examples/v2_ab_extract`
(CPU feature extraction), not added to the fleet.]**

**Net:** "backfill all with 720" = one JOIN (nonphoto+imazen26, ~free, fleet byproduct) +
one re-encode SWEEP (the two grids, the real remaining compute).

### Eval instruments

- **forward+panel** (`predict_features_with_bake` + `panel`): width-agnostic — works for 720 today (A/B-proven).
- **bake_verdict @720**: registry is 372-only → extend to ext_ parquets + width autodetect (infrastructure item I-1; owner = bake_verdict).
- **steer-mass / family-sensitivity** (`v2_steer_by_family.py`, `v2_combined_steer_mass.py`): work today.
- **coherence** (`diffmap_block_coherence`): 720 scalar-side works; runtime M3 needs E9.
- **dial panel @720**: blocked on eval-grid re-extraction (I-2).
- **RD probe** (jxl/zenjpeg worktrees): needs a 720 bake behind a custom profile — E10.

---

## The ladder (each step pre-registered before unblinding)

**E0 — 720 pipeline smoke.** Local jobexec cell + fleet first-chunk artifact
check: feature rows length **720** or stop-the-line. Cross-check jobexec-vs-
`v2_ab_extract` on ~10 shared pairs (≤5e-4 rel) — decoder/pipeline parity.
*(Fleet side owned by zenmetrics session; verify before trusting their ledger.)*

**E1 — backfill.** Fleet: T-big, T-safe (zenmetrics session). Local: H-aic4,
H/T-konjnd-JPEG, H-sdr25, T-cid201 + investigations (this session's agent).
Gate: row counts = pair counts (skip rate ≤0.1%); `_MANIFEST.json` with
build_commit + image digest per output (ML-discipline §2). Convert all to
parquet; index in DATA_PROVENANCE.

**E2 — ceiling model (the production append-only decision).** Train 720 MLP on
{T-big + T-safe + T-cid201 + guards}, target ssim2, `withinref,both` +
mse-weight, seeds {1,7,13}, val groups = ALL of H. Twin v1-372 arm, identical
argv (cap 372). Gates: best-epoch > 0 (instrument trains past epoch 0 — the
lab-recipe failure mode is gone); seed σ(CID22) < 0.02; then the pre-registered
append-only bands (WIN = ext ≥ v1 − 0.010 mean across compression holdouts, no
corpus ≤ −0.030). This SETTLES the seed-noisy lab CID22/LIVE verdict.

**E3 — masked variants at scale.** ext-luma (chroma transducers masked) and
ext-lumacoh (+ v1-nonspat f156-371 masked) under E2's recipe/seeds. Decides at
production scale: chroma-transducer deprecation + v1-nonspat deprecation
(currently supported by lab evidence: lumacoh = 100% spatializable at ~0 cost).

**E4 — feature-family LOO at the optimum.** In the E2/E3 winner config: mask
each v2 family (width-constant), ×3 seeds, full panel per holdout. Verdict per
family: load-bearing (|Δ| > seed-σ, hurts when removed) / redundant (drop-mask)
/ neutral. This replaces every lab-scale family claim (incl. blockiness-keep,
GMS-graduates) with marginal-at-the-optimum evidence.

**E5 — per-feature sensitivity (cheap cross-check).** `s_k` central-difference
importance on the winner (no retrain); grouped by family/scale/channel.
Agreement with E4 → confidence; disagreement → investigate before freezing.

**E6 (conditional) — transducer k-refit.** ONLY if the Y-transducer family
survives E4. Append k∈{2,8} variants at **f720+** (append-only), subset-screen
on ~100k pairs before any full extraction. Kill: no holdout gains > seed-σ.

**E7 (conditional) — GMS-deviation (real GMSD).** Append std-pooled GMS at
f720+ (reuses the materialized GMS map; near-free at extraction). Same subset-
screen protocol. (The one remaining validated utility candidate — GMS is
mean-pooled today and underused at ~2% steering mass.)

**E8 — feature-set freeze.** Survivor list + mask list → docs + memory +
`bake_verdict` registry (I-1). The production feature definition for the next
zensim generation; code change to `feature_v2.rs` emission (drop dead compute)
only AFTER this freeze ("when research is done we could change that").

**E9 — diffmap completion for survivors** (task #48). Wire
`compute_v2_diffmap_channel_scale` (landed `ce45a1ff`) into streaming
`compute_with_diffmap`; add per-pixel gradient maps ONLY for load-bearing
excluded families (masked/iw = `w·v/Σw`; dev = deviation term; soft-peak =
product rule; fragility = correct 0). Block-pool identity test per family;
measure DEPLOYED M3 vs the 1.0 ceiling (proxy-validated: v1's 0.54 ≈ its 53%).

**E10 — five-gate scorecard + ship decision.** G-RANK (bake_verdict@720),
G-DIAL (needs I-2), G-STEER (E9's M3), G-RD (jxl+zenjpeg probe with independent
judges), G-TARGET. Winner vs shipped B; swap is USER-GATED.

### Infrastructure items
- **I-1**: bake_verdict 720-corpus support (before E2's verdicts).
- **I-2**: eval-grid (dial+corruption) 720 re-extraction — investigate source
  pixels; blocks G-DIAL only.
- **I-3**: `score-pairs --feature-output` still emits 372 (WithIw) — latent
  footgun, fix or loudly document in zenmetrics.

### Cost/time (workstation-derived; smoke-verify before trusting)
T-big ≈ 109 CPU-h ⇒ ~7 h on 40×cx43 ≫ margin (in flight); T-safe ≈ 9 CPU-h;
local legs ≈ 1-2 h wall total. E2/E3 training ≈ 30-60 min/arm ×3 seeds ×~4 arms
(run-heavy, serialized; Hetzner trainer fan-out if it drags). E4 ≈ 11 families
×3 seeds — the big training bill; batch on Hetzner per feedback_hetzner_cpu.

### Standing rules that bind every step
Pre-register before unblinding; multi-seed for any claim; CID22-49 sacred;
results → `benchmarks/*.md` + sidecars same-day; no /tmp; run-heavy for
everything heavy; jj push-verify (`main@{u}` ancestor check) before "done".
