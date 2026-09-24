# Rev4 feature bank and the in-sample potential protocol (2026-09-23)

Design lane `featbank`. Design only: no training, no extraction, **no human label read**. Measurements
read feature values or schemas only. Commands:
[`../benchmarks/rev4_featbank_WORKLOG.md`](../benchmarks/rev4_featbank_WORKLOG.md). This plan carries four
deliverables: the feature ledger (§1), the f32 feature bank (§2), the in-sample potential protocol (§3), the
leave-one-dataset-out rotation (§3b), and three work orders (§4). They sit under
[REV4_EXPERIMENTS](REV4_EXPERIMENTS_2026-09-23.md), whose split and owner rules bind.

**User direction (2026-09-23).** In-sample testing can show the *potential* of a feature set if the right
steps are taken: use lasso, linear projection and possibly an MLP. Decide which features to add or fix.
Optimise the extractor so it can run over the training data, and store f32 features for lasso and subset
learning with good seed selection. Addendum: rotate leave-one-dataset-out, as in the V0_2 era.

**The one idea behind the design.** Most past feature verdicts came from one of two instruments:
- no-refit ablation (`bake_contrib`, the LOO records);
- MLP screens selected on the joint-core-v2 val aggregate (block5, gmsd). Its rows are dominated by
  SSIMULACRA2-target legs: safesyn has 38,758 of the 45,165 dev rows (block5 record, corpus line).

Neither can tell "useless" apart from "unrewarded". Block5 measured exactly that gap. Its block surfaces
gained +0.01 to +0.02 on the held-out dev legs, beating their own permuted controls (W_b5gate t = 4.19), yet
were invisible to the val objective that selected checkpoints (`zensim--transplant`
`benchmarks/block5_2026-09-22.md`, verdict 3). The potential protocol measures each family against **human**
labels directly, with a fitted model and reference-grouped CV. It is still only a ceiling, never a
performance claim.

---

## 1. Feature ledger

**Costs** are the records' own numbers:
- **[FC]** `benchmarks/feature_cost_frontier_2026-08-31.md` lines 286-296: zenbench, 3 rounds, ms per pair at
  576², 1152² and 2304², 1 thread. The record says to read its 8T/16T columns for shape only.
- **[SS]** `benchmarks/scale_selective_944_2026-09-13.md`: warm `BakeScorer::compute`, 1024², 15 interleaved
  samples.

"LOO" means refit-free zero-ablation (a lower bound on what a refit must recover; FC lines 157-164). E2 LOO rows
are pre-Rev3 era-1 720 tables; [SS] rows are Rev3.

| family (IDs) | measures | cost | evidence for | evidence against | status |
|---|---|---|---|---|---|
| `basic` f0-155 | per-channel/scale SSIM, edge-artefact, detail-loss, MSE; mean + L4 | fold156 6.50/30.8/128.8 ms [FC] | B ablation −0.706 CID22 (FC §2.1); E2 LOO Σ −2.103 (`e2_optimal_model_720_vs_372_2026-07-23.md` §LOO) | — | **keep** |
| `peaks` f156-227 | soft-peak (tail) pools of the same maps | free within the basic walk (fold228 7.30/28.6/123.5 ms [FC]) | B ablation −0.311 KonJND; carries 97% of the free set's CID22 gain (`free_features_2026-09-01.md` 2026-09-05 note) | peak sub-signal hurts FR in the E2 LOO (+0.227) | **keep** |
| `masked`, `iw` f228-371 | activity-weighted pools of the SSIM/ART/DET/MSE maps | +2.40/+9.80/+44.20 ms, +33-36% [FC :340] | B ablation masked+IW −0.399 CID22 / −0.525 KonJND (FC §2.1) | **F4**: SSIM `d` unbounded, `f313` = 5.8e6 (`FEATURE_DEFECTS_AUDIT_2026-09-05.md` §2.1); no spatial refinement support, so UNSUPPORTED in steering (`feature_ceiling_2026-09-13.md` spatial) | **fix** F4 (`v1ssimcap`, Proposed) |
| `v2` f372-719 (29 signals × 3 ch × 4 scales) | SSIM dev, HF gain/loss, weighted pools, PJND transducers, GMS, BLOCKINESS, RINGING, BANDING, edge width | per-scale dispatch landed: 619-layout 20.7 vs 51.7 ms ST [SS] | corruption-class MAE 8.9 (full944) vs 17.6 (basic228) [SS]; ringing LOO −0.608, gms −1.024 | 720 vs 372 flat on compression (E2 record); v2 spatial repairs fail 18/18 at 619 [SS] | **keep**; sub-rows below |
| v2 `BLOCKINESS` (local 25; f397, …) | Wang-Bovik oriented step excess on a fixed 8-px lattice | sparse scalar pass (`feature_v2.rs:4142`) | LOO −0.172 (CSIQ/TID) | inert: block5 `G_blk ≡ G` to 4 d.p., 5 seeds. Form `(a−b)⁺/(a+b+1e-3)` (`feature_v2.rs:742`, `C_BLOCK = 1e-3` `:215`) is **contrast-invariant**: it saturates for strong steps and peaks for faint steps in flat areas. The 8-px lattice is used at *every* scale, so it is grid-aligned only at scale 0 | **fix** → candidate C1 |
| v2 `RINGING` (26) | near-edge flat-region error | in the gradient pass | LOO −0.608 | one saturating product, no magnitude basis | **keep**; basis extension in C2 |
| v2 `BANDING` (27) | CAMBI-like contour detector | in the gradient pass | — | LOO Σ **+0.401**, removal lifts CSIQ/LIVE/TID (E2); superseded by BANDVIS | **drop** from new read sets (the ID stays) |
| v2 `PJND_FRAGILITY` (21) | reference-only fragility | — | — | **F15**: constant 1.0 on a v1-only walk | **fix** or exclude explicitly (slice) |
| v2 `PJND_TRANSDUCER_LOW/HIGH_K` (23-24) | masking transducer bank | `transducer_bank` toggle | — | LOO +0.072 (mildly harmful, KonJND) | **keep** in the bank; potential decides |
| `append` f720-923 | luminance-bin error, MSCN, contrast gain/loss, texture, GMS/ART/DET deviation, GLOBAL_* moments | v2 + append marginal +7.4/+38.9/+159.3 ms [FC :346] | class-C 24 slots cost ~1.3-1.5% (`free_features_classC_2026-09-04.md`) | **F5**: raw-moment slots fail route parity, 9.12% of cells over 2e-5, worst 3.63e-3 (catastrophic cancellation). Deviation pooling on the 228 surface: −0.0039, 0/5 seeds (`zensim--gmsd` `benchmarks/gmsd_2026-09-22.md`) | **fix** F5 before any bake reads those slots; keep the rest |
| `append2` BANDVIS f924-943 | curvature band-pass banding visibility + LUMA_MEAN_REF + HDR highlight bins | +1.79% (`append2_bandvis_gates_2026-07-27.md`) | LOO Σ −0.069 PASS (`bandvis_loo_944_2026-07-28.md`); LOSS s3 SROCC −0.447 on LIVE-YT-Banding | GAIN weak; dst-activity toggle falsified (`bandvis_dst_activity_2026-08-02.md`) | **keep**, dst-activity OFF |
| `csfw` f944-955 | luminance-CSF-weighted GLOBAL_* | ≤ +1.39% | cross-route SROCC 0.8497 → 0.9520 (`csf_tier1_gates_2026-07-28.md`) | LOO Σ **+0.061**, FAIL (`csfw_g6_loo_2026-07-29.md`) | **drop** from new read sets; the potential run re-tests it (cheap; the LOO was no-refit) |
| `dvifm` f956-985 | parametric visibility-weighted 5×5 block error + 5 log-contrast bins, 5 levels, luma | i16 serving kernel 0.94× of baseline at 1024² 1T (`LANE_GEOMETRY_DONE.md`) | JPEG/JP2K per-type in-sample rank 0.921/0.944 (`dvifm_faithful_2026-09-22.md` probes); edge-discount +0.055 KADID-nt, +0.037 CID22-A | standalone below fast-ssim2 on real labels (`dvifm_verdict_2026-09-20.md`); in-sample ceiling 0.792 on TID | **keep** (the DVIFM-ish components) |

### 1.1 Candidate additions (IDs from 986, append-only; next free = 986, `feature_defs.rs:2172`)

Each candidate answers a measured failure or a peer's measured strength. The zenpapers gap audit independently lists two of them:
W7 "blockiness cannot see phase-shifted (re-cropped) lattices" and W6 pooling gaps (lines 248-272). All candidates are Y **and** X/B
unless stated, 4 scales, and difference-form (0 on identity).

- **C1 — grid-phase blocking with a magnitude basis.** *Failure:* E1b. C over-rates JPEG in 484 of 510 wrong
  CID22-A pairs, while every zensim model under-rates JPEG on AIC-3 (C 85/94) and the AIC-4 crop (162/162)
  (`rev4_e1b_crosscodec_2026-09-23.md`). That is the flat-response signature. *Mechanistic hypothesis, to test
  and not assume:* the contrast-invariant ratio above cannot express "invisible faint grid" vs "visible
  strong grid". *Definition:*
  - per scale `s`, the grid period `P_s = 8/2^s` for luma (8, 4, 2; not defined at scale 3) and `16/2^s` for
    the 4:2:0 chroma lattice on X/B;
  - the phase is estimated from the reference-vs-distorted step-energy profile over the P phases (argmax;
    ties go to phase 0), so crops and shifts are handled;
  - boundary step excess `e = |Δd| − |Δs|` is compared **on grid vs off grid** (phase contrast removes blur
    and noise that raise steps everywhere);
  - `e` is normalised by the local activity of the two neighbouring blocks (divisive masking, the existing
    `C_ACTIVITY` saturator), in absolute Y units, not relative to the step;
  - output = **6 fixed log-magnitude bins** (triangular, as in DVIFM's 5 bins) + on-grid mean + on/off ratio
    = 8 signals per channel per scale.

  The bins give lasso and the linear projection a piecewise basis to *learn* the response shape. That is the
  difference from block5 and zgeom: those changed the *pooling* (5×5 lattice, p90, gates, softmax) of
  grid-agnostic or contrast-invariant statistics, and were selected on that val aggregate. C1
  changes the *per-location transducer* and the *grid alignment*, and is judged on human sets. Cost goal:
  ≤ 5% of fold944_full at 1024² ST (a sparse pass: only lattice rows and columns plus 1 phase scan).
- **C2 — ringing magnitude basis.** The same 6 log-magnitude bins over the existing RINGING per-pixel term.
  *Failure:* the same JPEG shape question (ringing is JPEG's second artefact). It reuses the gradient pass,
  so the cost goal is ≤ 2%.
- **C3 — tail pooling from mergeable histograms.** Per plane, a fixed 32-bin log histogram of the existing
  per-pixel SSIM/ART/DET/MSE maps, from which p95, p99 and max are emitted.
  - *Evidence:* ungated block-peak +0.0012, 4/5 seeds, codec leg +0.0139 5/5 (`LANE_ZGEOM_DONE.md`); block5
    dev +0.01 to +0.02; E4 exploratory: butteraugli-only disagreement caught 45/46 contract failures
    (`rev4_e4_agreement_gate_2026-09-23.md`); E5 needs max pooling.
  - *Not* std pooling: std of these maps is already a function of mean and L2, Spearman ≥ 0.999996
    (`gmsd_2026-09-22.md` Part 3).
  - Integer bin counts are thread-invariant and strip-mergeable by construction. That is the reason for
    histograms over the scalar p90 kernel block5 priced at 5.11× glob (scalar replica path, an upper bound).
  - Cost goal ≤ 8%.
- **C4 — artefact-type descriptors.** Ratios of accumulators that already exist:
  - blur: EDGE_WIDTH_CHANGE × HF_LOSS;
  - noise: HF_GAIN restricted to flat reference regions;
  - blocking: C1;
  - ringing: C2;
  - **colour bleeding** (new): X/B gradient energy displaced from luma edges. It is `dst` chroma-edge energy
    outside a ±1-px luma-edge mask, relative to `src`.

  *Target:* E1b's cross-codec deficit of C on CID22-A (−0.0089 [−0.0130, −0.0049]). Only colour
  bleeding is new arithmetic. Cost goal ≤ 3%.
- **C5 — visibility in cycles/degree.** No new extraction for the linear part. At a declared ppd, pyramid
  scale `s` covers roughly `ppd/2^(s+1)` cpd, so a CSF weight per scale is a *reparameterised linear
  projection* of the per-scale features already in the bank. The potential run fits it as a constrained
  projection at E2a's documented geometries: CID22 46.9, KonJND 24.3, UPIQ-SDR ~51-57 ppd
  (`rev4_e2a_cvvdp_display_2026-09-23.md`). A nonlinear post-CSF masking stage stays with E3, now a side
  experiment. Status: candidate, bank-free.
- **C6 — chroma coverage: no separate family.** zensim computes X/B throughout, and GMSD's colour blindness
  is a peer weakness (`gmsd_2026-09-22.md`). The chroma wave was *refuted* validation-first (Y-only matched
  full-944 on chroma-distortion MOS; `../zenpapers/docs/zensim-720-feature-gaps-2026-07-26.md` line 28), so
  C1's chroma lattice and C4's colour bleeding are **low prior**, included only because they are near-free.
- **C7 — DVIFM-ish components.** Already registered (f956-985, `--full-986 --dvifm-spec`). They enter the
  bank as a family; nothing new to build.

**Rev3 numerics every candidate inherits** (`PLAN_FEATURE_REV3_2026-09-09.md`, `FEATURE_DEFECTS_AUDIT_2026-09-05.md`):
- `ZENSIM_FORMULA_REV=3` fused stable SSIM, `ZENSIM_ROOT_FORM=sqrt`, the era-2 reduction (F9);
- pool `width`, never the padded width (F3); H-tiling at entries (F10); the `h_mirror_add_idx` owner (F11);
- **no `Σs²/n − (Σs/n)²`** (F5): direct error moments or Terriberry;
- row-ordered f64 accumulation (the `blockiness_sparse_rows` precedent), thread invariance, strided rows,
  exact zero on identity for difference-form slots;
- registry entries with `form`/`direction`/`cost`/`revisions` (`FEATURE_SYSTEM_DESIGN_2026-09-05.md` §2);
  the servability census stays at 0 refused.

---

## 2. The f32 feature bank

### 2.1 What exists already, measured today (no labels read)

The September 14 recovery caches
(`/var/tmp/zensim-validation-2026-09-14/baseline-recovery/`, `benchmarks/baseline_recovery_2026-09-14.md`):

| cache | rows | sha256 | stored as |
|---|---|---|---|
| `cid22-train944.parquet` | 17,611 | `fb666c42…` | 905 `double` feature cols + **39 all-zero cols typed `int64`** (IDs in the WORKLOG), one row group, **no key-value metadata** |
| `safesyn-train944.parquet` | 196,086 | `6044fdc8…` | same shape |

- **Keys:** `row_id` (admission order), `ref_basename` and `reference_pixels_sha256` in the parquet;
  `distorted_pixels_sha256` only in the audit JSONL (`cid22-train-audit.jsonl`, schema
  `canonical-feature-audit-v1`).
- **Manifest:** `*.csv.manifest.json` carries the era `ceiling_rev3`, `formula_revision "3"`, the
  feature_set_id `…@w944/ceiling_rev3#b782e349` and the populated IDs, but **no `build_commit`**. The
  extractor sha256 is only in `*_ADMISSION.json` (`7c7ffbbf…` for CID22).

**f32 cast, measured on the first 3,000 CID22 TRAIN rows:**
- every value is in f32 range with no f32-subnormals;
- the max relative cast error is **5.96e-8** (≤ half an f32 ulp);
- **only 11.6% of basic cells are exactly f32-representable** (masked/IW 0.45%, v2 1.1%).

So the extractor emits more than f32 precision (f64 accumulation). **f32 storage is a lossy, deterministic
cast (round-to-nearest-even) of the extractor's f64 output.** The zero-numeric-change requirement is
therefore stated on the extractor's f64 output, and the f32 bank is a pure function of it. The cast error
(6e-8 relative) is 3 orders below F5's measured skew (3.63e-3) and 2 orders below the fused-Rev3 locality
bound (4.277e-6 absolute).

**Storage, measured on all 17,611 CID22 TRAIN rows × 905 live columns, cast to f32** (feature columns only,
`benchmarks/rev4_featbank_2026-09-23/f32_cast_probe.py`):

| encoding | bytes | B/row | notes |
|---|---:|---:|---|
| source f64 parquet | 131,368,060 | 7,459 | — |
| f32, zstd-3, dictionary | 81,622,870 | 4,635 | — |
| **f32, zstd-3, BYTE_STREAM_SPLIT** | **48,164,863** | **2,735** | chosen: exact round-trip; zstd-9 only 1% smaller |

SafeSyn is not measured (EXTRACT measures it).

Also reusable (Rev3 944, fresh native decode, `~/work/zensim-validation-2026-09-13/ceiling/final/`,
`_MANIFEST.json` era `ceiling_rev3`):
- KADID TRAIN 5,000 + TID 3,000 (`human_fit/half/dev`);
- KADID SELECT 3,125 (`human_test`, eval role);
- the codec proxy panel (620) and the corruption panel (8,213).

### 2.2 Which sets

**TRAIN-role (extract or convert first):**

| set | rows | route | role note |
|---|---:|---|---|
| SafeSyn | 196,086 | **convert** the Sep-14 cache; new families = new sidecars | SSIM2-derived targets |
| CID22 TRAIN (201 refs) | 17,611 | **convert** | SSIM2 targets only; human MCOS never trains |
| KADID TRAIN view | 5,000 | convert from ceiling/final (verify row keys against `ext_kadid_train`) | TRAIN; memorised by several eras |
| TID2013 | 3,000 | convert | TRAIN-only (§8.1) |
| KonFiG `originsplit_train` | per view | extract | TRAIN |
| KonJND BPG half | 403 refs | extract | TRAIN (JPEG half never trains) |
| KADIS train view | 40,040 | extract (optional, phase 2) | TRAIN, SSIM2 targets |
| fresh imazen-26 legs | per `joint-core-v2` manifest | **coordinate with the lane extracting 944 for fresh legs now** (`.workongoing` 2026-09-23T11:57Z); convert its output, do not duplicate | digits {0,2,4,6,8} |
| HDR `hdr_v3mix` train | 7,410 | phase 2, separate era (`hdr-common-primaries-v2`) | TRAIN |
| MCL-JCI | tbd | after the `datasets` lane proposes a role | USER DECISION |

**Held-out (potential and 3b only; labels in a sealed directory, read only through the ledgered admission):**
- CID22-A(25) (already fit-exposed for DVIFM constants, 2026-09-19);
- AIC-3 CTC 600;
- AIC-4 sample 300 (crop and full resolution separately);
- KonJND JPEG SELECT 404;
- CSIQ 866;
- KonFiG `originsplit_val`;
- KADID SELECT 3,125.

**Never:** CID22-B(24), KonJND TERMINAL, KADID TERMINAL, `originsplit_test`, any secret holdout, and LIVE
(target defect §8.2 until audited).

Extracting held-out *features* reads no labels, but the September 13 ruling forbids extracting test segments
and the September 14 clarification only permits assessing published TEST populations. **USER DECISION D4:**
explicit permission to extract features for the held-out sets above.

### 2.3 Layout

```
<bank>/                                   # /var/tmp/rev4-featbank/bank/ + LAN store + Tower mirror
  <set>/                                  # e.g. safesyn, cid22_train, kadid_train
    keys.parquet                          # pair_key, row_id, ref_pixels_sha256, dist_pixels_sha256,
                                          # ref_group (reference/origin id), codec, knob/q, width, height
    features__<tokens>__<era>__<hash8>.parquet   # pair_key + one f32 column per feature id "f<ID>"
    labels__<source>.parquet              # TRAIN targets only; never human held-out labels
    _MANIFEST.json
<bank>/_sealed/<set>/labels__human.parquet  # held-out human labels; read only by the ledgered admission
```

- **Row key:** `pair_key = sha256(ref_pixels_sha256 ‖ dist_pixels_sha256 ‖ input_contract)`, hex. It is
  content-addressed, so re-decoding the same pixels joins, and a decoder-era change (F16) cannot silently
  join. `row_id` keeps the admission order.
- **Sidecars:**
  - one sidecar per (compute token set, era, formula revision), with columns only for **populated** IDs
    (structural zeros are not stored; the manifest's populated list says which);
  - adding a family writes a new sidecar and **never rewrites an existing column**;
  - consumers join on `pair_key` and must refuse a row-count or key-set mismatch;
  - encoding: f32, zstd-3, BYTE_STREAM_SPLIT, row groups of 65,536.
- **`_MANIFEST.json`** per sidecar:
  - identity: `feature_set_id` (from `zensim::feature_set_id`, never hand-written), `formula_revision`,
    `root_form`, `input_contract`, `era`;
  - build: `build_commit`, `binary_sha256`, `decoder_identities` (crate + version per format);
  - inputs: `admission_sha256`, `pairs_sha256`, `input_file_sha256s` (or a pointer to the admission file);
  - values: `populated_feature_ids`, `dtype: "f32"`, `cast: "f64->f32 round-nearest-even"`;
  - run: `env`, `row_count`, `created_utc`, per-file sha256.
- **Registration:** `benchmarks/feature_sets_registry.json` (append-only), `~/work/zen/DATA_PROVENANCE.md`,
  `docs/DATASET_HISTORY.md`.
- **Converting existing caches is not a relabel:** same Rev3 extractor (sha256 in the admission file). The
  converter records `source_parquet_sha256`, rebuilds `pair_key` from the audit JSONL pixel hashes, refuses a
  row without a distorted-pixel hash, and writes the missing `build_commit` as `unknown(binary <sha256>)`.

### 2.4 Extractor work and budgets

Measure before optimising. The IMPL lane measures every new family on the same binary, interleaved, against
`fold944_full`, with zenbench and heaptrack at 256², 1024², 2048² and 4096²; it fits `α + β·px` and reports
ST and MT8.

**Budgets at 1024²** (full944 = 58.726 ms ST / 19.160 ms MT8 [SS]):
- C1 ≤ +5%, C2 ≤ +2%, C3 ≤ +8%, C4 ≤ +3%;
- all four together ≤ +15% ST;
- speed below 512² does not gate (REV4 user direction).

**Reuse:** the fold/streaming engine (`fold_engine`, strip walk), the per-scale planner (`feature_plan`,
scale-selective dispatch, [SS]), `incant!` tiers with `#[rite]` helpers, and the DVIFM histogram precedent.
A scalar research kernel's cost is an upper bound only (the geometry lane measured 1.43× call-per-op vs 1.04×
fair).

**Wall time:**
- measured today: SafeSyn full944 took 4,347.6 s at 8 threads and CID22 TRAIN 134.7 s
  (`baseline_recovery_2026-09-14.md`);
- a new-families-only sidecar pass costs one decode plus the new families; its wall is **measured** on 2,000
  SafeSyn rows first and then reported for the full set;
- no per-node zenfleet extraction throughput exists in any record (verified: no throughput number in
  zenmetrics `docs/RUNNING_JOBS.md` or `crates/zenfleet-*`).

**Fleet:** zenfleet `JobKind::Feature { regime, formula revision }` exists
(`zenfleet-core/src/job.rs:174-190`). Add a `rev4bank-<tokens>` regime that emits the f32 sidecar schema.
Allowed nodes: i134, r5600g, r3500, and the tower in Docker (capped). Gate: a 1-reference smoke per node, a
bit-identity check against local output, then scale. There is a review gate before any fleet run (§4).

### 2.5 Zero numeric change

Existing IDs f0-985 must stay **bit-identical** (`to_bits` on the f64 output) with every candidate toggle on
and off. The golden set:
- 64 CID22 TRAIN pairs + 64 SafeSyn pairs + 16 KADID TRAIN pairs;
- plus the audit's odd, sub-64, non-tight and past-`H_TILE_WIDTH` geometries;
- × {serial, rayon 8} × tiers {v4x, v3, scalar}.

A 16-pair tier runs in CI. Sources of truth:
- `v1_golden_bytes` and `fold_engine_parity`;
- the 944 CSV sha256 `bd01507e…` from the scale-selective run (19,958 × 944, byte-identical across its
  arms);
- a fresh re-extraction of 500 CID22 TRAIN rows compared against `cid22-train944.parquet`, which must match
  exactly.

**Plan revision (2026-09-24 UTC, coordinator):** CI lacks TRAIN
access. Its unignored 16-pair tier is synthetic
(`rev4_synthetic_16_pair_identity`). TRAIN coverage runs via
`just rev4-corpus-tests`; the 144-pair × 6-mode extractor matrix
qualifies the corpus.

---

## 3. The in-sample potential protocol

**Output is a ceiling, never a performance claim.** Every table and file carries `POTENTIAL — ceiling, not a
model score`. Fits live in `/var/tmp/rev4-featpot/fits/`: never packed as a profile, never on the board, never
used to pick a shipped recipe.

**Arms (feature sets), all from the bank:**
- **R0**: the Rev3 944 surface as extracted;
- **R0 − F**: R0 minus one family, for every ledger family;
- **R0 + C_k**: for each candidate family;
- **R0 + all C**;
- the **basic228** surface as a cheap reference.

**Models:**
1. **Lasso, full path.**
   - Owner: `zensim_validate::gram_lasso::lasso_cd` via `bake_dial_refit gram` / `fit-lasso`
     (`zensim-validate/src/bin/bake_dial_refit.rs`).
   - Needed extension: **per-group grams**. `gram` emits one Gram per reference, so a fold's Gram is a sum of
     reference Grams; CV costs no refit I/O.
   - λ: a 50-point log grid from λ_max down to λ_max·1e-4.
   - Standardisation: on each fit fold only.
2. **Linear projection:** BVLS with the existing sign masks (`gram_lasso::box_cd`, `--bounds-tsv` with the
   `feature_sign_mask_2026-05-26.tsv` convention). New candidate IDs get sign masks declared from their
   registry `direction`, written *before* fitting.
3. **Small MLP (upper probe):**
   - `zensim_mlp_train`, H32 and H128, 60 epochs, frequent checkpoints;
   - checkpoint selection only on an inner fold, never on the outer test fold;
   - the non-negative-distance head, so the dial contract can be checked.

**Estimates per (arm, model, dataset):**
- (a) in-sample fit;
- (b) **reference-grouped nested CV**: 5 outer folds by reference, λ and the checkpoint chosen by an inner
  4-fold by reference;
- (c) gap = (a) − (b), the overfitting signal;
- (d) cross-dataset transfer (§3b).

Statistics: SROCC, KROCC and PLCC, plus within-reference pairwise ordering accuracy and tail-band SROCC in the
E1 bands, all via `panel` / `scripts/lib/zen_stats.py`. Reference-clustered paired bootstrap,
B = 2,000, seed 20260923, through `panel --batch` resample manifests.

**Seed and stability protocol:**
- **Separate seed streams.** `--init-seed` and `--sample-seed` (CHANGELOG; `WAVE_PLAYBOOK.md` step 4).
  Sample seeds are raw-stream offsets, so seeds must be well separated and preflighted with
  `subset_sim --require-disjoint-sampler-windows` at the actual epoch/draw budget.
- **k = 5 per cell** (init × sample on a 5×5 Latin square for the MLP, so the two variance components
  separate). Lasso and BVLS are deterministic: one run per fold. Their stability comes from resampling.
- **Stability selection.**
  - For lasso: B = 200 half-sample **reference** subsamples per dataset. For each, record the λ-path entry
    order and the selected set at λ_1se.
  - A family's selection frequency is the fraction of subsamples in which ≥ 1 of its IDs is selected at
    λ_1se.
  - The MLP analogue is permutation importance on outer folds, reported only, not a gate.
- **Never best-of-k; never a bare mean.** Report mean, min, max and every per-seed value.
- **Representative, not lucky, seeds.** The seed list is fixed in the prereg before any fit. Seeds are never
  chosen by outcome. Per the 2026-09-05 subset-quality study, the sample seed changes the visiting *order*,
  not *which* rows, so "good" seeds cannot be picked by coverage. Instead, report the order spread and the
  init spread separately, and prefer stratified pair sampling where the trainer supports it.

**Decision rule (a family earns a Rev4 slot only if all hold):**
1. its nested-CV potential gain (R0 + C vs R0, or R0 vs R0 − F for existing families) has a paired
   reference-clustered 95% CI excluding zero on **≥ 2 human sets**, and a point gain ≥ +0.005 SROCC;
2. stability-selection frequency ≥ **0.6** (preregistered) on those sets;
3. it is within its §2.4 budget, measured;
4. under the non-negative-distance MLP head, the dial-contract gates (monotone ladders, identity 100) do
   not regress.

A family that meets 1-3 only on TRAIN-role sets is "TRAIN-supported, unconfirmed".

**Data roles — proposal. Every line needs USER DECISION D1:**
- **Potential fitting (in-sample and CV), TRAIN-role human sets:** KADID TRAIN, TID2013, KonFiG
  originsplit_train, KonJND BPG half. These need no new permission.
- **Potential fitting on held-out sets (new exposure):** CID22-A(25) (already fit-exposed 2026-09-19);
  AIC-3 CTC; KADID SELECT; KonFiG originsplit_val. Once fitted in-sample, each is **potential-exposed**: it
  can never again be quoted as held-out for any model or feature choice informed by the potential run.
- **Untouched confirmation for Rev4:** CID22-B(24) (sealed), the AIC-4 sample (the AIC-3/AIC-4 split
  proposal), KonJND JPEG SELECT, CSIQ, and every secret holdout.
- **Ledger:** one `docs/DATA_SPLITS.md` entry per batch, purpose "rev4 featpot". It lists the populations
  fitted in-sample, the rows, and the statement "fitted models are diagnostic only". Potential-exposed sets
  get a column in `benchmarks/eval_annotations.json`.

---

## 3b. Leave-one-dataset-out rotation

**Precedents.**
- `913f725c` (2026-02-22, the V0_2 trainer): K-fold CV within a dataset plus `--leave-one-out` across
  `--also` datasets, with content-aware grouping of all variants of a reference into one fold. Result on
  TID2013: 0.912 ± 0.015 trained vs 0.896 embedded.
- `ac260c42` (wave 10, `sota944_campaign_2026-08-03.md` Appendix H) left one *training leg* out, over
  10 legs, k = 2, with bands frozen before the run. The largest effect was not a leg at all but an
  **inverted KADID target** (CSIQ +0.115 outside noise, H.R1), and `tkadis` had negative marginal value
  (H.R2). Lesson carried forward: **label orientation is audited before any fold is fitted.** The `datasets`
  lane owns this for MCL-JCI and the rest.

**Rotation.**
- The candidate list comes from `DATASETS_DONE.md`, not landed at writing time. Provisionally: KADID, TID,
  KonFiG, KonJND-BPG, CID22-A, AIC-3, KADID SELECT and MCL-JCI (if the user gives it a role).
- For each dataset `D`: fit on all others (SROCC-comparable targets per dataset via the existing per-dataset
  affine / rank-normalised target; mixture weights equal per dataset, preregistered), and test on `D`.
- Same models, seeds, stability rules and bank as §3.

**Reported per fold, next to each other:**
- (i) the LODO transfer score on `D`;
- (ii) `D`'s in-sample fit;
- (iii) `D`'s nested CV (from §3);
- (iv) the incumbents B, C (`W10L9PH_s4004`), D and R915 fast/rich, plus peers (SSIMULACRA2, butteraugli
  3-norm and max, GMSD, IW-SSIM, MS-SSIM, and CVVDP at its documented display), from stored per-pair
  outputs, all on `D`.

**Readings:**
- the transfer matrix: which datasets transfer to which (a dataset whose (i) ≪ (iii) is idiosyncratic);
- per-fold stability selection: families selected in all folds are general, and those selected only when
  one dataset is in training are dataset-specific;
- whether the JPEG response-shape gap appears only in folds that hold out JPEG-heavy sets (AIC-3, KonJND,
  MCL-JCI). The JPEG-vs-other residual of each fold's model is computed with E1b's pair classifier.

**Data roles (USER DECISION D2, listed first in the DONE file):**
- **Withheld from ALL folds:** CID22-B(24), the AIC-4 sample, KonJND JPEG SELECT and TERMINAL, CSIQ, KADID
  TERMINAL, and every secret holdout.
- Every set that appears as a training fold becomes **LODO-exposed** in the ledger.
- **Fold models are quarantined:** written under `/var/tmp/rev4-featpot/lodo/`, filenames prefixed `LODO_`,
  never packed into `zensim/weights`, never on the board, never an input to recipe selection. Their only
  output is the transfer matrix and the selection frequencies.

---

## 4. Work orders

In `/home/lilith/tmp/zensim-paper/rev4/`, binding `DEVIN_COMMON.md`:
- `FEATBANK_IMPL_brief.md`: C1-C4 kernels, registry entries, bit-identity, cost.
- `FEATBANK_EXTRACT_brief.md`: converter + extraction of new sidecars on TRAIN sets. Review gate
  `FEATBANK_FLEET_READY.md` → `FEATBANK_FLEET_GO.md` before any fleet run.
- `FEATBANK_POTENTIAL_brief.md`: §3 + §3b, preregistered before any label read. It starts only after the
  user rules on D1/D2.

Order: IMPL → EXTRACT (conversion of existing caches can start immediately) → POTENTIAL.

## 5. What this plan does not establish

- No feature here is shown to help; C1-C4 are hypotheses with a measured motivation.
- Unmeasured: SafeSyn f32 size, every candidate's cost, all fleet throughput.
- The C1 mechanism (a contrast-invariant ratio producing the flat JPEG response) is an explanation
  consistent with E1b's sign flip. It is not a measured cause.
