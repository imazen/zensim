# Parity + CID22-methodology + holdout-overlap plan

**Authored**: 2026-05-11 (post-V0_5 ship).
**Trigger**: user directive — "achieve parity and matching results not
just smoke tests. be thorough, methodical, and make rust code do
matching methodology to each page of the 30-page paper and make it
match and reproduce the ssim2 results on cid validation, but then add
additional testing against balanced and extensive synth corpus holdout
only. find a way to validate no holdout overlap incl cropped variants."

Expands the scoping doc `WASM_CUBECL_TRAINER_PLAN.md` from "smoke
test parity" (Phase 1 milestone) to a full **methodology-conformant
+ bit-faithful reproduction harness** before any WASM polish.

---

## Five user goals

| # | Goal | Status | Owner artifacts |
|---|---|---|---|
| 1 | **Rust trainer ↔ Python trainer parity** (same ZNPR v2 bytes for a fixed seed) | not started | `zensim-train-core/tests/parity_*.rs`, `scripts/v_next/dump_python_state.py` |
| 2 | **Methodology matches each of the 30 paper pages** | partially extracted | `docs/CID22_PAPER_PAGE_BY_PAGE_2026-05-11.md` (new) |
| 3 | **Reproduce CID22 paper SSIM2 numbers** (Tables 3, 5, 6) on the 49-ref held-out validation | not started | `zensim-bench/examples/reproduce_cid22_table_3.rs` etc. |
| 4 | **Balanced extensive synth-corpus holdout testing** | not started | `zensim-train-core/tests/balanced_holdout.rs`, new synth split spec |
| 5 | **No holdout overlap detection (including cropped variants)** | not started | `zensim-validate/src/holdout_overlap.rs`, `bin/check_holdout_overlap` |

Every goal lands as a tracked deliverable with success criteria and
a measurable artifact under git. No goal is "done" until the artifact
is committed and reproducible by a clean clone of the repo.

---

## Goal 1 — Rust ↔ Python trainer parity

### Definition

Given **the same input features file, the same seed, the same
hyperparameters**, both trainers must produce **a ZNPR v2 byte
sequence whose final-Linear-layer weights and biases differ by at
most fp32-quantization error** (~1e-7 per element) from each other.

Bit-identical fp64 internals are NOT the goal — Python uses
NumPy/PyTorch BLAS (which may use FMA / SIMD reorderings) and Rust
uses scalar f64 loops. Order-dependent ops can differ by ~1e-15.
What we DO require:

- **Same SROCC** on CID22 49-ref held-out validation, to within ±0.002.
- **Same per-band SROCC** (B0/B1/B2/B3 from `CLAUDE.md`), to ±0.005.
- **Same non-monotonic q-step rate** on JPEG unified parquet, to ±0.2%.
- **Same scaler_mean / scaler_scale**, byte-identical after f32 cast.
- **Same Layer-2 weights/bias direction** — cosine similarity > 0.9999.

### Subtasks

1. **Pin Python trainer behaviour** — add `scripts/v_next/dump_python_state.py`
   that runs `train_v_next_mlp.py --seed S --epochs E` AND emits per-step
   weight snapshots at epoch 1, 10, 100, end. Snapshots stored as
   `.npz` files under `benchmarks/parity/python_seed_S/`.
2. **Match init** — `SplitMix64` Xavier-Glorot init matches PyTorch's
   `torch.manual_seed(S)` default init? Almost certainly NO — Python uses
   Mersenne Twister and Kaiming init by default. We need to either
   (a) port the Rust trainer's SplitMix64 init to Python first, OR
   (b) port Python's exact init to Rust. Decision: do (a) since the Rust
   trainer is the future runtime; the Python trainer feeds it the
   features file but the actual weight init lives in Rust.
3. **Build parity harness** — `zensim-train-core/tests/parity_seed_42.rs`
   loads a small fixed feature matrix (synthetic from `SplitMix64(42)`),
   trains for 10 epochs, asserts ZNPR v2 bytes match a checked-in golden.
4. **Add cross-trainer parity test** — `zensim-validate/tests/cross_parity.rs`
   runs BOTH `zensim_mlp_train` (existing) and the new `train_mlp_core`
   on identical input and asserts:
   - scaler bytes identical
   - SROCC on a 10k-pair holdout differ by < 0.002

### Decision points

- **A** (recommended): freeze the random init in `SplitMix64` and update
  the Python trainer to use the same init. This makes the Rust trainer
  the source of truth.
- **B**: port PyTorch's init to Rust. Required IF we want to bit-match
  Python's existing baked weights, which is NOT a stated goal.

Go with A.

### Success criteria

- `cargo test -p zensim-train-core --test parity_seed_42` green.
- `cargo test -p zensim-validate --test cross_parity` green.
- A committed `benchmarks/parity/golden_seed_42_e10.bin` under git, plus
  a markdown comparison table at `docs/parity_results_seed_42.md`.

---

## Goal 2 — Methodology matches each of the 30 paper pages

### Definition

For each page of `CID22_wg1m99012.pdf`, produce one line in
`docs/CID22_PAPER_PAGE_BY_PAGE_2026-05-11.md` describing:
1. What that page says (1 sentence).
2. What load-bearing fact / methodology our pipeline must conform to.
3. Whether we currently conform (yes / no / partial).
4. The owner artifact that enforces it (test, doc, code).

The existing `CID22_PAPER_NOTES_2026-05-07.md` summarizes findings;
this new doc grounds every finding to a page so we don't lose any
methodology detail.

### Subtasks

1. Read the paper end-to-end via `pdf_oxide` (PDF→markdown converter,
   referenced in MEMORY.md) and produce a page-by-page log.
2. For each load-bearing methodology element, link to the owning
   artifact (test or code path).
3. Open issues for missing pieces with concrete reproduction tasks.

### Pages we know are load-bearing (from prior reading)

- **p. 1–5**: TSBPC + DSBQS protocols; MCOS bias correction. → must
  reflect in any cross-codec eval doc.
- **p. 8**: 14.7% TSBPC session discard rate via honeypot screening.
  → we don't crowdsource opinions; for trainer use, drop pairs where
  ssim2 ↔ butteraugli rank disagrees within (image, codec) group.
- **p. 10–14**: Table 3 (per-metric SROCC) + Table 5 (quality-scale
  mapping) + Table 6 (pairwise SROCC). → reproduction targets for
  Goal 3.
- **p. 15–17**: Honeypot rules + monotonicity smoothing (200 dummy
  opinions/pair). → enforce monotonicity penalty in trainer (already
  done as `--tv-weight`).
- **p. 22**: 49 / 250 references held-out from SSIMULACRA 2 weight
  tuning. → blocklist on synth generator (we do this).
- **p. 24**: Reference MCOS = 88.3 mean (not 100). → calibration
  target for `affine_calibrate_znpr_v2.py`.
- **p. 26**: SSIMULACRA 2 architecture (6 scales, XYB color space,
  108 sub-scores). → architecture parity target for future work.
- **p. 28**: Figure 7 sample-size guidance. → not directly applicable.

### Success criteria

- `docs/CID22_PAPER_PAGE_BY_PAGE_2026-05-11.md` exists with 30 entries.
- Every "load-bearing" entry references an owning artifact.
- Every "no conformance" entry links to a tracking issue or TODO.

---

## Goal 3 — Reproduce CID22 paper SSIM2 numbers

### Definition

Run **our** `zensim` (or `fast-ssim2`) on the 49-ref held-out CID22
validation set and reproduce:

- **Table 3 row**: `SSIMULACRA 2 KRCC=0.6934 / SROCC=0.882 / PCC=0.8601`
  (on the FULL 250-ref set, mostly in-training-set for SSIM2).
- **Table 3 49-ref-holdout** (from paper text): `KRCC 0.7033 / SRCC
  0.88541 / PCC 0.87448 / MAE 4.97`.
- **Table 5 anchors**: medium=50 / high=65 / lossless=90.
- **Table 6 row**: `pairwise within-source SSIM2 SROCC 0.9210`.

Reproduction needs to be within ±0.002 SROCC to count.

### Subtasks

1. **Verify our fast-ssim2 implementation against paper SSIM2**. The
   Sneyers reference implementation is at `github.com/cloudinary/ssimulacra2`;
   our `fast-ssim2` Rust port (in `~/work/zen/fast-ssim2/`) was
   validated against it but the exact SROCC numbers on CID22 need to
   be re-checked.
2. **Score all 22,153 CID22 distorted images** with `fast-ssim2`
   against their refs. Store in a parquet with `(image_path, codec,
   ssim2_score, mcos_score)`.
3. **Compute per-band SROCC** (Table 5 cutoffs) on:
   - Full 250-ref set (matches paper Table 3 row 1).
   - 49-ref held-out subset (matches paper held-out claim).
4. **Compute pairwise within-source SROCC** (matches Table 6).
5. **Document discrepancies** — anything > ±0.002 SROCC off is a bug
   in our SSIM2 port and gets a follow-up issue.

### Success criteria

- `zensim-bench/examples/reproduce_cid22_table_3.rs` runs in < 5 min,
  emits the reproduction table to `benchmarks/cid22_repro_2026-05-11.md`.
- All numbers within ±0.002 SROCC of paper values.
- Per-band SROCC matches CLAUDE.md's mandatory per-band reporting rule.

### Then: zensim numbers

After SSIM2 is reproduced, run zensim V0_5 against the same 49-ref
set and report the same metrics. Per-band SROCC must match-or-exceed
SSIM2's per-band per the V0_5 shipping criteria. If it doesn't, that
gives concrete page-by-page direction for V0_6.

---

## Goal 4 — Balanced extensive synth-corpus holdout testing

### Definition

Build a held-out synth corpus that is:
- **Balanced** across the 4 CID22 bands (B0/B1/B2/B3).
- **Balanced** across the 6 CID22 codec classes (JPEG, WebP, JXL,
  AVIF, MozJPEG-variants, …).
- **Balanced** across the ~15 content classes (portraits, screens,
  line art, photos, etc).
- **Disjoint from training** — neither the source image nor any
  cropped/resized variant of it appears in training.
- **Large enough** for per-cell SROCC at p25 / p50 / p75 — ~50
  pairs per (band × codec × content-class) cell ≈ **18,000 pairs
  minimum**.

This is a stricter holdout than CID22's 4,292 pairs because we
control the source corpus.

### Subtasks

1. **Define content classes** — start with the 7 from `zenanalyze`
   content classification (or expand to match the paper's 15).
2. **Sample source images** — cluster all 3,579 safe-synthetic
   sources into 7 clusters via k-means on `zenanalyze` features,
   pick 50 centroid-nearest per cluster (≈350 sources).
3. **Generate synth pairs at each band** — using the existing synth
   generator binary; target ~50 pairs per cell.
4. **Score with truth (ssim2)** and store as a parquet.
5. **Build the holdout eval harness** — `zensim-bench/examples/balanced_holdout_eval.rs`
   loads the holdout parquet, runs zensim, reports per-cell SROCC.

### Success criteria

- A committed `/mnt/v/output/zensim/balanced_holdout_2026-05/holdout.parquet`
  with at least 18,000 pairs across 4 bands × 6 codecs × 7 content classes.
- Manifest stored as `benchmarks/balanced_holdout_2026-05/manifest.json`
  with full source-image list, cluster IDs, generation seeds.
- An eval harness that runs end-to-end.

---

## Goal 5 — No holdout overlap detection (including cropped variants)

### Definition

Build a tool that, given any zensim training CSV/parquet and the
CID22 49-ref validation set, **proves zero overlap** including:
- Exact-image overlap (md5).
- Resized-image overlap (any aspect-preserving resize).
- Cropped-image overlap (any contiguous rectangular crop).
- Color-augmented overlap (brightness, contrast, gamma shifts).

### Approach

A two-stage detector:

1. **Stage 1: perceptual hash** — compute pHash (or
   `zenanalyze`-feature-vector hash) of every CID22 validation
   reference, every training source. Cluster training sources whose
   pHash distance < threshold to a CID22 ref. Stage 1 catches exact +
   resize + small-color overlap.
2. **Stage 2: cropped-variant check** — for each Stage-1 cluster, run
   feature-based template matching (use `zenanalyze` tier-1 features
   on a 4×4 grid over the training source; match against any rotation
   / crop of the CID22 ref). Slow but exhaustive — only run on
   clusters Stage 1 already flagged.

### Subtasks

1. **Implement Stage 1** as `zensim-validate/src/bin/check_holdout_overlap.rs`.
2. **Implement Stage 2** as the same binary's second mode.
3. **Run against the current safe-synthetic 218k corpus** and confirm
   ZERO Stage-1 hits and ZERO Stage-2 hits. If hits appear, that's a
   shipping bug.
4. **Run against the extended 340k corpus** (with the 122k zenjpeg-420-e1
   fill) — same expectation.
5. **Document the result** as `benchmarks/holdout_overlap_audit_2026-05-11.md`.

### Success criteria

- Binary builds, runs in < 30 minutes against the safe-synthetic 218k.
- Reports 0 Stage-1 hits and 0 Stage-2 hits.
- If any hits found, they're documented and a follow-up issue is
  opened. (Per CLAUDE.md: "CID22 training data still must NOT be
  added".)

---

## Phasing & priority

Suggested execution order across the next ~20 ticks:

1. **Goal 5** (overlap detector) — **HIGH priority**. We don't yet
   know with certainty that the 218k synth corpus is overlap-free.
   The 49-ref blocklist was on filename hashes, NOT on perceptual
   content. Cropped variants would have slipped through.
2. **Goal 3** (SSIM2 reproduction) — **HIGH priority**. We claim
   match-or-exceed-ssim2 as our shipping bar. Reproducing the paper
   numbers first proves our ssim2 implementation is correct; only
   then can we honestly say zensim V0_5 matches it.
3. **Goal 2** (page-by-page methodology) — **MEDIUM**. Background
   documentation work; can run interleaved.
4. **Goal 1** (parity test) — **MEDIUM**. The Rust trainer is not
   yet shipping any weight (the Python trainer's bakes are still
   what gets affine-calibrated and dropped into `zensim/weights/`),
   so parity is a future-proofing concern, not a current bug.
5. **Goal 4** (balanced holdout) — **MEDIUM**. We have CID22 as the
   gold standard already; balanced synth is "extra" but needed for
   per-band SROCC stability.

---

## What this displaces

This plan **does not displace** the WASM-trainer plan
(`WASM_CUBECL_TRAINER_PLAN.md`) — it precedes it. WASM Phase 1
(porting `train_mlp_with_tv` body to `zensim-train-core`) continues
as a parallel workstream when there's idle cycles. The user's
"smoke tests are not enough" feedback means **WASM Phase 1's
deliverable must include the parity tests from Goal 1**, not just a
"runs without panicking" smoke test.

---

## Tracking

Every tick that touches this plan documents its work in
`~/work/zen/zenanalyze/zensim_champion_log.md` with the tick number,
the goal it advanced, and the artifact produced. The plan itself
gets revised when assumptions change.
