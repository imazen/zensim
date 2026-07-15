# JXL near-lossless encoder bug — parquet contamination audit (2026-07-15)

**Status: COMPLETE.** Empirical on-disk identification (MEASURED) + three Sonnet
agents tracing the zensim / jxl-encoder / zenmetrics histories, all reconciled.

**Verdict in one line: the training corpora were never contaminated and shipped
bakes are unaffected; the only contaminated data was 33 cells of an eval-only dial
grid, now purged.**

## Timeline (from zensim chat history `a9bacddc`, cross-checked vs git)

- **2026-07-05** — user asked for a zenjxl near-lossless distance sweep to
  calibrate Profile B's dial; the sweep revealed the inversion: `distance 0.01
  → ssim2 33.94`, `0.02 → 33.95` (**worst quality, largest file**), `0.03 →
  96.02`. Committed as `00a56c12`, filed **imazen/zenjxl#18**.
- **2026-07-06** — two-stage fix. Stopgap `008499e1` (widen DC i16→i32 + a
  `VARDCT_MIN_LOSSY_DISTANCE=0.03` floor). User rejected the floor as a deferral
  ("fix, not defer"). Real fix `eeb52735` (closes **imazen/jxl-encoder#94**):
  the header field `modular_16bit_buffer_sufficient` was set true even when DC >
  i16, so conformant decoders reconstruct DC into i16 buffers → wrap → ANS
  desync. Fix sets `force_modular_32bit` when DC overflows i16; floor removed.
  Post-fix verified: distance 0.005 → 77.1 dB accept; ≥0.03 byte-identical.
- **2026-07-06** — re-sweep with the fixed encoder: **1200/1200 cells, 0
  failures**, distance 0.005–0.03 now ssim2 96.0–96.85 (commit `14b2f3c4`).
- **2026-07-07** — a *separate* downstream bug surfaced in the same
  investigation: Profile B's own **winsor bounds** were miscalibrated for
  near-lossless (not the encoder). Fixed + shipped
  `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin` (`aaa1ecac`). Already
  in `DATASET_HISTORY.md` §3.10.

Full record: `benchmarks/jxl_nearlossless_dial_2026-07-05.md` (11 parts).

## The bug (encoder side)

The JXL encoder produced **broken bitstreams at near-lossless distances
(butteraugli distance < 0.03)**. Root cause: a header lie —
`modular_16bit_buffer_sufficient = true` while the DC coefficients exceeded
i16 range, so conformant decoders (jxl-oxide, zenjxl-decoder) wrapped DC into
i16 → **ANS stream desync → garbage decode**. The trigger is **content-dependent**:
it fires only when DC energy exceeds i16, i.e. high-contrast / graphic content.

Symptom in metric data: a near-lossless cell that should score ssim2 ≈ 96–100
instead scores ≈ 34 (garbage), and its 372-D zensim feature vector explodes
(distortion features 30–40× their healthy magnitude — the reference-vs-garbage
"difference" is huge).

Fixed in jxl-encoder `008499e1` (2026-07-06T04:37Z — stopgap: widen DC i16→i32 +
a `VARDCT_MIN_LOSSY_DISTANCE=0.03` floor; closes zenjxl #18) and `eeb52735`
(2026-07-06T06:09Z — real fix: `force_modular_32bit` when DC>i16 so the header
stops lying, floor removed; closes jxl-encoder #94). Re-verified 2026-07-14 down
to distance 0.001 across jxl-rs + jxl-oxide + zenjxl-decoder (`a0f7e870`, CI
green) — **nothing remains broken post-fix.**

### Purge-critical facts (from jxl-encoder ground-truth trace)

- **Exact range: distance ≤ 0.02 reliably broken; the 0.021–0.029 band is
  content-dependent-suspect (onset ~0.025); distance ≥ 0.03 is ALWAYS fine —
  byte-identical / hash-proven pre vs post fix.**
- **Date bound: distance ≥ 0.03 is safe at EVERY date** (no bound needed). Only
  **sub-0.03 JXL generated before `eeb52735` (2026-07-06T06:09Z)** is suspect. The
  i16-DC flaw is structural to VarDCT since crate 0.1.3 (2026-02-14) — there is no
  "safe era" before the fix.
- **"qualities over 0.3d" decoded:** the broken zone is the *last ~0.3 quality
  points* — JXL-native quality ≥ 99.7 (`(100−q)/10 = distance`, so q=99.7→d=0.03).
  Not butteraugli distance 0.3 (that is ordinary lossy, q≈87, never affected).
- **Replacement rows are already correct for distance ≥ 0.03**; only sub-0.03
  cells ever needed the re-sweep.

## On-disk identification (zensim-side parquets — MEASURED 2026-07-15)

### CLEAN — no purge needed

| Parquet | JXL rows | Why clean |
|---|--:|---|
| `canonical-2026-05-21/train/safesyn.parquet` (+ source `training_safe_synthetic.csv`) | 26,362 `zenjxl-e7` | q5–q100; **q=100 → ssim2 med 95.13 (min 90.5), butteraugli med 0.229**. Highest quality sampled sits at butter ≈ 0.23, *above* the 0.03 broken floor — the corpus never entered the broken zone. ssim2/butteraugli monotone across all q. |
| `canonical-2026-05-21/train/cvvdp_iwssim_LARGE.parquet` | — | zenjxl distances only **{0.5, 1, 2, 5}** — 16–250× above the boundary. |
| `canonical-2026-05-21/scores/{ssim2,cvvdp,iwssim}_imazen*.parquet` | JXL q=10..90 | q=10..90 maps to distance ≥ 1 (far above the 0.03 floor). ssim2 max 95.4, healthy by-q. |
| `canonical-picker-2026-06-27/zenjxl_lossy`, `canonical-picker-2026-07-01-zensimA/zenjxl_lossy` | — | q ∈ [5, 90], generic-quality-only — structurally never resolves to native distance < 0.03. |
| `jxl-hqfill-2026-07-01/zenjxl_lossy_hqfill_2026-07-01.parquet` | 62,958 rows | distance range **[0.05, 1.3]** (direct duckdb query). |
| `hqfill_7metric_sidecar_2026-07-02.parquet` | 62,173 rows | distance range **[0.05, 1.3]**. |
| `s3://zentrain/multi-codec-2026-05-18/omni/` | 6,400 zenjxl | distance **[0.1, 10.0]**. See the correction below — this sweep DID complete (112/112 objects); it was never *merged* into LARGE. Clean either way. |

### CONTAMINATED — near-lossless eval dial grid

`dial_grid_372col_2026-05-29.parquet` (and its two byte-identical copies:
`..._quarantined.parquet` — the w11 quarantine dropped webp rows but **kept**
these JXL cells; and `qsweep_expanded_2026-05-29/dial_grid_372col.parquet`).

- JXL sweep is `param_kind=distance`, min distance **0.025** — 33 cells sit at
  d=0.025, inside the broken `<0.03` zone; grid dated 2026-05-29, **before** the
  fix.
- Healthy near-lossless ceiling (measured over the clean d=0.05..0.35 cells):
  **max|feat| = 1.56**, mean feature-L2 rising smoothly 0.109 → 0.246.
- At d=0.025: mean feature-L2 = **4.011**, max|feat| = **59.29** — a 37× distortion
  explosion at the *lowest* distance, backwards from the monotone trend = the
  garbage-decode signature.
- Content-dependent, as predicted by the DC>i16 mechanism — **4 of 33 cells are
  unambiguously broken** (max|feat| ≫ the 1.56 clean ceiling):

  | image_id | max\|feat\| @ d=0.025 | @ d=0.05 (healthy) |
  |---|--:|--:|
  | `b2e6e2b5969eaf25_1022x818` | 59.29 | 0.02 |
  | `85d6b54b6872b19b_512sq` | 5.61 | 0.04 |
  | `7f7998c62e54398f_1024sq` | 3.53 | 0.03 |
  | `3316926_opo25u_512sq` | 3.24 | 0.03 |

  The remaining 29 d=0.025 cells are within noise of the clean region but were
  still generated in the broken zone → the whole d=0.025 JXL slice is treated as
  suspect for purge.

**Impact:** eval-only (DIAL panel monotonicity). Not training data. Near-lossless
JXL was independently re-characterized in July with the fixed encoder (near-lossless
"Parts 4–11"), so these 2026-05-29 cells are already superseded for analysis; the
purge removes stale broken rows so no future run rescores them.

### RESOLVED — `cvvdp_iwssim_LARGE` is clean; its extreme features are a DIFFERENT bug

`canonical-2026-05-21/train/cvvdp_iwssim_LARGE.parquet` (73,300 rows) looked
suspicious on feature magnitude alone (max|feat| up to 1181) but that is
**confounded**: its references are `gen-chart__*` graphic content, which triggers
the *separate* known IW-feature explosion (DATASET_HISTORY §3.19,
[[iw-feature-normalization]]) — it has no codec column and null ssim2, so
feature magnitude cannot attribute it. Resolved by direct query instead: LARGE's
zenjxl distances are only **{0.5, 1, 2, 5}**. **Clean. Do not conflate §3.19 with
this bug.**

### ⚠ CORRECTION — "the fresh-jxl sweep never landed" is WRONG

An earlier revision of this doc and of DATASET_HISTORY §3.22 said LARGE's
2026-05-18 "fresh jxl" 8-distance sweep "was PLANNED but BLOCKED by a vast.ai/cudarc
bug and never landed (`c17447f5`)". That was read off a commit message and is **not
true**. Direct R2 verification (`aws s3 ls --recursive --summarize` on
`s3://zentrain/multi-codec-2026-05-18/omni/`) shows the sweep **COMPLETED**: 112/112
objects dated 2026-05-19, 24,800 cells, of which 6,400 zenjxl rows span distance
**[0.1, 10.0]**. It was never *merged* into canonical LARGE — "blocked" described the
merge, not the sweep. The clean verdict is unchanged (two independent routes), but the
"never landed" claim must not be repeated.

### Near-miss — the one sweep that would have mass-produced contamination

`jxl-dense-20260530` — 2k k-means sources × a 44-distance ladder **starting at
0.025** — is the only planned dataset that would have landed squarely in the broken
zone at scale. **Confirmed never launched:** `s3://coefficient/jobs/jxl-dense-20260530/`
holds only the 2 prep files (`chunks.jsonl`, `omni_backfill_chunk_worker.sh`, both
2026-05-30 09:08:06); no `chunks/`, `done/`, `features/`, or `variants/` ever appeared.
It was blocked on red sweep-image CI (`9d8f73a5`) and never resumed. Had it run in
that window, this incident would have been a genuine large-scale training-data purge.

### Where the record lives (zenmetrics has none)

The zenmetrics repo and its session history contain **zero** record of this incident —
exhaustive ripgrep across all zenmetrics sessions + ~100 subagent transcripts
(~123 MB, 2026-06-18 → 2026-07-13) found nothing. The diagnosis, both fixes, and the
re-sweep all happened in **zensim** + **jxl-encoder**, using the `zenmetrics` binary
only as a CLI dependency (`zenmetrics sweep --codec zenjxl …`). The contaminated dial
grid itself was built by `scripts/v_next/build_qsweep_expanded.py` →
`zenmetrics sweep --metric zensim-gpu --zensim-features-regime with-iw` as a **local**
run, not a fleet job. Authoritative record = this doc +
`benchmarks/jxl_nearlossless_dial_2026-07-05.md` + DATASET_HISTORY §3.22 + GitHub
`imazen/zenjxl#18` / `imazen/jxl-encoder#94` (both closed with verified-resolution
comments).

## Purge — DONE (zensim side), 2026-07-15

Followed the repo's established quarantine pattern (as used for w11) rather than
mutating a canonical grid: **originals are preserved untouched**, a clean sibling
is published, and historical dial numbers stay reproducible.

**`dial_grid_372col_2026-05-29_quarantined_v2.parquet`** — 4,424 rows
(4,457 → 4,424), sha256
`6546c43e6d9572dcf0740c6346cd604fd8cd3ff01ee2f7031aca998fd8fec2bd`, 7,819,228 B.
Drops **both** contaminations: the w11 masked/IW webp ladders (inherited from v1)
**and** the 33 pre-fix JXL cells at distance < 0.03.

Verified: 0 JXL cells below d=0.03 remain; 1,504 JXL cells retained with min
distance now 0.05 — i.e. **distance ≥ 0.03 is kept**, because it is
byte-identical/hash-proven at every date and needs no re-encode. Quarantine
provenance (mechanism, measured evidence, the 4 named broken images) is embedded
in the parquet's KV metadata under `quarantine_note_v2`, so it travels with the
file and cannot desync from this doc.

- Local: `/mnt/v/output/zensim/eval_panels_2026-05-29/`
- R2: `s3://zentrain/eval-grids/dial_grid_372col_2026-05-29_quarantined_v2.parquet`
- Pointer updated: `benchmarks/eval_grids_2026-05-29.pointer.md`

**Why drop the whole d<0.03 slice rather than only the 4 provably-broken cells:**
the encoder team's own boundary is that 0.021–0.029 is *content-dependent-suspect*
pre-fix (the DC>i16 onset is ~0.025 and depends on content), so a cell that looks
clean there is unverifiable rather than known-good. It is 33 of 4,817 cells (0.7%)
of an eval grid whose near-lossless reach is already covered by d≥0.05 and by the
separate post-fix refit corpus — dropping costs nothing and removes the risk of
future bakes being scored on garbage.

**Not spliced with replacements:** the post-fix re-sweep
(`/mnt/v/output/zensim-jxl-nearlossless/refit/`) covers a *different* 200-reference
set than the dial grid's 33, so there is no 1:1 replacement for these cells. Dropping
is the correct action; regenerating that slice would need a fresh extraction against
the fixed encoder.

## Standing guard for future JXL data

Any JXL cell is contaminated iff **distance < 0.03 AND generated before
2026-07-06T06:09Z** (`eeb52735`). Distance ≥ 0.03 is safe at every date; anything
generated after the fix is safe at every distance (re-verified to d=0.001). That
two-clause test is the whole rule — apply it to any new JXL corpus before use.

## Remaining

- `/mnt/v/output/zensim-jxl-nearlossless/refit/` (the corrected near-lossless
  re-sweep: 200 refs × 6 distances 0.005–0.03 = 1200 cells, 0 failures) is
  **local-only** — not mirrored to R2 or Tower. Worth mirroring per the
  ML-data-discipline rule, since it cost a GPU re-sweep to produce and is the only
  copy of the post-fix near-lossless data.
- Two zenmetrics bugs surfaced by the rebuild were "logged for deliberate fix" and
  could not be confirmed fixed by name in zenmetrics git log: a stale
  `zenwebp?/zencodec` feature ref, and `hdr.rs::rgb16_hlg_to_nits` using
  `cvvdp::params` under `cfg(feature="png")` without cvvdp as a direct dep (`cvvdp`
  is now a direct optional workspace dep — unverified whether that incidentally
  resolves it).

## Replacement rows

Believed to exist from a fixed-encoder re-sweep (zensim commit `14b2f3c4`
"Rebuilt zenmetrics vs fixed jxl-encoder … 1200/1200 cells: distance 0.005-0.02
now ssim2 96.0-96.85 (was broken ~34)") and the dense JXL fleet-sweep prep
(`f5e813e4`, 2k k-means sources × 44-distance ladder). Exact R2/local paths being
located by the zenmetrics history agent.
