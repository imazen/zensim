# kernel-int16 — DVIFM phases 1-3: build, screen, and an Int16 serving kernel

## What was asked

Build a DVIFM (5-level binomial-pyramid, Laplacian/Local-band, block-visibility)
feature family into zensim as an opt-in, default-off, additive-only kernel
(phase 1); screen it against basic228 on human-labelled TRAIN data with
increasingly aggressive fitting (phases 2, 2b, 2c); then, once the family was
judged not independently useful as a *score* but interesting as a mechanism,
build a bit-exact Int16 serving kernel and measure whether it clears the
production cost bars (phase 3). Briefs: `../briefs/dvifm_phase1_prompt.md`
through `dvifm_phase3_perf_prompt.md`.

## Verdict (from the reports — do not re-derive, read `../reports/`)

- **Phase 1 (build):** PASS on every correctness/parity/SIMD/byte-stability
  gate. Cost: +31.6 ms (+66%) @1024², +120.5 ms (+54%) @2048² for the f64
  oracle — real, not claimed cheap.
- **Phase 2 (Laplacian/Local screen):** NEGATIVE on all three rounds —
  every seed-paired diff of `basic228dvifm − basic228` is negative.
- **Phase 2b (10-seed confirmation):** INFO-NOT-USEFUL — DVIFM carries real
  within-image signal (beats its own row-permuted twin decisively) but is
  redundant on top of basic228 (net harmful, −0.0029 dev per-ref mean).
- **Phase 2c (native Y'CbCr, human + codec-proxy labels):** NOT-A-SUBSTITUTE
  on human rows (Δ −0.00454 vs y60); NEGATIVE on codec-proxy Q2, with JPEG
  losing worst (−0.1105).
- **Phase 3 (Int16 serving kernel):** the integer kernel is bit-exact against
  an independent Python reference (`dvifm_int_ref.py`) and **PASSES** both
  cost bars (31.8 ms @1024², 122.5 ms @2048², vs bars of 50/200 ms); the f64
  oracle sits AT/OVER the bars. `DvifmParams::int16()` is the fitted serving
  definition; `default()` stays the ExactF64 oracle. No re-screen was run in
  the integer domain — that's open work, not done here.

Net standing conclusion (confirmed independently by the `verdict-x1x2` lane):
DVIFM as a standalone score is not competitive with fast-ssim2 or the shipped
zensim bakes. The kernel exists, is fast, and is architecturally correct;
it is not qualified for production and stays `dvifm_block` default-OFF.

## Code in this directory

| file | phase | role |
|---|---|---|
| `dvifm_parity_fixture.py` | 1 | numpy reference used to generate the phase-1 parity fixture (still in main checkout at `scripts/dvifm_parity_fixture.py`) |
| `dvifm_int_ref.py` | 3 | independent integer (Q14/Q15) reference implementation; ground truth for the int16 kernel's bit-exactness fixture |
| `build_mech_subset.py`, `dvifm_cache.py`, `derive_constants.py`, `fit_params.py` | 2 | phase-2 mechanism subset builder, block-stats cache reader, and the screen-3/4 constant derivation + Adam fit |
| `run_probes.sh` | 2 | phase-2 probe driver (160-epoch budget probes) |
| `admit_2b.py` | 2b | admission/segment builder for the 10-seed confirmation |
| `admit_2c.py` | 2c | admission/segment builder for the human+codec-proxy Q1/Q2 screen |
| `screen2d/*` | 2d | the largest phase: pair building, imazen26 selection, standalone fitting (`fit_standalone.py`, 48 KB — over the 30 KB note-threshold, included as source per the task's size exception), CID22-B unsealing, surfaces report, parquet conversion, orientation-pair checks, shell drivers for extract/fit/score-all |

## Rust code (NOT copied here — see commit ids)

- Main checkout (`/home/lilith/work/zen/zensim`, already ancestors of the
  current tip, fully committed): `mzvwpxrq`/`35880863` (phase-2d prereg +
  ledger docs), `rnywwrvu`/`898b87ed` (native Y'CbCr input planes — touches
  `zensim/src/{color,dvifm,feature_v2,research,streaming}.rs`,
  `scripts/dvifm_parity_fixture.py`, a YCbCr parity fixture).
- Workspace `zensim--dvifm3` (`/home/lilith/work/zen/zensim--dvifm3`), clean
  working copy (`yrqrpqnr`/`12e5f301`, empty — nothing at risk), unpushed,
  not merged to main:
  - `srmtxyuk`/`3dc4e78a` — step-0 stage-cost split (bench only)
  - `wzrskmtl`/`943781e8` — step-1 exact shortcuts + step-2 `zensim/src/dvifm_int.rs`
    (the Int16 kernel), `scripts/dvifm_int_ref.py`, parity fixture
  - `tkllsmyz`/`fa0b60f0` — step-2b int16 monotone test + i686 fixes
  - `rllusmtv`/`0d3f8c00` — step-3 cost gate bench

## How to re-run

Everything here is TRAIN-only, non-destructive analysis over already-extracted
feature/cache data; nothing writes to a holdout.

1. Build the workspace with `training,feature-regime-v2` features enabled
   (see the dvifm3 workspace's `Cargo.toml` — this code was written against
   `zensim/src/dvifm.rs` + `dvifm_int.rs` at commit `rllusmtv`/`0d3f8c00`).
2. Parity check: `python3 dvifm_int_ref.py <args from the phase-3 report>`
   against the fixture at
   `zensim/tests/fixtures/dvifm_int_parity_2026-09-20.txt` (in the dvifm3
   workspace) — see `../reports/dvifm_PHASE3_DONE.md` for the exact sizes/arms.
3. Phase-2 mechanism screen: `run_probes.sh` drives the 160-epoch probes;
   `derive_constants.py` + `fit_params.py` derive/fit the Laplacian/Local
   constants from the TRAIN-only block-stats cache described in
   `../reports/dvifm_PHASE2_DONE.md` (two caches, 11,125 rows each, 8.25 GB
   each — not reproduced here, re-extract via
   `zensim-bench --full-986 --dvifm-block-stats` per that report).
4. Phase-2d (screen2d): `screen2d/build_pairs.py` →
   `screen2d/select_imazen26.py` → `screen2d/fit_standalone.py` (the main
   fitter; `test_fit_standalone.py` is its test) → `screen2d/fit_to_spec.py`
   → `screen2d/score_cid22b.sh`/`score_all.sh`. Inputs consumed: TRAIN-origin
   admitted rows per `../reports/dvifm_PHASE2B_DONE.md`'s N_max table
   (KADID-10k train refs, TID2013, KonFiG origin-split train+dev2 — never
   CID22/AIC/KonJND-val/KonFiG-test).

## Artifacts (reference by path — not copied, too large / not source)

- `/mnt/v/output/zensim/dvifm-screen-2026-09-19/` (phase 2, 20 GB: caches,
  `mech/` run logs, `probe4/`)
- `/mnt/v/output/zensim/dvifm-screen2b-2026-09-19/` (phase 2b, 969 MB)
- `/mnt/v/output/zensim/dvifm-screen2c-2026-09-19/` (phase 2c: `q1-human/`,
  `q2-codec/`, `segments/`)
- `/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/` (phase 2d: `caches/`,
  `fits/`, `imazen26/`, `logs/`, `pairs/`, `scores/`, `specs/`,
  `tables/done_tables.{json,md}`)
- `~/tmp/devin/dvifm3-ab/`, `~/tmp/devin/dvifm3-dev/` (phase 3: A/B cost
  blocks, 584 MB int block cache, deviation report)
- Committed benchmark records (permanent, in the repo already, not copied
  here): `benchmarks/dvifm_block_gates_2026-09-19.md`,
  `benchmarks/dvifm_screen*_2026-09-19.{md,json,pointer.md}`,
  `benchmarks/dvifm_perf_2026-09-19.md`,
  `benchmarks/dvifm_int16_{deviation,cost}_2026-09-20.md`.
