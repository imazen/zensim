# joint-core-v1 — DONE with a measured too-small finding (2026-09-20)

The build executed end to end and every gate is measured with numbers.
One gate fails **by the plan's own rule**: the 30-column permutation cost
(+0.0021) is indistinguishable from the ~0.003 seed-noise floor — the
core is too small for feature screening. Per the work order this is a
good measured outcome, not an unearned pass.

## Commits (jj, no push)

- `815c3424a85d395398c52912b6da709f62ba5905` — joint-core-v1 build
  pipeline + tooling (tools/joint_core/*, zensim-bench examples,
  DvifmSink cap/f16/hist, zen_decode cap, flag_confirm, prereg
  amendment 1, clippy fix).
- `b9ee7da96247dcae029ce81b419d6dbd1d61db4f` — record:
  `benchmarks/joint_core_v1_2026-09-20.{md,json,pointer.md}`,
  `docs/DATA_SPLITS.md` addendum, `board_discussion_sets.json` entry.

Dataset root: `/mnt/v/output/zensim/joint-core-v1/` (11 GB ≤ 25 GB cap;
`/mnt/v` left with 88 GB free). `_MANIFEST.json` at the root carries
build commits, 32 per-input sha256s, cluster/seed rules, kernel
provenance per leg.

## Coverage (gate 4 — PASS, measured)

52,963 pairs. Bands: mid 55.92% (≥55), small 24.42%, tiny 19.66%;
rung ≤1024 px. Classes: photo 81.63% (≥75), screen/doc/lineart 5.62%
each (≥5), AI 1.51% (<5). Legs: fresh_imazen26 62.2%, cid22 14.2%,
human 9.6%, fresh_safesyn 8.7%, hdr 4.7%, konfig 0.6%. Per-codec, rung,
decile tables in `coverage_report.json`.

## Gates 1–3 (measured)

1. **Leader reproduction:** core best-val geomean3 0.9696/0.9715/
   0.9711/0.9729/0.9718 (seeds 17101/03/07/11/13) vs frozen R915
   0.9788/0.9783/0.9775/0.9784/0.9778. Mean gap **−0.0068** (range
   −0.0055..−0.0092) — above seed noise, consistent with the ~3.3×
   smaller train pool; all seeds healthy. Stated, not excused.
2. **Convergence:** val(geomean3) ep50→ep99 +0.0018/−0.0018/+0.0216/
   +0.0002/+0.0016 (mean +0.0047) — flat-to-rising. The ep100 dip on
   17111 is the recipe's LR-restart transient (identical dip in frozen
   R915: 0.9743→0.9710). **PASS** — no sustained decline.
3. **Permuted-column control:** 30 used cols (f64..f93) row-permuted in
   train+dev; cost per seed −0.0005/+0.0010/+0.0024/+0.0029/+0.0048,
   mean **+0.0021 ≈ noise floor 0.003 → FAIL — core too small; grow the
   target before any feature screen.** Same signature as screen2c's
   codec panel at 373 rows; at 53k the measurement exists but the
   sensitivity does not.

## DVIFM constants fit (native3, Y′CbCr)

- Caches: 50,463 SDR rows × 3 planes, cap 1024/row, f16, + full-pop
  256×256 (C̃,m) histograms. Extraction ~195 s/plane, 0 failures.
- Histogram ≡ per-block grid-sum to **4e-16** on matched populations
  (400-pair seeded check); binning error 5e-4–3e-3 (256 bins).
- Fit @3,154 rows (seeded uniform, mean(y)=51.4): MSE 530.1→335.3,
  SROCC 0.7296→**0.8401**, KROCC 0.6531, wall 1448.5 s — over the
  ≤10-min gate. β runs to ~77 / C₀ to grid floor on most cells —
  recorded edge+sharpness diagnostics; masking-as-gate finding per the
  guards doc.
- Budget variant @788 rows (stride-64 uniform): **206.1 s — inside the
  gate**; SROCC 0.8190. Row count is the sanctioned lever.

## One-hour budget (measured)

generation ~21 min + 944-extraction 246.6 s + DVIFM caches ~9.8 min +
5-seed MLP ~8.7 min + constants fit 3.4 min ≈ **47 min ≤ 60**.

## Kernel finding for existing renditions

clean-picker-corpus-2026-06-26 = **PIL Lanczos → inadmissible**,
re-rendered; sdr-fps-1p5gp = zenresize Mitchell+sharpen, png-v1-sourced
and >1024-px rungs → not reused. Fresh legs: plain Mitchell sharpen=0.
cid22/human/konfig: native no-resample. hdr: Mitchell resize_sharpen=10
on linear PQ (recorded). 160 fresh cells are rung≥source passthroughs.

## dHash audit

21 flags vs CID22-49/AIC-3/AIC-4/AIC2026/SDR25/KonJND-val/KonFiG-test;
**all adjudicated false-positive** by pixel RMSE/NCC (min RMSE 50,
max NCC 0.61); AIC2026-S14 is the documented degenerate flat-page hash.
`audit/` in the evidence root.

## NOT done

- No eval-corpus label reads, no holdout fits, no CID22-B.
- HDR leg is PQ-era and excluded from SDR tables/caches by design.
- The constants' boundary pile-up is flagged but unresolved — the
  guards doc prescribes shipping the simplified (gate/flat) masking
  form; that decision is for the constants study, not this build.
- Feature screening deferred — the measured verdict is **grow the core**
  (gate 3); recommended next size ≥150k pairs or a 2× photo leg before
  any column screen.
- Hygiene: `cargo fmt --all -- --check`, `just clippy`,
  `just lint-scripts` all clean at HEAD (one pre-existing
  field-reassign lint fixed in `feature_v2.rs` test).
