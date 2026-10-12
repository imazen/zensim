# featacc WORKLOG — accuracy of candidate arithmetics across all 18 Rev4 families (2026-09-26)

Lane: `quarantine/devin/featacc`, workspace `~/work/zen/zensim--featacc`
(based on bookmark `quarantine/claude/featcanon-fix` = `13587bed`).
Brief: `~/tmp/devin/featacc/BRIEF_featacc.md`. Question: *"which is more
theory-grounded and math-correct"* — the featcanon lane shipped c32 on the
fused-V-blur pools + csfw only; the review (REVIEW_FEATCANON.md D7) showed
13/18 families had no candidate variation and the oracle did not cover their
kernels. This lane extends both to all 18 families and adds the
blur-recurrence axis the review named.

All scratch/output under `/var/tmp/featacc/` (never /tmp); builds through
`~/tmp/devin/heavy`, `CARGO_TARGET_DIR=/var/tmp/featacc/target`.

## Candidates and axes (oracle-gated; product build bit-unchanged)

`ZENSIM_FEATCANON` (existing): `off|c32|c64|neum|exact`.
`ZENSIM_FEATCANON_BLUR` (new axis): `rec` (production f32 sliding window) |
`rec64` (same sliding recurrence in f64) | `fresh` (per-window re-summed f64).
Default: `exact`→fresh, others→rec. `TIER_AUDIT_TAG` overrides dump names.
`ZENSIM_ERA2_DENSE=0` gives the era-1 dense baseline (`prod.era1`).
Every measurement is `research::extract(Request::everything())` at
`ZENSIM_FORMULA_REV=4`, tier v3, `RAYON_NUM_THREADS=1`.

## What was added (all `#[cfg(oracle)]`-gated or `Seq`-identity in product)



## Verification before measurement



## Corpus (reviewer probe set, `/var/tmp/review-featcanon/probe/pairs.tsv`)

12 pairs, all non-identical, sizes 8×8 … 2048×1536 mosaic:
kadid_I02_03_03 (512×384), konjnd_SRC0510_030 (640×480),
konfig_SRC06_j2k8 (384×512), 5 crops of kadid_I24_08_04 (8×8, 17×9, 64×64,
97×63, 131×65), kadid_mosaic4x4 (2048×1536), tid_I14_11_1 (512×384),
lanepair_tid512x384, lanepair_konjnd640x480. Roles: small crops exercise
tiny-n paths; the mosaic is the large-image stress; the rest are natural
mid-size TRAIN pairs.

## Numbers (dump: `/var/tmp/featacc/dump`, analysis: `/var/tmp/featacc/analyze2.py`, table: `/var/tmp/featacc/family_tables.tsv`)

Worst max relative error vs `exact` (rel floor 1e-9), 12 pairs, Rev4, v3:

- **Blur recurrence is the dominant structured error**, not accumulation.
  `exact.blur-rec` (f64 elements + Neumaier + production f32 sliding blur)
  still shows append 3.71, tailhist 1.0e-3, z1max 1.1e-3, gridblk 6.5e-3,
  v2 1.35e-3. `exact.blur-rec64` ≡ `exact` to ≤~1e-12 rel (85/21900 slots
  differ at f64-ULP level) — the f64 sliding recurrence is effectively
  exact; the f32 *storage* of the running sum is the whole blur error.
- **Accumulation order matters visibly only on csfw**: c32 2.70e-4 /
  c64 = neum 4.11e-5 (6.6×, same direction as the reviewer's 18× on the
  earlier set). Everywhere else c64/neum and c32 differ bitwise on ~15k/22k
  slot-values but at magnitudes *below* the element-error floor.
- **Element precision (f32 terms/stores) bounds everything else**:
  `*.blur-fresh` columns — append 0.57, tailhist 2.90, ringbasis 0.94
  (exact-zero bin), arttype 2.3e-3, gmsbank/gmsnative ~1e-3-1e-4,
  gridblk 0.66 — identical for c32/c64/neum → residual is element error.
- `v2_ssim_dev4` cancels catastrophically (raw-moment expansion
  `raw4−4μr3+6μ²r2−3μ⁴`): on the 17×9 crop m4 straddles 0 — c64/neum emit
  exactly 0 where exact emits 3.76e-4 (rel 1.0). A formula-conditioning
  issue, not accumulation order — only `exact`'s f64 elements keep the
  sign positive on every pair.
- `prod.era1` (era-1 dense) vs `prod` differ on 3405/21900 slot-values;
  era-1's f64 row reduction is slightly better on v2's worst slot
  (0.203 vs 0.306 rel) — same magnitude class.
- prod vs off: 3911 slot-values differ — the shipped Rev4 canon changed
  ~18% of emitted values relative to pre-canon arithmetic.

## Cost (zenbench, `benches/featacc_extract_ab.rs`, interleaved, pinned core 9, single-thread, Rev4)

Fit: t ≈ fixed_ms + slope_ns/px (slope from 1024²→4096², fixed = 64² residual).
Full output `/var/tmp/featacc/featacc_bench*.txt`. Contention note: shared
box, run under `~/tmp/devin/heavy`; load ~11; drift flags on some groups —
slope differences ≥5% are trustworthy, smaller deltas are noisy.

- off 415 ns/px, prod (shipped Rev4 canon) 659 ns/px — canonization cost
  ~+59% slope over pre-canon at these sizes.
- c32 808, c64 783, neum 1261, exact 2757 ns/px.
- Blur axis under c32 (focused run 3, `/var/tmp/featacc/featacc_bench3.txt`,
  256²/1024²): c32+rec64 57.3ms/0.91s vs c32+rec 95.5ms/1.46s vs
  c32+fresh 136.5ms/2.54s. `rec64` < `rec` is a measurement-path artifact:
  the axis arm streams row-major over full-width sums while the shipped SIMD
  kernel sweeps 8-column groups vertically (strided; the plane exceeds L2 at
  ≥256²). `fresh` +40–75% as expected for O(diam) windows. No 4096²
  blur-axis measurement (zenbench wall-time cap; 0 rounds in run 2).
- 4096² rows are single-round (cap) — slope fit leans on ≤1024² rounds.

## Recompute

- Accuracy: `ZENSIM_FORMULA_REV=4 ZENSIM_FEATCANON=<mode> \
  [ZENSIM_FEATCANON_BLUR=<rec|rec64|fresh>] [ZENSIM_ERA2_DENSE=0] \
  ZENSIM_FEATCANON_DUMP=<dir> TIER_AUDIT_ONLY=v3 \
  <oracle-build>/tier_audit_features /var/tmp/review-featcanon/probe/pairs.tsv`
  then `python3 /var/tmp/featacc/analyze2.py <dir>`.
- Cost: `RAYON_NUM_THREADS=1 ZENSIM_FORMULA_REV=4 taskset -c 9 \
  cargo bench -p zensim --bench featacc_extract_ab --features \
  custom-profiles,feature-regime-v2,threads,training,oracle`.
