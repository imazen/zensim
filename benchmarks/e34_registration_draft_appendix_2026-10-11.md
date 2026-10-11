# E34 registration draft — appendix (revisions 3–4, 2026-10-11)

Linked from [e34_registration_draft_2026-10-10.md](e34_registration_draft_2026-10-10.md). Everything here is
label-free stored evidence or historical context. Nothing here is a rule.

## A1. Seed-0 failures, per case

- **Without worst block:** M2 after dropping the block with the largest rank displacement.
- **Rel. L2:** ‖linear − true‖ / ‖true‖ in served units.
- **true > linear:** blocks breaking the concavity inequality in raw units (E33C_STEER PreFloor rows; A and C only).

Computed from the stored files outside `characterize.py`; the round-1 review verified them independently.

| Model | Case | Block | Image (origin) | Level | Blocks | M2 | M3f | Without worst | Rel. L2 | true > linear | Base |
|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | broad-157-b8 | 8 | 9380 product photo on white | 10 | 704 | .9795 | .9295 | .9814 | .037 | 277/704 | 92.8 |
| A | broad-220-b16 | 16 | 8384 text screenshot | 10 | 192 | .9868 | .9522 | .9901 | .037 | 107/192 | 90.5 |
| A | broad-10-b32 | 32 | 2010 dense photo | 10 | 56 | .9682 | .9699 | .9764 | .221 | 39/56 | 88.1 |
| A | broad-10-b64 | 64 | 2010 dense photo | 10 | 16 | .9353 | .9324 | .9643 | .426 | 6/16 | 88.1 |
| A | broad-136-b64 | 64 | 7066 line art | 10 | 16 | .9794 | .9794 | .9964 | .050 | 7/16 | 86.8 |
| A | broad-76-b64 | 64 | 6068 text scan | 13 | 16 | .9882 | .9971 | .9964 | .230 | 0/16 | 81.8 |
| C | broad-157-b8 | 8 | 9380 product photo on white | 10 | 704 | .9893 | .9030 | .9913 | .042 | 389/704 | 93.3 |
| C | broad-160-b32 | 32 | 9380 product photo on white | 13 | 48 | .9843 | .9690 | .9996 | .034 | 32/48 | 83.8 |
| C | broad-202-b32 | 32 | 8206 web screenshot | 13 | 40 | .9662 | .9670 | .9820 | .165 | 29/40 | 84.9 |
| C | broad-34-b32 | 32 | 1054 architecture photo | 13 | 48 | .9502 | .9468 | .9562 | .360 | 8/48 | 79.0 |
| C | broad-136-b64 | 64 | 7066 line art | 10 | 16 | .9882 | .9853 | .9964 | .022 | 16/16 | 88.2 |
| C | broad-202-b64 | 64 | 8206 web screenshot | 13 | 12 | .8601 | .9371 | .9273 | .220 | 7/12 | 84.9 |
| C | broad-31-b64 | 64 | 1054 architecture photo | 10 | 12 | .7273 | .7273 | .8182 | .351 | 0/12 | 90.6 |
| C | broad-34-b64 | 64 | 1054 architecture photo | 13 | 12 | .9720 | .9720 | .9909 | .378 | 1/12 | 79.0 |
| P | broad-206-b32 | 32 | 8206 web screenshot | 17 | 40 | .9698 | .9525 | .9826 | .114 | — | 57.5 |
| P | broad-206-b64 | 64 | 8206 web screenshot | 17 | 12 | .9441 | .8531 | .9545 | .146 | — | 57.5 |
| P | dog-256-20 | 32 | M3 fixture (dog, JPEG q20) | — | 64 | .9872 | .9857 | .9887 | .200 | — | 41.7 |
| P | I01/I21/I41/I61_10_05 | 8 | KADID JPEG | — | 3,072 | 0 | 0 | — | — | — | −16.9 |

Production's four KADID JPEG failures are floor ties (STEERFIX "CODE"); E33's identity knot and tail remove them for A
and C. All 44 A/C failures are M2 misses (A seed 1 broad-76-b64 also misses M3f), all "MODEL" by E33C_STEER.

## A2. Descriptive counts (`summary.json`, seven served models)

These are counts on a fixed engineering panel with few source pictures. §1.3 of the draft gives the clustered
intervals for the comparative claims.

| Item | Value |
|---|---|
| Failures by block (all / non-floor) | b8 10/224 (6 non-floor), b16 6/168, b32 12/385, b64 23/168 |
| Drop-one rescues (non-floor) | b8 2/6, b16 1/6, b32 3/12, b64 12/23 |
| Spearman swap budget at M2 ≥ .99 | Σd² ≤ 2.86 at n = 12, ≤ 6.8 at n = 16 (one or three adjacent swaps) |
| Broad failures by reference | 7066 10/84, 8206 9/84, 9380 9/84, 1054 4/84, 2010 4/84, 6610 4/84, 6068 2/84, 8384 2/84 |
| Failures by JXL level (broad) | 10: 21/224, 13: 13/224, 17: 10/224; median base 91.4 / 80.6 / 38.9 (seed-0 A, C, production) |
| Seed overlap | 36 failing (case, block) pairs; 9 fail in two or more models; broad-136-b64 in 5 of 7; counts 124–131 |
| Median Rel. L2 failing / passing (all panels, non-floor) | b8 .042/.013, b16 .082/.017, b32 .160/.016, b64 .125/.021 |
| Concavity breaks (raw units, seed-0 A/C failures) | 12 of 14 cases have a block with true > linear; excess 0.4–4.9% of the case's largest gain (round-1 review) |

The content observation (three references carry 28 of 44 broad failures) names specific pictures. With 8 broad
sources it is descriptive and makes no claim about new images.

## A3. Training revision (by analogy; R5STEER2, E24 `by_v2fy` bakes, uniform 3-seed ensembles, same `N` head)

| Weights | Arithmetic | Broad passes | Owner case (I61, seed index 2, b32) |
|---|---|---:|---|
| Rev4-trained | Rev4 | 92/96 | Rel. L2 .004 at full repair |
| Rev4-trained | Rev5 | 93/96 | M2 .9997 |
| Rev5-trained | Rev5 | 85/96 | Rel. L2 .19, M2 .888 |

R5STEER2 also measured 9 of 10 failing cases passing at 1% of the repair and failing at the full repair; the owner
case's Rel. L2 grows .02 → .19. These are counts on one engineering panel, not clustered intervals.

## A4. Prior evidence on sign-constrained and linear scores (historical, descriptive)

These are pooled SROCCs from other heads, feature sets and populations. They are neither split by pair class nor
source-clustered, so they are context for the owner, not evidence under the 2026-10-11 guidance.

| Record | Model | Result |
|---|---|---|
| `v46_v46b_monotone_cbc_pareto_2026-05-26.md` | V46b, `monotone_cbc` | CID22 −0.035, KADID −0.12 vs V39 (0.9251) |
| `v47_masked_monotone_2026-05-27.md` | masked-strict | KADID 0.803 vs 0.925 |
| `minmax_monotone_2026-07-16.md` | positive-weight `monotone_cbc` MLP / masked-monotone linear | imazen-26 0.032 / 0.862 |
| `feature_sign_mask_2026-05-26.tsv` | 372 v1 features | 300 `pin_geq0`, 72 `free`; 33 of A's 130 v1 inputs `free` |

## A5. Why arm M was dropped (revision 2)

Revision 1's M projected W1 ≥ 0 on `HigherIsWorse` columns with nonnegative minima. All 410 inputs qualify (C's
products are nonnegative too), so with `b1 = 0` no unit could switch: M was L's linear class at width 128.

A partial M (the May sign-safe subset) is not admitted either:
- that subset comes from a correlation census, not value signs;
- the v2 inputs have no census;
- A4 shows rank costs with no measured steering benefit.

## A6. Reproduction

| Step | Command | Output (sha256) |
|---|---|---|
| Stored-row characterization | `python3 benchmarks/e34_draft_2026-10-10/characterize.py <out>` | `per_case.json` `6a6ba56b…`, `summary.json` `81d18131…` |
| Clustered intervals (zenmetrics-stats `1e48c1ea`, B 10,000, seed 20261011, type 7) | `cargo run --locked --release --manifest-path benchmarks/e34_draft_2026-10-10/clustered/Cargo.toml -- <per_case.json> <clustered.json>` | `clustered.json` |
| Student interval beside it | `python3 benchmarks/e34_draft_2026-10-10/student.py <clustered.json>` (scipy `t.ppf` for the constant only) | `clustered.json` `c8a1ba8f…` (revision 4; updated in place) |

Raw roots: `/mnt/v/output/zensim/e34-draft-2026-10-10/`, `/mnt/v/output/zensim/e34-rev3-2026-10-11/` and
`/mnt/v/output/zensim/e34-rev4-2026-10-11/` (checked on dev), all mirrored to tower. Revision 4's tool reproduces
revision 3's seven rows bit for bit and adds the picture counts, the within-case rows and A7's table.

## A7. Between-picture contrasts, by picture (revision 4)

Revision 3 read these as "holds" at G = 15. Like for like they are descriptive: the sides draw on different pictures.

| Contrast (all panels, G = 15) | Pictures failing / passing | Estimate | Percentile 95% | Student 95% |
|---|---|---:|---|---|
| Rel. L2 ratio failing / passing, b8 | 2 / 12 | 3.32× | [2.46, 7.44] | [1.63, 6.77] |
| b16 | 4 / 8 | 4.28× | [2.92, 7.49] | [2.54, 7.24] |
| b32 | 8 / 15 | 7.42× | [4.25, 12.86] | [4.00, 13.75] |
| b64 | 6 / 8 | 6.45× | [3.31, 15.86] | [2.65, 15.69] |

At b8, 5 of the 6 non-floor failing rows are one picture (9380). Draws with no failing picture at a block size are
dropped (b8 11.9%).

**Failures at b64 and at b8–b32, per broad picture** (seven served models, floor ties excluded):

| Picture | b8–b32 | b64 |
|---|---:|---:|
| 1054 | 1/63 | 3/21 |
| 2010 | 1/63 | 3/21 |
| 6068 | 0/63 | 2/21 |
| 6610 | 2/63 | 2/21 |
| 7066 | 3/63 | 7/21 |
| 8206 | 3/63 | 6/21 |
| 8384 | 2/63 | 0/21 |
| 9380 | 9/63 | 0/21 |

Six of eight pictures fail more at b64, and two fail less.
