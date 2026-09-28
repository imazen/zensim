# Preregistration — lane `cvvdp-safesyn`, Rev4 E2b (CVVDP display selection on TRAIN human legs)

Registered **before any human-label read in this lane** (2026-09-23, devin lane
`cvvdp-safesyn`; brief `/home/lilith/tmp/zensim-paper/rev4/CVVDP_SAFESYN_brief.md`;
fleet addendum `/home/lilith/tmp/zensim-paper/rev4/CVVDP_SAFESYN_FLEET_addendum.md`).

This document preregisters the E2a proposal **as written** in
`benchmarks/rev4_e2a_cvvdp_display_2026-09-23.md` Deliverable 4 ("proposed E2b
design"). Nothing below is adjusted after seeing data; the only additions are
operational bindings (file paths + sha256, RNG procedure, CI definition) needed to
make the run reproducible and auditable.

## Arms (7 displays, fixed)

| arm | ppd | peak cd/m² | role |
|---|--:|--:|---|
| `standard_4k` | 75.40 | 200 | control — every past CVVDP teacher value |
| `sdr_4k_30` | 60.55 | 100 | program list; lower ppd and peak together |
| `standard_fhd` | 37.84 | 200 | program list; AIC/AIC2026 organiser setting; geometry-only change vs 4K |
| `sdr_fhd_24` | 37.84 | 100 | pairs with `standard_fhd` to isolate peak luminance at fixed geometry |
| `standard_phone` | 120.56 | 500 | program list |
| `iphone_14_pro` | 159.61 | 1025 | program list |
| `modern_oled_phone_indoor` | 109.97 | 400 | program list; **exploratory until a pycvvdp parity check with `config_paths` passes** — if no pycvvdp reference can run, it is reported labelled "unchecked" |

Scoring is the parity-verified CPU port (`zenmetrics-cli`, quarantine branch
`quarantine/devin/cvvdp-safesyn`, rebased cvvdpfix trio 27c69beb/d71922bd/dc1fca78
copies). One named display per run; output column names carry the display suffix so
arms can never be averaged together.

## Selection data (the only data used to choose)

TRAIN-role views per `docs/DATA_SPLITS.md` §8:

| leg | rows | refs | pairs TSV (sha256) | label parquet (sha256) |
|---|--:|--:|---|---|
| KADID-train | 5,000 | 40 | `/mnt/v/output/zensim/reports/refmetrics/kadid_pairs.tsv` (`ceaf324f…`) | `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_kadid_train_2026-08-29.parquet` (`1fb8ff9b…`) |
| TID-train | 1,440 | 12 | `/mnt/v/dataset/tid2013/tid_pairs_ab.tsv` (`e6a790e8…`) | `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_tid_train_2026-08-29.parquet` (`269c7a8c…`) |
| KonFiG-train | 327 | origin-split | `/mnt/v/output/zensim/konfig944/build/konfig_pairs.tsv` (`44bde2d8…`) | `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/konfig_originsplit_train_944.parquet` (`b919b9ad…`) |

Row binding: each pairs-TSV row is matched to its label row by
`(ref_basename, distorted-path stem)` identity and the stored human label is carried
through verbatim; the binding is verified by row count, key uniqueness, and a hash
of the joined key sequence recorded in the worklog. All three legs are TRAIN-role
(memorised by several zensim eras — irrelevant to CVVDP, relevant to later zensim
comparisons). None has documented viewing geometry, so a selected display is the one
that best fits crowd/unstated conditions, and the report says so.

## Statistic (all through the canonical owner)

- Per leg, **pooled SROCC** and **within-image (per-reference) SROCC** through
  `zensim_validate::panel` via `scripts/lib/zen_stats.py` (`panel --input --per-group`
  aggregate mode; `panel --batch` for resamples). No stat math is reimplemented.
- Paired difference Δ = display − `standard_4k` with **reference-clustered
  bootstrap, B = 2000, seed 20260923**: `rng = random.Random(20260923)`; per resample
  `picked = [rng.randrange(G) for _ in range(G)]` over G references, expanded to row
  indices (draw order preserved); the SAME index set is applied to all 7 arms, so Δ is
  paired. Driver script `benchmarks/cvvdp_safesyn/e2b_stats.py` (committed with the
  report) records its own sha256.
- **KonFiG**: triplet ordering accuracy on its `q_jnd` scale through the same owner —
  `panel --pairwise` rows `(group=ref, s_left, s_right, choice, weight=1)` built from
  all unordered within-reference stimulus pairs with `q_jnd_a != q_jnd_b` (choice =
  side with larger q_jnd, i.e. more distorted; metric quality = JOD, higher better);
  reference-clustered bootstrap via `panel --pairwise --resample` with the same
  `random.Random(20260923)` group-index stream (B = 2000).
- CI: two-sided percentile at 1 − 0.05/6 = 0.991667 ("Bonferroni 99%", α ≈ 0.008
  over the 6 challenger arms), computed on the Δ resample distribution.

## Decision rule (registered verbatim from E2a)

A display *beats `standard_4k`* only if, on **both** KADID-train and TID-train,
pooled Δ ≥ +0.010 SROCC **and** the 99% cluster CI of Δ excludes 0 (Bonferroni over
the 6 challenger arms, α = 0.05/6 ≈ 0.008), **and** KonFiG-train accuracy Δ is not
significantly negative (its 99.1667% CI does not lie entirely below 0). If several
pass, take the largest mean Δ over the two SROCC legs. If none passes, CVVDP's
showing as a teacher stands and E2c does not run.

## Reporting

- Every display × every leg reported (pooled SROCC, per-reference SROCC, Δ vs
  `standard_4k` with CI) — report all arms, select per the rule.
- `modern_oled_phone_indoor` carries its parity-check status in the report.
- Deliverables: `benchmarks/rev4_e2b_cvvdp_display_2026-09-23.{md,json}`; every read
  logged in `docs/DATA_SPLITS.md` exposure ledger; commands + hashes in
  `benchmarks/cvvdp-safesyn_WORKLOG.md`.

## Exclusions (binding)

No CID22-B, no AIC-3/AIC-4 label reads for selection (Part 0 uses stored score TSVs
only, never labels), no secret holdouts, no held-out fitting or selection. Part 2
(SafeSyn, no human labels) is descriptive only — no fitting on its comparisons.
