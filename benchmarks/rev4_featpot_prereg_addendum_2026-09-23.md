# Rev4 feature-bank potential preregistration ADDENDUM (2026-09-23)

**POTENTIAL — ceiling, not a model score.** This addendum is committed before this resumed lane decodes any bank label value. It supplements preregistration `d2169f5b`, after the Opus promotion of the Rev3 f32 bank in `/home/lilith/tmp/zensim-paper/rev4/REVIEW_FEATBANK_EXTRACT.md`. The addendum fixes the baseline input identity and registers the GMSD arms before any human-label read. Candidate sidecars are not available yet and their arms run later.

## Promoted input and label admission

Bank root: `/var/tmp/rev4-featbank/bank/`; `_MANIFEST.json` SHA-256 `6cce682dc4bc847507ba13a75c3ff5b8df91eb2dbdc924bc2658d2afba70440d`. Producer identity: `basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#b782e349`, 905 populated f32 feature columns plus 39 structural zeros. Each source uses `keys.parquet`, the feature sidecar named by its manifest, and **only** the role-allowed `labels__*.parquet`. Manifest file hashes bind each component. Join exclusively on unique `pair_key`; preserve `n_stimuli` multiplicity for collapsed pairs and reject missing/duplicate keys. The f32 values are promoted as the exact f32 casts of Rev3 f64 source values; the old mixed-era CSVs in the first prereg are superseded as inputs.

| Bank set | Stimuli / keys | Set-manifest SHA-256 | Role and target scale |
|---|---:|---|---|
| `kadid_train` | 5,000 / 4,880 | `02b00e89e09cfb6dd2bda0876e1884186a1d0398d95f9592f8395bf1591577c8` | TRAIN human `human_score`; retain native quality scale until fold-local transform |
| `tid2013` | 3,000 / 3,000 | `eb8fa95b9b439a3c7e80375fba17880896ee11fa19b5e7c796ecc08ab5de3e79` | TRAIN human `human_score`; native quality scale |
| `konfig_train` | 327 / 327 | `839819bfd761d8c45bbfa5dcc90e8e73dc9afaf58a097e4183a8dd06ca5fa0af` | TRAIN human-derived quality `1 - q_jnd/3.2` |
| `konjnd_bpg_train` | 8,060 / 8,060 | `1d0ecfb6e8117d162b521851e86c12678c35a731b09022760dda7f5ed3836abb` | TRAIN **SSIMULACRA2 oracle**, `ssim2_oracle = raw ssim2 / 100`, not a human-label accuracy set |
| `konjnd_bpg_val` | 2,020 / 2,020 | `c04360e73b23f4e6e3786c7fb4173fea8280388051f4af0e44f4328e5c228331` | D2 BPG fold evaluation; same oracle /100, reference-disjoint from BPG TRAIN |
| `cid22_a25` | 2,192 / 2,192 | `daa46bb04481add3534ebee51a8d0fede0139a58d1812c227dc55d0791a405d3` | D1 potential + D2 fold; **human MCOS/100** in `labels__human.parquet`, never an oracle |
| `aic3` | 600 / 600 | `a736a3c0ed3185f393577537c37068044c90875c0fb0b2553ae4270aa840bbd4` | D1 potential + D2 fold; human CTC quality |
| `kadid_select` | 3,125 / 3,050 | `6676851ee03d1bd89fe7f51f51ebe20203e9aa4582d47519982c3e8dbf903124` | D1 potential + D2 fold; human quality, preserve collapsed-key multiplicity |
| `konfig_val` | 436 / 436 | `1285e50a5669a791bdb5ea5587d5447f43413634461769ad98085f5644679e06` | D1 potential; D2 KonFiG fold evaluation; `1 - q_jnd/3.2` |

The D1 in-sample reporting populations are `kadid_train`, `tid2013`, `konfig_train`, `cid22_a25`, `aic3`, `kadid_select`, and `konfig_val`. The provisional D2 rotation is KADID TRAIN, TID, KonFiG TRAIN (with VAL as its reference-disjoint eval view), KonJND BPG TRAIN (with BPG VAL as its reference-disjoint eval view), CID22-A, AIC-3, and KADID SELECT. Dataset-level LODO training weights are equal; no KonFiG or BPG VAL row enters a fit unless a separately registered D1 in-sample diagnostic explicitly fits KonFiG VAL. The BPG target is an oracle and its fold is identified as oracle transfer, never a human generalization result. No CID22 TRAIN or SafeSyn oracle row enters a human LODO fit. If either is used in a separate control, `ssim2_oracle` is **raw 0–100** there, unlike BPG's /100; each set's unit is declared and converted before pooling. `cid22_a25` is **human MCOS/100**.

Before each held-out label read, update `docs/DATA_SPLITS.md` with the exact population, row count, purpose and `potential-exposed` or `LODO-exposed` status. Read label values **only** from these role-allowed bank `labels__*.parquet` files. Never read held-out `human_score` from `/var/tmp/rev4-featbank/pairs/`, `/var/tmp/rev4-featbank/raw/`, or their copies under `_sealed/`. CID22-B, AIC-4, KonJND JPEG, CSIQ, KADID TERMINAL, LIVE and secret holdouts remain unread. MCL-JCI remains outside every fit pending D3.

## Arms, controls, seeds, statistics and decision

The existing-family and C1–C4 candidate-family arms from prereg `d2169f5b` remain: R0 (Rev3 944), R0−F for each ledger family, R0+C1, R0+C2, R0+C3, R0+C4, R0+all C1–C4, and basic228. Each family gets its within-reference size-matched permuted-column control. C1–C4 and the following GMSD arms wait for reviewed Part B sidecars. This run executes **P0/R0 bank baseline** and the available baseline positive control only.

GMSD arms are preregistered now, exactly as GMSBANK item 7 proposes:

| Arm | Columns added to P0 | Permuted control |
|---|---|---|
| P0 | none: bank-944 baseline | sham 0-column permutation; use the R0−basic positive control to prove sensitivity |
| P1 | `gmsbank` f1322–f1501 | permute the 180 added columns within reference groups |
| P2 | peer GMSD `gmsd`, `gmsm` | permute the 2 added columns within reference groups |
| P3 | both P1 and P2 | permute the same 182 added columns within reference groups |

Each candidate arm and its control use identical D1/D2 rows, folds and seed protocol. Compare a permuted arm to P0; its incremental delta must have a paired 95% reference-clustered CI including zero or declare that dataset's instrument insensitive. P1/P2/P3 are **not** fitted until their reviewed sidecars exist. GMSBANK's peer columns are diagnostic controls, not a shipped feature claim.

The D5 adoption bar remains: stability-selection frequency ≥0.6; nested-CV gain ≥+0.005 SROCC with paired 95% CI excluding zero on ≥2 **human** sets; measured runtime cost within budget; no non-negative-head dial-contract regression. The same 50-lambda lasso, sign-masked BVLS and H32/H128 non-negative-head MLP, 5 outer/4 inner reference folds, 200 half-reference stability subsamples, 2,000 reference-clustered bootstrap resamples (seed 20260923), 5×5 Latin-square init/sample seeds, and `panel`/`zen_stats` statistic owner from the original prereg apply. Fold-local target normalization retains the original prereg's per-dataset q0.001/q0.999 affine rule; no raw 0–1/0–100 mixture is allowed.

Interpretation, fixed before labels: **P1 ≈ P3 > P0** means the bank captured GMSD's useful information; **P2 > P1** means the gamma domain or gradient operator matters; **neither P1 nor P2 > P0** means no human-label gain. These are conditional interpretations only after the controls, CIs, D5 bar and data-role checks pass. They do not turn a potential ceiling into a model score.

## 2026-09-23 23:07 UTC manifest-metadata repin, before any label value read

The extractor lane applied the review's manifest corrections while this lane was preparing admission. The first adapter invocation refused `kadid_train` on its manifest-hash check **before opening any Parquet file**. The current root manifest SHA-256 is `813cd9204912df55aedf9967f45fcb6a368a6ba9867271c00a68c3ad529268c3`. The nine per-set manifest SHA-256 values, in the table order above, are now:

`6976a3ce05fa52552883ac380c85bdfd4cbeb800be23077299069d2b78769b30`, `ec305b5e610dc36815a91c7faf533dd1000cdba5f4faa59a14f4ab924bb69c1d`, `2a48a0c3f3686dce24ef9e1288cec0907df17531df32f345d91eda0b1ad8e405`, `f0badfe8d26e69fb79839458749c385b32d04734e36116bd5c40f25749046292`, `59f7099930cde8b2899ad3f76f826a7e12f3615938356fa370ec0ad79d9c2cf9`, `38474bc10a3ad6d89965dcf3cabdc10e7fb677b0ce76606ccd9e73c4ef9009f4`, `e73441296ec64433198a537b5fe0dd340003a3bced446848a8b38c4ba22dc389`, `c8807a25089e0d354c8cff068477edca410b33cc82085ddc2ec86930af24183b`, `b2fc90e103312ec86968ede3fc0d820b4dc45a48df7e37c3bebd4efff945605e`.

Only the manifest hashes are repinned here. The adapter still verifies the current manifest's hashes for keys, features and role-allowed labels before decoding. The review's bank promotion and every role, arm, scale and exclusion above are unchanged.

## 2026-09-24 P2 peer gate and sign-mask clarification, before P2 fitting

The coordinator prioritised P2 as soon as `REVIEW_GMSBANK.md` gave a PROMOTE verdict covering the peer GMSD/GMSM columns. That review is now `PROMOTE WITH CORRECTIONS`; it independently verified the two columns, 18 sets, 249,227 stimulus rows, and zero key mismatches. P2 uses only the reviewed `/var/tmp/gmsbank/peer_gmsd/<set>.parquet` files plus the pinned Rev3 bank. P1 and P3 still require reviewed Part B C8 sidecars. The reviewer-required GMSBANK documentation/code corrections do not change the peer values; P2 remains a diagnostic peer-control arm, never an adoption claim.

P2 adds the **raw** `gmsd` and `gmsm` values as feature indices 944 and 945 to the 944-column baseline in that order. Its two-column permuted control moves the two values together under one seed-20260923 permutation of unique `pair_key`s within each reference group, then expands collapsed stimuli. It uses the identical D1/D2 rows, folds and target transforms. Standalone GMSD is scored as `-gmsd` against quality-oriented targets; standalone GMSM is scored as `+gmsm`. No standalone peer score enters fitting or selection.

For P2 BVLS, the existing 372-column sign-mask file applies unchanged to f0–f371 and all f372–f943 remain free, as in P0. Both peer columns are explicitly **free**: raw `gmsd` rises with distortion while raw `gmsm` rises with quality, and the Rust BVLS mask format supports only `pin_geq0` or `free`, not a negative pin. Their fitted coefficients and non-negative-head dial diagnostics must be reported before any interpretation. The P2 linear lasso is unconstrained. This mask decision is frozen before P2 model fits or score inspection.

## 2026-09-24 D2 MLP checkpoint protocol, before any D2 MLP fit

For each of the seven D2 held-out source sets, the H32/H128 MLP fits on the other six. Inner checkpoint selection leaves each of those six source sets out once, trains on the remaining five, and averages the six validation `geomean3` values at epochs 0, 5, …, 55. Select the earliest epoch attaining the maximum mean, then refit on all six sources and score only the held-out set's registered evaluation view (KonFiG VAL and BPG VAL are the two reference-disjoint substitutions). Each training source receives equal draw share by setting its `--group` train weight to 1 under the trainer's default `uniform` pair sampler; unlike raw Gram weighting, no inverse-row factor is applied to an MLP group. Every source target is independently normalized by its own q0.001/q0.999 fitted on that source's TRAIN rows. The held-out target and its affine bounds never enter training, checkpoint choice, or the predictor input target column. The BPG transfer fold remains oracle agreement, not a human generalization result.

Use the registered five init seeds and five sample seeds. For D2 held-out fold index `o=0..6` in the fixed source order, pair init seed `i` with sample seed `(i+o) mod 5`; the full refit uses that same pair. Keep 60 epochs, 50,000 pairs/epoch, non-negative-distance head and five-seed mean/min/max/spreads. Score each fold and the five-seed mean with `panel`, with B=2,000 paired reference-clustered resamples (seed 20260923). Apply the identical D2 protocol to P0, P2 and P2's matched permutation if their MLP D2 arms are run; compare P2 and permutation to P0 on the same fold rows and resample indices. This clarification adds no new label population or candidate feature definition.
