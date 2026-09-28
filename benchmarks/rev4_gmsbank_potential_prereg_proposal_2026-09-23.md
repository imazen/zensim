# GMSBANK potential-arm preregistration proposal (2026-09-23)

**Proposal only for the `featbank-potential` owner.** This lane reads no human labels and fits nothing. The coordinator must merge this into the potential preregistration before that owner reads labels. The input is the promoted f32 bank `/var/tmp/rev4-featbank/bank/` plus the reviewed GMSBANK and exact-peer sidecars, joined strictly by `pair_key`. Input hashes, producer commits and feature-set identities must be pinned by the potential owner after promotion; a missing or duplicate join fails.

| Arm | Added columns to the same bank-944 baseline | Size-matched negative control |
|---|---|---|
| P0 | none | baseline only |
| P1 | Revised 2026-09-24 C8 f1322–f1501, 180 columns (native Y; coarse XYZ + joint CS) | 180 independently permuted GMSBANK columns |
| P2 | exact peer `gmsd`, `gmsm`, 2 columns | 2 independently permuted peer columns |
| P2b (optional, coordinator decision pending) | bank + reference-exact MDSI peer distance, 1 added column | 1 independently permuted MDSI column |
| P3 | revised P1 + P2, 182 columns | 182 independently permuted added columns |

Permutations are made inside the same TRAIN fold and reference group protocol as the potential owner, with seed `20260923`, never across train/evaluation boundaries. The control keeps column count and marginal distributions. Use the existing `zen_stats` or `panel`/`bake_verdict` statistic owner for SROCC, ties and reference-clustered bootstrap confidence intervals; do not write a new statistic implementation. Use 10,000 reference-clustered bootstrap resamples with seed `20260924` and the owner's established percentile-CI convention. The potential owner's existing lasso, linear projection and optional MLP capacities and seed-selection rules are identical across arms. Report the best permitted fit within each arm under those frozen rules and compare to its own size-matched permuted control.

## Data roles (ruling 2026-09-23, `docs/DATA_SPLITS.md`)

**D1 in-sample potential:** CID22-A(25), AIC-3 CTC, KADID SELECT and KonFiG originsplit val are fitted in-sample and become potential-exposed. They must never be described as held-out evidence for a model chosen using this run. **D2 LODO folds:** KADID, TID, KonFiG, KonJND-BPG, CID22-A(25), AIC-3 and KADID SELECT rotate as the held-out fold, with models quarantined under `LODO_`. MCL-JCI enters folds only if the pending D3 ruling gives it a fitting role; otherwise confirmation-only. CID22-B(24), AIC-4 sample, KonJND JPEG SELECT/TERMINAL, CSIQ, KADID TERMINAL, LIVE and secret holdouts remain outside every potential fit/fold as specified by D2. Pixel-only sidecar extraction under D4 does not authorize label reads. Record any later human-label read in the exposure ledger.

## Decision rule

The D5 family adoption bar applies: stability-selection frequency ≥0.6; nested-CV SROCC gain ≥+0.005 with a reference-clustered CI excluding zero on at least two human sets; benefit pays its measured runtime; and the non-negative-distance head preserves the dial contract. A gain over a permuted control is necessary diagnostic evidence, not alone an adoption pass. The exact peer is a control for information content and operator/domain, not a candidate that silently joins the bank.

| Result pattern | Interpretation to test |
|---|---|
| P1 ≈ P3 > P0, and P1 beats its permutation | The bank captured the useful GMSD information. |
| P2 > P1, with P2 beating its permutation | Gamma-luma domain, Prewitt operator or 2× box sampling carries information the XYB bank misses; the arm comparison alone does not identify which. |
| Neither P1 nor P2 beats P0 and its own permutation | No measured gain on these human labels under the frozen fit protocol. |
| P3 > both P1 and P2 | Complementary information; inspect selection stability and runtime before any adoption. |

No interpretation is a claim until the potential owner runs the preregistered analysis and its review promotes the result.


## Chroma revision proposal — 2026-09-24

P1 now refers exclusively to the revised C8 era registered by the gmsd-chroma
lane. Prior C8 sidecars are incompatible even though both layouts are1502.
Optional P2b means the common bank baseline plus the reference-exact MDSI
peer **final distance** column (one scalar, lower is better); it is not
MS-GMSDc or C8's value-map summaries. The canonical zenmetrics gmsd crate
produces that peer, with its author software used only as an oracle.
The coordinator must decide P2b's admission, join hashes, orientation and
control before the potential owner reads any labels. This proposal performs
no potential fit and authorizes no additional label exposure. If a richer
set of MDSI peer columns is desired, its exact definition and count require
a new pre-label proposal; do not infer unnamed maps or intermediate columns.
