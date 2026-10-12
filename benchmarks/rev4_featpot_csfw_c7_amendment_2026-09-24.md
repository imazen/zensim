



## Authority and reason



## New arms (added to the plan §3 arm list)

| arm | columns added to R0 (Rev3 944) | permuted control |
|---|---|---|
| R0 + csfw | f944-f955 (12) | the same 12 columns permuted |
| R0 + C7 | f956-f985 (30) | the same 30 columns permuted |

- **Permuted control (per arm):** the arm's added columns are permuted **jointly across pair rows within each reference group** (rows moved together by unique `pair_key`, so collapsed stimulus copies stay together), size-matched to the real arm, same fixed seed procedure as the C1-C4 amendment's controls. Its incremental delta versus R0 must have a paired reference-clustered 95% CI including 0, otherwise that dataset's instrument is declared insensitive for the family and no family conclusion is drawn there.
- **Sets, folds, models, seeds, statistics, roles:** identical to the C1-C4 amendment — the same D1/D2 sets, 5 outer / 4 inner reference folds, the 50-lambda lasso, sign-masked BVLS and H32/H128 non-negative-head MLP, 5x5 Latin-square seeds, 200 half-reference stability subsamples, 2,000 reference-clustered bootstrap resamples (seed 20260923), `panel`/`zen_stats` as the only statistic owner. Sign masks for the new IDs come from the registry `direction` and are written into the pin note below before any label is read. Forbidden and secret sets stay unread. Arms of different feature revisions never share a table.
- **Decision bar:** the D5 bar as amended by `61d3c07a050d` — (1) nested-CV gain ≥ +0.005 SROCC with a paired 95% CI excluding zero on ≥ 2 human sets; (2) stability-selection frequency ≥ 0.6 on those sets; (4) no non-negative-head dial-contract regression. **No cost clause.** Cost (extractor wall time, pairs/s, memory) is measured and reported for these families where a measurement exists and never accepts or rejects.
- **Composition note:** the registered "R0 + all C" arm keeps its C1-C4 composition; csfw and C7 are reported as their own arms and are not folded into it.





- **Presence rule (decided from column metadata only, before any label read):** when the sidecar and its manifest/registry entry exist, this lane checks whether the 30 C7 columns carry both visibility forms. Only if **both forms are present as distinct, identified column sets** are two arms run: **R0 + C7[curve]** and **R0 + C7[gate]**, each with its own size-matched permuted control, under the same sets, models, seeds and D5 bar as above. The pair is compared to each other (paired reference-clustered CI on the difference) as well as to R0.
- **If only one form is present, or the form cannot be identified from the manifest, B1 is recorded as MISSING** in the report, naming what is absent. The lane will not derive, recompute or infer the other form.

## Run condition and pinning
