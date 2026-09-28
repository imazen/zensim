# Rev4 POTENTIAL amendment — cost is reported, never a gate (2026-09-24)

**POTENTIAL — ceiling, not a model score.** Supplements preregistration `d2169f5b`, addendum `044f00dc`, addendum `bf553563` and the C1-C4 amendment. Written at 2026-09-24T21:04:58Z by Claude Sonnet lane `featbank-potential` (continuing the paused Codex lane), before any candidate-arm result was read: `PARTB_C1C4_DONE.md` is a stop report, 0 `features__rev4c1c4.parquet` sidecars exist, and no C1-C4, all-C, P1, P2-MLP-comparison or P3 label-bearing result has been opened by this lane. (P2's existing deterministic fits were read earlier and are covered by the P2 section already in the report; this amendment changes only how their cost is used.)

## Authority

User directive relayed by the coordinator, 2026-09-24: *"remember not to reject things for the cost budget, and track all things and code and results of those you have. we can optimize and make things optional"*.

## Change

Decision rule clause 3 of plan §3 ("it is within its §2.4 budget, measured"), and its restatements — the addendum's "measured runtime cost within budget" and the P2 "no D5 cost gate without a prospective budget" note — are **removed from the D5 bar for every arm**: C1, C2, C3, C4, all-C, P1, P2, P3, and any arm added later.

- The remaining D5 conditions are unchanged: (1) nested-CV gain ≥ +0.005 SROCC with a paired reference-clustered 95% CI excluding zero on ≥ 2 human sets; (2) stability-selection frequency ≥ 0.6 on those sets; (4) no non-negative-head dial-contract regression.
- "TRAIN-supported, unconfirmed" now means conditions 1-2 (and 4 where evaluated) are met only on TRAIN-role sets.
- **Every candidate arm runs and is reported whatever its cost.** No arm is skipped, truncated or down-ranked for exceeding a §2.4 or any other budget.
- Cost is still **measured and reported per family** (extractor wall time ST/MT, pairs/s, peak memory where a measurement exists), next to the family's potential numbers, as information for later optimisation and for making a family optional. Where no measurement exists the report says "not measured"; a missing or over-budget cost never changes an outcome.
- Code, inputs, commands and results for every arm stay tracked in the worklog and manifest as before.

Nothing else in the preregistration changes: statistics, folds, seeds, controls, roles and the forbidden-set list are as registered.
