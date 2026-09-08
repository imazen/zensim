# Start here — zensim’s current product and science

**Reviewed 2026-09-07 against checkout `0205c45c`, September 6 results, and the
Claude session through September 7.** This is the navigation page. Dated
measurement records remain the evidence; an old “current state” paragraph is
not a new ruling. The earlier version of this page remains in Git history.

## The product

The user controls **one target score**. The codec/picker chooses its own
parameters and reaches that quality with few passes, small output, low latency
and low memory use. The score must remain useful across codecs and content,
including near-lossless settings and each codec’s actual floor. Negative
scores are valid; there is no arbitrary required negative depth. A stronger
rank correlation alone does not establish a better user dial.

Read [`docs/CODEC_TARGET_METRIC.md`](docs/CODEC_TARGET_METRIC.md) for current
integration, profile/bake mapping, and real consumer behavior. **B remains the
`codec_target()` / `latest_preview()` default; D is an explicit fast profile.**
The September 5 change updated D’s own calibration, not the default alias.

## What the latest evidence establishes

| Question | Current answer | Evidence |
|---|---|---|
| Is there a useful fast scorer? | Yes: D is served, with 28 declared feature IDs; the compared ADD156/D lineage meets all five codec-floor bars under the operative `resolvable` rule. This is not a claim of universal perceptual superiority. | [D dial change](benchmarks/d_ship_flip_2026-09-05.md), [current ladder comparison](benchmarks/board_ladder_ruler_2026-09-06.md) |
| Is a stronger common dial ready to replace it? | No unified winner has been established. The constrained MLP improves several human-rank axes at roughly D’s measured speed but still misses codec floors and trades KonJND. | [September 6 decisions, D1](docs/OPEN_DECISIONS_2026-09-06.md), [best-of-all record](benchmarks/best_of_all_2026-09-06.md) |
| Does “selected” mean qualified to ship? | No. The current selector can pick a recipe with incomplete G-ADDR coverage or failed codec floors. Product qualification requires both G-ADDR tiers plus the other scorecard gates. | [ladder selection result §6](benchmarks/board_ladder_ruler_2026-09-06.md), [scorecard](docs/MODEL_SELECTION_SCORECARD.md) |
| Can wide/subset models be served? | The feature-plan work closed the old width-only refusal. A/B/BHdr/D use explicit-ID bakes. C/CHdr still have an unresolved train/serve activity-toggle mismatch; their densification is pending. | [feature-system design/results](docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md), [dense serving](benchmarks/dense_serving_ungate_2026-09-06.md), [D3](docs/OPEN_DECISIONS_2026-09-06.md) |
| Is revision 2 the shipped default? | No. Deterministic roots/power and bounded-feature changes have measured opt-in paths. Default behavior and the remaining predictor-transform exposure are separate decisions. | [D2/D5](docs/OPEN_DECISIONS_2026-09-06.md), [score arithmetic owner](benchmarks/score_owner_consolidation_2026-09-06.md) |

The original Claude session’s last user question was whether we have something
really good, because the accumulated choices were overwhelming. There is no
subsequent ruling on the September 6 decision memo. Do not treat its
recommendations as adopted changes or present another six-question wall.

## One route through the work

**Current cleanup plan:** [the cruft purge, September 7 revision](docs/PLAN_CRUFT_PURGE_2026-09-06.md#current-cleanup-plan--2026-09-07)
names the deletion batches, owner migrations, feature/recipe ablations and
completion checks. Its §0 separates demonstrated parity from unresolved
replacement claims. **Prefer Rust for canonical implementations; keep Python
for fast invention and independent references.** Establish the intended method
and actual implementation evidence before retiring a useful prototype.

1. **Define the product claim and cheapest discriminating experiment.** Pin the
   baseline, one lever, data split, feature/decoder era, decision rule and cost
   budget. Use [`docs/WAVE_PLAYBOOK.md`](docs/WAVE_PLAYBOOK.md); the July
   [`ITERATION_PROTOCOL`](docs/ITERATION_PROTOCOL.md) contains historical cost
   measurements and the enduring efficiency principles.
2. **Resolve the bake and data before computing.** Read
   [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md),
   [`docs/FEATURE_SET_IDS.md`](docs/FEATURE_SET_IDS.md), and the relevant
   [`DATASET_HISTORY`](docs/DATASET_HISTORY.md) corrections. Use the canonical
   owners in `CLAUDE.md`; do not infer semantics from a 372/944 width.
3. **Evaluate and qualify separately.** `bake_verdict` owns the statistics;
   `scripts/run_full_eval.sh` adds coherence. Neither alone runs the real
   codec RD/target exam. Grade G-ADDR on the named ladder and compatible identity
   and negative probes. Missing measurements stay missing. Follow
   [`MODEL_SELECTION_SCORECARD`](docs/MODEL_SELECTION_SCORECARD.md).
4. **Interpret in the existing board.** The fair board is
   `/mnt/v/output/zensim/reports/summer_gauntlet_fair.html`; the full historical
   board is `summer_gauntlet.html` beside it. Generator:
   `scripts/v_next/bandwise_dashboard.py --fulleval-dir …`; run
   `scripts/v_next/gauntlet_gates.sh <html>` on every regenerated board.
   Codec curves use `dial.curves`; floor qualification uses `dial_ladder`.
   Keep their instrument identities visible.
5. **Publish a decision, not just a winner.** State what improves for the user,
   what still fails, what is unmeasured, and the smallest next test. Consult
   [`OPEN_DECISIONS_2026-09-06`](docs/OPEN_DECISIONS_2026-09-06.md) for changes
   that have not been adopted. Push only through `scripts/safe_push.sh` when
   pushing is part of the authorized task; follow workspace preservation rules.

## Chronology and lookup

- May’s Tuner/JND conventions and July’s additive-only arguments are historical.
  Later retractions and code take precedence over their opening summaries.
- The August [SOTA campaign](benchmarks/sota944_campaign_2026-08-03.md) and
  [balance campaign](benchmarks/balance_campaign_2026-08-28.md) retain the runs.
  [`TOP_MODELS_COOKBOOK`](docs/TOP_MODELS_COOKBOOK.md) is a dated recipe archive.
- September 5’s `distinct` floor result and September 6’s operative
  `resolvable` result answer different questions. Always name the rule,
  margin and instrument alongside the value.
- [Science/workflow audit](benchmarks/science_workflow_audit_2026-09-07.md)
  records this review’s evidence, coverage and prioritized cleanup.
  [`benchmarks/INDEX.md`](benchmarks/INDEX.md) indexes prior experiments.
