# ADJUDICATE comparator landing — 2026-10-08

C2 used to reject identical NaN placeholders at uncomputed feature slots,
charging codec saturation as model ties. The production `bake_verdict` now
calls its compiled private `missing_features` comparator. Finite slots retain
the same 1e-5 epsilon; NaNs must match at the same positions; infinities and
unequal lengths remain unequal. The lane's two regressions and the reviewer's
five controls are ordinary binary tests. Two parser regressions cover the
explicit engineering-only `--corpora none` option and refusal of mixed lists.
No corpus defaults, input values, model, scoring arithmetic or acceptance bar
changed. Investigation scripts and reports stay on the original adjudicate
bookmark and in its evidence archive.

C5 scores raw cached identity features without the pixel-identity shortcut.
Ten reference-only PJND_FRAGILITY inputs (422, 480, 509, 538, 567, 596, 625,
654, 683, 712) are nonzero in the admitted Rev5 identity probe. The source
comments and emitted note now describe this; the public legacy note constant
name is retained for compatibility. The historical zero-vector probe captured
placeholder features from the product shortcut. It cannot establish a general
zero-feature identity contract. Pixel identity remains exactly 100; the frozen
seed-0 raw C5 failure and G-STEER failures remain unresolved model properties.

## Verification and evidence

Verification receipts are recorded under
`/mnt/v/output/zensim/adjudicate-land-2026-10-08/`. The final pointer and done
record bind the locally committed source, compiled verdict binary, exact
commands, inputs and outputs. Replays use the frozen seed-0 bake SHA-256
`f803b74c4252952f337abdc0234c2930839d45dddc32ae9b8b5296d6c840f400`
and the admitted label-free standard/ladder/identity/negative engineering
instruments only. Empty human-corpus and unavailable integrity roots are
explicit; missing coverage remains unmeasured and cannot be shippable.

Required checks pass: `cargo test -p zensim --all-features` (900 passed,
27 existing ignores), `cargo test -p zensim-validate --bin bake_verdict`
(50 passed, one existing ignore), CI-exact `just clippy`, `just lint-scripts`,
and scoped formatting. Additional `dial_addressability::tests` coverage
passes 53/53. No assertions, expectations, ignores or numerical bars were
relaxed. The initial formatting failure is preserved in the logs; line
wrapping was corrected before these successful checks.

Both frozen seed-0 replays pass C2 at the unchanged 0.05 bar:
standard 28/4318 = 0.006484483557202408 and ladder 343/9411 =
0.03644671129529274. Prediction TSVs are byte-identical to the accepted
corrected receipts. All 14 verdict rows per grid agree with those receipts
except the corrected C5 explanatory note; every numeric value, state and
bar is unchanged. Relative to the original verdicts, only standard C2 and
the C5 note change. Both replays retain the raw C5 failure and unmeasured
coverage, and are not shippable. The compiled verdict SHA-256 is
`d79a3432a779ee447338940ca863849c823bba040f105a70bca55a2bb8b666a1`. The original instrument/model
bytes were rehashed against their frozen pins before either replay.

Base: `6a37f4e45af4f15e8076b82e45d4e8d805cfe02a`. One local landing change
on `quarantine/codex/adjudicate-land`; no push and no local main movement.
The final receipt binds this source to the compiled binary and both verdicts.
Evidence mirror: `/mnt/tower/output/zensim/adjudicate-land-2026-10-08/`.
The newly created Cargo target is removed after mirror verification; the
pinned release binary remains with the evidence.

## Oversized documentation errata excluded from this commit

The brief excludes edits to existing oversized non-Rust files. Proposed
explanatory corrections and a C2/C5 Known Bugs entry are in
`OVERSIZED_DOC_CORRECTIONS.patch` and `OVERSIZED_DOCS.json` in the evidence root.
They cover `CLAUDE.md`, `benchmarks/dial_addressability_gate_2026-09-04.md`,
`benchmarks/d_id100_2026-09-04.md`, `docs/PLAN_BEST_OF_ALL_2026-09-06.md`,
`docs/FEATURE_SET_IDS.md`, `docs/DATASET_HISTORY.md`,
`docs/history/CLAUDE-through-2026-09-07.md`,
`benchmarks/balance_campaign_2026-08-28.md`,
`benchmarks/d_free_id100_2026-09-05.md` and
`docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`. Older quoted explanations remain
historical observations rather than current assertions. The six false C5
notes in `benchmarks/cleanup_scientific_controls_2026-09-07.json` are superseded
by this erratum; its original verdict bytes are retained. The corrected live
verdict output no longer emits that claim. This paragraph is also the durable
Known Bugs record until the oversized project document can take the proposed
entry.
