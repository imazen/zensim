# Trainer and metadata cleanup — September 7, 2026

The Rust MLP trainer rejects unsupported configurations before allocation or
row loading. Table admission now checks actual CSV/Parquet headers through one
column-discovery owner, selected feature IDs against each producer, registered
formula revisions, and conflicting explicit root/file decoder declarations.
Malformed manifests and missing requested columns remain fatal even in
historical replay. Unknown or mixed semantic provenance requires an explicit
replay reason; missing decoder declarations cannot earn qualified provenance.
This validates declarations, not the truth of every historical decoder stamp.

The output bake receives one feature-set identity only when every training leg
resolves to that identity. An unknown leg can no longer disappear during
collection and leave a misleading known identity on the result. Dense/gapped
tables remain refused by unmigrated loaders; no silent zero fill is introduced.
CSV and Parquet share header validation; independent value-loading references
remain for parity tests. Transform parameters now reject malformed/nonfinite
values, and the shared capability gate covers the remaining parallel-head and
activation constraints. GPU admission is tested; GPU training was not run.

`bake_contrib` uses the feature-family registry and declared feature IDs.
Its report IDs and live-mask IDs are identities, not packed positions. The
producer-absence report is per corpus; a 944-column table is no longer assumed
to have structural zeros in the v1 pool. Its activation perturbation remains a
diagnostic with a full BakeScorer baseline gate.

Removed the unused positional training/CV/coordinate/CMA-ES/RankNet/FISTA
surface from `zensim-validate`, its duplicate private trainer modules, and the
now-unused CMA-ES dependency. Kept dataset extraction, caches, scale reports,
explicit-weight diagnostics and their historical Kendall instrument. Removed
silent feature/weight truncation. `zensim_mlp_train` owns model training and
`bake_verdict` owns candidate evaluation through the Rust surface.

Removed the stale `mlp_cross_check` example: it printed Rust values from one old
bake and did not assert JavaScript parity. The active browser JavaScript and
its assets remain. Searches found no executable callers of these removed
training flags/example in tracked scripts, CI, tests, zenmetrics, zenpapers or
shared scripts. Exact retired source remains in Git at `11123107b26a` and in
the private cleanup archive.

Validation: admission/capability/library tests; legacy dataset-loader test;
contribution tests including actual ID-family mapping; verdict tests;
nonnegative-distance/output-polarity gates; CSV/Parquet parity guard fixtures.
The two large mounted parity comparisons remain explicitly ignored by default.
CI-exact workspace clippy passes. The competitive reproduction and paired
floor control have separate records; removal of code is not proof of quality.
