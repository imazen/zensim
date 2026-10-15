# E34 preparation: feature census and statistics prerequisite

**Incomplete; not frozen or launchable.** P1 is complete. P2, P5, P7, the
noise read, P4, local P6 and the freeze draft remain outstanding. No fit,
noise read, fleet job or gate change occurred.

The [registration](e34_registration_draft_2026-10-10.md) remains the rule.
This record adds measurements and a dependency request; it does not amend it.

## P1: nonnegative difference inputs

Code: `760c301c`. Measurement snapshot: `cdab396b6a185ac026c097346a4483736c201e60`
(same census and column-owner source as that commit). The census delegates to
`canonical_corpus.check_table_integrity.column_stats`, with an explicit row
mask for coverage. All 410 registered difference columns have finite,
nonnegative minima on each table below. Columns below zero: **none**.

| Registered fit table | Rows |
|---|---:|
| SafeSyn fit | 141,054 |
| CID22 teacher fit | 12,163 |
| Human all fit | 11,430 |
| Human without CID22-A fit | 9,851 |
| Human without KADID fit | 4,771 |
| Human without KonFiG fit | 10,758 |
| Human without TID2013 fit | 8,910 |
| KADIS coverage, cf98 | 17,600 |

The four full-data legs total 182,247 rows. The eight-table census visits
216,537 rows, **including repeated LODO subsets**. This is no count of
independent observations and establishes no steering or serving gate.

Each table is admitted against E33's frozen fit contract
`3bb4705b8a07307b3dec3956f43d99ae649ea333b6c95817ac389167713ec4d7`.
The original `/var/tmp/rev4-featpot/v2e29` path is absent on dev; the retained
E33 packet extraction supplies the tables. Fit-time coverage is a derived
cf98 selection, not the whole stored pool. The first attempt refused that
manifest mismatch before decoding any feature payload. The completed census
checks the pinned Rev5 pool and key hashes, then verifies the selected row
index hash and ordered row identities against the frozen derived-table
contract. It does not recreate or decode its target column.

Receipt, checked on **dev**:
`/mnt/v/output/zensim/e34-prep-2026-10-15/p1-census.json`, SHA256
`747149eeb7e1175cf7e3b61d2c1bb7f2e1ad87c04cbc530c289952ae592f77c3`.
It contains each column's minimum and maximum per table and across tables,
the exact command, source snapshot and input bindings. No remote mirror was
created; mirroring remains outstanding.

Reproduction (fresh output required):

```sh
root=/mnt/v/output/zensim/e33-impl-2026-10-09/packet/container-scratch-1/fit-cell/9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008/rev4-featpot/v2e29
contract=/mnt/v/output/zensim/e33-impl-2026-10-09/packet/e33-fit-contract.json
TMPDIR="$HOME/tmp/e34prep" ~/work/claudehints/scripts/run-heavy --mem 24G --jobs 8 -- \
  just e34-feature-census "$root" "$contract" /path/to/fresh-census.json
```

Verification: admission refuses an altered contract before any Parquet read;
a synthetic table checks the masked census, its negative-value reporting,
feature-only column projection and unchanged unmasked behavior. Both tests
pass (`python3 scripts/tests/test_e34_prepare.py`, under run-heavy).
Baseline scoped formatting and trainer/panel Clippy also pass. Final
`just clippy` (CI-exact workspace/all-targets/all-features) and
`just lint-scripts` pass. Cargo reports an existing future-compatibility
notice for dependency `proc-macro-error2` 2.0.1; Clippy exits successfully.

## P7 dependency request

Source inspection on 2026-10-15 verified the pinned `zenmetrics-stats`
`1e48c1eafcaa8afaa8b632c4f12d8181a971a248` and current upstream
`54e046a9aafc2c98c2e6f26b920015a897fcdb66`.

Current upstream adds `Interval::student(groups, level)`, but its
`special::student_t_quantile` is crate-private and its Student CDF is private.
There is no public Student tail. `Interval` still exposes only the capped
two-sided `p_zero`, without directional draw-tail counts. `Design` has
Rows, OneWay and TwoWay, without dataset stratification.

The E34 rule requires the larger of its directional bootstrap p and the
Student upper-tail p at df 9. Neither replacing it with `p_zero / 2` in all
cases nor substituting a normal tail implements that rule. Computing these
statistics locally would conflict with the registered statistics ownership.
A pin bump to the inspected current upstream alone cannot finish P7.

Request to the statistics owner:

1. Public Student quantile and upper-tail probability, with finite-domain
   handling and independent tail/quantile tests (including df 9 and Holm's
   .05/9 threshold).
2. Directional lower-tail probability on `Interval`, retaining the exact
   `(1 + count(theta_star <= 0)) / (finite_draws + 1)` convention and zero
   ties. Preserve the existing two-sided statistic.
3. Dataset-stratified reference sampling crossed with a shared seed factor,
   with copy-aware expansion and arm/control pairing. The registration
   currently describes the unstratified design and its dropped KonFiG draws;
   adopting stratification still needs an explicit freeze record rather than
   silently changing the design.

Source checks used the pinned Cargo git checkout and `gh api` at the full
upstream commit for `crates/zenmetrics-stats/src/{lib,resample,special}.rs`.
No sibling repository was patched and no dependency pin changed. The E33
stored prediction/label read remains unused, with no noise numbers or Q9
measurement claimed. Add its ledger reservation before that future read.
