# E32 research training transport

The [registered E32 arm](../benchmarks/e32_palette_registration_2026-10-07.md)
uses the existing strict four-source trainer. Its consumer contract is
[e32_palette_training_transport_2026-10-07.json](../benchmarks/e32_palette_training_transport_2026-10-07.json).
The coordinator's [shared-control decision](../benchmarks/e29_e31_e32_shared_control_decision_2026-10-07.md)
requires a fresh matched control under the combined v40 program.

## Columns and identity

The projected table retains its physical `f0..f1852` columns and appends
`palette_f1825..palette_f1866`. Logical inputs below 1825 read their original
numeric columns. Logical IDs 1825..1866 read only the named palette columns.
The old numeric auxiliaries remain physically present and are never aliased
to palette inputs. Named columns may appear in any physical order.

The logical width is 1867. Its compound research identity, computed by the
existing Rust feature-set hash owner, is
`basic+peaks+v2+palette@w1867/e32_palette_v2#2ec9dcda`.
This combines the inherited Rev5 family with the separately pinned
`palette@w1867/palette_v2#30b09cd1` producer. It adds no serving registration.

The keep list must equal the registered ordered 462 IDs. Palette values are
cast from the instrument's Float64 values to Float32 during data assembly;
the consumer requires Float32 named columns. Every selected inherited and
palette value must be finite and non-null. Absent values cannot become zeros.

## Admission declarations

Each projected table's `.manifest.json` retains the existing decoder, row
selection, ordered-key, table-byte, source-root and D1 decision bindings. It
declares the compound `feature_set_id`, formula revision 5 and a
`research_palette` object with:

- The contract's `schema`, inherited and palette identities, producer commit,
  producer binary, bank/instrument manifests, ID-list hash and cast policy.
- `serving_allowed: false` and the exact 42-entry canonical-ID-to-named-column map.
- `inherited_table_sha256`, binding the original table before projection.
- Exact `member_sets`, `role` and `key_domain`.

Human roles are `D1-fit` or `D1-development`; teacher roles are
`TRAIN-oracle-fit` or `TRAIN-oracle-development`. The ordinal coverage pool
uses `TRAIN-ordinal` and `coverage-selection-ordinal`. Other tables use
`member-pair-observation`. The existing receipt-bound D1 decision remains
required. Fit/development roles must match their recipe slots and weights.

The root's wide receipt carries the full transport contract under
`research_palette` and width 1867. It remains bound by the existing admission
freeze. Every leg must carry the projection. E32 requires the unchanged E30
control recipe, head N, strict admission and train-only operation.

Python preflight checks all projected declarations and label-free key
populations before label-bearing payload access. Rust preflights all
declarations, then all key files, then table byte hashes before loading.
Palette_v1, altered pins/maps, unauthorized members, mixed projected/legacy
legs and historical replay are refused. Coverage retains the original pool
ancestry and key-byte pin; its existing family mask and row-filter owner stay
in use.

## Assembly and verification boundary

These consumers read already projected tables. They do not construct joins
or certify an unbuilt data archive. The later combined-package assembly must
prove inherited labels, feature bits, ordered observations, teacher/coverage
selection and roles against the original control tables. It must check the
producer's exact pinned instrument manifest before value joins.

Join by the registered member, pixel-pair and observation identities,
preserving duplicates and order. Instrument `row_id` and wide
`source_row_id` describe different coordinates and cannot be substituted for
one another. Coverage uses its original selection ordinal and separate path
key domain. Positional or pair-only fallbacks cannot establish parity.

`just e32-extension-build`, `just e32-extension-tests`,
`just e32-shippath-regression` and `just e32-serving-refusal` expose local
gates with explicit scratch, target and binary arguments.
`just e32-control-parity` runs one complete 120 × 50,000 control cell under
one Rayon thread and v3. It compares complete model bytes after removing only
`zentrain.repro` through the canonical strip owner, and separately compares
reproduction facts after removing run/build locations and timestamps. Input
SHA values, recipe, weights, seeds, sampling and admission facts remain in
the comparison. A mismatch is recorded; it never selects a different control.
