# Palette pair source authority

`palette_training_pairs.py` takes a caller-frozen `pool` authority alongside the
source inventory. Schema `e35-palette-pool-v2` requires `certified_clean=true`
and complete `available_families` for tiny, photo, screen, line and mixed.
`authority_sha256` is SHA-256 over every other field: JSON with sorted object
keys, compact separators, ASCII escapes, finite numbers, UTF-8 bytes and no
trailing newline. Array order is retained and extra provenance fields are bound.
A pruned pool carrying its former digest is rejected before capture processing.
A hash match does not establish dataset admission; the caller must supply the
independently approved authority.

Under the owner-authorized question 15 contract, select 64 families per class
or every certified clean family in a short class. Development counts use
`max(1,min(n-1,(14*n+32)//64))`; fewer than two families is incomplete.
Nominal weight is `2/(5*class_role_count)` before the existing R915 acceptance
correction. Development receives zero training weight. Teacher arithmetic,
feature identity, persistence, independent decode and alias checks remain fixed.

The registration names a frozen family-hash order without specifying the
algorithm, encoding, sort direction or tie-breaks. Exact ordering remains an
external freeze binding; the validator enforces counts and family disjointness.
`just palette-pairs-check` checks synthetic authority and capture metadata;
`just palette-admission-test` retains the existing verifier admission checks.
Neither recipe reads a dataset or starts generation, preflight or training.
