# One feature planner and the C activity correction — September 7, 2026

The September 7 user ruling authorizes replacing B/C/D scores and the listed
cleanup. This increment corrects the C/CHdr training/serving mismatch before
retiring the two alternate plan derivations. It does not retrain either model.

`Plan::for_bake` derives work from the actual read IDs for both explicit-ID and
legacy identity layouts. It uses the canonical training BANDVIS activity
variant (`append2_dst_activity = false`), preserves each bake's formula revision,
and retains the Off-to-Peaks working-set policy. `ComputeSet::from_block_profile`
and `fold_engine::wide_bake_v2_read` are removed; their tests now use the owner.
The old coverage-only equivalence check is replaced by consumed-feature parity.

## Artifacts and numerical scope

| Profile | Source caller/internal width | Declared IDs now | New bytes | New SHA-256 |
|---|---:|---:|---:|---|
| C | 944 / 667 | 667 | 151785 | `996dfbb16ee0abf1a4d7faeaeab7e390fdc20277bbcaa46c4f3b89b2604041f7` |
| CHdr | 944 / 697 | 697 | 182826 | `3ea640d3299632eae189bedb6ec47911e2c0dbf5ef8bce81a36815a83d20411a` |

The existing `bake_dial_refit densify --gate-rows 4096` owner produces both
artifacts with zero Drop transforms and bit-identical cached predictions on
4096 rows each. Original August 29 bytes and manifests remain packaged.
Explicit-ID metadata increases disk size by 2442 / 2631 bytes; this conversion
is a semantic simplification, not a compression claim.

The dense and corrected wide versions return bit-identical score, raw distance
and mean offset across the existing 20-geometry pixel matrix. The independent
canonical-producer check compares actual consumed features on 64×64 and 97×65
pairs: non-free IDs are bit-exact; the existing raw-moment/Class-C accumulators
retain their established absolute 2e-5 summation-order bound. Pixel and cached
surface scores are bit-exact. Six bundled bakes plus 486 filesystem bakes yield
984 passing model/pair cases. Re-enabling the old activity variant makes this
gate fail on consumed `C f924`, even with the activity-flag assertion removed:
0.0197271911 versus canonical 0.1770873485. The source was restored afterward.

A frozen before/after census on the same 96×96 pixel pair serves all 486 old
artifacts without refusals. **182 scores are unchanged; 304 change, all from
944-wide historical models.** Maximum absolute change on this fixture is
0.6369706833. This correction applies to custom wide bakes as well as C/CHdr;
old recorded verdicts retain their original instrument identity.

The 96-cell named-profile matrix changes 22 cells, all C/CHdr. Maximum absolute
change is 0.8089595774 for C and 0.6947678337 for CHdr. The separate shipped
quantization golden moves C 41.4265873778 → 38.6336270950 and CHdr
66.2051229558 → 66.0798119607. The CHdr single-LSB golden moves
96.9972785152 → 97.1320373283. Other profile pins and their tolerance are unchanged.

## Compute and retention decision

The existing `extract_paths_bench` gains two matched cheap-wide arms, raw
moments and raw moments plus bounded error, using the same pixels, process,
paired/interleaved group and scratch policy. Raw Off remains an explicitly
labeled historical control; it is not represented as current D serving.
These are extraction measurements, not complete scoring or codec-loop timings.

Buffered extraction remains necessary for supported compatibility paths and
its distinct padded-width semantics. The August 30 audit's claim that no fold
scoring or prepared-reference surface exists is superseded by September's
implementation; its numerical width divergence and remaining caller evidence
still prohibit blanket removal. No kernel or free feature is deleted on a
coefficient-count argument. The full-feature research producer remains available.

Detailed local artifacts: `~/tmp/zensim-science-audit-2026-09-07/plan-*` (frozen
before binaries, census and matrix TSVs, JSON comparisons, mutation evidence,
checks and cost logs). The following final check results are added after execution.

The fresh accepted wall-time comparison is **NOT MEASURED**: the 180-second
budget completed only 6 of 25 requested rounds and recorded 186 noisy checks
while an unrelated Miri test was active. The run ended naturally in 192 seconds;
its raw output is retained but excluded from performance conclusions. The gate
was not disabled and the other job was left alone. Existing Class-C measurements
remain dated evidence, not a new performance claim.

## Final validation

All-feature library: 422 passed, 6 explicitly ignored, including the 984-case
canonical pixel check. Candidate surface: 10 passed, including dense/wide HDR
PU-pixel versus canonical-feature parity. Shipped provenance and historical
bake/polarity gates: 6 passed. The eight-arm serving build matrix passes with
zero refusals and bit-identical scores within each environment. CI-exact
`just clippy`, `just api-doc-check`, formatting and diff checks pass. No public
signature changes in this increment; the C pair's intentional behavior and
artifact changes are in the changelog and manifests.
