# The dense-bake flip — four shipped profiles now declare the ids they read

Increment **2A** of [`../docs/PLAN_CRUFT_PURGE_2026-09-06.md`](../docs/PLAN_CRUFT_PURGE_2026-09-06.md)
(gates B.1 / B.2 / B.3 / B.5). Increment 1 built the tool, the declaration and
the consumer gather but **replaced no shipped bake**; this one replaces four.
Contract + the tool's own gates: [`dense_bake_contract_2026-09-06.md`](dense_bake_contract_2026-09-06.md).
Artifacts: `/mnt/v/output/zensim/purge-2026-09-06/inc2/`.

---

## 1. What flipped

| profile | shipped bake (before) | shipped bake (now) | declared → | bytes |
|---|---|---|--:|--:|
| **D** (default) | `d_sdr_add156_id100_negrich_dial_2026-09-05` | `…_byid_2026-09-06` | 372 → **28** (`f6..f155`) | 4,222 → 1,420 |
| **B** | `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07` | `…_byid_2026-09-06` | 372 → **95** (`f3..f369`) | 7,325 → 2,012 |
| **BHdr** | `bhdr_linear_shaped_cvvdpmix_2026-07-12` | `…_byid_2026-09-06` | 372 → **133** (`f1..f359`) | 11,826 → 5,331 |
| **A** | `v47_strict_qat_native_2026-05-27` | `…_byid_2026-09-06` | 372 → **285** (`f0..f371`) | 27,316 → 26,456 |

Every one has `caller_input_width() == n_inputs() == |declared ids|` and **zero
`Drop` transforms**. The wide originals are **not deleted** — they stay
committed, and the flip gate `include_bytes!`es both, so they are load-bearing
evidence rather than archive.

`C` and `CHdr` are **not** converted; see §4.

## 2. The identity evidence, at three levels

**Probe rows (the tool's own gate).** D and B bit-identical on 512/512; A and
BHdr on 496/512, the other 16 being the pre-existing NaN-absorption class
(`fma(NaN, 0.0, acc)` is NaN, so a wide bake poisons an accumulator where the
dense one has nothing to poison it with — only reachable on input that was
already out of domain). The DEAD-COLUMN gate — score the SOURCE with every
dropped line forced to `0.0` — passed on **all 512 rows for all four**, which is
what separates "the columns are dead" from "the probe avoided the difference".

**Real pixels through `Zensim::compute` (gate B.2).**
`zensim/src/profile.rs::dense_bake_flip_gate` scores each shipped profile and a
twin built from the SAME `ProfileParams` over the retired wide bytes, over the
20-cell `tests/common/parity_cells.rs` geometry matrix, and compares `score`,
`raw_distance` and `mean_offset` at `to_bits()`. **4 profiles × 20 geometries ×
3 scalars, all equal.**

That test carries a **negative control** and it was RUN, not assumed: reverting
D's `include_bytes!` path fails two tests with
*"shipped bake is byte-length-identical to the RETIRED wide bake — the dense
flip did not happen and the identity assertions are vacuous"*. Without it the
suite would go green the moment someone reverted a path.

**Full verdicts (gate B.1).** `bake_verdict --full-json` on the default
(current-extractor) 372 root, wide vs dense, compared field-by-field over the
whole JSON tree:

| profile | fields compared | STATISTIC diffs | bake-identity-block diffs |
|---|--:|--:|--:|
| D | 31,407 | **0** | 14 |
| A | 32,949 | **0** | 1,457 |
| B | 32,896 | **0** | 1,229 |
| BHdr | 33,250 | **0** | 1,709 |

Rank, dial, corruption, composite, gates, `m3_coherence` and every `per_pair`
array are bit-identical. The only fields that move are the bake's own identity
block — `bake`, `bake_sha256`, `n_inputs`, `model.*` (layer shapes, scaler,
the permuted `feature_transforms`), `feature_set.*` — which is exactly what a
conversion is supposed to move.

**An independent control on the numbers themselves.** Shipped **B** reads
CID22 SROCC **0.8821166166351724** and kon504 **|0.5193759178072009|** through
BOTH arms — the two values `CLAUDE.md` recorded for that bake on that root
before this lane existed (0.88212 / |0.51938|), to every published digit.

## 3. It found one more site reading POSITIONS as IDS

The `--full-json` **kadis multi-metric per-pair** block pre-sliced
`sample.feature_rows[i][..n_inputs]` before handing rows to
`Ensemble::score_rows`. For a dense bake `n_inputs` is the PACKED width, so the
slice handed the scorer the first 28 columns of a 720-column table and the
gather inside then read ids `f6..f155` out of them. **MEASURED on shipped D:
4,920 of 4,928 per-pair predictions wrong** — silently, in a block the
gauntlet's scatter matrix renders. Same defect class as the grid-admission bug
increment 1 fixed; this site was missed because it is the only one that does its
own slicing upstream of the canonical dispatch.

Fixed by handing the row over caller-laid-out and letting the gather work. That
is byte-identical for an identity bake, because `CallerGather::Positional::fill`
already copies exactly `min(n_inputs, row.len())` and zero-fills the tail.

**And fixing it surfaced a second, opposite fact.** The first fix admitted the
source with `accepts_row_width` — the GRID rule, an exact `==` — which dropped
the kadis block from every identity bake's verdict, because that source is 720
columns wide and has always fed 372-input bakes **by prefix**. Two genuinely
different historical rules exist and collapsing them moves numbers in both
directions, so the owner now carries both:
`accepts_row_width` (grid, `==`) and `accepts_prefix_row_width` (prefix, `>=`),
with the same reach test on the dense arm and a test asserting the two
**disagree** — if they ever agree everywhere, one is dead and the split is
unjustified.

## 3b. Three more sites, all the same bug class, all found by RUNNING it

Increment 1 wired `CallerGather` through the six evaluation scorers. Flipping
the shipped bakes proved that was not the whole surface — four more sites read
POSITIONS where the dense contract means IDS, and **none of them was found by
reading the code**. Each was a test going red on a real artifact.

**1. `fold_engine::bake_pool_need_from_model` — a SKIP decision, so the worst
place for it.** It folded `caller_line_reads` (positions) against the v1 family
bounds `f156..228` / `f228..300` / `f300..372`. Dense `B` has 95 live positions
`0..94` carrying ids `f3..f369`, so the fold reported
`{peaks: false, masked: false, iw: false}` for a bake that reads **49 lines
above `f227`**. Had pool-skipping been enabled for `B` (it is on for `D` only),
the walk would have deleted the masked and IW pools and the gather would have
read structural zeros. Fixed by routing through `feature_plan::bake_read_slots`,
which now maps through `declared_feature_ids` and is THE owner of "which ids
does this bake read" — the translation used to live in `Plan::for_bake`, which
is why the owner returned positions under an ids-shaped name.

**2. `Plan::for_bake`'s id-space branch bypassed the `Off`-is-never-served
policy.** `pools_mode_for_need` documents, with a measured footprint argument,
that `V1PoolsMode::Off` is never the right answer for a served v1 walk: it
costs the same arithmetic as `Peaks` and grows the hot set by disabling the
band-local self-blur. `from_block_profile` routes through that owner;
`derive_with_layout` derives from touched families and returns `Off`. Nothing
reached that branch until `D` declared its 28 basic ids — at which point D's
emitted vector went from real values at `f156..227` to zeros. **MEASURED**:
default-build `D` at 96×64 emitted 134 nonzero slots spanning `f0..f154`
against the buffered arm's 350 spanning `f0..f371`, diverging over exactly
`f156..371` where the documented asymmetry is `f228..371`. The promotion is
applied in `for_bake` for both branches, which restores byte-identical
behaviour and puts the policy back in one place.

**3. The batched FD-gradient entry did not gather.**
`score_features_fd_gradient_with_profile` — the entry the jxl H3 magnitude
steering loop calls — went straight to `prep_bake_input_f32`, whose
`n_inputs < features.len()` branch takes a POSITIONAL PREFIX. Against the
sequential recipe (which routes through `forward_one_bake_with_codec` and does
gather), component 0 read **−748.7 batched against the correct 0.0**, because
caller position 0 is id `f0` and `B` declares `f3..f369`. Its `k >= n_inputs`
zero-tail shortcut was wrong for the same reason — `n_inputs` is a dense bake's
PACKED width — and is now an id-membership test.

**4. And a short feature vector was being silently zero-filled at both
entries.** `score_features_with_profile(B, &vec![0.1; 156])` SUCCEEDED: the
156-wide row was gathered into `B`'s 95 ids and every id above `f155` became a
structural `0.0`. `Layout::gather` writing `0.0` is right for a DECLARED GAP
and wrong for a short row, because every id a dense bake declares is an id it
reads — this is the "a consumer cannot tell this zero from a measured zero"
failure the whole contract exists to end. Both entries now REFUSE with a named
reason. Gate: `a_dense_bake_refuses_a_feature_vector_that_does_not_reach_its_ids`,
with a positive control at the reach width so the refusal cannot be a blanket
rejection.

Every one of these is the shape the ruling names, and every one was invisible
while `position == id` held for all 11 shipped bakes.

## 4. Why `C` and `CHdr` did not flip

Not an oversight, and not a scope cut: for that pair the conversion is **not
score-neutral**, and the reason is a pre-existing train/serve skew densify
exposed. `Plan::for_bake`'s identity-layout branch falls back to `everything`,
which hard-sets `append2_dst_activity: true`; its id-space branch derives
`false`; and the canonical extractor that built their training tables defaults
**false**. So the shipped runtime already computes a BANDVIS formula those
weights never saw — **0.866 zensim points on C, 0.311 on CHdr** on one CID22
pair. Both available fixes change a shipped number, so the choice is the user's.
Measurement: [`dense_bake_contract_2026-09-06.md`](dense_bake_contract_2026-09-06.md) §5.

The banner is on both bake functions in `profile.rs`, and its executable form is
`flipped_bakes_are_dense_and_the_c_pair_is_deliberately_not`, which asserts the
pair is STILL the wide 944 shape and declares no ids — so densifying them while
the decision is open fails the suite.

## 5. Deliberately out of scope

The five non-shipped bakes in `zensim/weights/` (`b_sdr_..._anchored`,
`b_sdr_..._dense_dial`, `bhdr_..._anchored2`, `d_sdr_add156_dense_dial`,
`c_sdr_mlp944_corrmix`) are **not** converted. They are retired historical
artifacts that nothing serves; converting them would mint new bytes with no
consumer while making the sha256 of a published lineage harder to resolve.
Increment 1's dry run records that four of them densify cleanly if ever needed.
