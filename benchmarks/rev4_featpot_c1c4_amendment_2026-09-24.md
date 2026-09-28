# Rev4 POTENTIAL C1-C4 execution amendment (2026-09-24, before candidate fits)

**POTENTIAL — ceiling, not a model score.** This supplements preregistration
`d2169f5b` and addendum `044f00dc`. The coordinator's 2026-09-24 direction
starts the landed C1-C4 arms when `PARTB_C1C4_DONE.md` exists, before the
ongoing P0/P2 MLP grid completes. The report is the revised Part B gate for
these four families. C8/GMSD P1 and P3 still wait for their own sidecars.
No candidate label value was opened before this amendment.

## Input and comparison

For each of the nine ruling-admitted sets, the promoted Part B
`features__rev4c1c4.parquet` must contain the ordered, finite f32 columns
f986–f1321 and the same ordered, unique `pair_key`s as the promoted bank's
`keys.parquet`. The previous Rev3 f0–f943 bank bytes and role-allowed label
bytes remain pinned by `admit_bank.py` and its bank snapshot. Validate every
Part B sidecar, manifest and key hash before a candidate fit; commit a dated
receipt with their exact hashes and the report hash before reading candidate
labels. The join expands collapsed stimuli by `pair_key`, preserving
`source_row_id` multiplicity. Only the allowed bank `labels__*.parquet` file
supplies the target. No held-out `pairs/`, `raw/` or `_sealed/` label read.

R0+C1, R0+C2, R0+C3, R0+C4 and R0+all C1–C4 inherit the same D1/D2 rows,
folds, seeds, target transforms, models, reference bootstrap, stability and
D5 bar as P0. Each has a size-matched negative control: jointly permute the
added columns over unique keys within each reference using seed 20260923,
then expand collapsed rows. The permutation may disrupt features in a
reference but never changes labels or reference membership. The processing
order is C1, C4, C2, C3, all, with each family and its control kept adjacent;
the order is a compute schedule, not a selection rule. The registered five
families and all human-set D5 tests remain mandatory regardless of early
results. The in-progress P0/P2 grid does not block starting a candidate arm.

The registered widths are C1 `gridblk` f986–f1081 (96), C2 `ringbasis`
f1082–f1153 (72), C3 `tailhist` f1154–f1297 (144), C4 `arttype` f1298–f1321
(24), and all C f986–f1321 (336). The diagnostic model packs each added arm's
canonical IDs after f0–f943, so packed f944+j maps to that arm's ordered
canonical ID j. DVIFM f944–f985 is absent and never read; every result must
record the mapping. This makes a 1040, 968, 1016, 1088, or 1280-column
diagnostic fit, respectively.

## BVLS direction and mask, fixed before fitting

The source registry is `zensim/src/feature_defs.rs` in the landed Part B
code, SHA-256 `4966b213c93af2a6c834580321fb0d4b685b46f35e95dd87fbfb38ae634c4fc5`
as inspected 2026-09-24 12:21 UTC. The existing 372-row
`feature_sign_mask_2026-05-26.tsv` applies unchanged to f0–f371, and bank
f372–f943 remain free. The Rust BVLS sign-mask parser supports
`pin_geq0` and `free`; it has no nonpositive pin. Thus every new C1–C4
coefficient is **free**, without a sign claim:

| Family | Registry direction | Packed BVLS mask |
|---|---|---|
| C1 mag_bin1–6 and on_mean | `unsigned` (signed mass / mean) | free |
| C1 onoff_ratio | `higher_is_worse` | free |
| C2 mag_bin1–6 | `higher_is_worse` | free |
| C3 p95/p99/max across four maps | `higher_is_worse` | free |
| C4 blur/noise/bleed | `higher_is_worse` | free |

Every `higher_is_worse` new signal has an expected nonpositive quality
coefficient, but pinning it nonnegative would invert the direction. Free is
the preregistered feasible mask for this solver; fitted signs must be reported
and no direction inference may be made from an unconstrained coefficient.
Lasso remains unconstrained. The nonnegative-distance MLP head and dial checks
remain separate requirements.
