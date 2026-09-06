# Tables and the dense contract — the census, and why a value scan must not decide

Increment **C** of [`../docs/PLAN_CRUFT_PURGE_2026-09-06.md`](../docs/PLAN_CRUFT_PURGE_2026-09-06.md)
(gates C.1 / C.3 / C.4). Tool: `rescore_parquet --densify`. Census:
`/mnt/v/output/zensim/purge-2026-09-06/inc2/table_densify_census.txt`.

---

## 1. The headline, and it inverts the obvious plan

The plan expected to shrink wide tables by dropping structural-zero columns.
**MEASURED across 7 roots and 62 parquets: there is almost nothing to drop, and
the columns a naive converter WOULD have dropped are mostly not droppable.**

| root | regime | declared populated | all-zero columns MEASURED | declaration drops |
|---|---|--:|--:|--:|
| `2026-09-05-…-372-postC` | `v1-372` | 372 | **0–8** per leg | **0** |
| `2026-08-30-full-features-372` | `v1-372` | 372 | 0–8 | **0** |
| `2026-09-04-imazen26-anchor-372` | 372 | 372 | **0** on all 4 legs | **0** |
| `ext944-era2r4-2026-09-01` | `folded720append2pools` | **944** (`0-943`) | **39–49** | **0** |
| `r1b-pools944-2026-08-30` | `folded720append2pools` | 944 | 39–49 | **0** |
| `r1b-ctrl944-2026-08-30` | `folded720append2` | 728 (`0-155,372-943`) | 255–256 | **216 (22.9 %)** |
| `v2-eval-924-2026-07-27` (grids) | *(no regime in manifest)* | — | **247** of 924 | undeterminable |

One root family — the `folded720append2` arms, whose declaration genuinely
excludes `f156..371` — has a real reduction. Everything else is already
dense-by-declaration.

## 2. Finding 1 — an all-zero column is not an absent column

The postC 372 root declares every one of its 372 ids populated, and a full-column
scan still finds all-zero columns: **`f25` in `aic3` (600 rows), `f12` in
`konjnd` (1,008) and `konjnd_jpeg504` (504), and EIGHT columns in `ext_sdr25`
(50 rows)**. Those are small-corpus accidents — the same prune **class 3**
`zensim-validate/src/prune.rs` refuses to act on ("inert on a corpus … NOT
mathematically dead"). Dropping them would make those tables unreadable by every
bake that reads those ids, for a saving of 8 columns out of 372.

So `--densify` takes the populated set from the **DECLARATION** and uses the
full-column scan only as a **GATE**: every id the declaration drops must be
all-zero across every row, or the conversion REFUSES. The declaration decides
what to drop; the data has to agree. `--keep-ids` is mandatory (there is no
"infer it from the values" mode), and `--scan-only` reports the census without
writing.

## 3. Finding 2 — a 944 table declares 39 ids the walk never writes

Every leg of `ext944-era2r4-2026-09-01` has **the same 39** all-zero columns:

```
720 721 754-772 805 806 822 823 856 857 873 874 907 908 927 928 932 933 937 938 942 943
```

They are the `APPEND_SKIP_B_SCALE0` cells — `feature_defs::placement_admits`
says in so many words *"the 944 walk never computes that append cell"* — plus
the reference-only (`LUMA_MEAN_REF`) and HDR-gated (`HL_BIN1/2`) append2 slots.
**The registry's `folded720append2pools` regime declares `0-943`, and
`Plan::emit` agrees with it**, so both declarations overstate what is populated
by 39. That is the ruling's own shape ("a layout where … features aren't
computed") one level up from the bakes increment 2A fixed.

**It is registered, not fixed, and the reason is measured.** Teaching
`populated_slots` the placement rules would make `feature_set::check` start
reporting `SlotsNotPopulated` for any bake reading one of the 39 — and shipped
**CHdr reads 8 of them** (`f927/928`, `f932/933`, `f937/938`, `f942/943`:
`HL_BIN1`/`HL_BIN2` per scale, structurally zero on an SDR route). A new refusal
on a shipped bake is a product decision. Pinned by
`a_944_regime_declares_39_ids_the_walk_never_writes`, which fails when the fix
lands — which is the point of pinning it.

## 4. The converter, proven on real bytes

`rescore_parquet --densify` is a mode on the canonical parquet-rewrite owner,
not a new script. Two passes: a full-column scan, then a copy that moves kept
columns **by reference** (the arrays are not rebuilt, so a kept column's bytes
cannot change). It writes a `<output>.densify.json` carrying the source path +
sha256, the output sha256, the row count, and the kept and dropped id lists.

**MEASURED end to end** on `r1b-ctrl944-2026-08-30/ext_aic3.parquet`
(600 rows, 944 → 728 feature columns, `--keep-ids 0-155,372-943`):

* **Gate C.1 — 730 of 730 kept columns bit-identical** to the source
  (`to_bits()` on every float column, exact equality on the rest), row count
  equal, row order preserved.
* **Gate C.3 — all 216 dropped columns verified all-zero** on every one of the
  600 rows, by a full scan of the source. Zero carried a value.
* **Gate C.4** — manifest written with both sha256s and both id lists; the
  source root is **not modified and not deleted**.

## 5. What was NOT converted, and why that is the right answer

**No live root was rewritten.** A dense-by-id table is one the loaders cannot
read yet, so converting a root today would produce a correct artifact that
nothing can score — strictly worse than its source. The measured payoffs make
that trade obviously bad: 0 % on every 372 root and every `…pools` 944 root,
and 22.9 % on one experiment-arm family.

**The loaders now REFUSE such a table instead of truncating**, which was the
correctness-critical half and is landed. All five feature-column discovery walks
in `parquet_loader.rs` collapse into two owners (`feature_column_run`,
`feature_column_run_by_name`); both find the contiguous run byte-for-byte as
before, and both fail loud when a `f<id>` column exists past the end of the run.
Before this, a dense `f0..f155, f372..f943` table loaded as **156-wide with no
error at all** — the same defect class as a positional slice in a scorer: the
numbers come back and they are about the wrong columns. Gated by
`a_gapped_dense_by_id_table_is_refused_not_truncated`, with a negative control
proving the same table minus the columns past the gap still loads as before.

**Registered, not run:** the id-indexed loader (rows keyed by id rather than by
position), after which converting the `folded720append2` arms is one command per
file with the gates above already in place.
