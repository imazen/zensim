# FAIR GAUNTLET — 2026-09-06 board-hygiene re-issue

Board hygiene lane. The `fair_gauntlet_2026-09-04.md` mechanism (`gauntlet.fairness_of`,
§2 of that doc) is unchanged — this is a **re-run of the same mechanical criteria over
the current fulleval population**, not a new audit design. Nothing here re-derives a
statistic; every number is read from a fulleval JSON, the annotations registry, or the
freshly-written audit TSV.

## 0. Why this was needed

`summer_gauntlet_fair.html` was last built 2026-09-05 19:58 (10,884,706 B, 97 of 433
fullevals). The all-rows companion was rebuilt 2026-09-06 10:04 (508 fullevals) by a
concurrent lane, but nobody re-ran the fair-only build afterward, so every fulleval
promoted since 2026-09-05 19:58 was simply **absent from the fair board's file** — not
misclassified, just never rendered there. Verified directly (not assumed): the
currently-computed `fair` field for these cells is correct wherever it can be checked
(e.g. `BOA_A_plain_s4004` -> `VERIFIED-FAIR, k=3` in the already-regenerated all-rows
board), so the fix is a rebuild, not a code change.

## 1. Audit re-issued today

New audit: `/mnt/v/output/zensim/reports/fairness_tiers_2026-09-06.tsv` (508 rows,
902,912 bytes, sha256 `eded6d3227e299d6276bd4559180d9a8ceb7b5a1e4751bdfb6d95a336d1886e7`).
Pointer: [`fairness_tiers_2026-09-06.pointer.md`](fairness_tiers_2026-09-06.pointer.md).
**`fairness_tiers_2026-09-04.tsv` was never opened for writing by this pass** (sha256 and
mtime both unchanged: `652465` bytes, `2026-09-05 19:58`).

*Note found in passing, not fixed here*: that "2026-09-04" file was itself already at
479 rows on disk (not the 433 its own `.md` documents) before this pass touched
anything — some earlier lane grew it in place rather than dating a new file, exactly
the anti-pattern this task was told to avoid repeating. Left as-is per the instruction
to never touch it further; flagged so the drift doesn't get re-attributed to today.

### 1.1 Counts — eligible / excluded, whole 508-row population

| tier | rows | Δ vs the 481-row pre-508 snapshot (02:39 today) |
|---|--:|--:|
| **VERIFIED-FAIR** | 125 | +27 |
| **FAIR-NOTED** | 46 | +2 |
| **LEGACY** (excluded from the fair board) | 337 | +0 |
| **total** | 508 | +29 |

| failing criterion (LEGACY rows; not mutually exclusive) | rows |
|---|--:|
| `e_no_invalidated` | 278 |
| `a_repro` | 41 |
| `f_ens_units` | 31 |
| `b_era` | 2 |

(`c_no_train_eq_val`, `d_seed`-alone and `g_split` still never independently fail a
row today, matching the 2026-09-04 mechanism exactly — `d_seed` demotes to FAIR-NOTED,
never to LEGACY, per criterion (d)'s own rule.)

## 2. The 29 new cells, and what they actually turned out to be

Comparing the fresh 508-row audit against the immediate-predecessor snapshot (481
rows, generated 02:39 today, preserved as `fairness_tiers_2026-09-06_pre508.tsv.bak`)
isolates exactly what changed in this pass:

| family | count | tier | k | notes |
|---|--:|---|---|---|
| `BOA_A_plain_s{4004,4005,4006}` … `BOA_H_anchorlad_s{4004,4005,4006}` (8 recipes × 3 seeds) | 27 | **VERIFIED-FAIR** | 3 each | genuinely new since 02:39; each recipe is its own k=3 seed group |
| `D_guard12_p999@dguard2` | 1 | FAIR-NOTED | — (ungroupable) | already present at 02:39, absent only from the stale HTML |
| `D_shipped@dguard2` | 1 | FAIR-NOTED | — (ungroupable) | same |

**Correction to the brief's framing**: `D_guard12_p999@dguard2` / `D_shipped@dguard2`
and the fast-class `fc2_372_S156_H32_*` / `fc2_372_S228_H128_*` / `fc2_944_ORACLE_*` /
`fc2_944_S228_H32_*` (12 cells, all **VERIFIED-FAIR**, k=3) and `Dpeaks_lam1em3{,_minus_f162}`
(2 cells, FAIR-NOTED, k=1/ungroupable) were **already correctly classified** in the
02:39 snapshot — they were never `fair: {}`. Checked directly against the raw
`*.fulleval.json` files: none carries a `fair` field at all (it is computed at render
time by `gauntlet.fairness_of`, never stamped into the source JSON), so there is no
literal `fair: {}` state to find in any file on disk. The only genuinely-new rows in
this pass are the 27 `BOA_*` cells; the rest of the "not yet classified" families were
already classified and merely unrendered until this rebuild.

No literal cell named `S372_S228_H128_p` exists. Read as shorthand for the fast-class
372-regime family, verification below uses `fc2_372_S228_H128_s4004` (one of the
family; noted explicitly rather than silently substituted).

## 3. Board regeneration

Both boards rebuilt with the unchanged Reproduce commands from
`fair_gauntlet_2026-09-04.md`, `--fairness-tsv` pointed at the fresh file:

| board | rows | bytes | vs cap |
|---|--:|--:|---|
| `summer_gauntlet_fair.html` | **171 of 508** (was 97 of 433) | **12,419,271 (11.85 MiB)** | under the 12 MiB cap (margin ≈164 KB) |
| `summer_gauntlet.html` | 491 of 508 (17 coverage-gate exclusions, unchanged rule) | 25,294,787 (24.12 MiB) | over cap; reported per the standing rule, not trimmed |

### 3.1 The fair board exceeded the cap on first build, and what was stripped

The first fair-only build (fresh audit, no other change) came out at **12,655,576
bytes (12.067 MiB)** — over the 12 MiB cap by about 73 KB. Diagnosis: the size driver
was **not** the newly-fair rows' base stats (baseline JSON per row is small); it was
per-pair scatter embedding, which the existing render rule (`gauntlet.py` line ~1163,
2026-08-04) already restricts to `curated and tier != LEGACY` rows. Measuring per-cell
scatter payload found two pairs of **seed-duplicate curated rows both carrying full
scatter for the same k-group**:

| kept (stays full) | stripped (grid-interior sibling) | seed_group | k | why this pair |
|---|---|---|---|---|
| `LSTAR_s4021_packed` | `LSTAR_s4022_packed` | `9facbc8a2223` | 7 | same recipe, sibling seed of an already-represented k=7 group; `s4021` is the group's first-listed CURATED_BOARD member |
| `w11_s4014_final` | `w11_s4014_e050` | `4ec838fb58b9` | 7 | **same seed (4014)**, not just the same group — exactly the doc's own `duplicate-promotion-same-seed-2026-09-04` pattern ("one training run promoted twice"); `final` is the terminal checkpoint, `e050` an intermediate one |

Nothing was dropped silently and no row left the board: both cells keep every scalar
stat (composite, rank, dial, gates, model card, annotations) and their scoreboard rows.
Only the embedded per-image point cloud was removed, and it is **not lost**:

* The complete pre-strip fulleval JSON for both is preserved verbatim at
  `/mnt/v/output/zensim/reports/fulleval/_perpair_full_backup/<name>.fulleval.full_backup.json`
  (byte-identical to the pre-strip file, verified at write time).
  * `LSTAR_s4022_packed.fulleval.full_backup.json` — 1,325,875 bytes
  * `w11_s4014_e050.fulleval.full_backup.json` — 1,326,109 bytes
* Independently, both cells' `bake` paths exist on disk (`LSTAR_s4022_packed` ->
  `/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/LSTAR_s4022_packed.bin`;
  `w11_s4014_e050` -> `.../W11J_s4014_ckpts/ckpt_epoch050_s4014_packed.bin`), so the
  identical per-pair table is also re-derivable by re-running `bake_verdict` against
  the same features-root and corpora.
* The stripped source files carry `per_pair_stripped: true` and a
  `per_pair_stripped_note` explaining the reason and backup path in-line, mirroring
  (not duplicating) `promote_fulleval.py --strip-per-pair`'s own field names; that tool
  itself was not used here because these two fullevals have no `source_verdict` (they
  were not produced via `promote_fulleval.py`), so re-running it was not an option
  without risking a different re-promotion than what is already on the board.

Rebuilt after the strip: fair board **12,419,271 bytes**, comfortably under the 12 MiB
(12,582,912-byte) cap. Tier counts unchanged by the strip (171 of 508; the two cells
stay `VERIFIED-FAIR`, k=7 each).

## 4. Gates

`scripts/v_next/gauntlet_gates.sh` on both files, **PASS / rc=0** on both:

* fair board: node --check (2 script blocks), DOM-shim render harness (171 bakes, 15
  sections, 21 tables, 480 rows, 8 svgs), ECharts SSR on every panel kind, badge check
  (1,143 registry-annotated cells), failure panel (6 measured / 0 NOT MEASURED, 140
  findings), all five compare-fragment gates (4a-4e), strict-JSON validation of all 508
  source fulleval files.
* all-rows board: same suite, 491 bakes, 4,253 registry-annotated badge cells, all
  compare gates pass.

## 5. Serving + named-cell verification

```
curl -s -o /dev/null -w '%{http_code} %{size_download}' http://192.168.50.44:3300/zensim/reports/summer_gauntlet_fair.html
-> 200 12419271   (matches the file on disk exactly; also 200 via localhost:3300)
```

`node scripts/v_next/gauntlet_render_check.js summer_gauntlet_fair.html --hash
'#compare=BOA_F_nonneg32_s4004,D_guard12_p999@dguard2,fc2_372_S228_H128_s4004'
--expect-visible <same three> --expect-no-banner` — **PASS**: all three resolve, no
missing-id banner, scoreboard holds exactly the three rows in fragment order. Verified
individually first, then combined (all four invocations rc=0).

```
http://192.168.50.44:3300/zensim/reports/summer_gauntlet_fair.html#compare=BOA_F_nonneg32_s4004,D_guard12_p999@dguard2,fc2_372_S228_H128_s4004
```

## Reproduce

Unchanged from `fair_gauntlet_2026-09-04.md`'s Reproduce block; only the `--fairness-tsv`
destination path changes per re-issue date. See that doc for the exact commands.
