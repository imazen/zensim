# Board orientation fix on the JND axes, and AIC-4 is refreshable (2026-09-22)

Two defects, both verified before anything changed. Neither one was a model
finding: the display was wrong, and so was the registry.

Artifacts (block storage, not git): `/mnt/v/output/zensim/board-orientation-fix-2026-09-22/`
— fresh verdicts, repair roster (`apply.tsv`), sha256-checked pre-repair copies
of all 518 touched board cells (`board_before/`), before/after board HTML and
their data diff. AIC-4 read: `/mnt/v/output/zensim/aic4-refresh-2026-09-22/`.

## 1. Every correct model was marked INVERTED on AIC-4 and SDR25

### 1.1 What was wrong

`aic4` and `sdr25` store `q_jnd`: a JND *distance* from the original. It rises
with distortion, and `check_target_orientation.py` declares both `DISTORTION`.
`bake_verdict`'s `sign_is_meaningful` exempted only `konjnd`, so for aic4/sdr25
it did two wrong things:

- It pinned the per-reference statistic to `HigherIsBetter` (the 2026-08-04 pin,
  `730a386e`). A bake that ranks every ladder correctly then reads as −0.9 per
  reference, with 100% of references backwards.
- It printed `⛔INVERTED` in the SROCC cell whenever the signed pooled value was
  negative, which is the correct sign for a quality score against a distance.

The board's pooled accessor (`gauntlet._signed`) already took the declared
orientation. Only the per-reference statistic was never aligned. The registry
then explained the artifact away as a corpus property
(`aic4-corpus-wide-per-ref-inversion`), and the gauntlet special-cased it
(`corpusWide`).

### 1.2 Evidence (reproduced)

| check | measured |
|---|---|
| Frozen controls, stored aic4 per-ref mean / % backwards | MT914_B −0.9119 / 100%, MT914_D −0.9517 / 100%, R915_basic228 −0.9481 / 100%, R915_y60 −0.9408 / 100% |
| Same four on AIC-3 (quality-oriented) | +0.9173, +0.9554, +0.9520, +0.9561, 0% backwards |
| AIC-4 label monotonicity (`aic4_sample.parquet`) | `human_jnd` rises on **270 of 270** ladder steps (5 sources × 6 codecs × 9 steps) |
| Organisers' CVVDP vs `human_jnd`, within each source | −0.969, −0.984, −0.976, −0.906, −0.974; pooled −0.961 |
| Board scan, 524 cells with aic4 | pooled signed < 0 on 523. Per-ref < 0 on 332 (the pinned-era cells: 331 at 100% backwards, plus `ebothg_m504` at −0.018). Per-ref > 0 on 186 (pre-pin `Auto` cells, whose pooled-sign flip happened to give the declared orientation). |

The 60%-backwards median in the retired entry was that 332/186 split, not a
property of the corpus.

### 1.3 What changed

- `bake_verdict.rs`: `DISTORTION_ORIENTED = ["aic4","konjnd","sdr25"]` is a gated
  mirror of `EXPECTED_ORIENTATION`, using the same parse test `freeze_check`
  already has. The per-reference statistic takes `LowerIsBetter` on those three
  corpora and `HigherIsBetter` everywhere else, never `Auto`. The summary cell,
  the `⛔INV` flag and the SVG bars show the **orientation-aligned** signed SROCC
  (the raw value on quality-oriented corpora, its negation on aic4/sdr25).
  KonJND keeps its `|SROCC|` cell. The JSON `srocc`/`srocc_signed` are unchanged.
- Regression test `distortion_oriented_jnd_corpus_is_not_marked_inverted_for_a_correct_bake`
  covers aic4, sdr25, KonJND and cid22/aic3/kadid. **Mutation check:** with the old
  pin restored, the test fails (log `~/tmp/devin/boardfix_mutation.log`).
- `promote_fulleval.py`: `--repair-tag` records the program behind a repair
  (default keeps the old tag). The orientation-independent equality gate now
  compares numbers by value, because 43 stored blocks spell `or` as `0` against a
  fresh `0.0`. Any differing value is still refused.
- `gauntlet.py`: the `corpusWide` branch is gone.
- Registry: `aic4-corpus-wide-per-ref-inversion` is **retired** (scope → manual,
  kind invalidated, the original text kept). New entries:
  `aic4-sdr25-per-ref-declared-orientation-REPAIRED-2026-09-22`,
  `hya-w084-per-ref-stale-orientation-2026-09-22` and
  `sdr25-372-root-table-orientation-unverified-2026-09-22`.

### 1.4 The stored-row repair

Every one of the 518 board cells carrying an aic4 or sdr25 per-reference block
was re-verdicted for `aic4,sdr25` with the fixed binary (dial and corruption
grids skipped). Cells without a recorded root or regime got candidate roots,
and the gate picked among them. Each result went through the existing sha-gated
`promote_fulleval --repair-rank-orientation` (tag
`declared-orientation-2026-09-22`). A block is replaced only when every
orientation-independent field reproduces exactly and `per_ref_mean` is an
exact negation (|d| ≤ 1e-9).

| outcome | aic4 | sdr25 |
|---|---:|---:|
| repaired (exact sign flip) | **330** | **253** (5-reference q_jnd table) |
| reproduced unchanged (already in the declared orientation) | 139 | 155 |
| not reproduced, already declared-orientation (pre-pin `Auto` with pooled < 0 = `LowerIsBetter`) | 47 | 16 |
| not reproduced, stale | 1 (`HYA_w084`) | 1 (`HYA_w084`) |
| repaired, then restored (see below) | — | 62 |

Four cells needed member lists the cells don't carry. `BAL_E1/E2/E3` reproduce
as two `W10L9P` seeds at equal weight. `D_shipped@dguard2` reproduces as the
`zensim/weights` copy of its bake (same sha256) on the post-C root. `HYA_w084`
could not be reproduced: its Q7b member is not recorded. It stays stale and is
registered.

**The 62 restored sdr25 cells.** 372-class cells read sdr25 from the 372 roots'
`ext_sdr25.parquet` (sha256 `4f567646…`). That table is **different** from the
944/924 one: 10 references × 5 rows, with `human_score ∈ {2,6,7,9,10}` on every
reference. No builder or provenance is recorded for it. The distortion
declaration describes the q_jnd table, not this one, so its orientation is
unverified. Those 62 blocks were put back exactly as stored. The registry entry names
all 63 cells that read the table (one was never repaired in the first place). Consequence: `bake_verdict` now pins distortion for every sdr25 read,
so a fresh 372-root verdict prints the opposite per-ref sign from these stored
cells. The registry entry forbids comparing the two.

Integrity, checked independently of the tool's gates: across the 518 cells, the
only fields that differ from the pre-repair copies are `rank.aic4/sdr25.per_ref_mean`,
`.frac_negative`, `.or` (spelling only, `0` → `0.0`) and `rank_graft_sources`.

### 1.5 Board before / after

Both boards were rendered from the same fulleval directory: *before* with the
pre-fix gauntlet and registry (temporary workspace at `122ebeac`, removed
afterwards) on unrepaired data, *after* with this change on repaired data.
Every data difference, by bake name:

| field | cells | nature |
|---|---:|---|
| `rank.aic4.per_ref_mean` / `frac_negative` | 330 | sign flip / 100% → 0% |
| `rank.sdr25.per_ref_mean` / `frac_negative` | 253 | sign flip |
| `rank.aic4.or`, `rank.sdr25.or` | 43 / 42 | `0` → `0.0` |
| `rank.*.band_scheme` | 90 / 89 | absent → `null` (fresh block carries the key) |
| annotations | all | `aic4-corpus-wide…` removed from all 548 shared cells; `sdr25-372…` added on 63; `hya-w084…` on 1 |
| `colorIndex` | 234 | another lane added `peer_cvvdp_aicfhd` meanwhile (index shift) |
| `train_date` | 1 | bake-file mtime, resolved relative to the rendering workspace |

No fairness tier changed. Per-reference findings from the board's `failures()`
rule, mirrored for aic4/sdr25:

| | aic4 | sdr25 |
|---|---|---|
| before | 332 "corpus-wide" watch notes | 304 "corpus-wide" watch notes |
| after | 2: `HYA_w084` (stale, registered) and `ebothg_m504` (genuine: −0.018 / 60% backwards, unchanged by the fresh verdict) | 52: **50 on the 10-reference 372 table** (orientation unverified, now annotated on all 63 cells that read it), `HYA_w084` (stale, registered), and `sota944_Q_lin944_hdr` (genuine: pooled **+0.086** on the q_jnd table, per-ref −0.236 — it ranks sdr25 in the undeclared direction) |

Frozen controls, aic4 per-ref mean / % backwards: before −0.9119 / −0.9517 /
−0.9481 / −0.9408 at 100%; after **+0.9119 / +0.9517 / +0.9481 / +0.9408 at 0%**.
Their AIC-3 cells did not move.

The board gate script fails identically before and after, on
`TRAIN comparison must use a served report URL: 2026-09-19-dvifm-*`. That check
belongs to another lane's DVIFM entries. Because the gates must pass before the
served board is replaced, **`summer_gauntlet.html` was not overwritten**. Both
builds are in the artifact directory.

## 2. AIC-4 is not "unrefreshable"

The claim (registry entries `eval372-current-root-copied-corpora-2026-08-30` and
`eval-root-era3-2026-08-30-mixed-era-copies`, and `eval_roots.rs`) rested on the
reconstructed-JND CSV having been deleted. It was. But the same labels, with
confidence intervals and the organisers' metric columns, are committed in
`site/data/parquet/aic4_sample.parquet` (`709f4597`), and all 305 crop and 305
full-resolution PNGs are on disk.

- `build_fr_corpus_pairs.py aic4` now reads the committed labels. It reproduces
  the 2026-07-20 manifest: 300/300 rows, same order, labels equal to max |d|
  1.8e-7 (float32), SROCC 1.0. It writes a crop manifest and a full-resolution
  manifest to `/mnt/v/output/zensim/aic4-refresh-2026-09-22/`.
- **Re-extraction works.** `extract_features_372col --corpus pairs-tsv` gives
  300/300 rows in 3.5 s, with labels equal to the stored table. It is a current
  HEAD (post-option-C) read, so all four blocks differ from the stored pre-fix
  table (basic max |d| 5.9e-3, peaks 9.5e-2, masked 8.8e-4, IW 7.8e-4). It was
  **not** swapped into any root; that is a separate, deliberate decision.
- The registry and `eval_roots.rs` now say this.

### 2.1 Frozen read, crops vs full resolution (T0, eval-only)

Registered in `docs/DATA_SPLITS.md`'s exposure ledger before the read. Frozen
models only: nothing was fitted, calibrated or selected. **Five sources is a
sanity check** (is any ladder backwards?), **not a selection axis**. These
models have prior AIC-4 exposure from the September 14/15 panels. Scores come
from `score_pairs_tuner` (profiles + `BakeScorer::ensemble`) and
`peer_metric_pairs`. Statistics come from the Rust `panel` via
`scripts/canonical_corpus/aic4_refresh_read.py`. Full table:
`aic4_read.md`/`.json` in the artifact directory.

| model | crop SROCC | crop KROCC | full SROCC | full KROCC | Δ SROCC | per-source SROCC, crop | per-source SROCC, full |
|---|---:|---:|---:|---:|---:|---|---|
| R915_y60_h32_ens5 | 0.9136 | 0.7479 | 0.9141 | 0.7541 | +0.0004 | 0.881–0.984 | 0.902–0.973 |
| R915_basic228_h128_ens5 | 0.9159 | 0.7470 | 0.9174 | 0.7529 | +0.0015 | 0.890–0.980 | 0.912–0.978 |
| MT914 matched B (= profile B) | 0.8904 | 0.7078 | 0.8944 | 0.7153 | +0.0040 | 0.867–0.952 | 0.861–0.950 |
| MT914 matched D (= profile D) | 0.9331 | 0.7813 | 0.9466 | 0.8091 | +0.0135 | 0.892–0.984 | 0.925–0.978 |
| profile PreviewV0_2 | 0.9110 | 0.7444 | 0.9131 | 0.7539 | +0.0021 | 0.875–0.981 | 0.899–0.974 |
| profile C | 0.9144 | 0.7529 | 0.9171 | 0.7602 | +0.0027 | 0.876–0.979 | 0.880–0.975 |
| our ssim2 (fast-ssim2) | 0.9127 | 0.7460 | 0.9054 | 0.7383 | −0.0073 | 0.883–0.975 | 0.887–0.974 |
| our butteraugli p3 | 0.8969 | 0.7264 | 0.8933 | 0.7198 | −0.0037 | 0.856–0.967 | 0.897–0.960 |
| our butteraugli max | 0.8652 | 0.6859 | 0.7529 | 0.5683 | −0.1122 | 0.848–0.970 | 0.638–0.949 |

The organisers' published columns (crops only): CVVDP 0.9609 / 0.8410, IW-SSIM
0.9507, MS-SSIM 0.9409, HDR-VDP-3 0.9329, SSIMULACRA2 0.9125.

Read-outs:

- **Every model, peer and published column ranks in the declared direction,
  pooled and within all 5 sources, on both legs.** No per-source inversion anywhere.
- **The pixel crop read reproduces the board's feature-table AIC-4 SROCC to 6
  decimals** for all four frozen controls (0.913634, 0.915926, 0.890424,
  0.933075). The stored aic4 feature table and the pixel path agree for these
  models.
- The matched B/D bakes and profiles B/D produce the same numbers: the profiles
  embed those bakes (sha256 verified).
- Full-resolution scoring moves the zensim models by +0.0004 … +0.0135. It moves
  our ssim2 by −0.0073 and butteraugli-max by −0.112. With 5 sources this is
  descriptive, not a ranking.

## 3. What this does not do

- `HYA_w084` is still stale (Q7b member unknown). It is registered.
- The 372-root sdr25 table's orientation is unidentified. It is registered, and
  those 62 cells were left as stored.
- The served board was not replaced (the pre-existing gate failure above).
- The re-extracted AIC-4 table was not installed in any eval root.
- Dated historical records that repeat "aic4 unrefreshable" were not rewritten:
  `board_era_rows_2026-08-30.md:145`, `balance_campaign_2026-08-28.md:3611`,
  `eval372_current_root_2026-08-30.md:86`, `add156_ship_audit_2026-08-31.md:538`,
  `v1_extractor_drift_2026-08-30.md:380`, `DATASET_HISTORY.md:1344`,
  `feature_sets_registry.json:106` (live note), and
  `docs/history/CLAUDE-through-2026-09-07.md:556`. The registry is the live
  correction.
- The CVVDP comparator was not touched.

## 4. Addendum 2026-09-26: the served board is replaced

The gate failure in §1.5 was the three `2026-09-19-dvifm-*` TRAIN comparisons, whose `report_url`
pointed at `/zensim/benchmarks/*.md` (not served; 404). Each screen's committed record, JSON, pointer
and prereg are now published byte-identically under
`/mnt/v/output/zensim/reports/dvifm-{screen,screen2b,screen2c}-2026-09-19/` with a `FILES.json`
sha256 index, and the three entries point at those served directories (`659580a8`). Rebuilt from
that commit, both boards pass `gauntlet_gates.sh` (gates 1, 2, 3 and 4a–4e):

| file | before (kept) | now |
|---|---|---|
| `summer_gauntlet.html` | `summer_gauntlet_pre_orientation_2026-09-26.html` (29,816,306 B, sha256 `e3cddf74…`) | 29,796,685 B, sha256 `66504af3…`; 566 fulleval files, 549 rendered |
| `summer_gauntlet_fair.html` | `summer_gauntlet_fair_pre_orientation_2026-09-26.html` (15,734,811 B, sha256 `30c23a87…`) | 15,733,164 B, sha256 `fa587daf…`; 186 rendered |

On the served board the four frozen controls read aic4 per-reference +0.9119 / +0.9517 / +0.9481 /
+0.9408 at 0% backwards, and no cell carries the retired corpus-wide note. The fair board is still
above the 12 MB cap named in `gauntlet.py --fair-only`'s help, as the replaced one was.

