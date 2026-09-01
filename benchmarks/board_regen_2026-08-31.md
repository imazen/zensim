# Board currency pass — KonJND JPEG-504 ruler + registry badges + curation (2026-08-31)

**What this executes.** Three things changed under the summer-gauntlet board on
2026-08-31 that the rendered board did not yet reflect: the KonJND default ruler was
corrected (`6e508793`), the annotations registry was repaired (`cb76bd5c`), and era-2
went default-on (`515001dc`). This pass re-scores the affected cells **through the
owners**, regenerates the board through the canonical pipeline, and gates it. Nothing
here re-derives a statistic: `bake_verdict` measures, `promote_fulleval.py` grafts under
a byte-identity gate, the dashboard reads.

**Board HTML:** `/mnt/v/output/zensim/reports/summer_gauntlet.html` (19.95 MB, 379
fulleval files → 362 rendered rows after the 17 dominated exclusions), browseable at
<http://localhost:3300/zensim/reports/summer_gauntlet.html>. The pre-change file is kept
at `summer_gauntlet_pre_konjnd504_2026-08-31.html` (19.79 MB); nothing was deleted.
**Gates:** `scripts/v_next/gauntlet_gates.sh` **PASS** — `node --check` on both script
blocks; DOM-shim render (362 bakes, 12 sections, 18 tables, 512 rows, sort clicks reorder
the ATTACHED tables, ECharts mounts + per-panel SSR), **960 registry-annotated scoreboard
cells carry ⚠**.

Artifacts (as-run scripts, fresh verdicts, comparison tables):
`/mnt/v/output/zensim/board-konjnd504-2026-08-31/`. Pre-change copies of every touched
cell: `/mnt/v/output/zensim/reports/fulleval-snapshots/pre-konjnd504-2026-08-31/`
(+ `SHA256SUMS`).

---

## 1. The 17 diluted-ruler cells, re-scored

`bake_verdict`'s 372 corpus map defaulted to `konjnd_features_372col_2026-05-15.parquet`
(all **1,008** refs, JPEG *and* BPG) while every 720/944-class row scores the JPEG **504**
half. `6e508793` moved the default to `KONJND_JPEG504_372_SLOTS`. Registry entry
`konjnd-372-diluted-ruler-pre-2026-08-31` enumerated the 17 board cells that had been read
on the diluted file; the other 361 were already at n=504.

**All 17 are now on the JPEG-504 ruler, and every one of the 378 board cells reads
KonJND at n=504** (verified by scanning `rank.konjnd.n` across the board). The 361
correct cells were not touched.

| board cell | root | KonJND before (n=1008) | KonJND after (n=504) | Δ | composite before | after |
|---|---|---:|---:|---:|---:|---:|
| `ADD156_safesyn_only_raw_lasso@cur372` | current 372 (2026-08-30) | 0.4462 | 0.5332 | +0.0870 | 0.81606 | 0.82415 |
| `Ebothg_scr0_5_dial@cur372` | current 372 | 0.2707 | 0.4103 | +0.1397 | 0.82857 | 0.84156 |
| `T_appT_b372_lam1e-3` | stored 372 (2026-05-15) | 0.5306 | 0.5674 | +0.0368 | 0.82340 | 0.82545 |
| `T_appT_b372_lam1e-3@cur372` | current 372 | 0.5411 | 0.5510 | +0.0098 | 0.82290 | 0.82381 |
| `b_sdr_linear_cid80_inclwinsor_dense_dial` | stored 372 | 0.5466 | 0.5935 | +0.0468 | 0.82906 | 0.83342 |
| `b_sdr_linear_cid80_inclwinsor_dense_dial@cur372` | current 372 | 0.6497 | **0.5194** | −0.1303 | 0.84074 | 0.82862 |
| `bhdr_linear_shaped_cvvdpmix@cur372` | current 372 | 0.4804 | 0.6380 | +0.1576 | 0.79491 | 0.80957 |
| `cl_tfm_corruption_LQ_MLP_s13@cur372` | current 372 | 0.2720 | 0.4298 | +0.1578 | 0.84081 | 0.85549 |
| `mlp_2L_diverse_H128@cur372` | current 372 | 0.5048 | 0.5402 | +0.0354 | 0.86882 | 0.87211 |
| `peer_butteraugli` | peer per-pair TSV | 0.3578 | 0.2586 | −0.0992 | — | — |
| `peer_cvvdp` | peer TSV | 0.0482 | 0.0562 | +0.0080 | — | — |
| `peer_iwssim` | peer TSV | 0.1859 | **0.5704** | +0.3845 | — | — |
| `peer_ssim2` | peer TSV | 0.4786 | 0.5272 | +0.0486 | — | — |
| `v02_bvls_NO_shaping@cur372` | current 372 | **0.7275** | 0.3296 | −0.3979 | 0.81596 | 0.77895 |
| `v02_bvls_shaped@cur372` | current 372 | 0.1703 | 0.3908 | +0.2206 | 0.76328 | 0.78380 |
| `v47_strict_QAT_native@cur372` | current 372 | 0.3996 | 0.4431 | +0.0435 | 0.81800 | 0.82205 |
| `winner_dial_Ebothg_hfgain_winsor_dial@cur372` | current 372 | 0.3352 | 0.4305 | +0.0953 | 0.81703 | 0.82589 |

### 1.1 How, and what proves it is surgical

**13 bake cells.** `bake_verdict --full-json` on the same bake bytes (sha-verified against
the board's `bake_sha256`), at the same features root per cell — default 2026-08-30 root
for the eleven `@cur372` rows, `--features-root .../2026-05-15-full-features` for the two
genuine stored-root rows (`benchmarks/board_era_rows_2026-08-30.md` §2). As-run:
`board-konjnd504-2026-08-31/rerun_verdicts.sh`, 68 s for all 13.

Three independent checks:

* **Non-KonJND reproduction.** Every other rank corpus in the fresh verdict is
  **BIT-IDENTICAL** to the stored board value on **12 of 13** cells. The 13th
  (`T_appT_b372_lam1e-3`) differs only in band structure (`band_scheme` gains a
  `recut_from` key from `--rebuild-bands`; `kadid.bands` 10 vs 7) and by carrying an
  `sdr25` block the board row predates — **no `srocc` differs anywhere**. This also
  answers the era-2 question: the era-2 flip does not move a `bake_verdict` rank read,
  because that path scores stored parquet features and never runs the extractor.
* **Cross-run agreement.** The eleven `@cur372` KonJND values agree with round-4b's
  independent `kon504` shim run (`eval372-roster-2026-08-30/kon504/`, a *different*
  binary — pre-fix — and a *different* path to the same 504 rows) to **max |Δ| = 0.0e+00**.
* **Published-number agreement.** `6e508793`'s table (ADD156 0.5332, `B` 0.5194) and
  `konjnd-372-full-file-dilution-2026-08-29`'s stored-root `B` = 0.5935 all reproduce to
  4 dp.

The graft is `promote_fulleval.py --graft-into … --reslice-rank konjnd`, which refuses to
write unless every other key of the board file is byte-identical. It passed on all 13.

**4 peer rows.** Peers are not bakes — they are scored from the stored per-pair tables in
`reports/refmetrics/` by `build_peer_fullevals.py`, which read all 1,008 KonJND rows. The
owner now carries a per-corpus `ROW_FILTERS` entry keeping `dist_path` containing
`/jpeg/`. That subset was verified to **BE** the ruler population: 504 rows whose `pjnd`
is bit-identical to the 504 parquet's `human_score` for every ref (max |Δ| = 0.0). Every
non-KonJND block of all four peer rows is byte-identical to the pre-change snapshot.

---

## 2. Ordering changes

**KonJND column (378 ranked cells): 355 cells change rank position** — 16 of the 17
re-scored cells move, and 339 untouched cells are displaced by them (their values are
unchanged). The column head is completely rearranged:

| cell | KonJND rank before → after | value |
|---|---|---|
| `peer_iwssim` | 364 → **7** | 0.1859 → 0.5704 |
| `v02_bvls_NO_shaping@cur372` | **1** → 295 | 0.7275 → 0.3296 |
| `cl_tfm_corruption_LQ_MLP_s13@cur372` | 342 → 171 | 0.2720 → 0.4298 |
| `ADD156_safesyn_only_raw_lasso@cur372` | 130 → 14 | 0.4462 → 0.5332 |
| `b_sdr_linear_cid80_inclwinsor_dense_dial@cur372` | 2 → 25 | 0.6497 → 0.5194 |
| `bhdr_linear_shaped_cvvdpmix@cur372` | 62 → 2 | 0.4804 → 0.6380 |

**The board's KonJND leader was a ruler artefact**: `v02_bvls_NO_shaping@cur372` held
rank 1 on the diluted file and sits at 295 on the ruler every other cell uses. The new
leader is `bhdr_linear_shaped_cvvdpmix` (0.6403, untouched — it was already at n=504),
with its own `@cur372` twin second at 0.6380. And the D2 inversion is now on the board:
ADD156 (0.5332) reads **above** shipped `B` (0.5194) on the `@cur372` pair, where the
diluted file had `B` ahead by +0.204.

**composite column (374 ranked cells): 188 cells change rank position**, 12 of them
re-scored and 176 displaced. **The top 10 is unchanged.** Largest moves:
`cl_tfm_…@cur372` 167 → 75, `v02_bvls_NO_shaping@cur372` 231 → 287,
`b_sdr…@cur372` 168 → 211.

**`freeze_check --select` over the board: the selection is UNCHANGED** —
`W10L9P_s4005_packed`, 8/8 floors, `selection_composite` 0.9876, before and after.
Thirteen cells' selection numbers move, and three cross the F2 KonJND floor (≥ 0.43):

| cell | floors before → after | cause |
|---|---|---|
| `winner_dial_Ebothg_hfgain_winsor_dial@cur372` | 6/8 → **7/8** | KonJND 0.3352 → 0.4305 |
| `v47_strict_QAT_native@cur372` | 5/8 → **6/8** | 0.3996 → 0.4431 |
| `v02_bvls_NO_shaping@cur372` | 5/8 → **4/8** | 0.7275 → 0.3296 |

---

## 3. ADD156's standing, stated precisely

The round-35 claim `SELECTED: ADD156_safesyn_only_raw_lasso — 6/8 floors,
selection_composite 0.9644` reproduces **exactly**, in the pool it was run on — ADD156 vs
shipped `B`:

```
| 1 | ADD156_safesyn_only_raw_lasso            | era-bridge | 6/8 | 0.8213 | 0.9540 | 0.9644 |
| 2 | b_sdr_linear_cid80_inclwinsor_dense_dial | era-bridge | 6/8 | 0.8292 | 0.5968 | 0.9187 |
```

Two clarifications a reader needs, because "ranks FIRST" is pool-dependent:

* **Over the whole board (374 non-peer cells) ADD156 is rank 101**, and the rule selects
  `W10L9P_s4005_packed` (8/8 floors). The PRIMARY term is floor count, and no 372-class
  model reaches 8/8.
* **Within its own class (era-bridge) ADD156 is 3rd**, behind
  `winner_dial_Ebothg_hfgain_winsor_dial` (7/8, 0.9674) and `Ebothg_scr0_5_dial`
  (6/8, 0.9775). It leads shipped `B` by +0.0457 on `selection_composite` — a gap that
  *narrowed* from +0.0493 when `B`'s KonJND was corrected upward (0.5466 → 0.5935; the
  brief's 0.9151 for `B` is the pre-fix value, now **0.9187**).

**The board has no place to show `selection_composite` or the floor count.** Its "Gate
scorecard" is a different system (the CODEC_TARGET_GOALS soft-gates), and the scoreboard
carries no freeze-floor column. No panel was invented for this pass; the `--select` table
lives here and in `board-konjnd504-2026-08-31/`.

---

## 4. Badges: what the registry repair actually did

`cb76bd5c` fixed two silent-drop classes — 3 findings sat outside `entries[]`, and 19 had
scopes the matcher could not evaluate. **MEASURED correction to the premise that "badges
have been missing across the board":** the repair converted all 22 to the explicit
`{"manual": …}` documentation-only form, which by definition badges **no** cell. Counting
(cell, entry) matches with gauntlet's own `_ann_matches` over the same board:

| registry state | entries | match ≥1 cell | match ZERO cells | (cell,entry) matches |
|---|---:|---:|---:|---:|
| pre-repair (`cb76bd5c^`) | 42 | 23 | 19 | 2,959 |
| post-repair (`d4cf5383`) | 47 | 25 | 22 | 3,355 |
| + this pass | 51 | 27 | 24 | 3,632 |

The +396 between the first two rows comes from the two **machine-scoped** entries round 35
added, not from the repair of the 22. What the repair genuinely bought is *readability*:
`freeze_check` now prints all 22 by name, and the 3 formerly out-of-array findings
(`live-cross-root-targets-divergent-2026-08-29`,
`konjnd-372-full-file-dilution-2026-08-29`, `tid-retired-to-train-2026-08-29`) are now in
the page's `DATA.annRegistry` with their reasons — **verified present in the emitted
HTML**, where before they were absent from the file the loader reads.

Badge rendering verified end-to-end on real payload (`gauntlet_render_check.js
--dump-row`, which now uses deep text so it can see the appended badge span):

| cell | rendered evidence | finding |
|---|---|---|
| `ADD156_safesyn_only_raw_lasso@cur372` | `KonJND 0.533⚠`, `composite 0.824⚠` | `konjnd-372-diluted-ruler-repaired-2026-08-31` on exactly the two columns its `fields` cover |
| `linear924_M5_lam2e-3` | `composite 0.773⚠`, KonJND clean | `composite-stale-after-rank-graft-2026-08-28` (the worst-stale cell) |
| `peer_ssim2` | `KonJND 0.527⚠`, `composite —` | repaired-ruler badge + NOT-MEASURED composite (§5.2) |
| `peer_cvvdp` | row tooltip lists `peer-cvvdp-nonstrict-json-2026-08-31` | its `fields` are `per_pair.*`, which no scoreboard column maps to, so it surfaces in the row tooltip, not a column badge |
| `W10L9P_s4005_packed` | **every column clean** | the no-finding control: not resliced, not in the 17 |

`konjnd-372-diluted-ruler-pre-2026-08-31` was moved to documentation-only in the same
edit and superseded by `konjnd-372-diluted-ruler-repaired-2026-08-31` — leaving a
machine scope that asserts "this cell reads the diluted ruler" on 17 cells that no longer
do would have been a false badge. The entry itself is untouched in `reason`/`evidence`
(append-only), and its `fix_path` now points at the repair.

---

## 5. Two board-integrity defects found in passing (pre-existing, not from this pass)

### 5.1 `composite` is stale on 276 cells

`composite` (the Rust `product_composite`) is a weighted function of `rank`:
CID22 1.00 + imazen26 0.50 + nonphoto 0.30 + KonJND 0.20 + AIC-3 0.10 + AIC-4 0.05,
normalised by the present weights. The 2026-08-28 family-aware reslice replaced
`rank.imazen26` + `rank.nonphoto` — two of those six terms — on 276 board cells and did
**not** update the stored composite, so each disagrees with its own rank blocks.
Measured across all 378 cells by recomputing the published weights from each cell's own
rank: **276 stale, |Δ| up to 0.02370** (`linear924_M5_lam2e-3`), median 0.00170. That is
the scoreboard's default sort key. Registered `composite-stale-after-rank-graft-2026-08-28`
(kind `invalidated`, per-cell table in `board-konjnd504-2026-08-31/stale_composites.json`).

Not affected: `freeze_check`'s `balanced_composite` and `--select`, which recompute from
`rank` rather than reading the field.

**The mechanism is fixed at the owner.** `promote_fulleval.py --reslice-rank` now carries
the verdict's `composite` in the same gated write, under a gate that the verdict and the
board already agree on every rank corpus they share other than the replaced ones — so the
carried value is the board's own composite with only the replaced terms moved, and no
formula is duplicated. That fix also repaired `T_appT_b372_lam1e-3`'s stale value
(0.82340 → 0.82545) as a side effect of its KonJND graft. Re-running the 2026-08-28
grafts through the fixed promoter would clear the other 276.

### 5.2 The four peer rows led the default sort on an incomparable scale

Peer fullevals carry no `composite` (`bake_verdict` does not run on a reference metric),
so `load_fulleval` fell through to the legacy `_composite` fallback — the **unnormalised**
sum (max ≈ 1.65) — while every bake carries the **normalised** Rust value (max 1.0). The
four peers therefore occupied the top four rows of the board's default composite sort at
**1.112–1.420** against the best bake's 0.872, reading as "ssim2 beats every model".
Fixed: a peer/reference-metric row with no emitted composite now publishes **NOT
MEASURED** (em-dash, sorted last), which is the board's standing rule for a value nobody
measured. Top of the composite sort is now `mlp_2L_diverse_H128@cur372` 0.872.

That exposed a latent hole in the render gate: its scoreboard sort test mapped a null
composite to a `-1e9` sentinel, which encodes "nulls first when ascending" — the opposite
of the page's documented "nulls last either direction" — and it never checked *where*
nulls landed. The test now asserts both halves (non-null prefix ordered, nulls a suffix)
in all three directions. **Negative control:** a copy of the page patched to sort nulls
first fails all three assertions. This strengthens the gate; nothing was relaxed.

### 5.3 One board cell is not valid strict JSON

`peer_cvvdp.fulleval.json` carries **73 bare `NaN` tokens** in the carried
`per_pair.imazen26` (43/5,000) + `per_pair.nonphoto` (30/5,000) arrays, written by the
2026-08-28 reslice graft. Python's `json` accepts `NaN`, so the board renders; `serde_json`
does not, so `freeze_check --select` over a glob including it **aborts**. It is the only
such file of 378, and it is identical in the pre-change snapshot — not introduced here.
Scoping `--select` to the non-peer cells is the correct read (peers are reference metrics,
not selection candidates), not a workaround. Registered
`peer-cvvdp-nonstrict-json-2026-08-31`.

A second latent bug in the same area was fixed on contact: `build_peer_fullevals.py`
**overwrote** its output, silently deleting every block a later program had grafted onto a
peer row (the imazen26/nonphoto/hfnlproxy reslices and the dial/corruption blocks). It now
merges — it owns the corpora it computes and the identity scalars, and carries everything
else through byte-identical, printing what it carried.

---

## 6. Curation

`gauntlet.py CURATED_BOARD` already carried ADD156, shipped `B` (both era halves) and the
944 flagship `W10L9P_s4005_packed` (the board-wide `--select` winner). Added:

* **`Q7b_pools_g0.2_a0.2_b0.97`** — the W-LIN round-7b *candidate the registered rule
  names* (`benchmarks/wlin_round7_rawframe_2026-08-30.md`; 3,583 B, 944-input, 5/5 bars +
  G-RANGE PASS). It was not on the board at all; promoted from its own fulleval through
  `promote_fulleval.py` (+ `--set-block-profile`). It carries **7 of the 14 board
  corpora** — its verdict ran a partial corpus list — so the other seven render as
  NOT-MEASURED em-dashes, never zeros. Its KonJND was already on the 504 ruler (regime 944).
* **A `W-LIN` family toggle.** `Q7b_*` / `T7b_*` / `H7b_*` / `wlin*` / `copperline_wlin*`
  stems fell through `family_of` to "pre-944 era" — wrong (they are 944-input) and
  invisible. This is the standing "check `family_of` on every new stem pattern" rule.

`ADD156_safesyn_only_raw_lasso@cur372` was deliberately **not** promoted into the curated
set: the era doc's four curated pairs are the decision-relevant ones, the `@cur372` half
is per-pair-stripped by the registered size rule, and the row `--select` names is the
unsuffixed one, which is curated. It stays one family-toggle click away.

---

## 7. Reproduction

```sh
# 1. re-verdict the 13 bake cells on the corrected ruler (68 s)
/mnt/v/output/zensim/board-konjnd504-2026-08-31/rerun_verdicts.sh
# 2. graft rank.konjnd (+ the composite it feeds) onto the board, byte-identity gated
/mnt/v/output/zensim/board-konjnd504-2026-08-31/graft_konjnd.sh
# 3. rebuild the 4 peer rows on the JPEG-504 subset (merge, never clobber)
ZEN_PANEL_BIN=…/panel python scripts/v_next/build_peer_fullevals.py
# 4. regen + gates
python scripts/v_next/bandwise_dashboard.py --fulleval-dir /mnt/v/output/zensim/reports/fulleval \
    --out /mnt/v/output/zensim/reports/summer_gauntlet.html
scripts/v_next/gauntlet_gates.sh /mnt/v/output/zensim/reports/summer_gauntlet.html
```

Binaries: `bake_verdict` / `freeze_check` / `panel` / `bake_block_profile` built at
`6e6efb1a`+ (post the era-2 flip), pinned by sha256 in
`~/tmp/board-regen/bin/SHA256SUMS`; `bake_verdict` sha256
`dc9f1e50a79fbf77456873966076c7b3eda99e4a21bba54fb3182a77dfceae20`.
