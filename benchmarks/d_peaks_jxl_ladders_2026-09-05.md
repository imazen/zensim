# D-peaks JXL floor ladders — visual page + fair-board cells (2026-09-05)

**Lane:** `claude-dpeaks-show`, jj sibling workspace `~/work/zen/zensim--dpeaks-show`
(forgotten + removed on completion). **Scope discipline, unchanged from every prior
lane in this chain: `zensim/src/profile.rs` and `zensim/weights/` were not opened for
writing.** Nothing installs from this record — this is a presentation + board-promotion
pass over data three prior lanes already produced and gated:
[`d_peaks_jxl_floor_2026-09-05.md`](d_peaks_jxl_floor_2026-09-05.md) (classifies the 4
failing jxl ladders as raw-model INVERSIONs) and
[`d_peaks_slot_ablation_2026-09-05.md`](d_peaks_slot_ablation_2026-09-05.md) (isolates
the cause to feature f162; `minus_f162` cures A7r but trades in a new A4 floor
regression — neither arm ships).

---

## 1. The ladders page

`http://192.168.50.44:3300/zensim/dpeaks372-2026-09-05/jxlfloor/ladders/index.html`
(curl-verified `HTTP 200`, both via the LAN IP and `localhost:3300`).

For each of the 4 failing references, the page shows the reference plus 5 JXL steps —
the 4 lowest quality settings on the postC dial grid's normalized 0-100 scale (`q=0,
8, 16, 24`; `q=0` is the largest butteraugli distance, the most aggressive setting)
plus one mid-ladder step for context (the actual grid point nearest the ladder's own
`(q_min+q_max)/2`, which resolved to `q=48` — `distance≈13` — on all 4 ladders). Each
cell pairs a Mitchell-downscaled full frame (≤512px long side) with a native 1:1 crop
of the **same** detail region (origin stated per ladder, picked by eye against a
downscaled preview of the reference — never from a codec output) so the actual coded
bytes are visible, not a resample of them:

| ladder | crop origin (native px) | content |
|---|---|---|
| `2b79a18d1b7537e0_818x1022` | `(498, 0)`, 320×320 | camera body / lens edge, in-focus against a defocused foreground |
| `96a0024c685ead3f_1024sq` | `(704, 704)`, 320×320 | "APTEKA" shop sign + window mullions + a thin wire |
| `b2e6e2b5969eaf25_1022x818` | `(340, 498)`, 320×320 | handwritten music annotation ("gelt") over staff lines |
| `f65a24b7e176eb47_1022x818` | `(260, 150)`, 320×320 | dense notation with a slur crossing the staff |

Each tile is labelled with its q value and a per-ladder table gives all four scorers
(ssim2 truth, shipped `ZensimProfile::D`, `lam1em3` raw + dial, `minus_f162` dial) at
every shown step, with the inverting pair highlighted. Decoded/cropped/downscaled ONLY
through `zenpng` + `zenresize` via the new `zensim-bench/examples/ladder_tile_gen.rs`
example (`full`/`crop` modes, same decode/downscale primitives as the existing
`m3_fixture_gen.rs`); no ImageMagick or other foreign imaging tool touches a pixel
anywhere in this path — the page's own HTML/CSS lays out the 48 already-rendered PNG
tiles (no `montage` needed). Page builder:
[`scripts/dpeaks_jxl_ladders_page.py`](../scripts/dpeaks_jxl_ladders_page.py).

### 1.1 The four inversions

| ladder | failing step pair | ssim2 truth (monotone) | D shipped (monotone) | `lam1em3` raw (inverts) | `lam1em3` dial (inverts) | `minus_f162` dial (cured) |
|---|---|---|---|---|---|---|
| `2b79a18d1b7537e0_818x1022` | q0→q8 | 24.9818 → 25.2696 | 6.7495 → 7.2817 | **0.4375 → 0.4301** | **6.3552 → 5.0949** | 8.2588 → 8.4547 |
| `96a0024c685ead3f_1024sq` | q0→q8 | 11.3006 → 13.0728 | 2.8002 → 3.9489 | **0.4066 → 0.4058** | **0.8613 → 0.7105** | 2.3449 → 3.0907 |
| `b2e6e2b5969eaf25_1022x818` | q16→q24 | 22.0590 → 25.1280 | 23.8823 → 26.6100 | **0.5503 → 0.5453** | **21.4659 → 20.5533** | 20.7503 → 23.3394 |
| `f65a24b7e176eb47_1022x818` | q16→q24 | 55.9647 → 57.3499 | 29.9599 → 31.1435 | **0.6402 → 0.6280** | **36.6636 → 33.6663** | 33.5493 → 36.7432 |

All four numbers reproduce the two prior lanes' published tables exactly (cross-checked
positionally, not re-derived); `minus_f162` is monotone-increasing across the full
bottom-4 window on all 4 ladders, confirming the slot-ablation lane's `A7r 1.0000`
finding at the per-cell level shown here. ssim2's own truth is monotone on all 16 rows.

---

## 2. Board cells

Both candidates re-scored on the postC 372 root exactly as `dialgate_arms.sh` scores
them (`ZL_ERA=postC`: `instruments/dial_grid_372col_postC_2026-09-05.parquet`,
matching negtail/identity probes, `--gaddr-tail-pins product`, ssim2 grid truth from
`ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv`), with the full 12-corpus set
(`cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy` — the
postC root carries every one of them) rather than dialgate's 5-corpus quick-grade
subset, plus the M3/M3a diffmap-coherence sweep (`scripts/m3a_sweep.sh --grid full`,
27 cells each) `run_full_eval.sh` would add. `bake_verdict`'s own feature-set-id report
states the era mismatch explicitly rather than silently correcting across it: both
bakes were fit on era `v1pre` (the frozen `safesyn.npz` gram, sha256 `904a8e80…`,
built before the option-C flip) and are scored here on era `v1postc` (the runtime
extractor's own era) — the SAME era pairing the two prior lanes used, stated in each
fulleval's `features_root.era` field rather than left implicit.

| | `Dpeaks_lam1em3` | `Dpeaks_lam1em3_minus_f162` |
|---|---|---|
| source bake | `sweep/bakes/Dsweep_lam1em3_dial.bin` | `slots/bakes/minus_f162_dial.bin` |
| sha256 | `4490e64b…` | `fcf4e4d4a090…` |
| CID22 (signed SROCC) | 0.87179 | 0.87284 |
| KonJND (\|SROCC\|) | 0.59740 | 0.62349 |
| AIC-3 | 0.77722 | 0.78484 |
| TID | 0.79479 | 0.79647 |
| KADID | 0.80346 | 0.79865 |
| composite (`product_composite`) | 0.8252 | 0.8286 |
| M3 coherence (n=27) | 0.269833 | 0.211944 |
| M3a coherence (n=27) | 0.881859 | 0.831707 |
| G-ADDR (vs `peer_ssim2`) | jxl `A7r` FAIL (order_fail=4) | jxl `A7r` PASS, **new A4 FAIL** |
| G-ADDR headline | NOT SHIPPABLE — regression FAIL / contract PASS | NOT SHIPPABLE — regression FAIL / contract PASS |
| family (board toggle group) | `D-peaks candidates` (new; `gauntlet.family_of` gained a `Dpeaks*` branch) | same |

Every rank number above reproduces the two prior lanes' `dialgate_arms.sh` output
positionally (CID22/KonJND/AIC-3/TID/KADID all match to 5 decimal places).

**Pipeline, all owner tools, nothing re-derived:** `bake_verdict --regime 372
--features-root <postC root> --dial-grid <postC grid> --negtail-probe <postC>
--identity-probe <postC> --gaddr-tail-pins product --gaddr-grid-truth <ssim2 tsv>
--gaddr-json <out>.gaddr.json --full-json <out>.full.json` → `scripts/m3a_sweep.sh`
(same fixtures/grid `run_full_eval.sh` uses) → `jq` injects `m3_coherence`/`m3a_coherence`/
`m3_n`/`m3a_n`/`m3_dropped_mass_pct` into the `.full.json` (the exact three-line jq
pattern `run_full_eval.sh` itself runs) → `scripts/promote_fulleval.py --verdict
<out>.full.json --name <board-name> --out-dir /mnt/v/output/zensim/reports/fulleval`
→ `scripts/promote_fulleval.py --graft-into <board>.fulleval.json --graft-gaddr
<out>.gaddr.json`. The graft was **not** a no-op: `bake_verdict`'s own inline
`dial.addressability` (populated automatically now that the probe flags are supplied
directly to `--full-json`/`--fulleval`, unlike the 433 pre-existing board rows this
gate landed after) carries two extra keys (`tail_pin_set`/`tail_pins`) the graft's
canonical 14-key shape drops, and the graft adds the `dial_gaddr_source` provenance
stamp (path + sha256 of the `--gaddr-json` sidecar) neither direct-embed path writes on
its own — so running `--graft-gaddr` per the brief was correct, not redundant, even
though the *addressability verdict itself* (pass/fail per axis) was already right.

**`family_of()`** (`scripts/v_next/gauntlet.py`) gained one branch: any board name
starting `Dpeaks` returns `"D-peaks candidates"`, placed right after the `peer_`
check. Both new rows carry `"family":"D-peaks candidates"` in the regenerated board's
embedded DATA (grepped, not assumed).

### 2.1 Fairness tier — corrected from the brief's guess

Both rows render **FAIR-NOTED** (glyph `◐`, confirmed via `--dump-row` scoped by an
exact-match `#compare=` hash to dodge the name-collision below) — correctly on the
fair board — but as **`k = —` (UNGROUPABLE)**, not **`k=1` UNREPLICATED** as the task
brief predicted. Measured, not assumed: `zenpredict inspect --weights` on both bakes'
embedded `zentrain.repro` shows `{"tool":"bake_dial_refit fit-lasso", "solver":"lasso",
"argv":[...], "grams":[...]}` — no `seed`/`init_seed`/`sample_seed` key at all, because
a frozen-Gram coordinate-descent lasso fit is deterministic and has no seed to record.
`gauntlet.py`'s `k` computation (`fairness_of`, line ~880) reads
`(o.get("repro") or {}).get("seed")` for exactly this k=1-vs-ungroupable distinction;
absent seed → `k=None` → UNGROUPABLE. This does **not** change which tier the rows land
in — `FAIR_CRITERIA`'s own docstring states "(d) never fails on its own" and groups
UNGROUPABLE with UNREPLICATED under FAIR-NOTED — so **the practical outcome the brief
cared about (the rows appear on the fair board) holds**; only the specific badge text
differs from the prediction. Recorded here per "NEVER CLAIM FALSE COMPLETION" rather
than silently matching the brief's wording.

**A real, pre-existing naming-collision gotcha, found while verifying this:**
`--dump-row Dpeaks_lam1em3` (bare, without a disambiguating `--hash`) returns
`Dpeaks_lam1em3_minus_f162`'s row, because the harness's diagnostic `--dump-row` helper
resolves by `startsWith`, not exact match, and `Dpeaks_lam1em3` is a literal prefix of
`Dpeaks_lam1em3_minus_f162`. This is a debug-convenience gap in `--dump-row`, **not** a
bug in the page's own `#compare=` resolution (which checks exact match first, per its
own docstring, and passed both directions once scoped — see §3). Not fixed here (out of
scope; the harness's own doc comment already flags `--dump-row`'s matching as
"starting with").

### 2.2 `peer_ssim2` is absent from the regenerated FAIR board — a pre-existing, structural fact

The task's suggested 4-way compare (`…,peer_ssim2`) does not resolve cleanly against
`summer_gauntlet_fair.html` as regenerated: `peer_ssim2` fails criterion `a_repro` (its
fulleval carries `"repro": null`, `"bake": null` — it is scored from stored refmetrics
tables, not a trained bake, per its own `model.kind: "reference-metric"`), lands
LEGACY, and `--fair-only` drops LEGACY rows from the embedded page data entirely (130
of 467 bakes are *embedded*, not merely hidden). This applies to `peer_ssim2` on
**every** regen of the fair board, not something this pass introduced — grepping the
page source for `"peer_ssim2"` finds it once, but that occurrence is the app's own
search expression (`DATA.bakes.find(b=>b.name==='peer_ssim2')`), not an embedded row.
Not changed here: exempting reference-metric rows from `a_repro` is a fairness-filter
design decision (it would move which rows land in FAIR on every future regen, for every
board, not just these two), out of this presentation lane's scope. See §3 for the
compare URL that works as specified on the all-rows board, where `peer_ssim2` **is**
present (LEGACY rows are only hidden there, not dropped).

---

## 3. Compare URLs (harness-verified)

**FAIR board, 3-way (shipped D + both candidates) — clean, no banner:**
```
http://192.168.50.44:3300/zensim/reports/summer_gauntlet_fair.html#compare=d_id100_negrich@did100lane,Dpeaks_lam1em3_minus_f162,Dpeaks_lam1em3
```
Verified: `node scripts/v_next/gauntlet_render_check.js summer_gauntlet_fair.html --hash
'compare=d_id100_negrich@did100lane,Dpeaks_lam1em3_minus_f162,Dpeaks_lam1em3' --expect-visible
'd_id100_negrich@did100lane,Dpeaks_lam1em3_minus_f162,Dpeaks_lam1em3' --expect-no-banner` →
PASS (exact 3 rows, fragment order, no banner).

**All-rows board, 4-way (adds the ssim2 mentor row) — clean, no banner:**
```
http://192.168.50.44:3300/zensim/reports/summer_gauntlet.html#compare=d_id100_negrich@did100lane,Dpeaks_lam1em3_minus_f162,Dpeaks_lam1em3,peer_ssim2
```
Verified the same way, 4 rows, fragment order, no banner. `d_id100_negrich@did100lane`
is shipped `ZensimProfile::D`'s own board row (`bake_sha256` matches
`d_sdr_add156_id100_negrich_dial_2026-09-05.bin` exactly — grepped across
`reports/fulleval/*.fulleval.json`, not assumed from the name).

Both base HTML files curl-verified `HTTP 200` at the LAN URL (fragments are
client-side and are not sent to the file server, so the base-file check is what curl
can verify; the harness above is what verifies the fragment behavior).

---

## 4. Board regen + gates

```
export ZEN_PANEL_BIN=<repo>/target/release/panel
cd scripts/v_next
python3 bandwise_dashboard.py --fulleval-dir /mnt/v/output/zensim/reports/fulleval \
  --fair-only --out /mnt/v/output/zensim/reports/summer_gauntlet_fair.html
python3 bandwise_dashboard.py --fulleval-dir /mnt/v/output/zensim/reports/fulleval \
  --out /mnt/v/output/zensim/reports/summer_gauntlet.html
../../scripts/v_next/gauntlet_gates.sh /mnt/v/output/zensim/reports/summer_gauntlet_fair.html
../../scripts/v_next/gauntlet_gates.sh /mnt/v/output/zensim/reports/summer_gauntlet.html
```
(`--fairness-tsv` omitted per the brief; `fairness_of()` is computed fresh from each
fulleval + the registry, so the TSV is a cached copy of the same computation, not an
independent input.)

| board | rows embedded | bytes | cap |
|---|--:|--:|---|
| `summer_gauntlet_fair.html` | 130 of 467 | 10,250,231 (9.78 MB) | **under 12 MB** |
| `summer_gauntlet.html` | 467 of 467 | 23,094,166 (22.02 MB) | over cap, reported not trimmed (pre-existing documented policy — cutting further means dropping rank/dial data) |

`gauntlet_gates.sh` **PASS** on both files (GATE 1 `node --check`, GATE 2 DOM-shim
render, GATE 4a-4e `#compare=` behaviors, gate 3 all 467 fulleval JSONs strict-valid).
130-of-467 / 97-of-433 in the 2026-09-04 fair_gauntlet record reflects concurrent
lanes' promotions between that pass and this one, not a defect here — the +2 from this
lane are `Dpeaks_lam1em3` and `Dpeaks_lam1em3_minus_f162`.

---

## 5. Reproduction

```sh
cd ~/work/zen/zensim
cargo build --release -p zensim-validate --bins
cargo build --release --manifest-path zensim-bench/Cargo.toml --example ladder_tile_gen --features m3-fixtures
cargo build --release -p zensim --features custom-profiles,feature-regime-v2 --example diffmap_block_coherence

# ladders page
python3 scripts/dpeaks_jxl_ladders_page.py

# board cells (repeat per bake: lam1em3 -> Dpeaks_lam1em3, minus_f162 -> Dpeaks_lam1em3_minus_f162)
I=/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments
ROOT=/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC
SSIM2=/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv
target/release/bake_verdict --bake <bake.bin> --name <label> --regime 372 \
    --features-root "$ROOT" --dial-grid "$I/dial_grid_372col_postC_2026-09-05.parquet" \
    --negtail-probe "$I/negtail_probe_372_postC_2026-09-05.parquet" \
    --identity-probe "$I/identity_probe_372_postC_2026-09-05.parquet" \
    --gaddr-tail-pins product --gaddr-grid-truth "$SSIM2" \
    --gaddr-json <out>.gaddr.json \
    --corpora cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25,imazen26,nonphoto,hfnlproxy \
    --full-json <out>.full.json
scripts/m3a_sweep.sh --bake <bake.bin> --bin target/release/examples/diffmap_block_coherence \
    --grid full --label <label> --logdir <out-dir> --tsv <out>.m3a_cells.tsv > <out>.m3a.kv
# jq-inject m3_coherence/m3_n/m3a_coherence/m3a_n/m3_dropped_mass_pct from the kv file
python3 scripts/promote_fulleval.py --verdict <out>.full.json --name <board-name> \
    --out-dir /mnt/v/output/zensim/reports/fulleval
python3 scripts/promote_fulleval.py --graft-into /mnt/v/output/zensim/reports/fulleval/<board-name>.fulleval.json \
    --graft-gaddr <out>.gaddr.json

# board regen + gates: see §4
```

Artifacts: `/mnt/v/output/zensim/dpeaks372-2026-09-05/jxlfloor/ladders/` (page + 48
tiles), `/mnt/v/output/zensim/dpeaks372-2026-09-05/board/` (per-bake verdict/gaddr/m3a
work files), `/mnt/v/output/zensim/reports/fulleval/Dpeaks_lam1em3{,_minus_f162}.fulleval.json`.
