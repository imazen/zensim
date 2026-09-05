# D-peaks jxl A7r floor failures — per-ladder classification, and the lever is in the weights (2026-09-05)

**Lane:** `claude-dpeaks-jxlfloor`, jj sibling workspace `~/work/zen/zensim--dpeaks-jxlfloor`
(forgotten + removed on completion per the workspace-cleanup rule).
**Registered by:** [`dial_addressability_gate_2026-09-04.md`](dial_addressability_gate_2026-09-04.md)
§16.4 — every arm in [`d_peaks_lambda_sweep_2026-09-05.md`](d_peaks_lambda_sweep_2026-09-05.md)
fails `A7r` on `jxl` alone (0.818–0.909 against the mentor's **0.9697**), while shipped
`ZensimProfile::D` reads jxl **1.0000**. This lane classifies every failing jxl ladder for the
best-rank arm (`lam1em3`) and `Dpeaks`/`lam2em3`, and decides whether the fix belongs in the
spline or the fit.

**Scope discipline, unchanged from every prior lane in this chain:** `zensim/src/profile.rs` and
`zensim/weights/` were not opened for writing. Nothing installs from this record.

**Headline: the lever is in the WEIGHTS, not the spline.** All 8 failures (4 ladders × 2 arms) are
`INVERSION`s, zero `TIE`s, zero `CLAMP`s, and in every one of the 8 the pre-spline RAW model output
is *already* inverted at the exact same step pair the dial fails on. §5 below is the direct
evidence; §7 is why part 2 of the brief (the spline-only `--anchor-weight` lever) is correctly
skipped rather than attempted; §8 registers the two fixes that remain, unrun.

---

## 1. Method — owner tools only, no re-implemented stats

Binaries built fresh from this workspace (`main@origin` 90926c32f06f, the commit that landed
`A7r`/`FloorRepresentabilityRule`): `cargo build --release -p zensim-validate --bins`
(81 s, rc=0). Reused, not touched: the postC grid
(`instruments/dial_grid_372col_postC_2026-09-05.parquet`, 4,424 rows, sha256 `506bdadf…`), its
`sweep/work/dial_grid_postC_with_dummy_target.parquet` companion (same rows + an unused
all-zero `human_score` a prior lane appended so `predict` has a target column to load), the ssim2
truth table (`/mnt/v/output/zensim/ssim2-bar-2026-08-31/dialcells_ssim2_qv2grid.tsv`, 4,424 rows),
and the three bakes below (read-only, sha256-verified):

| label | path | sha256 |
|---|---|---|
| `lam1em3` | `/mnt/v/output/zensim/dpeaks372-2026-09-05/sweep/bakes/Dsweep_lam1em3_dial.bin` | `4490e64b…` |
| `Dpeaks` (= `lam2em3`) | `/mnt/v/output/zensim/dpeaks372-2026-09-05/bakes/Dpeaks372_id100negrich_dial.bin` | `85ae9c7c…` |
| `D` (shipped, control) | `zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin` | `921a8f67…` (matches `profile.rs`'s own doc comment) |

**Aggregate re-grade, `scripts/dialgate_arms.sh score <label> <bake> 372` at `ZL_ERA=postC`**
(`ZL_BV`/`ZL_BDR` pointed at this workspace's fresh binaries; output under
`/mnt/v/output/zensim/dpeaks372-2026-09-05/jxlfloor/arms/`) — reproduces §16.4 of the gate doc
exactly:

| bake | jxl `repr` | order_fail | clamp_fail | headline |
|---|--:|--:|--:|---|
| `lam1em3` | 0.8788 (29/33) | 4 | 0 | NOT SHIPPABLE (A7r fail / contract pass) |
| `Dpeaks`/`lam2em3` | 0.8788 (29/33) | 4 | 0 | NOT SHIPPABLE (A7r fail / contract pass) |
| `D` (shipped) | 1.0000 (33/33) | 0 | 0 | SHIPPABLE |

**Per-cell data**, all via the owner (`bake_dial_refit predict`, never a hand-rolled forward
pass): `predict --bake <bake> --corpus <dummy-target grid> --out <tsv>` for RAW (pre-spline)
units, `--score-units` added for DIAL (post-spline) units — six TSVs under
`/mnt/v/output/zensim/dpeaks372-2026-09-05/jxlfloor/work/{lam1em3,dpeaks,dship}_{raw,dial}.tsv`,
row-order-aligned to the grid parquet (`predict`'s own contract: `row_idx<TAB>pred`, file row
order).

**Classification script** (`/mnt/v/output/zensim/dpeaks372-2026-09-05/jxlfloor/classify_jxl_ladders.py`,
copy of `~/tmp/jxlfloor_classify.py`): one-off scratch, not a repo tool — same status as the
prior lane's own §4 per-codec analysis ("one-off scratch, not a repo tool"). It re-implements
**zero statistics**; it applies `dial_addressability.rs::FloorMeasure::from_grid`'s documented
rule (`bottom_k=3`, `clamp_eps=1e-9`, ordered = strictly increasing across the bottom 3 steps and
into the 4th, clamped = a bottom step within `clamp_eps` of the *instrument-wide* min unless sole
holder) directly to the owner's own predict dumps, and its aggregate output is the validation:
for every one of the three bakes it reproduces `n_ladders_at_min`, `n_fail_order`, and
`n_fail_clamp` **exactly** against `bake_verdict`'s own `codec_floor` JSON block
(`.dial.addressability.measured.codec_floor[]`, cross-checked directly from
`arms/verdict_{lam1em3,dpeaks,dship}.json`). Full stdout:
`jxlfloor/classify_jxl_ladders_output.txt`.

---

## 2. Classification — every failing jxl ladder is the SAME 4 images for both arms

**`lam1em3`** — instrument min `-57.488196` (sole holder: an avif ladder). **`Dpeaks`/`lam2em3`**
— instrument min `-56.190382` (also a sole-held avif ladder; different value because the two
lasso fits are different models, but the mechanism is identical). Both bakes fail on the
identical 4 of jxl's 33 ladders:

| image_id | `lam1em3` class | fail pair (q-steps) | `Dpeaks` class | fail pair |
|---|---|---|---|---|
| `2b79a18d1b7537e0_818x1022` | INVERSION | q=0 → q=8 | INVERSION | q=0 → q=8 |
| `96a0024c685ead3f_1024sq` | INVERSION | q=0 → q=8 | INVERSION | q=0 → q=8 |
| `b2e6e2b5969eaf25_1022x818` | INVERSION | q=16 → q=24 | INVERSION | q=16 → q=24 |
| `f65a24b7e176eb47_1022x818` | INVERSION | q=16 → q=24 | INVERSION | q=16 → q=24 |

**Zero TIEs, zero CLAMPs, in either arm.** `n_fail_order = 4, n_fail_clamp = 0` for both —
matching §1's owner cross-check exactly. (`q` is JXL's `param_kind=distance` axis; `q=0` is the
lowest configurable setting = the largest distance = the most aggressive compression, per the
gate doc's own registered rule — confirmed on the grid, not re-derived.)

---

## 3. Per-ladder detail — dial, raw, ssim2 truth, and shipped D, on the same cells

Bottom 4 q-steps for each failing ladder. `dial` = score-units (post-spline, what the product
serves); `raw` = pre-spline linear-model output (`predict` without `--score-units`); `ssim2` =
the mentor's own truth on the identical cell, from `dialcells_ssim2_qv2grid.tsv`; `D_dial`/`D_raw`
= shipped Profile D on the same cell, for reference (D passes all 4 of these ladders cleanly).
Boldface marks the failing adjacent pair.

### `2b79a18d1b7537e0_818x1022`

| q | lam1em3 dial | lam1em3 raw | Dpeaks dial | Dpeaks raw | ssim2 truth | D dial | D raw |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | **6.3552** | **0.4375** | **6.1012** | **0.4379** | 24.9818 | 6.7495 | 0.4450 |
| 8 | **5.0949** | **0.4301** | **4.4601** | **0.4283** | 25.2696 | 7.2817 | 0.4484 |
| 16 | 6.7677 | 0.4400 | 6.6800 | 0.4413 | 26.6214 | 8.1264 | 0.4541 |
| 24 | 10.0914 | 0.4621 | 10.1257 | 0.4634 | 29.2518 | 9.6499 | 0.4652 |

### `96a0024c685ead3f_1024sq`

| q | lam1em3 dial | lam1em3 raw | Dpeaks dial | Dpeaks raw | ssim2 truth | D dial | D raw |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | **0.8613** | **0.4066** | **0.8946** | **0.4082** | 11.3006 | 2.8002 | 0.4212 |
| 8 | **0.7105** | **0.4058** | **0.3171** | **0.4051** | 13.0728 | 3.9489 | 0.4278 |
| 16 | 3.6089 | 0.4216 | 3.2525 | 0.4214 | 15.2855 | 5.4783 | 0.4370 |
| 24 | 6.5857 | 0.4389 | 6.3444 | 0.4393 | 17.6785 | 7.5637 | 0.4503 |

### `b2e6e2b5969eaf25_1022x818`

| q | lam1em3 dial | lam1em3 raw | Dpeaks dial | Dpeaks raw | ssim2 truth | D dial | D raw |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 16.0888 | 0.5176 | 18.1648 | 0.5275 | 16.6793 | 19.3331 | 0.5498 |
| 8 | 17.8294 | 0.5297 | 19.5189 | 0.5388 | 19.5609 | 21.4914 | 0.5620 |
| 16 | **21.4659** | **0.5503** | **22.2387** | **0.5591** | 22.0590 | 23.8823 | 0.5747 |
| 24 | **20.5533** | **0.5453** | **21.0517** | **0.5509** | 25.1280 | 26.6100 | 0.5891 |

### `f65a24b7e176eb47_1022x818`

| q | lam1em3 dial | lam1em3 raw | Dpeaks dial | Dpeaks raw | ssim2 truth | D dial | D raw |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0 | 30.3470 | 0.6065 | 29.7955 | 0.6053 | 50.5351 | 26.4303 | 0.5881 |
| 8 | 31.6602 | 0.6168 | 31.0842 | 0.6157 | 52.6653 | 28.2158 | 0.5994 |
| 16 | **36.6636** | **0.6402** | **35.8194** | **0.6404** | 55.9647 | 29.9599 | 0.6156 |
| 24 | **33.6663** | **0.6280** | **32.5558** | **0.6254** | 57.3499 | 31.1435 | 0.6253 |

**ssim2's own truth is monotone on every one of these 16 rows** — the reference metric considers
all four ladders correctly ordered at the bottom; only the candidate models' RAW linear score
dips. `D_dial`/`D_raw` are strictly increasing across all 4 steps on all 4 images (D is `ADD156`,
`f0..155` only — no peaks block, see §6).

---

## 4. Raw-vs-spline verdict: the inversion is already in the raw model

For all 8 (4 ladders × 2 arms) failing cases, the RAW ordering at the failing pair matches the
DIAL ordering exactly — a monotone spline cannot un-invert what is already inverted underneath
it, and does not need to: it is preserving the raw model's own (wrong) order here, not causing
it.

| ladder | arm | raw pair | raw order | dial pair | dial order |
|---|---|---|---|---|---|
| `2b79a18d1b7537e0` | lam1em3 | 0.4375 → 0.4301 | ↓ inverted | 6.3552 → 5.0949 | ↓ inverted |
| `2b79a18d1b7537e0` | Dpeaks | 0.4379 → 0.4283 | ↓ inverted | 6.1012 → 4.4601 | ↓ inverted |
| `96a0024c685ead3f` | lam1em3 | 0.4066 → 0.4058 | ↓ inverted | 0.8613 → 0.7105 | ↓ inverted |
| `96a0024c685ead3f` | Dpeaks | 0.4082 → 0.4051 | ↓ inverted | 0.8946 → 0.3171 | ↓ inverted |
| `b2e6e2b5969eaf25` | lam1em3 | 0.5503 → 0.5453 | ↓ inverted | 21.4659 → 20.5533 | ↓ inverted |
| `b2e6e2b5969eaf25` | Dpeaks | 0.5591 → 0.5509 | ↓ inverted | 22.2387 → 21.0517 | ↓ inverted |
| `f65a24b7e176eb47` | lam1em3 | 0.6402 → 0.6280 | ↓ inverted | 36.6636 → 33.6663 | ↓ inverted |
| `f65a24b7e176eb47` | Dpeaks | 0.6404 → 0.6254 | ↓ inverted | 35.8194 → 32.5558 | ↓ inverted |

8/8. Over the full 4-step window the RAW rank order and the DIAL rank order are identical on
every one of the 4 ladders in both arms (e.g. `2b79a18d1b7537e0`/lam1em3: raw ranks
q8 < q0 < q16 < q24; dial ranks q8 < q0 < q16 < q24 — same order) — exactly what a strictly
increasing calibration spline must produce, and further evidence this is a rank-order property
of the fitted linear model, not an artifact the spline step introduces.

**Conclusion for the brief's branching rule: raw values are NOT ordered. The lever is in the fit
(weights), not the spline.**

---

## 5. Why: the peaks block, not the basic block

`lam1em3` (38 active coefficients) and `Dpeaks`/`lam2em3` (26 active coefficients) are different
lasso fits on the same `0..227` (basic+peaks) slice at different `--lam`, yet they invert on the
**identical 4 references** — evidence this is a property of the slice/data at those references,
not a coincidence of one λ. Shipped D fits only `0..155` (`ADD156`, no peaks block:
`bake_block_profile` on `Dpeaks` reports `v1_basic 20/156, v1_peaks 6/72`, per
`d_peaks_372_postC_2026-09-05.md` §4.2) and is clean on all 33 jxl ladders. The peaks block
(`f162-164, f211-212, f224` at λ=2e-3) is the one structural difference between a model that
inverts here and one that doesn't. This lane did not isolate which single peaks feature drives
it (out of scope — the brief asks for classification + a raw/spline verdict, not a feature
attribution), but the slice boundary is the natural first place to look if this is picked up
later.

---

## 6. Part 2 (spline-only lever) — SKIPPED per the brief's own branching rule

The brief: *"Try the spline-only lever only if the raw values are ordered."* §4 shows they are
not — every failure is a raw-level inversion. No `--anchor-weight 1.25`/`1.5` variant, no extra
spline knots, were run.

**This is not merely a scope call — the prior lane already proved a spline-only lever
structurally cannot reach this bug.** `d_peaks_lambda_sweep_2026-09-05.md` §3's "Registered
sanity control for grid (b)" establishes, by a `strip`+`cmp` control (not an assumption): *"the
CD lasso fit (`w`, `bias`, `mu`, `sd`) is computed from `--gram` alone (before any anchor row is
read) ... the lever cannot touch `w`/`bias`/`mu`/`sd` by construction."* `--anchor-weight`
(and, by the same mechanism, `extend-top`/`shared-anchor`'s knot placement) can only reshape the
**monotone output spline** — it cannot introduce a rank change relative to the raw model, because
a monotone function preserves rank order by definition. Since the fault here is a raw rank
inversion, every spline-only lever available in `bake_dial_refit` today is provably unable to fix
it, independent of which anchor rows or weights are chosen. (The brief's own exclusion of
anchoring on the grid's own jxl-floor rows — "never anchor on the eval instrument" — was moot for
the same reason: even an unrestricted spline refit cannot repair a raw inversion.)

---

## 7. Registered, NOT run: candidate fixes in the fit

Per the brief's fallback instruction. Neither was implemented or exercised this lane.

**(a) An isotonic/monotone shape term in `fit-lasso`'s solver.** The current CD lasso
(`zensim_validate::gram_lasso::lasso_cd_slice`) and its BVLS sibling (`box_cd_slice`) both fit
from a *frozen, pre-aggregated* Gram (`S`/`s`/`q`/`Y1` moments) — they never see individual rows,
only sufficient statistics, so there is no per-row ladder identity available to constrain against
inside the existing solve. A monotone-shape extension would need either (i) a genuinely new
solver path that additionally consumes a small set of *raw* `(image, codec, q)` ladder rows (not
moments) and adds an explicit non-negativity-of-adjacent-difference constraint along the q axis
for those ladders to the CD objective (a shape-constrained regression term, akin to isotonic
regression restricted to the ladder ordering), or (ii) a post-hoc monotone projection of `w`
restricted to the peaks-block coordinates, checked against a held ladder set for correctness
before packing. Both are solver-level changes, unbuilt.

**(b) Row-level up-weighting of the jxl-floor ladder rows in the training GRAM (not the spline
anchor).** This is a different mechanism from the existing `--anchor-weight` (§6): the anchor only
feeds `fit_spline_knots`, never the CD lasso's `w`. To make jxl's lowest-q rows influence `w`
itself, the frozen `.npz` gram the lasso reads (`safesyn.npz`) would need to be rebuilt (via the
`gram` subcommand) with additional moment mass contributed by those specific rows — a data-
pipeline change (extracting the jxl dial-grid's floor cells' own feature vectors and target, then
folding them into the gram at extra weight), not a CLI flag on the existing tool. Registered as
the brief specifies; not attempted, since (a) is the more direct mechanism and this lane's budget
went to classification, not solver work.

---

## 8. Nothing installed

`ZensimProfile::D` still resolves to `d_sdr_add156_id100_negrich_dial_2026-09-05.bin`;
`zensim/weights/` was read-only throughout. This record is a classification and a decision
("lever is in the fit"), not a shipped change.

---

## 9. Reproduction

```sh
cd ~/work/zen/zensim
cargo build --release -p zensim-validate --bins

BV=target/release/bake_verdict; BDR=target/release/bake_dial_refit
GRID=/mnt/v/output/zensim/dpeaks372-2026-09-05/sweep/work/dial_grid_postC_with_dummy_target.parquet
LAM1EM3=/mnt/v/output/zensim/dpeaks372-2026-09-05/sweep/bakes/Dsweep_lam1em3_dial.bin
DPEAKS=/mnt/v/output/zensim/dpeaks372-2026-09-05/bakes/Dpeaks372_id100negrich_dial.bin
DSHIP=zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin

# aggregate re-grade (reproduces the table in §1)
ZL_ERA=postC ZL_BV=$BV ZL_BDR=$BDR scripts/dialgate_arms.sh score lam1em3 "$LAM1EM3" 372
ZL_ERA=postC ZL_BV=$BV ZL_BDR=$BDR scripts/dialgate_arms.sh score dpeaks "$DPEAKS" 372

# per-cell raw + dial dumps (owner: bake_dial_refit predict)
$BDR predict --bake "$LAM1EM3" --corpus "$GRID" --out lam1em3_raw.tsv
$BDR predict --bake "$LAM1EM3" --corpus "$GRID" --score-units --out lam1em3_dial.tsv
$BDR predict --bake "$DPEAKS"  --corpus "$GRID" --out dpeaks_raw.tsv
$BDR predict --bake "$DPEAKS"  --corpus "$GRID" --score-units --out dpeaks_dial.tsv
$BDR predict --bake "$DSHIP"   --corpus "$GRID" --score-units --out dship_dial.tsv
$BDR predict --bake "$DSHIP"   --corpus "$GRID" --out dship_raw.tsv

# per-ladder classification (one-off scratch, applies dial_addressability.rs's own rule)
python3 /mnt/v/output/zensim/dpeaks372-2026-09-05/jxlfloor/classify_jxl_ladders.py
```

Artifacts: `/mnt/v/output/zensim/dpeaks372-2026-09-05/jxlfloor/{arms,work}/` (six predict TSVs,
three `gaddr_*.json`/`verdict_*.json` pairs) and the classification script + its stdout.
