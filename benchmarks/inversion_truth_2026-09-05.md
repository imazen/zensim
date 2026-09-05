# Two-reference inversion truth — charging a backwards rung to the ENCODER, not the dial (2026-09-05)

**USER DIRECTIVE, verbatim:** *"for inversions, we should choose say ssim2 and butter and only
flag true inversions where they agree, and we can then file or update tracking issues on codecs
for when they are nonmonotonic."*

**Lane:** `claude-invtruth`, jj sibling workspace `~/work/zen/zensim--invtruth`.
**Owner of the rule:** `zensim_validate::dial_addressability::encoder_inversion` — ONE function,
called by both the G-ADDR contract's `mono` input and `bake_verdict`'s ladder-inversion census,
so the gate and the census cannot drift apart on what an encoder inversion is.
**Registry:** the `inversion_truth` section of
[`benchmarks/dial_addressability_floor_2026-09-04.json`](dial_addressability_floor_2026-09-04.json)
(append-only, the ruling quoted verbatim).

---

## 1. What changed, and why it is not a relaxation

A dial ladder walks a codec's own quality settings. When a scorer reads a higher setting as
WORSE, that has two possible causes and the instrument could not tell them apart:

* the dial mis-ranked two images, or
* **the encoder actually emitted a worse image at the higher setting.**

Charging the second to the dial makes every scorer look defective in proportion to how
non-monotone its codecs are — and it is the *codec* that needs the bug report. Under the ruling,
a material backwards rung is charged to the ENCODER, and leaves `mono` (hence C1) and the zone
census, **only** where both reference metrics independently call the higher setting worse.

It is not a relaxation, because the exemption is exactly its own size and nothing more:

* **Unknown is never an exemption.** A pair whose reference values are not both on disk is NOT
  MEASURABLE and stays charged to the DIAL. `bake_verdict` counts those
  (`dial.inversion_truth.n_attribution_unknown`) so a thin truth table cannot look like a clean
  dial, and a requested `agree` with no usable table degrades to `single` **loudly**.
* **The butteraugli margin is rounded UP**, which makes agreement *rarer* — see §3.
* **`single` stays reproducible and is byte-identical.** MEASURED: a `--inversion-truth single`
  run against a binary built from pristine `main@origin` gives **0 JSON differences** on shipped
  Profile D over the 9,593-row ladder instrument.

## 2. The rule

For one adjacent DISTINCT-setting pair, as quality rises:

```
encoder_inversion  ⟺  Δssim2 ≤ −0.5 points     AND     Δbutteraugli ≥ +margin(variant)
```

ssim2 is quality-oriented (worse = negative); butteraugli is a distance (worse = positive).
`bake_verdict --inversion-truth single|agree` selects the reading; **`agree` is the default**
since the ruling. `--reference-truth <tsv>[:variant]` supplies the per-cell table
(`image_id/codec/q/ssim2/butteraugli`, butteraugli in DISTANCE units), built by
`scripts/build_reference_truth.py` from an instrument that persisted both metrics.

### 2.1 Why BOTH, not EITHER — measured

Of the **105** pairs on the ladder instrument where ssim2 alone calls a material inversion,
butteraugli-pnorm3 moves the worse direction **at all** on only **47**. A single-reference rule
would excuse more than twice as many pairs as two agreeing references do, and half of what it
excused would be one metric's opinion. Requiring agreement is what makes an exemption evidence.

## 3. The margins, and how they were derived

**ssim2's is not re-derived.** It is `bake_verdict`'s own `MATERIAL_INV_PT` = **0.5 dial
points**, so the reference is held to exactly the bar the dial is held to.

**butteraugli's could not come from measurement noise, because there is none.** The instrument's
own §8.0 reproducibility gate checked only encoded bytes and ssim2; this lane extended it.
MEASURED — re-running the jpeg leg from scratch against the original:

| column | identical | max abs delta |
|---|--:|--:|
| `encoded_bytes` | 2,574 / 2,574 | 0 |
| `score_ssim2` | 2,574 / 2,574 | 0 |
| **`score_butteraugli_pnorm3`** | **2,574 / 2,574** | **0** |
| **`score_butteraugli_max`** | **2,574 / 2,574** | **0** |
| `score_dssim` | 2,574 / 2,574 | 0 |

A reproducibility-derived margin would therefore be **0.0**, which is not a materiality bar at
all. So the margin is derived by **equivalence to ssim2's own materiality**, on the population
where the two metrics demonstrably track each other:

> **margin(variant) = the 85th percentile of |Δ variant| over FORWARD adjacent pairs whose
> Δssim2 lies in [0.45, 0.55] — the moves ssim2 itself calls exactly material — rounded UP to
> the next 0.05.**

MEASURED (n = 410 forward pairs in band):

| variant | p25 | median | p75 | **p85** | → margin |
|---|--:|--:|--:|--:|--:|
| **pnorm3 (PRIMARY)** | 0.0238 | 0.0315 | 0.0403 | **0.0481** | **0.05** |
| max (reported beside) | 0.0037 | 0.0677 | 0.1361 | **0.2189** | **0.25** |

The estimate is stable to **±0.0007** across the wider bands [0.4, 0.6] and [0.25, 0.75], so it
is not an artifact of band choice.

**Rounding UP is the conservative direction and is deliberate.** A larger butteraugli margin
makes agreement rarer, so fewer inversions are excused and MORE stay charged to the dial. The
ruling cannot launder a dial defect by being generous with this number. Sensitivity, on the 105
ssim2-material inversions:

| pnorm3 margin | 0.00 | 0.01 | 0.02 | **0.032 (median)** | **0.05 (shipped)** | 0.10 | 0.25 |
|---|--:|--:|--:|--:|--:|--:|--:|
| encoder-confirmed | 47 | 40 | 37 | 31 | **26** | 16 | 6 |

### 3.1 Why pnorm3 is PRIMARY — measured, not asserted

Over all **9,411** adjacent distinct pairs, the share on which the variant moves the direction
ssim2 moves:

| variant | direction agreement with ssim2 |
|---|--:|
| **pnorm3** | **94.30 %** (8,875 / 9,411) |
| max | 75.27 % (7,084 / 9,411) |

`max` is a maximum over pixels, so one localised artifact swings it: in the population where
ssim2 barely moves (|Δ| < 0.05), `max`'s own p95 excursion is **1.27** against pnorm3's **0.14**.
`max` is reported beside pnorm3, never instead of it.

There is a second, sharper reading in the same data. Restricted to BACKWARD ssim2 moves of the
same 0.5-point magnitude, butteraugli's median move is **−0.008** — i.e. it mostly says the
higher setting is *better*, disagreeing with ssim2 — while on FORWARD moves of that magnitude it
agrees essentially always (median −0.032, tight IQR). **Small ssim2 inversions are largely not
corroborated**, which is exactly the phenomenon the ruling exists to stop mis-charging.

---

## 4. THE RESULT — the per-codec encoder-inversion table

Bake-independent: this is a property of the instrument's two reference metrics, not of any dial.
Emitted by `bake_verdict --encoder-inversion-census <out.tsv>` (same rule, same owner, only the
filter differs — the per-run panel table lists just the pairs a given scorer also inverted on).

| codec | adjacent distinct pairs | ssim2-alone material inversions | **encoder-confirmed (pnorm3, PRIMARY)** | rate | encoder-confirmed (max) | refs touched | of which bytes UP |
|---|--:|--:|--:|--:|--:|--:|--:|
| `avif-rav1e` | 2,457 | 46 | **20** | 0.81 % | 23 | 14 | 13 |
| `avif-svt` | 1,599 | 1 | **1** | 0.06 % | 1 | 1 | 0 |
| `jpeg` | 1,950 | 49 | **5** | 0.26 % | 10 | 3 | 5 |
| `jxl` | 1,144 | 1 | **0** | 0.00 % | 0 | 0 | 0 |
| `webp` | 2,261 | 8 | **0** | 0.00 % | 4 | 0 | 0 |
| **all** | **9,411** | **105** | **26** | **0.28 %** | **38** | — | **18** |

**jxl and webp are CLEAN under the primary rule** — zero encoder-confirmed inversions on 1,144
and 2,261 pairs. `avif-rav1e` is the outlier at 0.81 %, and **13 of its 20 also produce a LARGER
file**, so those are RD points strictly dominated by their own lower setting. All five jpeg
confirmations cost bytes.

Ties and saturation (byte-identical bitstreams at two settings) are a **separate observation**,
not inversions, and are excluded by construction — the census walks only DISTINCT settings.
Recorded here because it is consequential for `avif-svt`: **936 of its 2,574 cells (36.4 %)** are
duplicate settings, against `avif-rav1e`'s 78 (3.0 %).

### 4.1 The pairs — full evidence

`/mnt/v/output/zensim/invtruth-2026-09-05/encoder_inversions_ladder_{pnorm3,max}.tsv`
(header carries the grid, the reference table and both margins). Per-codec detail is in the
three tracking issues filed from it (§7).

---

## 5. Dial-attributed re-grade — D, previous D, A, B, and the peers

Ladder instrument, 9,593 rows, sha256 `4c3874a78c469e15…`; reference table
`reference_truth_ladder_pnorm3.tsv` (9,593 cells).

| arm | `mono` under `single` | `mono` under `agree` | Δ | rungs re-attributed |
|---|--:|--:|--:|--:|
| **Profile D — SHIPPED** | 0.99310 | **0.99470** | +0.00160 | 15 |
| Profile D — previous (08-31) | 0.99420 | 0.99540 | +0.00120 | 12 |
| Profile A (`v47_strict_qat_native`) | 0.98030 | 0.98120 | +0.00090 | 8 |
| Profile B (shipped SDR) | 0.97760 | 0.97870 | +0.00110 | 11 |
| **`peer_ssim2` — the mentor** | 0.98880 | **0.99160** | +0.00280 | 26 |
| `peer_butteraugli_pnorm3` | 0.99970 | 0.99980 | +0.00010 | 1 |
| `peer_butteraugli_max` | 0.92650 | 0.92860 | +0.00210 | 20 |

**Shipped D's dial-attributed inversion rate is 0.53 % against the mentor's 0.84 %** — D is the
more monotone of the two on this instrument, and the ruling widens that gap slightly (it was
0.69 % vs 1.12 % under `single`). D was already ahead here; §3 of
[`d_inversions_2026-09-05.md`](d_inversions_2026-09-05.md) found the same reversal against the
old board grid and localised it to the floor-dense sampling.

Two readings worth stating plainly:

* **`peer_ssim2` absorbs all 26** encoder-confirmed pairs — necessarily, since an ssim2 material
  inversion is half the AND. Its remaining 79 material inversions are ones butteraugli does not
  corroborate, and they stay charged to it. That is not circular; it is the rule applied
  symmetrically to the mentor.
* **`peer_butteraugli_max` fails C1's 0.93 bar under BOTH readings** (0.9265 / 0.9286). The
  noisiest variant is also the least monotone dial measured here, which is a second, independent
  argument for pnorm3 as primary. (Its scores are in butteraugli distance units, so its 0.5-point
  materiality is not commensurate with a dial's — read the direction, not the magnitude.)

---

## 6. The board — measured, and it provably cannot move

**Before = after: 47 NOT-SHIPPABLE badges, 0 cells flipped.** This is a proof, not a
non-attempt, and it has two independent halves.

**(a) The ruling can only raise `mono`, and `mono` gates exactly one row.**
`mono_agree = 1 − dial/pairs` and `mono_single = 1 − (dial + encoder)/pairs`, so
`mono_agree ≥ mono_single` for every non-negative encoder count. `mono` feeds exactly ONE gate
row — **C1** (`dial_addressability.rs:2199`), a `≥` bar — and **all 130 board fullevals carrying
a G-ADDR block already read C1 PASS**. No C1 can flip PASS→FAIL (the value only rises) and none
can flip FAIL→PASS (none is failing). C2–C6 are dead-zone / negative-tail / identity rows and do
not read `mono` at all. Gate:
`dial_addressability::tests::encoder_attribution_moves_c1_up_and_never_down`.

**(b) The two-reference reading is NOT MEASURABLE on the board's own grids, and that is a
measurement too.** The canonical 372 grid and the 944 POOLS grid carry per-cell ssim2 tables, but
the only butteraugli on disk for them is `dialcells_butteraugli_944grid.tsv`, whose variant the
artifact does not name. Identified empirically: **median relative error 0.0029 against
`score_butteraugli_max` over 4,105 independently-encoded cells, against 0.58 for pnorm3** — so it
is the `max` variant and the PRIMARY variant does not exist for those cells. Re-scoring pnorm3
for them is not honestly possible: the pixels that produced those cells are a different
decoder/encoder generation, and this repo has already measured decoder era at **73 % of extractor
era** on shipped B's dial. Those cells therefore keep the `single` reading, with the loud NOT
MEASURABLE note the tool now prints.

Scoped in the registry as `inversion-counts-single-reference-pre-2026-09-05`
(`benchmarks/eval_annotations.json`): every published `mono_pct`, per-codec inversion count and
ladder-zone census predating 2026-09-05 is the `single` reading and must not be compared with or
averaged into an `agree` one.

### 6.1 The inversions page, re-marked

`.../ladder-2026-09-05/inversions/index.html` now badges each of D's ten worst inversions
**ENCODER-CONFIRMED** or **DIAL-ONLY**, with both reference deltas on the badge. Builder committed
at `scripts/v_next/build_inversions_page.py` (it was living in scratch); it joins the
owner-produced census rather than re-implementing the rule.

**MEASURED, and it is the headline of this whole lane: ssim2 alone confirmed 9 of 10. Both
references confirm 5 of 10.**

| # | image | codec | step | Δssim2 | Δbutteraugli-pnorm3 | verdict |
|--:|---|---|---|--:|--:|---|
| 1 | `f65a24b7e176eb47_1022x818` | jpeg | q12→q13 | −12.132 | +0.1277 | **encoder-confirmed** |
| 2 | `d01e6b7798bbe066_513x769` | jpeg | q0→q11 | −7.487 | +0.1088 | **encoder-confirmed** |
| 3 | `b2e6e2b5969eaf25_1022x818` | avif-rav1e | q2→q3 | −4.513 | +0.3257 | **encoder-confirmed** |
| 4 | `76c1e30469720c75_769x513` | avif-rav1e | q5→q6 | −2.224 | +0.0321 | dial-only (sub-margin) |
| 5 | `68845bbc29306de5_769x513` | avif-rav1e | q3→q4 | +0.239 | +0.4995 | dial-only (ssim2 flat) |
| 6 | `5a9b3b963f852e20_512sq` | jpeg | q16→q17 | −9.725 | +0.0269 | dial-only (sub-margin) |
| 7 | `a9143f4b78fe5a13_513x769` | jpeg | q16→q17 | −8.844 | **−0.0497** | dial-only (butteraugli says BETTER) |
| 8 | `20f63bf11ab2c911_512sq` | jpeg | q0→q11 | −7.254 | **−0.0169** | dial-only (butteraugli says BETTER) |
| 9 | `d01e6b7798bbe066_513x769` | avif-rav1e | q0→q2 | −6.204 | +0.2681 | **encoder-confirmed** |
| 10 | `b2e6e2b5969eaf25_1022x818` | jpeg | q12→q13 | −7.734 | +0.0819 | **encoder-confirmed** |

Rows 7 and 8 are the ones that matter for judgement: ssim2 reads a **7–10 point** quality loss
while butteraugli says the higher setting is *better*. A single-reference reading would have
credited the encoder with those and excused the dial. Row 4 is honestly borderline — +0.0321 sits
above the un-rounded 0.0315 median and below the shipped 0.05, so the conservative rounding is
what keeps it on the dial. **§5 of `d_inversions_2026-09-05.md` ("9 of 10 are confirmed by
ssim2") is correct for its single-reference reading and is superseded as an attribution claim.**

---

## 7. Codec tracking issues filed

Searched each repo's open AND closed issues for an existing monotonicity / RD-ordering item
first; none existed, so all three are new. Assigned to `lilith`. Each states the rule, the
margins, the exact steps with both metrics' deltas and byte deltas, the reproducibility gate,
and the instrument sha256 — and each says up front to re-run against current `main` before
investing, because the AVIF encode path in particular is under active change.

| repo | issue | codec | pairs | headline |
|---|---|---|--:|---|
| `imazen/zenjpeg` | **#201** | jpeg | 5 | 5 steps where quality+1 costs more bytes AND scores worse on both metrics |
| `imazen/zenrav1e` | **#42** | avif-rav1e | 20 | 20 steps across 14 images; 13 also larger |
| `imazen/zenav1-svt` | **#19** | avif-svt | 1 | one confirmed step (1/1,599), plus the 36 % setting-saturation observation |

No issue for `jxl-encoder` or `zenwebp` — **zero** encoder-confirmed inversions on either.

---

## 8. What shipped

* `zensim-validate/src/dial_addressability.rs` — `encoder_inversion`, `ButteraugliVariant`,
  `InversionTruth`, `ReferenceTruth`; margins as pinned constants; the derivation in the module
  note. **7 new tests** (three-outcome attribution incl. both single-reference directions and a
  sub-margin case, unknown-is-not-an-exemption, pinned margins with boundary cases, tag
  round-trip, and the C1 direction property through the real gate). 53 pass.
* `zensim-validate/src/bin/bake_verdict.rs` — `--inversion-truth single|agree` (default
  `agree`), `--reference-truth <tsv>[:variant]`, `--encoder-inversion-census <tsv>`; the DIAL
  panel gains the encoder row, the `single` comparison row, the unknown row, the loud
  NOT-MEASURABLE note and a named evidence table; `--full-json` gains
  `dial.inversion_truth` and per-zone-cell `inv_encoder` / `inv_unknown`.
* `scripts/build_reference_truth.py` — the sidecar emitter (computes nothing; reshapes only).
* `scripts/v_next/build_inversions_page.py` — the page builder, moved out of scratch and joined
  to the census.
* Registry `inversion_truth` section + annotation
  `inversion-counts-single-reference-pre-2026-09-05`.

## 9. Reproducing

```sh
cargo build --release -p zensim-validate --bin bake_verdict
L=/mnt/v/output/zensim/ladder-2026-09-05/instruments

python3 scripts/build_reference_truth.py --full $L/ladder_grid_ladder_full.parquet \
    --variant pnorm3 --out $L/reference_truth_ladder_pnorm3.tsv

# the bake-independent per-codec census (the codec-issue evidence)
target/release/bake_verdict --bake zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin \
    --dial-grid $L/dial_grid_372col_ladder.parquet --corpora cid22 \
    --reference-truth $L/reference_truth_ladder_pnorm3.tsv:pnorm3 \
    --encoder-inversion-census encoder_inversions.tsv --output /dev/null

# the pre-ruling reading, byte-identical to main@origin's binary
target/release/bake_verdict ... --inversion-truth single
```
