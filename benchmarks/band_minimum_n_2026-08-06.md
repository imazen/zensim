# How wide must a band be? — the minimum-usable-band rule, derived

**Campaign appendix V** (`benchmarks/sota944_campaign_2026-08-03.md`), 2026-08-06.
User directive: *"widen band 9, we can figure out what is statistically usable
minimum band sizes"*.

Every statistic here comes from `zenstats` through the canonical `panel --batch`
owner (via `scripts/lib/zen_stats.py`). The instrument is
`scripts/band_reliability.py`; it contains no Spearman. Nothing was re-scored:
the whole study runs on the board's **stored per-pair predictions**, so a
"recut" and a re-verdict are the same thing (gated below, bit-identically).

---

## 0. The short answer

A band is **USABLE** iff all three hold:

| # | condition | bar | why |
|---|---|---|---|
| 1 | **ESTIMABLE** — marginal 95 % CI half-width | ≤ 0.20 | the value can be reported at all |
| 2 | **DISCRIMINATING** — split-half model-ranking `r_SB` | ≥ 0.90 | it ranks models consistently (appendix O's registered threshold) |
| 3 | **RESOLVING** — model spread p10–p90 ÷ band LSD | ≥ 1 | the population differs by more than the noise |

Which, on the corpora this project bands, comes out as:

> **n ≥ 1000 pairs AND target span ≥ 0.08.**

**Both floors are load-bearing, and they bind through different mechanisms.**
That is the finding that matters, and it was not what the registration predicted:

* **n binds through the noise.** The CI half-width is almost exactly
  `1.06/√(n−3)` and cares about nothing else.
* **span binds through the signal.** At a fixed n of 200, the CI half-width
  moves only 0.140 → 0.086 across spans 0.02 → 0.20, while the correlation
  itself moves **0.056 → 0.632**. Range restriction attenuates what you are
  trying to measure and leaves the noise alone.

So an n-only rule admits range-restricted bands (CID22's equal-population
deciles: n = 429 each, spans 0.024–0.066, not one of them discriminating), and a
span-only rule admits tiny ones (CID22's `B3`: span 0.096, n = 57, `r_SB` 0.26).
**Neither substitutes for the other, and no amount of n rescues a narrow band.**

---

## 1. Instrument A — the pure-n curve (span held fixed)

Subsampling *within* a wide band leaves its span alone, so this isolates n.
Donors: CID22 `B7` + `B8` (spans ≈ 0.10); 5 models spanning the board's
aggregate range × 8 independent subsample draws per point; B = 2,000.

| n | measured 95 % CI half-width | Bonett–Wright prediction | ratio |
|---|--:|--:|--:|
| 8 | 0.7571 | 0.6826 | 1.109 |
| 12 | 0.5814 | 0.5451 | 1.067 |
| 16 | 0.4595 | 0.4524 | 1.016 |
| 24 | 0.3761 | 0.3678 | 1.023 |
| 32 | 0.3103 | 0.3145 | 0.987 |
| **43** *(= CID22 B9's n)* | **0.2666** | 0.2619 | 1.018 |
| 64 | 0.2090 | 0.2133 | 0.980 |
| 96 | 0.1684 | 0.1716 | 0.981 |
| 128 | 0.1477 | 0.1510 | 0.978 |
| 192 | 0.1195 | 0.1227 | 0.974 |
| 256 | 0.1048 | 0.1075 | 0.976 |
| 384 | 0.0849 | 0.0875 | 0.970 |
| 512 | 0.0725 | 0.0748 | 0.969 |
| 768 | 0.0592 | 0.0614 | 0.965 |
| 1024 | 0.0515 | 0.0532 | 0.969 |

The empirical curve tracks the closed form to within 3 % everywhere above
n = 16 — so the theory is a usable predictor here, and the two agree that
**estimability is cheap**: the 0.20 bar is crossed between n = 64 and n = 96.

**Estimability is therefore NOT the binding condition**, which is the single
most useful thing this table says. Three of CID22's old bands prove it directly:
`B4` (n=266), `B5` (615) and `B6` (836) all sit comfortably inside the 0.20 bar
(half-widths ≈0.10 / 0.07 / 0.06) and rank models at `r_SB` 0.441 / 0.650 /
0.778 — estimable, and unable to gate anything.

> **Correction, filed against an earlier revision of this document.** `B9` was
> first used as that example, on the strength of a "marginal bootstrap sd 0.178"
> quoted from appendix U. That is a standard DEVIATION, not a CI half-width, and
> repeating it as one was my error. Measured here directly at B=10,000 over 15
> models, **`B9`'s marginal 95 % CI half-width is 0.334** (reference-clustered
> 0.310) — ×1.96 of U's sd, as it should be. So `B9` does not "pass estimability
> and fail the rest": **it fails all three conditions.** The argument that
> estimability is not the binding constraint is unaffected — it rests on the
> curves above and the discrimination surface in §4, where the bars are crossed
> ~10× apart in n — but the illustration was wrong and is replaced.

## 2. Instrument B — the pure-span curve (n held fixed at 200)

Centred sub-slices of the target at decreasing width, each subsampled to a
common n, so this isolates span.

| span | sd ratio | \|SROCC\| | CI half-width | Thorndike prediction | **SNR** |
|---|--:|--:|--:|--:|--:|
| 0.020 | 0.044 | 0.056 | 0.1397 | 0.081 | **0.40** |
| 0.030 | 0.064 | 0.087 | 0.1366 | 0.117 | **0.64** |
| 0.040 | 0.089 | 0.182 | 0.1344 | 0.164 | 1.35 |
| 0.060 | 0.129 | 0.247 | 0.1304 | 0.233 | 1.89 |
| 0.079 | 0.177 | 0.320 | 0.1243 | 0.315 | 2.58 |
| 0.099 | 0.224 | 0.370 | 0.1223 | 0.346 | 3.02 |
| 0.147 | 0.323 | 0.513 | 0.1103 | 0.510 | 4.65 |
| 0.198 | 0.421 | 0.632 | 0.0887 | 0.612 | 7.13 |

`SNR = |SROCC| / CI half-width`. Two readings:

1. **The noise column is nearly flat** (0.140 → 0.089 over a 10× span change)
   while the signal column moves 11×. Span is a signal problem.
2. **Thorndike case-II range restriction predicts the attenuation closely** —
   `r_band ≈ r_full·u / √(1 + r_full²(u²−1))` with `u = sd_band/sd_full` —
   which is why this is a structural property of narrow slices and not something
   a better model or a bigger corpus fixes.

At span ≤ 0.03 the SNR is **below 1**: the band cannot establish that its
correlation is nonzero, even though its CI half-width "passes" the estimability
bar. That is the concrete proof that condition 1 alone is insufficient.

## 3. Instrument C — the joint (n, span) surface

95 % CI half-width (median over models × draws):

|  n | 0.02 | 0.03 | 0.04 | 0.06 | 0.08 | 0.10 | 0.15 | 0.20 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| 43 | 0.301 | 0.302 | 0.301 | 0.285 | 0.283 | 0.260 | 0.234 | 0.212 |
| 96 | 0.201 | 0.197 | 0.200 | 0.191 | 0.186 | 0.168 | 0.146 | 0.134 |
| 192 | 0.141 | 0.140 | 0.139 | 0.133 | 0.130 | 0.121 | 0.106 | 0.089 |
| 512 | · | · | · | 0.081 | 0.077 | 0.074 | 0.064 | 0.054 |
| 1024 | · | · | · | · | · | 0.053 | 0.046 | 0.039 |

Read across any row: **the half-width barely depends on span.** Read down any
column: it is essentially `∝ 1/√n`. The noise surface is a function of n alone.

## 4. Instrument E — discrimination, which is what actually binds

Split-half model-ranking SROCC with Spearman–Brown correction over the board's
120 per-pair-carrying cells (20 shuffles, seed 4242 — appendix O's constants),
on centred (n, span) slices:

|  n | 0.02 | 0.04 | 0.06 | 0.08 | 0.10 | 0.15 | 0.20 |
|---|--:|--:|--:|--:|--:|--:|--:|
| 43 | −0.075 | −0.053 | −0.176 | 0.088 | 0.158 | 0.352 | 0.273 |
| 128 | 0.168 | 0.381 | 0.432 | 0.471 | 0.675 | 0.665 | 0.740 |
| 256 | · | 0.371 | 0.431 | 0.746 | 0.683 | 0.773 | 0.791 |
| 512 | · | · | 0.659 | 0.812 | 0.847 | **0.890** | **0.894** |
| 768 | · | · | · | 0.854 | 0.877 | 0.898 | **0.929** |
| 1024 | · | · | · | · | **0.918** | **0.930** | **0.955** |
| 1400 | · | · | · | · | · | **0.946** | **0.965** |

**This is where both constants come from.**

* **`N_MIN = 1000`.** At span 0.10 the 0.90 bar is bracketed by n = 768 (0.877)
  and n = 1024 (0.918). CID22's REAL bands at the same span agree closely and
  independently: `B7` (n = 1092) measures **0.900**, `B8` (n = 1382) **0.949**,
  and everything smaller falls away fast — 836 → 0.778, 615 → 0.650, 266 →
  0.441, 57 → 0.260. 1000 sits at the top of the measured bracket; the smallest
  real band it admits (1092) measures exactly at the bar.
* **`SPAN_MIN = 0.08`.** Below 0.08 the bar is unreachable at **any** n CID22
  can supply: the best observed is 0.659 at span 0.06 (n = 512, the largest that
  fits), 0.407 at 0.04, 0.298 at 0.02. It IS reached at 0.10 / 0.15 / 0.20.
  **Stated plainly: 0.08 was not itself observed clearing the bar** — its trend
  (0.762 @384, 0.812 @512, 0.854 @768) heads for it near n ≈ 1200 — so the
  constant sits one grid step below the lowest span where the bar was actually
  seen. It also has a hard structural ceiling: a fixed decile's realised span is
  always just under 0.10, so any floor at 0.10+ would merge away every decile.

### The full per-band panel on CID22's OLD (fixed-decile) bands

n = 120 models, 20 shuffles, 30 model pairs × B = 2,000 for the LSD:

| band | n | span | `r_SB` | LSD | model p10 | p50 | p90 | DR |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| B0 / B1 / B2 | 0 / 0 / 1 | — | NOT-MEASURED | | | | | |
| B3 | 57 | 0.096 | 0.260 | 0.197 | −0.017 | 0.077 | 0.197 | 1.09 |
| B4 | 266 | 0.099 | 0.441 | 0.062 | 0.205 | 0.238 | 0.280 | 1.21 |
| B5 | 615 | 0.100 | 0.650 | 0.032 | 0.299 | 0.331 | 0.354 | 1.74 |
| B6 | 836 | 0.100 | 0.778 | 0.023 | 0.334 | 0.388 | 0.406 | 3.10 |
| B7 | 1092 | 0.100 | **0.900** | 0.024 | 0.290 | 0.379 | 0.398 | 4.60 |
| B8 | 1382 | 0.100 | **0.949** | 0.016 | 0.393 | 0.461 | 0.488 | 6.12 |
| **B9** | **43** | **0.019** | **0.711** | **0.132** | **−0.263** | **−0.187** | **−0.015** | 1.88 |

`B9`'s marginal 95 % CI half-width is **0.334** (reference-clustered 0.310) —
it fails the estimability bar too, so it fails all three conditions.

Only **two of ten** bands discriminate. And note `B3`: DR = 1.09 clears
condition 3 while `r_SB` = 0.26 fails condition 2 — so condition 3 is not
redundant either. Every condition earns its place in the rule.

`B9`'s whole model population is **negative** — the band is ordered backwards
for essentially every model on the board — which is what appendix U measured
and this reproduces from a different population.

---

## 5. The scheme

Registered candidates were fixed deciles (status quo), equal-population
quantiles, adaptive merging, and fixed quintiles. **Merging wins; quantile
banding is actively wrong.**

**Quantile banding is rejected on evidence.** It guarantees n by construction —
429 pairs per band on CID22 — and destroys span exactly where the target
distribution is dense:

| CID22 quantile band | Q0 | Q1 | Q2 | Q3 | Q4 | Q5 | Q6 | Q7 | Q8 | Q9 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| n | 430 | 429 | 429 | 429 | 429 | 429 | 429 | 429 | 429 | 430 |
| span | 0.245 | 0.066 | 0.057 | 0.049 | 0.042 | 0.040 | 0.037 | 0.034 | 0.024 | 0.050 |
| **`r_SB`** | 0.859 | **0.400** | **0.474** | **0.578** | **0.592** | **0.422** | **0.349** | **0.356** | **0.443** | **0.632** |

Nine of ten fail `SPAN_MIN`, and — measured, not inferred — **not one of the ten
reaches the 0.90 discrimination bar**, the widest getting 0.859. Equalising
counts is the intuitive fix and it makes the range-restriction problem *worse*:
it takes the corpus's densest regions and slices them thinnest.

**Fixed quintiles are not enough either** — the registered "maybe ten is simply
too many" null. Coarsening the same fixed grid to five bands on CID22:

| band | B0 | B1 | B2 | B3 | B4 |
|---|--:|--:|--:|--:|--:|
| n | **0** | 58 | 881 | 1928 | 1425 |
| `r_SB` | — | **0.361** | **0.864** | 0.969 | 0.949 |

Two of five pass, one is still **empty**, and one still rests on 58 pairs. Band
*count* is not the variable: the problem is that a FIXED grid cannot know where
a corpus's mass actually is. Only a scheme that reads the target distribution
adapts — which is the argument for merging over any fixed grid, coarse or fine.

**The shipped scheme** (`zensim_validate::bands`, `merged-decile-2026-08-06`):
sweep the fixed deciles low→high, close a band the moment it clears both floors,
fold a deficient remainder into the band before it. Deterministic in the target
column alone — it takes no predictions, so identical bands for every model is
structural, not a convention.

| corpus | pairs | bands |
|---|--:|---|
| **cid22** | 4292 | `B0-B6` (1775) · `B7` (1092) · **`B8-B9` (1425, span 0.119, all 49 refs)** |
| **tid** | 3000 | `B0-B4` (1418) · `B5-B9` (1582) |
| **kadid** | 10031 | `B0-B1` · `B2-B3` · `B4-B5` · `B6` · `B7` · `B8-B9` |
| **csiq** | 866 | `B0-B9` — **NOT-MEASURED**, too small to band |
| **live** | 779 | `B0-B9` — **NOT-MEASURED**, too small to band |

### Does the shipped scheme pass its own rule? Yes, on every band

Same instrument, same 120 models, same constants — CID22 under
`merged-decile-2026-08-06`:

| band | n | span | `r_SB` (≥0.90) | LSD | model p10 | p50 | p90 | DR (≥1) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| `B0-B6` | 1775 | 0.423 | **0.954** | 0.0119 | 0.620 | 0.682 | 0.706 | **7.15** |
| `B7` | 1092 | 0.100 | **0.900** | 0.0232 | 0.290 | 0.379 | 0.398 | **4.66** |
| `B8-B9` | 1425 | 0.119 | **0.949** | 0.0167 | 0.391 | 0.458 | 0.484 | **5.60** |

Three bands, three passes, and every model's value is **positive in every band**
— against the old grid's ten bands of which two discriminated and one ran
backwards for the entire population. Registered outcome **(A)** fires.

Two notes worth keeping:

* The mechanical rule lands on **`≥ 0.80`** for CID22's top band — the exact
  slice appendix U had already identified by hand as the honest high-fidelity
  read (n = 1425, span 0.119). Two different methods, same edge.
* **CSIQ and LIVE cannot be banded at all** at this bar, and now say so instead
  of publishing ten bands of 19–213 pairs. That is a result, not a failure.

A pairwise "merge the worst band into its smaller neighbour" greedy was tried
first and **rejected**: it is myopic. On TID it spent `B4` (677) on the
already-satisfied `B5` (705), stranded `B6-B9` (877) with no deficient
neighbour, and collapsed the corpus to a single band — where the sweep finds two
clean ones. Sweeping high→low gives identical bands on all five corpora, so the
direction is not a free parameter (gated by a test).

### Two defects the recut exposed

1. **The bottom band dropped rows.** The fixed grid closed at `0.0`, so LIVE's
   21 sub-zero DMOS pairs fell out of every band: its published band rows summed
   to **758 of 779**. The scheme's bottom band is now open below, symmetric with
   the top being open above, and a partition test asserts every row lands in
   exactly one band.
2. **KADID's stored per-pair is a 5,000-row subsample** of its 10,125, and the
   subsample differs per cell. It cannot be recut from the board and is left
   legacy + annotated, rather than publishing bands over half a corpus under a
   header claiming all of it.

---

## 6. F8's re-point, and its floor

F8 read `|B9| ≥ 0.15 ∧ |B3| ≥ 0.0` on the **absolute** value. It now reads the
**signed** top and bottom usable bands, in each corpus's declared orientation.

**The floor is derived.** F8's job is *non-collapse*, so the bar is the smallest
value at which a band ordering is significantly positive — the band's own
marginal 95 % CI half-width, measured on `B8-B9` at B = 10,000 over a stratified
25-model probe:

| bootstrap | median half-width |
|---|--:|
| pair-level | 0.0407 |
| **reference-clustered (49 refs)** | **0.0866** ← governs |

The reference-clustered figure governs because CID22's pairs cluster by
reference (up to 61 in this band), so a pair-level resample understates the
uncertainty — the confound was registered before the number was known.
`ceil` to 2 dp gives **`BAND_HIGH = 0.09`**. `BAND_LOW` stays `0.0`, which
against a *signed* value is a real bar for the first time (`|x| ≥ 0.0` was
unfalsifiable).

**Honest limitation: the new F8 has no discriminating power on today's board.**
All 120 recut cells pass — the top band's population runs **+0.262 … +0.514,
with zero negatives**. Nothing on the board collapses in the high-fidelity
region. That is the correct behaviour for a non-collapse floor and it is worth
stating plainly: F8 is now a **regression guard**, not a selector. Its
predecessor appeared to discriminate (167 of 280 passed) only because it was
ranking models by the depth of an inversion.

The band term in `balanced_composite` — `(B3 + B9)/2` at weight 0.15, on the
same absolute values — is fixed the same way, and that one *was* feeding
selection, not just a PASS/FAIL cell.

---

## 7. Reproduction

```sh
cargo build --release -p zensim-validate --bin panel --bin bake_verdict --bin freeze_check
export ZEN_PANEL_BIN=target/release/panel

python3 scripts/band_reliability.py --out benchmarks/appendixV/gv1_b9_sign_2026-08-06.tsv gv1
python3 scripts/band_reliability.py --out benchmarks/appendixV/curves_2026-08-06.tsv \
        curves --boot 2000 --reps 8 --model-quantiles 0.1,0.35,0.5,0.65,0.9
python3 scripts/band_reliability.py --out benchmarks/appendixV/discrim2d_2026-08-06.tsv discrim2d
python3 scripts/band_reliability.py --n-min 1000 --span-min 0.08 \
        --out benchmarks/appendixV/discrim_cid22_fixed10.tsv \
        discrim --corpus cid22 --scheme fixed10 --shuffles 20 --lsd-pairs 30 --lsd-boot 2000
python3 scripts/band_reliability.py --n-min 1000 --span-min 0.08 \
        --out benchmarks/appendixV/f8_floor_2026-08-06.tsv floor
python3 scripts/band_reliability.py selfcheck        # mirror-vs-owner parity
```

Board recut (no rescore): `promote_fulleval.py --rebuild-bands <cell.fulleval.json>`.

**Gate:** recutting a cell's bands from its stored per-pair reproduces what
`bake_verdict` emits for the same bake **bit-identically** (max abs difference
`0.0` across every band and every field on `E1_baseline_s42`). That is what
makes the 120-cell recut equivalent to 120 re-verdicts for the band block.

**Methodological caveat worth stating**: the (n, span) surfaces were measured on
CENTRED slices of CID22's dense middle (MOS median ≈ 0.72), while the bands that
matter sit at the ends. Both constants therefore rest on a mechanism argument
PLUS a corroboration, and the corroboration is what makes them credible —
CID22's REAL bands at span ≈ 0.10 land on the same `r_SB` curve as the centred
slices (1092 → 0.900, 1382 → 0.949, against the slices' 0.918 @1024), and the
resulting merged bands then pass the rule they were derived from. A tail slice
with a different reference mix could still behave differently; that is not
measured here.

Artifacts: `benchmarks/appendixV/`. Board backup before the recut:
`/mnt/v/output/zensim/reports/fulleval.pre-appendixV-2026-08-06.bak`.
