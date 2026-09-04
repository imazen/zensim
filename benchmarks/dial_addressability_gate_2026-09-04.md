# G-ADDR — the dial ADDRESSABILITY gate (PRE-REGISTERED 2026-09-04)

**Lane:** `claude-dialgate`.
**User rule (2026-09-04), verbatim spirit:** *"floor and ceiling dial addressability is
crucial … any model that limits dial range cannot ship"*, and *"I want to be able to reuse
ssim2 scores if possible."*
**Owner:** `zensim-validate/src/dial_addressability.rs` (registry
`benchmarks/dial_addressability_floor_2026-09-04.json`, embedded with `include_str!`),
wired into `bake_verdict`'s DIAL panel and emitted at `dial.addressability` in
`--full-json`.

> **This section (§1–§5) is PRE-REGISTRATION.** It was written and committed BEFORE any
> era-corrected anchor was fitted. The bars are the SHIPPED dial's own measured values plus
> the already-registered product conventions; nothing here was chosen after seeing a
> candidate's numbers, and nothing here may be lowered to let a candidate through. Results
> land in §6+ and in the companion sections of this file.

---

## 1. Why a new gate

`benchmarks/imazen26_anchor_2026-09-04.md` established that shipped **B**'s −5..−6 point
dial skew is an **era** term: re-anchoring the SAME 2,000 anchor rows read today recovers
+3.892 (CID22) / +4.798 (KonJND) / +3.864 (AIC-3), 78–82 % of the defect, with SROCC
identical to 5 dp. But every candidate that recovers it **loses reach**:

| | shipped | `B_safesyn_curera` | `B_im26anchor` | `B_im26topdense` |
|---|---:|---:|---:|---:|
| reach | 96.85 | 94.23 | 85.74 | 88.96 |
| p5 | 13.73 | 18.23 | 22.91 | 22.99 |
| dynamic range | 85.99 | 81.33 | 75.53 | 76.13 |

Under the user's rule none of those can ship. **Nothing in the existing eval sees this.**
SROCC is rank-invariant and therefore *structurally* blind to it (all five arms read
0.88212 on CID22). G3 is about ordering. G1 asks only `p5 ≤ 25 ∧ p95 ≥ 85` — a bar every
one of the compressed arms clears comfortably. So a dial can lose 11 points of reach and
every panel in the mandatory two-panel eval will call it fine.

G-ADDR is the missing measurement.

## 2. The two tiers — and why they must not be merged

| tier | bar | question |
|---|---|---|
| **REGRESSION** (`A1`–`A9`) | the SHIPPED dial's own value on the SAME instrument | *is this candidate worse at the ends than the dial users have today?* |
| **CONTRACT** (`C1`–`C6`) | absolute product requirements | *does this dial meet the product contract at all?* |

They are separate because **the shipped dial fails four contract rows today** (§4). Merging
them would make every candidate's report say "FAIL" for reasons it did not cause. Every
report therefore prints an `incumbent` column: what the shipped dial reads on that same
axis. A ship needs **both** tiers PASS.

`—` is **NOT MEASURED**, never a pass. An unregistered dial grid makes the whole regression
tier `NOT MEASURABLE` — a bar you can dodge by choosing a friendlier instrument is not a
bar.

## 3. The bars

Instrument: the **canonical** dial grid
`dial_grid_372col_2026-05-29_quarantined_v2.parquet`, sha `6546c43e6d9572dc…`, 4,424 rows.
(The non-canonical `_quarantined` grid that `imazen26_anchor_2026-09-04.md` measured on is
*also* registered, with its own row, so those published numbers stay checkable against
their own instrument. New work uses the canonical one.)

### REGRESSION tier — bar = shipped B on the same instrument

| id | axis | direction | bar (canonical grid) | bar (doc grid `b5d27f21…`) |
|---|---|---|---:|---:|
| A1 | ceiling — pooled dial `max` | ≥ | **99.98330778475787** | 99.98376794026095 |
| A2 | floor — pooled dial `min` | ≤ | **3.12950123756248** | 3.12950123756248 |
| A3 | robust ceiling — `p95` | ≥ | **99.72170874183841** | 99.71524151863066 |
| A4 | robust floor — `p5` | ≤ | **13.645032446453126** | 13.726111774203428 |
| A5 | `reach` = max − min | ≥ | **96.85380654719539** | 96.85426670269847 |
| A6 | `dynamic_range` = p95 − p5 | ≥ | **86.07667629538528** | 85.98912974442723 |
| A7 | negative-tail probe dial `min` | ≤ | **2.516685884084839** | (probe is grid-independent) |
| A8 | negative-tail probe dial `p1` | ≤ | **3.981383254902343** | — |
| A9 | negative-tail `frac_below_zero` | ≥ | **0.0** | — |

A4's bar on the doc grid (13.7261) and A5/A6 reproduce the imazen-26 record's `p5 13.73 /
reach 96.85 / DR 85.99` exactly, which is the cross-check that this gate measures the same
quantity that record does.

### CONTRACT tier — absolute

| id | axis | bar | provenance |
|---|---|---|---|
| C1 | monotonicity | ≥ 0.93 | registered G3, `docs/EVAL_PANEL_REQUIREMENT.md` |
| C2 | flat/clamp dead-zone | ≤ 0.05 | registered G3 |
| C3 | negative values WORK — `frac_below_zero` on an all-negative-truth probe | > 0 | CLAUDE.md *"NEGATIVE zensim values MUST work … do NOT clamp at 0"* |
| C4 | deepest probe dial | < 0 | same |
| C5 | `dial(ref==dist)` inside `[97.5, 100]` | 0 rows outside | `benchmarks/standard_bake_packing_2026-05-27.md` + the QAT record |
| C6 | dial-grid cells out-scoring a perfect copy | 0 | *"0 above-identity"*, same record |

## 4. The instruments (pinned, content-addressed)

**`negtail_probe_372_2026-09-04.parquet`** — sha `5609d19fa10aef81…`, 2,000 rows,
`/mnt/v/output/zensim/dialgate-2026-09-04/`. Cut from
`canonical-2026-07-15/train/kadis_negrich.parquet` (266,111 rows, 237,675 with
`ssim2_gpu < 0`) by **20 equal-count quantile bins over the negative population, lowest 100
row indices per bin** — deterministic, no RNG. Truth span `ssim2_gpu` −770.62 … −0.33;
**every row's reference metric is negative**, so a correct dial must go below zero somewhere
on it. Current era (extracted 2026-07-15, after both extractor fixes).

**`identity_probe_372_2026-09-04.parquet`** — sha `e6f9096b8e0ebd97…`, 38 rows. All 38
distinct reference images of the canonical dial grid, each paired with **itself**, extracted
by `extract_features_372col` through imazen decoders at this lane's HEAD.

> **MEASURED, and it changed the design: the identity feature vector is the ZERO vector, for
> every image.** All 38 rows come back byte-identical and all-zero — which is what a
> *difference* metric must do, and it makes the identity dial a **scalar property of the
> bake**, `dial(0⃗)`, not a per-image measurement. The probe is kept at 38 rows rather than
> 1 so that property stays gated instead of assumed.

Measured identity dial across the shipped bakes:

| bake | `dial(0⃗)` | inside [97.5, 100]? |
|---|---:|---|
| `v47_strict_qat_native_2026-05-27` (Profile A) | **97.6893** | ✓ |
| `b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07` (Profile B, SHIPPED) | **96.2412** | ✗ |
| `d_sdr_add156_dense_dial_2026-08-31` (Profile D) | **96.1157** | ✗ |

So `[97.5, 100]` is a **v47-era** property; both shipped LINEAR dials sit ~1.3 points below
it. Registered as specified and **not relaxed** — the incumbent's failure is a finding.

## 5. Where the incumbent stands — the gate's first reading

`bake_verdict --bake <shipped B> --dial-grid <canonical> --negtail-probe … --identity-probe …`

**`NOT SHIPPABLE — regression PASS / contract FAIL`**, 11 pass / 4 fail / 0 not measured.
Regression is a tie by construction (the bars are its own values, and the comparators are
inclusive — a dedicated test pins that). The four contract failures are **standing defects
in the shipped product dial**:

| id | shipped B reads | bar |
|---|---:|---|
| C3 negative values work | `frac_below_zero` **0.0000** on a probe whose every row's truth is negative | > 0 |
| C4 deepest tail dial | **+2.5167** | < 0 |
| C5 identity band | **96.2412** (38/38 rows outside) | [97.5, 100] |
| C6 nothing above identity | **266 of 4,424 cells (6.01 %)** out-score a perfect copy; worst `1a20ecb0c1b92466_1022x818` jxl d=0.05 at **99.9833** vs identity **96.2412** | 0 |

**Read C3/C4 together with the floor mechanism.** Shipped B's dial anchor
(`multiband_anchor_dial100.parquet`) stores `target_score = max(ssim2, 0)`. 147 of its 2,000
rows have a genuinely negative ssim2 — down to **−64.16** — and **every one is stored as
0**. `dial_spline::fit_spline_knots` then collapses that run of `y == 0` bins to a single
bottom knot (the `neg_tail` dedup), so the spline carries **no in-distribution evidence at
all** about how far below zero the dial should go; the whole negative tail is a linear
extrapolation off one knot, and it extrapolates to `+2.52` on inputs whose truth is −770.

That is also why a *pure* re-anchor fails A4 by construction: the era correction is a
near-uniform **+3.9…+4.8** lift, so the floor rises with everything else and the clamp is
the reason nothing pushes back. **The lever is the clamp, not the anchor.**

## 6. Gate implementation — what landed

- `zensim-validate/src/dial_addressability.rs` — the owner: registry reader, `GridMeasure` /
  `NegTailMeasure` / `IdentityMeasure`, `evaluate`, markdown + JSON renderers. 12 tests.
- `benchmarks/dial_addressability_floor_2026-09-04.json` — the append-only registry.
- `bake_verdict` — new `--negtail-probe` / `--identity-probe`; the DIAL panel now emits the
  G-ADDR section on **every** run; `--full-json` carries `dial.addressability` (verdict +
  every raw measurement) and `dial.min` / `dial.max`.
- Test `canonical_dial_grid_has_a_g_addr_floor_row` pins the registry to `bake_verdict`'s
  own `CANONICAL_DIAL_GRID_SHA256`, so rotating the canonical grid without measuring the
  reference on it fails the build instead of silently disarming the gate.

**A one-ULP defect found and fixed on the way.** `serde_json`'s default float parser is not
correctly rounded: the bar written as the shipped dial's own `99.98330778475787` parsed back
as `…788`, and the reference bake **failed its own bar by one ULP** (A1 and A5 read ✗ in the
first run). Fixed by enabling serde_json's `float_roundtrip` feature for `zensim-validate`
(parsing only — serialization already goes through ryu), with
`registry_floats_round_trip_bit_exactly` as the guard. Worth knowing beyond this gate:
**every float bar `freeze_check` reads out of a fulleval JSON had the same hazard.**

---

# RESULTS

_Sections 7–8 are filled in by the measurement runs; until each one carries its numbers it
reads REGISTERED, NOT RUN. Nothing is written here before it is measured._

## 7. ssim2 reuse across decoder eras — REGISTERED, NOT RUN

**Question (user):** *"I want to be able to reuse ssim2 scores if possible."* The anchors'
`target_score` is a **stored** ssim2 computed at the corpus's own decode era, while the
features beside it are decoded today — the "mixed-era caveat" §2c of the imazen-26 record
refused to paper over. Measuring it settles whether an anchor can keep its stored targets.

**Method (pre-registered).** Re-score both anchors' pairs with
`zenmetrics batch --metric ssim2` (imazen `fast-ssim2` through the umbrella's `cpu-ssim2`;
format detection and decode are imazen codecs) on **today's** decode, and compare to the
stored value. Report median / p99 / max `|Δ|` and the fraction past the **0.5 dial-point
materiality**. Decision rule, fixed in advance: **if the fraction past 0.5 is 0, stored
ssim2 targets are REUSABLE across decoder eras** and the anchor keeps them; otherwise the
anchor's targets are re-scored at the anchor's own decode era.

## 8. The candidates — REGISTERED, NOT RUN

All share **identical weights, identical scaler, identical winsor guards** — only the output
spline differs, so rank is invariant by construction (to be verified to 5 dp on all five
corpora for every candidate). All are built by the owner (`bake_dial_refit`); no bake bytes
are edited by hand.

| arm | what changes vs `B_safesyn_curera` | mechanism it targets |
|---|---|---|
| `curera` (control) | — (the imazen-26 record's arm, rebuilt here) | era only |
| **(a) `negfloor`** | anchor targets **UNCLAMPED** (`ssim2`, keeping the negative rows) + `--neg-tail` | the floor: give `fit_spline_knots`'s bottom bins real `y < 0` evidence |
| **(b) `negfloor_topdense`** | (a) + top band densified toward identity | (a) plus `extend-top`'s saturation |
| **(c) `im26_negfloor_topdense`** | (b) on the imazen-26 anchor | the long-term corpus |
| **(d) knot sweep** | (b) with the `shared-anchor` percentile-edge count swept | the fit's own resolution at the ends |

## 9. Reproduction

```sh
cargo build --release -p zensim-validate --bin bake_verdict
D=/mnt/v/output/zensim/dialgate-2026-09-04
./target/release/bake_verdict --bake <bake.bin> \
    --dial-grid /mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet \
    --negtail-probe  $D/negtail_probe_372_2026-09-04.parquet \
    --identity-probe $D/identity_probe_372_2026-09-04.parquet \
    --corpora cid22,konjnd,kadid,tid,aic3 --full-json verdict.json
```

Artifacts: `/mnt/v/output/zensim/dialgate-2026-09-04/{ref,build,arms}/`.
