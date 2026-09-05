# D-peaks-372 lambda sweep + negative-tail anchor-weight lever (2026-09-05)

**Lane:** `claude-dpeaks-sweep`, jj workspace `~/work/zen/zensim--dpeaks-sweep`.
**Registered by:** the task brief handed to this lane, itself grounded in
[`d_peaks_372_postC_2026-09-05.md`](d_peaks_372_postC_2026-09-05.md) §5 — the
peaks arm (λ=2e-3, slice 0..227) reproduces shipped D's rank gain (CID22
+0.00798) but fails G-ADDR **A8** (negative-tail p1 −167.7 against the
incumbent's −212.1 and the `peer_ssim2` bar's −187.1), and §5's own words:
*"What was NOT tried … λ. A λ sweep on the 228 slice is a different
experiment and is registered, not run."* [`d_id100_2026-09-04.md`](d_id100_2026-09-04.md)
§7.1 additionally establishes the floor rule this sweep's second leg exercises:
*"The lever for the floor axes was 'the anchor/fit rows extended with negrich
rows carrying unclamped negative targets'."*

**Scope discipline:** this lane produces a TABLE and this RECORD. Nothing here
touches `zensim/src/profile.rs` or `zensim/weights/` — no default ships from
this lane regardless of outcome.

**Status at time of writing this section: ZERO fits have been run.** Everything
below is committed to before the first `fit-lasso` invocation, per the task's
"pre-register before the first fit" instruction.

---

## 1. Pre-registration

### 1.0 A naming coincidence, noted so nobody chases a ghost

The pre-existing `multiband_anchor_dial100.parquet` carries a column literally
named `anchor_weight` (2,000 rows, constant `1.0`). It is **unrelated** to the
new CLI flag this lane adds (same name, different owner): that column is read
only by `zensim_mlp_train`'s per-sample training sampler
(`zensim-validate/src/bin/zensim_mlp_train.rs:910,3133`) as a row-sampling
weight for MLP training, and `bake_dial_refit`'s `parquet_loader::load_parquet`
does not read it at all (verified by reading `parquet_loader.rs`:
`OwnedLoadedGroup` carries `feature_rows` + `human_scores` only, no weight
field; `fit_spline_knots(preds, tgt, n_edges, neg_tail)` takes plain slices).
The new `--anchor-weight` CLI flag on `fit-lasso` is a **file-level** repeat
count paired with `--anchor-parquet`, entirely independent of that column.

### 1.1 The owner gap this lane closes

Checked before writing any code: `bake_dial_refit fit-lasso`'s anchor-parquet
loading (`--anchor-stride`, row subsampling) and `fit_spline_knots`
(percentile-edge binning + bin-median) have **no existing weight mechanism**.
`fit_spline_knots` cannot consume a continuous per-row weight — it sorts
predictions and takes bin medians of whatever rows are physically present.
So "up-weighting" a subset means **row duplication**, matching the precedent
`identity_anchor_sg_n21` already set (`d_id100_2026-09-04.md` §5: mass fraction
via literal row count). Added: `--anchor-weight <N>` (`Vec<usize>`, paired
positionally with `--anchor-parquet`, default all-1 = today's behavior,
byte-identical). Landed in commit `89f5eb4c3373` (pushed, verified on
`origin/main`), with 4 unit tests (count-resolution validation on both flag
names; weight=1 single push; weight=N bit-exact duplication via `to_bits()`,
NaN-safe; weight=0 appends nothing) — `cargo test -p zensim-validate --bin
bake_dial_refit`: 20/20 pass, clippy clean, fmt clean.

### 1.2 Grid (a) — lambda sweep on slice 0..227

Six values: **5e-4, 1e-3, 2e-3 (CONTROL), 4e-3, 8e-3, 1.6e-2**. Every other
flag is `d_peaks_372_postC_2026-09-05.md` §8's exact reproduction command for
the peaks arm, unchanged:

```sh
bake_dial_refit fit-lasso --space raw --target human_score --lam <LAM> --tau 0 \
    --n-sweeps 400 --tol 1e-10 \
    --slice-file /mnt/v/output/zensim/dpeaks372-2026-09-05/slices/a228.idx \
    --gram /mnt/v/output/zensim-multicodec-probe/linear-probe/grams/safesyn.npz --weight 1.0 \
    --anchor-parquet /mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet \
    --anchor-parquet /mnt/v/output/zensim/did100-2026-09-04/work/identity_anchor_sg_n21.parquet \
    --anchor-target ssim2_gpu \
    --embed-repro --feature-set-id 'basic+peaks+masked+iw@w372/v1pre#d16a1091' \
    --out sweep/bakes/Dsweep_<TAG>_raw.bin
bake_dial_refit extend-top --in sweep/bakes/Dsweep_<TAG>_raw.bin \
    --anchor /mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet \
    --target-col target_score --out sweep/bakes/Dsweep_<TAG>_dial.bin
```

(`sweep/` = `/mnt/v/output/zensim/dpeaks372-2026-09-05/sweep/`.) Reused
read-only inputs, verified present with the shas the referenced docs recorded
(all under existing `did100-2026-09-04`/`dpeaks372-2026-09-05` dirs, nothing
copied): gram sha `904a8e80…`, multiband anchor sha `594b3df5…`, identity
anchor sha `a8f32c69…`, slice file `a228.idx` verified == `seq 0 227`
(sha `e7807011…`), reused as-is rather than rebuilt.

Slice/lambda tags: `lam5em4`, `lam1em3`, `lam2em3` (control), `lam4em3`,
`lam8em3`, `lam16em3`.

**Control gate (must pass before trusting any other grid-(a) result):**
`lam2em3`'s `_dial.bin`, with `zentrain.repro` stripped (`bake_dial_refit
strip --key zentrain.repro`), must `cmp` byte-identical to
`/mnt/v/output/zensim/dpeaks372-2026-09-05/bakes/Dpeaks372_id100negrich_dial_fsid.bin`
with the same key stripped. (`feature_set_id` is not stripped from either side
— both use the identical stamp string, so it does not need to be excluded to
match.)

### 1.3 Grid (b) — negative-tail anchor-weight lever, spline step only

**Selection:** after grid (a) scores, take the **top two λ by CID22
`srocc_signed`** (postC era, `dialgate_arms.sh score` reads it). Call them
λ-best-1 and λ-best-2 (order does not matter for what follows — both get the
same two weight arms).

**The lever:** an extra `--anchor-parquet` holding ONLY the 147 rows of
`multiband_anchor_dial100.parquet` where `ssim2_gpu < 0` (built from that file,
read-only — `sweep/work/negrich_only_anchor.parquet`, verified 147 rows,
range [−64.160, −0.124], built and checked BEFORE this pre-registration was
written since it required no fit and is pure data prep). Because those 147
rows already appear ONCE in the base `multiband_anchor_dial100.parquet` pass,
adding the extra file at `--anchor-weight W_extra` gives the negative subset a
**total** representation of `1 + W_extra`× baseline:

- **weight ×2** (`W_extra = 1`): just append the third `--anchor-parquet`,
  default weight (omit `--anchor-weight` entirely — all-1 default already
  gives exactly this).
- **weight ×4** (`W_extra = 3`): `--anchor-weight 1 --anchor-weight 1
  --anchor-weight 3` (positional, multiband / identity / negrich_only).

For each of the two selected λ, at the SAME `--lam`, slice, gram and
`--tau`/`--n-sweeps`/`--tol` as its grid-(a) arm:

```sh
# w2 (negative subset at total weight x2)
bake_dial_refit fit-lasso <same flags as the grid-(a) arm for this lambda> \
    --anchor-parquet multiband_anchor_dial100.parquet \
    --anchor-parquet identity_anchor_sg_n21.parquet \
    --anchor-parquet sweep/work/negrich_only_anchor.parquet \
    --anchor-target ssim2_gpu --embed-repro --feature-set-id '<same id>' \
    --out sweep/bakes/Dsweep_<LAMTAG>_w2_raw.bin
# w4 (negative subset at total weight x4)
bake_dial_refit fit-lasso <same flags as the grid-(a) arm for this lambda> \
    --anchor-parquet multiband_anchor_dial100.parquet \
    --anchor-parquet identity_anchor_sg_n21.parquet \
    --anchor-parquet sweep/work/negrich_only_anchor.parquet \
    --anchor-weight 1 --anchor-weight 1 --anchor-weight 3 \
    --anchor-target ssim2_gpu --embed-repro --feature-set-id '<same id>' \
    --out sweep/bakes/Dsweep_<LAMTAG>_w4_raw.bin
```
then `extend-top` each exactly as in §1.2.

**Registered sanity control for grid (b):** the CD lasso fit (`w`, `bias`,
`mu`, `sd`) is computed from `--gram` alone (before any anchor row is read),
so the w2/w4 arms MUST have the identical underlying linear model to their
parent grid-(a) arm — only the spline may differ. Checked by `bake_dial_refit
strip --key zentrain.repro` (and, since the `w2`/`w4` arms do carry the
identical `feature_set_id` stamp already, no second key needs stripping)
followed by `cmp` against the parent λ's similarly-stripped `_raw.bin` for at
least one of the two selected λ. This is what makes "the negative-tail lever
in the SPLINE step only" a checked fact, not an assumption.

### 1.4 Grading — every arm, one path

`ZL_ERA=postC scripts/dialgate_arms.sh score <label> <bake.bin> 372` (the
runtime-era 372 ruler `d_peaks_372_postC_2026-09-05.md` built), which runs
`bake_verdict --full-json --gaddr-json` against the postC root + postC
instruments. Same tool, same era, same corpora list
(`cid22,konjnd,kadid,tid,aic3`) as the doc this lane extends.

**Additional axis — `hfnl_cid22band`** (CID22's top-MOS band, non-circular
near-lossless read; owner: `benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py`,
`BAND_LO=0.8`, per `benchmarks/hfnl944_2026-09-01.pointer.md` /
`benchmarks/free_features_2026-09-01.md` §3.2's established usage). Per-pair
CID22 dumps (`bake_verdict --regime 372 --features-root <postC root> --corpora
cid22 --per-pair-output <dir>/<label>_cid22.tsv --per-pair-refs --output
/dev/null`) feed both this script (by column NAME, `human`/`pred`/`ref`) and
`scripts/wave6_paired_bootstrap.py` (by column POSITION 0/1 — verified
identical convention: `format_per_pair` emits `human\tpred\tref`, so both
readers agree on which column is which). Reported number: pooled
`srocc_signed` at `BAND_LO=0.8` (the script's "signed point estimate
(pooled, in-band)" line).

**Paired bootstrap vs shipped D** (`scripts/wave6_paired_bootstrap.py`, same
postC per-pair dumps, `--corpus {cid22,tid,aic3,konjnd}`, `--ref D`, `--b 2000
--seed 20260905` — same seed `d_peaks_372_postC_2026-09-05.md`'s own headline
table used, for direct comparability). Default (unsigned) `|SROCC|` mode,
matching that doc's own usage and the KonJND-column convention the script's
own docstring states (*"read as \|SROCC\| exactly as bake_verdict reports
it"*).

### 1.5 What gets reported per arm

CID22 Δ vs shipped D with 95% CI; TID; AIC-3; KonJND (`|SROCC|`);
`hfnl_cid22band` (pooled, `BAND_LO=0.8`); G-ADDR CONTRACT n/6; G-ADDR
REGRESSION n/9 with the failing axis ids; negative-tail probe `p1` and `min`;
dial grid `min`/`p5`/`max`; identity (`dial(0⃗)`).

### 1.6 Ship rule (verbatim from the task brief)

> CID22 ≥ D (CI not excluding a gain) AND contract 6/6 AND no regression axis
> lost vs shipped D's 7/9 (A8 must PASS).

"No regression axis lost vs shipped D's 7/9" is read as: the candidate's set
of PASSING regression axes must be a superset of shipped D's seven (i.e. it
may not newly fail anything D passes), and A8 specifically must read PASS
(not just "regression count >= 7", since a candidate could pass a different
7-of-9 and still have re-broken A8, which is the exact failure mode
`d_peaks_372_postC_2026-09-05.md` §5 measured for the unweighted peaks arm).

### 1.7 Nothing installed regardless of outcome

Per the lane's scope discipline: even an arm that clears every bar above is a
**proposal**, reported with its exact reproduction command and shas — not a
default flip. `ZensimProfile::D` and `zensim/weights/` are untouched by this
lane under every circumstance.

---

## 2. Grid (a) results

Code landed: `89f5eb4c3373`. Control gate: `Dsweep_lam2em3_dial.bin`, repro
stripped, `cmp`-identical to `Dpeaks372_id100negrich_dial_fsid.bin` similarly
stripped (sha256 `3e6ded209020f081f129be6afacfd5caf10c3fcc6dcd031751e83b177dae4d21`
on both sides) — **PASS**. `raw.bin` == `dial.bin` byte-for-byte on all six
arms (`extend-top` is a no-op for this lineage, as `d_peaks_372_postC` §4.1
found for the original arm).

| arm | λ | act | CID22 | Δ vs D [95% CI] | TID | AIC-3 | \|KonJND\| | hfnl_cid22band | contract | reach/min/max/p5/p95/DR | identity |
|---|--:|--:|--:|---|--:|--:|--:|--:|:--:|---|--:|
| D (shipped) | — | 28 | 0.86333 | — | 0.82369 | 0.77700 | 0.53670 | 0.4339 | 6/6 | 156.437 / −57.109 / 99.328 / 8.772 / 96.122 / 87.350 | 100.0000 |
| lam5em4 | 5e-4 | 56 | 0.86621 | +0.00297 [+0.00009,+0.00561] | 0.77611 | 0.78358 | 0.61340 | 0.4458 | 6/6 | 158.566 / −58.566 / 100.000 / 8.747 / 97.595 / 88.848 | 100.0000 |
| lam1em3 | 1e-3 | 38 | 0.87179 | +0.00852 [+0.00544,+0.01146] | 0.79479 | 0.77722 | 0.59740 | 0.4654 | 6/6 | 156.812 / −57.488 / 99.324 / 9.058 / 96.949 / 87.891 | 99.9999 |
| **lam2em3** (=Dpeaks, control) | 2e-3 | 26 | 0.87126 | +0.00798 [+0.00476,+0.01116] | 0.80502 | 0.76939 | 0.53844 | 0.4609 | 6/6 | 155.647 / −56.190 / 99.456 / 9.357 / 96.495 / 87.139 | 100.0000 |
| lam4em3 | 4e-3 | 15 | 0.86923 | +0.00598 [+0.00254,+0.00937] | 0.80936 | 0.76575 | 0.52780 | 0.4539 | 6/6 | 157.476 / −57.941 / 99.535 / 8.907 / 96.434 / 87.527 | 99.9999 |
| lam8em3 | 8e-3 | 13 | 0.86583 | +0.00252 [−0.00116,+0.00635] | 0.81925 | 0.75922 | 0.50546 | 0.4427 | 6/6 | 158.735 / −59.048 / 99.688 / 7.870 / 96.507 / 88.637 | 100.0000 |
| lam16em3 | 1.6e-2 | 13 | 0.86371 | +0.00042 [−0.00345,+0.00436] | 0.81936 | 0.75643 | 0.50048 | 0.4358 | 6/6 | 158.412 / −58.741 / 99.672 / 8.449 / 96.578 / 88.129 | 100.0000 |

(SROCC columns are `|SROCC|`; CID22/TID/AIC-3/KonJND deltas and CIs are
`scripts/wave6_paired_bootstrap.py --ref D --b 2000 --seed 20260905`, same
postC per-pair dumps; AIC-3/KonJND deltas: lam5em4 AIC-3 +0.00649
[+0.00103,+0.01280], KonJND +0.07626 [+0.05204,+0.10214]; lam1em3 AIC-3
+0.00016 [−0.00579,+0.00651], KonJND +0.06013 [+0.03341,+0.08819]; lam2em3
AIC-3 −0.00770 [−0.01455,−0.00051], KonJND +0.00226 [−0.02894,+0.03144];
lam4em3 AIC-3 −0.01118 [−0.01870,−0.00388], KonJND −0.00871
[−0.04128,+0.02178]; lam8em3 AIC-3 −0.01770 [−0.02605,−0.00920], KonJND
−0.03109 [−0.06536,+0.00231]; lam16em3 AIC-3 −0.02047 [−0.02988,−0.01104],
KonJND −0.03691 [−0.07285,−0.00057]. TID always loses, every CI excludes zero
except lam16em3's [−0.00873,+0.00018].)

**Regression axes (A1-A9), old ssim2-anchor pins (for continuity, see §4 —
NOT the ship rule as of this task):** every arm passes A1-A6; A7 fails for
every arm including D; A9 fails for every arm including D. **A8 is the one
that varies**: D/lam5em4/lam1em3/lam8em3 PASS A8; lam2em3/lam4em3/lam16em3
FAIL it (this is the `d_peaks_372_postC` finding, reproduced: heavier
regularization on this slice pulls the negative-tail probe's p1 shallower
than the −187.131 ssim2 pin). No arm regresses A1-A6 or A9 relative to D.

**Selection for grid (b):** top two by CID22 point estimate are **lam1em3**
(0.87179) and **lam2em3** (0.87126) — both with CID22 CIs strictly excluding
zero, and their difference from each other (0.00053) is inside noise.

---

## 3. Grid (b) results — the negative-tail anchor-weight lever

`negrich_only_anchor.parquet` built: 147 rows, `ssim2_gpu` range
[−64.160, −0.124] (matches `d_id100_2026-09-04.md` §6's "147 rows genuinely
negative"). **Spline-only sanity control: PASS on all 4 arms** — stripping
`zentrain.repro` then `zentrain.output_calibration_spline` from every w2/w4
bake reproduces its parent λ's identically-stripped bytes exactly (e.g.
`lam1em3_w2`/`lam1em3_w4` both → the same bytes as `lam1em3`'s stripped raw
model). This is a measured fact, not an assumption: the CD lasso fit never
reads an anchor row, so the lever cannot touch `w`/`bias`/`mu`/`sd` by
construction, and the control confirms it did not.

| arm | negtail min | negtail p1 | frac<0 | grid min | CID22 | contract | other-axis vs D |
|---|--:|--:|--:|--:|--:|:--:|---|
| lam1em3 (parent) | −211.558 | −189.835 | 0.8620 | −57.488 | 0.87179 | 6/6 | none worse |
| lam1em3_w2 | −141.502 | −115.925 | 0.8680 | −55.437 | 0.87179 | 6/6 | none worse |
| lam1em3_w4 | −57.405 | −57.405 | 0.8865 | −56.013 | 0.87179 | 6/6 | none worse |
| lam2em3 (parent) | −213.149 | −167.715 | 0.8755 | −56.190 | 0.87126 | 6/6 | none worse |
| lam2em3_w2 | −130.716 | −107.382 | 0.8815 | −55.350 | 0.87126 | 6/6 | **A2 worse** |
| lam2em3_w4 | −57.405 | −57.405 | 0.8990 | −55.602 | 0.87126 | 6/6 | none worse |

CID22/TID/AIC-3/KonJND/hfnl_cid22band are **bit-identical to the parent λ**
for every w2/w4 arm except a sub-0.0003 TID drift at w4 (`lam1em3_w4` TID
0.79452 vs parent 0.79479; `lam2em3_w4` 0.80489 vs 0.80502) — the expected
residue of a near-monotone (not perfectly monotone) spline change, matching
the tiny C1 mono drop (0.9917→0.9912 for the lam2em3 family).

**Finding: heavier weight on the negative subset SHRINKS the probe's own
negative reach, and moved it opposite to the naive expectation.** More copies
of the 147 negrich rows in the spline anchor pulls the model's very-negative
raw predictions toward the anchor's OWN worst target (~−64), rather than
letting the OOD floor formula (`ys[0] − (ys[n−1] − ys[0])`) extrapolate
further past a sparser, more spread-out negative knot layout. w4 saturates
this at exactly −57.405 for BOTH λ (a floor-formula artifact, not a
coincidence — both hit the same OOD floor bound once the bottom knot's `y`
value stabilizes near the anchor's own extremes). **This is a real,
measured trade, not a bug**: the lever does what "up-weight the negative
rows" should do to the SPLINE's shape (pull the bottom knots' Y-values toward
the anchor's own negative evidence) — it just isn't the direction that helps
the old ssim2-pinned A7/A8 axes, which reward MORE extreme extrapolation, not
more anchor-faithful calibration.

**lam2em3_w2 is the one grid-(b) arm that regresses an axis untouched by
either rule change**: its canonical-grid `min` (−55.350) sits 0.005 short of
the ssim2 A2 bar (−55.354544) that shipped D clears. This is a genuine,
if narrow, floor regression on REAL codec-derived grid cells (not the
synthetic negtail probe) and disqualifies it regardless of which negative-tail
rule is in force.

---

## 4. MID-TASK USER RULING — the negative-tail bar is redefined

The coordinator relayed this ruling **verbatim**, after grid (a) and (b) had
already been built and graded against the pre-registered (ssim2-pinned) rule:

> "the negative tail bar is entirely arbitrary. below -5-50" — i.e. the
> ssim2-pinned A7 (min ≤ −770.6) and A8 (p1 ≤ −187.1) bars are NOT the ship
> rule; the product-meaningful negative range is below −5 down to about −50.

**A7/A8 at the old ssim2 pins are retained in §2/§3's tables for continuity
only — they are NOT the ship rule as of this ruling.** The registry re-pin
(`benchmarks/dial_addressability_floor_2026-09-04.json`,
`zensim-validate/src/dial_addressability.rs`) is explicitly **out of this
lane's scope** and is being done by a separate lane; nothing in either file
was touched here, and every number in §5 below comes from read-only
`bake_verdict`/`bake_dial_refit predict` reads plus a small analysis script
(`negtail_band_checks.py`, doing only `min`/percentile/fraction arithmetic on
a `predict --score-units` dump — no owner is bypassed).

**Corrected checks**, computed via `bake_dial_refit predict --bake <bake>
--corpus negtail_probe_372_postC_2026-09-05.parquet --score-units --out
<tsv>` (positional row-order join to the probe's own `ssim2_gpu` column — the
tool's documented "file row order is the join contract"):

1. negtail probe dial `min` ≤ −50
2. negtail probe dial `p1` ≤ −50
3. on probe rows whose ssim2 truth ∈ [−50, −5) (798 of 2000 = 39.9%): fraction
   the dial places below −5, and fraction below 0 (report only, per the
   ruling's own wording — no threshold stated for this one)
4. canonical (postC) dial **grid** `min` ≤ −5 (bake_verdict's own A2 read,
   not recomputed) — "the dial reaches below zero on real codec output"

**Corrected ship rule**: CID22 ≥ shipped D with CI not excluding a gain,
contract 6/6, checks (1)-(4) pass, and every OTHER regression axis (A1-A6,
A9 — i.e. everything except the now-superseded A7/A8) no worse than shipped
D. Still: **NEVER INSTALL**, regardless of outcome (§1.7 unchanged).

---

## 5. Master table — every arm, both rulings

| arm | ck1 min≤−50 | ck2 p1≤−50 | band frac<−5 | band frac<0 | ck4 grid_min≤−5 | other-axis vs D | CID22 CI excl. 0 | **NEW ship rule** | OLD ship rule (A8 pin) |
|---|:--:|:--:|--:|--:|:--:|:--:|:--:|:--:|:--:|
| D (shipped) | — | — | 0.8546 | 0.8772 | — | — | — | *(baseline)* | *(baseline)* |
| lam5em4 | PASS | PASS | 0.7331 | 0.7569 | PASS | none worse | **yes** [+0.00009,+0.00561] | **PASS** | PASS |
| lam1em3 | PASS | PASS | 0.7669 | 0.7907 | PASS | none worse | **yes** [+0.00544,+0.01146] | **PASS** | PASS |
| lam2em3 | PASS | PASS | 0.7782 | 0.8108 | PASS | none worse | **yes** [+0.00476,+0.01116] | **PASS** | FAIL (A8) |
| lam4em3 | PASS | PASS | 0.7832 | 0.8145 | PASS | none worse | **yes** [+0.00254,+0.00937] | **PASS** | FAIL (A8) |
| lam8em3 | PASS | PASS | 0.8045 | 0.8421 | PASS | none worse | no [−0.00116,+0.00635] | FAIL (CID22 CI) | FAIL (CID22 CI) |
| lam16em3 | PASS | PASS | 0.8158 | 0.8434 | PASS | none worse | no [−0.00345,+0.00436] | FAIL (CID22 CI) | FAIL (A8 + CID22 CI) |
| lam1em3_w2 | PASS | PASS | 0.7782 | 0.7982 | PASS | none worse | **yes** (=lam1em3) | **PASS** | FAIL (A8) |
| lam1em3_w4 | PASS | PASS | 0.7920 | 0.8296 | PASS | none worse | **yes** (=lam1em3) | **PASS** | FAIL (A8) |
| lam2em3_w2 | PASS | PASS | 0.7957 | 0.8221 | PASS | **A2 worse** | yes (=lam2em3) | FAIL (A2) | FAIL (A2 + A8) |
| lam2em3_w4 | PASS | PASS | 0.8133 | 0.8471 | PASS | none worse | **yes** (=lam2em3) | **PASS** | FAIL (A8) |

Every arm clears checks (1), (2), (4) — the re-scoped negative-tail bar turns
out to be **easy** for this whole family (the shallowest `min`/`p1` in the
entire sweep is `lam1em3_w4`/`lam2em3_w4`'s −57.405, still comfortably past
−50). The re-scoping did not "relax the rule to fit the arms" — it was
decided by the user BEFORE these numbers were re-read, on the stated grounds
that the ssim2-anchored pins were never the intended product bar (§4).

---

## 6. Verdict

**Seven of the ten built arms satisfy the corrected ship rule:** `lam5em4`,
`lam1em3`, `lam2em3`, `lam4em3`, `lam1em3_w2`, `lam1em3_w4`, `lam2em3_w4`.
**Three do not:** `lam8em3` and `lam16em3` (CID22 gain over shipped D is not
statistically distinguishable — 95% CI includes zero), and `lam2em3_w2`
(regresses A2, the canonical grid's pooled floor, by 0.005 against shipped
D — the one axis in this whole sweep that neither rule change touches).

**Under the ORIGINAL (pre-ruling) ship rule, only two arms passed**:
`lam5em4` and `lam1em3` — both because the negative-tail weight lever
*always* flips A8 from PASS to FAIL (§3's mechanism finding), so no
`_w2`/`_w4` variant could ever have passed the old rule regardless of λ.

**No arm dominates on every corpus.** `lam1em3` has the best CID22 gain
(+0.00852) and best KonJND gain (+0.06013) in the sweep, but loses TID
(−0.02878, CI excludes zero) and is a rank tie on AIC-3. `lam5em4` is the
only arm with a positive AIC-3 delta (+0.00649) but has the sweep's worst TID
loss (−0.04749). The weight-lever variants change nothing about this
trade — they are rank-identical to their parent λ on every corpus tested
(§3) — they only reshape the spline's negative-tail floor, which (per §4-§5)
turns out not to matter for any ranking axis.

**This is a table and a record, per the lane's scope. Nothing ships.**
`ZensimProfile::D` and `zensim/weights/` are unchanged; the registry re-pin
implied by §4 belongs to a separate, already-notified lane.

---

## 7. Additional reproduction (grid (b) + the negtail-band checks)

The grid-(b) fit commands are in §1.3. The negative-tail band-check tool
(§4) was one-off scratch, not a repo tool (per the ruling: "report values
only", no owner file touched):

```sh
# per arm, on the negtail probe (score units = the bake's own dial units)
bake_dial_refit predict --bake <bake.bin> \
    --corpus /mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/negtail_probe_372_postC_2026-09-05.parquet \
    --score-units --out <predict.tsv>
# then, per arm: min(pred), p1(pred), and on rows where ssim2_gpu in [-50,-5):
# fraction(pred < -5), fraction(pred < 0) -- ssim2_gpu read straight from the
# probe parquet, joined POSITIONALLY (predict's own documented contract).
```

All artifacts: `/mnt/v/output/zensim/dpeaks372-2026-09-05/sweep/{bakes,fits,
arms,gaddr,verdicts,work,logs}`. Per-arm gaddr JSON:
`sweep/arms/gaddr_<label>.json`; per-pair dumps: `sweep/work/<label>_<corpus>.tsv`;
negtail predict dumps: `sweep/work/predict_<label>_negtail.tsv`; band-check
table: `sweep/work/negtail_band_checks_grid{a,b}.tsv`; bootstrap logs:
`sweep/work/boot_grid{a,b}_<corpus>.txt`, `sweep/work/hfnl_cid22band_grid{a,b}.txt`.
