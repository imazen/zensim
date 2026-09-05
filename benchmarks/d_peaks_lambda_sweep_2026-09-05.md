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

## 4. MID-TASK USER RULING — the negative-tail bar is redefined, then CORRECTED to be PER CODEC

The coordinator relayed a ruling **verbatim**, after grid (a) and (b) had
already been built and graded against the pre-registered (ssim2-pinned) rule.
**A first transcription of that ruling was garbled** (it named "−5" as a
threshold) and was corrected by the coordinator in a follow-up message before
this section was finalized. **Only the corrected version is used below; no
"−5" or "[−50,−5]" number appears anywhere in this record.** The two
messages, verbatim, in order:

> (garbled, superseded) "the negative tail bar is entirely arbitrary. below
> -5-50" — read at the time as a [−50,−5) band.

> (correction, OPERATIVE) "the number is −50, NOT −5 (the first message was
> garbled), and the tail is judged PER CODEC — 'codecs are all different,
> some go lower than others'."

**A7/A8 at the old ssim2 pins are retained in §2/§3's tables for continuity
only — they are NOT the ship rule as of either ruling.** The registry re-pin
(`benchmarks/dial_addressability_floor_2026-09-04.json`,
`zensim-validate/src/dial_addressability.rs`) is explicitly **out of this
lane's scope** and is being done by a separate lane (now visible as sibling
jj workspace `gaddr-repin`); nothing in either file was touched here, and
every number below comes from read-only `bake_verdict`/`bake_dial_refit
predict` reads plus one-off scratch scripts doing only
min/percentile/fraction/join arithmetic — no owner is bypassed.

**Corrected checks, verbatim from the correction:**

1. negtail probe dial `min` ≤ −50, reported **pooled AND per codec family** —
   pass = every family whose ssim2 min is ≤ −50 also has dial min ≤ −50
   (families ssim2 never takes below −50 are **exempt**, listed below).
2. negtail probe dial `p1` ≤ −50, **pooled only**.
3. **per codec family**, the fraction of rows with ssim2 truth ≤ −50 that the
   dial places ≤ −50 (report; no hard bar yet).
4. canonical grid: **per codec family**, where ssim2's min on that family's
   cells is ≤ −50 the dial's min must be ≤ −50 too; report every family's
   dial min beside ssim2's.

**A naming problem the data forced, stated up front:** checks (1)-(3) are
scored on the **negative-tail PROBE**, which is a KADIS-derived
synthetic-distortion probe recovered by a join (§1.3 of the referenced doc)
— it carries **no codec/encoder information at all** (KADIS applies
distortion FUNCTIONS — blur, noise, color shift, JPEG/JP2K compression,
brightness shift, etc. — not zen encoders). Its only per-category axis is
KADIS's own 25-way `dist_type`/`dist_name` taxonomy. Recovered here by
joining the probe's `ssim2_gpu` column to
`kadis700k_canonical_gpu_2026-07-01.parquet`'s `score_ssim2_gpu` (exact
match; the probe's `f0/f1/f2` do NOT match `feat_0/feat_1/feat_2` closely
enough to use as a join key — the probe is a pre-option-C extraction, the
canonical table a separate era — verified: 0/2000 matched on the 4-column
key, 2000/2000 matched on `ssim2_gpu` alone, 49/2000 ambiguous across >1
candidate `dist_type`, first candidate taken for those). **Every table below
labels this axis "distortion family", never "codec family"**, to keep it
visibly distinct from check (4)'s real `codec` column (`jpeg`/`webp`/`avif`/
`jxl`) on the canonical dial grid.

### 4.1 Negtail probe — distortion families

24 families present (1 of KADIS's 25 codes, `denoise_dncnn`'s sibling
`jp2k`... — all 24 non-empty codes in this 2,000-row probe). **1 family is
EXEMPT** (ssim2 never reaches −50 there): `noneccentricity` (n=19, ssim2 min
−46.82). **23 families are GRADED** (ssim2 min ≤ −50), spanning n=8
(`mean_shift`) to n=232 (`color_block`).

**Check (1) result: 0 of 11 arms — including shipped D — pass, on every
single one.** The blocker is the same family for every arm:

| family | n | ssim2 min | D dial min | worst arm dial min |
|---|--:|--:|--:|--:|
| `mean_shift` | 8 | −63.548 | **−28.111** | −14.479 (lam2em3_w4) |

D's own 8 predictions on this family: truths `[−63.5,−59.1,−40.6,−36.9,
−34.8,−34.5,−33.3,−31.3]` pair with dial predictions
`[43.2,21.9,13.3,7.3,5.9,−7.0,−16.3,−28.1]` (sorted independently — this is
not a single-outlier artifact: D's DEEPEST prediction on this family, −28.1,
sits more than 35 points short of ssim2's shallowest qualifying truth,
−63.5). Every other one of the 23 graded families is cleared by every arm.
**mean_shift is thin (n=8)** — thin enough that a "no arm including the
incumbent ever meets this" result is itself informative about the bar's
robustness at this sample size, which is reported here rather than silently
dropped or worked around.

Full per-family table (all 11 arms × 24 families):
`sweep/work/percodec_checks_output.txt` §"(1) dial min, pooled + per family".
Pooled mins (unaffected by the family split): D −213.149; lam5em4 −211.558;
lam1em3 −211.558; lam2em3 −213.149; lam4em3 −213.149; lam8em3 −210.363;
lam16em3 −210.363; lam1em3_w2 −141.502; lam1em3_w4 −57.405; lam2em3_w2
−130.716; lam2em3_w4 −57.405.

**Check (2) (pooled p1 ≤ −50): all 11 arms PASS**, including D (`p1
−212.121`); the shallowest in the sweep is `lam1em3_w4`/`lam2em3_w4` at
−57.405, still 7 points past the bar.

**Check (3) (report only)**: 1,187 of 2,000 rows have ssim2 truth ≤ −50. Per
family, the fraction where the dial ALSO reaches ≤ −50 ranges from 0.000
(`blur_motion` n=10 and `contrast` n=38 — every arm scores 0.000 or close
to it on `blur_motion` up to 0.500 at higher λ; `contrast` is 0.000 for
every arm without exception) to 1.000 (`color_shift`, `compress_jp2k`,
`pixelate` for most arms). Full table: same file, §"(3)". This axis has no
bar, so it is reported and not scored into the verdict.

### 4.2 Canonical (postC) dial grid — real codec families

| codec | n | ssim2 min |
|---|--:|--:|
| jpeg | 880 | −8.045 (**EXEMPT** — never reaches −50) |
| jxl | 1504 | −39.686 (**EXEMPT**) |
| webp | 640 | **−51.847** (GRADED) |
| avif | 1400 | **−55.355** (GRADED) |

Only **webp and avif** are graded; jpeg and jxl are exempt on this grid by
construction (their own worst cell never reaches −50, matching the user's
"some codecs go lower than others").

**Check (4) result — dial min per codec, beside ssim2's:**

| arm | avif (ssim2 −55.35) | jpeg (ssim2 −8.05) | jxl (ssim2 −39.69) | webp (ssim2 −51.85) | **ck4 (avif+webp both ≤ −50)** |
|---|--:|--:|--:|--:|:--:|
| D (shipped) | −57.109 | −3.716 | −45.271 | −48.119 | **FAIL** (webp short by 1.9) |
| lam5em4 | −58.566 | −5.063 | −49.694 | **−50.830** | **PASS** |
| lam1em3 | −57.488 | −7.297 | −47.038 | −48.383 | FAIL (webp) |
| lam2em3 | −56.190 | −9.193 | −46.077 | −45.724 | FAIL (webp) |
| lam4em3 | −57.941 | −10.305 | −47.528 | −46.383 | FAIL (webp) |
| lam8em3 | −59.048 | −12.077 | −49.076 | −46.046 | FAIL (webp) |
| lam16em3 | −58.741 | −13.173 | −49.154 | −46.671 | FAIL (webp) |
| lam1em3_w2 | −55.437 | −9.863 | −47.037 | −48.155 | FAIL (webp) |
| **lam1em3_w4** | −56.013 | −7.675 | **−50.608** | **−51.355** | **PASS** |
| lam2em3_w2 | −55.350 | −11.748 | −47.291 | −46.997 | FAIL (webp) |
| **lam2em3_w4** | −55.602 | −8.437 | **−50.332** | **−50.136** | **PASS** |

**Shipped D itself fails check (4)**, by 1.9 points on webp — a fact worth
stating plainly rather than working around: the corrected rule, applied
literally, is not currently met by the incumbent either. Three arms DO clear
it outright: `lam5em4`, `lam1em3_w4`, `lam2em3_w4` — all three reach jxl below
−49 too (not required, jxl is exempt, but it is the closest any arm comes to
clearing every codec unconditionally).

### 4.3 D-relative comparison (supplementary — the checks above are absolute bars, not "no worse than D")

Because D itself fails checks (1) and (4) as literal bars, a per-family
comparison against D is reported as additional, genuinely informative
context — it is NOT part of the checks (1)-(4) pass/fail definition above,
which the correction stated as absolute thresholds with a *family-exemption*
mechanism already built in (that mechanism, not a D-relative comparison, is
what answers "codecs are all different, some go lower than others").

**Grid, per codec, vs D:** `lam5em4` and `lam1em3` are worse than D on
**zero** codecs. `lam2em3` and `lam2em3_w2` are worse on avif AND webp.
`lam4em3`/`lam8em3`/`lam16em3` are worse on webp only. `lam1em3_w2`/
`lam1em3_w4`/`lam2em3_w4` are worse on avif only (their webp is BETTER than
D's, which is exactly why the w4 pair clears check (4) outright).

**Negtail probe, per distortion family, vs D:** every arm is worse than D on
a LARGE fraction of families — `lam5em4`/`lam8em3` on 13/24, `lam4em3` on
15/24, `lam16em3` on 16/24, `lam1em3` on 21/24, `lam2em3` on 22/24, and
**all four weight-lever arms on 24/24 (every family, no exceptions)**. This
is the sharpest, most unambiguous finding of the whole re-check: **the
`--anchor-weight` lever trades a narrow, targeted gain (grid webp/avif, per
§4.2) for a UNIVERSAL loss of floor depth across every distortion family in
the negative-tail probe**, not just the families it was built to help. The
mechanism is the one §3 already measured (heavier weight pulls the spline's
bottom knots toward the anchor's own worst evidence, ~−64, rather than a
sparser layout's further extrapolation) — §4.3 shows that mechanism is
family-*blind*: it shallows the WHOLE probe, uniformly, regardless of which
distortion produced a given row.

Full per-family D-relative tables: `sweep/work/percodec_relative_to_D.txt`.

---

## 5. Verdict (SUPERSEDES the earlier "7 of 10 pass" conclusion)

**Applying the checks literally (absolute bars, family-exemption built in,
per the correction): 0 of 11 arms — including shipped D — pass check (1).**
The blocker is `mean_shift` (n=8), where no arm's dial reaches within 35
points of ssim2's shallowest qualifying truth. Check (2) passes for all 11.
Check (4) passes for exactly 3: `lam5em4`, `lam1em3_w4`, `lam2em3_w4` — and
**D itself does not pass check (4)** (webp −48.119, needs ≤ −50).

**Consequence for the ship rule** ("CID22 ≥ D with CI, contract 6/6, checks
(1)-(4) pass, other regression axes no worse than D", per the correction's
"ship rule otherwise unchanged"): **since check (1) is unmet by every arm
including the incumbent, ZERO of the 11 arms satisfy the full corrected ship
rule.** This is a genuine, measured result, not an artifact of this lane's
choices — the same 8-row family blocks D's own pre-existing bake.

**If the coordinator/user instead wants the OPERATIVE bar to be "no worse
than D, family-by-family" (a different, D-relative rule this task did not
originally specify — reported in §4.3 as context, not substituted for the
literal rule):** only `lam5em4` and `lam1em3` are worse-than-D on zero grid
codecs, but BOTH are still worse than D on the negtail probe's per-family
mins (13/24 and 21/24 families respectively) — so even under this relaxed
reading, **no arm is strictly Pareto-better than D across every family of
every instrument.** `lam5em4` is the closest to a clean win (zero grid
regressions, fewest negtail-family regressions of any non-identical arm,
CID22 CI positive [+0.00009,+0.00561], and it also clears check (4)
outright) — offered as the most defensible SINGLE candidate if a coordinator
wanted to pick one, not as a passing verdict under either stated rule.

**Everything from §2-§3 about rank (CID22/TID/AIC-3/KonJND/hfnl_cid22band)
and about the weight lever being spline-only is UNCHANGED by this
correction** — only the negative-tail pass/fail verdict moves. In particular
§3's mechanism finding ("heavier weight shrinks the probe's own reach") is
now sharpened by §4.3 to: **that shrinkage is uniform across every
distortion family, not just the ones the lever was built to help.**

**This remains a table and a record, per the lane's scope. Nothing ships.**
`ZensimProfile::D` and `zensim/weights/` are unchanged throughout; the
registry re-pin implied by §4 belongs to a separate, already-notified lane.

---

## 6. Additional reproduction (grid (b) + the corrected per-codec checks)

The grid-(b) fit commands are in §1.3. The per-codec-family analysis (§4) was
one-off scratch, not a repo tool (per the correction: "report values only",
no owner file touched):

```sh
# per arm: dial predictions on BOTH instruments, in score units
bake_dial_refit predict --bake <bake.bin> \
    --corpus /mnt/v/output/zensim/dpeaks372-2026-09-05/instruments/negtail_probe_372_postC_2026-09-05.parquet \
    --score-units --out <predict_negtail.tsv>
bake_dial_refit predict --bake <bake.bin> \
    --corpus <grid-parquet-with-a-dummy-human_score-column-appended> \
    --score-units --out <predict_grid.tsv>
    # the ORIGINAL dial_grid_372col_postC_2026-09-05.parquet has no target
    # column at all (codec/q cells only) -- `predict` requires one, unused
    # by the forward pass; a scratch copy with an all-zero `human_score`
    # column appended is used, the registered instrument is never modified.

# negtail distortion-family recovery: join the probe's ssim2_gpu to
# kadis700k_canonical_gpu_2026-07-01.parquet's score_ssim2_gpu (exact match,
# rounded 1e-6) -> dist_type/dist_name. 0 unmatched, 49/2000 ambiguous
# (first candidate taken).

# grid codec-family truth: join dial_grid_372col_postC's (image_id,codec,q)
# to benchmarks/ssim2_bar_2026-08-31's dialcells_ssim2_qv2grid.tsv on the
# same key (0 unmatched on all 4,424 rows).

# then, per instrument, per family: min(pred), and (negtail only) the
# fraction of ssim2<=-50 rows where pred<=-50 too.
```

All artifacts: `/mnt/v/output/zensim/dpeaks372-2026-09-05/sweep/{bakes,fits,
arms,gaddr,verdicts,work,logs}`. Per-arm gaddr JSON:
`sweep/arms/gaddr_<label>.json`; per-pair dumps: `sweep/work/<label>_<corpus>.tsv`;
predict dumps: `sweep/work/predict_<label>_{negtail,grid}.tsv`;
distortion-family map: `sweep/work/negtail_disttype.tsv`; corrected
per-codec/per-family checks: `sweep/work/percodec_checks_output.txt`,
`sweep/work/percodec_relative_to_D.txt`; bootstrap logs:
`sweep/work/boot_grid{a,b}_<corpus>.txt`, `sweep/work/hfnl_cid22band_grid{a,b}.txt`.
