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
