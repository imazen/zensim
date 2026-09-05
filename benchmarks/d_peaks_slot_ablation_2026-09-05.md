# D-peaks JXL-floor inversion — per-slot attribution and LOO ablation (2026-09-05)

**Lane:** `claude-dpeaks-slots-ablation`, jj sibling workspace
`~/work/zen/zensim--dpeaks-slots` (forgotten + removed on completion).
**Registered by:** [`d_peaks_jxl_floor_2026-09-05.md`](d_peaks_jxl_floor_2026-09-05.md)
(ledger ROUND 93) — *"this lane did not isolate which single peaks feature drives
[the inversion] (out of scope for that lane) … the slice boundary is the natural
first place to look if this is picked up later."* This lane does that isolation,
then tests whether removing the culprit(s) and refitting actually ships.

**Scope discipline, unchanged from every prior lane in this chain:** `zensim/src/profile.rs`
and `zensim/weights/` were not opened for writing. Nothing installs from this record.

**Headline: f162 alone is the driver.** Per-feature decomposition of the raw-prediction
inversion at all 8 failing (ladder × arm) cases shows `f162` is the dominant, and in every
case sufficient, wrong-way contributor — removing it alone (leave-one-out refit at the
best-rank λ=1e-3) reaches `A7r` **1.0000 on jxl** (order_fail 4→0). **But it does not ship**:
the same refit trades the jxl fix for a NEW regression on `A4` (robust floor — dial p5),
an axis shipped D passes. Every other single-peaks-slot LOO (161, 164, 166, 211, 212, 223,
224) leaves the jxl inversion completely unchanged. **Zero of the 9 ablation arms satisfy
the ship rule.**

---

## 0. Method — owner tools only, no re-implemented stats

Binaries built fresh from this workspace (`main@origin` `94914138`, ROUND 93's own commit):
`cargo build --release -p zensim-validate --bins` (85 s via `run-heavy`, rc=0). Reused,
not touched: the postC root (`/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC`),
the postC dial grid (`instruments/dial_grid_372col_postC_2026-09-05.parquet`, 4,424 rows), the
frozen gram (`safesyn.npz`, sha256 `904a8e80…`), the two anchor parquets
(`multiband_anchor_dial100.parquet`, `identity_anchor_sg_n21.parquet`), and the three input
bakes (`lam1em3` sha `4490e64b…`, `Dpeaks`/`lam2em3` sha `85ae9c7c…`, shipped `D` sha
`921a8f67…`).

**Control gate, run before any new fit (per the brief): reproduce `lam1em3` byte-identically.**
The exact argv is recovered from the bake's own embedded `zentrain.repro` (via `zenpredict
inspect`, the canonical CLI at `~/work/zen/zenanalyze/target/release/zenpredict`) and re-run
verbatim (`--lam 1e-3 --slice-file a228.idx --gram safesyn.npz --anchor-parquet
multiband_anchor_dial100.parquet --anchor-parquet identity_anchor_sg_n21.parquet
--anchor-target ssim2_gpu`, then `extend-top --anchor multiband_anchor_dial100.parquet
--target-col target_score`). Result: `act=38` (matches the doc), and after `bake_dial_refit
strip --key zentrain.repro` on both sides, **byte-identical** to the stored
`Dsweep_lam1em3_dial.bin` — sha256 `23f44a953e40fb2c06b7095782ae0e4378dbf99650332cb1059f4fe1744baee5`
on both. This is the "id100+negrich spline chain" every ablation arm below reuses verbatim
(only `--slice-file` changes).

**Attribution is pure linear algebra, not a re-implemented statistic.** `zenpredict inspect
--weights` (the owner CLI) dumps the bake's `scaler_mean`/`scaler_scale`/layer-0 weights;
`zenpredict/src/inference.rs` documents the forward pass as `x' = (x-mean)/scale` then
`pred = bias + Σ_k x'_k · W[k]` for a single dense identity-activation layer (confirmed: all
three bakes here are `n_layers=1`, `activation=identity`). So for two rows on the SAME bake,
`coef_k := W[k]/scale_k` and `raw_pred(hi) − raw_pred(lo) = Σ_k coef_k·(x_k(hi)−x_k(lo))`
exactly — bias and mean cancel. This was cross-checked, not assumed: the decomposition's
`Σ delta_k` reproduces the raw-value deltas already published in ROUND 93's table (e.g.
`2b79a18d1b7537e0`/lam1em3 q0→q8: published raw `0.4375→0.4301` = −0.0074; computed
`Σ delta_k = −0.007454`) on all 8 cases. `x_k` values come straight from the postC dial grid
parquet's own `f0..f371` columns (pyarrow, no re-extraction). Script (one-off scratch, not a
repo tool, same status as ROUND 93's own classification script): `decompose_inversion.py`,
committed alongside this doc's artifacts.

---

## 1. Attribution — f162 is the whole story

### 1.1 Per-arm aggregate (summed across the 4 failing ladders)

Active-peaks-slot rows only (full table in `attrib/decompose_output.txt`, Part A2):

| feat | block | lam1em3 coef | lam1em3 Σdelta | Dpeaks coef | Dpeaks Σdelta |
|---|---|--:|--:|--:|--:|
| **f162** | PEAKS | −0.138726 | **−0.054435** | −0.158280 | **−0.062108** |
| f224 | PEAKS | −0.791449 | −0.003753 | −0.756258 | −0.003586 |
| f166 | PEAKS | −0.251362 | −0.000607 | (inactive) | — |
| f161 | PEAKS | +0.027955 | −0.000004 | (inactive) | — |
| f163 | PEAKS | (inactive) | — | −0.009590 | −0.000369 |
| f164 | PEAKS | −0.103750 | +0.000996 | −0.101990 | +0.000979 |
| f223 | PEAKS | −0.188878 | +0.001380 | (inactive) | — |
| f212 | PEAKS | −0.183275 | +0.001639 | −0.094711 | +0.000847 |
| f211 | PEAKS | −0.250468 | +0.002451 | −0.200130 | +0.001959 |
| largest basic mover | BASIC (f136) | −20.550937 | +0.005469 | (f136) −20.985286 | +0.005585 |
| **total (all active features)** | | | **−0.025419** | | **−0.035975** |

**f162's own contribution (−0.054/−0.062) exceeds the entire net inversion (−0.025/−0.036) by
2×** — every basic slot, and every OTHER peaks slot, nets POSITIVE (pushes the correct
direction); f162 alone overwhelms them. f224 is a distant second, never close to sufficient on
its own (checked directly, §1.3).

### 1.2 Sign-agreement check — f162 is not a sign-mismatch bug

For each active peaks slot, majority direction of travel (q_min→q_max) over **all 33 JXL
ladders** (endpoint difference, no correlation computed):

| feat | majority direction (33 ladders) | coef sign implies | agreement |
|---|---|---|---|
| f161 (lam1em3 only) | DECREASES (26 down / 0 up / 7 flat) | INCREASES (coef +0.028) | **CONTRADICTS** (negligible: Σdelta ≈ −4e-6) |
| f162 | DECREASES (33/33) | DECREASES | AGREES |
| f163 (dpeaks only) | DECREASES (33/33) | DECREASES | AGREES |
| f164 | DECREASES (33/33) | DECREASES | AGREES |
| f166 | DECREASES (33/33) | DECREASES | AGREES |
| f211 | DECREASES (33/33) | DECREASES | AGREES |
| f212 | DECREASES (31/33, 2 flat) | DECREASES | AGREES |
| f223 | DECREASES (33/33) | DECREASES | AGREES |
| f224 | DECREASES (33/33) | DECREASES | AGREES |

**Only f161 shows sign disagreement, and its contribution is negligible** (~4e-6, four orders
below the ~0.03-0.06 magnitude that matters). f162's coefficient sign is globally CORRECT
(agrees with its own 33-ladder trend) — the defect is **local non-monotonicity within the
failing ladders' bottom region**, not a sign error. Directly observed (`attrib/decompose_output.txt`
Part D): on `2b79a18d1b7537e0_818x1022`, f162 runs `q0=0.7433 → q8=0.8126 (UP) → q16=0.7557 →
q24=0.7028 → q32=0.6886` — a local bump at q8 riding on top of an overall-decreasing trend.
The other three failing ladders show the same shape (a local up-tick exactly at the failing
step, net decrease over the full ladder).

### 1.3 Analytic leave-suspect-out prediction (same-bake coefficients, verified before refitting)

Subtracting f162's own delta from each case's total (no refit — just arithmetic on the
existing coefficients) flips the sign to POSITIVE (correctly ordered) on **all 8 of 8** cases;
f224 additionally removed changes the margin but not the verdict:

| ladder | arm | total | minus f162 | minus f162+f224 |
|---|---|--:|--:|--:|
| 2b79a18d1b7537e0 | lam1em3 | −0.007454 | **+0.002162** | +0.003909 |
| 2b79a18d1b7537e0 | dpeaks | −0.009614 | **+0.001358** | +0.003027 |
| 96a0024c685ead3f | lam1em3 | −0.000804 | **+0.005990** | +0.007559 |
| 96a0024c685ead3f | dpeaks | −0.003167 | **+0.004586** | +0.006084 |
| b2e6e2b5969eaf25 | lam1em3 | −0.005044 | **+0.010606** | +0.011044 |
| b2e6e2b5969eaf25 | dpeaks | −0.008283 | **+0.009572** | +0.009991 |
| f65a24b7e176eb47 | lam1em3 | −0.012117 | **+0.010258** | +0.010258 |
| f65a24b7e176eb47 | dpeaks | −0.014911 | **+0.010618** | +0.010618 |

This is a same-bake arithmetic prediction, not a refit result (a refit redistributes ALL
coefficients) — §2-3 test it for real.

---

## 2. Refits — 8 single-slot LOO + 1 all-suspects, at λ=1e-3

Slices: `0..227` minus one of lam1em3's 8 active peaks slots (`{161,162,164,166,211,212,223,224}`)
for the LOO arms, and minus `{161,162,166,224}` (the slots with NEGATIVE aggregate contribution
in §1.1 — the measured suspects, not a guess) for `minus_all_suspects`. The "keep only
sign-agreeing peaks slots" arm from the brief coincides exactly with `minus_f161` (§1.2 found
only f161 disagrees) — reported under both names, one fit.

Every flag byte-identical to the control-verified `lam1em3` command above; only `--slice-file`
changes. All 9 fits + `extend-top` ran clean; **`raw.bin == dial.bin` byte-for-byte on all 9**
(extend-top is a no-op for this lineage, matching ROUND 90/93's own finding):

| arm | active coefs | sha256 (dial) |
|---|--:|---|
| minus_f161 | 37 | `4e7f9c51411d…` |
| minus_f162 | 37 | `fcf4e4d4a090…` |
| minus_f164 | 37 | `12fe27e9d65c…` |
| minus_f166 | 37 | `610043f98efd…` |
| minus_f211 | 37 | `898c330de22a…` |
| minus_f212 | 37 | `0f4fd07ffae2…` |
| minus_f223 | 36 | `1c5dbb0a70fd…` |
| minus_f224 | 37 | `26d97ebc88be…` |
| minus_all_suspects | 38 | `4e3abd7c7cd6…` |

(`minus_all_suspects` keeps 38 active like the parent `lam1em3` — the lasso fills back in with
different coordinates once 4 are excluded from the candidate set; every LOO arm drops from 38
to 36-37, i.e. mostly the removed slot goes to zero with only occasional redistribution.)

---

## 3. Grading — postC instruments, `dialgate_arms.sh score`

### 3.1 G-ADDR summary (all 9 arms + both baselines)

`jxl repr` bar is 0.9697 (the mentor's own fraction); avif/webp bar is 1.0000. Contract is 6/6
for every arm here (no arm ever loses a contract row). `A8r` is not-measured for everyone
(structural, unrelated to this ablation).

| arm | jxl repr | jxl order_fail | avif | webp | contract | **blocking axis** |
|---|--:|--:|--:|--:|:--:|---|
| **D (shipped)** | **1.0000** | 0 | 1.0000 | 1.0000 | 6/6 | **none — SHIPPABLE** |
| lam1em3 (unmodified) | 0.8788 | 4 | 1.0000 | 1.0000 | 6/6 | A7r |
| minus_f161 | 0.8788 | 4 | 1.0000 | 1.0000 | 6/6 | A7r |
| **minus_f162** | **1.0000** | **0** | 1.0000 | 1.0000 | 6/6 | **A4** (new) |
| minus_f164 | 0.8788 | 4 | 1.0000 | 1.0000 | 6/6 | A7r |
| minus_f166 | 0.8788 | 4 | 1.0000 | 1.0000 | 6/6 | A7r |
| minus_f211 | 0.8788 | 4 | 1.0000 | 1.0000 | 6/6 | A7r |
| minus_f212 | 0.8788 | 4 | 1.0000 | 1.0000 | 6/6 | A7r |
| minus_f223 | 0.8788 | 4 | 1.0000 | 1.0000 | 6/6 | A7r |
| minus_f224 | 0.8788 | 4 | 1.0000 | 1.0000 | 6/6 | A7r |
| **minus_all_suspects** | **1.0000** | **0** | 1.0000 | 1.0000 | 6/6 | **A4** (new) |

**A4 = "robust floor — dial p5" (bar ≤ 10.26332105, ssim2's own grid p5).** D=8.772 (pass),
lam1em3=9.058 (pass), **minus_f162=10.445 (fail, +0.18 over bar), minus_all_suspects=10.330
(fail, +0.07 over bar)**. No arm ever fails BOTH A7r and A4 — it is a clean either/or trade:
removing f162 (the only single-slot removal that touches the jxl inversion at all) shallows
the grid's overall 5th-percentile floor past the bar shipped D and the untouched peaks arm both
clear. Every other regression axis (A1,A2,A3,A5,A6,A9) passes for every arm without exception
(only A7r or A4 ever appears in a `fails` list; confirmed by dumping all 11 gaddr JSONs).

### 3.2 Rank — point estimates, all 9 arms (`|SROCC|`, `dialgate_arms.sh score` output)

| arm | CID22 | KonJND | AIC-3 | TID | KADID |
|---|--:|--:|--:|--:|--:|
| D (shipped) | 0.86333 | 0.53670 | 0.77700 | 0.82369 | 0.80806 |
| lam1em3 | 0.87179 | 0.59740 | 0.77722 | 0.79479 | 0.80346 |
| minus_f161 | 0.87202 | 0.59786 | 0.77733 | 0.79476 | 0.80345 |
| minus_f162 | 0.87284 | 0.62349 | 0.78484 | 0.79647 | 0.79865 |
| minus_f164 | 0.86873 | 0.58611 | 0.77578 | 0.79679 | 0.80254 |
| minus_f166 | 0.87195 | 0.59523 | 0.77671 | 0.79454 | 0.80319 |
| minus_f211 | 0.87153 | 0.59800 | 0.77710 | 0.79579 | 0.80366 |
| minus_f212 | 0.87187 | 0.59781 | 0.77733 | 0.79428 | 0.80321 |
| minus_f223 | 0.87205 | 0.59516 | 0.77694 | 0.79770 | 0.80611 |
| minus_f224 | 0.87228 | 0.60119 | 0.77718 | 0.79533 | 0.80354 |
| minus_all_suspects | 0.87311 | 0.62656 | 0.78388 | 0.79678 | 0.79858 |

Every ablation arm beats D on CID22 by point estimate (0.868-0.873 vs 0.863) — consistent with
ROUND 90's finding that the peaks slice carries a real CID22 gain independent of which
particular peaks slot is dropped.

### 3.3 Paired bootstrap vs shipped D (B=2000, seed 20260905, `wave6_paired_bootstrap.py` +
`scripts/ssim2_bar_2026-08-31/paired_perref_boot.py`), for the two candidates that reach
A7r=PASS plus both baselines

| corpus | Lam1em3 − D | Fminus162 − D | Fminusall − D |
|---|---|---|---|
| CID22 | +0.00852 [+0.00544,+0.01146] P=1.000 | +0.00956 [+0.00670,+0.01237] P=1.000 | +0.00984 [+0.00701,+0.01271] P=1.000 |
| KonJND (\|SROCC\|) | +0.06013 [+0.03341,+0.08819] P=1.000 | +0.08637 [+0.06436,+0.11079] P=1.000 | +0.08942 [+0.06721,+0.11390] P=1.000 |
| AIC-3 | +0.00016 [−0.00579,+0.00651] P=0.517 | +0.00774 [+0.00310,+0.01297] P=1.000 | +0.00682 [+0.00237,+0.01187] P=0.999 |
| TID | −0.02878 [−0.03343,−0.02479] P=0.000 | −0.02711 [−0.03147,−0.02314] P=0.000 | −0.02681 [−0.03107,−0.02289] P=0.000 |
| KADID | −0.00462 [−0.00697,−0.00228] P=0.000 | −0.00942 [−0.01171,−0.00725] P=0.000 | −0.00951 [−0.01182,−0.00730] P=0.000 |
| CSIQ | +0.00636 [+0.00421,+0.00862] P=1.000 | −0.00124 [−0.00324,+0.00062] P=0.100 | −0.00166 [−0.00373,+0.00026] P=0.046 |
| LIVE | +0.00252 [−0.00003,+0.00524] P=0.973 | +0.00270 [+0.00018,+0.00553] P=0.983 | +0.00246 [−0.00019,+0.00538] P=0.964 |
| hfnl_cid22band (pooled signed) | 0.4654 vs D's 0.4339 (+0.0315) | 0.4703 vs D (+0.0364) | 0.4714 vs D (+0.0375) |

CID22, KonJND, and hfnl_cid22band gain cleanly for both candidates (CIs exclude zero, P≈1).
TID and KADID lose for every arm including the unmodified `lam1em3`/`Dpeaks` lineage (a
pre-existing trade this LOO work doesn't change). AIC-3 and CSIQ/LIVE are new, smaller
movements specific to which slot is dropped: `minus_f162` gains AIC-3 but loses CSIQ
(borderline, CI touches zero); `minus_all_suspects` is materially the same shape. None of this
changes the ship verdict, which is decided by G-ADDR (§3.1).

---

## 4. Verdict

**Ship rule** (this task's brief): *CID22 ≥ D with CI not excluding a gain, AND A7r PASS on
every codec, AND contract 6/6, AND no other regression axis lost vs shipped D.*

**Zero of the 9 arms pass.**

- **7 arms (`minus_f161`, `minus_f164`, `minus_f166`, `minus_f211`, `minus_f212`,
  `minus_f223`, `minus_f224`) fail on A7r** — the jxl floor inversion is completely unchanged
  (order_fail stays exactly 4, identical to unmodified `lam1em3`). Consistent with §1: none of
  these slots carries meaningful inversion mass: removing any one of them leaves f162 (and its
  large coefficient) untouched in the refit's candidate set, and the lasso keeps it.
- **2 arms (`minus_f162`, `minus_all_suspects`) fix A7r (jxl reaches 1.0000, matching D) but
  fail A4** — a NEW regression, not present in shipped D or in the unmodified peaks arm
  (`lam1em3`: A4=9.058 pass; `minus_f162`: A4=10.445 fail). Removing f162 (and, more so, all
  4 suspects) shallows the model's overall grid p5 past the bar shipped D clears. This is a
  genuine, measured trade — not a bug in the check — and it means the CID22/KonJND gains these
  two arms show (§3.3) do not clear the full ship bar.

**Nothing installed.** `ZensimProfile::D` still resolves to
`d_sdr_add156_id100_negrich_dial_2026-09-05.bin`; `zensim/weights/` was read-only throughout.

**Registered, not attempted:** a joint fix (remove f162 from the slice AND separately address
the A4 floor regression, e.g. via the negative-tail anchor-weight lever from ROUND 90 — though
that lever only touches the spline, and A4 reads off the RAW grid's own p5 after the spline, so
whether `--anchor-weight` on the negative rows could recover A4 without re-breaking A7r is
unmeasured) is the natural next step if this line of work continues. Also registered from
ROUND 93 and still untried: the isotonic/monotone-shape solver extension and the row-level
GRAM up-weighting of the jxl-floor ladders themselves (as opposed to just excluding the slot
that misbehaves on them).

---

## 5. Reproduction

```sh
cd ~/work/zen/zensim
cargo build --release -p zensim-validate --bins

BDR=target/release/bake_dial_refit; BV=target/release/bake_verdict
GRAM=/mnt/v/output/zensim-multicodec-probe/linear-probe/grams/safesyn.npz
MB=/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet
ID=/mnt/v/output/zensim/did100-2026-09-04/work/identity_anchor_sg_n21.parquet
FSID='basic+peaks+masked+iw@w372/v1pre#d16a1091'

# control: reproduce lam1em3 byte-identically (sha 4490e64b… after --embed-repro;
# 23f44a95… after stripping zentrain.repro from both sides)
$BDR fit-lasso --space raw --target human_score --lam 1e-3 --tau 0 --n-sweeps 400 --tol 1e-10 \
    --slice-file /mnt/v/output/zensim/dpeaks372-2026-09-05/slices/a228.idx \
    --gram "$GRAM" --weight 1.0 --anchor-parquet "$MB" --anchor-parquet "$ID" \
    --anchor-target ssim2_gpu --embed-repro --feature-set-id "$FSID" --out ctl_raw.bin
$BDR extend-top --in ctl_raw.bin --anchor "$MB" --target-col target_score --out ctl_dial.bin

# LOO / suspects: same command, --slice-file = a228.idx minus {one slot} or minus {161,162,166,224}
# (slice files built by dropping the target index/indices from `seq 0 227`)
$BDR fit-lasso ... --slice-file minus_f162.idx ... --out minus_f162_raw.bin
$BDR extend-top --in minus_f162_raw.bin --anchor "$MB" --target-col target_score --out minus_f162_dial.bin

# grade
ZL_ERA=postC scripts/dialgate_arms.sh score minus_f162 minus_f162_dial.bin 372

# per-pair dumps + paired bootstrap vs D
ROOT=/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC
$BV --bake minus_f162_dial.bin --regime 372 --features-root "$ROOT" \
    --corpora cid22 --per-pair-output pp_minus_f162_cid22.tsv --per-pair-refs --output /dev/null
python3 scripts/wave6_paired_bootstrap.py --dir <perpair-dir> --corpus cid22 \
    --series Dshipped Lam1em3 Fminus162 Fminusall --ref Dshipped --b 2000 --seed 20260905

# hfnl_cid22band
BAND_LO=0.8 CORPUS=cid22 O=<dir-with-pp_<name>_cid22.tsv> ARMS="Dshipped Lam1em3 Fminus162 Fminusall" \
    python3 benchmarks/ssim2_bar_2026-08-31/paired_perref_boot.py

# attribution decomposition (one-off scratch)
python3 decompose_inversion.py
```

Artifacts: `/mnt/v/output/zensim/dpeaks372-2026-09-05/slots/{attrib,bakes,arms,perpair,work}/`.
