# THE CARRIER HEAD — recipe recovery, reproduction, and the keyed-substrate re-run

**Lane:** recover the exact fit recipe behind the campaign's *"the carriers
enable the 944-class linear kon head"* result, commit it as a driver, and
re-run it on R1b's keyed pools-944 substrate — reproduce it or falsify it.

**Why this lane exists.** `benchmarks/r1b_keyed_rebuild_2026-08-30.md` §8.3
reports that a faithful reading of the recipe *as written in the ledger* gives
KonJND −0.1914 for the pools-live arm where the ledger reports +0.4887, and
concludes: *"the missing information is the ledger's unrecorded argv."* The
heads were fit ad hoc, with no driver and without `--embed-repro`, so no
`zentrain.repro` exists on any of them (verified: `zenpredict inspect` shows
only `feature_transforms`, `feature_transform_params`,
`output_calibration_spline` on all four bakes). This document recovers that
argv by **fingerprinting the artifacts**, gates the recovery bit-exactly, and
then runs the recovered recipe on the keyed rows.

**Ownership / bans.** Every number below is produced by `bake_dial_refit`
(`gram` / `fit-lasso` / `gate`) or `bake_verdict` → `zensim_validate::panel`.
No fit math, no statistic, and no loader is written in this lane. CID22 human
MOS is never a training target (it appears only as a validation corpus). No
post-2026-08-05 KADID value is negated. No gate is relaxed.

---

## 1. WHAT WAS RECOVERABLE, AND FROM WHERE

The artifacts of the original run survive in
`/mnt/v/output/zensim/bakes/wlin-2026-08-29/` and the frozen Gram store
`/mnt/v/output/zensim-multicodec-probe/linear-probe/grams/`. Nothing in git
records the commands (`git show c9753eac 0e815e1e 1f099964 3d7e049d` are
ledger-text-only commits; `git log --since=2026-08-29 -- scripts` shows no fit
driver), and no run log survives.

| recipe item | recovered value | source of the recovery | confidence |
|---|---|---|---|
| legs | `safesyn`, `cid22t`, `kadid`, `tid` | the four frozen grams `l954_{safesyn,cid22t,kadid,tid}.npz` (mtime 19:40Z, 2 min before the bake) | **bit-exact** (mu) |
| leg weights | `1.0 / 1.5 / 0.5 / 0.5` | MixGram `mu = Σw·s / Σw·n` reproduces the stored `head954_kon.npz` `mu` with **max abs diff 0.000e+00**; all-1.0 gives 5.05e-2 | **bit-exact** |
| gram artifacts | the frozen `l954_*` / `l944_*` / `l944n_*` npz (Python-built: they carry `Y2_<target>`, which only `scripts/v_next/linear_projections_2026-07-03.py cmd_gram` writes) | key-set + `mu`/`sd` identity | **bit-exact** |
| feature space | `shaped` | gram key prefix `shaped__` | **bit-exact** |
| **transform screen** | the **40-slot** screen the bakes carry: 30 `winsor_p99` + 10 `signed_cbrt`, all at **f0..f155**, rest identity — reconstructed verbatim from `zentrain.feature_transforms` + `zentrain.feature_transform_params` | bake metadata; independently confirmed by rebuilding a gram with it (below) | **exact** |
| target column | `human_score` | gram key `shaped__q_human_score` — there is **no** `__mm01` key | **exact** |
| target scale | **1.0** (not 100) | per-leg `Y1/n`: safesyn 0.595, cid22t 0.706, kadid 0.508, tid 0.497 | **exact** |
| per-corpus min-max framing | **NOT USED by the carrier arms** (`l944n_`/`l954_`) — but **USED by the no-carrier baseline** (`l944_`) | absence vs presence of the min-max target the `--target-minmax01` path writes; matched to 6 dp on all four legs against the canonical parquets (§4.1) | **exact — and it is the confound §4 is about** |
| solver | `bvls` | 614/954 non-zero (a lasso at any λ>0 on this gram is sparse: the sibling `head954_cid` is 55) + the ledger text | **bit-exact** (below) |
| sign mask | `benchmarks/feature_sign_mask_2026-05-26.tsv` (300 pinned ≥0, 72 free, f372+ absent ⇒ free) | ledger text + bit-exact fit | **bit-exact** |
| `--lam` | 0 | bit-exact fit | **bit-exact** |
| **`--tau`** | **0.005** | solved from the PACKED weights: the stored bake holds **530 of 954 exactly zero** against the fit's 614 non-zero, and tau=0.005 is the unique threshold reproducing both that count and the packed mean \|w\| (0.0077320594 vs the stored 0.0077320593); tau=0 gives 340 / 0.0081902 | **exact** (§3) |
| `--n_sweeps` / `--tol` | 200 / 1e-10 (defaults) | bit-exact fit | **bit-exact** |
| **anchor for the dial spline** | **NOT RECOVERED** — the stored bake has a 19-knot spline; a `safesyn --anchor-stride 37` anchor gives 17 | bake spline payload | **unrecovered; rank-irrelevance MEASURED, not assumed** — §3 reproduces every SROCC exactly with the 17-knot spline, and the two bakes differ by exactly the 16 bytes of the two extra knots |
| `--target-clip-min` | not used (grams show `clip>=-inf`, clipped 0) | gram rebuild log | exact |
| exact wall-clock argv string | **NOT RECOVERED** | no log, no `zentrain.repro`, no driver in git | — |

### 1.1 The two recipe items R1b could not have guessed

Both are invisible in the ledger text and both are recovered above:

1. **The shaping screen is NOT `scripts/sota944/screen944_monotone.tsv`.** That
   file is 689 identity / 190 `signed_cbrt` / 65 `log1p` across the full 944.
   The heads' own metadata is 914 identity / 30 `winsor_p99` / 10
   `signed_cbrt`, and **every one of its 40 shaped slots is in f0..f155** — it
   is the trained-bake screen the ledger's W-LIN round 2 describes as
   *"north-anchor's own trained 944 transforms … screen extracted from the
   frozen bake"*, not the monotone screen the SOTA-944 campaign uses. The two
   define different feature spaces, so they define different Grams and
   different BVLS solutions.
2. **The target frame is not the same in the two arms.** The carrier arms use
   the RAW `human_score` at scale 1.0; the no-carrier baseline uses the
   **per-corpus min-max frame**. R1b's registered arm applied
   `--target-minmax01 --target-scale 100` to *both* arms and its R-1 variant
   dropped it from *both* — either way symmetric, and therefore unable to
   reproduce an asymmetric baseline. (Scale is not a live difference: for a
   box-constrained least squares, scaling `y` scales `w` and leaves the active
   set and the rank untouched. The frame is a live difference, and §4 prices
   it.)

### 1.2 Re-anchor — the ledger's numbers are real, and these are the artifacts

Scored with the owner (`bake_dial_refit gate`, whose advisory SROCC is
`zensim_validate::panel::spearman(..).abs()` — the |SROCC| convention KonJND is
read under), each bake on the root it was fit for:

| bake (sha256 prefix) | root | kon-504 | ledger | cid22 | ledger |
|---|---|---|---|---|---|
| `lin944_konmatch` `f0ded234…` | `ext944-canonical-2026-08-01` | **0.1644** | 0.1644 | **0.8249** | 0.8249 |
| `lin944n_konmatch` `70f35ccd…` | `fused944native-2026-08-30` | **0.4570** | 0.457 | **0.8726** | 0.8726 |
| `head954_kon_bake` `9169d425…` | `fused954-2026-08-29` | **0.4887** | 0.4887 | **0.8502** | 0.8502 |

All six match to 4 dp. `lin954_konmatch.bin` and `head954_kon_bake.bin` are the
**same file** (identical sha256 `9169d425…`) — the "matched-pair 954 arm" and
the "standalone 954 head" of the ledger are one artifact under two names.

### 1.3 The bit-exact recovery gate

```
bake_dial_refit fit-lasso \
  --gram l954_safesyn.npz --weight 1.0  --gram l954_cid22t.npz --weight 1.5 \
  --gram l954_kadid.npz   --weight 0.5  --gram l954_tid.npz    --weight 0.5 \
  --space shaped --target human_score \
  --solver bvls --bounds-tsv benchmarks/feature_sign_mask_2026-05-26.tsv \
  --lam 0 --tau 0.005 \
  --transforms-tsv <screen reconstructed from the bake> \
  --anchor-parquet fused954/safesyn_954.parquet --anchor-stride 37 \
  --anchor-target human_score --anchor-scale 100 \
  --parity-fit /mnt/v/output/zensim/bakes/wlin-2026-08-29/head954_kon.npz
```
→ `bvls(lam=0) … W=144047 act=614 bias=0.611542`, `act=424 (pre-pack 614)`
after the tau prune, and **`parity gate 1 PASS: w/bias/mu/sd bit-exact`**
(`--parity-fit` compares every f64 by bit pattern and errors on one mismatch;
it runs BEFORE the tau prune, so it gates the solver, and §3's scored values
gate the pack). The emitted bake differs from the stored one by **16 bytes** —
the spline's two extra knots — with every weight, transform token and transform
param byte-identical.

**Screen cross-check (independent of the fit):** rebuilding the `tid` gram with
the reconstructed screen through the *Rust* owner (`bake_dial_refit gram`)
reproduces the Python `l954_tid.npz` to **rel 3.6e-9 on `S`, 4.2e-9 on `s`,
4.3e-9 on `q`**, `n` bit-exact — i.e. BLAS accumulation order, the difference
the tool's own docstring predicts. A wrong screen would differ by orders of
magnitude, not by 1e-9. This also establishes that **the gram builder is not a
live variable**: the Rust owner and the Python one agree numerically, so new
grams for the keyed substrate can be built with the committed owner.

### 1.4 The driver

`scripts/carrier_head_fit.sh` — owners only (`bake_dial_refit gram` /
`fit-lasso` / `gate`, `bake_verdict`), `--embed-repro` always on, frozen grams
reusable via `CHF_GRAM_DIR`, `--parity-fit` wired through `CHF_PARITY`. What
the original run lacked is now committed.

---

## 2. PRE-REGISTERED — written before any §3/§4 number exists

### 2.1 Reproduction gate (step 2)
Running `scripts/carrier_head_fit.sh` from the **frozen Python grams** on the
**original roots** must reproduce the §1.2 ledger values for all three arms to
**≤ 0.005 absolute** on kon-504 and cid22, or every residual must be explained
by a measured cause. `head954_kon`'s arm is additionally gated **bit-exactly**
by `--parity-fit` (§1.3, already PASS).

### 2.2 The keyed-substrate arms (step 3)
Same driver, same recipe, grams rebuilt with the committed Rust owner in the
**recovered shaped space**, on R1b's keyed roots:

| arm | root | f156..f371 |
|---|---|---|
| `K0-zero` | `ext944-canonical-2026-08-01` (stored) | structural zeros |
| `K2-pools` | `r1b-pools944-2026-08-30` (keyed, one-width, all-live) | all 216 slots live |

Bars read on R1b's keyed slices, signed SROCC from `bake_verdict --full-json`
(`srocc_signed`; `bands[].srocc` is never read), KonJND additionally as
|SROCC|. Round-6 bars, unchanged: `kon ≥ 0.40` AND `hfnl ≥ 0.40` AND
`cid22 ≥ 0.845` AND `nonphoto ≥ 0.865` AND `imazen26 ≥ 0.875`.

### 2.3 Verdict rule (frozen)
- **REPRODUCED** — §2.1 passes **and** `K2-pools` shows a KonJND lift over
  `K0-zero` of the same order as the ledger's `+0.3243` (≥ +0.20) at
  cid22 ≥ 0.845 on the keyed rows.
- **RECIPE-DEPENDENT** — §2.1 passes on the original roots but the keyed
  substrate does not carry the effect. The conditions under which it holds are
  named explicitly, and the ledger's number is then scoped to them.
- **FALSIFIED** — §2.1 fails: the ledger's number cannot be regenerated from
  any recoverable recipe.

No other outcome is promoted after the fact, and no arm is dropped for being
inconvenient. Whatever §3/§4 say is what this document says.

---

## 3. REPRODUCTION (step 2) — GATE PASSED, residual 0.0000

`scripts/carrier_head_fit.sh` on the frozen Python grams, each arm on the root
it was fit for. |SROCC| via `bake_dial_refit gate` (the same instrument §1.2
used on the surviving bakes).

| arm | gram set | root | kon-504 | ledger | cid22 | ledger | residual |
|---|---|---|---|---|---|---|---|
| `repro944zero` | `l944_*` | `ext944-canonical-2026-08-01` | **0.1644** | 0.1644 | **0.8249** | 0.8249 | 0.0000 |
| `repro944native` | `l944n_*` | `fused944native-2026-08-30` | **0.4570** | 0.457 | **0.8726** | 0.8726 | 0.0000 |
| `repro954` | `l954_*` | `fused954-2026-08-29` | **0.4887** | 0.4887 | **0.8502** | 0.8502 | 0.0000 |

**All six values reproduce exactly.** `repro954` additionally passes the
bit-exact `--parity-fit` gate, and its bake differs from the stored original by
**16 bytes** — exactly the two extra spline knots of the unrecovered anchor
(19 vs 17 × 2 floats × 4 B), with every weight, transform token and transform
param byte-identical.

**One residual that mattered, and how it was closed.** The first reproduction
attempt was bit-exact on the FIT yet read kon 0.4739 / cid22 0.8653 — because
`--tau` was still at its default. The stored bake carries **530 of 954 weights
exactly zero against the fit's 614 non-zero**, i.e. 190 were pruned before the
f16 pack. Solving for the threshold gives **`--tau 0.005`** (exactly 530 zeros;
packed mean |w| 0.0077320594 against the stored 0.0077320593). With it the
reproduction is exact. Recorded because it is the kind of parameter that is
invisible in a results table and decisive in a rank.

---

## 4. WHAT THE ARMS ACTUALLY DIFFERED IN — the confound, measured

The reproduction succeeded, so the arms could be compared directly instead of
described. Doing that immediately showed the two arms of the "matched
TRUE-linear pair" were not matched.

### 4.1 The features ARE matched. The TARGET is not.

| leg | `l944_` ybar (no-carrier arm) | `l944n_` / `l954_` ybar (carrier arms) |
|---|---|---|
| safesyn | 0.769937 | 0.595221 |
| cid22t | 0.674312 | 0.706471 |
| kadid | 0.516994 | 0.508158 |
| tid | 0.632768 | 0.497184 |
| **weighted (the fit's bias)** | **0.742083** | **0.611542** |

Row counts are identical on every leg, and the feature side is a textbook
matched pair: **`max |Δmu| = 2.8e-17` outside the pool block** (i.e. exact) and
non-zero only inside it, on all four legs — exactly the "identical everything
except the carrier columns" the registration claims.

The target is a different quantity. Reading it back from the canonical
parquets: for **all four legs**, `l944_`'s ybar equals the mean of
`clip((y − q0.001)/(q0.999 − q0.001), 0, 1)` to six decimals — the
**per-corpus min-max frame** (`--target-minmax01`) — while `l944n_`/`l954_`
carry the **raw** `human_score`.

**So the ledger's pair varied two things at once: the carriers, and the target
frame.** The registration text names only the first ("identical everything
except the 10 carrier columns"); the frame switch is not mentioned anywhere.

### 4.2 The 2×2 — decomposing the +0.3243

Missing cells filled with the same driver. `raw × no-carriers` uses
`--slice-file` on the `l944n_` gram to exclude the ten carrier coordinates
(forcing `w=0` on a coordinate is exactly equivalent to zeroing that column:
its `S[j,k]·w[k]` terms vanish, so no table surgery and no second binary);
`mm01 × carriers` is a fresh `--target-minmax01` gram on `fused944native`.

**kon-504 (|SROCC|), on the original fused tables:**

| | no carriers | + 10 carriers | **carrier Δ** |
|---|---|---|---|
| **min-max frame** | **0.1644** ← *the ledger's baseline* | 0.1894 | **+0.0250** |
| **raw frame** | 0.4403 | **0.4570** ← *the ledger's treatment* | **+0.0167** |
| **frame Δ** | **+0.2759** | +0.2676 | |

**cid22, same cells:**

| | no carriers | + 10 carriers | carrier Δ |
|---|---|---|---|
| **min-max frame** | 0.8249 | 0.8137 | −0.0112 |
| **raw frame** | 0.8726 | 0.8726 | **+0.0000** |
| frame Δ | +0.0477 | +0.0589 | |

The ledger's **+0.3243** decomposes as **+0.2759 target frame (85 %)**,
**+0.0167 carriers (5 %)**, **+0.0317 the 954-append-vs-944-native layout
(10 %)**. Its cid22 **+0.0253** is the frame's +0.0477 less the 954 layout's
−0.0224; **the carrier term on cid22 is 0.0000.**

### 4.3 Why this is coherent with the campaign's own findings

Round 6 of the same campaign measured, and registered, that **"per-corpus
min-max frames poison joint fits — ONE calibrated frame unblocks single-model
hf+kon+broad composition"** (ledger L1862, lane **R4(a) "global target-frame
rebuild for ALL legs"**), after tracing the hf leg's collapse to exactly this
("the hf leg's per-corpus min-maxed target lives in a LOCAL frame … mixed
frames poison the joint fit; **SROCC-invariance hides it in single-leg fits**",
L1543). The 954 heads were then fit on raw-frame grams — an instance of R4(a)
— while the baseline reused the older min-max grams. The measured
`+0.2759` kon is that already-registered frame effect, read against a
pre-fix baseline and attributed to the carriers.

**This is also the exact reason R1b could not reproduce the number.** R1b
applied `--target-minmax01` to *both* arms (its registered arm) and got a
near-null; its R-1 variant dropped it from *both* and got a near-null at a
higher KonJND (0.399/0.410 — the frame effect, showing up right where §4.2
predicts). R1b's diagnosis, "the recipe is the difference, and the missing
datum is the argv", is confirmed; the specific defect is that the ledger's
baseline and treatment were framed differently.

---

## 5. THE KEYED SUBSTRATE (step 3) — the first honest full-bar read

Same driver, same recipe, grams rebuilt with the committed Rust owner at
R1b's keyed one-width all-live root (`r1b-pools944-2026-08-30`,
`folded720append2pools`). One gram set per frame; the three arms differ only in
the admissible coordinate set (§4.2), so they are the same rows, the same
extraction, the same binary and the same standardization.

Signed SROCC from `bake_verdict --full-json` (`srocc_signed`); KonJND also
given as |SROCC| per its convention. n: cid22 4,292 · kon 504 · nonphoto 6,142
· imazen26 6,953 · hfnlproxy 7,717 · kadid 10,125 · tid 3,000.

| arm | frame | f156-371 | cid22 | \|kon\| | nonphoto | imazen26 | hfnl | kadid | tid |
|---|---|---|---|---|---|---|---|---|---|
| `M0zero` | min-max | none | +0.8249 | 0.1644 | +0.8250 | +0.8328 | +0.1459 | +0.8723 | +0.8295 |
| `M1carr` | min-max | 10 carriers | +0.8135 | 0.1890 | +0.8287 | +0.8406 | +0.1212 | +0.8634 | +0.8398 |
| `M2pools` | min-max | all 216 | +0.8156 | 0.1740 | +0.8372 | +0.8475 | +0.1277 | +0.8687 | +0.8425 |
| `K0zero` | raw | none | +0.8726 | 0.4403 | +0.8296 | +0.8470 | +0.2195 | +0.8570 | +0.8209 |
| `K1carr` | raw | 10 carriers | **+0.8769** | 0.4553 | +0.8187 | +0.8374 | +0.1972 | +0.8545 | +0.8198 |
| `K2pools` | raw | all 216 | +0.8440 | **0.4866** | +0.8127 | +0.8386 | +0.1957 | +0.8541 | +0.8262 |
| *B (shipped, 372) — R1b §8.4, see caveat* | — | — | *0.8763* | *0.5183* | *0.9093* | *0.9142* | *0.3553* | — | — |

**Two substrate results worth stating separately from the arm question:**

- **`M0zero` reproduces the ledger's baseline EXACTLY on keyed rows** — cid22
  +0.8249, |kon| 0.1644, identical to 4 dp to the 2026-08-29 fit on the
  2026-08-01 root, from an independent one-width re-extraction.
- **`K0zero` is numerically identical to the `raw × no-carriers` cell of §4.2**
  fit on `fused944native` (|kon| 0.4403, cid22 0.8726, same active count, same
  dial range). Two different roots, one answer. This tightens R1b §8.3b's
  "the rebuild validates itself" from 3e-4 to 0.

**Carrier effect on the keyed rows, at matched frame** (the measurement the
lane exists for):

| frame | carrier Δ\|kon\| (10) | Δ\|kon\| (216) | Δcid22 (10) | Δcid22 (216) |
|---|---|---|---|---|
| raw | **+0.0150** | **+0.0464** | +0.0043 | −0.0286 |
| min-max | **+0.0246** | +0.0096 | −0.0114 | −0.0093 |

Both frames agree: the carriers are worth **one to five hundredths** of kon,
not three tenths. Whole-block-live buys more kon than the ten carriers under
the raw frame (+0.046) and pays −0.029 cid22 for it. On the family axes the
pool block is **negative in the raw frame** (nonphoto −0.011/−0.017,
imazen26 −0.010/−0.008, hfnl −0.022/−0.024) and mildly positive in the min-max
frame — i.e. its sign is frame-dependent, which is itself a reason not to
build on it.

**Round-6 bars: FAIL on all six arms.** The best kon (K2pools 0.4866) clears
`kon ≥ 0.40` but at cid22 0.8440 (bar 0.845), nonphoto 0.8127 (0.865),
imazen26 0.8386 (0.875), hfnl 0.1957 (0.40). No arm passes more than one bar.

**Caveat on the B row:** it is quoted from R1b §8.4, which reads B on the
*same-pair-restricted* subset (rows that also have a full v1-372 vector, ~93.5 %
of each slice) while the arms above are read on the complete keyed slices. It
is a reference, not a same-ruler comparison, and is not used in any Δ here.

---

## 6. VERDICT — **RECIPE-DEPENDENT** (per the §2.3 rule, frozen before §3)

§2.1 passed: every ledger number regenerates exactly from a recorded recipe, so
this is **not FALSIFIED**. §2.2's keyed arms do not show a KonJND lift of the
ledger's order (+0.046 against the required ≥ +0.20) and K2pools misses
cid22 ≥ 0.845, so it is **not REPRODUCED**. The §2.3 rule requires the
conditions to be named:

**The ledger's `+0.3243` is reproducible if and only if the baseline arm carries
the per-corpus min-max target frame and the treatment arm does not.** At matched
frame — either frame — the ten carriers are worth **+0.015 to +0.025 kon** and
**+0.004 / −0.011 cid22**, and the whole 216-slot pool block **+0.010 to
+0.046 kon** at **−0.009 to −0.029 cid22**.

Stated for the record, since the recovery makes it unambiguous: **every number
the ledger reports is real and exactly reproducible; the attribution of them to
the carriers is not.** The claims that rest on that attribution should be read
as follows:

| ledger claim | status |
|---|---|
| "the 10 carriers take kon 0.164 → 0.489 (+0.324)" | **numbers exact, cause wrong** — 85 % is the target frame, 10 % the 954 layout, 5 % the carriers |
| "cid22 0.825 → 0.850 (+0.0253) from the carriers" | **cause wrong** — the carrier term on cid22 is **0.0000** at matched frame |
| "shaping-on-944-alone is falsified (kon 0.16)" | **withdrawn** — 0.16 is the min-max-framed 944 head; the raw-framed one reads **0.4403** |
| "the carriers ARE the linear class's kon backbone" | **not supported** — +0.015…+0.046 |
| "a good linear does NOT require the 372 front" | **still standing, and now on firmer ground** — a raw-framed 944 head with NO carriers reaches kon 0.4403 at cid22 0.8726 |
| "one 954 head equals the whole wlin4 blend on cid22" | unaffected (a property of `head954_kon`, which reproduces exactly) |
| round-6 kon+cid22 bars "PASS" for the 954 candidates | unaffected as a reading of those bakes; but no arm here passes the five-bar set |

**The lane's positive finding.** The largest lever in this head class is not a
feature block — it is the **target frame**: dropping the per-corpus min-max
frame is worth **+0.276 kon, +0.048 cid22, +0.074 hfnl, +0.014 imazen26,
+0.005 nonphoto** (costing −0.015 kadid, −0.009 tid, both train-side guards)
on keyed rows, at zero size and zero inference cost. That is the campaign's own
registered **R4(a)**, and this lane is an independent measurement of its value.
`scripts/carrier_head_fit.sh` carries `CHF_MM01` precisely so the asymmetry can
be reproduced deliberately and never repeated by accident.

**What is NOT claimed.** These are single fits, not a seed band (a convex
sign-constrained fit on a frozen Gram is deterministic, so there is no seed to
band — but there is also no confidence interval here, and the family-axis
deltas of 0.008–0.029 are small enough that they are reported as measured, not
as ordered). The `hdrmix` / `tbig` / `teacher` / `kadis` legs are absent at
this regime (R1b §9), so nothing here speaks to the full-mix `cid` head or the
`wlin954b` blend. The dial-spline anchor of the original heads remains
unrecovered and is measured to be rank-irrelevant, not assumed to be.

---

## 7. ARTIFACTS

`/mnt/v/output/zensim/carrier-recipe-2026-08-30/` with `_MANIFEST.json`
(`build_commit`, the recipe, regime-purity and bans statements, sha256 + bytes
for 25 inputs and 30 outputs). One directory per arm: `repro954`,
`repro944native`, `repro944zero` (§3), `raw_nocarr`, `mm01_carr` (§4.2),
`K0zero`/`K1carr`/`K2pools`, `M0zero`/`M1carr`/`M2pools` (§5, each with its
`.fulleval.json` + `.verdict.md`), plus the two rebuilt gram sets
`grams-pools944{,-mm01}`. Every bake carries `zentrain.repro`.

In git: `scripts/carrier_head_fit.sh`,
`scripts/extract_bake_transform_screen.py`,
`scripts/sota944/screen_carrierhead{944,954}.tsv`.
