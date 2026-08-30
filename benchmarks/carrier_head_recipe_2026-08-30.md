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
| per-corpus min-max framing | **NOT USED** | absence of the `human_score__mm01` keys the `--target-minmax01` path writes | **exact** |
| solver | `bvls` | 614/954 non-zero (a lasso at any λ>0 on this gram is sparse: the sibling `head954_cid` is 55) + the ledger text | **bit-exact** (below) |
| sign mask | `benchmarks/feature_sign_mask_2026-05-26.tsv` (300 pinned ≥0, 72 free, f372+ absent ⇒ free) | ledger text + bit-exact fit | **bit-exact** |
| `--lam` / `--tau` | 0 / 0 | bit-exact fit | **bit-exact** |
| `--n_sweeps` / `--tol` | 200 / 1e-10 (defaults) | bit-exact fit | **bit-exact** |
| **anchor for the dial spline** | **NOT RECOVERED** — the stored bake has a 19-knot spline; a `safesyn --anchor-stride 37` anchor gives 17 | bake spline payload | **unrecovered, and rank-irrelevant** (the spline is monotone by construction, so SROCC — every number in this lane — is invariant to it) |
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
2. **No per-corpus min-max target framing, and target scale 1.0.** R1b's
   registered arm used `--target-minmax01 --target-scale 100`; its declared
   R-1 variant dropped the min-max but kept scale 100 (rank-invariant for a
   box-constrained least squares, so scale is not a live difference — the
   framing is).

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
  --solver bvls --bounds-tsv benchmarks/feature_sign_mask_2026-05-26.tsv --lam 0 \
  --transforms-tsv <screen reconstructed from the bake> \
  --anchor-parquet fused954/safesyn_954.parquet --anchor-stride 37 \
  --anchor-target human_score --anchor-scale 100 \
  --parity-fit /mnt/v/output/zensim/bakes/wlin-2026-08-29/head954_kon.npz
```
→ `bvls(lam=0) … W=144047 act=614 bias=0.611542` and
**`parity gate 1 PASS: w/bias/mu/sd bit-exact`** (`--parity-fit` compares every
f64 by bit pattern and errors on one mismatch). The emitted bake differs from
the stored one only in the spline section (17 vs 19 knots — the unrecovered
anchor, §1), which no number here reads.

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

## 3. REPRODUCTION (step 2) — *pending*

## 4. THE KEYED SUBSTRATE (step 3) — *pending*

## 5. VERDICT — *pending*
