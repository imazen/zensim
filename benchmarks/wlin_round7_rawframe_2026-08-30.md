# W-LIN ROUND 7 — the RAW-FRAME full mix on the keyed pools-944 substrate

**Lane:** the linear B-replacement lane, re-run under the *corrected* recipe
(raw target frame, no per-corpus min-max) on R1b's keyed all-944-live
substrate, with the **full** mix — including the bigcodec / hf / teacher legs
that have never existed at this regime — against the **round-6 bars**.

**Why this round exists.** Round 6 ended with a registered falsifier fired for
the single 944 linear, on a diagnosis that has since been shown to be about the
*target frame*, not the features:

- R6 traced the hf leg's collapse to **per-corpus min-max target frames**
  ("mixed frames poison the joint fit; SROCC-invariance hides it in single-leg
  fits") and registered the repair as lane **R4(a)** — one global frame for
  every leg. R6's own arms were never re-run under it.
- `benchmarks/carrier_head_recipe_2026-08-30.md` §6 then **priced** that repair
  independently: dropping the per-corpus min-max frame is worth **+0.276 kon,
  +0.048 cid22, +0.074 hfnl, +0.014 imazen26, +0.005 nonphoto** on keyed rows,
  at zero size and zero inference cost — and 85 % of the ledger's "carriers"
  effect was this, not the carriers.
- `benchmarks/r1b_keyed_rebuild_2026-08-30.md` delivered the substrate on which
  all five bars are readable for a 944-class model **and for shipped B on
  identical pairs**, and left the full-mix legs "keyable, not built".

Round 7 is the composition of those three: **the registered full mix, at the
one frame, on the keyed substrate.** It is the first time the round-6 bars can
be read for a full-mix 944 linear at all.

**Ownership / bans.** Every fit goes through `scripts/carrier_head_fit.sh` →
`bake_dial_refit` (`gram` / `fit-lasso` / `blend-heads` / `gate`); every
statistic through `bake_verdict` → `zensim_validate::panel`. No fit math, no
statistic and no loader is written in this lane. CID22 human MOS is never a
training target (it is a validation corpus only). No post-2026-08-05 KADID
value is negated. No ZNPR v2. No bar is relaxed. No default is flipped — this
lane produces candidates and numbers.

---

## 0. WRITTEN BEFORE ANY FIT

Nothing below §5 exists at the time §1–§4 are written. The extraction (§2) is
not a fit and runs concurrently with this registration; the first `fit-lasso`
invocation of the lane happens after this file is committed.

**Binary provenance, and one hazard already caught.** The lane's binaries are
built in an isolated workspace (`zensim--wlin7`) with its own
`CARGO_TARGET_DIR`, at `main@origin` = `6d0a393ab9640bd6367505e8e03aa23bd747b225`:

| binary | sha256 |
|---|---|
| `bake_dial_refit` | `9844c16ee4b17a74e6b43c116d98b26031f1ca4ede783778d10c92e24c7dfaad` |
| `bake_verdict` | `59ef003cc4fa5678a089036bd611bc62cabf803fb177aeac02a645ab24db0c0c` |
| `examples/v2_ab_extract` | `fc0d780bf8b7739a6d7c5e6a4f35028e9250b69cedaaf52ebed662eb42e1afcd` |

The first two are **byte-identical** to the primary checkout's current
binaries; `v2_ab_extract` is **NOT** (`17c69d86…` there), because the primary
worktree carried another lane's uncommitted edits to `zensim/src/metric.rs` +
`streaming.rs`. Reusing the ambient binary would have extracted this lane's
entire substrate through unreviewed WIP feature code. Recorded because the
"consume binaries, build nothing" convenience rule has exactly this failure
mode when a sibling lane is editing the extractor.

**That lane has since LANDED** (`f9fac41e`, *"fix(v1-372 width): reflect-pad at
EVERY pyramid entry"*), and it changes v1 feature values — i.e. exactly the
f156–371 pool block this round is about. **Round 7 pins the PRE-FIX extractor
on purpose.** Every table it compares against — the whole R1b pools-944
substrate, the `folded720append2` control root, `tbig_944_200k`, the teacher
twins — was extracted before that fix, so extracting this lane's one new leg
*after* it would put a differently-computed pool block into the middle of an
otherwise matched pair and silently break the only comparison the round exists
to make. The right substrate for a post-fix comparison is a post-fix
re-extraction of **everything**, which is a different (and much larger) lane.

**G-X (registered, gates the pin):** before the tbig leg is assembled, a slice
of an EXISTING pools-944 leg is re-extracted with this lane's pinned binary and
must come out **bit-identical** to the stored R1b table. That is what makes
"pinned to the substrate's extractor" a measurement rather than an assumption.
A failure aborts the extraction; it is never downgraded to a tolerance.

---

## 1. THE SUBSTRATE (fixed)

| root | regime | role |
|---|---|---|
| `/mnt/v/zen/zensim-training/r1b-pools944-2026-08-30/` | `folded720append2pools` | 11 keyed local legs + the 3 keyed D1 validate slices (170,007 rows) |
| `/mnt/v/zen/zensim-training/wlin7-pools944-2026-08-30/` | `folded720append2pools` | THIS lane's new legs (§2) |
| `/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/` + `tbig_944_200k.parquet` + `bakes/sota944/teacher*/…_teacher944.parquet` | `folded720append2` | the ZERO-BLOCK control at **matched rows and matched target** |

**Regime purity (absolute).** `folded720append2pools` rows are never
column-mixed with `folded720append2`, `folded720append2carriers`, 924, 720 or
v1 rows. Every new file carries a `regime` column + `zensim_regime` parquet
metadata; the assembler refuses a `*pools` table whose f156-371 block is dead
and a folded table whose block is live (`G-R1`, in the owner).

**The control is a TRUE matched pair, not an approximation.** R1b's G-E proved
`ref_basename` + `human_score` row-for-row equality between the pools legs and
the stored 2026-08-01 legs for 11/11 corpora; §2's `G-K1` proves the same for
the tbig key set against `tbig_944_200k`. So treatment and control differ in
**exactly one thing**: whether f156–371 carry values or zeros.

---

## 2. THE LEGS — what is built, what is grafted, what is absent

The registered W-LIN full mix (ledger `## W-LIN REGISTRATION`, the W10L9
purity-regression legs) is:

```
safesyn 1.0 · cid22_train 1.0 · tbig_200k 0.5 · kadis50k 0.15
tsafesyn 0.5 · ttbig 0.5 · konjnd_bpg 1.2 · tbig_hf 1.0
```

Three structural facts, each verified on disk before this file was written,
collapse most of that into work already done:

1. **`tbig_hf ⊂ tbig`** — all 11,941 `encoded_filename` values of the fused
   `tbig_hf_954` leg are members of `tbig_954`'s 192,714. The HF near-lossless
   leg is the top band (`human_score ≥ 0.90`) of the tbig leg, not a separate
   corpus. Extracting tbig therefore yields tbig **and** tbig_hf.
2. **`tsafesyn` / `ttbig` are TARGET twins, not feature corpora** —
   `tsafesyn_954` is row-for-row `safesyn_954` with `human_score` replaced by
   the teacher prediction (verified: `ref_basename` sequences identical,
   targets differ), and `ttbig_954` is the same for `tbig_954`. That is exactly
   what `scripts/canonical_corpus/build_teacher944.py` documents ("features are
   carried verbatim, so the student sees identical rows under a different
   target"). A teacher target is a **scalar per row and has no regime**, so the
   teacher legs at pools-944 cost **zero extraction**: the pools features carry
   the teacher target from the committed teacher TSV, joined by row identity.
3. Only **tbig** is genuinely missing, at **208,169 rows** (the canonical
   `build_tbig_200k` stride, not the 192,714-row 954 subset).

### 2.1 Built by this lane

| leg | rows | how |
|---|---|---|
| `tbig` | 208,169 | key sidecar (`build_tbig_200k.py --keys-only`, **G-K1 PASS**: `ref_basename` + `encoded_filename` + `human_score` bit-identical row-for-row to `tbig_944_200k`) → `resolve_bigcodec_pair_uris.py --split train` (**100 % of 208,169 rows resolved**; 104,584 object-mode, 103,585 tar-range) → byte fetch → extraction at `foldapp2pools` → `build_tbig_200k.py --from-features` (safe_key_join_arrow on `encoded_filename`, 100 % required, key-table row order, target from the KEY table) |
| `tbig_hf` | (band of the above) | `human_score ≥ 0.90` slice of the tbig pools leg — the same rule the fused `tbig_hf_954` leg satisfies |
| `tsafesyn` | 111,068 | pools `ext_safesyn_full` features + the committed teacher target, row-identity gated |
| `ttbig` | 208,169 | pools tbig features + the teacher target for those rows, key-gated |

### 2.2 ABSENT, with the measured reason (declared, not discovered later)

| leg | weight in the registered mix | why absent |
|---|---|---|
| `kadis50k` | 0.15 | keyable (KADIS rows carry `source_id` + a persisted `distorted_url`) but not built at this regime; a second 50k-row fetch+extract on top of tbig's 208k, for the **smallest** weight in the mix. Cost, not blocker (R1b §9). |
| `konjnd_bpg` | 1.2 | not extracted at `foldapp2pools`. R6-K **measured** that this leg *hurts* its own axis in a linear head (kon 0.154 / 0.143 at w 1.0 / 2.0 vs 0.267 without), so the registered expectation for its absence is neutral-to-positive on kon and unknown on cid22. Stated here so its absence cannot be read post-hoc as either an excuse or a win. |
| `hdrmix` | — | **NOT KEYABLE at this regime**: no SDR-route extraction of `hdr_v3mix` exists (the leg is HDR-route 944-native). Unchanged from the campaign's own check. |

**Consequence, stated up front:** the round-7 mix is the registered mix **minus
two legs**, so it is not a like-for-like re-run of the W10L9 recipe and is not
claimed as one. It IS the largest mix any 944-class linear has been fit on at a
single width with keyed rows.

### 2.3 Gates on every new table (all reported, none may be relaxed)

| gate | statement |
|---|---|
| **G-K1** | key sidecar row-identity vs the canonical leg (`ref_basename`, `encoded_filename`, `human_score` bit patterns). **Already PASS** for tbig. |
| **G-U** | URI resolution 100 % or abort (`safe_key_join_arrow` on `encoded_filename`; a ref-only key is refused). **Already PASS** for tbig. |
| **G-R1** | regime purity: a `*pools` table must have 216/216 f156–371 slots live; a folded table 0/216. In the assembler; aborts. |
| **G-J** | assembly join 100 % resolution, key-table row ORDER preserved, target taken from the KEY table (never from the features table). |
| **G-B** | `check_target_orientation.py` on every target column of every new table. A corpus with no recoverable raw truth reports SKIPPED = "not checked", never "passed". |
| **G-C** | `check_holdout_overlap --threshold 10` for any newly introduced TRAINING rows against the CID22-49 references; d ≤ 10 flags require montage + sign-off before any blocklist action (screening only, per CLAUDE.md). |
| **G-T** | teacher-graft row identity: the twin's `ref_basename` sequence must equal the base leg's, row for row, before a target is grafted. |
| **G-X** | extractor pin (§0): a slice of an existing pools-944 leg, re-extracted with this lane's pinned binary, must be **bit-identical** to the stored R1b table. Aborts on failure. |

---

## 3. THE FITS (frozen)

### 3.1 The invariant recipe

Every head uses the recipe recovered in
`benchmarks/carrier_head_recipe_2026-08-30.md` §1, unchanged:

```
space        shaped, screen = scripts/sota944/screen_carrierhead944.tsv
             (the TRAINED-BAKE screen the heads carry: 914 identity /
              30 winsor_p99 / 10 signed_cbrt, all inside f0..f155)
target       human_score, target-scale 1.0, **RAW — no --target-minmax01**
solver       bvls + benchmarks/feature_sign_mask_2026-05-26.tsv (f372+ free)
lam 0 · n_sweeps 200 · tol 1e-10 · tau 0.005
anchor       the safesyn leg, --anchor-stride 37, --anchor-scale 100
driver       scripts/carrier_head_fit.sh, --embed-repro always on
```

`CHF_MM01` is **NOT** set on any round-7 arm. Its only appearance is the
declared control arm §3.4.

### 3.2 The heads

| head | legs (weights) | purpose |
|---|---|---|
| **K** | safesyn 1.0 · cid22t 1.5 · kadid 0.5 · tid 0.5 | the recovered kon head, verbatim. Re-run under the driver as the round-7 anchor; it must reproduce the recorded `K2pools` cell (cid22 0.8440 / \|kon\| 0.4866) or the discrepancy is reported. |
| **C1** | safesyn 1.0 · cid22t 1.0 · tbig 0.5 · tsafesyn 0.5 · ttbig 0.5 · tbig_hf 1.0 | the registered W10L9 mix minus the two absent legs — the broad/RD head |
| **C2** | C1 + kadid 0.5 + tid 0.5 | W-LIN ARM 2 measured that kadid/tid **help** linear cid22 at 944 (0.835 vs 0.797), reversing a 372-era exclusion. Both variants are fit; neither is dropped. |
| **H** | tbig_hf 1.0 alone, **lasso** at λ ∈ {2e-3, 3e-3, 5e-3} | the R6 discovery (`head_hf0.003`: kon 0.445 / hfnl 0.726 / cid22 0.808), refit at the raw frame. A single-leg fit is SROCC-invariant to the frame, so H alone is **expected to reproduce R6's numbers**; what changes is that its output now lives in the same unit as C and K, which is the round-7 thesis. |

### 3.3 The blend sweep

`bake_dial_refit blend-heads` (the Profile-B mechanism), with the shaped screen
passed through, anchor = the safesyn leg:

- **2-way** `(C, K)` at α ∈ {0.3, 0.4, 0.5, 0.6, 0.7} for the better of C1/C2 on
  cid22, and for the other if they differ by ≤ 0.005;
- **2-way** `(C, H)` at α ∈ {0.5, 0.6, 0.7, 0.8};
- **3-way** `(C, K, H)`: a second `blend-heads` pass over the best 2-way, which
  requires `blend-heads` to emit a fit-npz. If that extension does not land, the
  3-way is reported **NOT RUN** with this reason — never silently replaced by a
  hand-combined weight vector.

### 3.4 The controls (both declared here, both run)

- **ZERO-BLOCK control** — the identical mix and the identical frame on the
  `folded720append2` root (§1 row 3). Isolates the pool block's contribution at
  matched frame, matched rows, matched target, matched screen. This is the only
  arm-pair in the lane that answers "what are the 216 live slots worth to a
  full-mix linear".
- **MIN-MAX control** — `CHF_MM01=1` on the best round-7 arm only. Re-prices
  R4(a) on the full mix (the carrier-recipe lane priced it on the 4-leg head).

### 3.5 Seeds — there are none, and the determinism is gated instead

BVLS and lasso coordinate descent on a **frozen** Gram, from a fixed
initialization, with fixed `n_sweeps`/`tol`, are **deterministic**: the same
inputs give bit-identical `w`/`bias`/`mu`/`sd`. There is no seed to band, and a
"k=2 seed" report would be two copies of one number. Instead:

**G-DET (registered):** one arm is fit twice, in separate processes, and the
two `--emit-fit-npz` outputs must be **bit-identical** on `w`, `bias`, `mu`,
`sd`. A failure means a non-determinism in the fit chain and invalidates every
number in the round until explained.

This is a *weaker* robustness statement than a seed band, and is not presented
as an equivalent one: the round-7 numbers are single fits, and their
uncertainty against the bars is **not** characterized by a spread. Where two
arms differ by less than the axis LSDs already recorded on this instrument
(≈0.004 on the family axes, per campaign appendix O), they are reported as
**not ordered**.

---

## 4. THE BARS, THE INSTRUMENT, AND THE SELECTION RULE

### 4.1 Bars — verbatim from the round-6 registration, unchanged

> **Frozen bars (PASS):** kon ≥ 0.40 AND hfnl ≥ 0.40 AND cid22 ≥ 0.845 AND
> nonphoto ≥ 0.865 AND imazen26 ≥ 0.875.
> **STRETCH:** kon ≥ 0.45 AND hfnl ≥ 0.45.

### 4.2 The instrument, and an honesty note that must not be discovered later

All five axes are read on R1b's **keyed** slices with
`bake_verdict --regime 944 --full-json`, **signed** SROCC (`srocc_signed`);
`bands[].srocc` is never read. KonJND is additionally given as \|SROCC\| per its
convention. n: cid22 4,292 · kon 504 · nonphoto 6,142 · imazen26 6,953 ·
hfnlproxy 7,717.

**The bars were set on a DIFFERENT instrument from the one they are read on
here.** On this keyed instrument, shipped **B itself reads hfnl 0.3553**
(R1b §8.4) — *below* the 0.40 bar it is being used to justify. So `hfnl ≥ 0.40`
on this ruler is an **absolute** bar that the incumbent also misses, not a
"beat B" bar. Every arm is therefore reported against **both**: the five
absolute round-6 bars (the registered gate) **and** a per-axis "beats B on the
same pairs" column (context). The absolute bars remain the PASS criterion —
they are not relaxed, re-based, or re-scaled to B.

B's row on the same pairs (R1b §8.4, same-pair-restricted subset): cid22
0.8763 · \|kon\| 0.5183 · nonphoto 0.9093 · imazen26 0.9142 · hfnl 0.3553.
It is quoted, and its ~6.5 % row restriction (rows with no usable v1-372
vector) is restated wherever it appears.

### 4.3 Selection rule (frozen, applied mechanically)

1. **PRIMARY — bars cleared**, counted out of 5, absolute, on the keyed slices.
2. **TIE-BREAK 1 — maximin margin.** For each axis, margin = (value − bar) /
   bar; the arm's score is the **minimum** margin over the five axes. A
   five-bar gate is won on its weakest axis, so the tie-break is the weakest
   axis, not a sum that lets a strong axis buy a failed one.
3. **TIE-BREAK 2 — cid22** (the sacred human holdout).
4. Size is a **report**, not a selector, unless two arms tie through (3); the
   W-LIN size directive (≤ 12 KB packed) is then a hard filter.

`sdr25`, `product_composite` and M3a are **reported comparators only**, never
the primary — per the campaign's registered selection rule and because the
coherence instrument cannot read a blend of heads as a single ZNPR.

### 4.4 Verdict rule (frozen)

- **PASS** — at least one arm clears all five bars. The arm is named, its
  packed size reported, and it becomes a candidate; **no default is flipped**
  (user-gated).
- **RECIPE-DEPENDENT** — no arm clears five, but an arm clears strictly more
  bars than round 6's best single linear (which cleared **three at most**, and
  only across two different models) *and* the conditions under which it does so
  are named explicitly.
- **FAIL** — no arm clears more bars than round 6's best, on this instrument.
  Then the registered round-6 outcome stands: the pair-of-profiles shape, and
  the 944-class single linear does not replace B.

No other outcome is promoted after the fact. Every arm that is fit is reported,
including the ones that are inconvenient; no arm is dropped for its result.

### 4.5 What this round does NOT claim, whatever §5 says

- Nothing about `hdrmix`, `kadis` or `konjnd_bpg` at this regime — they are absent (§2.2).
- Nothing about seed robustness (§3.5) or about dial / M3a coherence / G-OUT /
  G-GRAN, which are separate batteries this round does not run. A rank result
  is not a ship result: the campaign's own round-6 pair passed its rank lanes
  and **failed the dial battery**.
- Nothing about B's ledger KonJND 0.5935 vs this instrument's 0.5183 — that
  discrepancy is an unadjudicated instrument difference (R1b §8.5c) and is not
  resolved here.
- No cross-document comparison to a pre-R1b published number is treated as
  same-ruler.


---

## 4b. AMENDMENT R7-A1 — the H head's `tau`, declared before the refit

The invariant recipe's `--tau 0.005` (§3.1) is a *pre-pack zero threshold*
recovered from a full-range BVLS head. The **H** head's target is the
near-lossless band `[0.90, 0.984]` — a range of **0.084** — so its coefficients
are ~12× smaller than a full-range head's, and the shared threshold **zeroes
every one of them**.

MEASURED, on the zero-block control, before this amendment was written:

| λ | pre-pack active | post-`tau 0.005` active | result |
|---|---|---|---|
| 2e-3 | 18 | **0** | constant bake, dial range `[68.33, 68.33]`, SROCC **0.0000 on all seven corpora** |
| 3e-3 | 16 | **0** | 〃 |
| 5e-3 | 11 | **0** | 〃 |

That is a degenerate artifact of a mis-scaled threshold, not a result about the
hf head, and `tau 0.005` is therefore **not a valid setting for this head at
all**.

**Amendment, applying to the H head ONLY (every other arm keeps `tau 0.005`):**

1. `--tau 0`.
2. λ is additionally swept **down**, to {1e-4, 2.5e-4, 5e-4, 1e-3}. The L1
   penalty is on the mean-loss scale, so its effective sparsity scales with the
   target range: round 6's operating point (mm01 frame, λ 3e-3, 74 coefficients)
   corresponds to λ ≈ 3e-3 × 0.084 ≈ **2.5e-4** at the raw frame. The registered
   {2e-3, 3e-3, 5e-3} are re-run at `tau 0` as well and reported.

This changes no bar, no selection rule and no other arm. It is recorded here,
before the refit, because a shared constant that silently produces a constant
model is exactly the class of defect this campaign keeps paying for.

---

## 5. THE B RULER — a correction that changes the reference row

R1b §8.4 read B on a **same-pair-restricted** subset (rows that also carry a full
v1-372 vector). The v1-width lane has since landed the fix (`f9fac41e`) and
re-extracted the three slices at full width
(`/mnt/v/output/zensim/v1width-fix-recheck-2026-08-30/`, `build_commit`
`f9fac41e…`), with two gates that make the two cuts directly comparable: the
19,444 previously-full rows are **byte-identical** to the pre-fix extraction, and
for the 453 previously-short imazen26 pairs the fixed v1 `f0..f155` is
**BIT-IDENTICAL to the stored 944 fold** — so the 372 and 944 sides agree on the
shared block for exactly the rows that were missing.

Scoring B (`b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin`, sha
`b6fe5233…`, 372 inputs) on **both** cuts with one binary isolates the effect of
the restriction alone:

| axis | B, restricted cut (n) | B, FULL cut (n) | Δ |
|---|---|---|---|
| cid22 | 0.8764 (4,292) | 0.8764 (4,292) | 0 |
| \|kon-504\| | 0.5186 (504) | 0.5186 (504) | 0 |
| nonphoto | 0.9093 (5,720) | **0.8505** (6,142) | **−0.0588** |
| imazen26 | 0.9144 (6,500) | **0.8609** (6,953) | **−0.0535** |
| hfnlproxy | 0.4213 (7,224) | **0.3496** (7,717) | **−0.0717** |

**The 6.5 % restriction was not neutral — it was size-correlated, and it
inflated B's three family axes by 0.05–0.07.** On the full pair set B clears
**2 of the 5 round-6 bars** (cid22, kon), not 4.

Consequences, applied throughout §6:

- Every arm in this round is read on the **FULL** slices (cid22 4,292 · kon 504 ·
  nonphoto 6,142 · imazen26 6,953 · hfnlproxy 7,717) and B is read on the same
  pairs. The restricted cut appears nowhere in a bars table.
- B's cid22 and kon come from the **STORED** canonical 372 tables
  (`cid22_features_372col_2026-05-15`, `konjnd_features_372col_2026-05-15`), never
  a fresh extraction: a fresh v1-372 extraction is known to drift from the stored
  masked/IW slots on 100 % of rows (under diagnosis in another lane), and the
  ledger's B numbers live on the stored tables.
- `scripts/wlin7_bars.py` therefore **reads** B's row out of a fulleval JSON
  (`--b-fulleval`) instead of carrying it as a constant, so the two rulers cannot
  be mixed by accident. The hardcoded fallback is explicitly labelled
  "RESTRICTED".
- One number is **not reconciled**: R1b §8.4 reports B's restricted hfnl as
  0.3553; the same bake on the same restricted root through this lane's binary
  reads **0.4213**. cid22, kon, nonphoto and imazen26 all reproduce R1b exactly,
  so this is one axis, not a systematic offset. It is recorded, not adjudicated —
  and it does not enter any comparison here, because every hfnl number in §6 is
  on the FULL cut.

B's row used everywhere below, all on the full pair set:
**cid22 0.8764 · |kon| 0.5186 · nonphoto 0.8505 · imazen26 0.8609 · hfnl 0.3496**
— **2/5 bars**, maximin margin **−0.126**.

---

## 6. RESULTS — the zero-block control substrate (`folded720append2`)

The zero-block root carries every leg of the mix already, so it is the substrate
where the round-7 recipe is first readable end to end. Signed SROCC from
`bake_verdict --full-json`; KonJND as \|SROCC\|; **`bands[].srocc` never read**.

### 6.1 The heads

| head | legs (weights) | cid22 | \|kon\| | nonphoto | imazen26 | hfnl | kadid | tid | bytes |
|---|---|---|---|---|---|---|---|---|---|
| **K** | safesyn 1.0 · cid22t 1.5 · kadid 0.5 · tid 0.5 | **0.8726** | **0.4403** | 0.8374 | 0.8501 | 0.2373 | 0.8570 | 0.8209 | 8,988 |
| **C1** | safesyn 1.0 · cid22t 1.0 · tbig 0.5 · tsafesyn 0.5 · ttbig 0.5 · tbig_hf 1.0 | 0.7861 | 0.3773 | **0.9080** | **0.9157** | 0.2882 | 0.6268 | 0.7403 | 9,135 |
| **C2** | C1 + kadid 0.5 + tid 0.5 | 0.8226 | 0.3819 | **0.9053** | **0.9121** | 0.3344 | 0.7814 | 0.8070 | 9,327 |
| **H** | tbig_hf 1.0, lasso λ 2.5e-4, `tau 0` (§4b) | 0.8001 | **0.4836** | 0.8089 | 0.8130 | **0.7272** | 0.6355 | 0.6975 | 8,141 |

**Two exact reproductions anchor the substrate.** `K` reads cid22 **0.8726** /
\|kon\| **0.4403** — identical to `carrier_head_recipe` §4.2's `raw × no-carriers`
cell and to R1b's `K0zero`. And the min-max control's `MM_K` (§6.4) reads cid22
**0.8249** / \|kon\| **0.1644** — the ledger's baseline, to 4 dp. Two independent
cells of a published 2×2 regenerate through this lane's driver.

**The full mix is what moves the family axes.** Against the 4-leg `K` head,
adding the bigcodec + teacher + hf legs is worth **+0.068 nonphoto and +0.062
imazen26** (C2 vs K) — the axes B was thought to own — while costing **−0.050
cid22** and **−0.058 kon**. The two heads are strongly complementary, which is the
blend premise, measured rather than assumed.

**The H head reproduces round 6's discovery at the raw frame.** R6's
`head_hf0.003` (mm01 frame, λ 3e-3, **74** coefficients) read kon 0.445 / hfnl
0.726 / cid22 0.808. The raw-frame refit at the λ the §4b rescaling predicts
(3e-3 × 0.084 ≈ 2.5e-4) lands at **75** coefficients and reads kon **0.4836** /
hfnl **0.7272** / cid22 **0.8001**. The λ-rescaling prediction was made from the
target range before the fit ran and is confirmed to one coefficient.

### 6.2 The blends — and the round-6 falsifier reversed

Round 6's verdict was: *"no single 944 linear reaches kon ≥ 0.40 ∧ hfnl ≥ 0.40
while holding cid22 ≥ 0.845 … composition itself fails (blend cancellation vs an
hfnl-anti generalist head; joint-fit frame incoherence across per-corpus
min-maxed legs)."* At the raw frame, composition does not fail.

| arm | cid22 | \|kon\| | nonphoto | imazen26 | hfnl | bars/5 | maximin | ≥B | bytes |
|---|---|---|---|---|---|---|---|---|---|
| **`T3_KH01_C1_b0.95`** | **0.8559** | **0.5434** | **0.8842** | **0.8891** | **0.4582** | **5** | **+0.013** | **4/5** | **3,190** |
| `T3_KH01_C2_b0.95` | **0.8553** | **0.5378** | **0.8837** | **0.8886** | **0.4575** | **5** | +0.012 | 4/5 | 3,190 |
| `T3_KH01_C2_b0.92` | **0.8539** | **0.5221** | **0.8911** | **0.8963** | **0.4405** | **5** | +0.010 | 4/5 | 3,190 |
| `T3_KH01_C1_b0.9` | **0.8533** | **0.5222** | **0.8955** | **0.9008** | **0.4310** | **5** | +0.010 | 4/5 | 3,190 |
| `T3_KH01_C2_b0.9` | **0.8527** | **0.5128** | **0.8947** | **0.9000** | **0.4304** | **5** | +0.009 | 3/5 | 3,194 |
| `T3_KH015_C2_b0.95` | **0.8642** | **0.5328** | **0.8771** | **0.8836** | **0.4197** | **5** | +0.010 | 4/5 | 3,190 |
| `T3_KH01_C1_b0.85` | **0.8496** | **0.5061** | **0.9017** | **0.9074** | **0.4092** | **5** | +0.005 | 4/5 | 3,190 |
| `B2_C2H_a0.2` (2-way) | 0.8255 | **0.4581** | **0.9100** | **0.9157** | **0.4236** | 4 | −0.023 | 3/5 | 3,175 |
| `B2_KH_a0.1` (2-way) | **0.8554** | **0.5601** | 0.8619 | 0.8667 | **0.4880** | 3 | −0.010 | 3/5 | 3,190 |
| `B2_C2K_a0.6` (2-way) | **0.8451** | **0.4425** | **0.8926** | **0.9016** | 0.2963 | 4 | −0.259 | 2/5 | 3,220 |
| *B (shipped, 372) — FULL pairs* | *0.8764* | *0.5186* | *0.8505* | *0.8609* | *0.3496* | *2* | *−0.126* | — | *7,325* |

**Seven 3-way blends clear all five round-6 bars.** The selection rule (§4.3)
picks `T3_KH01_C1_b0.95` on the maximin margin (**+0.013**, i.e. every axis is
above its bar with margin to spare). At **3,190 bytes** it is **2.3× smaller than
B** and beats B on **four of the five axes**; the one axis B keeps is cid22
(0.8764 vs 0.8559, −0.0205).

The 3-way is two nested `blend-heads` passes (`--emit-fit-npz`, added this round
so pass 2 can compose on pass 1's exact pre-pack weights). The pass-1 spine fixes
K:H at 0.1:0.9; pass 2 mixes that spine against C at β. That is the mechanism
round 6 registered and could not run.

### 6.3 What the blend does that round 6's could not

Round 6 measured that its generalist head was **hfnl-ANTI (−0.016)** so the blend
cancelled the hf head. Here, at matched everything except the frame:

| head | hfnl at RAW frame | hfnl at MIN-MAX frame |
|---|---|---|
| `C1` (the generalist) | **+0.2882** | **−0.0832** |

**The generalist head is hfnl-anti at the min-max frame and hfnl-positive at the
raw frame.** That is round 6's cancellation mechanism, isolated to the one
variable, and it is why the same composition succeeds here.

### 6.4 The MIN-MAX control — R4(a) re-priced on the full mix

The identical mix, screen, solver, λ, `tau` and blend chain, with **only** the
target frame switched (`CHF_MM01=1`; the H head's λ moved 2.5e-4 → 3e-3, the same
operating point under the §4b rescaling, so the two arms sit at matched sparsity):

| arm | frame | cid22 | \|kon\| | nonphoto | imazen26 | hfnl | bars/5 |
|---|---|---|---|---|---|---|---|
| `T3_KH01_C1_b0.95` | **raw** | 0.8559 | **0.5434** | 0.8842 | 0.8891 | **0.4582** | **5** |
| `MM_T3_KH01_C1_b0.95` | min-max | 0.8550 | 0.3896 | 0.8784 | 0.8825 | 0.3529 | 3 |
| **Δ (raw − min-max)** | | **+0.0009** | **+0.1538** | **+0.0058** | **+0.0066** | **+0.1053** | **+2** |

Per-head, same switch: `K` **+0.2759 kon / +0.0477 cid22** (the carrier-recipe
lane's number, reproduced), `C1` **+0.3714 hfnl / +0.0565 kon / +0.0038 cid22**,
`H` **+0.0014 kon / −0.0003 hfnl** (a single-leg fit is SROCC-invariant to the
frame — this near-null is the control on the control).

**So the frame is worth two bars on the full mix**, and its value is concentrated
exactly where round 6 failed: kon **+0.154** and hfnl **+0.105** on the shipped
blend. This is an independent second pricing of the campaign's registered lane
**R4(a)**, now on eight legs instead of four.

### 6.5 Gates

| gate | result |
|---|---|
| **G-X** (extractor pin) | **PASS** — `ext_tid` re-extracted at `foldapp2pools` with the pinned pre-fix binary is **944/944 columns bit-identical** to the stored R1b table |
| **G-DET** (determinism) | **PASS** — the C2 recipe fit in two separate processes gives `w`/`bias`/`mu`/`sd` **bit-identical** |
| recovery-gate regression | **PASS** — the extended driver still reproduces the recovered kon head **bit-exactly** (`--parity-fit`, W=144047, act=614, bias=0.611542) |
| **G-K1** (tbig key identity) | **PASS** — `ref_basename` + `encoded_filename` + `human_score` bit-identical row-for-row to `tbig_944_200k` |
| **G-U** (byte resolution) | **PASS** — 100 % of 208,169 rows resolve (104,584 object, 103,585 tar-range) |
| **G-T** (teacher graft) | **PASS** — 111,068 rows, target mean 0.6142450490816594, source bit-identical to the registered `tsafesyn_954` target |
| **G-R1** (regime purity) | in the assembler; aborts on a dead pool block in a `*pools` table or a live one in a folded table |

---

### 6.6 THE DIAL DEFECT — measured, and AMENDMENT R7-A2 declared against it

The round-6 bars are a **rank** gate. The campaign's own two-panel mandate says a
rank-only verdict is a regression, so the dial panel `bake_verdict` computes on
every invocation was read for every arm. It says something the bars cannot:

| arm | dial dynamic range | p5 | p95 | mono | bars/5 |
|---|---|---|---|---|---|
| B (shipped) | **86.08** | 13.65 | 99.72 | 0.9792 | 2 |
| `K` (4-leg head) | 69.96 | 24.32 | 94.28 | 0.9904 | 2 |
| `C1` / `C2` (full-mix heads) | 66.22 / 65.99 | 26.43 / 25.53 | 92.66 / 91.52 | 0.9909 / 0.9940 | 2 |
| `B2_C2K_a0.6` (no H) | 68.25 | 24.22 | 92.47 | 0.9951 | 4 |
| **`H` (hf-only head)** | **25.94** | **66.63** | 92.57 | 0.9949 | 2 |
| **every 5/5 arm** | **27.3 – 32.0** | **60.4 – 65.0** | 92.2 – 95.1 | 0.9953 – 0.9964 | **5** |

**The hf-only head cannot score below ~66 on the dial, and every arm that
inherits enough of it to clear `hfnl ≥ 0.40` inherits that floor.** The cause is
data coverage, not fitting: `tbig_hf` is the `human_score ≥ 0.90` band, so the
head has never seen a low-quality encode and its output saturates there. A
monotone output spline cannot undo saturation.

This is a **product-relevant failure** — the metric is a dial first — and it is
reported here rather than left for a later battery, because the five-bar gate
would otherwise have shipped past it. Monotonicity itself is *better* than B's on
every 5/5 arm (0.995–0.996 vs 0.979, tied ≤ 0.006), so the defect is specifically
the reachable RANGE, not curve shape.

**AMENDMENT R7-A2 (declared before the arms are fit).** The obvious in-recipe
repair is to give the hf head full-range rows and let the hf band carry extra
weight, instead of restricting it to the band:

> **H′** = `tbig 1.0 · tbig_hf w` at the raw frame, `w ∈ {3, 5, 10}`, both solvers
> (`bvls` with the sign mask, and `lasso` at λ ∈ {2.5e-4, 5e-4} with `tau 0`), then
> the same spine/blend chain as §6.2.

Round 6 tried hf-upweighting (`hf-w 5/10`) and measured **hfnl NEGATIVE, cid22
0.28–0.54** — but that was at the min-max frame, i.e. inside the exact
frame-poison §6.3 isolates, so it does not predict the raw-frame behaviour. The
prediction being tested is that at one frame the upweighted mix keeps the hf
signal *and* the low-quality rows. Bars, selection rule and every other arm are
unchanged. The result is reported whatever it says; if H′ does not clear the
bars, the §6.2 arms stand as they are, with the dial defect stated as their
limitation.

### 6.7 R7-A2 RESULT — the frontier is real, and it is a data-coverage limit

**H′ heads (`tbig 1.0 · tbig_hf w`, raw frame), on the zero-block control:**

| head | cid22 | \|kon\| | nonphoto | imazen26 | hfnl | dial dyn | p5 |
|---|---|---|---|---|---|---|---|
| `Hp_lasso_w10_l2.5e-4` | 0.7019 | 0.1862 | **0.9168** | **0.9215** | **0.4629** | **80.49** | 11.51 |
| `Hp_lasso_w5_l2.5e-4` | 0.6923 | 0.1827 | **0.9161** | **0.9216** | **0.4216** | **80.10** | 10.66 |
| `Hp_lasso_w3_l2.5e-4` | 0.6862 | 0.1839 | **0.9152** | **0.9215** | 0.3932 | **81.27** | 11.36 |
| `Hp_bvls_w10` | 0.7070 | 0.1806 | **0.9110** | **0.9174** | **0.4347** | 20.46 | 71.70 |
| (for reference) `H` hf-only | 0.8001 | **0.4836** | 0.8089 | 0.8130 | **0.7272** | 25.94 | 66.63 |

**The amendment's prediction holds, and round 6's hf-upweighting result does not
transfer to the raw frame.** R6 measured hf-w 5/10 as "hfnl NEGATIVE, cid22
0.28–0.54"; at the raw frame the same shape reads hfnl **+0.39…+0.46** and cid22
0.69–0.71 — still a specialist, but a coherent one, and with a **dial dynamic
range of 80–81, essentially B's 86**. It also posts the round's best family axes
outright (nonphoto 0.9168, imazen26 0.9215).

Note the solver split: the **lasso** H′ heads keep the full dial range; the
**BVLS** ones collapse it (dyn ~20). The sign box makes every admissible
contribution monotone-positive, which saturates the low end. Recorded as
measured; not diagnosed further here.

**The frontier, over 133 fitted cells.** Blending H′ into the K/C chain gives
healthy dials but cannot reach `hfnl ≥ 0.40` (best `P3_KHp4_H_b0.5`: hfnl 0.3987,
i.e. 0.0013 short, dyn 68.3); pulling in enough of the hf-only `H` to clear the
bar pulls the dial back down. The two best 5/5 cells sit at opposite ends of that
trade:

| arm | cid22 | \|kon\| | nonphoto | imazen26 | hfnl | bars | maximin | **dial dyn** | p5 | mono | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `T3_KH01_C1_b0.95` | 0.8559 | 0.5434 | 0.8842 | 0.8891 | 0.4582 | **5** | **+0.013** | 30.32 | 64.77 | 0.9953 | 3,190 |
| `P3_KHp6_H_b0.3` | 0.8580 | 0.4856 | 0.8801 | 0.8885 | 0.4043 | **5** | +0.011 | **37.67** | 56.13 | 0.9953 | 3,190 |
| `P3_KHp4_H_b0.5` (4/5 − hfnl by 0.0013) | 0.8362 | 0.4202 | 0.8963 | 0.9045 | 0.3987 | 3 | −0.010 | **68.28** | 23.44 | 0.9940 | 3,190 |
| *B* | *0.8764* | *0.5186* | *0.8505* | *0.8609* | *0.3496* | *2* | *−0.126* | *86.08* | *13.65* | *0.9792* | *7,325* |

**Applying the registered rule as written** (§4.3: PRIMARY bars, TIE-BREAK 1
maximin) selects **`T3_KH01_C1_b0.95`**. Stated plainly: the two 5/5 arms differ
by **0.002 in relative maximin margin — about 0.0017 absolute on cid22, below the
≈0.004 axis LSD recorded for this instrument** — so the tie-break does not
meaningfully order them, while the **dial panel separates them decisively** in
the other direction. The rule is applied as registered rather than rewritten
after the fact; the dial reading is reported beside it, and it is the reason
neither arm is put forward as a ship candidate.

### 6.8 MULTIPLICITY — stated, not buried

**133 arms were fitted and scored in this round**, and the five-bar gate is read
on a fixed evaluation set that includes **cid22, the sacred human holdout**. The
passing margins are small (maximin **+0.005 to +0.013**, i.e. ~0.004–0.011
absolute on the binding axis). With 133 cells and margins of that size, the
correct reading of "5/5" is **"this recipe family reaches the bars", not "this
particular α/β/λ is worth 0.002 more than that one"** — and the round-7 arms are
single deterministic fits with no confidence interval (§3.5).

The result that does NOT depend on the search is the mechanism (§6.3, §6.4): the
target frame is worth **+0.154 kon / +0.105 hfnl / two bars** on one fixed
composition, and the generalist head flips from hfnl-anti to hfnl-positive with
that single switch. That is a one-variable measurement, not a selection.

### 6.9 The third panel — G-RANGE (`bake_dial_refit gate`, cid22, n=4,292)

| bake | knot domain | raw pred | below/above knot | G-RANGE | Z-RMSE (adv) | OUT-RATIO (adv) |
|---|---|---|---|---|---|---|
| B (shipped) | — | — | 0 / 0 | **PASS** | 0.482 | — |
| `T3_KH01_C1_b0.95` | [−4.442, 0.682] | [−0.02, 0.67] | 0 / 0 | **PASS** | 0.574 | 0.0023 |
| `P3_KHp4_H_b0.5` | — | — | 0 / 0 | **PASS** | 0.550 | — |
| `P3_KHp6_H_b0.3` | [−3.309, 0.614] | [−0.23, 0.61] | 0 / **1 (0.023 %)** | **FAIL** | 0.530 | 0.0009 |
| `Hp_lasso_w10_l2.5e-4` | — | — | 0 / **10 (0.233 %)** | **FAIL** | 0.711 | — |

The arm the registered rule selects passes the hard gate; the dial-preferred
alternative and the standalone H′ head do not. Every arm carries an embedded
`zentrain.repro` (argv + head shas + code commit), verified on the winner.

---

## 7. THE B REFERENCE, ERA BY ERA — and one open ledger item CLOSED

A second substrate update from the drift lane says B's *runtime* behaviour on
current-extractor features differs from every published (stored-era) B number,
and asks for both columns. Producing them turned up two things worth more than
the columns.

### 7.1 The coordinator's slicing shortcut does NOT hold — measured

The suggestion was to read B's 372 inputs out of the pools-944 tables'
`f0..f371` block, "which is bit-identical to v1 at HEAD". Checked against the
post-fix full-width v1-372 extraction of the *same rows in the same order*:

| slice | rows | cols with max\|Δ\| > 1e-6 | max abs | rows with any \|Δ\| > 1e-6 |
|---|---|---|---|---|
| `ext_imazen26` | 6,953 | **372 / 372** | 29.10 | 4,181 (**60.1 %**) |
| `ext_nonphoto` | 6,142 | **372 / 372** | 44.01 | — |
| `ext_hfnlproxy` | 7,717 | **372 / 372** | 3.79 | — |

Row 0 agrees to 8 decimals, so the two are the *same quantity* — but 60 % of
rows differ materially. The explanation is era, not layout: **this lane's pools
tables are the PINNED PRE-FIX extraction** (§0), and the v1-width fix
(`f9fac41e`) changed v1 values well beyond the previously-short rows. Slicing
them would have produced a silently pre-fix "current-era" B. The shortcut is
recorded as **falsified**; B's 372 side is extracted, not sliced.

### 7.2 Both B columns, each axis labelled by the table it came from

| axis | B **stored-era** | B **runtime-era** | table used for runtime |
|---|---|---|---|
| cid22 | **0.8764** | **0.8821** | fresh HEAD extraction of `cid22val_pairs_ab.tsv` (matches the drift lane's 0.88212) |
| \|kon-504\| | 0.5186 | **0.5186** | fresh HEAD extraction of `konjnd_jpeg_val_pairs.tsv` |
| nonphoto | 0.8505 | 0.8505 | post-fix full-width slice (the only full-row 372 table that exists) |
| imazen26 | 0.8609 | 0.8609 | 〃 |
| hfnl | 0.3496 | 0.3496 | 〃 |

**kon-504 has no era difference on this instrument, and that is a measurement,
not an omission:** a fresh HEAD extraction of the 504 keyed pairs is
**BIT-IDENTICAL** (max abs **0.000e+00**, 0 of 372 columns differ) to the table
R1b used. These images are large enough that the width fix does not touch them.
Only cid22 moves (**+0.0057**), and it moves to exactly the drift lane's value.

**So B's bars verdict is unchanged by era: 2 of 5**, on both columns.

### 7.3 CLOSED — R1b §9's "B's ledger KonJND 0.5935 vs R1b's 0.5183, unadjudicated"

The ledger's number comes from a **different 504-row table**:
`/mnt/v/zen/zensim-training/2026-05-15-full-features/konjnd_jpeg504_372_2026-08-29.parquet`.
Scoring B on it reads **exactly −0.5935**.

Aligned against R1b's `ext_konjnd_jpeg_val` (names differ only by a `.png`
suffix):

- **the same 504 sources**, and the targets are **bit-identical** (mean 54.010,
  range [22.46, 69.98]);
- **371 of 372 feature columns differ, on all 504 rows** (max abs 0.120).

So the two are **two different 372 extractions of identical pairs with identical
targets**, and the choice between them is worth **0.075 SROCC on the kon axis for
one fixed bake**. That is the whole of the 0.5935-vs-0.5186 gap. R1b's open item
is closed as an instrument difference — now *localized* to the feature table
rather than merely suspected — and it is a standing caution: **on kon-504, "which
372 extraction" moves the number by more than most model changes in this round.**

Every kon figure in this document is on **R1b's keyed `ext_konjnd_jpeg_val`**,
for every model including B.

### 7.4 The `SRC0437` pair-list defect — footnoted, and its effect measured

The drift lane reports that one kon-504 row, `SRC0437`, is a pair-list defect
(PJND ties at exactly 58.50; the loader rounds to `_059` while the TSV names
`_058` — two different images), and that R1b's keyed 504 inherits it. It is
recorded here rather than left unlabelled, and the honest way to size it is to
recompute the axis without that row.

`SRC0437` is **row 436** of R1b's keyed 504. Recomputing the axis on the
remaining 503 pairs through the **`panel` owner** (`zensim_validate::panel`; the
per-pair `pred`/`jnd` vectors come straight out of `bake_verdict --full-json`,
nothing is re-derived):

| model | \|kon\| all 504 | \|kon\| without `SRC0437` (n=503) | Δ |
|---|---|---|---|
| `T3_KH01_C1_b0.95` | 0.5434 | 0.5434 | 0.0000 |
| `P3_KHp6_H_b0.3` | 0.4856 | 0.4856 | 0.0000 |
| `K_pools` | 0.4866 | 0.4868 | +0.0002 |
| B (both eras) | 0.5186 | 0.5189 | +0.0003 |

**The defect is real and now labelled, and its effect is ≤ 0.0003 — it changes no
bar decision for any model here.** (The `panel` run also reproduces each fulleval's
kon value exactly, which is a free cross-check that the per-pair vectors are the
ones the verdict used.) Every kon number in this document is the **all-504**
value, with this footnote attached.

---

## 8. THE EXTRACTION — cost, and why it ran LOCALLY rather than on the fleet

Measured, not estimated. 208,169 cells (the tbig leg), 2,136 distinct references,
4 codecs.

| phase | wall | throughput | footprint | caps |
|---|---|---|---|---|
| byte fetch (R2) | **72 min** (2,964 s + 1,357 s resume) | **48 cells/s** | 5.8 GB encodes | `run-heavy --mem 16G --jobs 8`, s5cmd/range workers 16–24 |
| decode → PNG | **31 min** | **112 cells/s** (197,301 written + 10,868 already present, 0 errors) | 31 GB | `run-heavy --mem 20G --jobs 14` |
| extract (`foldapp2pools`) | **5 min 39 s** | **614 cells/s** = 1.63 ms/pair wall (20.2 ms/pair compute-only) | 1.45 GB parquet | `run-heavy --mem 20G --jobs 8` |

The fetch resumed cleanly after one transient `Read timeout` on a single tar
member — the owner is idempotent (it skips cached members), so the resume
re-fetched only the outstanding 45,024.

**Why not the fleet — the blocker, stated explicitly.**

1. **The bigcodec corpus is not on the LAN store.** Measured this session:
   `s5cmd --endpoint-url <LAN> ls s3://zentrain/canonical/2026-06-27/` returns
   *"no object found"*, and the LAN `s3://codec-corpus/` holds only
   `imazen-26-variants/` and two `jobsys-demo-*` prefixes. The corpus is on R2.
2. **For zenavif and zenjxl-lossy — 103,585 of the 208,169 cells (49.8 %) — the
   per-file `encodes/` prefix is EMPTY** (R1b §7.3, re-verified here). Those
   bytes exist only as members inside `variants/box-N.tar` (32 tars, 151.9 GiB).
   `declare-scorefiles --full-uri` needs a per-object `dist_path` URI, and **a
   tar-member byte range is not expressible as a pair URI** — so the job system
   cannot address half the cells without a materialization pass first.
3. **That materialization pass is exactly the local fetch.** The fleet path is a
   strict *superset* of what ran: the same 72-minute R2 fetch, **plus** a
   208,169-object upload of 5.8 GB into the LAN store, **plus** declare → workers
   → compact → write-back — in order to move ~32 minutes of local CPU (31 decode
   + 6 extract) onto tower/i134.

With the compute phase projected — and then measured — at well under an hour,
finishing locally was the cheaper path by a wide margin. The brief's
"fleet for the big legs" rule assumes the bytes are fleet-addressable; for this
leg, half of them structurally are not.

**Pipelining and ref-reuse (asked, and already true):** `v2_ab_extract` groups
pairs **by reference** by default (`ZENSIM_AB_GROUPED=1`), so each of the 2,136
references is decoded and prepared **once per group**, not once per pair — the
owner already does this and no change was needed. The distorted side is decoded
once per distinct member (208,169 distinct, `decode_list.tsv`), never twice.
Fetch → decode → extract are separate phases because the decode owner consumes a
completed list; the fetch's own object and tar-range passes are internally
concurrent (16–24 workers).

**Shared-box discipline.** Every phase ran under `~/work/zen/scripts/run-heavy`
with an explicit memory cap. The decode was raised from `--jobs 8` to
`--jobs 14` after **measuring** the box at load average 10.78 on 32 cores with
~2 cores used by other lanes; that raise took the phase from a projected ~130 min
to a measured **31 min** and still left ≥16 cores free for the concurrent KB-43
censuses. Peak RSS stayed ≤ 0.6 GiB for the fetch phases and min-available RAM
never dropped below 31 GiB.

---

## 9. THE KEYED POOLS-944 SUBSTRATE — the registered treatment

The tbig leg landed at `folded720append2pools` (208,169 rows, 1.45 GB), and with
it the teacher twin `ttbig` and the band slice `tbig_hf`. Every gate passed:

| gate | result |
|---|---|
| **G-J** (assembly) | **PASS** — row-aligned, `ref_basename` sequence equal, target max\|Δ\| **0.000e+00** |
| **G-R1** (regime purity) | **OK — 216/216 f156-371 slots live** |
| **G-T** (teacher graft) | **PASS** — 208,169 rows, teacher mean 0.6033651451972182 |
| band slice | 12,743 of 208,169 rows at `human_score ≥ 0.90` — the same count the zero-block control produced, from an independent extraction |

### 9.1 The pool block's contribution, at matched arm, matched frame, matched rows

Every row below is the SAME recipe fit on the two substrates, which differ in
exactly one thing: whether f156–371 carry values or zeros.

| arm | Δcid22 | Δ\|kon\| | Δnonphoto | Δimazen26 | Δhfnl | **Δ dial dyn** |
|---|---|---|---|---|---|---|
| `T3_KH01_C1_b0.95` | −0.0013 | +0.0030 | −0.0088 | −0.0077 | +0.0074 | **+14.7** |
| `T3_KH01_C1_b0.9` | −0.0009 | +0.0119 | −0.0037 | −0.0033 | +0.0054 | **+19.4** |
| `T3_KH01_C1_b0.85` | −0.0004 | +0.0135 | −0.0008 | −0.0008 | +0.0033 | **+21.7** |
| `P3_KHp6_H_b0.3` | −0.0017 | +0.0059 | +0.0008 | +0.0026 | +0.0119 | **+21.7** |
| `C1` (head) | +0.0239 | +0.0210 | +0.0083 | +0.0033 | +0.0984 | −2.4 |
| `Hp` (head) | +0.0326 | +0.0421 | +0.0122 | +0.0112 | +0.0184 | −52.1 |
| `K` (head) | −0.0286 | +0.0464 | −0.0247 | −0.0115 | −0.0417 | −3.2 |

**On rank the pool block is worth hundredths** — +0.003…+0.014 kon and
−0.009…+0.012 elsewhere on the blends — which agrees with the carrier-recipe
lane's independent finding (+0.010…+0.046 kon, −0.009…−0.029 cid22) and is
nothing like the ledger's +0.32.

**The pool block's real contribution here is to the DIAL: +14.7 to +21.7 points
of dynamic range on every 3-way blend**, which is the round's actual defect
(§6.6). That is a new result — the carrier lane read rank only — and it is the
first thing measured in this campaign that the live pool block clearly buys.

### 9.2 All 5/5 arms, both substrates, ranked by the registered rule

17 of the 133 fitted arms clear all five bars. Ranked by PRIMARY (bars) then
TIE-BREAK 1 (maximin margin), exactly as registered:

| rank | arm | substrate | maximin | cid22 | \|kon\| | nonphoto | imazen26 | hfnl | dial dyn | G-RANGE | bytes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **`PL_P3_KHp6_H_b0.3`** | **pools** | **+0.01327** | 0.8562 | 0.4915 | 0.8809 | 0.8911 | 0.4162 | **59.37** | **FAIL** (1 row, 0.023 %) | 3,589 |
| 2 | `T3_KH01_C1_b0.95` | zero | +0.01293 | 0.8559 | 0.5434 | 0.8842 | 0.8891 | 0.4582 | 30.32 | PASS | 3,190 |
| 3 | `T3_KH01_C2_b0.95` | zero | +0.01220 | 0.8553 | 0.5378 | 0.8837 | 0.8886 | 0.4575 | 30.12 | — | 3,185 |
| 4 | `P3_KHp6_H_b0.3` | zero | +0.01076 | 0.8580 | 0.4856 | 0.8801 | 0.8885 | 0.4043 | 37.67 | FAIL | 3,241 |
| 5 | `T3_KH01_C2_b0.92` | zero | +0.01050 | 0.8539 | 0.5221 | 0.8911 | 0.8963 | 0.4405 | 29.89 | — | 3,184 |
| 6 | `PL_T3_KH01_C1_b0.92` | pools | +0.00999 | 0.8534 | 0.5400 | 0.8864 | 0.8921 | 0.4475 | 46.60 | **PASS** | 3,436 |
| … | (11 more, maximin +0.0049 … +0.0098) | | | | | | | | 27.3–53.7 | | 3,182–3,453 |
| 16 | `PL_T3_KH01_C1_b0.85` | pools | +0.00492 | 0.8492 | 0.5197 | 0.9009 | 0.9066 | 0.4125 | **53.67** | **PASS** | 3,453 |
| — | *B (runtime-era, same pairs)* | — | *−0.126* | *0.8821* | *0.5186* | *0.8505* | *0.8609* | *0.3496* | *86.08* | *PASS* | *7,325* |

**The registered rule selects `PL_P3_KHp6_H_b0.3`** — on the keyed pools
substrate, 3,589 bytes, all five bars with the largest weakest-axis margin, and
the best dial of any 5/5 arm. It **fails G-RANGE by one row of 4,292 (0.023 %
against a 0.010 % gate)**, which is not in the selection rule and is reported
beside it, not used to re-rank after the fact.

**The best arm that clears all five bars AND passes G-RANGE AND has the healthiest
dial is `PL_T3_KH01_C1_b0.85`** (maximin +0.0049, dyn 53.67, 3,453 B): cid22
0.8492 · \|kon\| 0.5197 · nonphoto 0.9009 · imazen26 0.9066 · hfnl 0.4125,
per-codec dial monotonicity 0.987–0.999 with **zero** tied rate. Against B on the
same pairs it wins **nonphoto +0.050, imazen26 +0.046, hfnl +0.063, kon +0.001**
and loses **cid22 −0.033**, at **47 % of B's size**.

Both are named because they are what the two panels respectively prefer; neither
is proposed as a default.

---

## 10. VERDICT — **PASS** (per the §4.4 rule, frozen before any fit)

At least one arm clears all five round-6 bars, so the registered verdict is
**PASS**. 17 arms do, across both substrates. **The round-6 falsifier — "no
single 944 linear reaches kon ≥ 0.40 ∧ hfnl ≥ 0.40 while holding cid22 ≥ 0.845"
— is reversed**, and the reversal is attributable to one variable.

**The candidate the rule names:** `PL_P3_KHp6_H_b0.3`, **3,589 bytes**, on the
keyed pools-944 substrate — cid22 0.8562 · |kon| 0.4915 · nonphoto 0.8809 ·
imazen26 0.8911 · hfnl 0.4162. **No default is flipped; that is user-gated.**

**What actually did it — and it is not the features.** The single-variable
min-max control (§6.4) prices the target frame at **+0.154 kon, +0.105 hfnl,
+0.006 nonphoto, +0.007 imazen26, +0.001 cid22 — two bars** on one fixed
composition, and shows the generalist head flipping from **hfnl −0.0832** to
**hfnl +0.2882** with that switch alone. Round 6 diagnosed its own failure as
"blend cancellation vs an hfnl-anti generalist head"; the frame is *why* the head
was hfnl-anti. This is the campaign's own registered lane **R4(a)**, priced a
second time and independently, on eight legs instead of four.

**What the pool block is worth (the registered treatment-vs-control question).**
At matched arm, frame, rows and target, the 216 live slots move rank by
**hundredths** (+0.003…+0.014 kon; −0.009…+0.012 elsewhere) — consistent with the
carrier-recipe lane and nothing like the ledger's +0.32 — but they add
**+14.7 to +21.7 points of dial dynamic range** to every 3-way blend. On the axis
this round found to be the binding product defect, the pool block is the largest
single lever measured.

**What is NOT claimed.**

- **Not a ship candidate.** Every 5/5 arm has a compressed dial (dyn 27–59 against
  B's 86; §6.6), because the hf leg is a `human_score ≥ 0.90` band and its head
  cannot score a low-quality encode. That is a **data-coverage** limit, not a
  fitting one, and no monotone spline can repair it. The rule-selected arm also
  fails **G-RANGE** by one row.
- **Not a seed band.** These are single deterministic fits (**G-DET PASS**:
  bit-identical `w`/`bias`/`mu`/`sd` across two processes). There is no confidence
  interval here, and 133 arms were fitted against a fixed gate that includes
  cid22 at margins of +0.005…+0.013 (§6.8). Read the result as *"this recipe
  family reaches the bars"*, not as an ordering of α/β/λ.
- **Not the registered mix.** `kadis50k`, `konjnd_bpg` and `hdrmix` are absent
  with measured reasons (§2.2).
- **Not the full battery.** M3a coherence, G-OUT v2, G-GRAN v1 and the corruption
  panel were not run this round.
- **Not a cross-era comparison.** Arms are on the pinned pre-fix extractor; B's
  372 side is stated per axis and per era (§7.2).

**The next lever, named from the measurement rather than guessed:** an hf head
trained on the FULL quality range with the hf band upweighted keeps the dial
(`Hp_lasso_w10`: dyn 80–81, essentially B's 86, and the round's best family axes
at nonphoto 0.9290 / imazen26 0.9327 on the pools substrate) but tops out at
hfnl ≈ 0.46 and cannot carry a blend past the bar on its own. Closing the last
0.04 of hfnl **without** the band-only head is the whole remaining distance
between this round's result and a ship candidate.

---

## 11. ARTIFACTS

- **Doc:** this file. **Driver:** `scripts/carrier_head_fit.sh` (extended, default
  recipe byte-unchanged and re-gated bit-exactly). **Reporting view:**
  `scripts/wlin7_bars.py` (reads fulleval JSONs; computes no statistic).
- **Owners extended:** `build_tbig_200k.py` (`--emit-keys` / `--keys-only` /
  `--from-features` / `--band-from`), `build_teacher944.py` (`--graft-from`),
  `bake_dial_refit blend-heads --emit-fit-npz`.
- **Tables:** `/mnt/v/zen/zensim-training/wlin7-{pools944,ctrl944,b372full,bruntime372}-2026-08-30/`,
  each with `_MANIFEST.json` (`build_commit`, regime, per-file sha256 + rows) and
  per-file manifests; registered in `~/work/zen/DATA_PROVENANCE.md`.
- **Arms:** `/mnt/v/output/zensim/wlin7-2026-08-30/arms/` — 133 bakes, each with
  `.fulleval.json` + `.verdict.md` and an embedded `zentrain.repro`.
- **Pinned extractor:** `/mnt/v/output/zensim/wlin7-2026-08-30/bin/v2_ab_extract_PREFIX_PINNED`
  (sha256 `fc0d780b…`), the G-X-gated pre-fix build every pools table was made with.
- **Machine-readable bars table:** `benchmarks/wlin7_bars_all_2026-08-30.tsv` —
  all 133 arms plus the two B rows, one line each: the five signed axis values
  (KonJND as |SROCC|), bars cleared, maximin margin, axes ≥ B, and the source
  fulleval path. Produced by `scripts/wlin7_bars.py --tsv`, which reads
  `rank.<corpus>.srocc_signed` and computes nothing.
