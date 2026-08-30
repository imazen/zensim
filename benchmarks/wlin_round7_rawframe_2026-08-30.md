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

