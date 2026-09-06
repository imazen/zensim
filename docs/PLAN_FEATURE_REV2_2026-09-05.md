# FEATURE REVISION 2 — the plan and its pre-registered gates

**User directive (verbatim, 2026-09-05):** *"we should fix arithmetic defects
aggressively before shipping, and perhaps change feature definitions and
formulas to make monotone linear models better. we have a fleet to
recalculate"*.

**This lane executes phases 2b and 3 of**
[`PLAN_FEATURE_SYSTEM_2026-09-05.md`](PLAN_FEATURE_SYSTEM_2026-09-05.md), through
the definition registry's revision mechanism (`zensim::feature_defs`), not
beside it. The architecture lane owns `zensim/src/features/` and the research
engine; **this lane owns the kernels' arithmetic and the revision entries.**

Source of truth for what is wrong:
[`FEATURE_DEFECTS_AUDIT_2026-09-05.md`](FEATURE_DEFECTS_AUDIT_2026-09-05.md).

Gates below are **pre-registered**: written before the code, never edited to
match a result. A gate that fails is reported failed, not re-scoped. The
standing gates of the feature-system plan (G-API, G-TEST, G-OWNER, G-APPEND,
G-SERVE) apply unchanged. **G-BYTE is deliberately scoped**: it holds
absolutely for revision 1, and revision 2 is the batched era break the user
authorized — see R1.

---

## 0. What "revision 2" is

A named set of registered `Revision`s flipped from `Proposed` to `Landed`
together, so that exactly one era boundary exists rather than one per fix.
Every entry keeps its already-registered era token (G-APPEND: tokens are
append-only, and `v1ssimcap` / `freecomp` were registered by the architecture
lane at `38e7a586`).

| defect | era token | status now | what rev2 does |
|---|---|---|---|
| **F4** unbounded SSIM per-pixel `d` | `v1ssimcap` | Proposed | bound the luminance term |
| **F5** free-40 raw-moment route parity | `freecomp` | Proposed | stabilise the second moment |
| **F12** `mu` scalar-tail vs vector-body ulp split | *(new, see §3)* | unregistered | one summation order |
| anything §4's sweep finds | *(new tokens)* | — | decided by measurement |

**Revision 1 remains reproducible.** The selector is a first-class part of the
registry, not a build flag, because the research engine (phase 2, another lane)
must compute both revisions on the same pixels.

---

## 1. F4 — the one live arithmetic defect

### 1.1 What is wrong

`zensim/src/fused.rs` and `zensim/src/simd_ops.rs` compute, per pixel, in f32:

```
num_m   = 1 - (mu1 - mu2)^2          // luminance, NO C1, UNBOUNDED BELOW
num_s   = 2*cov + C2                 // C2 = 0.0009
denom_s = var1 + var2 + C2
d       = (1 - num_m * num_s / denom_s).max(0)      // no upper cap
```

`num_s/denom_s` is bounded in `[-1, 1]` by construction. `num_m` is not: it is
`1 - D^2` for a mean difference `D`, so a large local mean difference makes it a
large NEGATIVE number and `d` a large POSITIVE one. MEASURED
(`benchmarks/ssim_moment_explosion_2026-07-16.md`, full scan of 2,322,579 rows):
`f313 = iw_ssim_4th s0 ch2` reaches **5,814,302**; `f241` 5,797,029; against a
photographic p99.9 of **0.48**. The mechanism reproduces analytically:
`d ~ D^2`, and `D ~ 2400` gives 5.76e6.

72 slots carry the defect (three `ssim_*` signals x masked + IW blocks);
**144 features hold a clamped-outlier value on the weakest content class**,
because the shipped `winsor_p99` bake transform clamps the symptom.

### 1.2 The fix, DERIVED not guessed

**The correct bounded form already exists in this repo as a named shared
primitive.** `zensim/src/feature_v2.rs:720`:

```rust
fn bounded_sim(a: f64, b: f64, c: f64) -> f64 { (2.0*a*b + c) / (a*a + b*b + c) }
```

documented "Bounded `(0, 1]`" and used at four v2 call sites (`C_EDGE`,
`C_GMS`, `C_CONTRAST`). It **is** the standard SSIM luminance term. The v1
SSIM kernel is the single place in the crate that hand-rolls an unbounded
`1 - D^2` instead of calling it, so F4 is *also* a G-OWNER duplication defect.

**Revision 2 sets `num_m = bounded_sim(mu1, mu2, C1)`.**

**The constant is derived twice and agrees:**

1. *From the family.* Every `bounded_sim` regularizer in the crate is
   `1e-4` (`C_EDGE`, `C_GMS`, `C_CONTRAST`, `C_BV`).
2. *From the SSIM constant already present.* `C2 = 0.0009 = (0.03)^2 = (K2*L)^2`
   with `K2 = 0.03, L = 1` — the textbook value. The matching `C1 = (K1*L)^2`
   with the textbook `K1 = 0.01` is **`1e-4`**.

So **`C_SSIM_LUMA = 1e-4`**, and it is named once, beside `C2`, in
`zensim/src/fused.rs`.

*Correction to the record:* `ssim_moment_explosion_2026-07-16.md` §7a evaluates
"C1(0.01)", which is 100x the derived value. Its conclusions are unaffected
(`C1` only matters as `mu1^2+mu2^2 -> 0`), but its photo row moves from +0.0003
to +0.00004 at the derived constant. To be re-derived and corrected in place.

### 1.3 Consequences that follow structurally, and are asserted

With `num_m` in `(0, 1]` and `num_s/denom_s` in `[-1, 1]`:

* `d = 1 - num_m*(num_s/denom_s)` lies in **`[0, 2]`**, so the `.max(0)` floor
  becomes provably redundant and the whole `ssim_*` family is bounded.
* **Severity ordering is preserved** where the old form saturated: §7a's own
  numbers, `0.016 -> 0.529 -> 0.959`, stay strictly ordered where a cap would
  map all three to the cap.

### 1.4 The open question this lane must MEASURE, not assume

`zensim/src/lib.rs:204` states the no-`C1` form is deliberate: *"(no C1), uses
`1 - (mu1-mu2)^2` directly. Correct for perceptually-uniform spaces."*
`bounded_sim` re-introduces a mean-dependent (Weber) normalisation, which is
exactly what that comment says was removed on purpose. This is a genuine design
tension and it is settled by R6, not by argument. Two alternatives are carried
into the probe so the decision is between measured options:

| arm | `num_m` | bounded | Weber | first-order agreement with rev1 |
|---|---|---|---|---|
| `rev1` | `1 - D^2` | **no** | no | — |
| `A: ssim-luma` | `(2*mu1*mu2 + C1)/(mu1^2+mu2^2+C1)` | (0,1] | **yes** | no (slope `1/(mu1^2+mu2^2)`) |
| `B: lorentz` | `1/(1 + D^2)` = `1 - saturate(D^2, 1)` | (0,1] | no | **yes**, to `O(D^4)` |
| `C: clamp` | `(1 - D^2).max(0)` | [0,1] | no | **exact** for `D^2 <= 1` |

Arm B is the family's own `saturate(x, c) = x/(x+c)` idiom at `c = 1`, which is
the unique scale making it agree with rev1 to first order — so it introduces no
new constant and keeps the perceptual-uniformity intent. Arm C is exact on the
overwhelming majority of pixels and flattens rank above `D = 1`.

**Prior on the answer, stated in advance so it cannot be retrofitted:** arm A is
the principled SSIM form and §7a preferred it; arm B is the one that honours
lib.rs:204. If the probe cannot separate them, **arm A ships**, because it
reuses an existing owner and an existing constant scheme rather than adding a
form.

---

## 2. F5 — free-40 route parity (phase 2b, and its window is closing)

`global_stats_from_raw_moments` (`feature_v2.rs:4721`) computes
`var = sum_s2/n - (sum_s/n)^2`, a catastrophic-cancellation form, from moments
the two routes accumulate at **different granularities**: the append kernel
reduces its f32 lane accumulator to f64 **per ROW** (`feature_v2.rs:4307`) and
its scalar tail goes straight to f64 **per pixel** (`:4382`); the free route
accumulates in f32 lanes across a whole **BAND** and reduces once
(`fused.rs:896`). MEASURED: 2,607 of 28,601 cells (**9.12 %**) over the 2e-5
bar, worst 3.63e-3, worst relative ~55x; class-C 0/18,552; basic+peaks
bit-identical.

**This is a train/serve skew** — it is the reason G1.8 is PARTIAL — and phase
2b's own gate says land it while no shipped bake reads the tranche.

### 2.1 The measurement that chooses the fix (R4)

There are two sub-defects and they have different prices:

* **(a) granularity** — the two routes disagree because they round differently.
* **(b) conditioning** — `sum_s2/n - mean^2` amplifies (a) by `mean^2/var`.

Fixing **(a) only** (free route reduces per row, scalar tail direct to f64, to
match the append kernel) moves **zero shipped bytes**: the append route is
untouched, so no 944 table changes. Fixing **(b)** requires changing what is
accumulated (shifted / compensated second moment) in **both** routes, which
moves the append kernel's `GLOBAL_*` values and is therefore an era break for
every 944 table.

**R4 decides between them by measuring the append route against a two-pass f64
reference variance on real pixels.** If the append route is itself materially
wrong, (b) is required and rev2 takes it (the era break is authorized and the
tables are being recalculated anyway). If the append route is accurate and only
the free route is bad, rev2 takes (a) and F5 costs nothing.

**G2b.2 is re-checked at landing time**, not assumed: `bake_block_profile` over
`zensim/weights/*.bin` must show no shipped bake reading the raw-moment tranche.
If one does, the phase stops and re-prices.

---

## 3. F12 and the sweep for the same patterns

**F12** (`fused_blur_h_mu_inner_*`): the scalar tail computes `sum + (add - rem)`
and the vector body `(sum + add) - rem`, measured 2528.7349 vs 2528.7344. The
production band shape hits it. `fused_blur_h_ssim` already fixed the same wart
with a masked vector tail. Rev2 unifies it to ONE order, which moves v1 bytes on
the last `height % 8` rows — free inside an era break, forbidden outside one.

**§4's sweep** covers every other kernel for the same four patterns:
uncapped ratios, missing stabilisers, cancellation-prone differences, and
order-dependent reductions that are not pinned. Anything found is fixed with a
failing-first test and a registered revision, or explicitly recorded as
BEHAVIOUR with a named reason (F13's `dense_block_kernel` is already such a
record and is **out of scope** — the audit prices its restructure at a 1.17x@8T
upper bound against re-training every 944 model, and the user has not asked).

---

## 4. Pre-registered gates

| # | gate | pass criterion |
|---|---|---|
| **R1** | **rev1 byte-identity control** | With rev1 selected (the default until the flip), `v1_golden_bytes`, `fold_engine_parity`, `v1_feature_width_pure_function` and `feature_invariants` pass unmodified, and a full `to_bits()` dump over the 20-geometry set is **byte-identical** to the pre-change dump. Proves the refactor is inert. |
| **R2** | **F4 boundedness** | At rev2, per-pixel `d` is in `[0, 2]` on every probe input, and the pooled `ssim_*` max over the ladder instrument + a real training slice drops from 5.8e6 to `<= 2`. Asserted structurally (the `.max(0)` floor is proved redundant), not just observed. |
| **R3** | **F5 route parity** | free vs append within **2e-5 on 100 %** of cells of the audit's 773-pair real-pixel population. The synthetic-only gate that MISSED it is replaced, not re-run. Report the max, not just the pass. |
| **R4** | **F5 accuracy** | Append-route `gvar` vs a two-pass f64 reference on the same pixels: report max relative error. Decides §2.1 (a)-vs-(b) **before** the fix is written. |
| **R5** | **rev2 keeps every invariant** | At rev2: deterministic and thread-invariant (rayon pools 1/2/3/8/16/28), engine-bit-exact (buffered / fold v1-only / 944 fold / both product entries), tier-parity within HEAD's policy `max(1e-6 abs, 1e-5*scale)`, 0 NaN/Inf on the five degenerate families. |
| **R6** | **monotone-linear probe** | Per feature family: fit the monotone linear class (sign-constrained) on ONE training slice under rev1 vs each rev2 arm; held-out **CID22, KonJND, AIC-3**; plus per-codec floor representability on the ladder, dial-attributed inversions, and identity-implies-zero. **Decision rule: keep rev1 for a family unless a rev2 arm wins a pre-registered majority of {CID22, KonJND, AIC-3} without regressing the other gates.** A tie ships arm A (§1.4). |
| **R7** | **identity** | The computed (not fabricated) identity vector at rev2 is no worse than rev1: no new non-zero slot above the audit's 2e-3 bar outside the 15 registered reference-only slots. |
| **R8** | **revision selector** | G3.2: selecting the current revision is byte-identical to not selecting one. G3.1: the set of slots each era moved, measured by re-extraction on a fixture, equals the registry's `revisions` entries exactly — a slot that moves and is not listed FAILS. |
| **R9** | **perf** | The revision switch costs nothing at the shipping revision. Measured on the named instrument (`zensim-bench/benches/ssim2_speed_bar.rs`) with the harness discipline this repo already records: identical-length env values, arms interleaved, min over >= 15 process starts with ASLR on, a bit-identical control arm, and **nothing else running on the box**. |

---

## 5. Recalculation manifest (fills as the survey lands)

Order is by cost, cheapest and highest-value first. Every table gets a
`_MANIFEST.json` carrying `build_commit`, `feature_set_id` (rev2 era token),
**decoder era per format**, and per-file sha256; LAN store + Tower mirror; a
`DATA_PROVENANCE.md` + `docs/DATASET_HISTORY.md` row.

| # | tables | why first |
|---|---|---|
| a | 372 + 944 eval roots, ladder + dial + corruption instruments, dial anchor | small, and every gate reads them |
| b | fast-class training legs (r1b-pools944 class), safesyn from bitstreams | the refit inputs |
| c | KADIS-700k (negrich subset first) | PNGs persisted |
| d | bigcodec (5.7M) | declare, TEST views first, stream the rest |

**Decoder era is a first-class output, not an afterthought.** DATASET_HISTORY
§3.34 measured re-decoding to move shipped B's dial by mean **-3.658** points —
73 % of the -4.98 extractor-era defect. One decoder era, chosen deliberately,
recorded per format, for every table in the wave.

Fleet: zenfleet only (`zenfleet-ctl declare` + `zenmetrics jobexec`), LAN +
tower only, no paid cloud. Bake the rev2 extractor into the canonical worker
image under a NEW TAG (never a new package name). First-cell gate before
scaling. Observe-before-load on every node; narrow cpusets; tower is
Docker-only and media has priority. Progress streams to
`~/tmp/rev2_progress.log`.

---

## 6. Refit + re-verdict at rev2

The shipped **D** lineage (ADD156 id100-negrich chain) and the fast-class
campaign's best servable recipe refit on rev2 legs; **G-ADDR** (contract +
resolvable floors + two-reference inversions) re-run on the rev2 instruments
with the `peer_ssim2` pins **re-derived in-era** (a pin read on rev1 pixels is
not a bar for a rev2 dial); ranked against the rev1 versions and the 944
leaders. Board cells land under a **distinct era suffix**, never mixed with
rev1.

**Ship rule unchanged:** install into `ZensimProfile::D` only on the full gate
AND CID22 >= today's D with CI. Otherwise the result is a proposal.

---

## 7. R6 — PRE-REGISTERED, 2026-09-05 (written and pushed before any extraction)

R6 is the one gate above that the lane could not run: §1.6 of
[`../benchmarks/feature_rev2_2026-09-05.md`](../benchmarks/feature_rev2_2026-09-05.md)
measured that the ladder proxy gives **identical** results for all four arms, so
the F4 arm has to be chosen by a real fit on real corpora. This section fixes
what will be run, on what, and how the answer is read off — before it is run.

### 7.1 One decision, not four

The plan's R6 line says *"keep rev1 for a family unless …"*. **AMENDED, and the
amendment is stated here rather than applied silently:** the arm is chosen
**GLOBALLY, once**, not per family. `ssim_form` exists precisely because the
per-pixel dissimilarity had 36 owners; letting `basic` take one arm and `IW`
another would re-create that defect on purpose. Per-family numbers are reported
as evidence, never as separate decisions.

### 7.2 What is extracted

**One binary, four arms, no rebuild between them** — `ZENSIM_SSIM_LUMA` ∈
`{ssim2, c1, lorentz, clamp}` (`ssim2` IS the shipped rev1 form). A rebuild
alone has been measured to move a timing ~10 % in this repo; it must not be
allowed to move a feature.

**Width: 372 (`extended+iw`, `num_scales=4`).** Justified, not assumed: F4's
registered blast radius is 132 slots — basic 36 + peaks 24 + masked 36 + IW 36 —
and at 372 **all 132 are live**. The folded 944 regimes zero `f156..371`, so a
944 read sees only the basic 36 and is a strictly weaker instrument for this
decision. A confirming 944 extraction is run on one corpus to check that
structural claim; the full 944 wave is part of the recalculation, not of R6.

| leg | rows | source of pixels |
|---|--:|---|
| safesyn (training) | 196,086 | bitstreams (`training_safe_synthetic.csv`; `.jpg` 111,068 / `.avif` 34,001 / `.jxl` 26,362 / `.webp` 24,655) — pre-scanned, **0 missing** |
| cid22val | 4,292 | `/mnt/v/dataset/cid22/CID22_validation_set` |
| kadid | 10,125 | `/mnt/v/dataset/kadid10k` |
| tid | 3,000 | `/mnt/v/dataset/tid2013` |
| konjnd | 1,008 | `/mnt/v/datasets/KonJND-1k` |
| aic3 | 600 | `/mnt/v/dataset/aic3_ctc_epfl` |
| csiq | 866 | `csiq_pairs.tsv` |
| live | 779 | `live_r2_pairs.tsv` |

**sdr25, aic4, nonphoto, imazen26, hfnlproxy and hf_nearlossless are byte-COPIES
in the postC root and are NOT re-extractable on this box** (record §3a). They are
declared ABSENT, never filled from a rev1 table — mixing eras inside one arm's
root is the exact defect the era ledger exists to prevent.

**Decoder era** is recorded per format in every manifest: extraction decodes
through `zensim-bench/examples/shared/zen_decode.rs` (magic-byte detect via
`zencodec`, then zenjpeg / zenpng / zenwebp / zenavif / zenjxl) at this lane's
build commit. §3.34 measured decoder era at 73 % of the extractor-era shift, so
it is an input, not a footnote.

### 7.3 Controls — run first, reported first

* **C1 pipeline control.** At arm `ssim2` the extraction must reproduce the
  registered postC 372 root **bit-exactly**. If it does not, nothing downstream
  is comparable and that is the finding.
* **C2 pathology detector.** `clamp` is exact for `D² ≤ 1` and differs from
  `ssim2` only above it, so *a row where `clamp` moves is a row containing at
  least one pixel in F4's pathological regime*. Report moved rows / cells /
  slots / max |Δ| per corpus. A corpus where `clamp` moves nothing **cannot
  discriminate an arm on pathology** and is reported as such rather than counted.
* **C3 identity.** At every arm, `ref == dist` must give the all-zero 372 vector
  (the registered 372 identity fact, `dial_addressability_gate` §15.3).

### 7.4 The fits — one owner, one recipe, one thing varied

`bake_dial_refit gram` → `bake_dial_refit fit-lasso`. No script computes a fit.

Flags are the did100 `ctl` recipe (which reproduces shipped Profile D
byte-identically) with **only the input tables changed**:
`--space raw --target human_score --lam 2e-3 --tau 0 --n-sweeps 400 --tol 1e-10`.

* **slices**: `0..155` (ADD156 / Profile-D lineage) and `0..227` (basic+peaks).
* **solvers**: `lasso` (the shipped recipe) **and** `bvls`
  (`--solver bvls --bounds-tsv benchmarks/feature_sign_mask_2026-05-26.tsv`) —
  the sign-constrained monotone-linear class the user's directive names. The
  bounded-variable CD solver already exists at the owner
  (`gram_lasso::box_cd_slice`, with an active-bound fixture test), so no new
  solver is written.
* **the sign mask is held FIXED across arms on purpose**: it encodes the
  structural direction of an error feature, and re-deriving it per arm would
  vary two things at once.
* **anchor**: each arm's OWN 2,000-row safesyn subset (stratified 6 codecs × 16
  quality points, fixed seed, identical row set across arms) — in-era, because a
  spline fit on rev1 pixels is not a spline for a rev2 dial. SROCC is invariant
  under the monotone spline, so the rank gates are unaffected by this choice;
  the dial gates are not, which is why it is in-era.

4 arms × 2 slices × 2 solvers = **16 fits**, every one differing from the others
only in its input table and its declared slice/solver.

### 7.5 Gates

| # | gate | criterion |
|---|---|---|
| **G1 RANK (primary)** | pooled SROCC on **CID22, KonJND (\|·\|), AIC-3** — the plan's held-out three — vs the `ssim2` arm, **paired** bootstrap (same resampled index sets, B = 2,000, seed 20260905) through `panel --batch` via `scripts/wave6_paired_bootstrap.py`. A win counts only if the 95 % CI on the DELTA excludes 0. |
| **G2 rank (reported)** | CSIQ, LIVE, TID. KADID is train==val for this class and is an integrity guard, never ranking signal. |
| **G3 OUTLIER REMOVAL** | over the 132 F4 slots on the union of all extracted rows, `max f_j` must be ≤ the arm's structural bound (`d ∈ [0,2]` ⇒ every `ssim_*` pool ≤ 2). rev1's value is reported beside it. **An arm that fails G3 cannot be Rev2 at any rank.** |
| **G4 HEALTHY-CELL PERTURBATION** | on rows C2 does NOT flag, no cell may move more than **1e-4 absolute** — ~6× the largest healthy-regime delta on record (1.62e-5 for `c1` on the invariant dump). Exceeding it means the arm is changing healthy content, which is a different claim than fixing an outlier, and it is reported as such rather than waved through. |
| **G5 DIAL** | identity ⇒ zero (C3); per-codec floor representability and two-reference dial inversions on the **ladder instrument at the same arm**. |

### 7.6 Decision rule — fixed before the numbers exist

1. An arm that **fails G3** is out, whatever its rank.
2. If exactly one surviving arm wins a **strict majority (≥ 2 of 3)** of
   {CID22, KonJND, AIC-3} against `ssim2` with CI-excluding deltas, and does not
   regress G4/G5, **that arm is Rev2**.
3. More than one qualifies → the one with more CI-excluding wins; still tied →
   **arm A (`c1`)**, per §1.4's prior, which was registered before any of this.
4. **No arm wins a rank majority → the arithmetic fix still lands.** The plan's
   original wording ("keep rev1") is amended here, explicitly and in advance,
   because keeping rev1 means shipping an unbounded metric and the user
   directive that authorized this lane is *"fix arithmetic defects aggressively
   before shipping"*. In that case Rev2 is the surviving arm with the **smallest
   healthy-cell perturbation** — moved-cell count on C2-unflagged rows, tie-broken
   by max |Δ| there.
5. **Stated in advance so it cannot be read as a retrofit:** rule 4 is expected
   to select `clamp`, because `clamp` is bit-identical to rev1 wherever
   `D² ≤ 1`. If the measurements land that way, that is a prediction confirmed,
   not a rule bent to fit.

### 7.7 Out of scope, named rather than omitted

The full recalculation (§5), the refit/re-verdict at rev2 (§6), R9 (perf), and
F12. `SHIPPED_REVISION` stays `Rev1` until the recalculation lands; R6 chooses
which arm the registry's `Rev2` **means**, and hands the fleet lane an arm token
plus a manifest.

---

## 11. R6b / F17 — PRE-REGISTERED, 2026-09-05 (written and pushed before any code)

R6 §9 reported, and deliberately did not fix, an unbounded feature that is **not
F4**: `contrast_inc` = `hf_energy_gain` = `max(0, hf_dst_L2 / hf_src_L2 − 1)`.
This section registers it as **F17**, fixes what will be run before it is run,
and states the decision rule before the numbers exist. It follows §7's shape
exactly — one owner, arms selected at RUNTIME from one binary, controls first.

The user directive that authorises the lane is *"we should fix arithmetic
defects aggressively before shipping … we have a fleet to recalculate"*.

### 11.1 The defect — F17

Site: `streaming.rs:590-608` (the buffered v1 `finalize`), `feature_v2.rs:5407-
5421` (the fold's v1-pool finalize) and `attribution.rs:575-587` (the
attribution integrand's gate) — three hand-copies of one expression, which is
the same "no owner ⇒ cannot be revised" position `ssim_form` was created to end.

```text
var_src = Σ(src − μ1)² / n      var_dst = Σ(dst − μ2)² / n
var_loss     = max(0, 1 − var_dst/var_src)      bounded [0, 1]
tex_loss     = max(0, 1 − mad_dst/mad_src)      bounded [0, 1]
contrast_inc = max(0, var_dst/var_src − 1)      UNBOUNDED ABOVE
```

The numerator of all three is `max(0, ·)` of a difference; the denominator of
all three is the **source** term. That bounds the two `loss` members
structurally — their numerator can never exceed their denominator — and bounds
nothing at all for the `gain` member, whose numerator is the distorted term.
A flat source region drives `var_src → 0` past the `> 1e-10` guard while the
distorted image still carries HF, and the ratio runs away. **The guard is a
threshold, not a stabiliser**: it decides whether to divide, never what to
divide by.

The crate already owns the fixed form. v2's `HF_GAIN` — the same quantity, same
units (squared XYB energy), one block over — is
`bounded_excess_pair(hf_dst_sq, hf_src_sq, C_HF)`, i.e. `max(0, a−b)/(a+b+C_HF)`,
with `C_HF = 1e-4` declared as *"stabilizer for the HF gain/loss/mag-loss
bounded-excess forms"*. So F17 is not a missing idea; it is one block that did
not use the idea.

### 11.2 Blast radius — MEASURED, not asserted

12 slots at 372 (`basic` block-local 12, all four scales × all three channels):
**f12 f25 f38 f51 f64 f77 f90 f103 f116 f129 f142 f155**. The same 12 at a
pools-live 944 (`foldapp2pools`) and at every zeroed `ext944`/`ext924` root,
because `contrast_inc` is in the basic block, which those roots keep. Unlike
F4, F17's slot count does **not** vary with pool state.

Measured over the R6 rev1 (`ssim2`-arm) tables — 216,756 rows, 8 legs, every
one of the 372 slots (`/mnt/v/output/zensim/rev2-2026-09-05/r6b/slot_audit_rev1.tsv`, script
`scripts/r6b_audit_slots.py`):

| leg | n | max `var_loss` | max `tex_loss` | max `contrast_inc` | rows w/ cell > 100 |
|---|--:|--:|--:|--:|--:|
| safesyn | 196,086 | 1.000000 | 1.000000 | **36,465.74** | 63 |
| LIVE | 779 | 0.999329 | 0.981655 | **3,598.21** | 60 |
| TID | 3,000 | 1.000000 | 0.999999 | **927.91** | 73 |
| KADID | 10,125 | 1.000000 | 1.000000 | **618.28** | 88 |
| CSIQ | 866 | 1.000000 | 0.999999 | **163.13** | 5 |
| KonJND | 1,008 | 1.000000 | 0.999601 | 6.24 | 0 |
| CID22 | 4,292 | 0.972878 | 0.845740 | 3.72 | 0 |
| AIC-3 | 600 | 0.768070 | 0.640974 | 0.62 | 0 |

**The twelve `contrast_inc` slots are the top twelve of all 372 by maximum, and
the thirteenth (`peaks_ssim_max_s0_Y`) is 1.972.** The population separates with
no overlap. Against the gold photographic holdout's own p99.9 over these slots
(CID22, **0.34687**) the worst value is **×105,127**. Pooled over 2,601,072
`contrast_inc` cells: 2.59 % exceed 1, 0.30 % exceed 10, **0.0198 % exceed 100**,
0.0012 % exceed 1,000. Nonzero on 12.1 % (CID22) to 52.1 % (KADID) of cells.

**This is not F4's shape.** F4's 5.8e6 belongs to a bigcodec sweep with no local
pixels and fires on **zero** of these 216,756 rows; F17 fires on all five of the
distortion corpora and on the training leg.

`r = var_dst/var_src` is EXACTLY recoverable from a stored rev1 table as
`1 + contrast_inc − var_loss`, because one of the two is always zero — verified,
**0 of 2,601,072 cells have both positive**. That makes every arm that is a pure
function of `r` auditable on tables that already exist.

### 11.3 Per-bake exposure — MEASURED (`bake_block_profile` ∩ the 12 slots)

Transform status is read from each bake's own `zentrain.feature_transforms`,
classified BOUNDED (`winsor_p99`, `quantile_bins`, `clip_then_log1p`, …),
COMPRESSING (`log1p`, `signed_cbrt`, `yeo_johnson` — smaller, still unbounded)
or RAW (`identity`, or no transform block at all).

| shipped bake | reads | F17 read | bounded | compressing | RAW | worst measured max on a non-bounded slot |
|---|--:|--:|--:|--:|--:|--:|
| **D** `d_sdr_add156_id100_negrich` (the SDR default) | 28 | 2 | 0 | 0 | **2** | **2,127** (f155, f116=1,380) |
| **CHdr** `c_hdr_l1t1944` | 697 | 12 | 0 | 0 | **12** | **36,466** |
| **C** `c_sdr_purity944` | 667 | 12 | 10 | 1 | 1 | 3,430 (f38 `signed_cbrt`), 43.3 (f25 raw) |
| **BHdr** `bhdr_linear_shaped_cvvdpmix` | 133 | 6 | 4 | 2 | 0 | 36,466 (f129 `yeo_johnson`) |
| **A** `v47_strict_qat_native` | 285 | 7 | **7** | 0 | 0 | — |
| **B** `b_sdr_linear_cid80_inclwinsor` | 95 | 6 | **6** | 0 | 0 | — |

The mitigation pattern R6 found for F4 repeats and is **worse**: A and B are
fully guarded on their F17 slots, **D and CHdr are not guarded at all**, and
D — today's SDR default — carries no `feature_transforms` block whatsoever. It
reads two slots whose measured maxima on real corpora are 1,380 and 2,127, into
a 28-input monotone linear head, unclamped. **A bake-side transform therefore
cannot be the answer**: it is exactly what is already deployed, and it is
exactly what the default does not have.

### 11.4 Sibling audit — the whole feature surface, once

Every other candidate was checked, empirically at 372 and structurally beyond it.

| # | site | form | bounded? | measured max | joins the fix? |
|---|---|---|---|--:|---|
| 1 | v1 `contrast_inc` (`hf_energy_gain`) ×3 copies | `max(0, a/b − 1)` | **NO** | **36,465.74** | **YES — F17** |
| 2 | v1 `var_loss` | `max(0, 1 − a/b)` | yes, `[0,1]` (num ≤ den) | 1.000000 | no |
| 3 | v1 `tex_loss` | `max(0, 1 − a/b)` L1 | yes, `[0,1]` | 1.000000 | no |
| 4 | v2 `HF_GAIN`/`HF_LOSS`/`HF_MAG_LOSS` | `bounded_excess(·,·,C_HF)` then `clamp01` | yes | n/a at 372 | no — it is the MODEL |
| 5 | v2 `GMS`, `EDGE_WIDTH_CHANGE` | `1 − bounded_sim(·,·,C)` | yes, `[0,1)` | n/a at 372 | no |
| 6 | v2 `RINGING` | `saturate·saturate·(1−saturate)` | yes, `[0,1)` | n/a at 372 | no |
| 7 | v2 `BANDING` | `bounded_excess · (1−saturate)` | yes, `[0,1)` | n/a at 372 | no |
| 8 | v2 rest (23 slots) | `clamp01` / `clamp02` / `saturate` | yes | n/a at 372 | no |
| 9 | append 17 slots incl. `GLOBAL_*`, `LUM_*` | `clamp01` / `clamp(0,2)` / `saturate` | yes | n/a at 372 | no |
| 10 | CSFW `W_GLOBAL_*` | `saturate` / `bounded_excess_pair` + `clamp01` | yes | n/a at 372 | no |
| 11 | CSFW `gvar{1,2}_w = Σws²/Σw − wmean²` | cancellation-prone, **output bounded** | yes (output) | n/a at 372 | **no** — see below |
| 12 | v1 peaks `ssim_max`/`ssim_l8` | F4's `d`, pooled | F4's problem | 1.972 | no — F4 owns it |
| 13 | diffmap `f = n/Σw` (masked/IW) | guarded `Σw > 1e-12` | not a feature | — | no |

Only row 1 exceeds the photographic p99.9 by more than 100× — it exceeds it by
**105,127×**, and every other slot in the crate's 372-wide surface tops out at
1.972. **Row 1 is the only one that joins the fix.**

**Row 11 is reported, not fixed, and the reason is stated so it is not mistaken
for an oversight.** `finish_csfw` carries the identical catastrophic-
cancellation form F5 names (`Σws²/Σw − wmean²`), and revision 2's
`paired_global_contrast` remedy is wired to `global_stats_from_raw_moments`
ONLY — it does not reach the CSFW twin. But CSFW's output goes through
`bounded_excess_pair` + `clamp01`, so this is a PRECISION defect with a bounded
output, not an unboundedness defect, and it has no second route to be skewed
against (F5's parity framing does not apply). It is registered here as an open
observation for the F5 lane, with no arm and no gate in R6b.

### 11.5 The arms — one owner, runtime-selectable, legacy bit-identical

New module `zensim/src/hf_gain_form.rs`, modelled on `ssim_form.rs` and for its
stated reason. `HfGainForm::for_revision(Rev1) = RatioExcess`; the arm Rev2
means is named ONCE at `HfGainForm::REV2_HFGAIN`. `ZENSIM_HF_GAIN` is a
MEASUREMENT override only, never a production path — the production switch is
the equal-byte-length `ZENSIM_FORMULA_REV`. Let `a = var_dst`, `b = var_src`
(the `b > 1e-10` gate is unchanged in every arm) and `g = max(0, a/b − 1)`.

| arm | token | form | bound | agrees with rev1 | new constant |
|---|---|---|---|---|---|
| **A0** legacy | `ratio` | `g` | none | exactly | — |
| **A1** | `bexcess` | `bounded_excess(a, b, C_HF)` = `max(0, a−b)/(a+b+C_HF)` | `[0, 1)` | to **half** (`≈ g/2` as `g → 0`) | none — `C_HF` is the family's own declared stabiliser **for this quantity** |
| **A2** | `log1p` | `ln(1 + g)` | **none** (log growth) | 1st order | none |
| **A3** | `satexcess` | `saturate(g, 1) = g/(g+1)` | `[0, 1)` | **1st order** (`g − g² + …`) | none — `c = 1` is the unique scale that agrees to 1st order, the same derivation `SsimLumaForm::Lorentz` used |
| **A4** | `cap` | `min(g, 1)` | `[0, 1]` | exactly for `g ≤ 1` | none — 1.0 is the bound the two `loss` siblings already have |

Two structural facts, both stated in advance because they will decide this:

* **A3 restores the family's src↔dst symmetry by changing only the broken
  member.** `g/(g+1) = max(0, 1 − var_src/var_dst)`, the exact reflection of
  `var_loss = max(0, 1 − var_dst/var_src)`. A1 makes the gain member
  `bounded_excess(a,b,C_HF)` while the loss member stays `max(0,1−a/b)`, so A1
  either leaves the family inconsistent or expands the blast radius from 12
  slots to 36 by re-forming two features that are not broken.
* **A2 and A4 are the only arms expressible as a bake-side transform** (both are
  pure functions of `g`). §11.3 measures that the bake-side layer is precisely
  what the SDR default lacks, so an arm's expressibility there is not a merit.

### 11.6 What is extracted, and the controls

Same instrument as §7.2: 372 (`extended+iw`, `num_scales = 4`), one binary, arm
selected at runtime, same eight legs, same decoder era, artefacts under
`/mnt/v/output/zensim/rev2-2026-09-05/r6b/`.

* **CB1 — reuse control (run FIRST).** The rev1 arm of this lane must reproduce
  R6's `ssim2` tables byte-for-byte. `1aa3a419` touched `feature_v2.rs` and
  `fold_engine.rs` between R6's build commit `ceb86c2d` and this lane's base, so
  this is a check, not an assumption. PASS ⇒ R6's `ssim2` tables and its four
  `ssim2_s{156,228}_{lasso,bvls}` bakes are this lane's rev1 control unchanged.
  FAIL ⇒ the control arm is extracted fresh and the mismatch is the finding.
* **CB2 — pathology detector.** A cell is PATHOLOGICAL when its rev1 value
  exceeds **0.34687**, the gold photographic holdout's own p99.9 over the 12
  slots; a row is flagged when any of its 12 is. Reported per corpus, so a
  corpus that flags nothing is reported as unable to discriminate rather than
  counted as evidence.
* **CB3 — identity.** `ref == dist` ⇒ the all-zero 372 vector at every arm.
* **CB4 — containment.** Every arm's table must differ from the rev1 arm's in
  **exactly** the 12 F17 columns and nowhere else, cell for cell. This is what
  makes the fits an A/B on one feature rather than on a rebuild.

### 11.7 The fits

`bake_dial_refit gram` → `fit-lasso`, the §7.4 did100 `ctl` recipe verbatim
(`--space raw --target human_score --lam 2e-3 --tau 0 --n-sweeps 400
--tol 1e-10`), only the input tables changed. **Slices `0..155` and `0..227`**
(both contain all 12 F17 slots; `0..371` adds only masked/IW, which F17 does not
touch), **solvers `lasso` and `bvls`** with the frozen sign mask
`benchmarks/feature_sign_mask_2026-05-26.tsv`, in-era 2,000-row safesyn anchor
per arm on the identical row set. 4 new arms × 2 slices × 2 solvers = **16 new
fits**; the rev1 control's 4 come from R6 under CB1.

### 11.8 Gates

| # | gate | criterion |
|---|---|---|
| **H1 RANK** | pooled SROCC on CID22, KonJND (\|·\|), AIC-3 vs the rev1 arm, **paired** bootstrap (same resampled index sets, B = 2,000, seed 20260905) through `panel --batch`. A win counts only if the 95 % CI on the delta excludes 0. |
| **H2 rank (reported)** | CSIQ, LIVE, TID. KADID is train==val for this class — integrity guard, never ranking signal. |
| **H3 OUTLIER REMOVAL (gating)** | over the 12 F17 slots on the union of every extracted row, `max f_j` ≤ the arm's **structural** bound. An arm with no structural bound fails H3 whatever it measures — **A2 `log1p` is expected to fail here and is carried anyway**, so the cost of the log family is measured rather than assumed. |
| **H4 ZERO PRESERVATION (gating)** | every arm gives **exactly** 0 on every cell where rev1 gives 0 (`a ≤ b`, or the `var_src ≤ 1e-10` branch). Keeps the slot a `Difference` form and the identity vector zero. |
| **H5 ORDER PRESERVATION (gating)** | the arm is strictly increasing in `g` for `g > 0`: Spearman(arm, rev1) over all cells with `g > 0` must be exactly 1.0. **A4 `cap` is expected to fail** — F4's `Clamp` paid this price because nothing else could; F17 has arms that do not have to. |
| **H6 HEALTHY-CELL PERTURBATION (RANKING, not gating)** | on CB2-unflagged rows: moved cells, max \|Δ\|, median \|Δ\| over moved cells, and the count exceeding 1e-4. **Deliberate deviation from §7.5's G4, stated before the numbers:** bounding an unbounded feature necessarily moves it wherever it is nonzero, so a 1e-4 pass/fail bar would reject every arm including the correct one. It ranks; it does not gate. |
| **H7 DIAL** | identity ⇒ zero (CB3); per-codec floor representability, monotonicity/tied rate and two-reference inversions on the **ladder instrument at the same arm**; G-ADDR block re-read in-era. No arm may regress it. |

### 11.9 Decision rule — fixed before the numbers exist

1. An arm that fails **H3**, **H4** or **H5** is out, whatever its rank.
2. If exactly one survivor wins a strict majority (≥ 2 of 3) of {CID22, KonJND,
   AIC-3} on **H1** with CI-excluding deltas and does not regress **H7**, that
   arm is F17's Rev2 form.
3. More than one qualifies → the one with more CI-excluding wins.
4. **No arm wins a rank majority → the arithmetic fix still lands.** Per the
   user directive, and per §7.6's precedent. In that case the form is the
   surviving arm with the smallest **H6** healthy-cell perturbation (moved-cell
   count on CB2-unflagged rows, tie-broken by max \|Δ\| there).
5. Exact tie on H6 → **A3 `satexcess`**, the registered prior, for §11.5's two
   structural reasons (1st-order agreement with rev1, and family symmetry
   restored by changing only the broken member).
6. **Stated in advance so it cannot be read as a retrofit:** rule 4 is expected
   to select A3, because A1 rescales even the smallest values by ~½ while A3
   agrees to first order. If the measurements land that way it is a prediction
   confirmed. If A1 wins the rank majority at rule 2, A1 ships and its 36-slot
   family-consistency question is opened as its own lane rather than settled
   here.

### 11.10 Landing

`REV2_HFGAIN` is named once; `FormulaRevision::Rev2` gains the era token
**`v1hfgain`** alongside `v1ssimcap` and `freecomp`, so `ZENSIM_FORMULA_REV=2`
covers F4 + F5 + F17 as **one** era boundary and one recalculation. Status stays
`Proposed` while `ssim_form::SHIPPED_REVISION` is `Rev1`; §5's manifest gains
F17's 12 slots keyed on `feature_set_id`, and §6's refit list gains **D** and
**CHdr** as the two profiles with unguarded exposure.

### 11.10a ANSWERED 2026-09-06 — `REV2_HFGAIN = SaturatingExcess`

Rule 2 fired: it is the sole survivor of H3/H4/H5 AND wins a strict majority of
the primaries with CI-excluding deltas in 2 of the 4 variants. Rule 6's
prediction held; one of the two arguments behind it (that `bexcess` preserves
order) was FALSIFIED by measurement and is recorded as such. Full numbers,
gate table and the reading-ambiguity handled both ways:
[`../benchmarks/feature_rev2_2026-09-05.md`](../benchmarks/feature_rev2_2026-09-05.md)
§11.8–§11.10.

### 11.11 Out of scope, named rather than omitted

The recalculation (§5), the refit/re-verdict (§6), R9 (perf), F12, and row 11 of
§11.4 (the CSFW cancellation twin, handed to the F5 lane). `SHIPPED_REVISION`
stays `Rev1`.

---

## 12. REV2-D — the D-class F17 arm study. PRE-REGISTERED 2026-09-06, pushed before any fit

**Lane:** REV2-D. **Record:** [`../benchmarks/rev2_d_arms_2026-09-06.md`](../benchmarks/rev2_d_arms_2026-09-06.md).

### 12.0 The question, and why §11.10a does not answer it

§11.10a set `REV2_HFGAIN = SaturatingExcess` from R6b's own fits — `--lam 2e-3`
against a ×100 target, **147 active coefficients**, CID22 **+0.00272**
CI-excluding. [`../benchmarks/rev2_refit_2026-09-06.md`](../benchmarks/rev2_refit_2026-09-06.md)
§3 then ran the **shipped D chain** — the same λ against a [0, 1] target, **28
active coefficients**, its own two anchors — on the same tables and measured the
**same arm losing 0.00456**, also CI-excluding, plus a per-codec floor
(`avif-rav1e`) going PASS → FAIL. §9 of that record names the cause:
**F17's benefit is model-class-dependent.**

The USER DIRECTIVE that authorised the rev2 work is *"we should fix arithmetic
defects aggressively before shipping, and perhaps change feature definitions and
formulas to make monotone linear models better."* So **the bound is mandatory
and is not up for re-litigation here.** What is open is the **FORM**, and it was
chosen by measurement on ONE model class. This lane measures every form on the
class that actually ships.

A structural fact measured before this section was written, from
`bake_block_profile` on the REFIT lane's own bakes, which is what makes the
question sharp: **the two arms do not read the same slots.** Against the twelve
F17 ids {12, 25, 38, 51, 64, 77, 90, 103, 116, 129, 142, 155} —

| bake | active coeffs | F17 slots read | which |
|---|--:|--:|---|
| `D_ratio_s156` (revision 1) | 28 | **1** | f77 |
| `D_satexcess_s156` (revision 2) | 33 | **5** | f12, f38, f64, f129, f155 |
| `D_ratio_s228` (revision 1) | 27 | **1** | f77 |
| `D_satexcess_s228` (revision 2) | 39 | **7** | f12, f25, f38, f51, f64, f129, f155 |

Bounding the statistic does not merely rescale an input the model already had:
**it changes which inputs L1 keeps.** So "rev1 vs rev2 in the D chain" is a
comparison of two different sparse supports, and a per-arm sweep is the only way
to see whether the loss belongs to *bounding* or to *this particular bound*.

### 12.1 Decision rule — fixed before any number exists

**Step 0 — recommendability.** An arm may only be RECOMMENDED if it passes
R6b's structural gates H3 (declared bound holds), H4 (identity ⇒ zero) and H5
(no inversions, no new ties) — §11.9's gate table. Today that is `satexcess`
alone (`log1p` fails H3, `bexcess` fails H5 with 263,195 inversions, `cap` fails
H5 with 67,224 new ties). Every arm is still RUN and REPORTED; a
structurally-failing arm that dominates on rank and floors is a FINDING for the
user, never an install.

**Step 1 — eligibility.** ELIGIBLE = arms that keep **G-ADDR contract 6/6** AND
whose `A7r` per-codec `repr` is **≥ the revision-1 in-era arm's on every one of
the five codecs**. (The bar is the rev1 in-era arm, not the mentor, because this
is a regression question: the brief's words are "floors ≥ rev1's on every
codec". The mentor bar is reported beside it and decides the gate's own
PASS/FAIL, which is reported unchanged.)

**Step 2 — preference.** Among eligible arms prefer the one whose paired-
bootstrap **CID22** delta against the revision-1 in-era arm is not worse than
zero — CI including 0, or CI excluding 0 on the positive side. Ties broken by
(a) CI-excluding wins minus losses over {CID22, KonJND, AIC-3, CSIQ, LIVE};
**TID and KADID are train==val integrity guards and are never ranking signal**;
then (b) the smaller H6 healthy-cell perturbation already measured in §11.9
(`cap` 0 < `log1p` 0.04907 < `satexcess` 0.08931 < `bexcess` 0.34410).

**Step 3 — no eligible arm.** Report the smallest CID22 loss among the arms
holding contract 6/6 and state plainly that the bound costs rank in this model
class. Do not relax a gate to manufacture a winner.

**Step 4 — one feature definition, not two.** The registry definition stays
`satexcess` for the wide-lasso class **unless a single arm wins for BOTH
classes**. A per-class revision is NOT allowed — one feature has one definition.
If the classes disagree, this lane presents the trade with numbers and **the
user rules**. Nothing is installed either way.

### 12.2 What is run

Five arms — `ratio` (= revision 1, the control), `cap`, `log1p`, `bexcess`,
`satexcess` — × two slices (`a156.idx`, `a228.idx`) = **10 bakes**, through the
shipped-D chain verbatim:

```
bake_dial_refit gram      --parquet <r6b>/tables/<arm>/safesyn.parquet \
                          --target human_score --target-scale 0.01 --space raw
bake_dial_refit fit-lasso --space raw --target human_score --lam 2e-3 --tau 0 \
                          --n-sweeps 400 --tol 1e-10 --slice-file a{156,228}.idx \
                          --anchor-parquet <r6b>/tables/<arm>/anchor.parquet \
                          --anchor-parquet identity_anchor_<arm>_n21.parquet \
                          --anchor-target human_score --embed-repro
bake_dial_refit extend-top                       # byte no-op in this lineage, re-verified per arm
bake_dial_refit densify   --gate-rows 512        # identity gate BIT-identical, per arm
```

**The winsor control — bounding the MODEL's input instead of the feature.**
Two variants, both on the **revision-1** (`ratio`) tables, so no era break is
involved at all:

* **`W-all`** — `add-winsor` at its shipped defaults ([p0.1, p99.9] fit on the
  training leg, every slot). This is the recipe shipped Profile B carries
  (`b_sdr_linear_cid80_inclwinsor`), applied to the D lineage.
* **`W-f17`** — winsor [p0.1, p99.9] on the **twelve F17 slots only**, identity
  everywhere else, applied at gram-build time (`gram --transforms-tsv --space
  shaped`) and declared into the bake (`fit-lasso --transforms-tsv`), so the fit
  space and the f32 runtime see the identical clamp. This is the variant that
  isolates F17; `W-all` also clamps 20-odd non-F17 inputs and is reported as the
  shipped-recipe comparator, not as the isolation.

Both are graded identically to the arms. **If a winsor variant matches the best
arm on rank AND on floors, that is a finding the user needs** — a bounded model
with no era break, no recalculation, and no re-extraction — and it will be
stated beside, not instead of, the arithmetic-correctness argument for fixing
the feature (§11.1: the crate owns `bounded_excess` one block over and F17 is
the one site that did not use it).

**Grading, identical for every arm, each read on its OWN era root:**

| axis | how |
|---|---|
| RANK | `bake_verdict` on cid22, konjnd, aic3, csiq, live, tid, kadid + **paired bootstrap** vs the rev1 in-era arm (B = 2,000, seed 20260905, same resampled index sets, through `panel --batch`) |
| G-ADDR | `--floor-rule resolvable --gaddr-tail-pins product`, contract C1–C6 + A7r per codec, on the arm's OWN `<arm>_ladder.parquet` / `<arm>_negtail.parquet` / `<arm>_identity.parquet` |
| floors | A7r `repr` per codec, against BOTH the live mentor fraction (the gate's own bar) and the rev1 in-era arm (step 1's bar) |
| inversions | `--reference-truth reference_truth_ladder_pnorm3.tsv:pnorm3 --inversion-truth agree` |
| identity | the arm's own identity table through the bake |
| **outlier ordering** | the LIVE rows carrying a revision-1 `contrast_inc` cell **> 100** (R6b: 60 of 779). Per arm, the SROCC on that subset and on its complement — *does each arm fix the ordering where the pathology actually fires?* |

### 12.3 Controls, run first, reported first

| # | control | pass condition |
|---|---|---|
| **C-A** | this lane's `ratio` chain reproduces the REFIT lane's | `D_ratio_s{156,228}_byid.bin` byte-identical modulo the embedded `zentrain.repro` (which carries the tool's absolute path), AND CID22/KonJND read **0.86367 / 0.56141** (s156) and **0.87263 / 0.58220** (s228) EXACTLY |
| **C-B** | this lane's `satexcess` chain reproduces the REFIT lane's | `D_satexcess_s156_byid.bin` likewise; CID22 **0.85911** |
| **C-C** | identity anchors | the 21-row anchor cut from every arm's own `identity.parquet` is byte-identical across all five arms (all-zero at every arm — R6b CB3) |
| **C-D** | `extend-top` is a byte no-op | in → out sha equal, every arm, both slices |
| **C-E** | `densify` identity gate | BIT-identical on 512 probe rows, every arm, both slices |

**If C-A fails, this lane's chain is not the REFIT lane's and every number below
is void.** It is reported as a failure, not worked around.

### 12.4 Confounds inherited, stated not discovered

* **CID22 decoder**: 12.5 % of the CID22 distorted side in the r6b lineage is
  `.jpg` decoded by the third-party `image` crate. Every arm here carries the
  SAME one, so the comparison is single-confound; the REFIT lane §6 priced it at
  **2e-5** CID22 SROCC against a 4.56e-3 effect.
* **Tables**: R6b's are the only rev2 extraction of these legs that exists. No
  re-extraction is run here.
* **KonJND** is the 504-row JPEG ruler on these roots, not the 1,008-row cut.
* **KADID / TID** are train==val on this lineage and are printed as integrity
  guards only.

### 12.5 What this lane does NOT do

No install. `ssim_form::SHIPPED_REVISION` untouched, `ZensimProfile::D`
untouched, `zensim/weights/` untouched, `benchmarks/feature_sets_registry.json`
untouched, and the G-ADDR floor registry **NOT appended** — the arm probes are
not the wave's canonical rev2 instruments and pinning bars to them would pin
bars to instruments about to be superseded. No fast-class retrain and no
corruption-head refit: the REFIT lane measured the tree CANDIDATE invariant to
the revision to nine decimals (§8.6r).

### 12.6 RESULTS — ANSWERED 2026-09-06. Step 3 fired; the registry does NOT move

Full numbers: [`../benchmarks/rev2_d_arms_2026-09-06.md`](../benchmarks/rev2_d_arms_2026-09-06.md).

**Controls all pass.** This lane's `ratio` and `satexcess` grams are
**BYTE-IDENTICAL** to the REFIT lane's, its four rev1/rev2 bakes are
byte-identical modulo the tool-path-bearing `zentrain.repro`, `extend-top` is a
byte no-op on all 10 arm bakes, `densify`'s identity gate is BIT-identical
10/10, and the `satexcess` a156 CID22 delta reproduces the REFIT lane's
**−0.00456 [−0.00549, −0.00364]** exactly from an independent gram, fit, verdict
and bootstrap.

**Step 0** leaves `satexcess` the only recommendable arm (`log1p` fails H3,
`bexcess` and `cap` fail H5). **Step 1** finds NO eligible candidate at a156 —
every bounded arm and every winsor variant drops `avif-rav1e` below revision 1's
0.6667, and `W-all-carried` additionally breaks contract C6. At a228 `cap` and
`satexcess` are eligible (floors equal on all five codecs), but **step 2** fails
for both: CID22 −0.00309 and −0.00406, CIs excluding zero. **Step 3 fires.**

> **The bound costs CID22 in the shipped D's model class on every form tested,
> and `satexcess` — the form chosen for the wide class — is the most expensive
> of the four at BOTH slices** (−0.00456 / −0.00406 vs `log1p`'s −0.00088 /
> `bexcess`'s −0.00206, both of which step 0 eliminates).

**Step 4** is not met: `satexcess` is best for R6b's 147-coefficient class and
worst for the 28–39-coefficient D class, and no arm wins both. So
**`REV2_HFGAIN` STAYS `SaturatingExcess`** and the trade goes to the user
(record §10.1). Nothing was installed.

**The bake-side control produced the finding the brief anticipated.**
`W-f17` — `winsor_p99 [p0.1, p99.9]` on the twelve F17 slots only, revision-1
features, clamp inside the fit — is **the only non-revision-1 model in the study
that keeps A7r 0 fails AND contract 6/6**, and it has the best outlier-row
ordering of all eight (+0.94348 against revision 1's +0.93491 on the 60 LIVE
rows whose `contrast_inc` exceeds 100). It is **not free** (CID22 −0.00348, 76 %
of `satexcess`'s cost) and it is **not eligible** either (its `avif-rav1e` 0.6410
sits on the mentor bar but below revision 1's 0.6667). Two neighbours settle its
scope: `W-all`, the same guard on all 372 slots (the shipped Profile-B recipe),
is the WORST model in the study (CID22 −0.01536, KonJND −0.02339, outlier-row
SROCC +0.87152); and `W-all-carried`, the literal `add-winsor` chain whose
spline is carried, ties revision 1 on CID22 (−0.00002, CI includes 0) and WINS
CSIQ by **+0.04862** — but breaks contract C6 (1 of 9,593 cells out-scores a
perfect copy), exactly the stale-spline defect §12.2 predicted from the owner's
source. A serve-time guard with a REFIT spline is registered as a follow-up, not
run.

---

### 12.7 REV2-D-GUARD — the serve-time guard with a REFIT spline. PRE-REGISTERED 2026-09-06, pushed before the first bake

**Lane:** REV2-D-GUARD. **Record:**
[`../benchmarks/rev2_d_arms_2026-09-06.md`](../benchmarks/rev2_d_arms_2026-09-06.md)
§11 addendum. **Artefacts:** `/mnt/v/output/zensim/rev2-d-arms-2026-09-06/guard/`.

**The registered follow-up this executes** is §12.6's last sentence and record
§10.2's last bullet: *"a serve-time guard whose spline is REFIT would keep the
gain without the C6 break — registered here, not run."*

#### 12.7.0 The question

`W-all-carried` — revision-1 WEIGHTS, all 372 slots clamped at serve time,
spline CARRIED — is the only model in §12.6 that ties revision 1 on CID22
(−0.00002, CI includes 0) and it WINS CSIQ by **+0.04862**, the largest single
rank gain in that study. It breaks contract **C6** (1 of 9,593 grid cells
out-scores a perfect copy) and the `avif-svt` floor, and the cause was read
from `cmd_add_winsor`'s source before it was run: the tool writes the transform
tokens and then carries *"everything the raw bake had (incl. its spline)
verbatim"*, so the output calibration was fitted on the **un**-clamped net.

`W-f17` — twelve slots, clamp INSIDE the fit — is the only non-revision-1 model
with A7r 0 fails AND contract 6/6 AND the best LIVE-outlier ordering
(+0.94348), but it REFITS the weights on the clamped gram, lands a different
sparse support, and costs CID22 −0.00348.

**Neither arm is the combination the follow-up names.** This lane runs it:
**revision-1 weights, unchanged; the guard applied at serve time; the spline
refit on the guarded net.** If the C6 break is the carried spline and nothing
else, the refit removes it while keeping `W-all-carried`'s tie-and-CSIQ-win —
and the scope finding says the twelve-slot scope should also keep the outlier
ordering that `W-all` destroys.

#### 12.7.1 Decision rule — fixed before any number exists

An arm is a **SHIP CANDIDATE** iff ALL FOUR hold:

1. **G-ADDR contract 6/6.**
2. **`A7r` per-codec `repr` ≥ the revision-1 in-era arm's on EVERY one of the
   five codecs** (`avif-rav1e` 0.6667, `avif-svt` 1.0000, `jpeg` 0.7179, `jxl`
   1.0000, `webp` 1.0000 at a156 — the same bar §12.1 step 1 used).
3. **Paired-bootstrap CID22 delta vs revision 1 not worse than zero** — CI
   including 0, or excluding 0 on the positive side.
4. **Two-reference inversions PASS** (G3, rate ≤ 0.07).

A ship candidate goes to the USER beside the era-break options; it is **not
installed by this lane**. If no arm qualifies, the best arm and the axis it
misses on are reported, and nothing moves. `SHIPPED_REVISION` stays `Rev1`,
`ZensimProfile::D`, `zensim/weights/` and the feature-set / G-ADDR-floor
registries are untouched either way.

#### 12.7.2 The chain — one thing varied

Base bakes are §12.6's own `D_ratio_s{156,228}_raw.bin`: 372-wide, **no
transform block**, revision-1 features, revision-1 weights, and the fit-lasso
spline over `ratio/anchor.parquet` (2,000 rows, `human_score` ∈ [−64.22,
98.08], 70 negative — the negrich anchor) + `identity_anchor_ratio_n21.parquet`
(21 rows at exactly 100.0 — the id100 anchor).

```
add-winsor    --in D_ratio_s<S>_raw.bin --slots <spec> --lo-pct 0.1 --hi-pct <Q>
              --fit-corpus <r6b>/tables/ratio/safesyn.parquet     [OWNER EXTENSION]
shared-anchor --anchor <r6b>/tables/ratio/anchor.parquet
              --anchor identity_anchor_ratio_n21.parquet
              --target-col human_score                            [OWNER EXTENSION]
extend-top    --anchor <r6b>/tables/ratio/anchor.parquet --target-col human_score
densify       --gate-rows 512
```

**The carry defect is avoided by ORDERING, not by patching `add-winsor`:**
`shared-anchor` forwards the anchor through `build_fw_ops(&model)`, i.e. through
the bake's declared transforms, so a refit that runs AFTER the guard is fitted
on exactly what the runtime sees. Two owner extensions are required and are the
only code this lane changes:

* **`add-winsor --slots`** — restrict which feature indices receive
  `winsor_p99`; every other slot emits the `identity` token with empty params,
  the same convention `fit-lasso --transforms-tsv` already emits. Omitting the
  flag must be **byte-identical** to today (gated).
* **`shared-anchor --anchor` becomes REPEATABLE** — anchors concatenated in
  argv order, mirroring `fit-lasso --anchor-parquet`. A single `--anchor` must
  be **byte-identical** to today (gated).

**Held constant:** the base weights, the two anchors, `--lo-pct 0.1`, the spline
fitter and its 18 percentile edges, `extend-top`'s band, `densify`'s gate size,
the eval corpora and era root (revision 1 — these models ARE revision 1), the
dial instruments, the floor rule, the tail pins, and the bootstrap's seed
(20260905) and index sets. **Varied:** the guard's SCOPE and its upper
percentile.

#### 12.7.3 Arms — 5 × 2 slices = 10 bakes

| arm | guard scope | hi-pct | answers |
|---|---|--:|---|
| **`R-refit`** | none | — | **CONTROL.** Isolates the refit path itself: same weights, same anchors, no guard. Every guarded arm is read against THIS as well as against published `ratio`. |
| **`W-f17-refit-p999`** | the twelve F17 slots | 99.9 | brief (a) — the registered follow-up at the published window |
| `W-f17-refit-p99` | the twelve F17 slots | 99 | brief (b) |
| `W-f17-refit-p95` | the twelve F17 slots | 95 | brief (b) — does a tighter window fix the LIVE outliers' ordering without moving healthy rows? |
| `W-f17cell-refit-p999` | the twelve F17 slots **+ the two D-read cells entire** (f104–f116, f143–f155): 36 slots | 99.9 | brief (c), see below |

**Brief (c), resolved against the owner before it was run.** The exposure table
is `feature_defs::DEFECT_F17`'s note, which states that Profile D *"has no
transform block at all and reads f116 (max 1,380) and f155 (max 2,127) raw"*.
The basic block is 13 signals per `(scale, channel)` cell at `13·cell + k`
(`def_at`: `scale = cell/3`, `channel = TRIPLE[cell%3]`), so f116 is cell 8 =
(scale 2, channel 2) and f155 is cell 11 = (scale 3, channel 2). Their
**immediate scale siblings are f77, f116 and f155 — every one of which is
already inside the twelve-slot F17 set**, so (c) as literally written degenerates
to (a). The nearest non-degenerate owner-grounded reading is run instead: the
twelve F17 slots plus the **full 13-signal basic cells** of the two exposed
slots, which adds `mse`, `var_loss`, `tex_loss` and the nine ssim/edge signals
at exactly the two (scale, channel) cells the SDR default reads raw. It also
fills in §12.6's scope curve between `W-f17` (12 slots) and `W-all` (372).

#### 12.7.4 Grading — identical to §12.6

Paired-bootstrap rank (B = 2,000, seed 20260905, same index sets both sides) on
CID22, KonJND, AIC-3, TID, KADID, CSIQ, LIVE; G-ADDR contract 6/6 and A7r
resolvable per-codec floors; two-reference inversions; identity; the LIVE
outlier-row ordering on the SAME 60-row subset cut once at revision 1; and one
new axis the brief requires:

* **HEALTHY-CELL PERTURBATION** — how many of the 196,086 × 12 (or × 36)
  training-leg cells the guard actually clamps, per arm, with the fraction of
  ROWS touched. A guard is only cheap if it moves the tail and nothing else;
  this is the number that says so, and it is what separates the three
  percentiles.

#### 12.7.5 What this lane does NOT do

No re-extraction, no era break, no feature-definition change, no perf claim, no
install. `W-f17-refit` reads on the revision-1 era root because its features ARE
revision 1. Confounds inherited unchanged from §12.4 (the CID22 `image`-crate
decoder, the 504-row KonJND ruler, KADID/TID train==val on this lineage).
The single-fit CI caveat of §12.6 applies identically: these are deterministic
convex solves with no seed, so the CIs mean what they say.
