# Subset-quality study — what a "good" training subset looks like, and whether it explains seed spread

**Status: PRE-REGISTERED 2026-09-04. Hypotheses, descriptors, regression form and
falsification below were written BEFORE any descriptor→score analysis was run.**
The only measurements taken before locking this file were *mechanism* reads of the
trainer source (which RNG stream drives what) and a **power probe** sizing the
variance of the coverage descriptors under the real wave-r4 group structure — a
design input, not an outcome. Section 3 states what that probe was allowed to
change (grid/window sizing) and what it was not (hypotheses, targets, falsifier).

Lane: `claude-subsets`. Workspace `~/work/zen/zensim--subsets`.
User direction (2026-09-04): *"log good seeds and the subsets they find, across
multiple models, and figure out what good diversity/subsets look like and make
such more intentionally."*
Coordinator amendment (2026-09-04): *"init seed and sampling should be different
seeds"* — see §6.

---

## 1. Mechanism (read from source, not assumed)

`zensim-validate/src/mlp_train/mod.rs` runs **two independent RNG streams**, both
`SplitMix64`, both derived from the single `--seed`:

| stream | construction | drives |
|---|---|---|
| init | `SplitMix64::new(seed)` | `w1`/`w2` He-normal init |
| **sample** | `SplitMix64::new(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(0xDEADBEEFCAFEBABE))` | every RankNet pair draw |

(the 2-layer / min-max entry point at `mod.rs:6873` uses
`.wrapping_add(0x0123456789ABCDEF)` instead; `zensim-train-core`'s three legacy
heads use `seed ^ 0x5A5A…` / `seed ^ 0xA5A5…`.)

The separation is deliberate and documented in-source: it exists so a 228-vs-372
A/B sees the *same* pair draws even though init consumes a different number of
normals. **Consequence for this study: sampling is already decoupled from init in
the RNG, but not in the CLI — one `--seed` sets both, so every existing bake
confounds them.** Existing bakes are therefore attributed JOINTLY (§6).

The per-draw sequence (identical at `mod.rs:2265`, `3169`, `4003`; `7709` adds a
stratified variant) is:

```
u = rng.next_f64_unit()                       // 1 draw
train_pos = cdf.partition_point(|c| c < u).min(len-1)
n = groups[train_indices[train_pos]].features.len()
if n < 2 { continue }                          // consumes 1 draw
(ia, ib) = within_ref ? rb.draw(next_u64 ×3)   // 3 draws
         : per_row_cdf ? (next_f64_unit ×2)    // 2 draws
         : (next_u64 % n, next_u64 % n)        // 2 draws
if ia == ib { continue }
```

`train_indices` = groups with `train_weight > 0.0`, **in declaration order**;
`cdf` = cumulative normalised `train_weight`.

**Therefore the drawn multiset is a pure function of
`(seed, [train_w], [rows], epochs, pairs_per_epoch, boost flags, within_ref flags)`**
— and every one of those is recorded in the fulleval's embedded `zentrain.repro`
block (`inputs[].rows` is written as `g.human_scores.len()`, which is exactly the
`n` the draw uses). **No feature matrix is needed to reconstruct a subset.**

## 2. Population

312 board fullevals carry a usable `repro.argv` + `seed`. Canonicalising argv with
the seed value and `--out` elided gives **124 distinct arms, 83 of which have k≥2
seeds** (largest k=12, then 9, 9, 6, 6, 5, 5, 5, then twenty-two k=3 arms).
Model classes covered: the 944-class sota944 arms (`C_em944`, `C_co3a`, `C_nt944`,
`FS_K944`, `Q_hdr944`), the W8–W12 balance waves, the HDR944 arms, the
fastclass 156+free wave (`FC_C0/D1..D4/F1/G1` ×3), and the wave-r4 A-arms.

## 3. What the power probe was allowed to decide

Under the real wave-r4 structure (10 train groups, 708,301 train rows,
120 × 50,000 = 6,000,000 draws → ≈17 row-hits per row) whole-run coverage is
**expected a priori to saturate**, which would make binary coverage descriptors
degenerate and the study unfalsifiable-by-construction. The probe measures that
saturation so the descriptor set can be sized honestly. It may change:

- the **window** over which coverage descriptors are computed (whole-run vs
  epoch-1 vs first-N-draw prefixes), and
- whether multiplicity-balance descriptors replace binary-coverage ones.

It may **not** change the hypotheses, the targets, the regression form, or the
falsifier below. If whole-run coverage is saturated, that is reported as a
**result** (H1), not quietly dropped.

## 4. Hypotheses (pre-registered)

- **H1 (degeneracy).** At production settings, whole-run subset coverage is
  saturated: every coverage descriptor's between-seed relative spread within an
  arm is < 1 %. *If true, whole-run coverage cannot mechanically explain a
  seed-to-seed CID22 spread of order 0.01–0.05 SROCC, regardless of any
  correlation observed.*
- **H2 (early-window).** Coverage/balance measured over the first epoch (50,000
  draws, ≈14 % row coverage) carries real between-seed variance and predicts
  held-out score with |ρ| > 0.3 and consistent sign across ≥2 model classes.
- **H3 (balance).** Among balance descriptors (per-group share vs declared
  weight; entropy and CV of per-row multiplicity; near-threshold band share),
  at least one predicts held-out score with |ρ| > 0.3 and consistent sign across
  ≥2 model classes.
- **H4 (decomposition, needs §6 split seeds).** The seed-to-seed spread in
  held-out score decomposes into an init component and a sampling component. The
  pre-registered directional guess is **init ≫ sampling**.
- **H5 (lucky seeds).** Seeds that are outliers-high within their arm
  (`A4b_s4004`; the `a4bkon` K-arm best seeds) do **not** have descriptor vectors
  distinguishable from their siblings' (i.e. luck is not coverage).

## 5. Descriptors, targets, regression, falsifier

**Descriptors**, per (model class, arm, seed), computed on the whole run AND on
the epoch-1 prefix:

| # | descriptor |
|---|---|
| D1 | distinct rows touched / total train rows (per group, and pooled) |
| D2 | distinct refs touched / total refs (groups with `ref_ids`) |
| D3 | distinct (ref, quality-band) cells touched / total cells |
| D4 | per-group pair share vs its declared normalised `train_w` (L1 deviation, and χ²) |
| D5 | Shannon entropy of the per-row multiplicity distribution, normalised by log n |
| D6 | CV of per-row multiplicity (pooled and per group) |
| D7 | near-threshold band share — fraction of drawn pairs whose max endpoint score ≥ 90 (the KonJND-relevant zone) |
| D8 | within-image vs cross-image pair ratio |
| D9 | duplicate-pair rate — fraction of draws repeating an already-drawn (g, ia, ib) |
| D10 | identical-index skip rate (`ia == ib`) — a pure RNG-luck control that *should* carry no signal |

**Targets** — genuine holdouts only: `rank.cid22.srocc_per_ref_mean` (within-image
CID22, the product-relevant form), `rank.cid22.srocc_signed`, `rank.konjnd`,
`rank.aic3`. **KADID and TID are excluded** — they are train==val on these arms
and would reward memorisation, not skill. CID22 MOS is never a training target
anywhere in this study.

**Regression.** Seed is the unit. Within arm: Spearman ρ of each descriptor
against each target (owner: `zenstats` via `zensim_validate::panel` / the `panel`
binary — no hand-rolled stats anywhere). Pooled across arms: rank-transform both
descriptor and target **within arm** (removing arm fixed effects by construction,
which is the only defensible pooling at k=3), then Spearman on the pooled
residual ranks. Report per-class as well as pooled, plus effect size in target
units per descriptor unit from a within-arm OLS on the raw values.

**FALSIFIER (pre-registered).** *If no descriptor predicts held-out score across
≥2 model classes with |ρ| > 0.3 and consistent sign, subset coverage is not the
driver of seed spread.* On that outcome the study reports the null, lands the
registry and the stratified sampler, and does **not** ship coverage-targeted
steering (§7c is skipped, by pre-registration, not by choice after the fact).

## 6. Split seeds — coordinator amendment

`--seed` currently sets both streams. The `claude-ownerfix` lane is splitting it
into `--init-seed` + `--sample-seed` (backward-compatible: bare `--seed N` stays
byte-identical). Consequences adopted here:

1. **Every existing bake's single seed drove BOTH** init and sampling. Phase-2
   findings are therefore attributed **jointly**; no Phase-2 result can separate
   the two components, and none will be claimed to.
2. **The Phase-3 pilot uses the split flags** to decompose: replay the
   best-covered subset under 3 init seeds (sampling fixed) AND hold init fixed
   under 3 sample seeds. Spread within the first triple is the **init component**;
   within the second, the **sampling component**. This is the study's cleanest
   deliverable and the direct test of H4.
3. **The registry keys on the SAMPLE seed / sample-sequence digest**, with the
   init seed recorded as a separate field.

## 7. Deliverables

- (a) `benchmarks/good_subsets_registry.json` — append-only, schema header; per
  (model class, arm, sample seed) the sample-sequence digest + descriptor vector
  + held-out scores + init seed. Small by construction; the full descriptor
  tables live under `/mnt/v/output/zensim/subset-study-2026-09-04/` with a
  committed pointer.
- (b) A prescription: the descriptor profile of the top-quartile subsets per
  class, as a target coverage spec — **or the explicit null**.
- (c) `--pair-sampling stratified`, and coverage-targeted steering **only if the
  §5 falsifier does not fire**.
- (d) The pilot: fastclass `C0` recipe, seed-split triples per §6.2.

## 8. Owner discipline

The re-simulation must not become a second sampler. The draw step is extracted
into ONE owner used by the training loops **and** the simulator; a bake
byte-identity gate plus a sample-sequence digest prove the extraction changed
nothing. Any descriptor stat goes through `zenstats`. Default paths stay
byte-identical.
