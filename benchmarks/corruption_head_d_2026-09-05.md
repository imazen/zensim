# Profile D's companion corruption head, rebuilt at the RUNTIME extraction era (2026-09-05)

**Lane:** `claude-corrhead-2026-09-05`, jj sibling workspace `~/work/zen/zensim--corrhead`.
**Artifacts:** `/mnt/v/output/zensim/corruption-head-2026-09-05/`.
**Question this answers:** the fair gauntlet flags shipped Profile D
(`d_sdr_add156_id100_negrich_dial_2026-09-05.bin`, cell `d_id100_negrich@did100lane`)
on the corruption axis — `pass_q20` **0.269** / `pass_q10` **0.152**, against
`peer_ssim2` 0.345 / 0.201, shipped B 0.188 and Profile A 0.196.

The registered design (`corruption_head_2026-07-24.md`, CLAUDE.md "corruption HEAD")
does not ask the dial to win that axis — it makes a **companion head** the owner.
D had none (`corruption_head: null` on its board cell). This lane builds one at the
era D actually runs at, and wires it in.

---

## 1. What existed, and why none of it was usable as-is

Five findings, each measured, in the order they blocked the work.

**1.1 The corruption pixels were never persisted — by design — but they ARE
reproducible.** `build_corruption_corpus.py` streams generate → extract → `rmtree`,
because a corruption is a pure function of `(ref_id, seed, params)`. Confirmed in
the generator: `corruption_corpus` takes `--seed` (default 1) and drives its own
`prng.rs`. So the corpus is a **rebuild**, not a recovery.

**1.2 The 2026-07-24 sources point into the QUARANTINED imazen-26 tree.**
`sources.tsv` names 174 paths under `/mnt/v/imazen-26/`, which the 2026-08-27 user
directive rules is the *inspiration* collection, never the canonical corpus
(canonical = `/mnt/v/output/imazen-26-png-v3` + `imazen26_manifest.tsv`, 4-digit
ids). The directory was renamed to `/mnt/v/imazen-26-inspo`, so **154 of 174 paths
are dead** today; all 174 resolve after the one-token rewrite. This lane rebuilds
on the SAME 174 (rewritten) so the era comparison is an era comparison and nothing
else — **rebuilding on the canonical population is a separate, registered change**,
not folded in here.

**1.3 The 2026-07-24 build never finished.** Its log ends `rc=1` after **5,420 s**
with `ValueError: 'f627' is not in list` — a truncated extractor CSV — having
written **141 of 174** refs. The record's "(142/174 refs; 32 skipped)" is that
failure. This run completes **174/174 with zero skips**; the truncation class is
now impossible because the extractor aborts loudly on any undecodable pair and
`run_extract()` is the one place that knows each extractor's argv.

**1.4 `bake_verdict --corruption-head` takes a ZNPR BAKE, never the JSON head — so
no 372 head has ever been through the gate.** `--corruption-head` calls
`Model::from_bytes` and checks `head.caller_input_width() == grid.n_features`. The
2026-07-24 artifacts are `.json` (weights + isotonic + a deploy formula). The only
corruption-head *bakes* on disk are `corrhead944_s{13,42}.bin` — 944-wide MLPs
(493 KB, `layer0 944→128`) belonging to Profile C. The registered 372 design and
the wired-up evaluator have never met.

**1.5 The 2026-07-24 head is not a D companion at any price.** D's block profile is
`basic@w372/unknown#1008b687` — **28 of 156 basic lines used, 0 of 216 pool lines**.
Its walk therefore runs `V1PoolsMode::Peaks`, and `fold_engine.rs` states the cost
rule explicitly: *"`Off` and `Peaks` cost the SAME to compute (the peak
accumulators are the fused V-blur kernel's unconditional L8/max tier); the real
compute boundary is `{Off, Peaks}` vs `Full`."* So for D:

| slice | cost at D's runtime |
|---|---|
| `f0..155` (basic) | **free** — already computed |
| `f0..227` (basic + peaks) | **free** — same `Peaks` mode, superset of emitted slots |
| `f228..371` (masked / IW) | **NOT free** — forces `V1PoolsMode::Full` |

The 2026-07-24 head reads all 372, and its own ablation puts the top features
across *"basic `f17-21` at the finest scale + mask/iw/peak `f255-334`"* — i.e.
squarely in the block D does not compute. A D companion has to be re-fit on
`f0..155` or `f0..227`, and whether that costs detection is a real open question
this lane answers.

---

## 2. The era gap, measured twice

The shipped runtime is one extraction era ahead of every stored 372 table:
`56bbcda2` (option C — v1 stops pooling phantom columns, 2026-08-30 15:43) landed
after the default 372 eval root was built (`ea16c7ee`, 13:21 the same day). Both
2026-07-24 tables are older still.

Re-extracting the same pixels at HEAD, cell-over-tolerance at
`max(1e-6, 1e-5·|stored|)`:

| table | basic `f0..155` | peaks | masked | IW | rows moved |
|---|--:|--:|--:|--:|--:|
| corruption corpus, ref `gen-line-concentric…` (1024²) | **51.8 %**, max \|Δ\| 0.336 | 41.6 % | 66.9 % | 68.9 % | 664 / 674 |
| the gate grid `corruption_grid_372col_2026-05-28` | **73.7 %**, max \|Δ\| 4.35 | 53.5 % | 82.1 % | 84.2 % | 2013 / 2016 |

Both are exactly the padded-width class option C removed (1024 and 512 both take
`simd_padded_width`'s extra 16). **So D's published corruption numbers are read on
a grid one era behind the runtime**, and a head fit on the 2026-07-24 tables would
be scoring a different quantity than the product produces.

**But the era is NOT the explanation for D's score.** Re-extracting the gate grid at
HEAD (`corruption_grid_372col_postC_2026-09-05.parquet`, same 2,016 persisted PNGs,
0 failures) and re-running D's dial:

| grid era | D `pass_q20` |
|---|--:|
| `corruption_grid_372col_2026-05-28` (stored) | **26.9 %** |
| `corruption_grid_372col_postC_2026-09-05` (HEAD) | **26.8 %** |

The features move on 73.7 % of basic cells and the *ordering* does not. **D's
corruption weakness is intrinsic, not an era artifact** — which is what makes a
companion head the fix rather than a re-extraction.

One more property of that instrument, worth stating because it bounds every number
read on it: the gate grid is **672 triples from ONE reference** (`gb82_dog`) — the
single-source limitation the 2026-07-24 record itself calls out. Nothing in this
lane's training corpus comes from `gb82`, so the head's gate numbers are
source-held-out by construction.

---

## 3. What was rebuilt, and what it cost

| table | rows | note |
|---|--:|---|
| `im26_corruption_372_postC.parquet` | 116,928 corruption + 348 anchors | 173 ref_ids, 44 families, **0 skips** (2026-07-24: 141 refs, 95,424 rows, died rc=1) |
| `negrich_372_postC.parquet` | 60,000 | seed-0 sample of the 280,384, 48,067 distinct `source_id`; PNGs from **R2** — the LAN store has no `distorted/` prefix |
| `corruption_grid_372col_postC_2026-09-05.parquet` | 2,016 | the persisted `gb82_dog` gate PNGs, re-extracted |
| broad-honest = the ladder instrument | 9,593 | HONEST current-era imazen codec cells, floor-dense, already at HEAD |

Ran **locally under `run-heavy`**, not on the fleet: generation is the pole
(25 s/ref single-threaded) at 3,073 s wall for the corpus, while extraction runs
at 163 pairs/s on the ~1 MP corruption pairs and 1,013/s on the ~0.25 MP KADIS
pairs — 59 s for all 60,000 negrich rows. Total well under the 2 h fleet
threshold, and the R2 fetch (60,000 PNGs, 17 GB, 0 errors) is network-bound.

negrich moved on **100 %** of rows (basic 83.3 % of cells over tolerance), row
alignment gated on `source_id` equality across all 60,000.

## 4. Results

### 4.1 The head, on held-out sources

**⚠ Read the model form before reading the number.** `train_corruption_head.py`
reports its threshold curve from a `CalibratedClassifierCV` (isotonic, `cv=3`),
then persists a **different** model — a plain `LogisticRegression` refit on train
with an isotonic calibration fit on val. That has been true since 2026-07-24 and
means the published table has never described the artifact. Both are given below;
**the bake row is the one the product runs and the one every other number in this
document uses.**

As SHIPPED (the ZNPR bake, held-out fold, T = 0.9):

| arm | slice | free for D? | detection | FP severe-honest | FP ladder | FP matched anchor |
|---|---|---|--:|--:|--:|--:|
| `d156` | `f0..155` | yes | 84.9 % | 0.41 % | 11.94 % | 0.00 % |
| **`d228`** | `f0..227` | **yes** | **85.9 %** | **0.31 %** | **11.22 %** | 0.00 % |
| `d228nb` | `f0..227`, no codec negatives | yes | 95.6 % | 0.74 % | — | 20.93 % |
| `2026-07-24` | `f0..371` | **NO** | **90.7 %** | 0.31 % | 10.25 % | 0.00 % |

At T = 0.95: `d156` 75.8 / 0.22 / 6.92 / 0.00; `d228` 77.1 / 0.19 / 6.71 / 0.00;
`d228nb` 93.0 / 0.38 / — / 10.47; `2026-07-24` 85.2 / 0.22 / 2.87 / 0.00.

As REPORTED by the trainer (the `CalibratedClassifierCV`, same folds, T = 0.9) —
kept because it is what the 2026-07-24 record published: `d156` 88.1 / 0.36 /
15.22 / 1.72; `d228` 89.5 / 0.22 / 15.83 / 0.00; `d228nb` 94.5 / 0.58 / — / 18.60;
`2026-07-24` (its own record, different corpus) 84.6 / 0.06 / 0.34.

**Two conclusions, and the second corrects a claim this document made earlier.**

1. **Peaks are free and strictly better.** `f156..227` buys +1.0 point of
   detection and cuts severe-honest FP from 0.41 % to 0.31 % at literally zero
   extraction cost (`Off` and `Peaks` are the same walk). Any D companion should
   be the 228 slice, not the 156 one.

2. **Masked/IW is worth +4.8 points of detection — and is NOT free for D.**
   Bake-vs-bake on the identical held-out rows, the 2026-07-24 head reads
   **90.7 %** to `d228`'s 85.9 % at the same severe-honest FP (0.31 %). So the
   2026-07-24 ablation's conclusion — that the signal needs the mask/iw/peak
   block — is **SUPPORTED**, not refuted. An earlier draft of this document
   claimed the opposite by comparing that head's *reported* `clf` number (84.6 %)
   against `d228`'s *reported* `clf` number (89.5 %) across different corpora;
   that comparison was invalid in both respects and the corrected reading is the
   bake-vs-bake row above.

   So D's companion is a genuine trade, not a free lunch: **85.9 % at zero
   marginal extraction, or 90.7 % by forcing D's walk from `V1PoolsMode::Peaks`
   to `Full`.** Pricing that walk change is registered, not run.

### 4.2 The gate, wired (`bake_verdict --corruption-head`, postC grid, 672 triples)

| scorer | `pass_q20` | `pass_q10` |
|---|--:|--:|
| **D dial alone** | **26.8 %** | 15.3 % |
| head `d228` alone | 99.9 % | 99.7 % |
| head `d156` alone | 99.4 % | 99.3 % |
| 2026-07-24 head alone | 100.0 % | 99.9 % |
| **DEPLOY `min(dial, gate)`, `d228`** | **91.4 %** | 88.2 % |
| DEPLOY, `d156` | 92.4 % | 88.8 % |
| DEPLOY, 2026-07-24 | 94.6 % | 92.3 % |

So the registered design does what it claims: D's corruption axis goes from
**26.8 % to ~91 %** with a companion head, and the head is the owner of that axis
exactly as the design says.

### 4.3 …but the gate alone must not select the head

`d228nb` — the arm with **no** codec negatives — wins the deploy gate outright
(**99.1 % / 97.6 %**) because it is more trigger-happy. It is also the worst head
by a wide margin on honest content. **A head cannot be chosen on the corruption
gate**, whose only honest rows are two anchors from one reference.

Per-codec FP on honest current-era imazen codec output (`corruption_head_honest_fp.py`,
held-out ladder images only for the trained heads, all rows for the 2026-07-24 head
which trained on none):

| head | jpeg | webp | avif-svt | avif-rav1e | jxl | ALL |
|---|--:|--:|--:|--:|--:|--:|
| `d228` | 2.70 % | 2.12 % | 2.68 % | **27.15 %** | **22.22 %** | 11.22 % |
| `d156` | 4.66 % | 2.12 % | 2.98 % | 27.73 % | 23.11 % | 11.94 % |
| 2026-07-24 | 3.43 % | 4.88 % | 4.17 % | 21.88 % | 16.44 % | 10.25 % |

**And the FP is entirely at HIGH quality, which inverts the obvious worry.**
q-binned, `d228`, all ladder rows:

| codec | q0-5 | q5-20 | q20-50 | q50-80 | q80-95 | q95-100 |
|---|--:|--:|--:|--:|--:|--:|
| avif-rav1e | 0.0 % | 0.0 % | 0.0 % | 0.6 % | 41.8 % | **97.2 %** |
| jxl | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 0.6 % | 55.9 % |
| webp | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 6.7 % | 27.5 % |
| jpeg | 0.0 % | 0.0 % | 0.0 % | 0.0 % | 2.3 % | 18.9 % |
| avif-svt | 0.0 % | 0.0 % | 0.0 % | 0.9 % | 10.9 % | 14.5 % |
| **ALL** | **0.0 %** | **0.0 %** | **0.0 %** | 0.3 % | 13.0 % | **53.7 %** |

**1,134 of 1,139 flagged cells sit at q >= 80.** The head does not fire on
aggressive compression at all — it fires on **near-lossless** output, hardest on
avif-rav1e (97.2 % of its q95+ cells). The mechanism is visible in the corpus: a
small-region corruption (`sq8`, `sq16`) is *also* almost identical to its
reference, so in globally-pooled v1 features "nearly identical, with a little
localized structure" describes both a near-lossless encode and a tiny structural
break. Adding the ladder to training did **not** fix this — it is a separability
limit of the feature set, not a coverage gap.

### 4.4 A dial guard makes it loop-safe, at a measured price

Because the two populations DO separate on D's own dial — flagged honest cells sit
at dial p5/p50/p95 = **89.8 / 93.3 / 96.9**, flagged corruptions at **-53.5 / 81.2
/ 99.1** — gating only when the dial is also low recovers most of the win:

| head | guard `dial <` | deploy `pass_q20` | deploy `pass_q10` | honest-cell FP |
|---|---|--:|--:|--:|
| `d228` | none | 91.4 % | 88.2 % | 11.87 % |
| `d228` | 95 | 78.0 % | 74.9 % | 9.11 % |
| **`d228`** | **90** | **64.0 %** | 60.9 % | **0.74 %** |
| `d228` | 80 | 47.9 % | 44.8 % | **0.00 %** |
| `d228nb` | 80 | 53.1 % | 51.6 % | 20.88 % |
| D dial alone | — | 26.8 % | 15.3 % | 0.00 % |

At `dial < 90` the head still more than **doubles** D's corruption ordering
(26.8 % -> 64.0 %) while firing on 0.74 % of honest codec output; at `dial < 80` the
honest cost is zero and the gain is still 1.8x. The guard is a **proposal**, not
something this lane baked in — `bake_verdict`'s DEPLOY section implements the
registered unguarded `min(perceptual, gate)` and nothing else.

### 4.5 Runtime cost

**Extraction: zero marginal.** The head reads `f0..227`, which D's walk already
emits.

**Forward: <= 2.5 us per compare** — 60,000 rows minus 600 rows through
`predict_features_with_bake`, min of 5, and that bound still *includes* reading an
89 MB blob and formatting 60,000 output lines. Against D's own compare that is
**<= 0.03 %** at 576^2/1T and **<= 0.008 %** at 1152^2/1T.

The zenbench arm (`add156_plus_corrhead`, identical extraction and profile forward
to `add156_156basic` plus the head's forward, interleaved) **cannot resolve it**,
which is the expected result once the marginal is ~3,000x under the base arm's
spread. Reported for completeness, not as a measurement of the head:

| threads | size | `add156_156basic` | `add156_plus_corrhead` |
|---|---|--:|--:|
| 1 | 576^2 | 8.0 +-0.3 ms | 9.0 +-2.8 ms |
| 1 | 1152^2 | 33.2 +-2.3 ms | 35.2 +-4.2 ms |
| 8 | 1152^2 | 8.1 +-0.8 ms | 8.3 +-1.0 ms |

Every delta is inside its own error bar. The 8T/576^2 cell is **discarded as
degenerate** — `add156_156basic` read 59.9 +-6.1 ms there while
`add156_plus_corrhead`, which strictly contains it, read 4.9 ms; an arm cannot be
12x slower than its own superset. That is the failure mode
`benchmarks/profile_d_notax_2026-09-01.md` §4 documents, and it is why the
per-row bound above is the number to quote.

---

## 5. Proposed API shape (NOT implemented — no public API changed by this lane)

The head is bytes plus a dot product; the only design question is who owns the
composition rule and how the caller pays for features. Proposal, for the user to
accept or reject:

```rust
// zensim, behind a NEW default-OFF feature `corruption-head` (zero new deps —
// zenpredict is already in the graph for every bake-bearing profile).
pub struct CorruptionHead { /* zenpredict::Model + the baked deadband */ }

impl CorruptionHead {
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, CorruptionHeadError>;
    /// The caller_input_width this head demands, so a mismatch is a compile-time
    /// -ish contract rather than a silent prefix read.
    pub fn caller_input_width(&self) -> usize;
    /// P(corruption) in [0, 1] from a feature vector the score pass ALREADY built.
    pub fn probability(&self, features: &[f32]) -> Result<f32, CorruptionHeadError>;
    pub fn deadband(&self) -> f32;                 // the baked recommended T
}

/// THE composition rule, in one place. `bake_verdict`'s DEPLOY section must call
/// this same function so the shipped behaviour and the reported gate cannot drift
/// (that drift is exactly how bake_verdict's inline stat copy reported the wrong
/// OR + PWRC for months — see CLAUDE.md "NO DUPLICATE IMPLEMENTATIONS").
pub fn gate_score(perceptual: f32, p_corruption: f32, deadband: f32) -> f32 {
    if p_corruption > deadband { perceptual.min(0.0) } else { perceptual }
}
```

Four constraints that are NOT negotiable, each for a measured reason:

1. **Never on `Zensim::compute`'s default path.** D's whole value is that it is the
   cheap profile; a second forward pass belongs behind an explicit call, not inside
   the function every product path already calls.
2. **The head's block profile MUST be a subset of the profile's.** If a 372-layout
   head that reads `f228..371` is attached to D, `score_pool_mode` is forced from
   `Peaks` to `Full` and the extraction cost changes — silently, since nothing in
   the type system connects the two. A `debug_assert`/`Result` that compares the
   head's read-set against the profile's is cheap insurance. This is precisely the
   trap the 2026-07-24 head would have walked into.
3. **The bytes ship beside the profile, never inside a codec** (per
   `feedback_no_zenpredict_in_codecs`). `zensim/weights/` is the natural home; this
   lane installs NOTHING there.
4. **Two bakes, one fit.** The corruption GRID and the 372 eval root speak the v1
   372 layout; the runtime fold emits the 944 layout. `f0..155` (basic) and
   `f156..227` (peaks) are at the same indices in both, so the emitter writes the
   same coefficients at both caller widths rather than fitting twice. Whichever
   layout the caller holds, one of the two bakes accepts it and neither is a
   reinterpretation of the other's bytes.

---

## 6. Verdict, and what is NOT done

One bookkeeping note for anyone reading D's board cell: the grafted
`corruption_head` block was measured on the **postC** grid while the cell's
`corruption` (dial) block is the board's original **stored-era** read. Both are
measured here and they differ by 0.1 point (26.9 % vs 26.8 %); the graft leaves
every other key byte-identical by design.

**The companion-head design is validated for D and the head is built, baked and
wired** — `bake_verdict --corruption-head` now reports three sections (dial alone,
head alone, and the registered `min(perceptual, gate)` composition) and D's board
cell carries a `corruption_head` block for the first time.

**It is not yet loop-safe unguarded**, and the reason is specific and measured:
53.7 % of near-lossless honest cells false-fire, because globally-pooled v1
features cannot separate "near-lossless encode" from "corruption confined to an
8x8 square". The `dial < 90` guard is the cheap mitigation (0.74 % honest FP,
still 2.4x D's dial alone) and wants a user decision before it becomes the
composition rule.

**Registered, NOT run:**

1. Rebuild the corruption corpus on the CANONICAL imazen-26 (`imazen-26-png-v3` +
   `imazen26_manifest.tsv`), replacing the quarantined inspo tree. This lane kept
   the old population deliberately so the era comparison stayed clean.
2. Replace the PIL Lanczos reference downscale and the `image`-crate q10/q20
   anchors with `zenresize` / `zenjpeg`, per IMAZEN-ONLY. Both change the pixels,
   so this is its own change with its own before/after.
3. A multi-source gate grid. 672 triples from one reference is the whole
   corruption instrument today, and every gate number in this document inherits
   that.
4. Region-localized corruption features. §4.3 is a separability limit, not a data
   gap — more negatives will not fix it, and the head's own confusion (tiny local
   break vs near-lossless) names exactly the feature that is missing.
5. Price the `Peaks -> Full` walk change. §4.1 shows masked/IW is worth +4.8
   points of detection; `fold_engine.rs` says it costs a mode change, and nobody
   has measured that delta for D's walk specifically. That measurement decides
   whether D should carry a 372 head after all.
6. Make `train_corruption_head.py` report the model it ships. It has published a
   `CalibratedClassifierCV`'s numbers while persisting a `LogisticRegression` +
   isotonic since 2026-07-24; the gap is 4.6 points of ladder FP on `d228`. Left
   as-is here (both forms are reported in §4.1 and every other number in this
   document comes from the bake) because changing it re-dates every published
   head number.
7. The `dial < G` guard as a first-class composition, if the user wants it. It is
   measured here and implemented nowhere.

**Nothing was installed in `zensim/weights/`** and no public API changed.
