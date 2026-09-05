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
