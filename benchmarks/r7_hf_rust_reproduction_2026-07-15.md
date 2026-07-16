# Round-7 HF near-lossless — Rust-trainer reproduction (2026-07-15)

Answers the standing question *"did you try a bake including it using the
optimal pairwise/per-ref"* and the directive *"ensure key bakes are
reproducible with the rust paths and update rust when needed to do so."*

**Verdict: the HF FINDING ITSELF reproduces under two different training
objectives — that is the load-bearing result and it is solid (HF per-ref
+0.2539 @ 35 % backwards → +0.9500 @ 0 % backwards).**

**The COST numbers do not survive a seed sweep, and an earlier revision of this
doc wrongly claimed they did.** Corrected 2026-07-15 eve:

- **CID22 (−0.005): RETRACTED as noise.** The delta flips sign across seeds
  (−0.0047 at seed 1, +0.0012 at seed 7); Python's −0.0041 sits inside that
  spread. Two draws from one noise band agreeing is not a reproduction.
- **Non-photo (−0.002): now UNVERIFIED.** Never seed-swept; the same
  ~0.004-scale objection applies. Do not cite it either.
- **KonJND (−0.03): NOT attributed, and not the HF group's doing.** Rust's
  KonJND *level* is uniformly ~0.05 below Python's in every arm, both
  objectives, all three seeds — a systematic offset that exists before the HF
  group is added. The objective hypothesis was raised, tested, and falsified.

The lesson is the one this repo already has a rule for: a single seed cannot
measure a 0.004 effect, and reporting an unswept delta as a reproduction is the
"post-selection small-n" family in `docs/DATASET_HISTORY.md` §0.

## What was run

| | Python round-7 (`blend_search.py` ROUND7) | Rust (`zensim_mlp_train`) |
|---|---|---|
| main groups | `safesyn 1, bigcodec 1.5, kadis 0.3` (`_HON`) | same |
| HF group | `hf_rank_weight 0.1`, within-ref RankNet | `hf_nearlossless:…:0.1:0.0:withinref` |
| arch | 2-layer, H=128 | 2-layer, H=128 (`--n-hidden-layers 2 --hidden 128`) |
| **main-group loss** | **`smooth_l1_loss` (Huber — absolute target)** | **RankNet (`mse_weight: 0.0`)** — and, after per-group loss mode landed, an `:mse` re-run matching Python; both reported below |
| **seeds** | **2 (1, 7), averaged** | **1** |
| epochs | 400 full-batch | 120 × 50k sampled pairs |

Rust command (from `/mnt/v/output/zensim/r7_rust/train_hf01.log`):

```
zensim_mlp_train --out r7_hf01.bin \
  --group safesyn:canonical-2026-05-21/train/safesyn.parquet:1.0:0.0 \
  --group bigcodec:bigcodec_hqdedup_traindigits_2026-07-02.parquet:1.5:0.0 \
  --group kadis:canonical-2026-07-15/train/kadis_negrich.parquet:0.3:0.0 \
  --group hf_nearlossless:canonical-2026-07-15/train/hf_nearlossless_train.parquet:0.1:0.0:withinref \
  --group cid22_val:canonical-2026-05-21/train/cid22_train.parquet:0.0:1.0 \
  --target-column human_score --target-scale 100 \
  --hidden 128 --n-hidden-layers 2 --epochs 120 --seed 1
```

`hf0` is the identical command with the `hf_nearlossless` group removed.

## Split integrity (checked, not assumed)

`hf_nearlossless_train` = 900 rows / **150 refs**; `hf_nearlossless_val` =
300 rows / **50 refs**; `train ∩ val = 0 refs`; `train ∪ val` is exactly the
200-ref parent. The held-out readout is honest.

Per-ref n = **6** for every held-out ref (a 6-rung ladder). SROCC on 6 points
is coarse — read `%bwd` (share of refs ranked backwards) as the primary, since
it is robust to that.

## Result — the HF axis

| HF near-lossless, 50 held-out refs | hf0 | hf0.1 |
|---|---:|---:|
| **per-ref SROCC** | +0.2539 | **+0.9500** |
| **% refs ranked BACKWARDS** | **35%** | **0%** |
| PLCC | 0.3184 | 0.6275 |
| PWRC | 0.6510 | 0.8241 |
| Z-RMSE | 0.948 | 0.779 |
| DS-AUC | 0.5073 | 0.5842 |
| pooled SROCC | 0.2093 | 0.2810 |

A 900-pair group at weight 0.1 takes held-out near-lossless ordering from *a
third of refs backwards* to *none*. The pooled-vs-per-ref gap (0.281 vs 0.950)
is the documented within-vs-cross-image confound — the HF ladder moves ~0.92
ssim2 pts within an image vs ~6 pts between images, so the pooled number is
carried by cross-image scale. It is not a defect, and it is exactly why the
group is consumed rank-only and within-ref.

### The HF group helps under BOTH objectives — but the baseline is not fixed

Once per-group loss mode existed, the same experiment ran under Python's
objective. All four arms, HF axis (50 held-out refs):

| recipe | HF per-ref | %bwd | CID22 | KonJND | non-photo |
|---|---:|---:|---:|---:|---:|
| rank-only, hf0 | +0.2539 | **35%** | 0.8842 | 0.5024 | 0.9669 |
| rank-only, hf0.1 | **+0.9500** | **0%** | 0.8795 | 0.4675 | 0.9653 |
| mse-recipe, hf0 | +0.6393 | 6% | 0.8745 | 0.4932 | 0.9666 |
| mse-recipe, hf0.1 | +0.7262 | **0%** | 0.8714 | 0.4618 | 0.9651 |

The HF group drives %bwd to 0 under both objectives, so the finding is robust
to the recipe. But the *size* of the win is not: **the objective, not the HF
corpus, is what most of the "35% backwards" was measuring.** An absolute target
alone — with no HF data whatsoever — already gets near-lossless ordering to
+0.6393 / 6% backwards, versus rank-only's +0.2539 / 35%. So the headline
"a third of refs ranked backwards" is a property of a *pure-rank* objective,
not of a model that lacks near-lossless data.

That reframes the corpus's value: it is worth +0.09 per-ref and the last 6% of
backwards refs on top of an absolute target, not +0.70. Still a real gain (and
0% backwards is the number that matters for a near-lossless dial), but an order
of magnitude smaller than the rank-only comparison implies. Rank-only + HF
remains the best HF number measured (+0.9500), at the cost of having no
absolute dial at all.

## Result — the cost

| Δ (hf0 → hf0.1) | Python r7 | Rust r7 | |
|---|---:|---:|---|
| CID22 | −0.0041 | **−0.0047** | reproduces (within 0.0006) |
| non-photo | −0.0017 | **−0.0016** | reproduces (within 0.0001) |
| KonJND | **+0.0334** | **−0.0349** | **does NOT reproduce** |

Absolute values — Python `hf0`: CID22 0.8862, non-photo 0.9549, KonJND 0.5187;
`hf0.1`: CID22 0.8821, non-photo 0.9532, KonJND 0.5521. Rust `hf0`: CID22
0.8842, non-photo 0.9669, KonJND 0.5024; `hf0.1`: CID22 0.8795, non-photo
0.9653, KonJND 0.4675.

Two of the three cost axes reproduce to within 0.0006. KonJND is the lone
outlier. The obvious suspect was the objective mismatch in the table above —
that suspect has since been tested and cleared; see below.

## Why KonJND does not reproduce — hypothesis raised, then FALSIFIED

**First hypothesis (WRONG — recorded because it was published, then killed by
measurement).** The two runs optimized different objectives: `mse_weight` was
wired only inside `train_mlp_per_sample_alpha_head`, so the standard
`train_mlp` path this run used was RankNet-only by construction, while Python
trains its main groups on `smooth_l1` and adds rank only for HF. KonJND is the
corpus most sensitive to absolute calibration (its `human_score` is a PJND
threshold, not a ladder position), so a pure-rank objective plausibly moved it
differently.

**Test.** Per-group loss mode was built precisely to remove this confound
(`GroupLossMode::{Rank, Mse, Both}`, commit `9af0142b`), plus the polarity
reconciliation the first attempt needed (`7e7f49c8`). The Python recipe is now
expressible verbatim: `safesyn/bigcodec/kadis:…:mse` + `hf:…:withinref,rank` +
`--mse-weight 1.0`.

**Result — the hypothesis is dead.** With the objective matched, KonJND STILL
moves the wrong way:

| Δ (hf0 → hf0.1) | Python r7 | Rust rank-only | Rust MSE-recipe |
|---|---:|---:|---:|
| CID22 | −0.0041 | −0.0047 | −0.0031 |
| non-photo | −0.0017 | −0.0016 | −0.0015 |
| KonJND | **+0.0334** | −0.0349 | **−0.0314** |

Matching the loss recipe moved KonJND's delta by 0.0035 — it did not flip it.
So the objective is **not** the explanation. Two of three cost axes reproduce
under *both* objectives; KonJND reproduces under neither.

**The seed sweep ran (2026-07-15 eve).** Rank-only recipe, seeds 7 and 17
against the seed-1 baseline. It settled two things and **retracted one of this
doc's own claims**:

| arm | seed 1 | seed 7 | seed 17 | Python r7 (2-seed avg) |
|---|--:|--:|--:|--:|
| CID22 hf0 | 0.8842 | 0.8842 | 0.8840 | 0.8862 |
| CID22 hf0.1 | 0.8795 | 0.8854 | *(pending)* | 0.8821 |
| **CID22 Δ (hf0.1 − hf0)** | **−0.0047** | **+0.0012** | — | **−0.0041** |
| KonJND hf0 | 0.5024 | 0.4944 | 0.4978 | 0.5187 |
| KonJND hf0.1 | 0.4675 | 0.4843 | *(pending)* | 0.5521 |
| **KonJND Δ** | **−0.0349** | **−0.0101** | — | **+0.0334** |

**RETRACTED — the CID22 delta is seed noise, and this doc previously cited it
as a reproduction.** It flips sign across seeds (−0.0047 at seed 1, +0.0012 at
seed 7). Python's −0.0041 sits inside that spread. So "Rust's −0.0047
reproduces Python's −0.0041" was never a measurement of the HF group; it was
two draws from the same noise band agreeing by luck. The earlier
non-photo agreement (Rust −0.0016 vs Python −0.0017) has NOT been
seed-swept and is now equally suspect — treat it as unverified until it is.
One seed cannot measure a ~0.004 effect on this corpus.

**KonJND is NOT seed noise.** The delta stays negative at both seeds
(−0.0349, −0.0101 — a 3.5× magnitude swing but no sign flip) while Python's is
**positive** (+0.0334). With n=2 that is suggestive rather than decisive on the
delta alone, but it does not stand alone:

**The level offset is the real signal.** Rust's absolute KonJND is uniformly
BELOW Python's — 0.4675–0.5024 vs 0.5187–0.5521 — in **every arm, both
objectives, all three seeds**. A ~0.05 gap that survives the objective change,
the HF group's presence, and the seed is not the HF group's effect and not
variance. Something systematic differs in how the two trainers handle KonJND.
That, not the HF group, is where the remaining suspects point (batching: Rust
120 epochs × 50k sampled pairs vs Python 400 full-batch; LR schedule: Rust
decays 0.00100 → 0.00068, Python holds 1e-3).

**So `KonJND −0.03` remains NOT attributed to the HF group** — and the sweep
narrowed why: the disagreement lives in the KonJND level, which is already
wrong before the HF group is added.

## Honest scope of the HF win

The 50 held-out refs are held out **by reference**, but share the corpus's
distortion family and its 6-rung ladder design. So the result says: *the model
generalizes this ladder's ordering to unseen images.* It does **not** show that
near-lossless ordering improved anywhere else — CID22/KonJND/non-photo all moved
slightly down or sideways. No broad near-lossless transfer is claimed or
measured here.

Measuring transfer needs a near-lossless-specific readout on the other corpora.
`blend_lib.panel` has one (`srocc_hightail` / `srocc_lowtail`, added
2026-07-15); `zenstats` and `bake_verdict` do **not**. That is a Python-only
stat which, per the standing directive, belongs in `zenstats` — queued.

Neither bake is a ship candidate: both are raw rank output with no calibration
spline (G1 dynamic range 0.00, p5=−16.1 p95=2.5). They are research bakes.

## Gap closed (and a bug it exposed)

`within_ref` was per-group but `mse_weight` was global AND rejected by the
plain path, so the Python recipe was not expressible in Rust. Closed by
`GroupLossMode::{Rank, Mse, Both}` (`9af0142b`), selected per group via the
5th spec field:

```
--group safesyn:PATH:1.0:0.0:mse  --group hf:PATH:0.1:0.0:withinref,rank  --mse-weight 1.0
```

Two defects surfaced doing it, both worth recording:

1. **The guard meant to prevent this class of mistake was dead code.** It sat
   inside `train_mlp_per_sample_alpha_head` testing `!per_sample_alpha_head` —
   unreachable there by construction. Its doc ("trainer panics if set on other
   heads") was false: `--mse-weight` on a non-α head silently trained pure rank
   and discarded the flag. Caught by a `should_panic` test that did not panic.
   AUDIT: 0 of 142 weight manifests are affected — every one that sets
   `mse_weight=0.6` also sets `per_sample_alpha_head = true`, so the
   uselessness was latent and no shipped bake was ever mistrained by it.

2. **The rank and absolute terms trained OPPOSITE polarities** (`7e7f49c8`).
   Rust's legacy RankNet is distance-shaped (higher quality → lower y);
   regression onto `human_score` is score-shaped. Mixed, they fight, and the
   rank group's own corpus inverts: the first MSE-recipe run scored HF at
   **−0.3454 per-ref / 75% backwards** — adding rank supervision to a corpus
   made that corpus rank backwards. Python has no such conflict (its RankNet
   is `BCE(s_i − s_j, 1 if quality_i > quality_j)`, agreeing with its
   `smooth_l1`); Rust now flips the rank target when any group carries the
   absolute term. Rank-only runs keep the legacy distance convention
   bit-for-bit, which matters because every shipped bake's calibration assumes
   it.

Note what caught (2): only the held-out **per-ref / %bwd** readout. Pooled
SROCC could not — the panel reports |SROCC|, so the inverted bake read 0.4110,
a mediocre number rather than an alarming one. This is the §8.39 failure mode
that no pooled or per-band stat can see.

## Reproduce

```
cargo build --release --bin bake_verdict -p zensim-validate
./target/release/bake_verdict --bake /mnt/v/output/zensim/r7_rust/r7_hf01.bin \
  --corpora hf_nearlossless,cid22,konjnd,nonphoto \
  --output /mnt/v/output/zensim/r7_rust/v2_hf01.md
```

Both arms must be scored by the **same** `bake_verdict` build — the binary that
auto-emitted `r7_hf01.verdict.md` at train time predates the `Orientation` fix
and the `hf_nearlossless` corpus, so its per-ref column is not comparable.

- Bakes: `/mnt/v/output/zensim/r7_rust/r7_hf{0,01}.bin`
- Panels: `/mnt/v/output/zensim/r7_rust/v2_hf{0,01}.md`
- Train logs: `/mnt/v/output/zensim/r7_rust/train_hf{0,01}.log`
- Python reference: `benchmarks/blend_search_r7_2026-07-15.tsv`
