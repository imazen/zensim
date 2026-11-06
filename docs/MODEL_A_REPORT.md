# How zensim's model `A` works and how it was built

`zensim` 0.3.0 ships one default model, profile `A` (`zensim-a`). This report explains how it
computes a score, what the trained network turned out to be, what it was trained on, how it was
developed and chosen, how it was evaluated, and where it is weak.

Every number comes from a committed measurement. The raw runs and registrations are in
[imazen/zensimx](https://github.com/imazen/zensimx), chiefly
[`benchmarks/e36_subset_test_2026-11-01.md`](https://github.com/imazen/zensimx/blob/main/benchmarks/e36_subset_test_2026-11-01.md).
[`PROFILES.md`](PROFILES.md) is the shorter companion that maps profiles to bakes.

## 1. How a score is computed

`A` is a fixed extractor of hand-designed statistics followed by a small trained network. The
extractor keeps the vocabulary of SSIMULACRA2: an SSIM error and asymmetric edge maps in XYB, the
colour space butteraugli introduced. It changes the plumbing for speed and adds further detectors.

1. **Colour.** Both images become XYB at full resolution. sRGB input goes through a lookup table
   and SIMD; the cube root is the key nonlinearity. Images smaller than 64 pixels are reflect-padded.
2. **Pyramid.** Four dyadic scales are built by 2×2 box averaging of the XYB planes.
3. **Local statistics.** At each scale and channel, an 11-pixel box window feeds the SSIM variant,
   edge artifact and detail-loss ratios, MSE, high-frequency energy and magnitude changes,
   blockiness, ringing, gradient similarity, near-threshold transducers and banding.
4. **Pooling.** Each map is pooled several ways: mean, L2, L4, soft peaks, and weights that fall or
   rise with local reference activity. This yields **492 damage features**, each a difference
   between the two images, in revision-7 arithmetic.
5. **Fragility.** Twelve reference-only values, one per scale and channel, measure how little the
   reference masks damage: one minus a saturating function of its mean gradient. Each damage
   feature also enters the network multiplied by the fragility of its own scale and channel, so a
   product is exactly zero whenever its damage is. That adds **492 fragility products**.
6. **Network.** The 984 inputs feed 128 hidden units. The first 64 form 32 tied pairs that compute
   `c·max(a·x, b·x − t)` with `a ≥ 0`, `b ≥ 0`, `t ≥ 0` and `c ≤ 0`; the rest are monotone hinges.
7. **Score.** A monotone PCHIP spline, fitted on the CID22 training table, maps the network's output
   to the dial. It caps at 100 and extends below 0 for severe damage. Identical images score 100.

The structure is checked from the bake's bytes by a monotonicity certifier: raising any damage
feature cannot raise the score, apart from rounding. On 12,163 CID22 rows the ten highest-impact
inputs showed zero rises in 4,000 sampled comparisons each. The largest rise found on any input
was 0.000012 points, which is floating-point cancellation inside the signed pair. This guarantee
covers the features, not perceived damage: the features themselves can move the wrong way, and
section 6.3 reports how often the final score misorders close encoder settings.

The per-region map that steers encoders (`compute_with_diffmap`) is separate from the network. It
uses a fixed 156-coefficient projection.

## 2. What the trained network turned out to be

The 128 units are capacity for training, not what the model uses. In the shipped bake:

| Units | Count | Role |
|---|---:|---|
| 0 to 94 | 95 | All-zero input weights: never active |
| 95 to 125 | 31 | Identical weights, output coefficient −0.000031: tiny fragility-weighted term |
| 126, 127 | 2 | One tied max pair, output coefficient −0.498 each: 99.998% of the mean raw contribution |

So `A` is close to `score = spline(pin − 0.498·max(a·x, b·x − t))`: the larger of two additive damage
budgets. The second budget wins on 58.3% of the CID22 rows. Seeds 1 and 2 reach the same shape, with
over 99.999% in the pair. A structurally pruned copy gives identical scores in every registered
read, and its head takes 2.84 µs against 5.77 µs. A whole score takes 3.5 ms at 256², so the head is
under 0.2% of it and the unpruned bake ships.

Training width still matters even though the served model ends up this small; see section 5.

**Which features carry the score.** Setting a whole signal family to zero moves CID22 scores by:

| Signal family | Mean change, points |
|---|---:|
| Edge detail loss (basic) | 8.23 |
| Artifact soft peak | 4.16 |
| High-frequency magnitude loss | 3.11 |
| Blockiness | 2.59 |
| High-frequency loss | 2.48 |
| SSIM deviation (4th power) | 1.59 |
| Edge artifacts (basic) | 1.08 |
| Activity-weighted artifacts | 0.94 |

These effects overlap and do not add. All 492 damage features have a nonzero effect on CID22 scores,
yet 13 carry half of the summed single-feature effect, 69 carry 90% and 149 carry 99%. The ten
strongest features are all on the Y channel. Full tables:
[`a_weight_audit_2026-11-04.md`](https://github.com/imazen/zensimx/blob/main/benchmarks/a_weight_audit_2026-11-04.md).

## 3. Speed

Single-call p50 in milliseconds on an Intel Core Ultra 7 270K Plus at the v3 (AVX2) tier: 20 real
images per size, zenjpeg q40 reconstructions, 32 clean rounds per image. Lower is better. These runs
predate the parallel cutoff below, which affects none of the cells shown.

| Threads | Size | `A` | fast-ssim2 | margarine |
|---:|---|---:|---:|---:|
| 1 | 64² | 0.75 | 0.40 | 1.48 |
| 1 | 256² | 3.53 | 4.21 | 6.05 |
| 1 | 1024² | 33.5 | 73.9 | 56.2 |
| 1 | 12 MP | 351 | 1,374 | 666 |
| 16 | 256² | 2.65 | 4.75 | 7.31 |
| 16 | 1024² | 11.2 | 33.8 | 48.4 |
| 16 | 12 MP | 105 | 357 | 550 |

The fit for `A` is `time = 1.60 ms + 29.0 ms/MP` on one thread and `2.04 ms + 8.67 ms/MP` on sixteen.
Record: [`a_speed_vs_fastssim2_margarine_2026-11-04.md`](https://github.com/imazen/zensimx/blob/main/benchmarks/a_speed_vs_fastssim2_margarine_2026-11-04.md).

**Parallel cutoff.** Parallel extraction was slower than serial on small images, so `A` now runs
serially below 36,864 padded pixels (192²) even when parallelism is enabled. The rule was the first
size where parallel wins p50: 160² at 16 threads and 192² at 4 threads, taking the higher. Below the
cutoff, both "after" columns run the same serial walk.

| Threads | Size | Before: serial / parallel, ms | After: serial / parallel, ms |
|---:|---|---:|---:|
| 16 | 64² | 0.86 / 1.85 | 0.70 / 0.70 |
| 16 | 160² | 2.24 / 2.06 | 2.02 / 1.98 |
| 16 | 384² | 5.84 / 5.91 | 5.84 / 5.03 |
| 16 | 1024² | 30.6 / 14.1 | 30.7 / 15.3 |
| 16 | 12 MP | 345.7 / 117.5 | not measured |
| 4 | 64² | 0.83 / 1.95 | 0.74 / 0.74 |
| 4 | 192² | 2.50 / 2.46 | 2.39 / 2.29 |
| 4 | 320² | 4.71 / 4.81 | 5.42 / 4.28 |

The grid is not monotone above the cutoff: serial beat parallel at 384² on 16 threads and 320² on 4
threads in the first run. The 1024² parallel p50 rose 8.5% in the second run, and these measurements
do not establish that every larger size is unchanged. All 980 serial/parallel score comparisons were
bit-identical. This is one host's measurement, not a universal optimum. Record:
[`a_parallel_gate_2026-11-04.md`](https://github.com/imazen/zensimx/blob/main/benchmarks/a_parallel_gate_2026-11-04.md).

**Where the time goes, and why it stays.** At 256² on one thread the top costs are the dense detector
kernel (24% of cycles), horizontal blur (17%), vertical blur (15%), colour conversion (10%) and the
fused SSIM kernel (10%). The f64 divider is active for 22 to 24% of cycles at every size. Three
bit-identical fusions of these passes each lost paired timing on at least one required cell, so none
shipped. Removing every blur and pyramid pass would give at most 1.85×. Omitting the 156 lowest-effect
damage features removes no pass and saves no measurable time. A larger speedup needs an arithmetic
change that alters scores, such as f32 moments. Per-score peak memory has not been isolated. Record:
[`a_extraction_fusion_2026-11-04.md`](https://github.com/imazen/zensimx/blob/main/benchmarks/a_extraction_fusion_2026-11-04.md).

## 4. Training data

`A` is research profile `F`, recipe `mixF3`: "the TID2013 + MDID mix on F3's recipe". Training has two
steps and a pack, each run at seeds 0, 1 and 2 on one thread. Seed 0 ships; the others agree with it
within their intervals on every evaluation.

**Base fit (120 epochs, from scratch).** A group's share of the draws is its rows times its weight.

| Group | What it is | Rows | Weight | Loss |
|---|---|---:|---:|---|
| safesyn | synthetic distortion pairs, SSIMULACRA2 as teacher | 141,054 | 1.017 | rank + squared error |
| cid22 | CID22 training images, teacher labels | 12,163 | 1.012 | rank + squared error |
| coverage | coverage ladders: light, spatial, new families | 17,600 | 21.622 | rank |
| human | TID2013 MOS rows, training split | 1,680 | 18.018 | rank |
| MDID | MDID MOS rows, training split | 800 | 18.018 | rank |
| BAPPS | two-alternative human judgments | 223,068 | 1.000 | rank |
| chroma_teacher | chroma ladder teacher rows | 231,367 | 1.042 | rank + squared error |

Rank means RankNet pairs within a reference. The two human MOS groups split the recipe's whole MOS
weight (36.035) equally. KADID, KonFiG, CID22-A25, PIPAL, LIVE-MD, CID:IQ, CID:OA, IRCCyN/IVC and
Colourlab trained earlier recipes and do not train `A`.

**Continuation (60 epochs at learning rate 1e-5).** The same groups, plus the JPEG AIC-2026 public
training set (3,920 crops of 28 sources, weight 256) and the human leg again as a cross-reference. The
AIC leg takes 48.3% of the continuation's draws and 16.1% of all draws over both steps. Its labels are
available to JPEG committee members only, so no AIC-2026 result appears in this repository.

**Pack.** Epoch 59 becomes the production f16 bake with the dial spline refit on the CID22 fit table and
the derived-input declaration embedded. Removing training provenance gives the shipped file,
`weights/rev7_mix_h128_2026-11-03.bin`, SHA-256 `d5e2a56d…`.

## 5. Development process

### How the program ran

Each experiment was registered in the research record before it ran, with its measures and reading
rule. Each read of held-out labels was entered in an exposure ledger
([`docs/DATA_SPLITS.md`](https://github.com/imazen/zensimx/blob/main/docs/DATA_SPLITS.md)) first. Results
were read per seed with whole-reference bootstrap intervals.

| Date (2026) | Step |
|---|---|
| to September | Linear previews published as 0.2.7. Rank-only selection on MOS let an unbounded MLP through: a blurred fractal scored above its identical copy. Selection gained dial and identity gates |
| Oct 19 to 25 | Formula revision 7 and certified-monotone networks. The max pair is the best exactly monotone arm tested |
| Oct 25 | Selection moved to five product gates: certified monotone with identity pin; no global-edit inversion; dial behaviour along ladders; codec uniformity; encoder steering. Pooled human parity became a diagnostic |
| Oct 26 to 30 | AIC-2026 public training set admitted as a leg. Chroma program added 82 full-resolution chroma inputs and chroma ladders. SSIMULACRA2 found invalid at 64 pixels and below |
| Nov 1 to 3 | Loss-term program on the encoder-ladder set; experiment 5 searched which human MOS sources help (section 5.2) |
| Nov 3 | Mix carried through the full recipe; product gates, the JPEG XL loop and held-out reads |
| Nov 4 | Owner adopted the mix. HDR front end moved to the anchored cube root |
| Nov 4 to 5 | Studies of what could still change: heads, width, input transforms, extraction speed, parallel cutoff |

### How the mix was chosen

The question was which human MOS source, or pair, gives the lowest error on the other human MOS sets.
The registered measure was the median absolute residual around a four-parameter logistic fit of score
to MOS, divided by the collection's rating span, averaged over the nine evaluation collections an arm
did not train on. Arms kept the total MOS weight fixed, so an arm changed which rows were compared.

- **Singles.** Every source but IRCCyN/IVC and Colourlab beat the full set of 11 sources on the
  collections it did not train on. TID2013 alone scored 8.33% against the full set's 8.81%. With no MOS
  leg at all, the error on the three collections no arm trained on was 8.42%: the MOS legs mostly fit
  the collections they come from.
- **Pairs.** TID2013 + MDID was best at 8.31%.
- **Three seeds.** The pair's worst seed (8.53%) beat the full set's best (8.81%).
- **Controls.** Routing all original sources through the new path reproduced the previous scores
  exactly, and the training build reproduced its checkpoints bit for bit.

Those nine collections chose the mix, so they are selection data for it. Further evidence comes from the
held-out reads in section 6.2 and the frozen-A assessment in section 6.5, whose prior-exposure history is
disclosed there.

### What was tried and not adopted

| Variant | What it gained | Why it was not adopted |
|---|---|---|
| Ladder hinge, λ = 30 | Encoder-ladder misordering 2.39% against `A`'s 3.39% | Shared-three MOS error 8.96% against 8.27%; lower CLIC 2021 agreement |
| Mix + ladder hinge | Misordering 2.36% | CLIC 2021 −0.0047 [−0.0091, −0.0003]; 131 to 132 of 135 steering cases |
| Map loss term, weight 300 | Repaired the steering case | CLIC 2021 −0.0047 [−0.0084, −0.0010] against the hinge alone |
| Codec-only continuations | +0.003 SROCC on compression rows | Misordering 4.38% against 3.40%; MOS error 8.7% against 8.3% |
| Non-negative linear heads | Dense head misordering 3.15% against 3.40% | MOS error 8.35% against 8.03%; chroma level fails at all seeds |
| Hidden width 4 to 512 | Widths 64 and 256 match 128 within seed noise | Every width fails the strict chroma-level gate, by 0.02 to 0.38 points; width 4 worsens ordering by 0.24 points. Narrow fits reach a worse pair |
| Input transforms | Yeo-Johnson ordering 3.23% against 3.39% | Failed the zero-allowance development rule, MAE 3.51 against 3.23 |
| HDR retrain through the cube root | +0.01 on 48 4Kdtb rows; otherwise its own training rows | UPIQ development rows 0.924 against 0.943 at seed 0; ladder inversions 108 against 50 |
| Extraction fusions | Identical scores | Lost paired timing (section 3) |

## 6. Evaluation

### 6.1 Against human ratings (selection data for the mix)

Seed 0: SROCC, PLCC after the four-parameter logistic fit, KROCC and the error defined above, over each
collection's 1 to 99% rating span. `F3` is the previous recipe with all MOS sources. MDID's references
differ from those `A` trained on; the other eight collections are not trained on by `A`.

| Collection | n | F3: SROCC / PLCC / KROCC / error | `A`: SROCC / PLCC / KROCC / error |
|---|---:|---|---|
| CID:IQ | 210 | 0.863 / 0.865 / 0.684 / 8.3% | 0.868 / 0.869 / 0.692 / 7.8% |
| CID:OA | 91 | 0.905 / 0.921 / 0.749 / 8.7% | 0.911 / 0.923 / 0.753 / 8.3% |
| Colourlab | 192 | 0.689 / 0.684 / 0.516 / 12.7% | 0.735 / 0.728 / 0.562 / 10.7% |
| IQA-ODI | 48 | 0.858 / 0.924 / 0.674 / 7.5% | 0.867 / 0.928 / 0.679 / 7.6% |
| IRCCyN/IVC | 60 | 0.926 / 0.942 / 0.757 / 6.4% | 0.919 / 0.931 / 0.752 / 6.8% |
| LIVE-MD | 180 | 0.892 / 0.915 / 0.716 / 7.0% | 0.902 / 0.926 / 0.730 / 7.0% |
| MDID | 480 | 0.886 / 0.895 / 0.705 / 7.5% | 0.927 / 0.931 / 0.765 / 6.1% |
| NITS-IQA | 405 | 0.708 / 0.709 / 0.518 / 12.2% | 0.734 / 0.734 / 0.542 / 11.3% |
| VCL@FER | 192 | 0.939 / 0.937 / 0.778 / 6.6% | 0.950 / 0.950 / 0.800 / 5.9% |

Beside published metrics, on the same rows. "Eight" is every collection but MDID; "shared three" are
IQA-ODI, NITS-IQA and VCL@FER, in no candidate's training; compression rows are the 417 JPEG, JPEG 2000,
JPEG XT and LAR rows.

| Score | Eight: error / SROCC | Shared three: error / SROCC | Compression: SROCC / error |
|---|---|---|---|
| `A` | 8.18% / 0.861 | 8.27% / 0.850 | 0.895 / 6.95% |
| F3 | 8.67% / 0.848 | 8.79% / 0.835 | 0.895 / 7.03% |
| SSIMULACRA2 | 9.13% / 0.831 | 8.52% / 0.859 | 0.896 / 6.92% |
| Butteraugli 3-norm | 11.57% / 0.709 | 10.47% / 0.796 | 0.854 / 8.61% |
| CVVDP | 9.82% / 0.823 | 9.52% / 0.829 | 0.876 / 7.07% |
| FSIMc | 9.97% / 0.830 | 8.84% / 0.864 | 0.906 / 6.73% |
| HaarPSI | 8.92% / 0.843 | 8.99% / 0.828 | 0.888 / 6.90% |

On compression rows FSIMc leads, and SSIMULACRA2, `A` and F3 are within 0.001 of each other.
SSIMULACRA2 and FSIMc rank better than `A` on the shared three, with higher error.

On the 871 rows with no codec in them, row-weighted SROCC and error: `A` 0.790 / 8.66%, HaarPSI 0.774 /
9.76%, F3 0.767 / 9.28%, SSIMULACRA2 0.759 / 10.59%, CVVDP 0.755 / 10.76%, FSIMc 0.745 / 10.79%,
butteraugli 0.661 / 13.07%. Contrast and brightness changes and geometric distortions defeat every
score here, `A` included.

### 6.2 Held-out confirmation reads

Each read was registered first, and no model was selected on it.

| Read | What | `A` | Others |
|---|---|---|---|
| CLIC 2021 validation | 4,812 two-alternative questions on 315 source-independent crops, learned codecs | 0.642 at seed 0; 0.644 over three seeds | F3 0.643 over three seeds, difference +0.001 [−0.002, +0.004]. FSIM 0.638, FSIMc 0.637, SSIMULACRA2 0.626, butteraugli 0.606 |
| CIQA | 7,902 two-alternative questions between JPEG qualities of 6,596 photographs | 0.675 | Every score 0.675 to 0.676. "The higher quality wins" also scores 0.675 against a ceiling of 0.698, so this set separates nothing |
| MMSPG JPEG XR | 186 rated conditions, 6 photographs | SROCC 0.920 | F3 0.920, SSIMULACRA2 0.934, FSIMc 0.929, butteraugli 0.902 |

CLIC 2021 has been read three times over the program; the third read included `A`. On questions `A` calls
within 2.4 points, it leans to the smoother crop where raters prefer fine detail. That costs 0.3 to 0.6
points of agreement overall and returns at moderate differences, which is why the totals tie with F3.

### 6.3 Product gates

| Gate | Result |
|---|---|
| Dial contract C1 to C6: monotone along ladders, no flat zones, negative tail, identity, nothing above a perfect copy | Pass, seeds 0 to 2, both grids |
| Identity at 12 SIMD tier permutations | Pass |
| Near-identity ladders | Pass; 132 of 144 ladders monotone at each seed |
| Chroma level | Pass; worst MAE ratio to F3 is 1.008 against a bar of 1.05 |
| Chroma trade pairs misordered | 2.47% against F3's 2.50%; pass |
| Map steering cases, of 135 | 133 / 134 / 134 at seeds 0 to 2. One text-screenshot case misses its map-rank bar at every seed (0.69 against 0.70) |
| Encoder-ladder misordering, 646,677 held-out pairs where butteraugli and CVVDP agree | 3.39%; F3 3.35% |
| JPEG XL loop against F3, bytes at equal judged quality | +0.19% [−0.09, +0.46] by SSIMULACRA2, −0.21% [−0.56, +0.12] by butteraugli |

The missed steering case has no byte cost in the JPEG XL loop and was accepted with the model. Against a
loop steered by butteraugli, zensim's map still costs bytes: 2.18% by SSIMULACRA2 and 5.31% by butteraugli.

### 6.4 HDR

`compute_hdr` decodes declared linear, PQ or HLG input to absolute light and uses the cube-root front end
with SDR white at 203 cd/m². `A` was fitted to SDR with no HDR human rows, so this is a zero-shot use of
the SDR model. The front end was chosen for continuity: an SDR image shown as HDR at that white scores
within 0.0003 of its SDR score on the research panel. At seed 0, `A` reads SROCC 0.943 on UPIQ's 50
development rows through it, against 0.835 through the earlier PU21 front end, and 0.811 on all 380 UPIQ
rows. A retrain through the cube root was worse held out (section 5). HDR scores are not a calibrated
human scale. See [HDR input](PROFILES.md#hdr-input).

### 6.5 Registered frozen-A final assessment

Registered on 2026-11-05 and read once for the shipped bake, with SSIMULACRA2 and butteraugli's 3-norm on the same
pixels. All eight populations are complete. Signed SROCC below, with distortion-oriented
targets keeping their sign. Intervals, the other statistics and every disclosure are in
[`FINAL_READ_A_2026-11-05.md`](FINAL_READ_A_2026-11-05.md).

| Population | Pairs / references | `A` | SSIMULACRA2 | Butteraugli quality |
|---|---:|---:|---:|---:|
| CID22-B | 2,100 / 24 | 0.909 | 0.913 | 0.781 |
| CSIQ | 866 / 30 | 0.946 | 0.905 | 0.814 |
| LIVE release 2 | 779 / 29 | 0.967 | 0.960 | 0.916 |
| LIVE, 10 content-disjoint references | 267 / 10 | 0.967 | 0.961 | 0.911 |
| KonJND JPEG TERMINAL | 100 / 100 | −0.368 | −0.401 | +0.053 |
| KADID TERMINAL | 2,000 / 16 | 0.893 | 0.800 | 0.572 |
| AIC-4 sample | 300 / 5 | −0.922 | −0.913 | −0.897 |
| SDR25, a subset of AIC-4 | 50 / 5 | −0.855 | −0.841 | −0.842 |
| MCL-JCI | 5,000 / 50 | −0.897 | −0.885 | −0.851 |
| MCL-JCI, without KonFiG's ten sources | 4,000 / 40 | −0.904 | −0.891 | −0.856 |

CLIC 2022 uses two-alternative agreement.

All 57,300 questions over 57,243 reference crops were retained. Majority
agreement is the registered primary; historical soft-vote agreement is the
approved supplement. The cached targets match on all 57,300 questions, so
both panels coincide. Tied majorities and tied predictions receive half credit.

| Metric | Majority primary, 95% CI | Soft-vote supplement, 95% CI |
|---|---|---|
| A | 0.746632 [0.743006, 0.750179] | 0.746632 [0.743006, 0.750179] |
| SSIMULACRA2 | 0.736230 [0.732504, 0.739863] | 0.736230 [0.732504, 0.739863] |
| Butteraugli quality = −p3 | 0.688342 [0.684479, 0.692035] | 0.688342 [0.684479, 0.692035] |

Intervals use 10,000 whole-reference draws, seed `20261105`, percentile/type7.
CLIC 2022 shares the protocol of the earlier CLIC 2021 confirmation read.

**KADID TERMINAL failed both of its registered confirmation rules.** The rules compare `A` with two research models,
not with published metrics.

| Comparison | Comparator's SROCC | Difference | Registered limit | Result |
|---|---:|---:|---:|---|
| `A` minus research `B` | 0.942 | −0.049 | −0.008 | Failed |
| `A` minus research `by_v2fy` | 0.933 | −0.040 | −0.005 | Failed |

The failure stands, and nothing was changed on it. Both comparators trained on KADID's other references and `A` did
not train on KADID at all. That may contribute, but it has not been tested.

Earlier frozen research models were scored on most of these sets, and third-party-metric studies read some of their
labels. `A` was neither trained nor selected on them; the record lists the known content overlaps with its training
sources.

## 7. Known limits

- KADID TERMINAL failed its registered confirmation against two research models (section 6.5), though `A` leads SSIMULACRA2 and butteraugli there.
- One map-steering case on a text screenshot misses its bar (0.69 against 0.70).
- Near-ties on learned-codec pairs lean to the smoother crop where raters prefer fine detail.
- CID:IQ's SGCK gamut mapping reads 0.87 against F3's 0.91.
- Contrast and brightness changes and geometric distortions are weak for every score tested. NITS-IQA's
  D4 contrast change is the worst case at 0.22; the best metric reaches 0.46.
- The score is monotone in its features, but 3.39% of close encoder-ladder pairs are still misordered.
- HDR and SDR readings of the same encode are different regimes; an HDR score needs its white and boost
  stated.
- SSIMULACRA2 is not valid at 64 pixels and below, so it cannot serve as a reference at those sizes.
  Images smaller than 64 pixels are reflect-padded here.
- AIC-2026 results are members-only and are not reported.

## 8. Reproducing it, and parity

**Recipe.** The recipe declares each command, trainer build and expected hash. A reproduction archive
is available at an unlisted URL: pixel archives, tables and tools, the extractor, and a README. Each
object passed a complete SHA-256 readback and a separately verified archive mirror. The URL is recorded
in the private training repository.

The archive initially excluded five AIC-2026 inputs required for full replay. On November 6,
the owner authorized those exact files as an unlisted supplement; each passed full hash readback
and mirror verification. Access instructions remain in the private training repository. This
limited distribution is not a generally redistributable public dataset release.
The final seed-0 replay using the archive plus those five inputs passed all recorded checks: 2/2 base-fit,
62/62 continuation, 1/1 packing and 1/1 stripping. It reproduced the shipped bake with SHA-256
`d5e2a56db35c32620d9f9d9c19ba3bc78bd8fea2b8cd933cde8447428d85e62a`.
This verifies seed 0, not every seed declared in the recipe.

**Parity.** This crate serves frozen models. `examples/split_parity_check.rs` here and the research crate's
`split_parity_golden`, built on one machine, must produce identical output; the gate is described in
[PROFILES.md](PROFILES.md#parity-with-the-research-crate). The 2026-11-04 parallel-cutoff run matched all
135,625 normalized contract rows.

**Paper.** A companion paper on the metric and its evaluation exists as a private draft, updated to this model on
2026-11-05. It is not cited here.
