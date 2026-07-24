# Top-model v2 replication — FULL scoreboard (rank + dial + corruption) + v3 design (2026-07-23)

Corrects `top5_v2_replication_2026-07-23.md`, which reported *rank only*. The
five-gate scorecard was built so a bake can't be called "best" on rank while its
**dial monotonicity** is broken — and that's exactly what happened.

## The full scoreboard

Rank SROCC per corpus + DIAL panel (monotonicity G3, bar ≥0.93) on the densified
multi-codec q-sweep, matched-width grid per bake (504-reindexed grid for 504 bakes):

| model | CID22 | im26_rc | im26_np | CSIQ | LIVE | KonJND | **G3 mono** | inv |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| winner-156 | 0.888 | 0.807 | 0.807 | 0.366 | 0.239 | 0.199 | 0.910 ✗ | .090 |
| winner-372 (best v1) | 0.885 | 0.736 | 0.747 | 0.372 | 0.462 | 0.178 | **0.950 ✓** | .050 |
| **winner-504 (+v2)** | 0.892 | 0.828 | 0.829 | 0.452 | 0.457 | 0.360 | **0.922 ✗** | .078 |
| winner-720 | 0.892 | 0.787 | 0.797 | 0.241 | 0.302 | 0.366 | 0.909 ✗ | .092 |
| ebothg-156 | 0.890 | 0.879 | 0.866 | 0.320 | 0.272 | 0.452 | 0.912 ✗ | .088 |
| **ebothg-504 (+v2+bigcodec)** | 0.884 | **0.930** | **0.927** | 0.488 | **0.684** | 0.424 | **0.960 ✓** | .040 |
| ADD-156 (additive linear) | 0.810 | 0.837 | 0.841 | 0.531 | 0.487 | 0.386 | **1.000 ✓** | .000 |
| ADD-504 (+v2) | 0.780 | 0.851 | 0.865 | 0.787 | 0.921 | 0.476 | **1.000 ✓** | .000 |
| **B** (shipped linear-372) | 0.882 | 0.896 | 0.899 | 0.934 | 0.897 | 0.519 | 0.976 ✓ | .024 |

**Corruption gate (broken decode MUST rank below honest q20): 3.7% ✗** on the v2
MLP bakes — the known MLP weakness (butteraugli-max wins this 2–4×). v2 does not fix
it (it's an architecture property, not a feature one).

**G1 dial range (p5≤25 ∧ p95≥85): only B passes.** The fresh bakes lack a calibrated
dial spline; 156 bakes can take one (the 372-anchor works), but **504/720 can't —
the `multiband_anchor` has no v2 features**, so G1 for the v2 bakes is blocked on a
v2-extended anchor (small extraction, not yet done).

## The corrected verdict

- **winner-504 (+v2) FAILS G-DIAL** (mono 0.922 < 0.93) *and* the corruption gate.
  My earlier "clean net win" was **rank-only** — the dial regressed. This is the
  exact failure the gauntlet exists to catch.
- **ebothg-504 (+v2 *with bigcodec*) is the real rank+dial winner** — mono **0.960**
  (best MLP, passes) AND the biggest FR gains (LIVE 0.684, imazen26 0.93, beating
  shipped B's 0.90 on imazen26). **The bigcodec/multicodec mass is what stabilises the
  v2 dial** (winner-504 without it fails; ebothg-504 with it passes).
- **ADD-504 (additive linear +v2) has a perfect dial** (mono 1.000, by construction)
  and the strongest CSIQ/LIVE, but a CID22 cost (−0.030) — the steerable/additive
  option.

## Gates NOT run (honest blockers)

- **G-STEER (diffmap coherence M2/M3):** the runtime diffmap fold is ≤372-hardwired
  (task #48) — it cannot read the v2 block, so M3 for any 504/720 bake is unmeasurable
  until the fold is extended. The proxy (spatializable-mass) predicts ext-lumacoh ≈1.0
  M3 once wired; that wiring is the remaining ship-engineering.
- **G-RD / G-TARGET (codec-in-loop):** not run (~30 min probe, worktree binaries).
  Rank+dial gains must survive the equal-judged-quality byte comparison before ship.

## The 100% bigcodec parquet (located)

The fleet's **exact** `encode_sha`-keyed join: `s3://zentrain/ext720-canonical-2026-07-22/bigcodec/`
— sidecar `tbig_720_full.parquet` (5,742,660 rows × 720, 0 unmatched) + `views/`
(21 per-codec train/val/test split views, ALL match_rate 1.0). This SUPERSEDES the
91.8% NN-recover as the definitive training input (the NN was only needed because the
`hqdedup` parquet had dropped `encode_sha`). Local original-372 mirrors:
`/mnt/v/output/zensim-multicodec-probe/bigcodec_*.parquet`.

## What v2 replicated from v1 (the overlap)

v2 = 29 signals/channel/scale. **18 of 29 (62%) are bounded/foldable RE-IMPLEMENTATIONS
of v1 families:**

| v2 family | signals | replicates v1? |
|---|---|---|
| basic (ssim_mean, art, det, mse, hf_gain/loss/mag) | 7 | **YES** — v1 basic |
| soft-peak (ssim/art/det) | 3 | **YES** — v1 peak (bounded version) |
| masked (ssim/art/det/mse) | 4 | **YES** — v1 masked |
| iw (ssim/art/det/mse) | 4 | **YES** — v1 iw |
| ssim_dev2, ssim_dev4 | 2 | **NEW** — GMSD deviation moments |
| pjnd transducer, fragility | 2 | **NEW** — masking transducer |
| gms | 1 | **NEW** — gradient-magnitude similarity |
| transducer low_k, high_k | 2 | **NEW** — transducer bank |
| blockiness, ringing, banding, edge_width | 4 | **NEW** — structural detectors |

The v2 replications aren't wasted: they're **bounded-by-construction so they FOLD into
an exact per-pixel diffmap** where v1's don't (the coherence-maxed finding: v2 basic/
masked/iw are 100% spatializable vs v1's ~53%). The genuinely-new value is DEV2/4, GMS,
transducers, and the 4 structural detectors.

## Hybrid v3 — keep what the evidence says is best

Combining every result (LOO, dial, coherence, this scoreboard):

**KEEP:**
- **v1 basic-156** (the foundation — CID22/rank anchor).
- **v2 bounded masked + iw + soft-peak** (replace v1's non-foldable versions → the
  steering/coherence win, and LOO-load-bearing).
- **GMS** (LOO "graduates" — largest FR carrier, no downside), **ringing** (LOO
  load-bearing), **DEV2/DEV4** (GMSD moments), **blockiness** (mild-positive).
- Train **with bigcodec/multicodec mass** — the dial-monotonicity stabiliser (ebothg-504
  passes G3, winner-504 fails it).

**DROP:**
- **banding** — LOO Σ+0.40, *actively harmful*; cut outright.
- **edge_width** — the one non-per-pixel-foldable v2 feature (breaks exact steering) and
  LOO mild-harmful.
- **chroma transducers (X/B)** — luma-gate to Y only (the 2026-07-19 screen winner).
- **pjnd fragility** for the scalar path (correctly 0 for steering; keep only if a
  fragility signal proves out).

**Shape = `basic-156 ++ v2{masked, iw, soft-peak, gms, dev2/4, ringing, blockiness,
transducer-Y}`** ≈ 156 + ~270 = ~430 features — the 504 config minus banding +
edge_width + chroma-transducers. Bake as an MLP with the winner recipe **+ bigcodec**
(for the dial), then the v2 diffmap fold (task #48) makes it a coherent steerer.

### v3 MEASURED (trained + gauntleted, winner+bigcodec recipe, 504-width, 48 harmful cols zeroed)

| model | CID22 | im26_rc | im26_np | CSIQ | LIVE | KonJND | G3 mono | corruption |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| **v3-hybrid** | 0.880 | 0.928 | 0.923 | 0.441 | 0.661 | 0.422 | **0.960 ✓** | **10.4%** |
| ebothg-504 (all v2) | 0.884 | 0.930 | 0.927 | 0.488 | 0.684 | 0.424 | 0.960 ✓ | 5.7% |
| winner-504 (all v2) | 0.892 | 0.828 | 0.829 | 0.452 | 0.457 | 0.360 | 0.922 ✗ | 3.7% |

**The design holds.** Dropping banding + edge_width + chroma-transducers (the LOO's
harmful/non-foldable set):
- **keeps the dial** (mono 0.960, passes G3) and near-identical CID22/imazen26 rank;
- **nearly doubles corruption robustness** (3.7%→10.4% — the largest of any v2 MLP;
  the harmful families were part of why broken decodes out-ranked honest q20);
- costs a little FR rank (CSIQ −0.05, LIVE −0.02 vs ebothg-504) — the LOO's linear
  "harmful" verdict didn't fully transfer to the MLP+bigcodec, where those families
  carry a little FR signal.

So v3 is the **cleaner, more robust** hybrid: same dial, best-of-v2 corruption, ~same
rank on the product axes (CID22/imazen26). The pure-FR-rank optimum is ebothg-504
(all v2); v3 trades a little of that for coherence (drops the non-foldable edge_width
+ chroma) + robustness. Which to ship is the rank-vs-robustness call the G-RD probe
should settle. Corpora: `/mnt/v/zen/zensim-training/ext504-v3-2026-07-23`; bake
`/mnt/v/output/zensim/bakes/top5/v3_hybrid.bin`.

**Still to run before any ship:** the v2 diffmap fold (task #48 — unblocks G-STEER on
v3/504), a v2-extended dial anchor (unblocks G1), and the G-RD codec-in-loop probe.
