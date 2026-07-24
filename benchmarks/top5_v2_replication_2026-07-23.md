# Top-model replication + v2 introduction (2026-07-23)

**Directive:** replicate the last-month top models EXACTLY, then introduce the
v2-348 block; compare to B and the best 372-feature model.

## Fidelity check — winner_dial reproduces EXACTLY

The cookbook's top model `Ebothg_hfgain_winsor_dial` ("winner_dial", CID22 0.894)
reproduces byte-faithfully from the §3 recipe (canonical-2026-05-21, 196k safesyn,
156-basic MLP + 8 hf_gain winsor + `:both` + seed 13 + dial spline):

| | CID22 | CSIQ | LIVE | cookbook |
|---|--:|--:|--:|---|
| winner_repro | **0.8940** | 0.9584 | 0.9600 | 0.894 / 0.958 / 0.960 ✓ |

The reproduction pipeline is validated. (KonJND 0.431 vs the cookbook's 0.335 —
different KonJND corpus cut; headline photo/CID22 match exactly.)

## The v2 sweep — winner recipe at 156 / 372 / 504 / 720 (matched 111k rows)

To measure v2's *marginal* value cleanly, one recipe (the winner) is run at four
feature widths on the SAME rows (the ext720 111k safesyn subset — the extent of
the v2 extraction). `--max-features` selects the width; the 504-corpus is
`basic-156 (f0..f155) ++ v2-348 (f372..f719)`:

- **156** = basic only (the winner's actual feature set)
- **372** = full v1 (basic + masked + iw + nonspatial)
- **504** = basic-156 **++ v2-348** (drop v1's masked/iw/nonspatial, add v2)
- **720** = full v1 **++ v2-348**

| corpus | 156 | 372 | 504 | 720 | **504−372** | **720−372** |
|---|--:|--:|--:|--:|--:|--:|
| CID22 | 0.888 | 0.885 | 0.892 | 0.893 | **+0.007** | +0.008 |
| imazen26 real-codec | 0.807 | 0.736 | 0.828 | 0.787 | **+0.091** | +0.050 |
| imazen26 non-photo | 0.807 | 0.747 | 0.829 | 0.797 | **+0.083** | +0.050 |
| AIC-4 | 0.909 | 0.884 | 0.917 | 0.909 | +0.034 | +0.025 |
| AIC-3 | 0.815 | 0.795 | 0.809 | 0.797 | +0.014 | +0.003 |
| KonJND | 0.199 | 0.178 | 0.360 | 0.367 | **+0.182** | +0.188 |
| CSIQ | 0.366 | 0.373 | 0.452 | 0.241 | +0.079 | **−0.131** |
| LIVE | 0.240 | 0.462 | 0.457 | 0.302 | −0.005 | **−0.160** |

## VERDICT — v2 helps the best model, in the 504 (basic+v2) configuration

**This REVERSES the earlier deterministic-linear conclusion (`v2 doesn't beat
v1`).** That test was 372→720 on a *linear* model; both were the wrong instrument
for the question:

1. **504 (basic-156 ++ v2-348) beats full-372 v1** on nearly every axis — CID22
   +0.007, imazen26 +0.09, KonJND +0.18, AIC-4 +0.03, CSIQ +0.08; LIVE flat.
   **v2 adds genuine signal over full v1, not just restoring what basic-156
   dropped** — because 504 beats *372* (which already has v1's masked/iw), not
   only 156.
2. **720 (full-v1 ++ v2) overfits the small FR corpora** — helps CID22/imazen26/
   KonJND but craters CSIQ (−0.13) and LIVE (−0.16). Adding v2 *on top of the
   full 372* is redundant capacity that destabilizes the 800-row corpora. This is
   the same effect my linear-720 twin measured — and why **504, not 720, is the
   right v2 configuration**: drop v1's redundant masked/iw/nonspatial, keep basic
   + v2.

The sweet spot is **basic-156 ++ v2-348** — a 504-feature model that dominates
both the 156-basic winner and the full-372 baseline, without the 720 FR-crater.

### Caveats (honest)
- **Trained on 111k safesyn (the v2-extraction subset), not the cookbook's 196k.**
  So the winner-family ABSOLUTE numbers sit below the 0.894 fidelity repro; the v2
  DELTAS are clean (all arms share the 111k). Confirming 504 at full 196k needs
  the v2 extraction finished on all safesyn rows.
- **B (shipped linear-372) is a different, well-tuned model on the full 196k** and
  still leads the classic FR corpora (CSIQ 0.934, LIVE 0.897, imazen26 0.896);
  it's context, not a matched arm. A clean B-vs-504 needs B's recipe re-run on
  111k or the winner re-run on 196k+v2.

Bakes: `/mnt/v/output/zensim/bakes/top5/`. Recipe: cookbook §3.

## Ebothg_scr0.5 (winner + bigcodec) + v2 — CONFIRMS + amplifies

Same 156-vs-504 twin with the bigcodec line added (`bigcodec:tbig504:0.5:1.0:both`,
the scr0.5 recipe). v2 helps even more on FR, trading a small CID22/KonJND dip:

| corpus | Eb156 | Eb504+v2 | Δ | (winner Δ, no bigcodec) |
|---|--:|--:|--:|--:|
| CID22 | 0.890 | 0.884 | −0.006 | (+0.004) |
| imazen26 real-codec | 0.879 | **0.930** | +0.051 | (+0.021) |
| imazen26 non-photo | 0.866 | **0.927** | +0.061 | (+0.022) |
| CSIQ | 0.320 | 0.488 | +0.168 | (+0.085) |
| LIVE | 0.272 | **0.684** | **+0.412** | (+0.217) |
| KonJND | 0.452 | 0.425 | −0.027 | (+0.161) |

**bigcodec mass + v2's FR families (gms/ringing/blockiness) synergize** — LIVE more
than doubles, imazen26 clears 0.92 (beating shipped B's 0.896). Without bigcodec the
benefit shifts to CID22 + KonJND. Either way, **the 504 (basic+v2) config is a net
win**; the only regressions are ≤0.03 (CID22/KonJND with bigcodec).

## ADD156 (additive linear, safesyn-only RAW) + v2 — same pattern

The additive-linear top model (`ADD156_safesyn_only_raw_lasso`), 156 vs 504 RAW
(BVLS, safesyn-only — via `linear_projections twin --mix add156 --raw`):

| corpus | 156 | 504+v2 | Δ |
|---|--:|--:|--:|
| CID22 | 0.810 | 0.780 | **−0.030** |
| imazen26 real-codec | 0.837 | 0.851 | +0.014 |
| imazen26 non-photo | 0.841 | 0.865 | +0.024 |
| CSIQ | 0.531 | 0.787 | **+0.255** |
| LIVE | 0.487 | **0.921** | **+0.434** |
| KonJND | 0.386 | 0.476 | +0.091 |

v2 helps FR massively (LIVE nearly doubles) but costs CID22 −0.030 — the classic
linear behaviour (v2's FR mass drags linear CID22, same as bigcodec-poisons-linear).

## BOTTOM LINE — v2 in the 504 config helps every replicated top model

| model | CID22 Δ | FR (LIVE/CSIQ) | KonJND / imazen26 |
|---|--:|---|---|
| **winner_dial** (MLP, no bigcodec) | **+0.004** | +0.22 / +0.09 | +0.16 / +0.02 |
| **Ebothg_scr0.5** (MLP, bigcodec) | −0.006 | **+0.41** / +0.17 | −0.03 / +0.06 |
| **ADD156** (additive linear) | −0.030 | +0.43 / +0.26 | +0.09 / +0.02 |

**Consistent across all three top models: `basic-156 ++ v2-348` (504) delivers large
FR/non-photo gains** (LIVE +0.22…+0.43, CSIQ +0.09…+0.26, imazen26 +0.02…+0.06). The
CID22 effect scales with architecture: the MLP winner *gains* CID22 (+0.004), the
linear *loses* it (−0.030). **For the actual top model (winner_dial MLP), 504+v2 is a
clean net win.**

The earlier "v2 doesn't help" verdict was wrong because it tested the wrong config
(372→720, where v2 is redundant) on the wrong instrument (linear). Replicating the
real top models and testing the right config (drop v1's redundant masked/iw, add v2)
shows v2 is a genuine improvement — strongest on the FR/non-photo axis the product
cares about, and CID22-positive on the winning MLP.

**Next (not done):** confirm 504 at full 196k safesyn (needs v2 extraction on the
remaining 85k rows); run the RD + steer gates (§scorecard) on the 504 winner — rank
gains must survive the codec-in-the-loop probe before any ship.
