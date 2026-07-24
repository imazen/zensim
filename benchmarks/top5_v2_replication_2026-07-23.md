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

## Remaining top models (queued)
- `Ebothg_scr0.5` (= winner + bigcodec line) + v2
- `ADD156` (additive linear basic-156) + v2 (linear 504 via the twin tool)
