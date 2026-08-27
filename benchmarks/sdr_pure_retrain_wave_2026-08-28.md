# SDR purity retrain wave — pre-registered (2026-08-28)

REGISTERED BEFORE ANY FIT. Trigger: the user's SDR-freeze answer ("SDR purity
retrain first") — W10L9_s4003 is the unanimous two-lens SDR selection, but its
training is campaign-era (saw the channel-A synthetic-v2 files and the
pre-family bigcodec splits). This wave retrains the EXACT winner recipe on
policy-clean views; the freeze moves to its result.

## Data (frozen; `/mnt/v/zen/zensim-training/sdr-pure-2026-08-28/_MANIFEST.json`)
Recipe inputs from the winner's embedded repro, sha-verified against the
originals, with the policy filters applied (one substantive, one measured no-op):
- `safesyn_pure` 111,068 → **111,068 — the purge is a MEASURED NO-OP**: the
  safesyn training table contains ZERO gen-token refs and its ref set
  intersects the d≤2 sharing sources ZERO times (verified against
  `canon_vs_train_synth.tsv`, all 79 sharing sources gen-named). The
  channel-A files lived in the sources DIRECTORY, never in this table —
  so the campaign winner's training NEVER saw channel-A content, and the
  earlier "training saw the shared files" framing is CORRECTED here.
- `safesyn_teacher944_pure` — same, no-op by the same measurement
- `tbig_944_200k_pure` 208,169 → **192,714** (−15,455 rows whose origin's
  FAMILY bucket ≠ train per `split_map_family.tsv`)
- `tbig_teacher944_pure` — same predicate, −15,455
Unchanged groups (separate corpora, standing clean audits): cid22_train201
(metric-anchored, never human-MOS), kadid, tid, kadis 50k, konjnd_bpg
train/val.

## Recipe (frozen = the W10L9 embedded argv verbatim, paths swapped, seeds {4003,4004,4005})
L0 (0-hidden), target human_score ×100, epochs 120, pairs/epoch 50k,
coarse-decay 1e-5, max-features 944, the full winsor/signed_cbrt transform
list from the repro, group weights identical.

## Gates (frozen)
Same as the campaign selection: freeze_check E.4 over the fullevals
(--regime 944; the resliced boards), M3a measured, packed via
`bake_dial_refit pack` (default anchor/verify as W10L9_s4003_packed).
**Comparison row:** the incumbent W10L9_s4003 on the same panels — the wave
answers "does purity-clean training cost or gain?" with the E.4 + gate-panel
numbers side by side. No auto-freeze: the result returns to the user.

## RESULTS — the purity retrain WINS on both lenses (2026-08-28)

3/3 seeds trained (verbatim recipe, pure views), packed with the exact
campaign parity invocation (identity gates BIT-identical on all 2,035
anchors). Selection (freeze_check E.4, incumbent included):

| rank | bake | floors | bal_comp | M3a | sel_comp | sdr25 |
|---|---|---|---|---|---|---|
| **1** | **W10L9P_s4005_packed** | **8/8** | 0.8565 | **0.8744** | **0.9876** | 0.9612 |
| 2-3 | W10L9_s4003(_packed) — incumbent | 8/8 | 0.8549 | 0.8626 | 0.9843 | 0.9527 |
| 4 | W10L9P_s4003_packed | 8/8 | 0.8567 | 0.8278 | 0.9808 | 0.9539 |
| 5 | W10L9P_s4004_packed | 7/8 | 0.8491 | 0.8822 | 0.9814 | 0.9678 |

Gate panel AGREES: winner weighted_goal **0.764 vs 0.727**, g1=g7=1.0, dial
mono 0.9947, reach 90.6. Axis detail: cid22 0.8901 (+0.003), imazen26 0.9298
(+0.009), nonphoto 0.9342 (+0.008); honest dips: konjnd 0.4446 (−0.054),
hfnlproxy 0.3781 (−0.042) — the family-filtered rows carried near-threshold/
near-lossless signal; floors hold 8/8 regardless. **Answer to the wave's
question: family-clean training GAINS on the headline axes and the selection
rule — the freeze proposal moves to `W10L9P_s4005_packed`** (user-gated).
Sha: recorded in the board fulleval; bakes at
`/mnt/v/output/zensim/bakes/sdr-pure-2026-08-28/`.
