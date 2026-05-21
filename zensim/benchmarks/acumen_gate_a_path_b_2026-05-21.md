# Gate A Path B — acumen Mode A retrain test (preliminary)

**Date**: 2026-05-21
**Branch**: `imazen/zensim feat/acumen-foundation` + `imazen/zenmetrics feat/acumen-gpu`
**Tracking**: [imazen/zensim#40](https://github.com/imazen/zensim/issues/40)

## Question

Path A showed that applying castleCSF Mode A modulation to the
existing V_0_2 MLP causes a small (-0.005 to -0.009) SROCC loss.
The expected interpretation: the MLP wasn't trained for the new
feature scale. Path B asks the deeper question: does training a
**fresh MLP** on acumen-modulated features find useful signal that
the un-modulated training corpus doesn't already have?

## Method

1. Extracted 228-feature parquets via `zensim-gpu`
   `extract_acumen_features` example for KADID (10125), TID (2950
   after 50 NaN failures from rare unsupported image variants),
   AIC-3 (600), CID22 (4292) pairs. Both `--acumen-mode-a` (ppd=56,
   peak=100 cd/m², ambient=5 cd/m²) and baseline.
2. Joined human_score from canonical val parquets (row-position
   alignment verified at 5 sample rows for KADID).
3. Trained two MLPs via `zensim_mlp_train`:
   - **Acumen Path B**: train on KADID + TID + AIC-3 acumen
     features, validate on CID22 acumen features
   - **Baseline Path B**: same recipe on baseline features
4. Hidden 64, epochs 50, pairs-per-epoch 30k, lr 0.001, seed 1
   (identical hyperparameters for both bakes)

## Result

| Variant | KADID | TID | AIC-3 | **CID22 (held-out)** |
|---|--:|--:|--:|--:|
| Acumen Path B (best epoch 30) | 0.9101 | 0.2795 | 0.9493 | **0.6924** |
| Baseline Path B (best epoch 30) | 0.9124 | 0.2635 | 0.9467 | **0.7044** |

**Δ-SROCC on held-out CID22: -0.012** (acumen worse by 0.012).

## Interpretation

Acumen Mode A modulation, as currently wired (only the 3 HF
band-energy slots per scale-channel), reduces information content
slightly. The retrained MLP cannot recover the per-band CSF signal
that other unmodulated features (mean, L2, L4, SSIM-art,
mutual-masking) already implicitly capture via the multi-scale
pyramid.

This **falsifies Gate A as currently formulated** for the V_0_2
slot in zensim-gpu's 228-feature regime. Three doors remain open:

1. **Wider modulation**: scale ALL 19 features per scale-channel
   by the band CSF weight, not just HF band-energy slots
   10/11/12. May break feature meanings (mean / L2 norms get
   rescaled) but tests stronger acumen application.
2. **Different viewing conditions**: ppd=90 (mobile retina),
   peak=300 / 1000 nits (HDR). Per the 2-anchor falsification,
   different ppd produces meaningfully different weights —
   maybe HDR-specific viewing helps.
3. **Mode B (per-pixel L_adapt)**: instead of image-mean L,
   compute L per-pixel and apply castleCSF per-pixel. Higher
   compute cost (~14 ms/MP instead of ~170 ns/image) but more
   faithful to the paper. Adds local-adaptation signal that
   image-mean misses.

## Compute economics

- Extraction: 4 val corpora × 2 variants in ~12 min on local
  RTX 5070. Free (electricity).
- Training: 2 MLPs × 27s each = ~1 min total on CPU.
- Total Gate A loop: ~13 min from features to verdict.
- vs vast.ai fleet estimate: ~3 hours + ~\$30.

Local CUDA + the new `extract_acumen_features` GPU example reduces
the Gate A loop to under 15 minutes. Future iterations on
alternative acumen wirings can run trivially.

## Files

- Feature parquets: `/home/lilith/acumen-data/<corpus>_features_{acumen,baseline}_2026-05-21.parquet`
- Trainer-ready parquets (with human_score): `/home/lilith/acumen-data/<corpus>_train_{acumen,baseline}.parquet`
- Trained MLPs: `/home/lilith/acumen-data/mlp_path_b_3of4_{acumen,baseline}.bin`
- Extractor binary: `target/release/examples/extract_acumen_features`
- Pipeline script: `imazen/zenmetrics scripts/sweep/gate_a/gate_a_pipeline.sh`

## Verdict

Gate A as currently wired: **FALSIFIED preliminary** (-0.012 SROCC
on held-out CID22 with retrained MLP). Three follow-up
investigations queued (wider modulation / different viewing /
Mode B per-pixel). Safesyn extraction (196k pairs) is running in
background for a future full-corpus retrain that confirms the
preliminary result on the canonical training distribution.

The acumen FOUNDATION (LUT loader, ViewingCondition, Phase 4
weighting wiring, CLI flags, sweep image, local pipeline) remains
useful infrastructure even with this null result — it's the
prerequisite for the three follow-up directions and for any
future per-pixel HDR work.
