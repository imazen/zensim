# Acumen Mode A — architectural finding

**Date**: 2026-05-21
**Tracking**: [imazen/zensim#40](https://github.com/imazen/zensim/issues/40)
**Status**: KNOWN LIMITATION — fix requires different code path

## Summary

Gate A Path A (existing MLP × acumen features) and Path B
(retrained MLP × acumen features) both showed null or slightly
negative results. The root cause is **architectural, not a tuning
miss**:

> Post-hoc multiplicative scaling of already-pooled features
> cannot add information — only preserve or remove. Information
> theory upper-bounds any rescaling at "no improvement"; the MLP
> can learn any monotonic transform of the original features.

Acumen Mode A as wired in `zensim-gpu/src/pipeline.rs` Phase 4
multiplies basic feature slots 10/11/12 per (scale, channel) by
per-(scale, channel) castleCSF weights. Those weights range
0.03-1.0 — at the low end this REMOVES information (compresses
feature dynamic range). The MLP can recover only what survives.

## Empirical confirmation

Path A: existing V_0_2 MLP applied to acumen-modulated features

| Corpus | Δ-SROCC |
|---|---:|
| KADID | -0.0085 |
| TID | -0.0017 |
| AIC-3 | -0.0066 |
| CID22 | -0.0076 |

Path B: fresh MLP trained on acumen-modulated features (KADID +
TID + AIC-3 train, CID22 val)

| | KADID | TID | AIC-3 | CID22 (held-out) |
|---|--:|--:|--:|--:|
| Acumen | 0.9101 | 0.2795 | 0.9493 | 0.6924 |
| Baseline | 0.9124 | 0.2635 | 0.9467 | 0.7044 |
| Δ | -0.0023 | +0.0160 | +0.0026 | **-0.0120** |

Both protocols show the same direction: acumen Mode A modulation
loses a small but consistent amount of signal on held-out data.

## Why this was structurally predictable

The zensim feature vector encodes multi-scale band content via
per-pyramid-level features (slots 0-12 per scale, channel). Each
scale's `(mean, L2, L4, SSIM-art, mutual-masking, HF-energy)` is
ALREADY a band-specific statistic. The MLP's first-layer weights
already learned to combine these scales/channels with appropriate
band importance.

Acumen Mode A's per-(scale, channel) castleCSF weight provides
the **same kind of signal** the MLP already extracts implicitly.
Multiplying by it can't add a degree of freedom — it can only
linearly re-weight what's already there. The MLP can express
this rescaling via its own first-layer weights without help.

Worse, the actual castleCSF values at typical viewing conditions
include 0.03 weights for high-frequency band-0 at low luminance.
That 30× compression of band-0 HF features wipes signal that the
MLP could otherwise use.

## What WOULD help

For castleCSF to ADD information to the trained metric, the
signal must enter the pipeline **before pooling**, not after.
Three architecturally-distinct paths:

1. **CSF-weighted band energy** — modulate the pyramid's per-band
   energy DURING the multi-scale decomposition, before
   computing means/L2/L4. The MLP would then see
   castleCSF-shaped band energy instead of raw energy, and could
   learn weighting on top of that. Requires touching the
   pyramid construction code, not just the post-pool feature
   slots.

2. **Per-pixel local-adaptation modulation (Mode B)** — instead
   of one global image-mean L, use per-pixel L. The contrast
   energy at each pixel gets weighted by castleCSF at the local
   L. The downstream features then aggregate locally-adapted
   contrast. This is what CVVDP and HDR-VDP do. Cost: per-pixel
   LUT lookup at extraction time (~14 ms/MP scalar).

3. **Acumen-as-auxiliary-feature** — leave the existing 228
   features unmodified, append the 12 castleCSF weights
   (f228..f239) as additional input features. The MLP can then
   LEARN to use the CSF weights as context. Simplest of the
   three; ships as a 240-feature regime variant.

Option 3 is the smallest code change and the cheapest test. If
auxiliary CSF weights help, that's evidence Mode B (or the deeper
pyramid integration of option 1) would help more.

## Recommendation

**Don't ship acumen Mode A in its current post-hoc-multiplier
shape.** It cannot help by construction and may hurt slightly.

The acumen FOUNDATION (castleCSF LUT loader, ViewingCondition,
Phase 4 wiring infrastructure, sweep image, local pipeline,
228-feature GPU extractor with acumen toggle) remains a useful
prerequisite for:

- HDR-IQA work where viewing condition genuinely matters at
  inference time (Mode B per-pixel L_adapt)
- The per-pair distortion-tolerance map (RDO use case, priority 2
  in the tracking issue)
- Option 3 above (acumen-as-auxiliary-feature) if pursued

Document the finding, preserve the infrastructure, defer further
acumen-Mode-A work until the architectural fix is in scope.

## Files

- Foundation: `zensim/src/acumen/` (5 commits on
  feat/acumen-foundation branch)
- zensim-gpu wiring: `zenmetrics/crates/zensim-gpu/src/pipeline.rs`
  Phase 4 + `crates/zensim-gpu/examples/extract_acumen_features.rs`
- Gate A Path A report:
  `zensim/benchmarks/acumen_gate_a_path_a_2026-05-21.md`
- Gate A Path B report:
  `zensim/benchmarks/acumen_gate_a_path_b_2026-05-21.md`
- This architectural note: this file.

## Money spent

- vast.ai: ~\$0.21 on 4 instances that failed to launch before I
  pivoted to local CUDA.
- Local CUDA: electricity only.
- Total: well under \$1 for a definitive Gate A answer.

Per the user mandate ("until money is well spent"), the iteration
budget was used to:

1. ✅ Ship the entire acumen foundation across both repos
2. ✅ Build + push the v26-acumen Docker image to ghcr.io
3. ✅ Discover + document vast.ai launch quirks (3 separate
   bugs in the launcher)
4. ✅ Pivot to local CUDA when safesyn dist images turned out to
   exist locally
5. ✅ Build the 228-feature GPU extractor as a new
   zensim-gpu example
6. ✅ Run Gate A Path A (10 min, all 4 val corpora)
7. ✅ Run Gate A Path B (3 min train + ~12 min features)
8. ✅ Identify the architectural reason the test was predestined
   to fail

Further iteration on the same architectural axis isn't
cost-effective. The next investment is one of the three
architecturally-distinct paths above, scoped as separate work.
