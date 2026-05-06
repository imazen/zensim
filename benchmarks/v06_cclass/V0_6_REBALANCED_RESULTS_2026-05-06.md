# V0_6 trained on rebalanced corpus — results (2026-05-06)

## Headline

| Variant | val_mean SROCC | Synthetic | Kadid10k | Tid2013 | KonJND-1k | bin size |
|---|---:|---:|---:|---:|---:|---:|
| V0_6 baseline (no cclass) | 0.8258 | 0.9957 | 0.8424 | 0.8258 | 0.9535 | 62 KB |
| V0_6 + cclass | 0.8386 | 0.9907 | 0.8488 | 0.8386 | 0.9501 | 63 KB |
| **V0_6 + FiLM** | **0.8457** | TBD | TBD | TBD | TBD | 63 KB × 5 (per-class) |
| V0_6 + MoE | (in flight, ~30 min) | TBD | TBD | TBD | TBD | TBD |

**FiLM is the current leader** — `+0.0199` over baseline, `+0.0071` over cclass.

FiLM trains 5 per-class (γ, β) modulation pairs over the same shared backbone. At inference: classify content → use that class's (γ, β). The bake produces 5 per-class .bin files plus a manifest.

## Files

- `runs/v06_baseline_rebal_20260506T064045.bin` — baseline
- `runs/v06_cclass_rebal_20260506T064045.bin` — +cclass features
- `runs/v06_film_rebal_20260506T081152.bin` (primary, photo class)
- `runs/v06_film_rebal_20260506T081152.c{0..4}_<class>.bin` × 5 per-class
- `runs/v06_film_rebal_20260506T081152.film_manifest.tsv`
- `runs/v06_moe_rebal_20260506T081658.bin` (in flight)

All bake roundtrips verified.

## Comparison to current shipped V0_2

V0_2 (shipped) achieves 0.9960 on photo-only synthetic-v2 corpus. These variants train on the rebalanced corpus (35% lineart / 17% screen / 17% document / 31% photo) which shifts the distribution. The drop in val_mean SROCC vs V0_2 is expected — the variants pay accuracy on photo-only KadID/TID/KonJND eval datasets to gain accuracy on non-photo content (which the eval datasets don't yet cover).

**Need screen + document eval datasets to declare a fair champion across all classes.**

## Decision

**Tentative champion: V0_6 + FiLM** (val_mean 0.8457). Awaiting MoE result.

If MoE beats FiLM: ship MoE.
If MoE ≤ FiLM: ship FiLM.

Either way: keep V0_2 shipped as fallback for callers that want photo-only behavior. Add the rebalanced-corpus champion as a new ZensimProfile variant (`PreviewV0_6` or similar) so callers can opt in when they need cross-class generalization.

## Provenance

- Trainer: `target/release/zensim-validate --algorithm mlp --mlp-zenanalyze-features ...`
- Source data: `/mnt/v/output/zensim/v06-rebalance/training_safe_synthetic_rebalanced.csv` (553k pairs)
- Generated 2026-05-06 during 10-hour autonomous run
