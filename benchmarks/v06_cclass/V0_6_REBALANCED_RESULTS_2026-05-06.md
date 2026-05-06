# V0_6 trained on rebalanced corpus — preliminary results (2026-05-06)

| Variant | val_mean SROCC | Synthetic | Kadid10k | Tid2013 | KonJND-1k | bin size |
|---|---:|---:|---:|---:|---:|---:|
| **V0_6 baseline** (no cclass) | 0.8258 | 0.9957 | 0.8424 | 0.8258 | 0.9535 | 62 KB |
| **V0_6 + cclass** | **0.8386** | 0.9907 | 0.8488 | 0.8386 | 0.9501 | 63 KB |
| V0_6 + FiLM (in flight) | TBD | TBD | TBD | TBD | TBD | TBD |
| V0_6 + MoE (queued) | TBD | TBD | TBD | TBD | TBD | TBD |

**cclass beats baseline by +0.0128 SROCC val_mean** — small but real improvement on a held-out 4-dataset average. Mostly comes from Tid2013 (+0.013) and Kadid10k (+0.006), with a slight regression on KonJND-1k (-0.003). Synthetic regresses 0.005 (tiny).

## Caveats

- This is the FIRST training on the rebalanced corpus (17,629 sources, 31% photo / 35% lineart / 17% screen / 17% document) — not directly comparable to V0_2 shipped (0.9960 on photo-only synthetic-v2 corpus).
- val_mean drops vs V0_2 because the eval datasets (Kadid/Tid/KonJND) test photo distortions; rebalanced corpus shifts training distribution toward non-photo, hurting photo-eval performance ~0.16 SROCC.
- This is the right tradeoff if the goal is a metric that handles ALL content types well (vs only photo). Need to add screen/document eval datasets before declaring a champion.

## Files

- `runs/v06_baseline_rebal_20260506T064045.bin` — 61,724 bytes
- `runs/v06_cclass_rebal_20260506T064045.bin` — 63,044 bytes

Both at `/mnt/v/output/zensim/v06-rebalance/runs/`. Bake roundtrip verified.

## Next steps

- Wait on FiLM (in flight ~10-15 min)
- Wait on MoE (queued after FiLM)
- Eval all 4 + V0_2 + V0_6 dct_hf reigning champion on a STANDARD eval (CID22 photo + screen subset + lineart subset)
- Pick champion → bake → replace `Self::latest()` in zensim/src/profile.rs
