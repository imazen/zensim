# V_20b — Distortion-Manifold Pre-Training (design + scope)

**Status**: design doc + Python-prototype scope, 2026-05-15.
**Paper**: Su et al. 2023 — distortion-manifold contrastive pre-training
for IQA. Claim: +0.04 SROCC over BRISQUE on LIVE-Challenge low-q
cluster.
**Priority**: HIGHEST per `docs/v0_20_path_evaluation_2026-05-14.md`
ranking for B0..B5 lift (per CLAUDE.md "B0..B5 lift is the dominant
priority" directive).

## The gap to close (concrete)

Per `benchmarks/v0_18_ship_reference_card_2026-05-14.md`:

| CID22 band | n | V_18 SROCC | fast-ssim2 SROCC | Δ |
|---|---:|---:|---:|---:|
| **B3 [30, 40)** | **57** | **0.0246** | **0.1335** | **−0.109 LOSS** |
| B6 [60, 70) | 836 | 0.3943 | 0.4173 | −0.023 |
| B7 [70, 80) | 1092 | 0.3936 | 0.3974 | −0.004 |

The CID22 B3 [30, 40) band (heavy compression, low-quality regime) is
where V_18 underperforms most. **Acceptance gate for V_20b**: CID22
B3 [30, 40) ≥ 0.13 (matches fast-ssim2 there) AND aggregate ≥ 0.8933
(matches V_18) AND no >0.005 KADID/TID regression.

## Mechanism (paraphrased from Su 2023)

Standard IQA training (RankNet on labeled MOS pairs) wastes 99% of
the corpus: only the small labeled subset (~10k images with MOS
labels) drives gradient. Distortion-manifold pre-training uses **all
218k (ref, dist) pairs** even without MOS labels by:

1. **Encoder φ** maps a feature vector (or image, in the paper's
   case) into an embedding space.
2. **Self-supervised pre-training objective**: pairs with similar
   distortion strength should embed close together; pairs with
   different distortion strength should embed apart.
   - Specifically: triplet loss — for anchor `(ref, mild-dist)`,
     positive `(ref, similar-distortion-strength-on-same-ref)`,
     negative `(ref, very-different-distortion-strength)`,
     ‖φ(anchor) − φ(positive)‖² + α < ‖φ(anchor) − φ(negative)‖².
3. **Distortion-strength proxy**: codec quality parameter (zq). Within
   `(image, codec)` group, sort rows by zq, sample triplets where
   anchor + positive are at adjacent zq values and negative is far.
4. **Fine-tune**: freeze encoder (or partially), train a small
   regression/ranking head on the labeled MOS subset (KADID + TID +
   CID22-train-fold).

## Our adaptation

The Su 2023 paper uses raw images + a CNN encoder. We have
**pre-extracted 228 features per pair**, so our encoder operates on
the feature vector directly. Architecturally:

```
input: 228-dim pair-features  (existing)
      ↓
encoder φ: 228 → 64 (LeakyReLU)
      ↓
embedding e ∈ R^64
      ↓
head ψ: 64 → 1
      ↓
output: zensim score
```

Pre-training: encoder φ only, on 218k unlabeled pairs via triplet
loss + zq-based triplet sampler.

Fine-tune: head ψ on labeled corpora (CID22 + KADID + TID), with
encoder either frozen or partially unfrozen (smaller LR).

This is a **two-layer MLP, same wire format as V_18**. Bakes go through
ZNPR v3 unchanged. The only architectural change is the training
procedure (which becomes Python during prototyping, since the Rust
trainer is RankNet-only).

## Concrete training plan

### Phase 0 — Corpus prep (existing)

- ✅ `/mnt/v/zen/zensim-training/2026-05-07/v06-features/safe_synth_ssim2_features.parquet`
  (340k rows, 228+ features; `ref_basename` carries the codec quality
  info via path stem)
- Need to recover `zq` per row from `ref_basename` or join from the
  source `info_with_bitrates.csv` (TBD: confirm the path)

### Phase 1 — Triplet sampler (Python, ~2 days)

```python
# Pseudocode
def sample_triplet(rows_per_image):
    """Per (ref_image, codec) group, sample (anchor, positive, negative)."""
    sorted_by_zq = sorted(rows, key=lambda r: r.zq)
    n = len(sorted_by_zq)
    anchor_idx = random.randint(0, n - 2)
    # Positive: adjacent zq (within 5)
    pos_idx = anchor_idx + 1
    # Negative: zq far away (random non-adjacent)
    neg_pool = [i for i in range(n) if abs(i - anchor_idx) > 4]
    neg_idx = random.choice(neg_pool) if neg_pool else (n - 1)
    return (sorted_by_zq[anchor_idx], sorted_by_zq[pos_idx], sorted_by_zq[neg_idx])
```

Run-length validation: pre-training reaches stable encoder loss in
~200 epochs at batch 256 (per Su 2023 Fig. 4). ~6 GPU-hours on 4090.

### Phase 2 — Contrastive pre-training (Python + PyTorch, ~2 days)

Triplet loss with margin α = 0.5 (paper default):

```
L = max(0, ‖φ(a) − φ(p)‖² − ‖φ(a) − φ(n)‖² + α)
```

InfoNCE alternative tracked as ablation; default is triplet for paper
fidelity.

### Phase 3 — Fine-tune head (~1 day)

Either: (a) freeze encoder, train head with RankNet on
KADID+TID+CID22-train-fold; OR (b) joint fine-tune at LR/10 for both.
Output bake = standard 228→64→1 MLP, ZNPR v3.

Validation: V_18 reference card gate (CID22 0.8933 + B3 ≥ 0.13).

### Phase 4 — Sweep (~2 days compute)

Embedding dim ∈ {32, 64, 128}, encoder depth ∈ {1, 2, 3 layers},
margin α ∈ {0.2, 0.5, 1.0}, triplet sampler radius ∈ {3, 5, 10}.
Pareto-rank by (CID22 agg, B3) SROCC.

### Phase 5 — Bake + ship decision (~half day)

Convert best PyTorch checkpoint to ZNPR v3 via the existing zentrain
PyTorch→ZNPR pipeline. Validate cross-corpus, write methodology doc,
ship as V_20b candidate or archive based on V_18 ref card numbers.

## Risks + open questions

1. **`zq` extraction**: ref_basename may not carry zq directly. Need
   to confirm the join path. If unavailable, fall back to using
   `human_score` (the ssim2 score baked into the corpus) as the
   strength proxy — slightly noisier but workable.

2. **Synth corpus generalization**: contrastive pre-training on
   synthetic distortions may not transfer to CID22's
   "authentically-distorted" images (the FRIQUEE caveat —
   Ghadiyaram 2017 found synthetic-trained models broke on authentic
   data). Mitigation: include CID22-train-fold in fine-tune; report
   cross-corpus generalization as primary acceptance.

3. **MLP capacity ceiling**: V_18 ship's 372→384→1 architecture may
   already saturate the available information in our 228 features.
   The encoder pre-training adds non-linear feature recombination
   that single-pass training may not discover, but if features are
   the bottleneck, the lift is bounded. Mitigation: V_20c (LMS + opponent
   features) adds new feature columns — could compose.

4. **Python prototype vs Rust port**: prototyping in Python is faster;
   shipping requires Rust to keep the trainer self-hosted. After
   falsification/validation in Python, port the winning recipe to the
   Rust trainer (zensim-validate/src/mlp_train.rs adds a contrastive
   loss path).

## Effort summary

| Phase | Effort | Cumulative |
|---|---|---|
| 0 Corpus prep + zq join | 0.5 day | 0.5 |
| 1 Triplet sampler | 2 days | 2.5 |
| 2 Contrastive pre-train + GPU compute | 2 days + 6 GPU-hr | 4.5 |
| 3 Fine-tune head | 1 day | 5.5 |
| 4 Hyper sweep | 2 days compute | 7.5 |
| 5 Bake + methodology doc + ship decision | 0.5 day | 8 |
| **Total** | **~1 week + ~12 GPU-hours** | |

Total compute cost: ~$5-10 on vast.ai (4090) or local if a free GPU window opens.

## Acceptance gate (from V_18 ref card)

A V_20b candidate ships iff ALL hold:

- CID22 aggregate SROCC ≥ 0.8933 (V_18 ship floor)
- CID22 B3 [30, 40) SROCC ≥ 0.13 (fast-ssim2 floor — the gap V_20b targets)
- CID22 B4–B8 within ±0.02 of V_18 (no mid-band damage)
- KADID aggregate ≥ 0.9377 (V_18 −0.005 tolerance)
- TID aggregate ≥ 0.9476 (V_18 −0.005 tolerance)
- AIC-3 Anchor cross-corpus SROCC ≥ 0.90

Falsification: if no candidate clears B3 ≥ 0.13 after Phase 4 sweep,
the V_20b path is **falsified for B0..B5 lift via distortion manifold
on our feature space**. Pivot to V_20c (LMS + opponent features) or
revisit feature-extraction layer.

## Open question for the user

Should the Python prototype be merged into the existing zentrain
pipeline at `/home/lilith/work/zen/zenanalyze/zentrain/`, or live
standalone in `/home/lilith/work/zen/zensim/scripts/v_next/`?

The zentrain PRINCIPLES.md (2026-05-14) was updated to reference
zensim's ZNPR v3 + 2026-05-14-clean corpus. Putting V_20b in
zentrain keeps it next to the picker work it complements. Putting
it in zensim/scripts keeps it self-contained for this codebase.

Recommendation: scripts/v_next/v0_20b/ — self-contained, doesn't
couple to zentrain release cadence, easier to falsify and discard
if the mechanism doesn't transfer.

## References

- Su et al. 2023 — original paper (need to locate the exact citation;
  the path eval doc reference is to "Su 2023" with claim +0.04 SROCC
  over BRISQUE on LIVE-Challenge low-q cluster)
- `benchmarks/v0_18_ship_reference_card_2026-05-14.md` — acceptance gate
- `docs/v0_20_path_evaluation_2026-05-14.md` — path ranking
- `benchmarks/v0_20_v0_21_design_2026-05-14.md` — V_20 umbrella design
