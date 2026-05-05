# Mixture-of-Experts (MoE) for V0_6 zensim

This document describes the MoE architecture, the on-disk bake format,
and how to train + evaluate MoE models. The implementation is gated
behind the `moe` cargo feature on `zensim-validate` and `zensim-bench`.
Without that feature the existing V0_6 + concat + FiLM paths are
unchanged.

## Why MoE

V0_6 ships a single 228+3 → 64 → 1 MLP — one set of weights for every
content class. FiLM adds per-class affine modulation to that hidden
layer; on the current 99.5 %-photo corpus FiLM degenerates to "single
photo modulation" because every eval pair lives in class 0. The
modulation acts as a training-time regularizer on photo content,
which is why FiLM ships on KADID + TID but barely moves CID22.

MoE is the next step up the routing ladder: K full sub-networks, each
free to specialize on its own slice of the input distribution. With
the rebalanced corpus in flight (`v06-rebalance` worktree), every
content class will see enough pairs to actually train its expert
distinctly — at which point a learned gate over (content features +
cclass) routes pairs through the right expert and the K models can
diverge usefully. The architecture is in place now so we can fire as
soon as the corpus lands.

## Architecture

```text
input  x ∈ R^N            (N = n_features = n_content + n_cclass = 228 + 3 + 5 = 236)

gate (small MLP):
  h_g = ReLU(W_g1 · x + b_g1)                  (R^Hg, default Hg = 32)
  z   = W_g2 · h_g + b_g2                      (R^K, raw logits)
  w   = softmax(z / τ)                         (R^K, mixture weights, default τ = 1)

experts k ∈ 0..K (each is a V0_6-shaped 2-layer MLP):
  h_k = LeakyReLU(W_e1[k] · x + b_e1[k])       (R^H, default H = 64)
  y_k = w_e2[k] · h_k + b_e2[k]                (scalar)

output:
  y = Σ_k w_k · y_k
```

Defaults match V0_6: `n_hidden = 64`, `leaky_alpha = 0.01`,
RankNet pairwise loss, magnitude matching at λ ≈ 1e-3, L2 1e-5 on
weights only.

K defaults to the number of trailing `cclass_*` columns in the
zenanalyze TSV — typically 5 for the current schema. Override with
`--mlp-moe-experts <N>` to let the gate learn its own routing
without per-class identity.

### Loss

```text
L = L_RankNet                          # pairwise sigmoid cross-entropy
  + λ_mag · (Δy − α · Δh)²             # magnitude-matching auxiliary
  + λ_l2  · ‖W‖²                       # weight decay (per-pair)
  + λ_lb  · KL(uniform || w)           # load-balance: penalize gate collapse
```

`λ_lb · KL(uniform || w)` is what stops the gate from snapping to
"always expert 0" early. The KL is applied per-pair (not over a
running batch mean); the gradient flows through softmax exactly like
the RankNet term. Default `λ_lb = 0.01`. Set to 0 to disable.

### Why a learned gate over the FULL input, not just cclass?

Three reasons:

1. **Robustness to mis-classification.** The cclass tail is itself a
   prediction (from `gen_zenanalyze_features`); near class boundaries
   it can flip between classes without the underlying perceptual
   target moving much. A gate that sees only cclass would deliver a
   discontinuous routing decision at those boundaries. With the full
   input, the gate can smooth over cclass uncertainty using the 228
   content features.
2. **Specialization beyond named classes.** With K > n_cclass (or K
   < n_cclass), the gate can group images differently from the cclass
   labels — e.g. routing "screenshots with synthetic gradients" away
   from "screenshots with hard text" even though both are class 1.
3. **Hard top-1 fallback gives back the cclass dispatch when warranted.**
   When the gate is confident (max softmax > 0.95) the runtime drops
   to a single forward — same cost as FiLM's bake-per-class dispatch,
   no soft-mixing tax. The threshold is in the manifest, retunable
   without rebaking.

## On-disk bake format

Each artifact is a standard ZNPR v3 bake (the same format the V0_6
baseline uses) — *no* new bake format is introduced. Existing
`zensim::mlp::Model::from_bytes` loads every artifact unchanged.

Files written per training run with stem `<S>`:
- `<S>.bin`               — primary bake (= expert 0). Lets existing
                              single-bake eval pipes load *something*.
- `<S>.gate.bin`          — gate model. Shape: `n_features → gate_hidden → K`,
                              activations `ReLU → Identity`. The runtime
                              applies softmax(z / τ) on the K outputs.
- `<S>.e<k>_<name>.bin`   — expert k bake (k = 0..K-1). Shape:
                              `n_features → n_hidden → 1`,
                              activations `LeakyReLU → Identity`.
                              Same shape as V0_6 baseline — every
                              expert is "just an MLP".
- `<S>.moe_manifest.tsv`  — manifest tying it all together.
- `<S>.log`               — training log.

### Manifest format (`<S>.moe_manifest.tsv`)

```
# zensim MoE manifest v1
# n_experts            5
# gate_temperature     1
# hard_top1_threshold  0.95
# n_features           236
# n_hidden             64
role	index	name	bake_path
gate	-1	gate	/path/to/<S>.gate.bin
expert	0	photo	/path/to/<S>.e0_photo.bin
expert	1	screenshot	/path/to/<S>.e1_screenshot.bin
expert	2	line_art	/path/to/<S>.e2_line_art.bin
expert	3	mixed	/path/to/<S>.e3_mixed.bin
expert	4	other	/path/to/<S>.e4_other.bin
```

- `# key\tvalue` metadata lines come first; recognised keys are
  `n_experts`, `gate_temperature`, `hard_top1_threshold`,
  `n_features`, `n_hidden`. Unknown keys are ignored — the format
  is forward-compatible.
- `role` is `gate` (exactly one row, `index = -1`) or `expert`
  (K rows, `index ∈ 0..K-1`).
- `bake_path` is read as-is. Use absolute paths.

The format version is `v1`. Any forward-incompatible change must bump
the header comment + add a load-time check.

## Inference

The `dataset_metric_baseline` example loads a manifest with
`--moe-manifest <path>` and routes every pair through it instead of
the single `--v04-bake`:

```text
1. Standardize features (the gate + experts share the same scaler).
2. Run gate model → K logits → softmax(z/τ) → weights w[K].
3. argmax_k = argmax(w); max_w = w[argmax_k].
4. If max_w > hard_top1_threshold: run expert[argmax_k], return its score.
5. Otherwise: run all K experts in sequence, return Σ w_k · y_k.
```

The hard top-1 shortcut saves K-1 forwards in the high-confidence
case (which is most of the corpus). With K = 5, threshold 0.95, and a
gate that's confident 80 % of the time, the average inference cost
is `0.8 · 1 + 0.2 · 5 = 1.8` forwards — only 80 % above the V0_6
baseline of 1 forward.

## CLI invocation — training (DEFERRED until rebalanced corpus lands)

> The user explicitly requested code-only delivery. No training has
> been run on this branch. The exact invocation below is what the
> corpus-rebalance follow-up agent should use once
> `~/work/zen/zensim--v06-rebalance/` lands its rebalanced TSV.

```bash
cd ~/work/zen/zensim--v06-moe

# Run with the moe feature enabled.
cargo run --release -p zensim-validate --features moe -- \
  --algorithm mlp \
  --dataset /mnt/v/dataset/synthetic-v2 \
  --format synthetic \
  --also kadid:/mnt/v/dataset/kadid10k,tid:/mnt/v/dataset/tid2013,cid22:/mnt/v/dataset/cid22/CID22_validation_set \
  --mlp-train-also-weight 1.0 \
  --mlp-human-train-fraction 0.7 \
  --mlp-validation-policy min \
  --mlp-zenanalyze-tsv /mnt/v/dataset/zenanalyze/synthetic-v2-rebalanced.tsv \
  --mlp-zenanalyze-features dct_hf_log,dct_hf_p95,dct_hf_max,cclass_photo,cclass_screen,cclass_line,cclass_mixed,cclass_other \
  --mlp-hidden 64 \
  --mlp-epochs 200 \
  --mlp-pairs-per-epoch 50000 \
  --mlp-magnitude-match-lambda 0.001 \
  --mlp-magnitude-match-alpha 30 \
  --mlp-low-band-oversample 0.5 \
  --mlp-moe-onehot-content-class \
  --mlp-moe-experts 5 \
  --mlp-moe-gate-hidden 32 \
  --mlp-moe-gate-temperature 1.0 \
  --mlp-moe-load-balance-lambda 0.01 \
  --mlp-moe-hard-top1-threshold 0.95 \
  --mlp-output /mnt/v/output/zensim/v06-moe/runs/v06_moe_$(date -u +%Y%m%dT%H%M%S).bin
```

Expected wall-clock training cost on the production rig (Ryzen 9 7950X):
- 5 experts × ~7.3K weights each + gate ~1.5K weights ≈ 38K parameters.
- Roughly 2-3× the V0_6 baseline training time per epoch (5 experts +
  1 gate forward/backward per pair).
- 200 epochs × 50K pairs ≈ 10M pairs total. Estimate: 90-120 minutes.

## CLI invocation — evaluation (DEFERRED)

```bash
cargo run --release -p zensim-bench --features moe \
  --example dataset_metric_baseline -- \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --max-pairs 500 \
  --zenanalyze-tsv /mnt/v/dataset/zenanalyze/synthetic-v2-rebalanced.tsv \
  --zenanalyze-features dct_hf_log,dct_hf_p95,dct_hf_max,cclass_photo,cclass_screen,cclass_line,cclass_mixed,cclass_other \
  --moe-manifest /mnt/v/output/zensim/v06-moe/runs/v06_moe_<TIMESTAMP>.moe_manifest.tsv \
  --per-pair-output /mnt/v/output/zensim/v06-moe/per_pair_$(date -u +%Y%m%dT%H%M%S).csv
```

The eval reports SROCC for the V0_4 column — when `--moe-manifest` is
present that column is the MoE score. Compare against the FiLM and
V0_6-baseline numbers reigning on the rebalanced corpus to gate.

### Acceptance gates (proposed)

- KADID SROCC: `≥ FiLM` (FiLM ships +0.0165 over baseline at last
  measurement). MoE should match or exceed.
- TID2013 SROCC: `≥ FiLM` (FiLM ships +0.0051).
- CID22 SROCC: `≥ baseline V0_6` (FiLM regresses CID22 slightly; MoE
  should at minimum not make CID22 worse).
- Holdout (rebalanced corpus) min SROCC across all groups: > V0_6.

## Known limitations / DEFERRED follow-ups

- **No training has been run.** The architecture compiles, the
  trainer round-trips synthetic data (`mlp_train_moe::tests`), and
  inference works end-to-end on synthetic gate weights
  (`tests/moe_smoke.rs`). The first real run waits on the rebalanced
  corpus.
- **Inference loads the gate + K experts on every pair.** This is
  fine for the eval harness (rayon over pairs amortizes the load) but
  for production scoring we'd want a `MoeRuntime` that pre-loads the
  models once. That's a follow-up — out of scope here.
- **No `zensim::mlp` runtime API for MoE.** The eval harness
  hand-implements gate + softmax + mixture in
  `dataset_metric_baseline.rs`. Promoting it into `zensim::mlp` once
  the architecture lands would let zenpicker / production callers use
  it without reimplementing the routing logic. Tracked separately.
- **Single ZNPR v3 expert + single gate format.** A future "MoE-aware"
  ZNPR v4 could pack experts + gate into a single file; not done
  here, by design — keeping experts as standard V0_6 baseline files
  means existing tooling (`zenpredict`, dump utilities, regress
  harnesses) loads each piece directly.
