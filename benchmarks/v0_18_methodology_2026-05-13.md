# V0_18 methodology + reproducible recipe

**Ship date**: 2026-05-13
**Bin**: `zensim-experimental/weights/v0_18_2026-05-13.bin` (93,064 B, md5
`2cc537470e68f7379e759811ddd22900`)
**Architecture**: 228→384→1 LeakyReLU MLP, I8 weights with per-output
f32 scales (zenpredict v3 wire format, `WeightDtype::I8`)
**SROCC (held-out)**: CID22 0.8934, KADID10k 0.9427, TID2013 0.9525,
AIC-3 CTC 0.7998, AIC-4 0.9153
**Calibration**: affine `y' = 28.0366·y + (-5.0738)` baked into the
final layer weights/bias (inherited from V0_16)

This document is the **canonical methodology record** for V0_18. It
exists so a future reader (human or agent) can reproduce the bake
end-to-end without reading the entire `zensim_champion_log.md`.

Going forward, **every shipped V_X bake MUST land with a paired
methodology doc at `benchmarks/v0_X_methodology_YYYY-MM-DD.md`** —
see "Required per-bake documentation" section at the end.

---

## 1. Pipeline overview

V0_18 is a re-bake of **V0_17 weights** at I8 quantization. V0_17 is
itself a **3-way concat construction** built from three trained
228→128→1 single MLPs averaged at the output. The construction is
mathematically equivalent to a 3-bake output ensemble, implemented
as a single 228→384→1 MLP (3 × 128 hidden blocks concatenated) for
runtime efficiency.

```
+---------------+
| V0_16 (h=128) |\
+---------------+ \
+---------------+  \  output average  +-------+   I8     +-------+
| cycle-14 s=1  |--->  (= 3-way      ->  V0_17 | ------> | V0_18 |
+---------------+  /  concat MLP    /  +-------+   bake  +-------+
+---------------+ /  228→384→1)
| cycle-14 s=42 |/
+---------------+
       ↑              ↑                 ↑              ↑
  V0_16 recipe   per-band TV weights   weight ratios   re-bake
  4 groups       --tv-band-weights     0.65/0.30/0.05  dtype only
  seed=1 (s1)    10,30,10,30
  seed=42 (s42)
```

Quantization is bit-equivalent in SROCC: V0_18 CID22 = V0_17 CID22 =
0.8934 (4 decimals). The 73.8 % bin shrink comes for free.

## 2. Reproducing each component

### 2.1 V0_16 (h=128, F32, the base) — trained 2026-05-12

Trainer: `zensim-validate/src/bin/zensim_mlp_train.rs` at commit
`af1773ed` (zenpredict 0.2 prep) — earlier commits work too if you
adjust the `--out-dtype` flag absence.

Recipe:

```sh
cargo run --release -p zensim-validate --bin zensim_mlp_train -- \
  --group safesyn_purged:/tmp/zensim_loop/safe_synth_clean_features.csv:1.0:0.0 \
  --group kadid:/mnt/v/zen/zensim-training/2026-05-07/v06-features/kadid_features.csv:0.3:1.0 \
  --group tid:/mnt/v/zen/zensim-training/2026-05-07/v06-features/tid_features.csv:0.3:1.0 \
  --group konjnd:/tmp/zensim_loop/konjnd_aligned_features.csv:0.5:1.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 0.001 --seed 1 \
  --out-dtype f32 \
  --out /tmp/zensim_loop/v0_16_purged_tv20_seed1.bin
```

Training data hashes (verify before re-running):

| Path | MD5 | Rows |
|---|---|---:|
| `/tmp/zensim_loop/safe_synth_clean_features.csv` | `03a1725015b0ddd8cde8ce21e82b45a3` | 144,791 |
| `/mnt/v/zen/zensim-training/2026-05-07/v06-features/kadid_features.csv` | `6f07b25ce854fd6f8dc596cf055747d1` | 10,125 |
| `/mnt/v/zen/zensim-training/2026-05-07/v06-features/tid_features.csv` | `7189f57a832161c82aee3a8860c677a1` | 3,000 |
| `/tmp/zensim_loop/konjnd_aligned_features.csv` | `209ebdb6586cf76c5d57a62883b314e3` | 76,104 |

Hyperparameters (every field shown for reproducibility):
- `n_hidden=128`, `n_epochs=300`, `pairs_per_epoch=50000`
- `lr=0.001` initial, cosine annealing with 50-epoch period
- `leaky_alpha=0.01`
- `seed=1`
- `log_every=10`
- `l2_lambda=1e-5`
- `early_stop_patience=50`
- `validation_policy=Min` (best by worst-group SROCC, NOT mean)
- `low_q_boost=1.0`, `mid_q_boost=1.0` (both no-op)

V0_16 best val_mean SROCC = 0.9403, early-stopped at epoch 190.

Affine calibration applied post-training via
`scripts/v_next/affine_calibrate_znpr_v2.py`:

```sh
python3 scripts/v_next/affine_calibrate_znpr_v2.py \
  --in-bake  /tmp/zensim_loop/v0_16_purged_tv20_seed1.bin \
  --out-bake zensim-experimental/weights/archive/v0_16_2026-05-12.bin \
  --alpha 31.1041 --beta -4.3882
```

Resulting CID22 SROCC = **0.8919**.

### 2.2 cycle-14 seed=1 (per-band TV weights) — trained 2026-05-12

Same recipe as V0_16 except add `--tv-band-weights 10,30,10,30` and
`--tv-pairs-file <path>`. The TV regularizer constrains the
RankNet trajectory with per-band weights:
- B0 weight 10, B1 weight 30, B2 weight 10, B3 weight 30
- B1 and B3 pushed harder (where ssim2 has biggest gaps)

```sh
cargo run --release -p zensim-validate --bin zensim_mlp_train -- \
  <same 4 groups as V0_16> \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 0.001 --seed 1 \
  --tv-pairs-file /tmp/zensim_loop/combined_purged_tv_pairs_bands.tsv \
  --tv-weight 1.0 \
  --tv-band-weights 10,30,10,30 \
  --tv-apply-every 50 --tv-batch 32 \
  --out /tmp/zensim_loop/cycle14_seed1.bin
```

**Original doc said `tv_pairs_with_bands.tsv`** — that file name was
never on disk (audit 2026-05-14). The actually-used TV pairs file is
`/tmp/zensim_loop/combined_purged_tv_pairs_bands.tsv` (3.0 MB,
present on disk; produced by `scripts/v_next/regen_tv_pairs.py
--emit-bands` from the V0_18 base training CSVs). For V0_19+ this
file gets regenerated against the new contamination-purged training
CSV — same script, same flags, different input.

cycle-14-s1 alone: CID22 SROCC ≈ 0.8932 (per
`benchmarks/cycle_14_per_band_tv_outcomes_2026-05-13.md`).

### 2.3 cycle-14 seed=42 — same recipe, seed=42

cycle-14-s42 alone: CID22 SROCC ≈ 0.8901 but AIC-4 0.9201 (best of
any V_X). The seed lottery picks complementary failure modes vs s1.

### 2.4 3-way concat construction → V0_17

Output ensemble: `y = 0.65 · y_V0_16 + 0.30 · y_s1 + 0.05 · y_s42`.

Mathematically rewritten as a single 228→384→1 MLP by stacking the
three sub-MLPs' hidden weights into one wide hidden layer (with
diagonal block structure) and combining the three output layers
into one row vector with the mix coefficients baked in:

- Layer 0 weights: `[W_V0_16 | W_s1 | W_s42]` (228 × 384 = 3 × 128)
- Layer 0 biases: `[b_V0_16 | b_s1 | b_s42]` (384)
- Layer 1 weights: `0.65 · [w_V0_16 | 0 | 0] + 0.30 · [0 | w_s1 | 0] + 0.05 · [0 | 0 | w_s42]` (384 → 1)
- Layer 1 bias: `0.65 · b_out_V0_16 + 0.30 · b_out_s1 + 0.05 · b_out_s42`

The LeakyReLU between layers means weight-averaging (W₁ = avg of three
W₁'s) does NOT equal output-averaging — only the concat construction
is faithful. Single-MLP weight average was tested at cycle-14 and
gave CID22 0.8719 (regression).

**Reproduction script** (canonical, committed 2026-05-14 after audit
caught the original doc's broken references): use the Rust binary
`zensim-validate/src/bin/concat_three_way.rs`:

```sh
cargo run --release -p zensim-validate --bin concat_three_way -- \
  --base benchmarks/v0_19_base_seed1_2026-05-14.bin \
  --s1   benchmarks/v0_19_cycle14_s1_2026-05-14.bin \
  --s42  benchmarks/v0_19_cycle14_s42_2026-05-14.bin \
  --coeffs 0.65:0.30:0.05 \
  --out  benchmarks/v0_19_concat_3way_2026-05-14.bin
```

**Hallucination audit 2026-05-14**: The original methodology (prior
to this edit) referenced `/tmp/zensim_loop/concat_construct.py`
"during cycle-14" and `scripts/v_next/concat_three_way.py` "added
in tick 645". Both filenames were searched against the entire git
history (`git log --all --diff-filter=A --name-only`) and **neither
file was ever committed to this repo**. The on-disk V0_17 / V0_18
bake bytes are correct (their math matches the formula in this
section), but the script the original doc directed readers to was
vapor — running the documented `python3 concat_three_way.py …`
command would have failed. The Rust binary committed today is the
*real* canonical reproduction path. Future methodology docs should
ONLY reference scripts that are actually committed; check with
`git log -- <path>` before claiming a script exists.

V0_17 affine inherits from V0_16 (α=28.0366, β=-5.0738) because the
output is a linear combination of three affine-calibrated heads with
their biases baked in.

Final V0_17 bake: `benchmarks/rust_v0_X_2026-05-13_concat_3way_65_30_5.bin`
(md5 `2775812d…`, 355 KB F32).

### 2.5 V0_18 — re-bake V0_17 at I8 quant

```sh
cargo run --release -p zensim-bench --example quant_compare -- \
  zensim-experimental/weights/archive/v0_17_2026-05-13.bin /tmp/quant
# Output: /tmp/quant/v0_17_2026-05-13_i8.bin (93,064 B)

cp /tmp/quant/v0_17_2026-05-13_i8.bin zensim-experimental/weights/v0_18_2026-05-13.bin
```

The `quant_compare.rs` tool reads V0_17's f32 weights, dequantizes
to a clean f32 working buffer, then re-bakes with
`WeightDtype::I8` (per-output `max(|W_col|) / 127.0` scaling, round-
to-nearest, clamp to [-128, 127]).

Quant cost is essentially nil because V0_17 is a 3-way concat: each
column's max is dominated by one of the three sub-MLPs, so per-output
scale captures the column's dynamic range. Random-input divergence:
mean |Δ| / mean |y| = 6.9e-3 ≈ 0.7 % raw output error, which the
affine calibration absorbs without SROCC impact.

## 3. Validation harness

Every metric in §1 was produced by:

```sh
cargo run --release -p zensim-bench --example dataset_metric_baseline -- \
  --kadid /mnt/v/dataset/kadid10k \
  --tid /mnt/v/dataset/tid2013 \
  --cid22 /mnt/v/dataset/cid22/CID22_validation_set \
  --aic3 /tmp/zensim_loop/aic3_ctc_pairs.csv \
  --pairs-tsv AIC4:/tmp/aic4_pairs.tsv \
  --konjnd /mnt/v/datasets/KonJND-1k/KonJND-1k \
  --v04-bake zensim-experimental/weights/v0_18_2026-05-13.bin \
  --max-pairs 20000
```

The harness emits both 4-band CID22 Table 5 cuts AND step-5 (20-bin)
per-band SROCC per the 2026-05-13 user directive ("require more than
4 bands for eval always").

Non-monotonic q-step rate measured by:

```sh
python3 scripts/v_next/score_unified_with_bake.py \
  --bake zensim-experimental/weights/v0_18_2026-05-13.bin \
  --parquet /mnt/v/zen/zensim-training/2026-05-07/unified/unified_v15r_zenjpeg.parquet
```

V0_18 raw non-mono rate: **5.47 %** (V0_17 was 5.49 %, ssim2 GT is
5.08 %). Soft-iso projection drops both to 0 % (see
`scripts/v_next/soft_iso_smooth.py`).

## 4. Data lineage

- **Safe-synth source corpus**: 3,579 unique hex-hashed references
  from CLIC 2025 + non-CID22 portion of zensim's source pool,
  generated at `/mnt/v/input/zensim/sources/`.
- **CID22 quarantine**: 49 validation references held out across all
  generators and 361 perceptual-near-duplicates purged 2026-05-12
  per the V0_8 contamination audit. The clean safe-synth CSV
  contains 144,791 rows.
- **KADID-10k**: 81 reference images (Kodak + others), no overlap with
  hex-hashed training sources. Used for both training (weight 0.3)
  and validation.
- **TID2013**: 25 reference images, no overlap. Same dual use.
- **KonJND-1k**: 1,008 reference images, used at training weight 0.5
  for PJND alignment and at evaluation as the at-PJND ≈ 63 ± 5
  anchor (Cloudinary Table 4 calibration).
- **CID22 (validation only)**: 49 references, 4,292 distorted pairs.
  Gold standard for compression-output human MOS per CLAUDE.md goal
  #1.

## 5. Required per-bake documentation

**Effective 2026-05-13, every shipped V_X bake MUST land with a
paired methodology doc.** Add it as
`benchmarks/v0_X_methodology_YYYY-MM-DD.md` (this file is the
template) before flipping the `include_bytes!` in `profile.rs`.

The methodology doc MUST include:

1. **Architecture** — layer dims, activations, dtype, parameter count,
   final bake size (bytes + md5).
2. **Recipe** — full trainer command line, data file paths + MD5s,
   every hyperparameter (no "defaults assumed"), seed, epoch count,
   early-stop epoch reached.
3. **Lineage** — if the bake is built from prior bakes (ensemble,
   concat, finetune, knowledge distillation), document each
   component's recipe with the same rigor.
4. **Calibration** — affine α/β or whatever post-training transform
   was applied, with the script and its arguments.
5. **Validation** — held-out SROCC on the 5 canonical corpora
   (KADID, TID, CID22, AIC-3, AIC-4) plus KonJND PJND anchor; report
   both 4-band CID22 Table 5 cuts and step-5 (20-bin) per-corpus.
6. **Smoothness** — non-mono q-step rate on
   `unified_v15r_zenjpeg.parquet` (raw + after soft-iso), aggregate
   AND per-band per `zensim/CLAUDE.md` rule.
7. **Data lineage** — every training-data file: path, MD5, row count,
   provenance, contamination audit status against CID22 49 validation
   references.
8. **Honest gaps** — what the bake does WORSE than its predecessor and
   why shipping it anyway is the right trade.

A bake without methodology = **untrustworthy bake**. The model's
numbers can be reproduced; without methodology, they can't be
verified, can't be improved on, and can't survive context loss.

This rule lives in `zensim/CLAUDE.md` shipping policy and applies
to every future V_X.

---

## 2026-05-14 addendum: 10-band re-validation + KADID/TID overlap audit

After the CLAUDE.md per-band-reporting bump from 4 to 10 bands, V0_18
was re-validated on the new harness (`dataset_metric_baseline` with
the 10-band table). The bake under test is the same V0_17→V0_18 I8
re-quantization documented above, additionally rebaked through
zenpredict-bake 0.2.0 which always applies hidden-unit L2-asc reorder
(consumer-invisible, predict-output-equivalent). The compressed
variant (17,694 B via whole-bake LZ4) is numerically identical to
the uncompressed 93,064 B variant at the predict-output level.

### Aggregate (max-pairs 50000)

| Corpus | n | V0_4 | fast-ssim2 | vs ssim2 | vs prior V0_18 |
|---|--:|--:|--:|--:|--:|
| CID22 | 4292 | 0.8933 | 0.8895 | +0.0038 | −0.0001 |
| KADID10k | 10125 | 0.9427* | 0.8133 | +0.1294* | 0.0000* |
| TID2013 | 3000 | 0.9526 | 0.8460 | +0.1066 | +0.0001 |

`*` KADID number is **inflated** by training/holdout perceptual overlap
— see audit below.

### Perceptual-overlap audit (dHash-64, 2026-05-14)

Re-ran `check_holdout_overlap` against the synth-v2 safe-synthetic
training CSV (3,218 unique training sources) for both KADID and TID
references. CID22 was previously purged in the 2026-05-12 sweep
(361 contaminated sources removed, manifest at
`benchmarks/contaminated_sources_purged_2026-05-12.txt`).

| Corpus | strict d≤10 | loose d≤16 | refs affected |
|---|--:|--:|---|
| KADID10k | 6 | 118 | I04/I18/I25/I41/I71 (I18 has 4 size variants in training) |
| TID2013 | 1 | 33 | I12 |

TSVs: `benchmarks/kadid_overlap_2026-05-14.tsv`,
`benchmarks/tid_overlap_2026-05-14.tsv`.

### Interpretation

- **CID22 0.8933 is honest** — the gold-standard ship gate per
  CLAUDE.md is met (≥ fast-ssim2's 0.8895, Δ=+0.0038).
- **KADID 0.9427 is over-stated.** The 6 strict-match sources +
  118 loose-match sources mean some KADID test pairs are essentially
  in-distribution. True KADID SROCC is bounded above by 0.9427 and
  below by the corpus-uniform ssim2 floor (0.8133). Until the
  KADID-overlap sources are purged from training (deferred), treat
  this number as "V0_4 ≥ ssim2 by an unknown margin ≥ 0.0; exact
  delta inflated."
- **TID 0.9526 is mostly honest.** 1 of 25 references with strict
  overlap (and 33 of 3218 training sources at loose threshold) is
  unlikely to move the aggregate by more than ±0.005. Treat as
  approximately accurate.

### Ship gate

Per CLAUDE.md the **CID22 SROCC is the gold standard** for cross-band
evaluation. CID22 passes at 0.8933 (+0.0038 over ssim2 0.8895, within
±0.0001 of prior V0_18 baseline). The compressed V0_18 ships under
the existing zensim 0.3.0 (never published — no version bump).

KADID and TID stay as **integrity guards** but their inflated numbers
explicitly do NOT load-bear the ship decision.

### Follow-up (queued, not blocking ship)

- Purge KADID-overlap training sources (6 strict + 118 loose) and
  TID-overlap sources (1 + 33).
- Retrain V_X on the truly-clean CSV.
- Re-validate to get the honest KADID/TID numbers.
- Update CLAUDE.md's "Dataset contamination rules" to note that the
  file-name "no overlap" claim was insufficient — perceptual-hash
  audit is required.
