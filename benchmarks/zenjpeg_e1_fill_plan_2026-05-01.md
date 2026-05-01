# zenjpeg-420-e1 fast-fill plan for the SSIM2 0-90 band

Goal: densify the synthetic training set with a fast jpegli-AQ
sequential variant (`zenjpeg-420-e1`, no Huffman opt, no trellis), at
a q-grid that lands roughly uniformly across SSIM2 0-90.

This is cheap because zenjpeg at effort=1 is the fastest mode in the
zenjpeg family — ~20-40 ms/MP encode + decode, vs ~500-2000 ms/MP for
AVIF (-e6) or JXL (-e7). We can backfill 26+ q-levels for ~3,579
sources in ~30-45 minutes wall time on the existing GPU pipeline.

## Codec patch

Apply `benchmarks/zenjpeg_e1_codec_2026-05-01.patch` to
`coefficient/examples/generate_zensim_training.rs`. It adds one new
entry in `build_codecs()`:

```rust
codecs.push(Box::new(ZenjpegCodec::with_effort(
    Subsampling::S420,
    false,
    1,  // jpegli AQ baseline + sequential, no Huffman opt
)));
```

The codec naming machinery (`config_id` → `e1` suffix at stock effort)
will produce the codec ID `zenjpeg-420-e1-v0.3.1`, distinct from the
existing `zenjpeg-420-e2-v0.3.1`.

## q-grid

Target SSIM2 distribution per band after the fill, by extrapolating
from existing zenjpeg-420-e2 q-→SSIM2 medians (e1 lands roughly
3 SSIM2 points lower at the same q since no Huffman optimization):

| q | est. ssim2 median (e1) | target band |
|--:|--:|---|
| 2 | 5 | 0-25 |
| 3 | 10 | 0-25 |
| 4 | 14 | 0-25 |
| 5 | 15 | 0-25 |
| 6 | 18 | 0-25 |
| 7 | 21 | 0-25 |
| 8 | 24 | 0-25 |
| 9 | 27 | 25-40 |
| 10 | 29 | 25-40 |
| 12 | 33 | 25-40 |
| 14 | 37 | 25-40 |
| 15 | 41 | 40-60 |
| 16 | 42 | 40-60 |
| 18 | 45 | 40-60 |
| 20 | 50 | 40-60 |
| 22 | 52 | 40-60 |
| 24 | 54 | 40-60 |
| 25 | 56 | 40-60 |
| 26 | 57 | 40-60 |
| 28 | 58 | 40-60 |
| 30 | 59 | 40-60 |
| 32 | 60 | 40-60 |
| 35 | 61 | 60-75 |
| 38 | 62 | 60-75 |
| 40 | 62 | 60-75 |
| 42 | 63 | 60-75 |
| 45 | 64 | 60-75 |
| 48 | 64 | 60-75 |
| 50 | 65 | 60-75 |
| 52 | 65 | 60-75 |
| 55 | 66 | 60-75 |
| 58 | 66 | 60-75 |
| 60 | 68 | 60-75 |
| 65 | 70 | 60-75 |
| 70 | 73 | 60-75 |
| 75 | 74 | 60-75 |
| 80 | 76 | 75-90 |
| 87 | 80 | 75-90 |
| 90 | 83 | 75-90 |
| 95 | 87 | 75-90 |

That's 39 q-levels. 3,579 sources × 39 q × 1 codec = 139,581 new pairs.

## Generator command

```bash
cd /home/lilith/work/coefficient

# unlock ledger before re-running (generator chmod 0o444's it after writes)
chmod 644 /mnt/v/output/zensim/synthetic-v2/metric-ledger.jsonl

CUDA_PATH=/usr/local/cuda-12.6 \
LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/lib64/stubs:/usr/lib/wsl/lib \
LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/lib64/stubs:/usr/lib/wsl/lib \
    cargo run --release --features "gpu,all-codecs,zenwebp" \
    --example generate_zensim_training -- \
    --sources /mnt/v/input/zensim/sources \
    --codecs zenjpeg-420-e1 \
    --jpeg-qualities "2,3,4,5,6,7,8,9,10,12,14,15,16,18,20,22,24,25,26,28,30,32,35,38,40,42,45,48,50,52,55,58,60,65,70,75,80,87,90,95" \
    --remote /mnt/v/input/zensim \
    --output /mnt/v/output/zensim \
    2>&1 | tee /tmp/gen-zenjpeg-e1.log
```

Notes:
- `--codecs zenjpeg-420-e1` filters via substring match; only the new
  codec runs (mozjpeg, zenjpeg-e2, zenjpeg-xyb-e2, zenwebp, avif, jxl
  are all skipped).
- `--jpeg-qualities` applies to all JPEG-family codecs that pass the
  filter, which is just zenjpeg-420-e1 here.
- Existing pairs skip via the generator's
  `encoded_path.exists() && decoded_path.exists()` resumption check
  (no-op if accidentally re-run).
- `--avif-qualities` / `--jxl-qualities` left unset — those codec
  families don't run because the filter excludes them.

After completion: rebuild the safe-synthetic CSV by appending the new
`metric-ledger.jsonl` rows minus the 49 CID22 validation stems, then
extract features for the new pairs (the existing feature-cache code
handles append).

## Wall-time estimate

| stage | per-pair | 139,581 new pairs |
|---|--:|--:|
| zenjpeg-e1 encode (CPU, jpegli AQ baseline) | ~25 ms × 0.5 MP | ~18 min @ 32 cores |
| decode + re-decode for SSIM2 | ~5 ms | ~4 min |
| GPU SSIMULACRA 2 + Butteraugli scoring | ~5-8 ms | ~12-18 min |
| ledger + CSV writes | trivial | ~1 min |
| **Total** | | **~35-45 min** |

For comparison: a full 6-codec sweep at the same q-grid would be
~6h dominated by AVIF -e6 + JXL -e7.

## Expected band redistribution

After fill (218,089 existing + 139,581 new = 357,670 pairs):

| band | existing pairs | new pairs (est.) | total | new % |
|---|--:|--:|--:|--:|
| ≤ 0 | 13,183 | ~3,000 | ~16,000 | 4.5% |
| 0-25 | 16,239 | **~25,000** | ~41,000 | 11.5% |
| 25-40 | 17,016 | **~14,000** | ~31,000 | 8.7% |
| 40-60 | 38,020 | **~43,000** | ~81,000 | 22.6% |
| 60-75 | 45,686 | **~46,000** | ~92,000 | 25.7% |
| 75-90 | 60,964 | ~14,000 | ~75,000 | 21.0% |
| ≥ 90 | 26,981 | ~0 | ~27,000 | 7.5% |

The 25-60 SSIM2 band that was the user-flagged gap goes from 25%
of training to 31% — over 100,000 pairs. The 0-25 band roughly
doubles from 7.4% to 11.5%. Distribution is much more uniform.

## Post-fill retrain plan

The new dataset balance has zenjpeg-family at ~63% of training pairs
(up from 56%) — risk of MLP over-fitting to JPEG artifacts. Two
options:

### Option A: weight the new codec down at training time

```bash
zensim-validate \
  --dataset /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv \
  --format synthetic \
  --target-metric gpu-ssim2 \
  --feature-cache <new timestamped cache> \
  --feature-tier peaks \
  --train --algorithm mlp --mlp-hidden 64 --mlp-epochs 200 \
  --mlp-magnitude-match-lambda 0.001 --mlp-magnitude-match-alpha 30.0 \
  --mlp-zenanalyze-tsv /mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1.tsv \
  --mlp-zenanalyze-features dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio \
  --mlp-output /mnt/v/output/zensim/synthetic-v2/runs/v06_dct_hf_e1fill_<TS>.bin \
  --also kadid10k:/mnt/v/dataset/kadid10k,tid2013:/mnt/v/dataset/tid2013,konjnd1k:/mnt/v/datasets/KonJND-1k/KonJND-1k \
  --mlp-validation-policy min \
  --dataset-weights "zenjpeg-420-e1:0.4,zenjpeg-420-e2:1.0,zenjpeg-420-xyb-e2:1.0,mozjpeg-rs-420-e4:1.0,zenavif-s5-e6:1.0,zenjxl-e7:1.0,zenwebp-default-m4:1.0"
```

`zenjpeg-420-e1:0.4` halves-and-some the gradient contribution per pair
so the JPEG-family share in the *effective* training signal stays close
to the pre-fill ~56%. Other codecs stay at 1.0 (full weight).

(Note: `--dataset-weights` currently keys on dataset *name*, not
codec. The synthetic dataset is one group regardless of codec. To
weight per-codec we'd need a small change to mlp_train.rs to support
codec-level pair weighting from the synthetic CSV's codec column. See
Option B if that's not in scope.)

### Option B: sampler bias on the SSIM2-band axis

Modify `mlp_train.rs` triplet sampler to draw 50% of training pairs
from the SSIM2 [0, 60] band and 50% from [60, 100]. This already
does the right thing without per-codec weighting: codec balance is
naturally maintained because all 6 codecs contribute to both bands,
just unevenly. Code change: maybe 30 lines in the
`sample_pair_indices` helper.

### Recommended: Option B + retrain

It avoids the per-codec weighting complexity and directly attacks the
band-imbalance problem the fill is meant to solve. The expected
effect after sampler bias + e1 fill:
- 25-40 band SROCC: 0.90 → ~0.95 (V0_6 dct_hf)
- 40-60 band SROCC: 0.96 → ~0.97
- 0-25 band SROCC: 0.91 → ~0.94
- 75-90 band SROCC: 0.98 → 0.97 (slight regression, acceptable)

## Sequence of commits

If you go ahead, here's the suggested order:

1. **Apply codec patch** to `coefficient/examples/generate_zensim_training.rs`,
   commit + push to coefficient.
2. **Run generator** with the q-grid above. ~35-45 min wall.
3. **Append new pairs** to `training_safe_synthetic.csv` (create
   `training_safe_synthetic_extended.csv` to keep the original immutable),
   re-extract features for the new rows only.
4. **Generate updated zenanalyze TSV** (existing TSV covers all source
   stems, so this is a no-op unless any new sources were added).
5. **Land sampler-bias change** in `zensim-validate/src/mlp_train.rs`,
   commit + push.
6. **Retrain V0_6 dct_hf** with sampler bias + extended dataset.
   ~15-20 min training (cached features, hot-path inner loop).
7. **Re-run 4-metric eval** on the new V0_6 bake. ~5 min.
8. **Update final report** with the new per-band SROCC numbers.

Total wall: ~80-100 min from "go" to "updated report committed."
