#!/usr/bin/env python3
"""Convert a ZSFC v3 features.bin + paired training CSV
into the (ref_basename, human_score, f0..f227) format that
train_v_next_mlp.py's --human-csv flag expects.

ZSFC v3 layout (per zensim-validate/src/main.rs:340-461):
  4 bytes  magic = b"ZSFC"
  4 bytes  version (u32, must be 3)
  4 bytes  num_scales
  1 byte   blur_passes
  4 bytes  blur_radius
  4 bytes  _reserved
  4 bytes  n_pairs (u32)
  2 bytes  n_features (u16)
  2 bytes  name_len (u16)
  name_len bytes of utf8 dataset name
  n_pairs × 4 bytes valid_indices (u32)
  n_pairs × n_features × 4 bytes f32 features
  for each pair: 2 bytes klen (u16), klen bytes of utf8 ref_key
"""
import argparse
import struct
import pandas as pd
from pathlib import Path

def parse_zsfc_v3(path: Path):
    data = path.read_bytes()
    pos = 0
    def take(n):
        nonlocal pos
        b = data[pos:pos+n]; pos += n
        return b
    magic = take(4)
    assert magic == b"ZSFC", f"Bad magic: {magic!r}"
    version = struct.unpack("<I", take(4))[0]
    assert version == 3, f"v3 only, got {version}"
    num_scales = struct.unpack("<I", take(4))[0]
    blur_passes = take(1)[0]
    blur_radius = struct.unpack("<I", take(4))[0]
    _reserved = struct.unpack("<I", take(4))[0]
    n_pairs = struct.unpack("<I", take(4))[0]
    n_features = struct.unpack("<H", take(2))[0]
    name_len = struct.unpack("<H", take(2))[0]
    name = take(name_len).decode("utf-8", errors="replace")
    print(f"  ZSFC v3: {n_pairs:,} pairs × {n_features} features, "
          f"name={name!r}, scales={num_scales}, blur={blur_passes}/{blur_radius}")

    valid_indices = struct.unpack(f"<{n_pairs}I", take(4*n_pairs))
    # Features as f32 little-endian
    nbytes = n_pairs * n_features * 4
    feat_flat = struct.unpack(f"<{n_pairs*n_features}f", take(nbytes))
    # Reshape
    import numpy as np
    features = np.asarray(feat_flat, dtype=np.float32).reshape(n_pairs, n_features)
    # Ref keys
    ref_keys = []
    for _ in range(n_pairs):
        klen = struct.unpack("<H", take(2))[0]
        ref_keys.append(take(klen).decode("utf-8", errors="replace"))
    print(f"  parsed {len(ref_keys):,} ref_keys, features shape {features.shape}")
    return valid_indices, features, ref_keys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True, help="ZSFC v3 features.bin path")
    ap.add_argument("--csv", required=True, help="Paired training CSV with source_path + gpu_ssimulacra2")
    ap.add_argument("--out", required=True, help="Output CSV path")
    ap.add_argument("--clip-mode", default="clip01", choices=["clip01", "raw100", "minmax"],
                    help="How to rescale target into trainer space: "
                         "clip01 = clip(0,100)/100 (default, safe for safesyn-shape corpora); "
                         "raw100 = divide by 100 only, no clip (preserves negative ssim2 for "
                         "very-low-q pairs in KonJND); "
                         "minmax = per-corpus (target - min) / (max - min), maps to [0, 1].")
    ap.add_argument("--target-col", default="gpu_ssimulacra2",
                    help="Column from --csv to use as score (will be clipped to [0,100] then divided by 100)")
    args = ap.parse_args()

    print(f"Parsing {args.bin}...")
    valid_idx, feats, ref_keys = parse_zsfc_v3(Path(args.bin))

    print(f"Loading {args.csv}...")
    csv = pd.read_csv(args.csv)
    print(f"  {len(csv):,} rows, columns: {list(csv.columns)}")
    if args.target_col not in csv.columns:
        raise SystemExit(f"--target-col {args.target_col!r} not in CSV columns")
    if "source_path" not in csv.columns:
        raise SystemExit("expected source_path column")

    # The valid_indices map .bin rows to original CSV rows (0-indexed).
    # The features.bin only contains pairs where features were successfully
    # extracted; some original CSV rows may be missing.
    n_keep = len(valid_idx)
    print(f"  features.bin has {n_keep:,} valid pairs (subset of CSV's {len(csv):,})")

    # Use valid_indices to align CSV rows to .bin rows.
    csv_rows = csv.iloc[list(valid_idx)].reset_index(drop=True)

    # Build ref_basename: stem of source_path
    ref_basenames = csv_rows["source_path"].map(lambda p: Path(p).stem)

    # Sanity check: ref_keys from .bin should at least somewhat match
    # ref_basenames. They might differ if .bin was built with a different
    # naming scheme.
    matches = sum(1 for a, b in zip(ref_basenames[:1000], ref_keys[:1000])
                  if str(a) == str(b))
    print(f"  ref_basename / ref_key match in first 1000: {matches}/1000")
    if matches < 100:
        print(f"  WARNING: sample mismatch — using ref_keys from .bin instead")
        ref_basenames = pd.Series(ref_keys)

    raw_target = csv_rows[args.target_col].astype(float)
    if args.clip_mode == "clip01":
        target = raw_target.clip(0, 100) / 100.0
    elif args.clip_mode == "raw100":
        target = raw_target / 100.0
    elif args.clip_mode == "minmax":
        lo, hi = raw_target.min(), raw_target.max()
        if hi <= lo:
            raise SystemExit(f"--clip-mode minmax: target column has zero range ({lo}..{hi})")
        target = (raw_target - lo) / (hi - lo)
    else:
        raise SystemExit(f"unknown --clip-mode {args.clip_mode}")
    print(f"  target range after {args.clip_mode}: [{target.min():.4f}, {target.max():.4f}], mean {target.mean():.4f}")

    out_df = pd.DataFrame({
        "ref_basename": ref_basenames.values,
        "human_score": target.values,
    })
    n_features = feats.shape[1]
    print(f"  building {n_features} feature columns...")
    for i in range(n_features):
        out_df[f"f{i}"] = feats[:, i]

    print(f"Writing {args.out} ({len(out_df):,} rows × {len(out_df.columns)} cols)...")
    out_df.to_csv(args.out, index=False)
    sz = Path(args.out).stat().st_size
    print(f"  done; {sz/1024/1024:.1f} MB")

if __name__ == "__main__":
    main()
