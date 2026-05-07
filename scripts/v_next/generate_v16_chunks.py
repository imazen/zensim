#!/usr/bin/env python3
"""Generate sweep-v16 chunk JSONL files for cross-codec coverage.

Per the v_next handoff TODO §4.2: zensim training data is JPEG-heavy.
We need ~50–100k cells per non-JPEG codec on the same 1024 px corpus
that fed v15r/v15rc, with knob grids that cover the structural-distortion
axes the model hasn't seen at scale.

This script produces three JSONL files (one per codec) following the
chunk-per-image convention used by scripts/sweep/v15/chunks_gpu.jsonl
in zenmetrics. Each line is one image × full knob grid; the worker
encodes the Cartesian product per cell.

Outputs go to /mnt/v/zen/zensim-training/2026-05-07/v16-chunks/ — the
caller is responsible for uploading them to
s3://coefficient/jobs/<run_id>/chunks.jsonl when launching workers.

Usage:
    python3 scripts/v_next/generate_v16_chunks.py
"""
from __future__ import annotations

import json
from pathlib import Path

# Source: 981 PNGs Lanczos3-resampled to 1024 px max-dim, mirrored from
# /tmp/v15r-prep/stage_1024 to s3://zentrain/sweep-v15r-2026-05-06/sources/.
# We keep the same source list for direct cross-codec comparability; image
# basenames are read from the v15r features TSV which has one row per source.
FEATURES_TSV = Path("/mnt/v/zen/zensim-training/2026-05-07/v15r-prep/features_v15r_combined.tsv")

# 11-step quality grid covering the production range.
# Densely sampled at low-q where the structural-distortion modes
# (block boundary, transform-edge ringing, gaborish patterns) live.
Q_GRID = "5,10,15,20,25,30,35,40,50,60,70,80,90,95"

METRICS = ["zensim", "ssim2-gpu", "butteraugli-gpu"]

# Per-codec knob grids. Sized to ~30–60 knob variants so cell-count
# per image stays in the [400, 1000] band. With 981 images this yields
# 400k–1M cells per codec (close to the §4.2 acceptance criterion of
# ~500k per codec without going overboard).
KNOB_GRIDS: dict[str, dict] = {
    # zenwebp expert knob axes:
    #   - method 0..6: speed/effort tradeoff (default 4)
    #   - segments 1..4: rate-distortion segments (default 1)
    #   - sns_strength 0..100: spatial noise shaping (default 50)
    #   - filter_strength 0..100: deblock filter (default 60)
    "zenwebp": {
        "method": [4, 6],
        "segments": [1, 4],
        "sns_strength": [0, 50, 80],
        "filter_strength": [0, 60],
        # method×segments×sns×filter = 2×2×3×2 = 24 knob variants
    },
    # zenavif expert knob axes:
    #   - speed 0..10: encoder effort (default 6)
    #   - lossless: occasional lossless reference cells
    #   - partition_range: AV1 partition tree depth control
    "zenavif": {
        "speed": [4, 6, 8],
        "lossless": [False],
        # speed×lossless = 3 knob variants (smaller — AVIF cells are slow)
    },
    # zenjxl expert knob axes (require feature=jxl on the metrics binary):
    #   - effort 1..9: encoder effort (default 7)
    #   - patches: patch-based prediction
    #   - gaborish: gabor smoothing pre-pass
    #   - error_diffusion: rate-distortion ED
    #   - progressive: progressive scan ordering
    "zenjxl": {
        "effort": [3, 7],
        "gaborish": [True, False],
        "patches": [True, False],
        # effort×gaborish×patches = 2×2×2 = 8 knob variants
    },
}


def load_image_basenames() -> list[str]:
    """Read the 981-image v15r corpus list from the zenanalyze features TSV."""
    if not FEATURES_TSV.exists():
        raise SystemExit(f"missing {FEATURES_TSV}; run from a host with the "
                         f"v15r mirror at /mnt/v/zen/zensim-training/")
    images: list[str] = []
    with FEATURES_TSV.open() as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            ip_idx = header.index("image_path")
        except ValueError:
            raise SystemExit("features TSV has no `image_path` column")
        for line in f:
            cols = line.rstrip("\n").split("\t")
            ip = cols[ip_idx]
            base = ip.rsplit("/", 1)[-1]
            images.append(base)
    return sorted(set(images))


def main() -> int:
    out_dir = Path("/mnt/v/zen/zensim-training/2026-05-07/v16-chunks")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = load_image_basenames()
    n_q = len(Q_GRID.split(","))
    print(f"Corpus: {len(images)} images × {n_q} q-levels")

    summary = []
    for codec, knobs in KNOB_GRIDS.items():
        run_id = "sweep-v16{}-2026-05-07".format(
            {"zenwebp": "w", "zenavif": "a", "zenjxl": "j"}[codec])
        out_path = out_dir / f"chunks_{run_id}.jsonl"
        knob_json = json.dumps(knobs)
        n_knob_combos = 1
        for vs in knobs.values():
            n_knob_combos *= len(vs)
        n_cells_per_image = n_q * n_knob_combos
        n_total = n_cells_per_image * len(images)

        chunks = []
        for i, img in enumerate(sorted(images)):
            chunk_id = f"{codec}-{i:04d}"
            chunks.append({
                "codec": codec,
                "chunk_id": chunk_id,
                "q_grid": Q_GRID,
                "knob_grid": knob_json,
                "metrics": METRICS,
                "images": [img],
                "run_id": run_id,
            })
        with out_path.open("w") as f:
            for c in chunks:
                f.write(json.dumps(c) + "\n")
        summary.append((codec, run_id, len(chunks), n_knob_combos,
                        n_cells_per_image, n_total))
        print(f"  {codec:<10} → {out_path.name}: {len(chunks)} chunks × "
              f"{n_q} q × {n_knob_combos} knobs = {n_total:,} cells")

    print("\nUpload to R2 + adapt scripts/sweep/v15/launch_gpu.sh:")
    for codec, run_id, n_chunks, _, _, n_total in summary:
        print(f"  aws s3 cp {out_dir}/chunks_{run_id}.jsonl "
              f"s3://coefficient/jobs/{run_id}/chunks.jsonl")
    print()
    print("Don't forget to mirror sources to each run prefix:")
    for _, run_id, _, _, _, _ in summary:
        print(f"  aws s3 sync s3://zentrain/sweep-v15r-2026-05-06/sources/ "
              f"s3://zentrain/{run_id}/sources/  (or a CopyObject loop)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
