#!/usr/bin/env python3
"""Build chunks.jsonl for EXP-MULTI-CODEC sweep.

Targets the same 200 sources used in the existing canonical LARGE corpus,
re-encoding them with v06-style multi-codec grids: zenwebp (denser),
zenavif (denser), zenjxl (full v06 grid for replacement of stale v12 data).

Output format matches v06 sweep / onstart_v3.sh.
"""
import json
import sys
from pathlib import Path

# Sources: 200 basenames from existing iwssim sidecar.
SOURCES_FILE = Path('/tmp/iwssim_200_sources.txt')
# Source directory on disk (also mirrored to R2 for workers).
SOURCES_DIR = Path('/mnt/v/input/zensim/sources_gen_v06rb/')
OUT_PATH = Path('/tmp/chunks_multi_codec_2026-05-18.jsonl')

# Quality grid — same 10 q points as v06.
Q_GRID = "5,15,25,35,45,55,65,75,85,95"

# Grids: trim each from v06 to keep budget in line. Aim for ~150 cells/source
# zenwebp: 2 methods × 10q = 20 cells/src × 200 = 4,000 (vs 1,000 existing)
# zenavif: 4 speeds × 2 complex × 10q = 80 cells/src × 200 = 16,000 (vs 3,900)
# zenjxl: 4 effort × 19 dist × 3 biters = 228 cells/src × 200 = 45,600 (vs 32,000)
# Total: ~65,600 cells. CHUNK_SIZE=2 → ~32,800 chunks.
# Each chunk is ~300 cells (2 sources × ~150 cells avg). At ~30 min/chunk, 10 boxes ≈ 16hr wall.
# That's TOO BIG. Cut aggressively:

KNOB_GRIDS = {
    "zenwebp": json.dumps({
        "method": [4, 6],
    }),  # 2 × 10 = 20 cells/src
    "zenavif": json.dumps({
        "speed": [3, 5, 7],
        "complex_prediction_modes": [False, True],
    }),  # 3 × 2 × 10 = 60 cells/src
    "zenjxl": json.dumps({
        "effort": [5, 7],  # Cut from 4 efforts to 2 (mid + high quality)
        "distance": [0.1, 0.5, 1.0, 1.5, 2.5, 4.0, 6.0, 10.0],  # 8 distances (cut from 19)
        "butteraugli_iters": [0, 1],  # 2 (cut from 3)
    }),  # 2 × 8 × 2 = 32 cells/src (jxl ignores q)
}

# Cell counts (per source):
# zenwebp: 2 × 10 = 20
# zenavif: 3 × 2 × 10 = 60
# zenjxl: 2 × 8 × 2 = 32 (q is ignored for jxl)
# Total: 20+60+32 = 112 cells/source × 200 = 22,400 cells
# CHUNK_SIZE=2 → ~11,200 chunks. STILL too many.
# Bumping CHUNK_SIZE=10 (10 src × 112 cells = ~1,120 cells/chunk, ~3-4 hr/chunk)
# → 200/10 = 20 chunks per codec × 3 codecs = 60 chunks
# On 10 boxes at 30 min average per box-allocated-chunk-share = 3 hr wall, ~$15.

METRICS = ["zensim", "ssim2_gpu", "butteraugli"]
CHUNK_SIZE = 10  # 10 sources per chunk → 200/10 = 20 chunks per codec


def main():
    if not SOURCES_FILE.exists():
        print(f"ERROR: {SOURCES_FILE} not found", file=sys.stderr)
        sys.exit(1)
    with SOURCES_FILE.open() as f:
        sources = [line.strip() for line in f if line.strip()]
    sources.sort()
    print(f"# {len(sources)} source images from {SOURCES_FILE}", file=sys.stderr)

    n_chunks = 0
    with OUT_PATH.open("w") as f:
        for codec, knob_grid in KNOB_GRIDS.items():
            for i in range(0, len(sources), CHUNK_SIZE):
                chunk_srcs = sources[i:i + CHUNK_SIZE]
                chunk_id = f"{codec}-{i // CHUNK_SIZE:04d}"
                spec = {
                    "codec": codec,
                    "chunk_id": chunk_id,
                    "q_grid": Q_GRID,
                    "knob_grid": knob_grid,
                    "metrics": METRICS,
                    "images": chunk_srcs,
                }
                f.write(json.dumps(spec))
                f.write("\n")
                n_chunks += 1

    print(f"# wrote {n_chunks} chunks to {OUT_PATH}", file=sys.stderr)
    cells_per_src = {
        "zenwebp": 2 * 10,
        "zenavif": 3 * 2 * 10,
        "zenjxl": 2 * 8 * 2,
    }
    total_cells = sum(cells_per_src.values()) * len(sources)
    print(f"# total cells (estimate): {total_cells}", file=sys.stderr)
    print(f"# Q grid: {Q_GRID}", file=sys.stderr)
    print(f"# Metrics: {METRICS}", file=sys.stderr)
    print(f"# CHUNK_SIZE: {CHUNK_SIZE}", file=sys.stderr)


if __name__ == "__main__":
    main()
