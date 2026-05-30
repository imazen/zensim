#!/usr/bin/env python3
"""Convert the stored dial-grid parquet (image_id, codec, q, f0..f371) into the
(features CSV + manifest TSV) pair that `qsweep_eval` consumes. Used by
`scripts/eval_panel.sh` to rescore any bake against the stored feature set
WITHOUT re-encoding/re-extracting.

Usage:
  python3 scripts/dial_grid_to_qsweep.py <dial_grid.parquet> <out_features.csv> <out_manifest.tsv>
"""
import sys
import pyarrow.parquet as pq

def main():
    grid, feat_csv, manifest = sys.argv[1], sys.argv[2], sys.argv[3]
    t = pq.read_table(grid)
    n = t.num_rows
    img = t.column("image_id").to_pylist()
    codec = t.column("codec").to_pylist()
    q = t.column("q").to_pylist()
    fcols = [t.column(f"f{i}").to_pylist() for i in range(372)]
    with open(feat_csv, "w") as fc, open(manifest, "w") as mf:
        fc.write("ref_basename,human_score," + ",".join(f"f{i}" for i in range(372)) + "\n")
        mf.write("ref_path\tdist_path\timage_id\tcodec\tq\n")
        for r in range(n):
            fc.write(f"{img[r]},{q[r]}," + ",".join(f"{fcols[i][r]:.6g}" for i in range(372)) + "\n")
            mf.write(f"-\t-\t{img[r]}\t{codec[r]}\t{q[r]}\n")
    print(f"wrote {feat_csv} + {manifest} ({n} rows)", file=sys.stderr)

if __name__ == "__main__":
    main()
