#!/usr/bin/env python3
"""Exercise Rev5 validation CLIs using audited vectors and synthetic labels.

Engineering smoke tests only: these labels do not measure model quality.
Requires pyarrow; binaries must already be built with cargo --release.
"""

import argparse
import json
import os
import pathlib
import struct
import subprocess

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bin-dir", type=pathlib.Path, required=True)
    parser.add_argument("--bake", type=pathlib.Path, required=True)
    parser.add_argument("--vectors-root", type=pathlib.Path, required=True)
    parser.add_argument("--work-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    root = args.work_dir
    root.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, ZENSIM_FORMULA_REV="5", TMPDIR=str(root))
    records = []

    def run(name, binary, *arguments):
        command = [str(args.bin_dir / binary), *map(str, arguments)]
        result = subprocess.run(command, env=env, capture_output=True, text=True)
        (root / f"{name}.stdout").write_text(result.stdout)
        (root / f"{name}.stderr").write_text(result.stderr)
        records.append({"test": name, "rc": result.returncode, "command": command})
        (root / "results.json").write_text(json.dumps(records, indent=2))
        assert result.returncode == 0, (name, result.stdout[-2000:], result.stderr[-2000:])
        return result.stdout

    files = sorted(args.vectors_root.glob("new_scalar_*.f64bin"))
    assert len(files) >= 8, "provide the registered Rev5 audit vectors"
    rows = [list(struct.unpack(f"<{p.stat().st_size // 8}d", p.read_bytes()))[:720] for p in files]
    assert all(len(row) == 720 for row in rows)
    columns = {
        "ref_basename": pa.array([f"ENGINEERING_{i:02d}" for i in range(len(rows))]),
        "human_score": pa.array([(i + 1) / (len(rows) + 1) for i in range(len(rows))]),
    }
    columns.update({f"f{id}": pa.array([row[id] for row in rows]) for id in range(720)})
    table = pa.table(columns)
    source_table = root / "ext_kadid.parquet"
    pq.write_table(table, source_table, compression="zstd")
    (root / "_MANIFEST.json").write_text(json.dumps({
        "formula_revision": 5, "regime": "720",
        "purpose": "Rev5 CLI engineering tests; synthetic labels, never training or quality evidence",
        "sources": list(map(str, files)),
    }, indent=2))

    stamped = root / "stamped.bin"
    run("stamp_rev5", "bake_stamp_revision", args.bake, "5", stamped)
    dense_bake = root / "dense.bin"
    run("densify_bake_rev5", "bake_dial_refit", "densify", "--in", stamped,
        "--out", dense_bake, "--gate-rows", "64")
    before = root / "before.tsv"
    after = root / "after.tsv"
    for name, bake, output in [("verdict_rev5", stamped, before), ("verdict_dense_rev5", dense_bake, after)]:
        run(name, "bake_verdict", "--bake", bake, "--regime", "720", "--corpora", "kadid",
            "--features-root", root, "--per-pair-output", output,
            "--dial-grid", root / "absent-dial.parquet", "--ramp-grid", root / "absent-ramp.parquet",
            "--corruption-grid", root / "absent-corruption.parquet")
    assert before.read_bytes() == after.read_bytes(), "densifying must preserve every scored row"
    panel_rows = root / "panel.tsv"
    panel_rows.write_text("predicted\ttarget\n" + "".join(
        f"{row[13]}\t{(i + 1)/(len(rows)+1)}\n" for i, row in enumerate(rows)))
    panel = json.loads(run("panel_rev5_values", "panel", "--input", panel_rows, "--json"))
    assert panel, "panel must emit parsed statistics"
    dense_table = root / "dense-table.parquet"
    run("densify_table_rev5", "rescore_parquet", "--densify", "--input", source_table,
        "--output", dense_table, "--keep-ids", "0-227,372-719")
    dense = pq.read_table(dense_table)
    assert all(dense[f"f{id}"].equals(table[f"f{id}"]) for id in list(range(228)) + list(range(372, 720)))
    assert not any(f"f{id}" in dense.column_names for id in range(228, 372))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
