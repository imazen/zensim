#!/usr/bin/env python3
"""Join the per-encode identity, the ssim2 sidecar, and every candidate's score
into ONE table keyed on `encoded_filename`.

Inputs
  keys_tbig_pools944.parquet   identity + score_ssim2 (the bigcodec TRAIN view)
  scores_<model>.parquet       from score_encodes.py, one per candidate
  bytes/{encodes,decoded}/     the persisted bitstreams + decoded PNGs

Output
  encodes.parquet   one row per encode that has (a) bytes on disk, (b) an ssim2
                    score, and (c) a score from EVERY candidate. Rows failing
                    any of those are dropped and counted, never silently
                    imputed — a missing score is not a zero.

The reference-rendition path is resolved here too, so the staging step never
has to guess where a source PNG lives.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

KEYS = Path(
    "/mnt/v/zen/zensim-training/wlin7-pools944-2026-08-30/keys_tbig_pools944.parquet"
)
BYTES_ROOT = Path("/mnt/v/zen/zensim-training/wlin7-pools944-2026-08-30/bytes")
REF_DIR = Path("/mnt/v/output/clean-picker-corpus-2026-06-26")
# The 39 dial-grid reference images are the DIAL panel's own instrument. Human
# labels on them would feed back into the axis they are used to measure, so
# they are excluded by name even though the tbig pool does not contain them
# (verified intersection 0 — this stays as a standing guard, not a live filter).
DIAL_CLASSES = Path("benchmarks/dial_grid_content_classes_2026-08-31.tsv")


def dial_grid_refs(repo: Path) -> set[str]:
    p = repo / DIAL_CLASSES
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        first = line.split("\t")[0].strip()
        if first and first != "image_id":
            out.add(first)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    a = ap.parse_args()

    t0 = time.time()
    man = json.loads((a.scores_dir / "_MANIFEST_scores.json").read_text())
    models = [r["model"] for r in man["records"]]

    keys = pq.read_table(
        KEYS,
        columns=[
            "ref_basename",
            "encoded_filename",
            "codec",
            "q",
            "score_ssim2",
            "origin_id",
            "view",
        ],
    ).to_pydict()
    n_all = len(keys["encoded_filename"])
    print(f"[join] keys rows={n_all}", flush=True)

    score_maps: dict[str, dict[str, float]] = {}
    for m in models:
        t = pq.read_table(a.scores_dir / f"scores_{m}.parquet").to_pydict()
        score_maps[m] = dict(zip(t["encoded_filename"], t["score"]))
        print(f"[join] {m}: {len(score_maps[m])} scores", flush=True)

    guard = dial_grid_refs(a.repo)

    cols: dict[str, list] = {
        "encoded_filename": [],
        "ref_basename": [],
        "ref_path": [],
        "encode_path": [],
        "decoded_path": [],
        "codec": [],
        "q": [],
        "origin_id": [],
        "view": [],
        "ssim2": [],
        **{f"m_{m}": [] for m in models},
    }
    drops = {"no_bytes": 0, "no_decoded": 0, "no_ssim2": 0, "no_ref": 0, "dial_guard": 0}
    drops.update({f"no_score_{m}": 0 for m in models})
    ref_cache: dict[str, str] = {}

    for i in range(n_all):
        ef = keys["encoded_filename"][i]
        rb = keys["ref_basename"][i]
        if rb.rsplit(".", 1)[0] in guard or rb in guard:
            drops["dial_guard"] += 1
            continue
        s2 = keys["score_ssim2"][i]
        if s2 is None:
            drops["no_ssim2"] += 1
            continue
        ms = {}
        miss = None
        for m in models:
            v = score_maps[m].get(ef)
            if v is None:
                miss = m
                break
            ms[m] = v
        if miss is not None:
            drops[f"no_score_{miss}"] += 1
            continue
        ep = BYTES_ROOT / "encodes" / ef
        if not ep.exists():
            drops["no_bytes"] += 1
            continue
        dp = BYTES_ROOT / "decoded" / (ef.rsplit(".", 1)[0] + ".png")
        if not dp.exists():
            drops["no_decoded"] += 1
            continue
        rp = ref_cache.get(rb)
        if rp is None:
            cand = REF_DIR / rb
            if not cand.exists():
                drops["no_ref"] += 1
                continue
            rp = str(cand)
            ref_cache[rb] = rp

        cols["encoded_filename"].append(ef)
        cols["ref_basename"].append(rb)
        cols["ref_path"].append(rp)
        cols["encode_path"].append(str(ep))
        cols["decoded_path"].append(str(dp))
        cols["codec"].append(keys["codec"][i])
        cols["q"].append(keys["q"][i])
        cols["origin_id"].append(keys["origin_id"][i])
        cols["view"].append(keys["view"][i])
        cols["ssim2"].append(float(s2))
        for m in models:
            cols[f"m_{m}"].append(float(ms[m]))

    a.out.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(cols), a.out, compression="zstd")
    kept = len(cols["encoded_filename"])
    print(
        f"[join] kept={kept} of {n_all}  refs={len(set(cols['ref_basename']))}  "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )
    print(f"[join] drops={drops}", flush=True)
    (a.out.parent / "_MANIFEST_encodes.json").write_text(
        json.dumps(
            {
                "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "keys_source": str(KEYS),
                "bytes_root": str(BYTES_ROOT),
                "ref_dir": str(REF_DIR),
                "models": models,
                "rows_in": n_all,
                "rows_kept": kept,
                "refs_kept": len(set(cols["ref_basename"])),
                "drops": drops,
                "dial_grid_guard_names": len(guard),
                "score_manifest": str(a.scores_dir / "_MANIFEST_scores.json"),
            },
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
