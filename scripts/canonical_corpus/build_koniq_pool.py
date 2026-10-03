"""KonIQ-10k as a SOURCE-IMAGE POOL (registered 2026-10-03, docs/DATA_SPLITS.md registry row).

KonIQ-10k has no reference images, so its MOS can never label a full-reference pair and is never a target. It is a pool of
content-diverse in-the-wild photos (Hosu et al., TIP 2020) for synthetic coverage ladders; MOS and the authors' indicators are
kept only to choose clean sources (e.g. high MOS, high JPEG quality factor).

  python3 build_koniq_pool.py [--root /mnt/v/datasets/koniq10k_extracted]

Input: the official zips in /mnt/v/datasets/koniq10k/ (sha256 in SHA256SUMS there), unpacked into --root:
  1024x768/*.jpg, koniq10k_scores_and_distributions.csv, koniq10k_indicators.csv
Output: <root>/koniq10k_pool.parquet (+ .manifest.json), one row per SCORED image (10,073). The 1024x768 zip carries 300 further
JPEGs that are not in the score file; they are listed in the manifest and excluded.

Split: by source image, `int(sha256(image_name), 16) % 10 < 8` -> "train", else "holdout". Any coverage leg built on the pool
uses the train rows only; the holdout rows are reserved for a later read of that leg.
"""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

N_SCORED = 10073
SIZE = (1024, 768)
INDICATORS = ("brightness", "contrast", "colorfulness", "sharpness", "quality_factor", "bitrate", "deep_feature")


def split_of(name: str) -> str:
    return "train" if int(hashlib.sha256(name.encode()).hexdigest(), 16) % 10 < 8 else "holdout"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="/mnt/v/datasets/koniq10k_extracted")
    args = ap.parse_args()
    root = Path(args.root)
    img_dir = root / "1024x768"
    scores = list(csv.DictReader(open(root / "koniq10k_scores_and_distributions.csv", newline="")))
    ind = {r["image_id"]: r for r in csv.DictReader(open(root / "koniq10k_indicators.csv", newline=""))}
    if len(scores) != N_SCORED or len({r["image_name"] for r in scores}) != N_SCORED:
        raise ValueError(f"expected {N_SCORED} distinct scored images, got {len(scores)}")
    on_disk = {p.name for p in img_dir.glob("*.jpg")}
    scored = {r["image_name"] for r in scores}
    if not scored <= on_disk:
        raise ValueError(f"{len(scored - on_disk)} scored images missing from {img_dir}")
    cols = {k: [] for k in ("image_name", "path", "sha256", "width", "height", "mos", "sd", "mos_zscore", "c1", "c2", "c3", "c4",
                            "c5", "c_total", *INDICATORS, "split")}
    for r in sorted(scores, key=lambda r: r["image_name"]):
        name = r["image_name"]
        p = img_dir / name
        with Image.open(p) as im:
            if im.size != SIZE:
                raise ValueError(f"{name}: size {im.size}, expected {SIZE}")
        i = ind.get(Path(name).stem)
        if i is None:
            raise ValueError(f"{name}: no indicators row")
        cols["image_name"].append(name)
        cols["path"].append(str(p))
        cols["sha256"].append(hashlib.sha256(p.read_bytes()).hexdigest())
        cols["width"].append(SIZE[0])
        cols["height"].append(SIZE[1])
        cols["mos"].append(float(r["MOS"]))
        cols["sd"].append(float(r["SD"]))
        cols["mos_zscore"].append(float(r["MOS_zscore"]))
        for c in ("c1", "c2", "c3", "c4", "c5", "c_total"):
            cols[c].append(int(r[c]))
        for c in INDICATORS:
            cols[c].append(float(i[c]))
        cols["split"].append(split_of(name))
    out = root / "koniq10k_pool.parquet"
    pq.write_table(pa.table(cols), out)
    commit = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    sums = (root.parent / "koniq10k" / "SHA256SUMS")
    manifest = {
        "rows": N_SCORED,
        "split_counts": {s: cols["split"].count(s) for s in ("train", "holdout")},
        "split_rule": "int(sha256(image_name), 16) % 10 < 8 -> train",
        "unscored_excluded": sorted(on_disk - scored),
        "parquet_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "zip_sha256": sums.read_text() if sums.is_file() else None,
        "build_commit": commit,
        "builder": "scripts/canonical_corpus/build_koniq_pool.py",
        "role": "source-image pool; MOS never a target",
    }
    Path(str(out) + ".manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: manifest[k] for k in ("rows", "split_counts", "parquet_sha256")} |
                     {"unscored_excluded": len(manifest["unscored_excluded"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
