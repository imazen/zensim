#!/usr/bin/env python3
"""Write `_MANIFEST.json` for an AIC2026 scoring artifact directory.

Every generated-data directory in this workspace carries a manifest with a
`build_commit`, per-file hashes and the exact argv that produced it (ML Data
Pipeline Discipline §2: without `build_commit`, "is this still valid?" becomes
a forensic audit). This writes that manifest for the AIC2026 read.

    python3 scripts/aic2026_manifest.py --dir /mnt/v/output/zensim/aic2026/2026-09-19
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Binaries whose bytes determine the numbers in this directory.
BINARIES = [
    "target/release/score_pairs_tuner",
    "zensim-bench/target/release/examples/peer_metric_pairs",
    "target/release/check_holdout_overlap",
    "target/release/examples/zensim_score",
]

# Inputs whose bytes the scores depend on, beyond the images themselves.
INPUTS = [
    "/mnt/v/datasets/aic2026/metrics_cropped.csv",
    "/mnt/v/datasets/aic2026/metrics_fullres.csv",
    "/mnt/v/datasets/aic2026/SHA256SUMS",
]

FROZEN_DIR = "/var/tmp/zensim-validation-2026-09-15/recovery/calibrated"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--build-commit", default=None)
    args = ap.parse_args()

    commit = args.build_commit or subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()

    outputs = {}
    for name in sorted(os.listdir(args.dir)):
        p = os.path.join(args.dir, name)
        if not os.path.isfile(p) or name == "_MANIFEST.json":
            continue
        outputs[name] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}

    binaries = {}
    for rel in BINARIES:
        p = os.path.join(REPO, rel)
        if os.path.exists(p):
            binaries[rel] = sha256(p)

    inputs = {p: sha256(p) for p in INPUTS if os.path.exists(p)}

    frozen = {}
    frozen_json = os.path.join(FROZEN_DIR, "FROZEN.json")
    if os.path.exists(frozen_json):
        fr = json.load(open(frozen_json, encoding="utf-8"))
        for name, m in fr.get("models", {}).items():
            frozen[name] = {
                "weights": m.get("weights"),
                "members": [
                    {"path": x["path"], "sha256_declared": x["sha256"],
                     "sha256_measured": sha256(x["path"])
                     if os.path.exists(x["path"]) else None}
                    for x in m["members"]
                ],
            }

    manifest = {
        "artifact": "JPEG AIC2026 first read (metric-agreement panel)",
        "date": datetime.date.today().isoformat(),
        "build_commit": commit,
        "human_labels_present": False,
        "caveat": (
            "AIC2026 ships no human scores. Everything in this directory is "
            "metric-vs-metric agreement or ladder behaviour, never accuracy "
            "against human judgment. CVVDP placed the distortion levels, so "
            "CVVDP is monotone on these ladders by construction."
        ),
        "split_role": (
            "jpeg-aic-family-holdout-2026-09-01 — T0-family, EVAL-ONLY, never "
            "a training input, membership by content (zensim docs/DATA_SPLITS.md)"
        ),
        "license": "CC BY-SA 4.0; per-source attribution in the sources CSV",
        "source": "DaRUS doi:10.18419/DARUS-6156 v2.0; arXiv:2607.22783",
        "models_scored": [
            "ZensimProfile::PreviewV0_2",
            "ZensimProfile::B (codec_target)",
            "ZensimProfile::C",
            "ZensimProfile::D",
            "BakeScorer::ensemble R915_y60_h32_ens5 (5 members, equal weights)",
            "BakeScorer::ensemble R915_basic228_h128_ens5 (5 members, equal weights)",
        ],
        "frozen_ensembles": frozen,
        "binaries_sha256": binaries,
        "inputs_sha256": inputs,
        "outputs": outputs,
        "decoder": (
            "zenpng via zensim-validate score_pairs_tuner; zen_decode "
            "(zencodec dispatch) via zensim-bench peer_metric_pairs. No "
            "third-party image decoder is on either scoring path."
        ),
    }
    with open(os.path.join(args.dir, "_MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, sort_keys=True)
    print(f"wrote {os.path.join(args.dir, '_MANIFEST.json')}")


if __name__ == "__main__":
    main()
