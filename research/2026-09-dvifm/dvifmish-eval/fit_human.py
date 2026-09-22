#!/usr/bin/env python3
"""Human-label DVIFM fits for the dvifmish presets (work order §4 and the
talk-faithful three-plane configuration).

Two fitting sets, each written as a fit_dvifmish.py segment list:
  cid22     all 4,292 CID22-49 human pairs (the talk's own training data; exposure
            ledgered in docs/DATASET_HISTORY.md before any label read). Every preset
            fitted here has zero held-out CID22 claim.
  humanmix  the faithful lane's 3,092-pair mix: CID22-A (2,192), TID2013 JPEG+JPEG2000
            (250), KADID-10k JPEG+JPEG2000 on the 65 non-terminal references (650).
Labels are the lists' own quality-oriented 0..1 scales (label_scale 1).

Usage: fit_human.py fit <work> <set> <config> [<config> ...]
  config = <planes>-<vis>-<pyramid>  e.g. luma-curve-talk, ycbcr3-gate-ours
"""
import json
import subprocess
import sys
from pathlib import Path

EV = Path(__file__).resolve().parent
PAIRS = Path("/mnt/v/output/zensim/dvifmish-eval-2026-09-22/pairs")
CACHE = Path("/var/tmp/dvifmish/cache")
TOOLS = Path("/home/lilith/work/dvifmish/tools")
PLANES = {"luma": ["ycbcr_y"], "ycbcr3": ["ycbcr_y", "ycbcr_cb", "ycbcr_cr"],
          "xyby": ["xyb_y"], "xyb3": ["xyb_y", "xyb_x", "xyb_b"]}
PYR = {"talk": ("laplacian", "binomial121", "plain", "free", "ycc_lap121"),
       "ours": ("local", "binomial121", "edge", "tied", "ycc_local121"),
       "1331": ("local", "binomial1331", "edge", "tied", "ycc_local1331")}
SETS = {"cid22": ["cid22_49"], "humanmix": ["cid22a", "tid2013_codec", "kadid10k_nt_codec"]}
DESC = {"cid22": "all 4,292 CID22-49 human pairs (the talk's training data; fit-domain on every CID22 row)",
        "humanmix": "the faithful lane's 3,092 human pairs: CID22-A (2,192), TID2013 JPEG+JPEG2000 (250), "
                    "KADID-10k JPEG+JPEG2000 on 65 references (650)"}


def fit(work, setname, config):
    planes_k, vis, pyr = config.split("-")
    band, kernel, contrast, pooling, records = PYR[pyr]
    planes = PLANES[planes_k]
    if planes_k.startswith("xyb"):
        records = records.replace("ycc_", "xyb_")
    tag = f"{config}-{setname}"
    # the talk's structure learns A and B per level; our forms share one beta
    beta_mode = "level" if pyr == "talk" and vis == "curve" else "shared"
    segs = [{"name": ds, "pairs": str(PAIRS / f"{ds}.tsv"),
             "bins": {p: str(CACHE / f"{records}_{ds}_{p}.bin") for p in planes},
             "label_scale": 1.0} for ds in SETS[setname]]
    seg = work / "segs" / f"{tag}.json"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_text(json.dumps(segs, indent=1))
    art = work / "fits" / f"{tag}.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    log = work / "logs" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        if not art.exists():
            rc = subprocess.run(["python3", str(EV / "fit_dvifmish.py"), "fit", "--segs", str(seg),
                                 "--planes", ",".join(planes), "--vis", vis, "--contrast", contrast,
                                 "--pooling", pooling, "--constants", "fit", "--beta-mode", beta_mode,
                                 "--out", str(art), "--name", tag],
                                stdout=f, stderr=subprocess.STDOUT).returncode
            if rc != 0:
                return f"FAIL fit {tag}"
        prov = (f"dvifmish human-label fit 2026-09-22 ({config}): {vis} visibility, {band} band, "
                f"{'edge-discounted' if contrast == 'edge' else 'plain'} contrast, {pooling} pooling, "
                f"{kernel}, {'one beta per level' if beta_mode == 'level' else 'one shared beta'}; every constant "
                f"fitted by fit_dvifmish.py (refit-map MSE) on {DESC[setname]}.")
        out = work / "presets" / f"{tag}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        rc = subprocess.run(["python3", str(TOOLS / "artefact_to_params.py"), "--format", "dvifmish",
                             "--artefact", str(art), "--band", band, "--kernel", kernel, "--name", tag,
                             "--provenance", prov, "--out", str(out)],
                            stdout=f, stderr=subprocess.STDOUT).returncode
    return f"ok {tag} -> {out}" if rc == 0 else f"FAIL params {tag}"


def main():
    if sys.argv[1] != "fit":
        sys.exit(__doc__)
    work, setname = Path(sys.argv[2]), sys.argv[3]
    for c in sys.argv[4:]:
        print(fit(work, setname, c), flush=True)


if __name__ == "__main__":
    main()
