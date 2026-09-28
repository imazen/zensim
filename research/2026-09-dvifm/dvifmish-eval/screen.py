#!/usr/bin/env python3
"""dvifmish variant screen — one factor at a time from a baseline.

User selection (2026-09-22): planes {Y'CbCr-3, Y', XYB-3, XYB-Y} ×
visibility {curve, gate, off} × pyramid {talk: Laplacian band + plain block
range + free Lp pooling; ours: local band + 3×3 corner edge discount + tied
pooling; [1 3 3 1] kernel with ours otherwise} × constants {shared-β prior,
SafeSyn-fitted, full CID22 human}. Baseline: Y'CbCr-3 / gate / ours / prior.
The cross-product is not run: each arm changes one factor.

Every arm is fitted on three seeded 4,000-row SafeSyn TRAIN subsets (teacher
target: signed SSIMULACRA2/100, never clipped) with `fit_dvifmish.py`, then
scored with the dvifmish crate on the selection legs:
    cid22a      CID22-A human (fit-allowed for DVIFM constants since 2026-09-19)
    konfig_val  KonFiG-IQA validation human
    codec_dev   codec renditions, SSIMULACRA2 teacher
    safesyn_dev SafeSyn development, SSIMULACRA2 teacher
Decision composite = mean of the two HUMAN legs' global SROCC; the teacher
legs are reported beside it. The CID22-fitted constants arm is handled by the
separate §4 step (CID22-A would be in its fit domain).

Usage: screen.py fit|score|report <workdir>
       screen.py promote <workdir> <arm>   (the arm's seed-1 fit -> named preset)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

EV = Path(__file__).resolve().parent
PAIRS = Path("/mnt/v/output/zensim/dvifmish-eval-2026-09-22/pairs")
CACHE = Path("/var/tmp/dvifmish/cache")
BIN = "/var/tmp/dvifmish/bin/dvifmish"
TOOLS = Path("/home/lilith/work/dvifmish/tools")
SEEDS = (1, 2, 3)
LEGS = ("cid22a", "konfig_val", "codec_dev", "safesyn_dev")
HUMAN = ("cid22a", "konfig_val")

YCC = ["ycbcr_y", "ycbcr_cb", "ycbcr_cr"]
XYB = ["xyb_y", "xyb_x", "xyb_b"]
OURS = {"records": "ycc_local121", "band": "local", "kernel": "binomial121",
        "contrast": "edge", "pooling": "tied"}
TALK = {"records": "ycc_lap121", "band": "laplacian", "kernel": "binomial121",
        "contrast": "plain", "pooling": "free"}
K1331 = {"records": "ycc_local1331", "band": "local", "kernel": "binomial1331",
         "contrast": "edge", "pooling": "tied"}

BASE = dict(OURS, planes=YCC, vis="gate", constants="prior", beta=0.693)


def arm(**kw):
    d = dict(BASE)
    d.update(kw)
    return d


ARMS = {
    "base": arm(),
    "planes-ycbcr_y": arm(planes=["ycbcr_y"]),
    "planes-xyb3": arm(planes=XYB, records="xyb_local121"),
    "planes-xyb_y": arm(planes=["xyb_y"], records="xyb_local121"),
    "vis-curve": arm(vis="curve"),
    "vis-off": arm(vis="off"),
    "pyr-talk": arm(**TALK),
    "pyr-1331": arm(**K1331),
    "const-safesyn": arm(constants="fit"),
    # Added 2026-09-23 after the fit-domain (SafeSyn training) numbers showed that the
    # prior protocol leaves a gate's only constant, its knee, at the 10th percentile of
    # block contrast; before any selection leg was scored. With these, every visibility
    # form also has a SafeSyn-fitted arm (the gate's is const-safesyn).
    "vis-curve-fit": arm(vis="curve", constants="fit"),
    "vis-off-fit": arm(vis="off", constants="fit"),
}


# Preset name of each arm (its seed-1 fit, see promote_one): <planes>-<visibility>-<pyramid>-<constants>.
PRESET = {
    "base": "ycbcr3-gate-ours-prior",
    "planes-ycbcr_y": "luma-gate-ours-prior",
    "planes-xyb3": "xyb3-gate-ours-prior",
    "planes-xyb_y": "xyby-gate-ours-prior",
    "vis-curve": "ycbcr3-curve-ours-prior",
    "vis-off": "ycbcr3-off-ours-prior",
    "pyr-talk": "ycbcr3-gate-talk-prior",
    "pyr-1331": "ycbcr3-gate-1331-prior",
    "const-safesyn": "ycbcr3-gate-ours-safesyn",
    "vis-curve-fit": "ycbcr3-curve-ours-safesyn",
    "vis-off-fit": "ycbcr3-off-ours-safesyn",
}


def preset_name(a):
    """<planes>-<visibility>-<pyramid>-<constants source> for any arm."""
    planes = {("ycbcr_y", "ycbcr_cb", "ycbcr_cr"): "ycbcr3", ("ycbcr_y",): "luma",
              ("xyb_y", "xyb_x", "xyb_b"): "xyb3", ("xyb_y",): "xyby"}[tuple(a["planes"])]
    pyr = {"ycc_local121": "ours", "xyb_local121": "ours", "ycc_lap121": "talk",
           "ycc_local1331": "1331"}[a["records"]]
    return f"{planes}-{a['vis']}-{pyr}-{'prior' if a['constants'] == 'prior' else 'safesyn'}"


def r2_arms(base):
    """Round 2 (SCREEN_ROUND2_PREREG.md): planes and pyramid values from `base`."""
    b = ARMS[base]
    out = {}
    for name, kw in (("planes-ycbcr_y", dict(planes=["ycbcr_y"])),
                     ("planes-xyb3", dict(planes=XYB, records="xyb_local121")),
                     ("planes-xyb_y", dict(planes=["xyb_y"], records="xyb_local121")),
                     ("pyr-talk", TALK), ("pyr-1331", K1331)):
        d = dict(b)
        d.update(kw)
        out[f"r2-{name}"] = d
    return out


if os.environ.get("DVIFMISH_R2_BASE"):
    ARMS.update(r2_arms(os.environ["DVIFMISH_R2_BASE"]))
    for _n, _a in ARMS.items():
        PRESET.setdefault(_n, preset_name(_a))

def run(cmd, log):
    with open(log, "a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n")
        f.flush()
        return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT).returncode


def fit_one(work, name, seed):
    a = ARMS[name]
    tag = f"{name}-s{seed}"
    seg = work / "segs" / f"{tag}.json"
    seg.parent.mkdir(parents=True, exist_ok=True)
    seg.write_text(json.dumps([{
        "name": f"safesyn_fit_s{seed}", "pairs": str(PAIRS / f"safesyn_fit_s{seed}.tsv"),
        "bins": {p: str(CACHE / f"{a['records']}_safesyn_fit_s{seed}_{p}.bin") for p in a["planes"]},
        "label_scale": 0.01}], indent=1))
    art = work / "fits" / f"{tag}.json"
    art.parent.mkdir(parents=True, exist_ok=True)
    log = work / "logs" / f"{tag}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    if not art.exists():
        rc = run(["python3", EV / "fit_dvifmish.py", "fit", "--segs", seg, "--planes", ",".join(a["planes"]),
                  "--vis", a["vis"], "--contrast", a["contrast"], "--pooling", a["pooling"],
                  "--constants", a["constants"], "--beta", a["beta"], "--out", art, "--name", tag], log)
        if rc != 0:
            return f"FAIL fit {tag}"
    prov = (f"dvifmish variant screen 2026-09-22, arm '{name}' seed {seed}: fitted by fit_dvifmish.py "
            f"({a['constants']} constants) on 4,000 seeded SafeSyn TRAIN pairs whose target is signed "
            f"SSIMULACRA2/100 (a metric teacher's opinion, not human labels).")
    params = work / "params" / f"screen-{tag}.json"
    params.parent.mkdir(parents=True, exist_ok=True)
    rc = run(["python3", TOOLS / "artefact_to_params.py", "--format", "dvifmish", "--artefact", art,
              "--band", a["band"], "--kernel", a["kernel"], "--name", f"screen-{tag}",
              "--provenance", prov, "--out", params], log)
    return f"ok {tag}" if rc == 0 else f"FAIL params {tag}"


def promote_one(work, name, seed=1):
    """Name one screen arm's seed-`seed` fit as a preset: the constants the
    selection legs actually scored, frozen before any test set is read."""
    a = ARMS[name]
    tag = PRESET[name]
    src = work / "params" / f"screen-{name}-s{seed}.json"
    p = json.loads(src.read_text())
    if a["constants"] == "prior":
        what = {"curve": f"shared-beta prior (beta = {a['beta']}, the middle of the identified band "
                         "[0.607, 0.779]; knees at the fit rows' 10th percentile of block contrast)",
                "gate": "prior constants (gate knees at the fit rows' 10th percentile of block "
                        "contrast; g = p = 1)",
                "off": "prior constants (no masking; g = p = 1)"}[a["vis"]]
        what += "; only the level/plane weights and the output map fitted"
    else:
        what = {"curve": "every constant fitted (knees, g, p, sharpness, one shared beta, weights, "
                         "output map)",
                "gate": "fitted gate knees and error exponents p, weights and output map (g = 1)",
                "off": "fitted error exponents p, weights and output map (no masking)"}[a["vis"]]
        if a["pooling"] == "free":
            what += "; the free pooling exponent L per level fitted too"
    p["name"] = tag
    p["provenance"] = (f"dvifmish variant screen 2026-09-22/23, arm '{name}', seed {seed}: {what}; "
                       f"fitted by fit_dvifmish.py on 4,000 seeded SafeSyn TRAIN pairs whose target is "
                       f"signed SSIMULACRA2/100 (a metric teacher's opinion, not human labels). "
                       f"Seeds 2 and 3 were fitted the same way on other subsets and scored for the spread.")
    out = work / "presets" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(p, indent=1) + "\n")
    return f"ok promote {name} -> {out}"

def main():
    mode, work = sys.argv[1], Path(sys.argv[2])
    work.mkdir(parents=True, exist_ok=True)
    if mode == "fit":
        # one arm-seed per invocation, so the caller can parallelise
        name, seed = sys.argv[3], int(sys.argv[4])
        print(fit_one(work, name, seed), flush=True)
    elif mode == "promote":
        print(promote_one(work, sys.argv[3]), flush=True)
    elif mode == "list":
        for name in ARMS:
            for s in SEEDS:
                print(name, s)
    elif mode == "score":
        params = sorted((work / "params").glob("screen-*.json"))
        for leg in LEGS:
            out = work / "scores" / leg
            out.mkdir(parents=True, exist_ok=True)
            cmd = [BIN, "batch", "--pairs", PAIRS / f"{leg}.tsv", "--out-dir", out, "--threads", "16"]
            for p in params:
                cmd += ["--params", p]
            rc = run(cmd, work / "logs" / f"score_{leg}.log")
            print(f"score {leg}: rc={rc}", flush=True)
    elif mode == "report":
        models = [p.stem for p in sorted((work / "params").glob("screen-*.json"))]
        for leg in LEGS:
            (work / "eval").mkdir(parents=True, exist_ok=True)
            rc = run(["python3", EV / "eval_scores.py", "table", "--pairs", PAIRS / f"{leg}.tsv",
                      "--scores", work / "scores" / leg, "--models", ",".join(models),
                      "--out", work / "eval" / f"{leg}.json"], work / "logs" / f"eval_{leg}.log")
            print(f"eval {leg}: rc={rc}", flush=True)


if __name__ == "__main__":
    main()
