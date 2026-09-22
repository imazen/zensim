#!/usr/bin/env python3
"""The §5 evaluation table: every frozen dvifmish preset (float and integer
paths) and the frozen peers on the talk's five test sets, global SROCC/KROCC
first, per-codec and per-source means second.

Sets are scored once at full size with the dvifmish crate; subsets (the
JPEG + JPEG 2000 slices, CID22-A/B) are cut from those per-row scores by
(ref_path, dist_path), so a subset number can never come from a different
run than its parent. Peers come from `dvifmish_peers.sh` (the Rev3 extractor's
fast-ssim2 audit channel and `ensemble_score_rows` for zensim B, D and the
R915 basic228 ensemble).

Usage:
  table5.py score  <work> <preset.json>...        # dvifmish batch, float + int
  table5.py peers  <work>                         # peer CSVs -> per-set tables
  table5.py eval   <work>                         # eval_scores.py per set / path
"""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

EV = Path(__file__).resolve().parent
PAIRS = Path("/mnt/v/output/zensim/dvifmish-eval-2026-09-22/pairs")
# dvifmish commit 49aaf667; byte-identical to the screen binary on a 60-pair x 3-preset x float/int check
BIN = "/var/tmp/dvifmish/bin/dvifmish-49aaf667"
PEERS = Path("/var/tmp/dvifmish/peers/scores")
PEER_TAGS = {"fastssim2": "fast-ssim2", "zensim_b": "zensim B", "zensim_d": "zensim D",
             "rev3_basic228_ens5": "Rev3 basic228 ensemble"}

# name -> (parent scored set or None, label orientation)
SETS = {
    "tid2013_full": (None, "quality"),
    "tid2013_codec": ("tid2013_full", "quality"),
    "kadid10k_nt": (None, "quality"),
    "kadid10k_nt_codec": ("kadid10k_nt", "quality"),
    "nncd": (None, "quality"),
    "aic4_ptc": (None, "distortion"),
    "aic4_full": (None, "distortion"),
    "cid22_49": (None, "quality"),
    "cid22a": ("cid22_49", "quality"),
    "cid22b23": ("cid22_49", "quality"),
}
PATHS = ("float", "int")


def run(cmd, log):
    with open(log, "a") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        return subprocess.run(list(map(str, cmd)), stdout=f, stderr=subprocess.STDOUT).returncode


def keys(tsv):
    return [(r["ref_path"], r["dist_path"]) for r in csv.DictReader(open(tsv), delimiter="\t")]


def subset_rows(parent_tsv, child_tsv):
    idx = {k: i for i, k in enumerate(keys(parent_tsv))}
    return [idx[k] for k in keys(child_tsv)]


def cmd_score(work, presets):
    for name, (parent, _) in SETS.items():
        if parent is not None:
            continue
        for path in PATHS:
            out = work / "scores" / name / path
            out.mkdir(parents=True, exist_ok=True)
            cmd = [BIN, "batch", "--pairs", PAIRS / f"{name}.tsv", "--out-dir", out,
                   "--threads", os.environ.get("DVIFMISH_THREADS", "16")]
            for p in presets:
                cmd += ["--params", p]
            if path == "int":
                cmd.append("--int")
            rc = run(cmd, work / "logs" / f"score_{name}_{path}.log")
            print(f"score {name} {path}: rc={rc}", flush=True)


def cut(src_tsv, rows, dst):
    lines = open(src_tsv).read().splitlines()
    head, body = lines[0], lines[1:]
    with open(dst, "w") as f:
        f.write(head + "\n")
        for i in rows:
            f.write(body[i] + "\n")


def cmd_subsets(work):
    for name, (parent, _) in SETS.items():
        if parent is None:
            continue
        rows = subset_rows(PAIRS / f"{parent}.tsv", PAIRS / f"{name}.tsv")
        for path in PATHS + ("peers",):
            pdir = work / "scores" / parent / path
            if not pdir.exists():
                continue
            out = work / "scores" / name / path
            out.mkdir(parents=True, exist_ok=True)
            for t in pdir.glob("*.tsv"):
                cut(t, rows, out / t.name)


def cmd_peers(work):
    for name, (parent, orient) in SETS.items():
        if parent is not None:
            continue
        for tag in PEER_TAGS:
            src = PEERS / f"{tag}__on__{name}.csv"
            if not src.exists():
                continue
            rows = list(csv.DictReader(open(src)))
            want = keys(PAIRS / f"{name}.tsv")
            assert [(r["ref_path"], r["dist_path"]) for r in rows] == want, (src, "row order")
            out = work / "scores" / name / "peers"
            out.mkdir(parents=True, exist_ok=True)
            with open(out / f"{tag}.tsv", "w") as f:
                f.write("row\tscore\n")
                for i, r in enumerate(rows):
                    q = float(r["score"])           # peers are quality-oriented
                    f.write(f"{i}\t{q if orient == 'quality' else -q!r}\n")
    cmd_subsets(work)


def cmd_eval(work):
    for name, (_, orient) in SETS.items():
        for path in PATHS + ("peers",):
            d = work / "scores" / name / path
            models = sorted(p.stem for p in d.glob("*.tsv")) if d.exists() else []
            if not models:
                continue
            out = work / "eval" / f"{name}__{path}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            rc = run(["python3", EV / "eval_scores.py", "table", "--pairs", PAIRS / f"{name}.tsv",
                      "--scores", d, "--models", ",".join(models), "--orient", orient, "--out", out],
                     work / "logs" / f"eval_{name}_{path}.log")
            print(f"eval {name} {path}: {len(models)} models rc={rc}", flush=True)


def main():
    mode, work = sys.argv[1], Path(sys.argv[2])
    (work / "logs").mkdir(parents=True, exist_ok=True)
    if mode == "score":
        cmd_score(work, sys.argv[3:])
        cmd_subsets(work)
    elif mode == "peers":
        cmd_peers(work)
    elif mode == "eval":
        cmd_eval(work)
    else:
        sys.exit(f"unknown mode {mode}")


if __name__ == "__main__":
    main()
