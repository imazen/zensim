#!/usr/bin/env python3
"""Rev4 E1 — JPEG-AIC forced choice by question type x fidelity band.

Registered in benchmarks/rev4_e1_prereg_2026-09-23.md (§4 forced-choice bands,
§5.3). Reuses the response semantics and score tables of
benchmarks/hfhuman_2026-09-01/build_triplets.py (the pivot is always the
original; `response` names the side judged MORE different; trap and bias rows
excluded) and adds one split: the fidelity band of a triplet,
m = max(dlevel_left, dlevel_right): HF m<=3, MF 4-6, LF 7-10.

No statistic is computed here: every number comes from `panel --pairwise`
(zensim_validate::pairwise). This file owns the rows, the groups and the RNG.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess

HF = "/mnt/v/output/zensim/hfhuman-2026-09-01"
STUDIES = {
    "aic3_btc": "/mnt/v/datasets/aic3-btc-ptc/JPEG-AIC_BTC_final_response_data_2024.01.10.csv",
    "sdr25_btc": "/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data/JPEG_AIC_SDR_BTC_JPEG_AI_responses_2025.02.28_v1.csv",
    "sdr25_ptc": "/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data/JPEG_AIC_SDR_PTC_JPEG_AI_responses_2025.02.28_v1.csv",
    "aic3_iptc": "/mnt/v/datasets/aic3-btc-ptc/JPEG-AIC_IPTC_final_response_data_2024_06_28 (1).csv",
}
ARMS = {
    "btc_native": (["aic3_btc", "sdr25_btc"], [f"{HF}/btc_native_scores.tsv"]),
    "btc_displayed": (["aic3_btc", "sdr25_btc"], [f"{HF}/btc_displayed_scores.tsv"]),
    "native_ptc_iptc": (["sdr25_ptc", "aic3_iptc"], [f"{HF}/iptc/ptc_native_scores.tsv", f"{HF}/iptc/iptc_native_scores.tsv"]),
}
SCORERS = {"peer_ssim2": "ssim2", "W10L9PH_s4004_packed": "C", "B": "B",
           "W10L9P_s4005_packed": "ctx_W10L9P_s4005", "ADD156": "ctx_ADD156",
           "Q7b_pools_g0.2_a0.2_b0.97": "ctx_Q7b"}
UNDECIDED = {"notsure", "notSure", "skip", ""}
SEED = 20260923


def pct(v, q):
    v = sorted(x for x in v if not math.isnan(x))
    return v[min(len(v) - 1, max(0, int(q * len(v))))] if v else float("nan")


def fband(m):
    return "HF" if m <= 3 else ("MF" if m <= 6 else "LF")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel-bin", required=True)
    ap.add_argument("--scratch", default="/var/tmp/rev4-e1/fc_scratch")
    ap.add_argument("--out", default="/var/tmp/rev4-e1/fc_full.json")
    ap.add_argument("-B", type=int, default=2000)
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    resp = {k: list(csv.DictReader(open(p))) for k, p in STUDIES.items()}
    result = {"seed": SEED, "B": a.B, "panel_bin": a.panel_bin, "arms": {}}
    for arm, (studies, score_files) in ARMS.items():
        srows = []
        for sf in score_files:
            srows += list(csv.DictReader(open(sf), delimiter="\t"))
        score = {r["stimulus"]: r for r in srows}
        assert len(score) == len(srows)
        counts = {}
        for st in studies:
            for r in resp[st]:
                if r["response"] in UNDECIDED or r["is_bias"] == "1" or r["is_trap"] == "1":
                    continue
                cl, cr = r["codec_left"], r["codec_right"]
                qt = "vs_original" if (cl == "0" or cr == "0") else ("same_codec" if cl == cr else "cross_codec")
                m = max(int(r["dlevel_left"]), int(r["dlevel_right"]))
                g = f"{st}:{r['question_id']}"
                e = counts.setdefault(g, {"left": 0.0, "right": 0.0, "L": r["img_left"], "R": r["img_right"],
                                          "img": int(r["img_num"]), "qt": qt, "fb": fband(m), "m": m})
                assert e["L"] == r["img_left"] and e["R"] == r["img_right"] and e["m"] == m
                e[r["response"]] += 1.0
        res_arm = {}
        for fb in ("ALLF", "HF", "MF", "LF"):
            for qt in ("all", "vs_original", "same_codec", "cross_codec"):
                keys = sorted(g for g, e in counts.items() if (fb == "ALLF" or e["fb"] == fb) and (qt == "all" or e["qt"] == qt))
                if not keys:
                    continue
                n = len(keys)
                gimg = [counts[k]["img"] for k in keys]
                imgs = sorted(set(gimg))
                by_img = {i: [j for j, v in enumerate(gimg) if v == i] for i in imgs}
                tag = f"{arm}__{fb}__{qt}"
                man_i = f"{a.scratch}/{tag}__img.tsv"
                man_q = f"{a.scratch}/{tag}__q.tsv"
                rng_i = random.Random(SEED)
                rng_q = random.Random(SEED + 1)
                with open(man_i, "w") as f:
                    f.write("POINT\t*\n")
                    for b in range(a.B):
                        pick = []
                        for _ in range(len(imgs)):
                            pick.extend(by_img[imgs[rng_i.randrange(len(imgs))]])
                        f.write(f"B{b}\t" + ",".join(map(str, pick)) + "\n")
                with open(man_q, "w") as f:
                    f.write("POINT\t*\n")
                    for b in range(a.B):
                        f.write(f"B{b}\t" + ",".join(str(rng_q.randrange(n)) for _ in range(n)) + "\n")
                per = {}
                for col, name in SCORERS.items():
                    rf = f"{a.scratch}/{tag}__{name}.tsv"
                    with open(rf, "w") as f:
                        f.write("group\ts_left\ts_right\tchoice\tweight\n")
                        for k in keys:
                            e = counts[k]
                            sl, sr = score[e["L"]][col], score[e["R"]][col]
                            for ch in ("left", "right"):
                                if e[ch] > 0:
                                    f.write(f"{k}\t{sl}\t{sr}\t{ch}\t{e[ch]:g}\n")
                    out = {}
                    for ctag, man in (("img", man_i), ("q", man_q)):
                        p = subprocess.run([a.panel_bin, "--pairwise", rf, "--resample", man], capture_output=True, text=True)
                        if p.returncode != 0:
                            raise SystemExit(p.stderr[-2000:])
                        lines = p.stdout.strip().split("\n")
                        hdr = lines[0].split("\t")
                        recs = [dict(zip(hdr, l.split("\t"))) for l in lines[1:]]
                        out[ctag] = recs
                    pt = out["img"][0]
                    assert pt["label"] == "POINT"
                    per[name] = {"acc": float(pt["acc_response"]), "ceiling": float(pt["ceiling_response"]),
                                 "tie_rate": float(pt["tie_rate"]), "n_groups": int(pt["n_groups"]),
                                 "n_responses": float(pt["n_responses"]),
                                 "boot_img": [float(r["acc_response"]) for r in out["img"][1:]],
                                 "boot_q": [float(r["acc_response"]) for r in out["q"][1:]]}
                cell = {"n_triplets": n, "n_images": len(imgs), "n_responses": per["ssim2"]["n_responses"],
                        "ceiling": per["ssim2"]["ceiling"], "scorers": {}, "deltas_vs_ssim2": {}}
                for name, v in per.items():
                    cell["scorers"][name] = {"acc": v["acc"], "tie_rate": v["tie_rate"],
                                             "ci_img": [pct(v["boot_img"], .025), pct(v["boot_img"], .975)]}
                    if name == "ssim2":
                        continue
                    d_i = [x - y for x, y in zip(v["boot_img"], per["ssim2"]["boot_img"])]
                    d_q = [x - y for x, y in zip(v["boot_q"], per["ssim2"]["boot_q"])]
                    cell["deltas_vs_ssim2"][name] = {
                        "point": v["acc"] - per["ssim2"]["acc"],
                        "ci_img": [pct(d_i, .025), pct(d_i, .975)], "p_gt0_img": sum(x > 0 for x in d_i) / len(d_i),
                        "ci_q": [pct(d_q, .025), pct(d_q, .975)], "p_gt0_q": sum(x > 0 for x in d_q) / len(d_q)}
                res_arm[f"{fb}/{qt}"] = cell
                print(f"{arm} {fb}/{qt}: triplets={n} images={len(imgs)} responses={cell['n_responses']:.0f}", flush=True)
        result["arms"][arm] = res_arm
    json.dump(result, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
