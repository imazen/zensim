#!/usr/bin/env python3
"""Rev4 E1b — same-codec vs cross-codec within-reference ordering (confirmatory).

Registered in benchmarks/rev4_e1b_prereg_2026-09-23.md. Reuses E1's tables,
draws, bootstrap-percentile read and summarize() (benchmarks/rev4_e1_2026-09-23/
analyze.py) and the E1 forced-choice row semantics (fc.py). No statistic is
computed here: every accuracy comes from `panel --pairwise`
(zensim_validate::pairwise). This file owns the pair lists and their classes,
the paired draws, the decision bookkeeping, and plain tallies for the error
tables.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "rev4_e1_2026-09-23"))
import analyze as A  # noqa: E402

SEED = A.SEED
TABLES = "/var/tmp/rev4-e1b/tables"
RM = "/mnt/v/output/zensim/reports/refmetrics"
FE = "/mnt/v/output/zensim/reports/fulleval"
HF = "/mnt/v/output/zensim/hfhuman-2026-09-01"
FC_STUDIES = {
    "aic3_btc": "/mnt/v/datasets/aic3-btc-ptc/JPEG-AIC_BTC_final_response_data_2024.01.10.csv",
    "sdr25_btc": "/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data/JPEG_AIC_SDR_BTC_JPEG_AI_responses_2025.02.28_v1.csv",
}
FC_CODE = {"1": "AVIF", "2": "JPEG-1", "3": "JPEG-2000", "4": "JPEG-XL", "5": "VVC", "6": "JPEG-AI"}
FC_SCORERS = {"peer_ssim2": "ssim2", "W10L9PH_s4004_packed": "C", "B": "B"}
UNDECIDED = {"notsure", "notSure", "skip", ""}
PEERS = A.PEERS + ["cvvdp_gpu_unrec"]
ZENSIM = A.ZENSIM
AVIF = {"aom/s1", "aom/s7", "cld_avif", "vis_avif"}
FORMAT = {"cld_heic": "HEIC", "cld_jp2": "JP2", "cld_webp": "WebP", "libjxl/e7": "JXL", "mozjpeg": "JPEG"}


# ---------------------------------------------------------------- codec identity

def cid22_config(stim):
    _, _, enc, f = stim.split("/")
    return f"{enc}/{f.split('_')[0]}" if "_" in f else enc


def cid22_format(stim):
    c = cid22_config(stim)
    return "AVIF" if c in AVIF else FORMAT[c]


UNITS = {
    # unit: (table, codec(stim) for "same", format(stim) for "cross", row filter, gap kind)
    "cid22a": ("cid22a", cid22_config, cid22_format, None, "median"),
    "cid22a_encdir": ("cid22a", lambda s: s.split("/")[2], lambda s: s.split("/")[2], None, "median"),
    "csiq": ("csiq", lambda s: s.split(".")[1], lambda s: s.split(".")[1],
             lambda r: r["stim"].split(".")[1] in ("JPEG", "jpeg2000"), "median"),
    "aic3": ("aic3", lambda s: s.split("_")[0], lambda s: s.split("_")[0], None, "jnd"),
    "aic4crop": ("aic4crop", lambda s: s.split("_")[1], lambda s: s.split("_")[1], None, "jnd"),
    "aic4full": ("aic4full", lambda s: s.split("_")[1], lambda s: s.split("_")[1], None, "jnd"),
    "tid": ("tid", lambda s: TID_CODEC[s.split("_")[1]], lambda s: TID_CODEC[s.split("_")[1]],
            lambda r: r["stim"].split("_")[1] in ("10", "11"), "median"),
}
TID_CODEC = {"10": "JPEG", "11": "JPEG2000"}


def build_tid():
    """TID2013 rows (TRAIN-role, descriptive). Joins stored peer TSVs + C's fulleval by row order."""
    keys = list(csv.DictReader(open(f"{RM}/tid_iwssim.tsv"), delimiter="\t"))
    rows = [{"stim": os.path.basename(r["dist_path"]), "ref": os.path.basename(r["ref_path"]),
             "t": r["human_score"], "iwssim": r["iwssim_imazen_v0_0_1"]} for r in keys]
    for fn, cols in (("tid_ssim2_gpu.tsv", {"ssim2_gpu": "ssim2"}),
                     ("tid_butteraugli_gpu.tsv", {"butteraugli_pnorm3_gpu": "butter_p3", "butteraugli_max_gpu": "butter_max"}),
                     ("tid_cvvdp_gpu.tsv", {"cvvdp_imazen_v0_0_1": "cvvdp_gpu_unrec"})):
        src = list(csv.DictReader(open(f"{RM}/{fn}"), delimiter="\t"))
        assert [s["dist_path"] for s in src] == [k["dist_path"] for k in keys], fn
        for r, s in zip(rows, src):
            for c, o in cols.items():
                v = float(s[c])
                r[o] = repr(-v if o.startswith("butter") else v)
    pp = json.load(open(f"{FE}/W10L9PH_s4004_packed.fulleval.json"))["per_pair"]["tid"]
    assert len(pp["pred"]) == len(rows)
    for r, p, m in zip(rows, pp["pred"], pp["mos"]):
        assert abs(float(r["t"]) - m) < 1e-9
        r["C"] = repr(float(p))
    return rows, ["ssim2", "butter_p3", "butter_max", "iwssim", "cvvdp_gpu_unrec", "C"]


def load_unit(u):
    table, same_f, cross_f, filt, gapk = UNITS[u]
    if table == "tid":
        rows, cols = build_tid()
    else:
        rows, cols = A.load(TABLES, table)
    if filt:
        rows = [r for r in rows if filt(r)]
    return rows, cols, same_f, cross_f, gapk


# ---------------------------------------------------------------- owner calls

def run_panel(pairs, rows, metrics, rd, scratch, tag, panel_bin):
    """pairs: [(ref, i, j, choice)]. Returns {m: {point, boot}} from panel --pairwise."""
    groups = sorted({p[0] for p in pairs})
    gidx = {g: k for k, g in enumerate(groups)}
    man = f"{scratch}/{tag}__resample.tsv"
    with open(man, "w") as f:
        f.write("POINT\t*\n")
        for b, dr in enumerate(rd):
            s = [str(gidx[r]) for r in dr if r in gidx]
            if s:
                f.write(f"B{b}\t{','.join(s)}\n")
    out = {}
    for m in metrics:
        rf = f"{scratch}/{tag}__{m}.tsv"
        with open(rf, "w") as f:
            f.write("group\ts_left\ts_right\tchoice\tweight\n")
            for (r, i, j, ch) in pairs:
                f.write(f"{r}\t{rows[i][m]}\t{rows[j][m]}\t{ch}\t1\n")
        out[m] = parse_panel(panel_bin, rf, man, len(rd))
    return out


def parse_panel(panel_bin, rf, man, B):
    p = subprocess.run([panel_bin, "--pairwise", rf, "--resample", man], capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"panel --pairwise failed ({rf}):\n{p.stderr[-2000:]}")
    lines = p.stdout.strip().split("\n")
    hdr = lines[0].split("\t")
    boot = [float("nan")] * B
    point = tie = float("nan")
    for l in lines[1:]:
        rec = dict(zip(hdr, l.split("\t")))
        if rec["label"] == "POINT":
            point, tie = float(rec["acc_response"]), float(rec["tie_rate"])
        else:
            boot[int(rec["label"][1:])] = float(rec["acc_response"])
    return {"point": point, "boot": boot, "tie_rate": tie}


def dvec(sa, sb):
    return [x - y for x, y in zip(sa["boot"], sb["boot"])]


def ci_of(v):
    v = [x for x in v if not math.isnan(x)]
    return [round(A.pct(v, 0.025), 4), round(A.pct(v, 0.975), 4)] if v else [float("nan")] * 2


def best_of(stat, peers):
    ps = [p for p in peers if p in stat and not math.isnan(stat[p]["point"])]
    return max(ps, key=lambda p: stat[p]["point"]) if ps else None


# ---------------------------------------------------------------- tallies (not statistics)

def tally(pairs_meta, score, metrics, peer):
    """pairs_meta: [(cpair, sl, sr, choice)] with choice 'left'/'right' = side judged worse.
    Counts per codec pair: metric wrong & peer right, metric right & peer wrong, metric tie."""
    def verdict(m, key):
        a, b = score(m, key, "L"), score(m, key, "R")
        if a == b:
            return 0
        worse = "left" if a < b else "right"
        return 1 if worse == key[3] else -1
    out = defaultdict(lambda: defaultdict(lambda: [0, 0, 0, 0]))  # [n, m_wrong_peer_right, m_right_peer_wrong, m_tie]
    for key in pairs_meta:
        cp = key[0]
        vp = verdict(peer, key)
        for m in metrics:
            vm = verdict(m, key)
            c = out[cp][m]
            c[0] += 1
            if vm == -1 and vp == 1:
                c[1] += 1
            elif vm == 1 and vp == -1:
                c[2] += 1
            if vm == 0:
                c[3] += 1
    return {cp: dict(v) for cp, v in out.items()}


# ---------------------------------------------------------------- table units

def table_unit(u, B, scratch, panel_bin):
    rows, cols, same_f, cross_f, gapk = load_unit(u)
    peers = [c for c in cols if c in PEERS]
    zmods = [c for c in cols if c in ZENSIM]
    metrics = peers + zmods
    t = [float(r["t"]) for r in rows]
    cfg = [same_f(r["stim"]) for r in rows]
    fmt = [cross_f(r["stim"]) for r in rows]
    by_ref = defaultdict(list)
    for i, r in enumerate(rows):
        by_ref[r["ref"]].append(i)
    for r, ii in by_ref.items():
        assert len({fmt[i] for i in ii}) >= 2, f"{u}: ref {r} has <2 codecs"
    cls = {"same": [], "mid": [], "cross": []}
    meta = {"same": [], "mid": [], "cross": []}
    n_tie = Counter()
    for r, ii in by_ref.items():
        for a in range(len(ii)):
            for b in range(a + 1, len(ii)):
                i, j = ii[a], ii[b]
                k = "same" if cfg[i] == cfg[j] else ("mid" if fmt[i] == fmt[j] else "cross")
                if t[i] == t[j]:
                    n_tie[k] += 1
                    continue
                ch = "left" if t[i] < t[j] else "right"
                cls[k].append((r, i, j, ch))
                cp = " vs ".join(sorted((fmt[i], fmt[j]))) if k != "same" else f"{fmt[i]} (same)"
                meta[k].append((cp, i, j, ch, abs(t[i] - t[j])))
    refs = sorted(by_ref)
    rd = A.draws(refs, B, SEED)
    res = {"n_stim": len(rows), "n_refs": len(refs), "peers": peers, "zensim": zmods,
           "label_ties_dropped": dict(n_tie), "classes": {}}
    stats = {}
    for k in ("same", "mid", "cross"):
        if not cls[k]:
            continue
        st = run_panel(cls[k], rows, metrics, rd, scratch, f"{u}__{k}", panel_bin)
        stats[k] = st
        bp = best_of(st, peers)
        allright = sum(all((float(rows[i][m]) < float(rows[j][m])) == (ch == "left") and rows[i][m] != rows[j][m]
                           for m in metrics) for (_, i, j, ch) in cls[k])
        res["classes"][k] = {
            "n_pairs": len(cls[k]), "n_refs": len({p[0] for p in cls[k]}), "best_peer": bp,
            "frac_all_metrics_right": round(allright / len(cls[k]), 4),
            "acc": {m: {"point": round(st[m]["point"], 4), "ci": ci_of(st[m]["boot"]),
                        "tie_rate": round(st[m]["tie_rate"], 4)} for m in metrics},
            "d_best": {z: [round(st[z]["point"] - st[bp]["point"], 4)] + ci_of(dvec(st[z], st[bp])) for z in zmods},
            "d_ssim2": {z: [round(st[z]["point"] - st["ssim2"]["point"], 4)] + ci_of(dvec(st[z], st["ssim2"])) for z in zmods},
        }
    dec = {}
    if "same" in stats and "cross" in stats:
        bx, bs = res["classes"]["cross"]["best_peer"], res["classes"]["same"]["best_peer"]
        for z in zmods:
            dx = dvec(stats["cross"][z], stats["cross"][bx])
            ds = dvec(stats["same"][z], stats["same"][bs])
            ds_fix = dvec(stats["same"][z], stats["same"][bx])
            dec[z] = {"dx": [round(stats["cross"][z]["point"] - stats["cross"][bx]["point"], 4)] + ci_of(dx),
                      "ds": [round(stats["same"][z]["point"] - stats["same"][bs]["point"], 4)] + ci_of(ds),
                      "did": [round((stats["cross"][z]["point"] - stats["cross"][bx]["point"])
                                    - (stats["same"][z]["point"] - stats["same"][bs]["point"]), 4)]
                      + ci_of([a - b for a, b in zip(dx, ds)]),
                      "ds_fixedpeer": [round(stats["same"][z]["point"] - stats["same"][bx]["point"], 4)] + ci_of(ds_fix),
                      "did_fixedpeer": ci_of([a - b for a, b in zip(dx, ds_fix)])}
    res["decision_inputs"] = dec
    # descriptive strata on cross pairs, best peer fixed at the unit's cross best peer
    strata = {}
    if "cross" in stats:
        bx = res["classes"]["cross"]["best_peer"]
        sm = list(dict.fromkeys(zmods + [bx, "ssim2"]))
        gaps = sorted(m[4] for m in meta["cross"])
        med = gaps[(len(gaps) - 1) // 2] if gapk == "median" else 1.0
        sel = {"gap_small": lambda m: (m[4] <= med) if gapk == "median" else (m[4] < 1.0),
               "gap_large": lambda m: (m[4] > med) if gapk == "median" else (m[4] >= 1.0)}
        for cp in sorted({m[0] for m in meta["cross"]}):
            sel[f"pair:{cp}"] = (lambda c: (lambda m: m[0] == c))(cp)
        for name, f in sel.items():
            idx = [n for n, m in enumerate(meta["cross"]) if f(m)]
            if not idx:
                continue
            pl = [cls["cross"][n] for n in idx]
            st = run_panel(pl, rows, sm, rd, scratch, f"{u}__cross__{name.replace(' ', '').replace(':', '_')}", panel_bin)
            strata[name] = {"n_pairs": len(pl), "n_refs": len({p[0] for p in pl}),
                            "acc": {m: round(st[m]["point"], 4) for m in sm},
                            "d_best": {z: [round(st[z]["point"] - st[bx]["point"], 4)] + ci_of(dvec(st[z], st[bx])) for z in zmods}}
        strata["_gap_threshold"] = med if gapk == "median" else "1 JND"
        # error tables vs the unit's cross best peer, all pair classes
        def score(m, key, side):
            return float(rows[key[1] if side == "L" else key[2]][m])
        errs = {}
        for k in ("same", "mid", "cross"):
            if meta[k]:
                errs[k] = tally(meta[k], score, [m for m in metrics if m != bx], bx)
        res["error_tables"] = {"peer": bx, "cols": "[n, metric_wrong_peer_right, metric_right_peer_wrong, metric_tie]",
                               "by_class": errs}
    res["strata_cross"] = strata
    return res


# ---------------------------------------------------------------- forced choice

def fc_unit(B, scratch, panel_bin):
    srows = list(csv.DictReader(open(f"{HF}/btc_native_scores.tsv"), delimiter="\t"))
    score = {r["stimulus"]: r for r in srows}
    assert len(score) == len(srows)
    counts = {}
    for st, p in FC_STUDIES.items():
        for r in csv.DictReader(open(p)):
            if r["response"] in UNDECIDED or r["is_bias"] == "1" or r["is_trap"] == "1":
                continue
            cl, cr = r["codec_left"], r["codec_right"]
            if cl == "0" or cr == "0":
                continue
            qt = "same" if cl == cr else "cross"
            g = f"{st}:{r['question_id']}"
            e = counts.setdefault(g, {"left": 0.0, "right": 0.0, "L": r["img_left"], "R": r["img_right"],
                                      "img": int(r["img_num"]), "qt": qt,  # source image; AIC-3 BTC and SDR25 BTC share images 2,6,7,9,10
                                      "cp": " vs ".join(sorted((FC_CODE[cl], FC_CODE[cr]))) if qt == "cross" else f"{FC_CODE[cl]} (same)",
                                      "gap": abs(int(r["dlevel_left"]) - int(r["dlevel_right"]))})
            assert e["L"] == r["img_left"] and e["R"] == r["img_right"]
            e[r["response"]] += 1.0
    imgs = sorted({e["img"] for e in counts.values()})
    rng = random.Random(SEED)
    img_draws = [[imgs[rng.randrange(len(imgs))] for _ in range(len(imgs))] for _ in range(B)]
    metrics = list(FC_SCORERS.values())

    def run(keys, tag):
        by_img = defaultdict(list)
        for n, k in enumerate(keys):
            by_img[counts[k]["img"]].append(n)
        man = f"{scratch}/{tag}__img.tsv"
        with open(man, "w") as f:
            f.write("POINT\t*\n")
            for b, dr in enumerate(img_draws):
                pick = [str(n) for im in dr for n in by_img.get(im, [])]
                if pick:
                    f.write(f"B{b}\t" + ",".join(pick) + "\n")
        out = {}
        for col, name in FC_SCORERS.items():
            rf = f"{scratch}/{tag}__{name}.tsv"
            with open(rf, "w") as f:
                f.write("group\ts_left\ts_right\tchoice\tweight\n")
                for k in keys:
                    e = counts[k]
                    sl, sr = score[e["L"]][col], score[e["R"]][col]
                    for ch in ("left", "right"):
                        if e[ch] > 0:
                            f.write(f"{k}\t{sl}\t{sr}\t{ch}\t{e[ch]:g}\n")
            out[name] = parse_panel(panel_bin, rf, man, B)
        return out

    res = {"n_images": len(imgs), "peers": ["ssim2"], "zensim": ["C", "B"], "classes": {}}
    stats = {}
    for k in ("same", "cross"):
        keys = sorted(g for g, e in counts.items() if e["qt"] == k)
        st = run(keys, f"fc__{k}")
        stats[k] = st
        res["classes"][k] = {
            "n_questions": len(keys), "n_responses": sum(counts[g]["left"] + counts[g]["right"] for g in keys),
            "n_images": len({counts[g]["img"] for g in keys}), "best_peer": "ssim2",
            "acc": {m: {"point": round(st[m]["point"], 4), "ci": ci_of(st[m]["boot"]),
                        "tie_rate": round(st[m]["tie_rate"], 4)} for m in metrics},
            "d_best": {z: [round(st[z]["point"] - st["ssim2"]["point"], 4)] + ci_of(dvec(st[z], st["ssim2"])) for z in ("C", "B")},
        }
    dec = {}
    for z in ("C", "B"):
        dx, ds = dvec(stats["cross"][z], stats["cross"]["ssim2"]), dvec(stats["same"][z], stats["same"]["ssim2"])
        dec[z] = {"dx": res["classes"]["cross"]["d_best"][z], "ds": res["classes"]["same"]["d_best"][z],
                  "did": [round(res["classes"]["cross"]["d_best"][z][0] - res["classes"]["same"]["d_best"][z][0], 4)]
                  + ci_of([a - b for a, b in zip(dx, ds)])}
        dec[z]["ds_fixedpeer"], dec[z]["did_fixedpeer"] = dec[z]["ds"], dec[z]["did"][1:]
    res["decision_inputs"] = dec
    strata = {}
    cross_keys = sorted(g for g, e in counts.items() if e["qt"] == "cross")
    sel = {"gap_small": lambda e: e["gap"] <= 1, "gap_large": lambda e: e["gap"] >= 2}
    for cp in sorted({counts[g]["cp"] for g in cross_keys}):
        sel[f"pair:{cp}"] = (lambda c: (lambda e: e["cp"] == c))(cp)
    for name, f in sel.items():
        keys = [g for g in cross_keys if f(counts[g])]
        if not keys:
            continue
        st = run(keys, f"fc__cross__{name.replace(' ', '').replace(':', '_')}")
        strata[name] = {"n_questions": len(keys), "acc": {m: round(st[m]["point"], 4) for m in metrics},
                        "d_best": {z: [round(st[z]["point"] - st["ssim2"]["point"], 4)] + ci_of(dvec(st[z], st["ssim2"])) for z in ("C", "B")}}
    strata["_gap_threshold"] = "|dlevel_left - dlevel_right| <= 1 small"
    res["strata_cross"] = strata
    # error tables: per-question strict majority (exact splits excluded)
    errs = {}
    for k in ("same", "cross"):
        pm = []
        for g, e in counts.items():
            if e["qt"] != k or e["left"] == e["right"]:
                continue
            pm.append((e["cp"], e["L"], e["R"], "left" if e["left"] > e["right"] else "right"))
        col = {v: c for c, v in FC_SCORERS.items()}
        errs[k] = tally(pm, lambda m, key, side: float(score[key[1] if side == "L" else key[2]][col[m]]), ["C", "B"], "ssim2")
    res["error_tables"] = {"peer": "ssim2", "cols": "[n, metric_wrong_peer_right, metric_right_peer_wrong, metric_tie]",
                           "note": "per-question strict majority response; split questions excluded", "by_class": errs}
    return res


# ---------------------------------------------------------------- decision

def classify(d):
    dx_lo, dx_hi = d["dx"][1], d["dx"][2]
    ds_lo, ds_hi = d["ds"][1], d["ds"][2]
    did_hi = d["did"][2]
    if dx_hi < 0 and ds_hi >= 0:
        return "CONF"
    if ds_hi < 0 and not did_hi < 0:
        return "NONSPEC"
    if not dx_hi < 0 and not ds_hi < 0:
        return "NONE"
    return "PARTIAL"


def classify_fixed(d):
    return classify({"dx": d["dx"], "ds": d["ds_fixedpeer"], "did": [None] + d["did_fixedpeer"]})


def decide(units, z, cls_fn, family_once=True):
    fam_members = ["aic3", "aic4crop", "fc_btc_native"]
    lab = {}
    for u in ["cid22a", "csiq"] + fam_members:
        d = units.get(u, {}).get("decision_inputs", {}).get(z)
        lab[u] = cls_fn(d) if d else "MISSING"
    if family_once:
        mem = [lab[m] for m in fam_members if lab[m] != "MISSING"]
        if not mem:
            fam = "MISSING"
        elif "CONF" in mem and "NONSPEC" not in mem:
            fam = "CONF"
        elif "NONSPEC" in mem and "CONF" not in mem:
            fam = "NONSPEC"
        elif "CONF" in mem:
            fam = "MIXED"
        else:
            fam = "NONE" if all(x == "NONE" for x in mem) else "PARTIAL"
        counted = {"cid22a": lab["cid22a"], "csiq": lab["csiq"], "jpeg_aic_family": fam}
    else:
        counted = dict(lab)
    nc = sum(v == "CONF" for v in counted.values())
    nn = sum(v == "NONSPEC" for v in counted.values())
    if nc >= 2 and nn == 0:
        out = "CONFIRMED"
    elif nn >= 1 and nn >= nc:
        out = "REFUTED"
    else:
        out = "UNRESOLVED"
    return {"outcome": out, "n_conf": nc, "n_nonspec": nn, "units": counted, "members": lab}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-B", type=int, default=2000)
    ap.add_argument("--scratch", default="/var/tmp/rev4-e1b/scratch")
    ap.add_argument("--out", default="/var/tmp/rev4-e1b/e1b_full.json")
    ap.add_argument("--units", default=",".join(list(UNITS) + ["fc_btc_native"]))
    a = ap.parse_args()
    os.makedirs(a.scratch, exist_ok=True)
    panel_bin = A.zen_stats._find_panel_bin()
    result = {"seed": SEED, "B": a.B, "panel_bin": panel_bin, "units": {}}
    for u in a.units.split(","):
        r = fc_unit(a.B, a.scratch, panel_bin) if u == "fc_btc_native" else table_unit(u, a.B, a.scratch, panel_bin)
        result["units"][u] = r
        print(u, {k: v.get("n_pairs", v.get("n_questions")) for k, v in r["classes"].items()}, flush=True)
        json.dump(result, open(a.out, "w"), indent=1)
    U = result["units"]
    dec = {}
    for z in ZENSIM:
        dec[z] = {"primary": decide(U, z, classify),
                  "sens_members_separate": decide(U, z, classify, family_once=False),
                  "sens_fixed_peer": decide(U, z, classify_fixed)}
        if "cid22a_encdir" in U:
            U2 = dict(U)
            U2["cid22a"] = U["cid22a_encdir"]
            dec[z]["sens_cid22_encdir"] = decide(U2, z, classify)
        dec[z]["aic4full_class"] = classify(U["aic4full"]["decision_inputs"][z]) if z in U.get("aic4full", {}).get("decision_inputs", {}) else "MISSING"
        dec[z]["tid_class"] = classify(U["tid"]["decision_inputs"][z]) if z in U.get("tid", {}).get("decision_inputs", {}) else "MISSING"
    result["decisions"] = dec
    json.dump(result, open(a.out, "w"), indent=1)
    print(json.dumps({z: (d["primary"]["outcome"], d["primary"]["units"]) for z, d in dec.items()}, indent=0))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
