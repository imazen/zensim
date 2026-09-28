#!/usr/bin/env python3
"""Rev4 E1 — assemble per-stimulus tables from EXISTING per-pair outputs.

No metric is computed here and no statistic: this only joins stored scores to
stimulus keys (reference, distorted file) and orients them (higher = better
quality). Every join is asserted (row counts, key uniqueness, target equality),
so a mis-aligned row fails loudly instead of producing a plausible number.

CID22: rows are filtered to the 25 CID22-A references by `ref_path` BEFORE any
target value is extracted. CID22-B(24) is sealed (docs/DATA_SPLITS.md,
exposure ledger 2026-09-19) and never leaves this filter.

Output: <out>/<corpus>.tsv  (stim, ref, t, <metric>...), <out>/manifest.json.
Registered in benchmarks/rev4_e1_prereg_2026-09-23.md.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys

import pyarrow.parquet as pq

FE = "/mnt/v/output/zensim/reports/fulleval"
RM = "/mnt/v/output/zensim/reports/refmetrics"
A4 = "/mnt/v/output/zensim/aic4-refresh-2026-09-22"
SITE = "/home/lilith/work/zen/zensim/site/data/parquet"
LAD = "/home/lilith/work/zensim-validation-2026-09-14/rev3-public-human-eval/ladder"
KON_SELECT = "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_konjnd_jpeg_select_2026-08-29.parquet"
PUB = "/mnt/v/repos/iqa-tools/jpeg-aic__JPEG-AIC-4-datasets/JPEG-AIC_metric_scores.csv"

CID22_A = {
    "1189261.png", "1531677.png", "159550.png", "1624487.png", "162520.png",
    "164595.png", "2079234.png", "21169144185_3f7977cb5a_o.png", "225228.png",
    "2389166.png", "2936831.png", "3316926.png", "3653963.png", "373965.png",
    "3762075.png", "4215100.png", "6078297.png", "6292444.png", "70497.png",
    "7062219.png", "844297.png", "pexels-photo-2686358.png",
    "pexels-photo-2802032.png", "pexels-photo-4210863.png",
    "ularapi_Semarang_City_Logo.png",
}
assert len(CID22_A) == 25

# Metric provenance labels: printed with every table (one era / identity per column).
LABELS = {
    "ssim2": "SSIMULACRA2 (fast-ssim2, our impl.; parity vs published AIC-4 column SROCC 1.0000)",
    "butter_p3": "butteraugli 3-norm (butteraugli-gpu; negated)",
    "butter_max": "butteraugli max-norm (butteraugli-gpu; negated; board headline norm)",
    "iwssim": "IW-SSIM (iwssim-gpu, our impl.)",
    "cvvdp_4k": "CVVDP our port @ standard_4k (NOT the documented display)",
    "cvvdp_fhd": "CVVDP our port @ standard_fhd (AIC CTC display; parity with organisers 0.00027 JOD) [cvvdpfix lane, unmerged]",
    "dssim": "DSSIM (zen-metrics dssim-gpu via site/data/parquet; negated)",
    "pub_iwssim": "IW-SSIM, organisers' published column",
    "pub_msssim": "MS-SSIM, organisers' published column",
    "pub_psnry": "PSNR-Y, organisers' published column",
    "pub_cvvdp": "CVVDP, organisers' published column (pycvvdp 0.4.2 standard_fhd)",
    "ctx_ssim": "SSIM, organisers' published column [context]",
    "ctx_vmafneg": "VMAF-neg, organisers' published column [context]",
    "ctx_hdrvdp2": "HDR-VDP-2 Q, organisers' published column [context]",
    "ctx_hdrvdp3": "HDR-VDP-3 Q, organisers' published column [context]",
    "B": "zensim B",
    "C": "zensim C = W10L9PH_s4004_packed",
    "D": "zensim D",
    "R915_fast": "zensim Rev3 fast R915_y60_h32_ens5",
    "R915_rich": "zensim Rev3 rich R915_basic228_h128_ens5",
    "V0_2": "zensim PreviewV0_2 (AIC-4: profile pixel read; CID22/AIC-3: site parquet score_v0_2_linear, May era, negated)",
}
PEERS = ["ssim2", "butter_p3", "butter_max", "iwssim", "cvvdp_4k", "cvvdp_fhd", "dssim",
         "pub_iwssim", "pub_msssim", "pub_psnry", "pub_cvvdp"]
CONTEXT = ["ctx_ssim", "ctx_vmafneg", "ctx_hdrvdp2", "ctx_hdrvdp3"]
ZENSIM = ["B", "C", "D", "R915_fast", "R915_rich", "V0_2"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def tsv(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


_FE_CACHE = {}


def fe(name):
    if name not in _FE_CACHE:
        _FE_CACHE[name] = json.load(open(f"{FE}/{name}.fulleval.json"))["per_pair"]
    return _FE_CACHE[name]


def fe_vec(name, corpus):
    pp = fe(name)[corpus]
    tgt = pp.get("mos") if "mos" in pp else pp["jnd"]
    return pp["pred"], tgt


def base(p):
    return os.path.basename(p)


def by_key(rows, key):
    d = {}
    for r in rows:
        k = r[key]
        assert k not in d, f"duplicate key {k}"
        d[k] = r
    return d


def close(a, b, tol):
    return abs(float(a) - float(b)) <= tol


def attach_fe_in_tsv_order(tab, keep_idx, name, corpus, tsv_targets, scale, col, tol=1e-6):
    """Model per-pair vector in the peer TSV's row order (verified on kept rows only)."""
    pred, tgt = fe_vec(name, corpus)
    assert len(pred) == len(tsv_targets), f"{name}/{corpus}: {len(pred)} vs {len(tsv_targets)} rows"
    for out_row, i in zip(tab, keep_idx):
        assert close(tgt[i] * scale, tsv_targets[i], tol), f"{name}/{corpus} row {i}: target mismatch"
        out_row[col] = float(pred[i])
    return {"source": f"{FE}/{name}.fulleval.json", "per_pair": corpus}


def write(out, corpus, tab, cols, meta):
    p = f"{out}/{corpus}.tsv"
    with open(p, "w") as f:
        f.write("\t".join(["stim", "ref", "t"] + cols) + "\n")
        for r in tab:
            f.write("\t".join([r["stim"], r["ref"], repr(float(r["t"]))] +
                              [repr(float(r[c])) if c in r and r[c] is not None else "" for c in cols]) + "\n")
    meta["rows"] = len(tab)
    meta["refs"] = len({r["ref"] for r in tab})
    meta["columns"] = {c: LABELS[c] for c in cols}
    meta["sha256"] = sha(p)
    return meta


def cid22(out):
    keys = tsv(f"{RM}/cid22_ssim2.tsv")
    keep = [i for i, r in enumerate(keys) if base(r["ref_path"]) in CID22_A]
    # From here only kept (A) rows are touched.
    tab = [{"stim": keys[i]["dist_id"], "ref": base(keys[i]["ref_path"]),
            "t": float(keys[i]["MCOS"]), "ssim2": float(keys[i]["ssim2"])} for i in keep]
    tsv_t = {i: float(keys[i]["MCOS"]) for i in keep}
    tvec = [tsv_t.get(i, float("nan")) for i in range(len(keys))]
    for fn, cols in (("cid22_butter.tsv", {"butteraugli_pnorm3": "butter_p3", "butteraugli_max": "butter_max"}),
                     ("cid22_iwssim.tsv", {"iwssim_gpu": "iwssim"}),
                     ("cid22_cvvdp.tsv", {"cvvdp_cpu_imazen_v0_1_0": "cvvdp_4k"})):
        d = by_key(tsv(f"{RM}/{fn}"), "dist_path")
        for r, i in zip(tab, keep):
            src = d[keys[i]["dist_path"]]
            assert close(src["MCOS"], keys[i]["MCOS"], 1e-9)
            for c, o in cols.items():
                v = float(src[c])
                r[o] = -v if o.startswith("butter") else v
    srcs = {}
    for name, col in (("MT914_matched_B", "B"), ("W10L9PH_s4004_packed", "C"), ("MT914_matched_D", "D"),
                      ("R915_y60_h32_ens5", "R915_fast"), ("R915_basic228_h128_ens5", "R915_rich")):
        srcs[col] = attach_fe_in_tsv_order(tab, keep, name, "cid22", tvec, 100.0, col)
    # site parquet (May era): DSSIM + PreviewV0_2, joined by dist_path.
    t = pq.read_table(f"{SITE}/cid22.parquet", columns=["ref_path", "dist_path", "score_dssim", "score_v0_2_linear"]).to_pylist()
    site = {}
    for r in t:
        if base(r["ref_path"]) in CID22_A and r["dist_path"] != r["ref_path"]:
            assert r["dist_path"] not in site, r["dist_path"]
            site[r["dist_path"]] = r
    for r, i in zip(tab, keep):
        s = site[keys[i]["dist_path"]]
        r["dssim"] = -float(s["score_dssim"])
        r["V0_2"] = -float(s["score_v0_2_linear"])  # distance-oriented (label-free check vs ssim2: SROCC -0.99)
    cols = ["ssim2", "butter_p3", "butter_max", "iwssim", "cvvdp_4k", "dssim", "B", "C", "D", "R915_fast", "R915_rich", "V0_2"]
    return write(out, "cid22a", tab, cols, {"population": "CID22-A(25) rows only", "models": srcs,
                                           "site_parquet": f"{SITE}/cid22.parquet"})


def csiq(out):
    keys = tsv(f"{RM}/csiq_ssim2_gpu.tsv")
    tab = [{"stim": base(r["dist_path"]), "ref": base(r["ref_path"]), "t": float(r["human_score"]),
            "ssim2": float(r["ssim2_gpu"])} for r in keys]
    idx = list(range(len(keys)))
    tvec = [float(r["human_score"]) for r in keys]
    for fn, cols in (("csiq_butteraugli_gpu.tsv", {"butteraugli_pnorm3_gpu": "butter_p3", "butteraugli_max_gpu": "butter_max"}),
                     ("csiq_iwssim_gpu.tsv", {"iwssim_gpu": "iwssim"}),
                     ("csiq_cvvdp.tsv", {"cvvdp_cpu_imazen_v0_1_0": "cvvdp_4k"})):
        rows = tsv(f"{RM}/{fn}")
        assert [r["dist_path"] for r in rows] == [r["dist_path"] for r in keys], fn
        for r, src in zip(tab, rows):
            for c, o in cols.items():
                v = float(src[c])
                r[o] = -v if o.startswith("butter") else v
    srcs = {}
    for name, col in (("b_sdr_linear_cid80_inclwinsor_dense_dial@cur372", "B"), ("W10L9PH_s4004_packed", "C"),
                      ("D_shipped@dguard2", "D")):
        srcs[col] = attach_fe_in_tsv_order(tab, idx, name, "csiq", tvec, 1.0, col, 1e-12)
    cols = ["ssim2", "butter_p3", "butter_max", "iwssim", "cvvdp_4k", "B", "C", "D"]
    return write(out, "csiq", tab, cols, {"population": "CSIQ 866/30", "models": srcs})


def konjnd(out):
    sel = set(pq.read_table(KON_SELECT, columns=["ref_basename"]).column("ref_basename").to_pylist())
    assert len(sel) == 404
    keys_all = tsv(f"{RM}/konjnd_ssim2_heldout.tsv")
    jidx = [i for i, r in enumerate(keys_all) if "/jpeg/" in r["dist_path"]]
    assert len(jidx) == 504
    src_of = lambda r: base(r["ref_path"]).rsplit(".", 1)[0]
    keep504 = [k for k, i in enumerate(jidx) if src_of(keys_all[i]) in sel]  # positions within the 504
    assert len(keep504) == 404
    keys = [keys_all[jidx[k]] for k in keep504]
    tab = [{"stim": base(r["dist_path"]), "ref": src_of(r), "t": -float(r["pjnd"]), "ssim2": float(r["ssim2_gpu"])} for r in keys]
    for fn, cols in (("konjnd_butteraugli_heldout.tsv", {"butteraugli_pnorm3_gpu": "butter_p3", "butteraugli_max_gpu": "butter_max"}),
                     ("konjnd_iwssim_heldout.tsv", {"iwssim_gpu": "iwssim"}),
                     ("konjnd_cvvdp_heldout.tsv", {"cvvdp_cpu_imazen_v0_1_0": "cvvdp_4k"})):
        d = by_key(tsv(f"{RM}/{fn}"), "dist_path")
        for r, k in zip(tab, keys):
            src = d[k["dist_path"]]
            for c, o in cols.items():
                v = float(src[c])
                r[o] = -v if o.startswith("butter") else v
    # 504-order fullevals (TSV jpeg order): C.
    t504 = [float(keys_all[i]["pjnd"]) for i in jidx]
    srcs = {"C": attach_fe_in_tsv_order(tab, keep504, "W10L9PH_s4004_packed", "konjnd", t504, 1.0, "C", 1e-9)}
    # 404-order fullevals (rev1 / rev3 roots carry ref_basename per row).
    pos = {r["ref"]: j for j, r in enumerate(tab)}
    for name, col, par in (("MT914_matched_B", "B", f"{LAD}/features-rev1/konjnd_features_372col_2026-05-15.parquet"),
                           ("MT914_matched_D", "D", f"{LAD}/features-rev1/konjnd_features_372col_2026-05-15.parquet"),
                           ("R915_y60_h32_ens5", "R915_fast", f"{LAD}/features-rev3/ext_konjnd_jpeg_val.parquet"),
                           ("R915_basic228_h128_ens5", "R915_rich", f"{LAD}/features-rev3/ext_konjnd_jpeg_val.parquet")):
        refs = [x.split(":", 1)[1] for x in pq.read_table(par, columns=["ref_basename"]).column("ref_basename").to_pylist()]
        pred, tgt = fe_vec(name, "konjnd")
        assert len(refs) == len(pred) == 404 and set(refs) == set(pos)
        for rf, p_, tg in zip(refs, pred, tgt):
            r = tab[pos[rf]]
            assert close(tg, -r["t"], 1e-6), (name, rf)
            r[col] = float(p_)
        srcs[col] = {"source": f"{FE}/{name}.fulleval.json", "per_pair": "konjnd", "row_ids": par, "row_ids_sha256": sha(par)}
    cols = ["ssim2", "butter_p3", "butter_max", "iwssim", "cvvdp_4k", "B", "C", "D", "R915_fast", "R915_rich"]
    return write(out, "konjnd404", tab, cols, {"population": "KonJND JPEG SELECT 404 (TERMINAL-100 dropped by ref membership)",
                                               "models": srcs, "select_view": KON_SELECT})


def aic3(out):
    keys = tsv(f"{RM}/aic3_ssim2_heldout.tsv")
    tab = [{"stim": base(r["dist_path"]), "ref": base(r["ref_path"]), "t": float(r["jnd"]), "ssim2": float(r["ssim2_gpu"])} for r in keys]
    tvec = [float(r["jnd"]) for r in keys]
    for fn, cols in (("aic3_butteraugli_heldout.tsv", {"butteraugli_pnorm3_gpu": "butter_p3", "butteraugli_max_gpu": "butter_max"}),
                     ("aic3_iwssim_heldout.tsv", {"iwssim_gpu": "iwssim"}),
                     ("aic3_cvvdp_heldout.tsv", {"cvvdp_cpu_imazen_v0_1_0": "cvvdp_4k"})):
        rows = tsv(f"{RM}/{fn}")
        assert [r["dist_path"] for r in rows] == [r["dist_path"] for r in keys], fn
        for r, src in zip(tab, rows):
            for c, o in cols.items():
                v = float(src[c])
                r[o] = -v if o.startswith("butter") else v
    srcs = {}
    idx = list(range(len(keys)))
    for name, col in (("MT914_matched_B", "B"), ("W10L9PH_s4004_packed", "C"), ("MT914_matched_D", "D"),
                      ("R915_y60_h32_ens5", "R915_fast"), ("R915_basic228_h128_ens5", "R915_rich")):
        srcs[col] = attach_fe_in_tsv_order(tab, idx, name, "aic3", tvec, 1.0, col, 1e-9)
    site = by_key(pq.read_table(f"{SITE}/aic3_ctc_epfl.parquet", columns=["dist_path", "human_jnd", "score_dssim", "score_v0_2_linear"]).to_pylist(), "dist_path")
    for r, k in zip(tab, keys):
        s = site[k["dist_path"]]
        r["dssim"] = -float(s["score_dssim"])
        r["V0_2"] = -float(s["score_v0_2_linear"])  # distance-oriented (label-free check vs ssim2: SROCC -0.99)
    cols = ["ssim2", "butter_p3", "butter_max", "iwssim", "cvvdp_4k", "dssim", "B", "C", "D", "R915_fast", "R915_rich", "V0_2"]
    return write(out, "aic3", tab, cols, {"population": "AIC-3 CTC 600/10, decoded CTC PNGs at CTC source resolution",
                                          "models": srcs, "site_parquet": f"{SITE}/aic3_ctc_epfl.parquet"})


def aic4(out, leg):
    pairs = tsv(f"{A4}/aic4_pairs.tsv" if leg == "crop" else f"{A4}/aic4_fullres_pairs.tsv")
    crop = tsv(f"{A4}/aic4_pairs.tsv")
    assert [r["human_score"] for r in pairs] == [r["human_score"] for r in crop]
    hkey = lambda x: f"{float(x):.7e}"
    tab = [{"stim": base(c["dist_path"]).replace("PTC_", ""), "ref": base(c["ref_path"]).split("_")[1],
            "t": -float(r["human_score"]), "_h": hkey(r["human_score"]), "_crop_dist": c["dist_path"]} for r, c in zip(pairs, crop)]
    assert len({r["_h"] for r in tab}) == 300
    pos = {r["_h"]: j for j, r in enumerate(tab)}
    zs = pq.read_table(f"{A4}/zensim_scores_{leg}.parquet").to_pylist()
    assert len(zs) == 300
    zmap = {"score_b": "B", "score_c": "C", "score_d": "D", "score_r915_y60_h32_ens5": "R915_fast",
            "score_r915_basic228_h128_ens5": "R915_rich", "score_v0_2": "V0_2"}
    for z in zs:
        r = tab[pos[hkey(z["human_score"])]]
        for k, o in zmap.items():
            r[o] = float(z[k])
        # the MT914 matched bakes are the embedded profile bakes: must be identical
        assert z["score_mt914_matched_b"] == z["score_b"] and z["score_mt914_matched_d"] == z["score_d"]
    ps = tsv(f"{A4}/peer_scores_{leg}.tsv")
    assert len(ps) == 300
    for p in ps:
        r = tab[pos[hkey(p["human_score"])]]
        r["ssim2"] = float(p["ssim2"])
        r["butter_p3"] = -float(p["butter_p3"])
        r["butter_max"] = -float(p["butter_max"])
    cols = ["ssim2", "butter_p3", "butter_max"]
    meta = {"population": f"AIC-4 sample {leg} 300/5", "zensim_scores": f"{A4}/zensim_scores_{leg}.parquet",
            "peer_scores": f"{A4}/peer_scores_{leg}.tsv", "note": "boardfix lane pixel read (unmerged lab record)"}
    if leg == "crop":
        byd = {r["_crop_dist"]: r for r in tab}
        for fn, c, o in (("aic4_iwssim_gpu.tsv", "iwssim_gpu", "iwssim"), ("aic4_cvvdp.tsv", "cvvdp_cpu_imazen_v0_1_0", "cvvdp_4k"),
                         ("aic4_cvvdp_standard_fhd.tsv", "cvvdp_cpu_imazen_v0_1_0_standard_fhd", "cvvdp_fhd")):
            rows = tsv(f"{RM}/{fn}")
            assert len(rows) == 300
            for s in rows:
                r = byd[s["dist_path"]]
                assert close(s["human_score"], -r["t"], 1e-6)
                r[o] = float(s[c])
        site = by_key(pq.read_table(f"{SITE}/aic4_sample.parquet", columns=["dist_path", "human_jnd", "score_dssim"]).to_pylist(), "dist_path")
        for r in tab:
            s = site[r["_crop_dist"]]
            assert close(s["human_jnd"], -r["t"], 1e-6)
            r["dssim"] = -float(s["score_dssim"])
        with open(PUB) as f:
            pub = by_key(list(csv.DictReader(f)), "img_distorted")
        for r in tab:
            s = pub[base(r["_crop_dist"])]
            assert close(s["distortion"], -r["t"], 1e-6)
            r["pub_iwssim"] = float(s["IW-SSIM"])
            r["pub_msssim"] = float(s["MS-SSIM"])
            r["pub_psnry"] = float(s["PSNR-Y"])
            r["pub_cvvdp"] = float(s["CVVDP"])
            r["ctx_ssim"] = float(s["SSIM"])
            r["ctx_vmafneg"] = float(s["VMAF-neg"])
            r["ctx_hdrvdp2"] = float(s["HDR-VDP-2 Q"])
            r["ctx_hdrvdp3"] = float(s["HDR-VDP-3 Q"])
        cols += ["iwssim", "cvvdp_4k", "cvvdp_fhd", "dssim", "pub_iwssim", "pub_msssim", "pub_psnry", "pub_cvvdp"] + CONTEXT
        meta["published"] = PUB
    cols += ["B", "C", "D", "R915_fast", "R915_rich", "V0_2"]
    return write(out, f"aic4{leg}", tab, cols, meta)


def sdr25(out):
    keys = tsv(f"{RM}/sdr25_ssim2.tsv")
    tab = [{"stim": base(r["dist_path"]), "ref": r["img_num"], "t": -float(r["q_jnd"]), "ssim2": float(r["ssim2"])} for r in keys]
    tvec = [float(r["q_jnd"]) for r in keys]
    for fn, cols in (("sdr25_butteraugli_gpu.tsv", {"butteraugli_pnorm3_gpu": "butter_p3", "butteraugli_max_gpu": "butter_max"}),
                     ("sdr25_iwssim_gpu.tsv", {"iwssim_gpu": "iwssim"}),
                     ("sdr25_cvvdp.tsv", {"cvvdp_cpu_imazen_v0_1_0": "cvvdp_4k"})):
        rows = tsv(f"{RM}/{fn}")
        assert [r["dist_path"] for r in rows] == [r["dist_path"] for r in keys], fn
        for r, src in zip(tab, rows):
            for c, o in cols.items():
                v = float(src[c])
                r[o] = -v if o.startswith("butter") else v
    srcs = {"C": attach_fe_in_tsv_order(tab, list(range(50)), "W10L9PH_s4004_packed", "sdr25", tvec, 1.0, "C", 1e-12)}
    cols = ["ssim2", "butter_p3", "butter_max", "iwssim", "cvvdp_4k", "C"]
    return write(out, "sdr25", tab, cols, {"population": "SDR25 q_jnd table 50/5", "models": srcs})


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    man = {"script": os.path.abspath(__file__), "tables": {}}
    for fn in (cid22, csiq, konjnd, aic3, lambda o: aic4(o, "crop"), lambda o: aic4(o, "full"), sdr25):
        m = fn(out)
        name = m["population"]
        man["tables"][name] = m
        print(f"{name}: rows={m['rows']} refs={m['refs']} cols={list(m['columns'])} sha256={m['sha256'][:16]}")
    json.dump(man, open(f"{out}/manifest.json", "w"), indent=1)


if __name__ == "__main__":
    main()
