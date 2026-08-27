#!/usr/bin/env python3
"""Peer reference-metric rows for the summer-gauntlet board (user request
2026-08-28: "put ssim2 and butter and cvvdp as peers").

Builds `peer_<metric>.fulleval.json` rows from the STORED per-pair reference
scores in `/mnt/v/output/zensim/reports/refmetrics/` (ref_path, dist_path,
human label, metric value — cid22/kadid/tid/aic3/konjnd; produced by
`run_gpu_metrics.sh` in that dir). No image is rescored; every stat comes
from the canonical panel (`zen_stats.panel` → zenstats). Butteraugli is a
DISTANCE: predictions are negated for quality orientation (recorded in the
row). Corpora without stored per-pair peer scores are OMITTED from the row —
the board renders absent axes as em-dashes (never zeros). imazen26/nonphoto/
hfnlproxy are deliberately omitted for ssim2 (the slice target IS
ssim2-derived — a trivially perfect self-row would mislead) and for the
others until the encode-key sidecar join lands (registered follow-up).
"""
import json, os, sys, csv

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from zen_stats import panel  # noqa: E402

RM = "/mnt/v/output/zensim/reports/refmetrics"
OUT = "/mnt/v/output/zensim/reports/fulleval"

# (peer, orientation sign, {board corpus: (tsv, human col, metric col)})
PEERS = {
    "ssim2": (+1, {
        "cid22": ("cid22_ssim2.tsv", "MCOS", "ssim2"),
        "kadid": ("kadid_ssim2_gpu.tsv", None, None),
        "tid": ("tid_ssim2_gpu.tsv", None, None),
        "aic3": ("aic3_ssim2_heldout.tsv", None, None),
        "konjnd": ("konjnd_ssim2_heldout.tsv", None, None),
    }),
    "butteraugli": (-1, {
        "cid22": ("cid22_butter.tsv", None, None),
        "kadid": ("kadid_butteraugli_gpu.tsv", None, None),
        "tid": ("tid_butteraugli_gpu.tsv", None, None),
        "aic3": ("aic3_butteraugli_heldout.tsv", None, None),
        "konjnd": ("konjnd_butteraugli_heldout.tsv", None, None),
    }),
    "cvvdp": (+1, {
        "cid22": ("cid22_cvvdp.tsv", None, None),
        "kadid": ("kadid_cvvdp_gpu.tsv", None, None),
        "tid": ("tid_cvvdp_gpu.tsv", None, None),
        "aic3": ("aic3_cvvdp_heldout.tsv", None, None),
        "konjnd": ("konjnd_cvvdp_heldout.tsv", None, None),
    }),
    "iwssim": (+1, {
        "cid22": ("cid22_iwssim.tsv", None, None),
        "kadid": ("kadid_iwssim.tsv", None, None),
        "tid": ("tid_iwssim.tsv", None, None),
        "aic3": ("aic3_iwssim_heldout.tsv", None, None),
        "konjnd": ("konjnd_iwssim_heldout.tsv", None, None),
    }),
}

# 2026-08-28 completion (user: "don't skip any"): csiq/live/aic4/sdr25 —
# cvvdp scored CPU locally (the CPU rung is sanctioned for cvvdp); ssim2/
# butter/iwssim scored on the wsl RTX 5070 through the baked exec-gpu-cuda13
# container (the GPU-only rule); LIVE runs on the PNG-converted mirror.
# Label semantics per corpus: csiq human_score = 1-DMOS (quality); live =
# the canonical builder's quality orientation; aic4 = signed JND
# (|SROCC| convention — srocc_signed may read negative by design); sdr25 =
# q_jnd (the 50-pair instrument = the board axis population exactly).
for peer in list(PEERS):
    sign, corp = PEERS[peer]
    stem = {"ssim2": "ssim2", "butteraugli": "butter", "cvvdp": "cvvdp", "iwssim": "iwssim"}[peer]
    for c in ("csiq", "live", "aic4", "sdr25"):
        for cand in (f"{c}_{stem}_gpu.tsv", f"{c}_{stem}.tsv"):
            if os.path.exists(os.path.join(RM, cand)):
                corp[c] = (cand, None, None)
                break

HUMAN_CANDS = ["MCOS", "DMOS", "MOS", "mos", "jnd", "q_jnd", "human", "human_score", "dmos", "pjnd", "label"]

def load_pairs(path):
    with open(path) as f:
        rd = csv.DictReader(f, delimiter="\t")
        rows = list(rd)
    if not rows:
        return None
    cols = rows[0].keys()
    hcol = next((c for c in HUMAN_CANDS if c in cols), None)
    # metric col = the last non-path, non-human, non-id numeric column
    mcands = [c for c in cols if c not in ("ref_path", "dist_path", "dist_id", "ref_basename") and c != hcol]
    # butteraugli tables carry two norms; the family's headline is _max
    mmax = [c for c in mcands if "max" in c]
    mcol = (mmax[0] if mmax else (mcands[-1] if mcands else None))
    if hcol is None or mcol is None:
        return None
    xs, ys = [], []
    for r in rows:
        try:
            m = float(r[mcol]); h = float(r[hcol])
        except (TypeError, ValueError):
            continue
        xs.append(m); ys.append(h)
    return xs, ys, hcol, mcol

def main():
    for peer, (sign, corpora) in PEERS.items():
        rank = {}
        prov = {}
        per_pair_all = {}
        for corpus, (tsv, _, _) in corpora.items():
            p = os.path.join(RM, tsv)
            if not os.path.exists(p):
                continue
            got = load_pairs(p)
            if not got:
                continue
            xs, ys, hcol, mcol = got
            pred = [sign * v for v in xs]
            st = panel(pred, ys)
            # panel()'s srocc is ABS (zenstats convention); the signed value
            # comes from the batch full path — never alias abs into signed.
            try:
                from zen_stats import panel_batch
                sb = panel_batch([("p", pred, ys)])[0]
                signed = sb.get("srocc_signed")
            except Exception:
                signed = None
            import random as _r
            _r.seed(11)
            k = min(5000, len(pred))
            samp = sorted(_r.sample(range(len(pred)), k))
            per_pair_local = {"pred": [pred[i] for i in samp], "mos": [ys[i] for i in samp]}
            rank[corpus] = {"srocc": st["srocc"],
                            **({"srocc_signed": signed} if signed is not None else {}),
                            "plcc": st["plcc"], "krocc": st["krocc"], "or": st["or"],
                            "pwrc": st["pwrc"], "z_rmse": st["z_rmse"], "n": st["n"]}
            per_pair_all[corpus] = per_pair_local
            prov[corpus] = {"tsv": tsv, "human_col": hcol, "metric_col": mcol,
                            "n": st["n"], "oriented": "negated" if sign < 0 else "as-is"}
        doc = {
            "name": f"peer_{peer}",
            "regime": "reference-metric",
            "n_inputs": None,
            "bake": None,
            "model": {"kind": "reference-metric",
                      "note": ("classical reference metric scored from the STORED per-pair "
                               "tables in reports/refmetrics/ via the canonical panel; "
                               "butteraugli negated for quality orientation. Axes without "
                               "stored per-pair scores are absent (em-dash), incl. "
                               "imazen26/nonphoto (ssim2-derived targets — a self-row "
                               "would be trivially perfect).")},
            "rank": rank,
            "per_pair": per_pair_all,
            "m3_coherence": None, "m3a_coherence": None,
            "peer": True,
            "peer_provenance": prov,
        }
        out = os.path.join(OUT, f"peer_{peer}.fulleval.json")
        json.dump(doc, open(out, "w"), indent=1)
        print(f"peer_{peer}: {len(rank)} corpora -> {out}")
        for c, b in sorted(rank.items()):
            print(f"   {c}: srocc {b['srocc']:.4f} (n={b['n']})")

if __name__ == "__main__":
    main()
