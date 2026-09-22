#!/usr/bin/env python3
"""Collect fit artefacts -> side-by-side comparison JSON for the report.

Usage: collect.py --out report/compare.json report/fits/artefact_*.json
Emits per-arm rows: structure chosen, dev metrics, per-level betas,
identification flags, prior selected, detector status.
"""
import argparse, glob, json, sys
import numpy as np


def row(A):
    fin = A["final"]
    st = A["stages"]
    bm = fin["params"]["beta_mode"]
    groups = fin["params"]["groups"]
    betas = {g: [c["beta"] for c in levs] for g, levs in groups.items()}
    prof = st.get("E6_profile") or {}
    det = st.get("E5_detectors") or {}
    e4 = st.get("E4_prior") or {}
    return {
        "name": A["name"],
        "weber_eps": A.get("weber_eps"),
        "domain_weights": A.get("domain_weights"),
        "n_rows": A["n_rows"], "n_refs": A["n_refs"],
        "n_pairs": A["n_pairs"], "y_range": A["y_range"],
        "y_neg_frac": A.get("y_neg_frac"),
        "beta_mode": bm,
        "chroma_shared": fin["params"].get("chroma_shared"),
        "betas": betas,
        "lambda_prior": fin.get("lambda_prior"),
        "e4_selected": e4.get("selected_lambda"),
        "e4_sweep": [{"lambda": s.get("lambda"),
                      "dev_rank_loss": (s.get("dev") or {}).get("rank_loss"),
                      "betas": s.get("betas")}
                     for s in e4.get("sweep", [])],
        "dev": fin.get("dev"),
        "sanity_srocc": (A.get("sanity") or {}).get("srocc"),
        "untie": {k: {"accepted": (st.get(k) or {}).get("accepted"),
                      "dev_delta": (st.get(k) or {}).get("dev_delta")}
                  for k in ("U1", "U2", "U3") if st.get(k)},
        "profile": {k: {"hat": v.get("hat"),
                        "interval": v.get("interval"),
                        "edge": v.get("interval_edge"),
                        "identified": v.get("identified")}
                    for k, v in prof.items()},
        "detectors": {k: v for k, v in det.items() if k != "fit"},
        "det_fit": det.get("fit"),
        "boot": st.get("E7_bootstrap"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artefacts", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = []
    for pat in a.artefacts:
        files += glob.glob(pat)
    rows = [row(json.load(open(f))) for f in sorted(set(files))]
    rows = [r for r in rows if not r["name"].startswith("smoke")]
    out = {"arms": rows,
           "beta_table": {r["name"]: r["betas"] for r in rows}}
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"{len(rows)} arms -> {a.out}")
    for r in rows:
        b = {g: np.round(v, 3).tolist() for g, v in r["betas"].items()}
        dev = r["dev"] or {}
        print(f"  {r['name']:22s} mode={r['beta_mode']:6s} "
              f"lam={r['e4_selected']} devL={dev.get('rank_loss')} "
              f"srocc={dev.get('srocc')} betas={b}")


if __name__ == "__main__":
    main()
