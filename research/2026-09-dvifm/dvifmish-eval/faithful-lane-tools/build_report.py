#!/usr/bin/env python3
"""faithful lane — build the attribution record (md + json).

Reads fits/*.json + evals/*.json under the lane output dir, writes
benchmarks/dvifm_faithful_2026-09-22.{md,json} in the repo.

Exposure roles are stated per dataset per side — the author's CID22
number is in-sample (their fit domain was the CID22 human pairs); their
TID2013/KADID10K numbers sit on sets whose JPEG/JP2K subsets were in
their fit mix. Ours: cid22a in-sample; TID2013 8.3% fit-domain
(JPEG/JP2K rows); KADID10K-nt 8.0% fit-domain.
"""
import json
import sys
from pathlib import Path

OUT = Path('/mnt/v/output/zensim/dvifm-faithful-2026-09-22')
REPO = Path('/home/lilith/work/zen/zensim')

AUTHOR = {
    # author's published luma-only numbers (the talk's table)
    "tid2013": {"srocc": 0.95185, "krocc": 0.80667},
    "kadid10k": {"srocc": 0.93746, "krocc": None},
    "cid22": {"srocc": 0.88289, "krocc": 0.69446},
    "aic4_trn": {"srocc": 0.91609, "krocc": None},
    "nncd": {"srocc": 0.88659, "krocc": None},
}
SURF_AUTHOR = {"tid2013_full": "tid2013", "kadid10k_nt": "kadid10k",
               "cid22a": "cid22", "tid2013_heldout": "tid2013",
               "kadid10k_heldout": "kadid10k"}

ARMS = ["faithful", "band_local", "edge_disc", "gate", "tied_pool",
        "fitmix", "all_ours", "ours_full"]
ARM_DESC = {
    "faithful": "talk config: Laplacian band, plain block range, "
                "smooth curve, per-band Lp (free L), human-mix fit",
    "band_local": "deviation 1 — LOCAL band G-B^2G instead of Laplacian",
    "edge_disc": "deviation 2 — 3x3-corner edge-discount contrast "
                 "instead of plain block range",
    "gate": "deviation 3 — two-state visibility gate instead of "
            "smooth curve",
    "tied_pool": "deviation 4 — Lp exponent folded into error power "
                 "(E=s^{1/P}) instead of free per-band L",
    "fitmix": "deviation 5 — fitted on cid22_dev ssim2/100 pseudo-labels "
              "instead of the human-label mix",
    "all_ours": "all four structural deviations together "
                "(local+edge+gate+tied), human-mix fit",
    "ours_full": "production configuration: all deviations + "
                 "pseudo-label fit (the verdict-lane setup)",
}

EXPOSURE = {
    "author": {
        "cid22": "IN-SAMPLE — the ~4.3-4.9k-pair fit domain was the "
                 "CID22 human pairs themselves",
        "tid2013": "mostly held-out: only the JPEG/JP2K subset "
                   "(~250/3000 rows) was in the fit mix",
        "kadid10k": "mostly held-out: only the JPEG/JP2K subset "
                    "(~650/10125 rows) was in the fit mix",
    },
    "ours": {
        "cid22a": "IN-SAMPLE for human-mix arms (fit domain); held-out "
                  "for the fitmix/ours_full arms",
        "tid2013_full": "250/3000 rows (JPEG/JP2K subset) in the "
                        "human-mix fit domain; 91.7% held-out",
        "kadid10k_nt": "650/8125 rows (JPEG/JP2K, non-terminal refs) in "
                       "the fit domain; 92.0% held-out; the 16 terminal "
                       "refs (2000 rows) stay untouched",
    },
}


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if v is not None else "—"


def main():
    fits, evals = {}, {}
    for arm in ARMS:
        fp, ep = OUT / "fits" / f"{arm}.json", OUT / "evals" / f"{arm}.json"
        if fp.exists():
            fits[arm] = json.loads(fp.read_text())
        if ep.exists():
            evals[arm] = json.loads(ep.read_text())

    rec = {
        "lane": "faithful",
        "date": "2026-09-22",
        "question": "does DVIFM in the author's own configuration "
                    "reproduce his published luma-only numbers, and "
                    "which of our deviations costs what?",
        "author_numbers": AUTHOR,
        "author_fit_domain": "CID22 human pairs + TID/KADID JPEG-JP2K "
                             "subsets + NLCD + AIC-4 (~4.3-4.9k pairs)",
        "our_fit_domain": "cid22a (2192) + tid JPEG/JP2K (250) + kadid "
                          "JPEG/JP2K non-terminal (650) = 3092 rows; "
                          "NLCD unavailable, AIC-4 is holdout (T0) — "
                          "fit budget comparable, ~0.65x theirs",
        "exposure": EXPOSURE,
        "surfaces": {
            "tid2013_full": {"rows": 3000, "role": "T2 train-only"},
            "kadid10k_nt": {"rows": 8125,
                            "role": "T1; 16 terminal refs excluded"},
            "cid22a": {"rows": 2192, "role": "fit-domain half"},
            "tid2013_heldout": {"rows": 2750,
                                "role": "fit-mix rows excluded"},
            "kadid10k_heldout": {"rows": 7475,
                                 "role": "fit-mix rows excluded"},
        },
        "label_orientations": {
            "kadid_nt.tsv": "raw DMOS (higher=worse) — inverted to "
                            "1-y at segment load via invert_label; "
                            "verified complement of verdict-lane "
                            "kadid_dev labels (SROCC -1.0, 1-dmos=mos)",
            "tid_full.tsv": "MOS (higher=better; label-vs-level "
                            "SROCC -1.0 within ref+type)",
            "cid22a.tsv": "quality-oriented (consistent with prior "
                          "in-sample fits)",
            "cid22_dev.tsv": "signed fast-ssim2/100 pseudo-labels",
        },
        "probes": {
            "tid_insample_ceiling": {"srocc": 0.7922, "krocc": 0.6226,
                                     "note": "faithful arm fitted ON "
                                             "all 3000 TID rows"},
        },
        "arms": {},
    }

    lines = []
    a = lines.append
    a("# DVIFM faithful-replication attribution — 2026-09-22")
    a("")
    a("Lane `faithful`. Luma-only standalone DVIFM, author's (talk) "
      "configuration restored, our deviations added back one at a time. "
      "All fits luma-only; evaluation on TID2013 full and KADID10K "
      "non-terminal-refs (16 terminal refs untouched). No holdout spent.")
    a("")
    a("## Verdict")
    a("")
    a("**The faithful talk configuration does not reproduce the "
      "author's published numbers** — it lands ~0.25/0.30 SROCC below "
      "them (TID 0.70 vs 0.95185; KADID-nt 0.64 vs 0.93746). And the "
      "gap is NOT explained by our five deviations: every deviation is "
      "neutral-to-POSITIVE relative to the faithful arm, and our full "
      "production configuration (ours_full: local band + edge discount "
      "+ gate + tied pooling + pseudo-label fit) is the BEST arm of "
      "the eight. The correct restatement of the earlier verdict is "
      "therefore: our replication cannot reach the author's numbers, "
      "but our deviations were never the cause — they are small "
      "improvements on a mechanism that in our hands tops out far "
      "below the claim.")
    a("")
    a("Two probes bound where the gap lives: (a) an IN-SAMPLE ceiling "
      "test — fitting the faithful model directly on all 3000 TID rows "
      "reaches only SROCC 0.792, so no parameter setting of this model "
      "family produces the author's number on this data as we "
      "implement it; (b) per-distortion-type ranking is strong "
      "(in-sample median ~0.88; JPEG/JP2K types 0.92–0.97) while the "
      "pooled score collapses — the missing piece is cross-type scale "
      "calibration, which lives in details the talk does not specify "
      "(see 'what is left').")
    a("")
    a("## Exposure roles — never compare across roles")
    a("")
    a("| dataset | author's number is | our number is |")
    a("|---|---|---|")
    a(f"| CID22 | {EXPOSURE['author']['cid22']} | "
      f"{EXPOSURE['ours']['cid22a']} |")
    a(f"| TID2013 | {EXPOSURE['author']['tid2013']} | "
      f"{EXPOSURE['ours']['tid2013_full']} |")
    a(f"| KADID10K | {EXPOSURE['author']['kadid10k']} | "
      f"{EXPOSURE['ours']['kadid10k_nt']} |")
    a("")
    a("Their fit domain was the CID22 human pairs — so their CID22 "
      "0.88289/0.69446 is in-sample; our cid22a in-sample "
      "(0.9167/0.7464 luma-only, 2192 rows, prior screen) is ABOVE it. "
      "On CID22 we are not behind; the defensible gap is TID2013 "
      "(0.95185) and KADID10K (0.93746). Their fit budget ~4.3-4.9k "
      "pairs vs ours 3092 — comparable, not the explanation.")
    a("")
    a("## Attribution table")
    a("")
    a("| arm | deviation | TID2013 SROCC | TID2013 KROCC | KADID-nt "
      "SROCC | KADID-nt KROCC | CID22-A SROCC | CID22-A KROCC | "
      "TID-ho SROCC | KAD-ho SROCC | fit rows |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    a(f"| **author** | — | **{AUTHOR['tid2013']['srocc']:.5f}** | "
      f"**{AUTHOR['tid2013']['krocc']:.5f}** | "
      f"**{AUTHOR['kadid10k']['srocc']:.5f}** | — | "
      f"**{AUTHOR['cid22']['srocc']:.5f}** | "
      f"**{AUTHOR['cid22']['krocc']:.5f}** | ~92% held-out | "
      f"~94% held-out | ~4.3-4.9k |")
    for arm in ARMS:
        if arm not in evals:
            continue
        ev = evals[arm]
        t, k, c = ev.get("tid2013_full", {}), ev.get("kadid10k_nt", {}), \
            ev.get("cid22a", {})
        th, kh = ev.get("tid2013_heldout", {}), \
            ev.get("kadid10k_heldout", {})
        nrows = fits.get(arm, {}).get("fit_rows", "?")
        a(f"| {arm} | {ARM_DESC[arm].split('—')[-1].strip() if arm != 'faithful' else '(talk config)'} | "
          f"{fmt(t.get('srocc'))} | {fmt(t.get('krocc'))} | "
          f"{fmt(k.get('srocc'))} | {fmt(k.get('krocc'))} | "
          f"{fmt(c.get('srocc'))} | {fmt(c.get('krocc'))} | "
          f"{fmt(th.get('srocc'))} | {fmt(kh.get('srocc'))} | {nrows} |")
        rec["arms"][arm] = {
            "desc": ARM_DESC[arm],
            "fit_rows": nrows,
            "eval": ev,
            "delta_vs_author": {
                surf: (ev.get(surf, {}).get("srocc") or 0)
                      - AUTHOR[v]["srocc"]
                for surf, v in SURF_AUTHOR.items()
                if ev.get(surf, {}).get("srocc") is not None},
            "fit": fits.get(arm, {}).get("fit"),
        }
    a("")
    fth = evals.get("faithful", {})
    if fth:
        a("### Per-deviation cost (Δ SROCC vs `faithful`)")
        a("")
        a("| arm | Δ TID2013 | Δ KADID-nt | Δ CID22-A |")
        a("|---|---|---|---|")
        for arm in ARMS[1:]:
            if arm not in evals:
                continue
            ev = evals[arm]
            d = [fmt((ev.get(s, {}).get("srocc") or 0)
                     - (fth.get(s, {}).get("srocc") or 0), 4)
                 for s in ("tid2013_full", "kadid10k_nt", "cid22a")]
            a(f"| {arm} | {d[0]} | {d[1]} | {d[2]} |")
        a("")
    a("## Probes")
    a("")
    a("- **In-sample ceiling**: faithful model fitted ON tid_full "
      "(3000 rows) reaches SROCC 0.7922 / KROCC 0.6226 on the same "
      "rows — `fits/probe_tid_insample.json`. The feature+pooling "
      "form itself does not carry enough cross-type-calibrated signal "
      "to reach 0.95 even with zero generalisation gap.")
    a("- **Per-type in-sample ranking (TID2013)**: median ~0.88 across "
      "the 24 types; types 10/11 (JPEG/JP2K) 0.921/0.944; worst types "
      "15/17/18 (0.46–0.58) drag the pooled figure. The block-"
      "visibility mechanism orders distortions within a type well; "
      "it does not place them on a common scale.")
    a("")
    a("## What is left (unattributed residual vs the author)")
    a("")
    a("Candidates our five deviations do not cover: (a) the expand "
      "step E in G_l − E(G_{l+1}) is explicitly unspecified in the "
      "talk — we use zero-insert + [1 2 1] ×4 (standard Burt–Adelson); "
      "(b) Y′CbCr conversion/range details; (c) their eval protocol "
      "may report numbers including fit-domain rows or a different "
      "pooled/per-ref/per-type aggregation; (d) label preprocessing; "
      "(e) a fundamentally different block statistic than our "
      "max|δ|^P / φ_g-range records.")
    a("")
    if "faithful" in fits:
        a("## Fitted constants — faithful arm (per level)")
        a("")
        a("| level | g | P | c0 | β | ς | L | head w |")
        a("|---|---|---|---|---|---|---|---|")
        fcells = fits["faithful"]["cells"]
        fw = fits["faithful"]["wl"]
        import math as _m
        sm = [_m.exp(x) for x in fw]
        sm = [x / sum(sm) for x in sm]
        for l, c in enumerate(fcells):
            a(f"| {l} | {c['g']:.3f} | {c['P']:.3f} | {c['c0']:.4g} | "
              f"{c['beta']:.3f} | {c['sharp']:.3f} | {c['L']:.3f} | "
              f"{sm[l]:.3f} |")
        a("")
    a("## Provenance")
    a("")
    a("- extraction: `extract_features_372col --full-986 --dvifm-spec` "
      "luma-only, cap 1024 blocks/row over 5 levels, f16; band mode is "
      "the only spec field affecting records")
    a("- bands: `lap` = Laplacian G_l - E(G_{l+1}) (talk); `local` = "
      "G - B^2 G (our drift)")
    a("- fitter: `tools/fit_faithful.py` — Amendment-1 protocol "
      "(refit-map MSE objective, c0×β grid init, Adam, ≤3 sweeps), "
      "same machinery family as fit_standalone.py")
    a("- visibility (curve arms): v(c)=exp(-softplus(β·ς·(ln c − ln "
      "c0))/ς); gate arm: v = [min(cs,cd) ≤ κ]; model: s_l = "
      "mean_b(max(v_s,v_d)·m^P), E_l = s_l^L (free) or s_l^{1/P} "
      "(tied), E = softmax-mix")
    a("- fit domains per arm in `fits/<arm>.json`; eval surfaces in "
      "`evals/<arm>.json`")
    a("")
    md = "\n".join(lines) + "\n"
    (REPO / "benchmarks/dvifm_faithful_2026-09-22.md").write_text(md)
    (REPO / "benchmarks/dvifm_faithful_2026-09-22.json").write_text(
        json.dumps(rec, indent=1) + "\n")
    print("wrote benchmarks/dvifm_faithful_2026-09-22.{md,json}")


if __name__ == "__main__":
    main()
