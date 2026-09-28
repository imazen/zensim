#!/usr/bin/env python3
"""cvvdp-safesyn Part-2 descriptive comparison (brief: "rank agreement
between CVVDP at each display, the fresh fast-ssim2 labels and the original
oracle labels; break down by codec family and quality band; report where
the teachers disagree most, with examples by row key").

Joins the harvested sidecar parquet (row_id -> cvvdp_jod_<display>, ssim2_fresh)
to the SafeSyn cache parquet (row_id -> human_score [fresh ssim2 label],
original_oracle) and the pairs TSV (row_id -> dist_path -> codec family).

  safesyn_compare.py --sidecar /var/tmp/cvvdp-safesyn/safesyn_cvvdp_sidecar.parquet \
      --out-md /home/lilith/work/zen/zensim--cvvdp-safesyn/benchmarks/rev4_safesyn_cvvdp_descriptive_2026-09-23.md

Descriptive only — no fitting (brief exclusion).
"""
import argparse, csv, json, math, sys
from collections import defaultdict

CACHE = "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train944.parquet"
PAIRS = "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train-pairs.tsv"

CVVDP_COLS = ["cvvdp_jod_standard_4k", "cvvdp_jod_standard_fhd", "cvvdp_jod_sdr_fhd_24"]
# quality bands on original_oracle (ssim2-scale, negative; higher = better).
# Band edges mirror the admitted q ladder's natural breaks; descriptive only.
BANDS = [("oracle>=-10", -1e9, -10.0), ("-10>oracle>=-25", -25.0, -10.0),
         ("-25>oracle>=-50", -50.0, -25.0), ("oracle<-50", -1e9, -50.0)]
# note: (-1e9,-10) and (-1e9,-50) — fix below via explicit range checks


def spearman(x, y):
    """Spearman = Pearson on midranks (exact under ties; matches scipy)."""
    rx, ry = ranks(x), ranks(y)
    n = len(rx)
    mx, my = sum(rx) / n, sum(ry) / n
    dx = [v - mx for v in rx]; dy = [v - my for v in ry]
    num = sum(a * b for a, b in zip(dx, dy))
    den = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return num / den if den else float("nan")


def ranks(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        m = (i + j) / 2.0
        for k in range(i, j + 1):
            r[order[k]] = m
        i = j + 1
    return r


def band_of(o):
    if o >= -10.0: return "oracle>=-10"
    if o >= -25.0: return "-10>oracle>=-25"
    if o >= -50.0: return "-25>oracle>=-50"
    return "oracle<-50"


def fam_of(dist):
    # /mnt/v/input/zensim/images/<ref>/<family>/<file>
    p = dist.split("/")
    return p[-2] if len(p) >= 2 else "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sidecar", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    sc = pq.read_table(a.sidecar).to_pydict()
    ca = pq.read_table(CACHE, columns=["row_id", "human_score", "original_oracle"]).to_pydict()
    fam = {}
    with open(PAIRS) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            fam[int(r["row_id"])] = fam_of(r["dist_path"])

    idx = {rid: i for i, rid in enumerate(ca["row_id"])}
    n = len(sc["row_id"])
    rows = []
    for i in range(n):
        rid = sc["row_id"][i]
        j = idx.get(rid)
        if j is None:
            continue
        rows.append(dict(
            row_id=rid, family=fam.get(rid, "?"),
            oracle=ca["original_oracle"][j],
            ssim2_label=ca["human_score"][j],
            ssim2_fresh=sc["ssim2_fresh"][i],
            **{c: sc[c][i] for c in CVVDP_COLS}))
    print(f"joined {len(rows)} rows", file=sys.stderr)

    # ssim2 sanity: fresh == stored label (same-buffer control)
    s2diff = max(abs(r["ssim2_fresh"] - r["ssim2_label"]) for r in rows) if rows else float("nan")
    print(f"max |ssim2_fresh - stored label| = {s2diff}", file=sys.stderr)

    targets = {"oracle": [r["oracle"] for r in rows],
               "ssim2_label": [r["ssim2_label"] for r in rows]}
    rep = {"n": len(rows), "max_ssim2_drift": s2diff, "overall": {}, "by_family": {}, "by_band": {},
           "disagreement": {}}
    for tname, tv in targets.items():
        for c in CVVDP_COLS:
            rep["overall"].setdefault(tname, {})[c] = spearman(tv, [r[c] for r in rows])
        rep["overall"][tname]["ssim2_fresh"] = spearman(tv, [r["ssim2_fresh"] for r in rows])

    for group_key, groups in (("by_family", sorted({r["family"] for r in rows})),
                              ("by_band", [b[0] for b in BANDS])):
        for g in groups:
            sel = [r for r in rows if (r["family"] == g if group_key == "by_family"
                                       else band_of(r["oracle"]) == g)]
            if len(sel) < 30:
                continue
            e = {}
            for c in CVVDP_COLS + ["ssim2_fresh"]:
                e[c] = spearman([r["oracle"] for r in sel], [r[c] for r in sel])
            rep[group_key][g] = {"n": len(sel), **e}

    # disagreement: |cvvdp@4k - sdr_fhd_24| largest rows (rank deltas)
    diffs = sorted(rows, key=lambda r: abs(r["cvvdp_jod_standard_4k"] - r["cvvdp_jod_sdr_fhd_24"]),
                   reverse=True)[:20]
    rep["disagreement"]["top20_4k_vs_sdr_fhd_24"] = [
        {"row_id": r["row_id"], "family": r["family"], "oracle": r["oracle"],
         "jod_4k": r["cvvdp_jod_standard_4k"], "jod_fhd24": r["cvvdp_jod_sdr_fhd_24"],
         "delta": r["cvvdp_jod_sdr_fhd_24"] - r["cvvdp_jod_standard_4k"]}
        for r in diffs]

    # markdown
    L = []
    L.append("# SafeSyn × CVVDP displays — descriptive comparison (Part 2)\n")
    L.append(f"Rows: {len(rows)} | max |ssim2_fresh − stored label| = {s2diff:.3g}\n")
    L.append("\n## Overall SROCC vs stored targets\n")
    L.append("| metric | vs original_oracle | vs ssim2 label |")
    L.append("|---|---|---|")
    for c in CVVDP_COLS + ["ssim2_fresh"]:
        L.append(f"| {c} | {rep['overall']['oracle'][c]:.4f} | {rep['overall']['ssim2_label'][c]:.4f} |")
    L.append("\n## By codec family (SROCC vs original_oracle)\n")
    L.append("| family | n | " + " | ".join(CVVDP_COLS + ["ssim2_fresh"]) + " |")
    L.append("|---|---|" + "---|" * 4)
    for g, e in rep["by_family"].items():
        L.append(f"| {g} | {e['n']} | " + " | ".join(f"{e[c]:.4f}" for c in CVVDP_COLS + ["ssim2_fresh"]) + " |")
    L.append("\n## By oracle quality band (SROCC vs original_oracle)\n")
    L.append("| band | n | " + " | ".join(CVVDP_COLS + ["ssim2_fresh"]) + " |")
    L.append("|---|---|" + "---|" * 4)
    for g, e in rep["by_band"].items():
        L.append(f"| {g} | {e['n']} | " + " | ".join(f"{e[c]:.4f}" for c in CVVDP_COLS + ["ssim2_fresh"]) + " |")
    L.append("\n## Largest 4k-vs-sdr_fhd_24 disagreements (top 20 by |Δ JOD|)\n")
    L.append("| row_id | family | oracle | jod_4k | jod_fhd24 | Δ |")
    L.append("|---|---|---|---|---|---|")
    for d in rep["disagreement"]["top20_4k_vs_sdr_fhd_24"]:
        L.append(f"| {d['row_id']} | {d['family']} | {d['oracle']:.2f} | {d['jod_4k']:.3f} | {d['jod_fhd24']:.3f} | {d['delta']:+.3f} |")
    open(a.out_md, "w").write("\n".join(L) + "\n")
    if a.out_json:
        json.dump(rep, open(a.out_json, "w"), indent=1)
    print(f"wrote {a.out_md}", file=sys.stderr)


if __name__ == "__main__":
    main()
