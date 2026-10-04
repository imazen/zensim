"""CHROMAQ exchange rate: for each chroma cell, the luma-ladder multiplier that the same metric scores equally (per image)."""
import csv, math, statistics as st, sys, collections, json
rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
ms = sys.argv[2].split(",")
by = collections.defaultdict(list)
for d in rows:
    by[d["img"]].append(d)
LO, HI = -3.0, 4.0   # base mapped to log2 k = -3 (k=1/8); ladder top k=16 = 4
def k_eq(img_rows, m, s):
    lad = sorted([(math.log2(float(d["lvl"])) if d["fam"] == "luma" else LO, float(d[m]))
                  for d in img_rows if d["fam"] == "luma" or d["fam"] == "base"])
    if s >= lad[0][1]:
        return LO
    for (x0, y0), (x1, y1) in zip(lad, lad[1:]):
        if (y0 - s) * (y1 - s) <= 0 and y0 != y1:
            return x0 + (x1 - x0) * (y0 - s) / (y0 - y1)
    return HI + 1.0   # worse than the coarsest luma step
out = {}
for fam in ("chroma", "chroma_lf", "cb_only", "cr_only", "s420"):
    lvls = sorted({float(d["lvl"]) for d in rows if d["fam"] == fam})
    for lvl in lvls:
        rec = {}
        for m in ms:
            ks = [k_eq(v, m, float(d[m])) for v in by.values() for d in v if d["fam"] == fam and float(d["lvl"]) == lvl]
            rec[m] = st.median(ks)
        out[f"{fam}:{lvl:g}"] = rec
print("median log2 of the equivalent luma multiplier (−3 = no worse than base, 5 = worse than luma×16)")
print(f"{'cell':14s} " + " ".join(f"{m:>9s}" for m in ms))
for k, rec in out.items():
    print(f"{k:14s} " + " ".join(f"{rec[m]:9.2f}" for m in ms))
json.dump(out, open(sys.argv[3], "w"), indent=1)
