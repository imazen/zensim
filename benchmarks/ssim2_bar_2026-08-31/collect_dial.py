#!/usr/bin/env python3
"""Collect the pooled + q>=85-zone dial rows out of bake_verdict markdown
reports into one TSV. Parses ONLY; every number was computed by bake_verdict."""
import re, sys, os, glob
def grab(p):
    t=open(p).read(); i=t.find("## DIAL panel")
    if i<0: return None
    sec=t[i:]
    out={"file":os.path.basename(p)}
    m=re.search(r"\| monotonicity \(1 − inversions\) \| ([\d.]+)", sec);  out["mono"]=m.group(1) if m else ""
    m=re.search(r"\| \*\*inversions\*\* \(backwards > [\d.]+pt\) \| ([\d.]+)", sec); out["inv"]=m.group(1) if m else ""
    m=re.search(r"\| ↳ strict backwards \(any > 1e-9\) \| ([\d.]+)", sec); out["strict"]=m.group(1) if m else ""
    m=re.search(r"dead-zone.*?\| ([\d.]+) \| G3", sec); out["flat"]=m.group(1) if m else ""
    m=re.search(r"\| dial p5 / p95 \| ([-\d.]+) / ([-\d.]+)", sec); out["p5"],out["p95"]=(m.group(1),m.group(2)) if m else ("","")
    for row in re.finditer(r"\| (all|codec|class) \| (\S+) \| (q<50|q50-85|q>=85) \| (\d+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \| (\d+) \| (\d+)% \| (\d+)% \|", sec):
        split,key,zone,pairs,inv,rate,worst,flat,lad,winv,ends = row.groups()
        out[f"{key}|{zone}"] = f"{pairs}/{inv}/{rate}/{lad}/{winv}%/{ends}%"
    return out
rows=[grab(p) for p in sorted(sys.argv[1:])]
rows=[r for r in rows if r]
keys=["file","mono","inv","strict","flat","p5","p95","all|q<50","all|q50-85","all|q>=85",
      "avif|q>=85","jpeg|q>=85","jxl|q>=85","webp|q>=85",
      "photo|q>=85","text_lineart|q>=85","nonphoto|q>=85"]
print("\t".join(keys))
for r in rows: print("\t".join(str(r.get(k,"")) for k in keys))
