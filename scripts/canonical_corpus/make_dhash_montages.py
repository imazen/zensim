#!/usr/bin/env python3
"""Labeled side-by-side montages for the imazen-26 dHash audit eye pass.

v2 (2026-08-27): rebuilt for the CANONICAL-estate sweeps after the wrong-copy
correction (the v1 sweeps indexed /mnt/v/imazen-26 — the quarantined
inspiration dir, now imazen-26-inspo; see the audit md CORRECTION section).
Refs are the repo-manifest members of /mnt/v/output/imazen-26-png-v3, so every
estate label carries the canonical 4-digit id natively (user requirement).

Inputs (produced by check_holdout_overlap + the analysis step):
  canon_vs_train_synth.tsv   — canonical refs vs synthetic-v2 metric sources
  canon_vs_train_picker.tsv  — canonical refs vs picker-ladder renditions
Montage sets:
  synth d<=2 pairs (POOL vs CANON)  — the generator-sharing channel
  picker CROSS-ID d<=2 pairs (CANON vs CANON) — split-piercing duplicate candidates
Near-blank sides are ANNOTATED, never refused (flat-content dHash FP class).
"""
import argparse, collections, os
from PIL import Image, ImageDraw

def load_checked(path, side):
    im = Image.open(path).convert("RGB")
    g = im.resize((16, 16)).convert("L")
    px = list(g.getdata())
    near_blank = (max(px) - min(px)) < 8
    return im, near_blank

def fit(im, h):
    w = max(1, round(im.width * h / im.height))
    return im.resize((w, h))

def canon_index(refs_dir):
    idx = {}
    for f in os.listdir(refs_dir):
        tok = f.split("_")[0]
        if tok.isdigit() and len(tok) == 4:
            idx[f] = os.path.join(refs_dir, f)
    return idx

def montage(out_dir, name, lp, ltag, rp, rtag, foot, manifest, row):
    Lim, Lb = load_checked(lp, "left")
    Rim, Rb = load_checked(rp, "right")
    L, R = fit(Lim, 360), fit(Rim, 360)
    band = 34
    canvas = Image.new("RGB", (L.width + R.width + 12, 360 + band * 2), (24, 26, 30))
    canvas.paste(L, (0, band)); canvas.paste(R, (L.width + 12, band))
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 8), f"{ltag}  {os.path.basename(lp)}", fill=(220, 220, 210))
    dr.text((L.width + 16, 8), f"{rtag}  {os.path.basename(rp)}", fill=(220, 220, 210))
    note = "".join(["   LEFT NEAR-BLANK (flat-content dHash class)" if Lb else "",
                    "   RIGHT NEAR-BLANK (flat-content dHash class)" if Rb else ""])
    dr.text((4, 360 + band + 8), foot + note,
            fill=(230, 170, 120) if note else (150, 200, 180))
    canvas.save(os.path.join(out_dir, name))
    manifest.write(row + "\n")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit-dir", default="/mnt/v/output/zensim/imazen26-dhash-2026-08-27")
    ap.add_argument("--refs-dir", default="/mnt/v/output/zensim/imazen26-dhash-2026-08-27/canon_refs")
    ap.add_argument("--out", default="/mnt/v/output/zensim/imazen26-dhash-2026-08-27/montages_v3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    idx = canon_index(a.refs_dir)
    manifest = open(os.path.join(a.out, "manifest.tsv"), "w")
    manifest.write("montage\tset\td\tleft\tright\n")
    made = 0

    # Set 1: synth-pool sharing, d<=2 (one montage per (source-stem, ref) pair)
    pairs = collections.OrderedDict()
    for line in open(os.path.join(a.audit_dir, "canon_vs_train_synth.tsv")):
        f = line.rstrip("\n").split("\t")
        if len(f) < 5 or not f[4].isdigit() or int(f[4]) > 2:
            continue
        key = (os.path.basename(f[0]).split("_")[0], f[2])
        if key not in pairs or int(f[4]) < pairs[key][2]:
            pairs[key] = (f[0], f[2], int(f[4]))
    for lp, rf, d in pairs.values():
        rp = idx.get(rf)
        if rp is None:
            print(f"SKIP (ref not in canon_refs): {rf}"); continue
        cid = rf.split("_")[0]
        name = f"synthshare_id{cid}_d{d}_{made:03d}.png"
        montage(a.out, name, lp, "SYNTH-POOL", rp, f"CANON id {cid}",
                f"synthetic-v2 source vs canonical  hamming d={d}",
                manifest, f"{name}\tsynth-share\t{d}\t{lp}\t{rf}")
        made += 1

    # Set 2: canonical-internal cross-id duplicates, d<=2 (split-piercing candidates)
    seen = set()
    for line in open(os.path.join(a.audit_dir, "canon_vs_train_picker.tsv")):
        f = line.rstrip("\n").split("\t")
        if len(f) < 5 or not f[4].isdigit() or int(f[4]) > 2:
            continue
        aid = os.path.basename(f[0]).split(".")[0].replace("o_", "")
        bid = f[2].split("_")[0]
        if aid == bid or (min(aid, bid), max(aid, bid)) in seen:
            continue
        seen.add((min(aid, bid), max(aid, bid)))
        la = [p for n, p in idx.items() if n.startswith(aid + "_")]
        if not la or f[2] not in idx:
            continue
        d = int(f[4])
        name = f"crossid_{aid}x{bid}_d{d}_{made:03d}.png"
        montage(a.out, name, la[0], f"CANON id {aid}", idx[f[2]], f"CANON id {bid}",
                f"cross-id duplicate candidate  hamming d={d}",
                manifest, f"{name}\tcross-id\t{d}\t{la[0]}\t{f[2]}")
        made += 1
    manifest.close()
    print(f"wrote {made} labeled montages -> {a.out} (+ manifest.tsv)")

if __name__ == "__main__":
    main()
