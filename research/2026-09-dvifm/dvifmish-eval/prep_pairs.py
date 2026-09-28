#!/usr/bin/env python3
"""Build the dvifmish evaluation pair lists (and the decode lists they need).

Every list is a TSV with `ref_path, dist_path, human_score, source, codec,
orig_dist_path` — `source` is the reference picture and `codec` the
distortion type / codec family, for the per-source and per-codec
aggregations; `orig_dist_path` is the file on disk before any decode step.

Encoded inputs (JPEG, WebP, AVIF, JXL) are scored from PNGs decoded once by
zensim's own decode owner (`verify_bitstream_decode --decode-list`, the same
zencodec decoders the block-record caches and the peer tables were built
with), so every metric sees identical pixels. `decode` mode writes the lists
of files to decode (as uniquely named symlinks); `lists` mode writes the pair
TSVs pointing at the decoded PNGs.

Labels and orientation (higher = better quality everywhere except AIC-4):
  tid2013      MOS/9 from mos_with_names.txt
  kadid10k     (dmos − 1)/4 from dmos.csv ("dmos" is quality-oriented)
  cid22        MCOS/100 (cid22val_pairs_ab.tsv)
  nncd         MOS from MOS_scores_sorted.csv
  konfig_val   1 − q_jnd/3.2 (verdict-lane list)
  codec_dev / safesyn_*   signed SSIMULACRA2 (/100 for codec_dev) — a
               metric teacher's opinion, never clipped
Held back: KADID-10k terminal references (last digit 7 or 9) are never
listed; the AIC-4 sample is built in a separate, later step after every
model is frozen.

Usage: prep_pairs.py decode|lists <out_pairs_dir> <decoded_root>
       prep_pairs.py aic4 <out_pairs_dir>   (after every model is frozen)
"""
import csv
import os
import random
import sys
from pathlib import Path

TID = Path("/mnt/v/dataset/tid2013")
KADID = Path("/mnt/v/dataset/kadid10k")
CID22 = Path("/mnt/v/dataset/cid22/CID22_validation_set")
NNCD = Path("/var/tmp/dvifmish/datasets/nncd-iqa")
NNCD_MOS = Path("/mnt/v/datasets/nncd-iqa/MOS_scores_sorted.csv")
VERDICT = Path("/mnt/v/output/zensim/dvifm-verdict-2026-09-20/pairs")
LOSS = Path("/mnt/v/output/zensim/dvifm-loss-2026-09-20/pairs")

CID22_A = {
    "1189261.png", "1531677.png", "159550.png", "1624487.png", "162520.png",
    "164595.png", "2079234.png", "21169144185_3f7977cb5a_o.png", "225228.png",
    "2389166.png", "2936831.png", "3316926.png", "3653963.png", "373965.png",
    "3762075.png", "4215100.png", "6078297.png", "6292444.png", "70497.png",
    "7062219.png", "844297.png", "pexels-photo-2686358.png",
    "pexels-photo-2802032.png", "pexels-photo-4210863.png",
    "ularapi_Semarang_City_Logo.png",
}
CID22_DUP_IN_B = "3316926_opo25u.png"   # same picture as 844297.png (A)

SAFESYN_FIT_ROWS = 4000      # per seed
SAFESYN_SEEDS = (1, 2, 3)
SAFESYN_DEV_ROWS = 3000


def rows_tid():
    out = []
    for line in open(TID / "mos_with_names.txt"):
        mos, name = line.split()
        stem = Path(name).stem                      # e.g. I01_01_1 / i01_01_2
        ref = f"I{stem[1:3]}.png" if stem[1:3] != "25" else "i25.png"
        refp = TID / "reference_images_png" / ref
        if not refp.exists():
            refp = TID / "reference_images_png" / ref.lower()
        dist = TID / "distorted_images_png" / f"{stem}.png"
        assert refp.exists() and dist.exists(), (refp, dist)
        out.append({"ref_path": str(refp), "dist_path": str(dist),
                    "human_score": float(mos) / 9.0, "source": refp.stem.upper(),
                    "codec": f"tid{int(stem[4:6]):02d}", "orig_dist_path": str(dist)})
    assert len(out) == 3000
    return out


def rows_kadid():
    out = []
    for r in csv.DictReader(open(KADID / "dmos.csv")):
        ref = r["ref_img"]
        refnum = int(ref[1:3])
        if refnum % 10 in (7, 9):
            continue                                 # terminal references: sealed
        dist = KADID / "images" / r["dist_img"]
        refp = KADID / "images" / ref
        out.append({"ref_path": str(refp), "dist_path": str(dist),
                    "human_score": (float(r["dmos"]) - 1.0) / 4.0,
                    "source": Path(ref).stem, "codec": f"kadid{int(r['dist_img'][4:6]):02d}",
                    "orig_dist_path": str(dist)})
    assert len(out) == 8125, len(out)
    return out


def cid22_decoded(root, dist):
    p = Path(dist)
    if p.suffix.lower() != ".jpg":
        return dist
    ref = p.parent.parent.name
    return str(Path(root) / "cid22" / f"{ref}__{p.parent.name}__{p.stem}.png")


def rows_cid22(root):
    out = []
    for r in csv.DictReader(open(CID22 / "cid22val_pairs_ab.tsv"), delimiter="\t"):
        dist = r["dist_path"]
        codec = Path(dist).parent.name
        out.append({"ref_path": r["ref_path"], "dist_path": cid22_decoded(root, dist),
                    "human_score": float(r["human_score"]),
                    "source": Path(r["ref_path"]).name, "codec": codec,
                    "orig_dist_path": dist})
    assert len(out) == 4292, len(out)
    return out


def rows_nncd():
    out = []
    for r in csv.DictReader(open(NNCD_MOS)):
        rel = r["distorted"]
        folder, name = rel.split("/")
        n = int(name.split("_")[2])
        refp = NNCD / "original_images_kodak" / f"image{n}.png"
        dist = NNCD / rel
        assert refp.exists() and dist.exists(), (refp, dist)
        codec = {"rec_im_ycbcr": "FCNN-LS"}.get(folder, folder)
        out.append({"ref_path": str(refp), "dist_path": str(dist),
                    "human_score": float(r["mos"]), "source": refp.stem,
                    "codec": codec, "orig_dist_path": str(dist)})
    assert len(out) == 320
    return out


def encoded_key(root, leg, i, dist):
    ext = Path(dist).suffix.lower()
    if ext == ".png":
        return dist, None
    link = Path(root) / "links" / leg / f"{leg}_{i:06d}{ext}"
    return str(Path(root) / leg / f"{leg}_{i:06d}.png"), (link, dist)


def rows_verdict_leg(root, leg, teacher):
    out, dec = [], []
    for i, r in enumerate(csv.DictReader(open(VERDICT / f"{leg}.tsv"), delimiter="\t")):
        d, job = encoded_key(root, leg, i, r["dist_path"])
        if job:
            dec.append(job)
        if leg == "konfig_val":
            codec = Path(r["dist_path"]).parent.name
            source = r["source"]
        else:
            codec = Path(r["dist_path"]).suffix.lstrip(".")
            source = Path(r["ref_path"]).name
        out.append({"ref_path": r["ref_path"], "dist_path": d,
                    "human_score": float(r["human_score"]), "source": source,
                    "codec": codec, "orig_dist_path": r["dist_path"]})
    return out, dec


def rows_safesyn(root, which, n, seed):
    src = LOSS / ("safesyn_fit.tsv" if which == "fit" else "safesyn_development.tsv")
    allrows = list(csv.DictReader(open(src), delimiter="\t"))
    rng = random.Random(20260922 * 100 + seed)
    idx = sorted(rng.sample(range(len(allrows)), n))
    leg = f"safesyn_{which}"
    out, dec = [], []
    for i in idx:
        r = allrows[i]
        d, job = encoded_key(root, leg, i, r["dist_path"])
        if job:
            dec.append(job)
        fam = Path(r["dist_path"]).parent.name
        out.append({"ref_path": r["ref_path"], "dist_path": d,
                    "human_score": float(r["human_score"]),
                    "source": Path(r["ref_path"]).name, "codec": fam.split("-")[0],
                    "orig_dist_path": r["dist_path"], "row_id": i})
    return out, dec


AIC4 = Path("/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset")
AIC4_JND = Path("/mnt/v/repos/iqa-tools/jpeg-aic__JPEG-AIC-4-datasets/JPEG_AIC_reconstructed_jnd_scores.csv")


def rows_aic4(kind):
    """The AIC-4 public sample: `ptc` = the 620x800 crops the subjects saw,
    `full` = the full-resolution encodes with the same labels. Label = the
    reconstructed JND `distortion` (DISTORTION-oriented: rises with
    distortion)."""
    out = []
    for r in csv.DictReader(open(AIC4_JND)):
        num = f"{int(r['img_num']):05d}"
        ref, dist = r["img_source"], r["img_distorted"]
        if kind == "full":
            ref, dist = ref.removeprefix("PTC_"), dist.removeprefix("PTC_")
            folder = AIC4 / "full_resolution_images" / num
        else:
            folder = AIC4 / "PTC_images" / num
        refp, distp = folder / ref, folder / dist
        assert refp.exists() and distp.exists(), (refp, distp)
        codec = Path(dist).stem.removeprefix("PTC_").split("_")[1]   # PTC_00002_AVIF_01 -> AVIF
        out.append({"ref_path": str(refp), "dist_path": str(distp),
                    "human_score": float(r["distortion"]), "source": num,
                    "codec": codec, "orig_dist_path": str(distp)})
    assert len(out) == 300
    return out


def write(path, rows):
    cols = ["ref_path", "dist_path", "human_score", "source", "codec", "orig_dist_path"]
    if rows and "row_id" in rows[0]:
        cols.append("row_id")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    mode, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    if mode == "aic4":
        for kind in ("ptc", "full"):
            write(outdir / f"aic4_{kind}.tsv", rows_aic4(kind))
            print(f"aic4_{kind}.tsv 300 rows")
        return
    root = Path(sys.argv[3])
    legs = {}
    decode_jobs = {}
    # CID22 JPEGs
    cid = rows_cid22(root)
    decode_jobs["cid22"] = [(Path(r["dist_path"]).with_suffix(".jpg").name, r["orig_dist_path"])
                            for r in cid if r["dist_path"] != r["orig_dist_path"]]
    for leg in ("konfig_val", "codec_dev"):
        legs[leg], decode_jobs[leg] = rows_verdict_leg(root, leg, teacher=(leg == "codec_dev"))
    for seed in SAFESYN_SEEDS:
        legs[f"safesyn_fit_s{seed}"], decode_jobs[f"safesyn_fit_s{seed}"] = rows_safesyn(
            root, "fit", SAFESYN_FIT_ROWS, seed)
    legs["safesyn_dev"], decode_jobs["safesyn_dev"] = rows_safesyn(root, "dev", SAFESYN_DEV_ROWS, 0)
    if mode == "decode":
        for leg, jobs in decode_jobs.items():
            group = "cid22" if leg == "cid22" else ("safesyn_fit" if leg.startswith("safesyn_fit")
                                                     else leg)
            ldir = root / "links" / group
            ldir.mkdir(parents=True, exist_ok=True)
            lst = root / f"decode_{leg}.tsv"
            with open(lst, "w") as f:
                f.write("dist_path\n")
                for link, target in jobs:
                    link = ldir / Path(link).name
                    if not link.exists():
                        os.symlink(target, link)
                    f.write(f"{link}\n")
            print(f"{leg}: {len(jobs)} files to decode -> {lst} (out dir {root / group})")
        return
    tid, kad, nn = rows_tid(), rows_kadid(), rows_nncd()
    write(outdir / "tid2013_full.tsv", tid)
    write(outdir / "tid2013_codec.tsv", [r for r in tid if r["codec"] in ("tid10", "tid11")])
    write(outdir / "kadid10k_nt.tsv", kad)
    write(outdir / "kadid10k_nt_codec.tsv", [r for r in kad if r["codec"] in ("kadid09", "kadid10")])
    write(outdir / "cid22_49.tsv", cid)
    write(outdir / "cid22a.tsv", [r for r in cid if r["source"] in CID22_A])
    write(outdir / "cid22b23.tsv", [r for r in cid if r["source"] not in CID22_A
                                    and r["source"] != CID22_DUP_IN_B])
    write(outdir / "nncd.tsv", nn)
    for leg, rows in legs.items():
        write(outdir / f"{leg}.tsv", rows)
    for p in sorted(outdir.glob("*.tsv")):
        n = sum(1 for _ in open(p)) - 1
        missing = sum(1 for r in csv.DictReader(open(p), delimiter="\t")
                      if not Path(r["dist_path"]).exists() or not Path(r["ref_path"]).exists())
        print(f"{p.name:28} {n:6d} rows  missing files {missing}")


if __name__ == "__main__":
    main()
