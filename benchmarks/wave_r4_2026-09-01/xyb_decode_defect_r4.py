#!/usr/bin/env python3
"""wave-r4 A6 finding: `zenjpeg-420-xyb-e2` rows read DIRECTLY through `zen_io`
carry no usable codec-quality signal.

Discovered while measuring the decode-route drift this wave was asked to
measure. It is reported here because it affects a leg this wave did NOT build:
`ext_safesyn_full.parquet`'s 28,182 XYB rows (25.4% of that 111,068-row leg).

THE GROUND TRUTH. The ~403 GiB `q<N>.png` decode cache was deleted 2026-06-22,
but 32 files escaped: 16 `zenjpeg-420-xyb-e2` and 16 `zenjxl-e7`. They are the
generator's own decode, frozen at generation time -- the only surviving
reference for safesyn distorted pixels.

THREE ARMS, one extractor invocation, same reference, same bitstream:
  A  dist = q<N>.jpg          (v2_ab_extract reads it via zen_io -> zenjpeg)
  B  dist = q<N>.png          (decoded by the zencodec decode owner)
  C  dist = q<N>.png          (the SURVIVING ORIGINAL CACHE = ground truth)

MEASURED (this script regenerates all of it):
  * f0, XYB q20:  A 0.1122744 | B 0.0000763 | C 0.0002732
  * f0, XYB q50:  A 0.1125013 | B 0.0003553 | C 0.0003145
  * max|d| vs C:  A 0.966 | B 0.076  -- the zencodec route is 12.7x closer
  * the q-LADDER through arm A is destroyed: f0 spans only 0.1123..0.1145
    across q5..q100 and is NOT monotone (q100 = 0.1142 is WORSE than
    q10 = 0.1124). The colorspace error is ~370x the real codec distortion,
    so what arm A measures is a constant plus noise.
  * pixel-level, decoded vs surviving cache: XYB max|d| 9 / mean 0.73;
    JXL max|d| 1 / mean 0.24 (the documented jxl-oxide vs zenjxl-decoder
    rounding difference) -- i.e. the decode owner reproduces the cache, and
    it is the direct `.jpg` read that does not.

MECHANISM. `zen_io::decode_jpeg_rgb8` calls `zenjpeg::decoder::Decoder` and
takes `pixels_u8()`; nothing on that path applies the XYB->sRGB inverse an
XYB-encoded JPEG needs. The decode owner instead goes through zenjpeg's
zencodec `DecoderConfig -> DecodeJob -> Decode` path, which lands within 9 LSB
of the generator's own decode.

SCOPE. safesyn is the only corpus in the wave root with an XYB JPEG family, so
no other leg is affected. Not fixed here (zen_io is another lane's file, and
the wave's other legs depend on its current behaviour); reported with evidence
so nobody trains on those 28,182 rows believing they carry a quality signal.
The wave-r4 big leg ships a corrected variant alongside the as-briefed one --
see `stage_safesyn_big_r4.py`'s xybfix block.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DECODER = Path("/mnt/v/zen/cargo-targets/waver4-decode/release/examples/verify_bitstream_decode")
EXTRACTOR = Path("/mnt/v/zen/cargo-targets/waver4/release/examples/v2_ab_extract")
IMAGES = Path("/mnt/v/input/zensim/images")
SOURCES = Path("/mnt/v/input/zensim/sources")
WORK = Path(os.environ.get("XYB_WORK", "/mnt/v/output/zensim/waver4-run-2026-09-01/xybdefect"))
LADDER_Q = ["5", "10", "20", "30", "50", "75", "90", "100"]


def die(m: str) -> None:
    print(f"ABORT: {m}", file=sys.stderr)
    sys.exit(2)


def extract(pairs: list[tuple[str, str]], tag: str) -> tuple[list[str], list[list[str]]]:
    p = WORK / f"pairs_{tag}.tsv"
    o = WORK / f"out_{tag}.csv"
    with open(p, "w") as f:
        f.write("ref_path\tdist_path\thuman_score\n")
        for a, b in pairs:
            f.write(f"{a}\t{b}\t0.5\n")
    r = subprocess.run(
        [str(EXTRACTOR), str(p), str(o)],
        capture_output=True,
        text=True,
        env=dict(os.environ, ZENSIM_AB_MODE="foldapp2pools"),
    )
    if r.returncode != 0:
        print(r.stderr[-1500:], file=sys.stderr)
        die(f"extractor rc={r.returncode} for {tag}")
    rows = list(csv.reader(open(o)))
    return rows[0], rows[1:]


def main() -> None:
    for p in (DECODER, EXTRACTOR):
        if not p.exists():
            die(f"missing {p}")
    (WORK / "dec").mkdir(parents=True, exist_ok=True)
    (WORK / "links").mkdir(parents=True, exist_ok=True)

    # ---- find the surviving cache ---------------------------------------
    survivors = sorted(
        str(p)
        for p in IMAGES.glob("*/*/q*.png")
        if p.parent.name in ("zenjpeg-420-xyb-e2", "zenjxl-e7")
    )
    if not survivors:
        die("no surviving q<N>.png cache files found -- ground truth unavailable")

    # ---- decode their bitstreams through the owner ----------------------
    triples = []
    listing = WORK / "decode_list.tsv"
    with open(listing, "w") as lf:
        for orig in survivors:
            parts = orig.split("/")
            codec, ref = parts[-2], parts[-3]
            stem = os.path.splitext(parts[-1])[0]
            ext = "jxl" if codec == "zenjxl-e7" else "jpg"
            bs = orig[:-4] + "." + ext
            if not os.path.isfile(bs):
                continue
            uniq = f"{ref}__{codec}__{stem}"
            link = WORK / "links" / f"{uniq}.{ext}"
            if link.is_symlink():
                link.unlink()
            link.symlink_to(bs)
            lf.write(f"{link}\n")
            triples.append(
                {
                    "codec": codec,
                    "ref": ref,
                    "q": stem,
                    "bitstream": bs,
                    "orig_png": orig,
                    "dec_png": str(WORK / "dec" / f"{uniq}.png"),
                    "src": str(SOURCES / f"{ref}.png"),
                }
            )
    r = subprocess.run(
        [str(DECODER), "--decode-list", str(listing), "--out-dir", str(WORK / "dec"), "--jobs", "4"],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        die(f"decode owner rc={r.returncode}: {r.stderr[-500:]}")

    # ---- pixel-level: decoded vs surviving cache ------------------------
    px: dict[str, list[tuple[float, float]]] = {}
    for t in triples:
        a = np.asarray(Image.open(t["orig_png"]).convert("RGB")).astype(np.int32)
        b = np.asarray(Image.open(t["dec_png"]).convert("RGB")).astype(np.int32)
        if a.shape != b.shape:
            continue
        d = np.abs(a - b)
        px.setdefault(t["codec"], []).append((float(d.max()), float(d.mean())))
    pixel_stats = {
        c: {"n": len(v), "max_abs_lsb": max(x[0] for x in v), "mean_abs_lsb": float(np.mean([x[1] for x in v]))}
        for c, v in px.items()
    }

    # ---- three-arm feature comparison -----------------------------------
    arms: list[tuple[str, str]] = []
    labels: list[tuple[str, str, str, str]] = []  # arm, codec, ref, q
    for t in triples:
        if t["q"] not in ("q20", "q50") or not os.path.isfile(t["src"]):
            continue
        if t["codec"] != "zenjxl-e7":  # arm A only exists where zen_io can read it
            arms.append((t["src"], t["bitstream"]))
            labels.append(("A_jpg_direct", t["codec"], t["ref"], t["q"]))
        arms.append((t["src"], t["dec_png"]))
        labels.append(("B_zencodec_decode", t["codec"], t["ref"], t["q"]))
        arms.append((t["src"], t["orig_png"]))
        labels.append(("C_ORIGINAL_CACHE", t["codec"], t["ref"], t["q"]))
    hdr, rows = extract(arms, "3arm")
    pos = {c: i for i, c in enumerate(hdr)}
    fcols = [c for c in hdr if c.startswith("f")]

    by_key: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    f0: list[dict[str, object]] = []
    for (arm, codec, ref, q), row in zip(labels, rows):
        by_key.setdefault((codec, ref, q), {})[arm] = [float(row[pos[c]]) for c in fcols]
        f0.append({"arm": arm, "codec": codec, "ref": ref, "q": q, "f0": float(row[pos["f0"]])})

    vs_truth = []
    for (codec, ref, q), a in by_key.items():
        if "C_ORIGINAL_CACHE" not in a:
            continue
        for arm in ("A_jpg_direct", "B_zencodec_decode"):
            if arm not in a:
                continue
            d = [abs(x - y) for x, y in zip(a[arm], a["C_ORIGINAL_CACHE"])]
            vs_truth.append(
                {"codec": codec, "ref": ref, "q": q, "arm": arm,
                 "max_abs_vs_truth": max(d),
                 "n_feat_differing": sum(1 for x in d if x != 0.0)}
            )

    # ---- the q-ladder through arm A -------------------------------------
    ladder_ref = next((t["ref"] for t in triples if t["codec"] == "zenjpeg-420-xyb-e2"), None)
    ladder: list[dict[str, object]] = []
    if ladder_ref:
        src = str(SOURCES / f"{ladder_ref}.png")
        pl, ql = [], []
        for q in LADDER_Q:
            j = IMAGES / ladder_ref / "zenjpeg-420-xyb-e2" / f"q{q}.jpg"
            if j.is_file() and os.path.isfile(src):
                pl.append((src, str(j)))
                ql.append(q)
        if pl:
            h2, r2 = extract(pl, "ladder")
            p2 = {c: i for i, c in enumerate(h2)}
            ladder = [{"q": int(q), "f0_jpg_direct": float(row[p2["f0"]])} for q, row in zip(ql, r2)]

    vals = [x["f0_jpg_direct"] for x in ladder]
    res = {
        "finding": "zen_io's direct .jpg read of an XYB JPEG omits the XYB->sRGB inverse",
        "affects": {
            "leg": "/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/ext_safesyn_full.parquet",
            "codec": "zenjpeg-420-xyb-e2",
            "rows": 28182,
            "of_leg_rows": 111068,
            "pct_of_leg": round(100 * 28182 / 111068, 1),
        },
        "ground_truth": {
            "what": "q<N>.png files that survived the 2026-06-22 decode-cache deletion",
            "n_found": len(triples),
            "by_codec": {c: v["n"] for c, v in pixel_stats.items()},
        },
        "pixel_decoded_vs_cache": pixel_stats,
        "f0_three_arm": f0,
        "feature_distance_vs_truth": vs_truth,
        "q_ladder_arm_A": ladder,
        "q_ladder_span": (max(vals) - min(vals)) if vals else None,
        "q_ladder_monotone": (vals == sorted(vals) or vals == sorted(vals, reverse=True)) if vals else None,
        "verdict": (
            "arm A (direct .jpg) is a constant-plus-noise floor for XYB; "
            "arm B (zencodec decode) tracks the original cache"
        ),
    }
    (WORK / "xyb_decode_defect.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "f0_three_arm"}, indent=1))


if __name__ == "__main__":
    main()
