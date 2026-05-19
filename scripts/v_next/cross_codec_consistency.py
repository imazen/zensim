#!/usr/bin/env python3
"""Cross-codec consistency eval for PreviewV0_5Tuner (2026-05-18).

For each of K source images and a target zensim score T:
  1. For each codec in {JPEG, WebP, AVIF}, binary-search for the q value
     whose zensim score (under the eval bake) is closest to T.
  2. Decode all 3 outputs back to PNG.
  3. Compute pairwise butteraugli (max + p-norm-3) between the 3 decoded
     PNGs. The pair-wise mean should be SMALL — meaning the codecs landed
     at perceptually similar quality at the same zensim target.

Output: a TSV with per-image rows of (image_id, jpeg_q, webp_q, avif_q,
zensim_jpeg, zensim_webp, zensim_avif, pairwise_butter_max_mean,
pairwise_butter_p3_mean, ...).
"""

import argparse
import io
import json
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image

QSWEEP_DIR = Path("/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep")
OUT_DIR = Path("/mnt/v/output/zensim/exp_tuner_2026-05-18/cross_codec")

# Per-codec (encoder_callable, decoder_callable) — both use PIL.
def encode_jpeg(src_path, q, out_jpeg):
    img = Image.open(src_path).convert("RGB")
    img.save(out_jpeg, "JPEG", quality=q, subsampling=2, optimize=False, progressive=False)

def encode_webp(src_path, q, out_webp):
    img = Image.open(src_path).convert("RGB")
    img.save(out_webp, "WEBP", quality=q, method=4)

def encode_avif(src_path, q, out_avif):
    img = Image.open(src_path).convert("RGB")
    img.save(out_avif, "AVIF", quality=q, speed=6)


def decode_to_png(in_path, out_png):
    img = Image.open(in_path).convert("RGB")
    img.save(out_png, "PNG")


def zensim_with_bake(ref_png, dist_png, bake_path, bake_post_mode, zensim_score_tool):
    """Call the score-one-pair binary returning a float zensim score."""
    # Use the qsweep_eval-equivalent on a single (ref, dist) pair via a
    # one-row feature CSV + manifest TSV. Too slow; instead shell out to
    # `examples/score_pair_with_bake` if it exists, OR use zen-metrics.
    # For tractability, the simplest path is the `zensim_score_profiles`
    # binary which honors a list of profiles — but we need to score
    # against an *arbitrary bake*, not just the embedded profiles.
    #
    # Use the standalone score_pair_with_bake binary we'll write under
    # zensim-validate/src/bin/. Pass the bake bytes + ref + dist directly.
    out = subprocess.check_output(
        [
            zensim_score_tool,
            "--bake",
            bake_path,
            "--bake-post",
            bake_post_mode,
            "--ref",
            str(ref_png),
            "--dist",
            str(dist_png),
        ],
        text=True,
    )
    return float(out.strip())


def binary_search_q(target_score, encode_fn, decode_fn, src_path, codec_name, bake, mode, tool, work_dir, image_id, q_min=1, q_max=99):
    """Return the q that yields a zensim score closest to target_score."""
    best_q = None
    best_score = None
    lo, hi = q_min, q_max
    iters = 0
    cache = {}
    # Bracket: first check endpoints.
    def measure(q):
        if q in cache:
            return cache[q]
        enc_path = work_dir / f"{image_id}_{codec_name}_q{q:03d}{('.jpg' if codec_name == 'jpeg' else '.webp' if codec_name == 'webp' else '.avif')}"
        dec_path = work_dir / f"{image_id}_{codec_name}_q{q:03d}.png"
        if not dec_path.exists():
            encode_fn(src_path, q, enc_path)
            decode_fn(enc_path, dec_path)
        s = zensim_with_bake(src_path, dec_path, bake, mode, tool)
        cache[q] = (s, dec_path)
        return cache[q]
    # Simple binary search assuming monotonicity (zensim ↑ with q).
    while lo <= hi and iters < 8:
        mid = (lo + hi) // 2
        s, dec_path = measure(mid)
        if best_q is None or abs(s - target_score) < abs(best_score - target_score):
            best_q = mid
            best_score = s
            best_path = dec_path
        if s < target_score:
            lo = mid + 1
        elif s > target_score:
            hi = mid - 1
        else:
            break
        iters += 1
    return best_q, best_score, best_path


def butteraugli_pair(ref_png, dist_png, zen_metrics):
    """Return (max, p3) butteraugli for the pair."""
    out = subprocess.check_output(
        [
            zen_metrics,
            "score",
            "--metric",
            "butteraugli",
            "--reference",
            str(ref_png),
            "--distorted",
            str(dist_png),
        ],
        text=True,
    )
    # zen-metrics emits `metric=butteraugli butteraugli_max=X butteraugli_pnorm3=Y`
    bmax, bp3 = None, None
    for tok in out.split():
        if tok.startswith("butteraugli_max="):
            try:
                bmax = float(tok.split("=", 1)[1])
            except Exception:
                pass
        elif tok.startswith("butteraugli_pnorm3="):
            try:
                bp3 = float(tok.split("=", 1)[1])
            except Exception:
                pass
    return bmax, bp3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--bake", required=True, help="bake bytes to use for zensim scoring")
    ap.add_argument("--bake-post", default="clamp", help="post-processing mode for the bake")
    ap.add_argument("--n-images", type=int, default=20)
    ap.add_argument("--tool", default="/home/lilith/work/zen/zensim--exp-tuner/target/release/score_pair_with_bake")
    ap.add_argument("--zen-metrics", default="/home/lilith/work/zen/zenmetrics/target/release/zen-metrics")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    work_dir = OUT_DIR / f"bake_{Path(args.bake).stem}_t{int(args.target)}"
    work_dir.mkdir(parents=True, exist_ok=True)
    print(f"work dir: {work_dir}", file=sys.stderr)

    # Read manifest to get source paths (one row per image_id, q=50 for the source link).
    src_by_id = {}
    with open(QSWEEP_DIR / "qsweep_manifest.tsv") as f:
        f.readline()  # header
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue
            src_by_id.setdefault(cols[2], cols[0])

    images = sorted(src_by_id.items())[: args.n_images]
    print(f"selected {len(images)} images", file=sys.stderr)

    rows = []
    for image_id, src_path in images:
        print(f"\nimage: {image_id}", file=sys.stderr)
        codecs = [
            ("jpeg", encode_jpeg),
            ("webp", encode_webp),
            ("avif", encode_avif),
        ]
        results = {}
        for cn, fn in codecs:
            q, s, dec_path = binary_search_q(
                args.target, fn, decode_to_png, src_path, cn,
                args.bake, args.bake_post, args.tool, work_dir, image_id,
            )
            print(f"  {cn} q={q} zensim={s:.2f}", file=sys.stderr)
            results[cn] = (q, s, dec_path)

        # Pairwise butteraugli between the 3 decoded outputs.
        names = ["jpeg", "webp", "avif"]
        pair_b_max = []
        pair_b_p3 = []
        for i in range(3):
            for j in range(i + 1, 3):
                _, _, pi = results[names[i]]
                _, _, pj = results[names[j]]
                bmax, bp3 = butteraugli_pair(pi, pj, args.zen_metrics)
                print(f"  butter[{names[i]} vs {names[j]}] max={bmax} p3={bp3}", file=sys.stderr)
                if bmax is not None:
                    pair_b_max.append(bmax)
                if bp3 is not None:
                    pair_b_p3.append(bp3)
        mean_max = sum(pair_b_max) / len(pair_b_max) if pair_b_max else float("nan")
        mean_p3 = sum(pair_b_p3) / len(pair_b_p3) if pair_b_p3 else float("nan")
        rows.append({
            "image_id": image_id,
            "target": args.target,
            "jpeg_q": results["jpeg"][0], "jpeg_zensim": results["jpeg"][1],
            "webp_q": results["webp"][0], "webp_zensim": results["webp"][1],
            "avif_q": results["avif"][0], "avif_zensim": results["avif"][1],
            "pairwise_butter_max_mean": mean_max,
            "pairwise_butter_p3_mean": mean_p3,
        })

    with open(args.out, "w") as f:
        if not rows:
            print("no rows", file=sys.stderr)
            return 1
        cols = list(rows[0].keys())
        f.write("\t".join(cols) + "\n")
        for r in rows:
            f.write("\t".join(str(r[c]) for c in cols) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    # Summary.
    pmax = [r["pairwise_butter_max_mean"] for r in rows if r["pairwise_butter_max_mean"] == r["pairwise_butter_max_mean"]]
    pp3 = [r["pairwise_butter_p3_mean"] for r in rows if r["pairwise_butter_p3_mean"] == r["pairwise_butter_p3_mean"]]
    if pmax:
        print(f"mean butter_max across images: {sum(pmax)/len(pmax):.3f}", file=sys.stderr)
    if pp3:
        print(f"mean butter_p3  across images: {sum(pp3)/len(pp3):.3f}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
