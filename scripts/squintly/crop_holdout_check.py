#!/usr/bin/env python3
"""Reverse-direction holdout check: is a squintly candidate source image a
CROP of a full-resolution AIC-3 / AIC-4 reference image?

The project's existing dHash-64 overlap detector (`check_holdout_overlap`,
`zensim_validate::content_clusters::dhash_64`) compares two WHOLE images by
resizing each to a 9x8 grid. Its own doc comment says it plainly: "Robust to
resampling and mild recompression; blind to crops." A crop's 9x8 downsample
reflects a different sub-region composition than the full source's 9x8
downsample, so a small squintly stimulus that was cut out of a large AIC
source would sail through that check with a large Hamming distance even
though the pixels are literally the same photograph.

This script mirrors the SAME algorithm (Lanczos resize to 9x8 luma,
horizontal-adjacent-pixel bit, 64 bits, Hamming distance, d<=10 strict /
d<=16 screening per zensim/CLAUDE.md's dHash policy) but applies it to many
CROP WINDOWS of each large AIC source instead of the whole image, so a crop
match shows up as a small Hamming distance against the matching window
instead of being diluted by the rest of the big image's composition.

Why a second implementation instead of extending check_holdout_overlap: that
binary's whole-image contract (one directory of references, one training CSV)
has no notion of a sliding window, and this is a one-off audit script, not a
permanent trained/gating path — the numbers it produces feed a human review
step (per zensim/CLAUDE.md dHash policy: d<=16 is a SCREENING threshold
requiring montage + human sign-off, never an automatic blocklist), not a
source constant. The distance semantics and threshold convention are kept
identical to the canonical implementation on purpose, so results are directly
comparable to every other dHash audit in this project.

Inputs
  --big-sources DIR     directory (searched recursively) of large AIC-3 CTC /
                         AIC-4 full-resolution reference PNGs
  --candidates-tsv PATH provenance TSV with columns source_file, width,
                         height, tier, and a rendition filename column
  --candidates-dir DIR  directory holding the rendition files named in the TSV
  --out-tsv PATH        one row per candidate: best big-source match + min
                         Hamming distance + the matching window's bbox

Usage
  crop_holdout_check.py \
    --big-sources /mnt/v/dataset/aic3_ctc_epfl/original \
    --candidates-tsv /mnt/v/output/clean-picker-corpus-2026-06-26/_provenance.tsv \
    --candidates-dir /mnt/v/output/clean-picker-corpus-2026-06-26 \
    --out-tsv ~/tmp/squintly-prep/crop_holdout_d10.tsv
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

# Same 6 fixed points the project already uses for dHash screening: strict
# leak threshold (10) and the human-review screening threshold (16). See
# zensim/CLAUDE.md "dHash threshold" section.
STRICT_THRESHOLD = 10
SCREEN_THRESHOLD = 16

# Crop-window width, as a fraction of the big image's limiting dimension,
# covering "tight crop" to "nearly the whole image". Six log-ish steps, same
# spirit as the project's mandated log-spaced size sweeps.
SCALE_FRACTIONS = (0.15, 0.22, 0.32, 0.46, 0.66, 0.95)
# Fractional stride between window positions at a given scale (0.5 = 50%
# overlap). Smaller = denser search = slower.
STRIDE_FRAC = 0.5
# The base resolution each big image is downsampled to before windowing.
# Ample headroom over the final 9x8 hash target; keeps windowing/resizing
# cheap without losing crop-detection recall (a screening pass, not a
# pixel-exact match).
BIG_BASE_MAX_DIM = 900


def dhash_bits_from_luma(luma: np.ndarray) -> int:
    """Same algorithm as zensim_validate::content_clusters::dhash_64:
    resize to 9x8 luma (Lanczos), one bit per horizontally adjacent pair,
    set if left > right, 64 bits total."""
    img = Image.fromarray(luma, mode="L").resize((9, 8), Image.Resampling.LANCZOS)
    a = np.asarray(img, dtype=np.int16)  # (8, 9)
    bits = a[:, :8] > a[:, 1:9]  # (8, 8)
    h = 0
    bit = 0
    for y in range(8):
        for x in range(8):
            if bits[y, x]:
                h |= 1 << bit
            bit += 1
    return h


def load_luma(path: Path, max_dim: int | None = None) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    if max_dim is not None and max(img.size) > max_dim:
        scale = max_dim / max(img.size)
        img = img.resize(
            (max(1, round(img.width * scale)), max(1, round(img.height * scale))),
            Image.Resampling.LANCZOS,
        )
    return np.asarray(img.convert("L"), dtype=np.uint8)


def candidate_best_match(
    cand_path: str, cand_hash: int, big_paths: list[str]
) -> tuple[str, int, tuple[int, int, int, int], float]:
    """Search every big source for the crop window closest to this
    candidate's own dHash. Returns (best_big_name, min_hamming, bbox,
    window_width_frac_of_big)."""
    cand_luma = load_luma(Path(cand_path), max_dim=BIG_BASE_MAX_DIM)
    ch, cw = cand_luma.shape
    ar = cw / ch

    best = (None, 65, (0, 0, 0, 0), 0.0)
    for big_path in big_paths:
        big = load_luma(Path(big_path), max_dim=BIG_BASE_MAX_DIM)
        bh, bw = big.shape
        for frac in SCALE_FRACTIONS:
            # Window sized to the candidate's AR, limited by the big image.
            win_w = max(9, round(bw * frac))
            win_h = max(8, round(win_w / ar))
            if win_h > bh:
                win_h = bh
                win_w = max(9, round(win_h * ar))
            if win_w > bw or win_h > bh:
                continue
            stride_x = max(1, round(win_w * STRIDE_FRAC))
            stride_y = max(1, round(win_h * STRIDE_FRAC))
            xs = list(range(0, max(1, bw - win_w + 1), stride_x)) or [0]
            ys = list(range(0, max(1, bh - win_h + 1), stride_y)) or [0]
            if xs[-1] != bw - win_w:
                xs.append(max(0, bw - win_w))
            if ys[-1] != bh - win_h:
                ys.append(max(0, bh - win_h))
            for y0 in ys:
                for x0 in xs:
                    window = big[y0 : y0 + win_h, x0 : x0 + win_w]
                    if window.size == 0:
                        continue
                    wh = dhash_bits_from_luma(window)
                    d = bin(wh ^ cand_hash).count("1")
                    if d < best[1]:
                        best = (
                            Path(big_path).name,
                            d,
                            (x0, y0, win_w, win_h),
                            round(win_w / bw, 4),
                        )
    return (cand_path, *best)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--big-sources", required=True, type=Path)
    ap.add_argument("--candidates-tsv", required=True, type=Path)
    ap.add_argument("--candidates-dir", required=True, type=Path)
    ap.add_argument("--out-tsv", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap candidate count")
    a = ap.parse_args()

    big_paths = sorted(
        str(p)
        for p in a.big_sources.rglob("*")
        if p.suffix.lower() in (".png", ".ppm", ".bmp", ".jpg", ".jpeg")
    )
    if not big_paths:
        raise SystemExit(f"no big source images found under {a.big_sources}")
    print(f"[crop-check] {len(big_paths)} big sources: {[Path(p).name for p in big_paths]}")

    # One rendition (the largest by pixel area) per unique source_file.
    best_rend: dict[str, tuple[int, str, int, int]] = {}
    with open(a.candidates_tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sf = row.get("source_file")
            rend = row.get("rendition")
            if not sf or not rend:
                continue
            try:
                w, h = int(row["width"]), int(row["height"])
            except (KeyError, ValueError):
                continue
            area = w * h
            if sf not in best_rend or area > best_rend[sf][0]:
                best_rend[sf] = (area, rend, w, h)

    candidates = []
    missing = 0
    for sf, (_, rend, w, h) in best_rend.items():
        p = a.candidates_dir / rend
        if not p.exists():
            missing += 1
            continue
        candidates.append((sf, str(p)))
    candidates.sort()
    if a.limit:
        candidates = candidates[: a.limit]
    print(f"[crop-check] {len(candidates)} unique candidate sources ({missing} missing on disk)")

    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {}
        for sf, path in candidates:
            cand_hash_luma = load_luma(Path(path), max_dim=BIG_BASE_MAX_DIM)
            ch = dhash_bits_from_luma(cand_hash_luma)
            futs[ex.submit(candidate_best_match, path, ch, big_paths)] = sf
        done = 0
        for fut in as_completed(futs):
            sf = futs[fut]
            cand_path, big_name, dist, bbox, frac = fut.result()
            rows.append(
                {
                    "source_file": sf,
                    "candidate_path": cand_path,
                    "best_big_source": big_name,
                    "min_hamming": dist,
                    "window_x": bbox[0],
                    "window_y": bbox[1],
                    "window_w": bbox[2],
                    "window_h": bbox[3],
                    "window_frac_of_big_width": frac,
                    "flag_strict_d10": dist <= STRICT_THRESHOLD,
                    "flag_screen_d16": dist <= SCREEN_THRESHOLD,
                }
            )
            done += 1
            if done % 50 == 0 or done == len(candidates):
                print(f"[crop-check] {done}/{len(candidates)}  {time.time() - t0:.0f}s", flush=True)

    rows.sort(key=lambda r: r["min_hamming"])
    a.out_tsv.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_tsv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
        w.writeheader()
        w.writerows(rows)

    n_strict = sum(r["flag_strict_d10"] for r in rows)
    n_screen = sum(r["flag_screen_d16"] for r in rows)
    print(
        f"[crop-check] DONE {len(rows)} candidates x {len(big_paths)} big sources "
        f"({time.time() - t0:.0f}s). strict(d<=10)={n_strict} screen(d<=16)={n_screen} "
        f"min_dist_overall={rows[0]['min_hamming'] if rows else None}"
    )
    meta = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "big_sources_dir": str(a.big_sources),
        "big_source_files": [Path(p).name for p in big_paths],
        "candidates_tsv": str(a.candidates_tsv),
        "n_candidates": len(rows),
        "scale_fractions": list(SCALE_FRACTIONS),
        "stride_frac": STRIDE_FRAC,
        "base_max_dim": BIG_BASE_MAX_DIM,
        "strict_threshold": STRICT_THRESHOLD,
        "screen_threshold": SCREEN_THRESHOLD,
        "n_flagged_strict_d10": n_strict,
        "n_flagged_screen_d16": n_screen,
        "elapsed_s": round(time.time() - t0, 1),
    }
    (a.out_tsv.parent / (a.out_tsv.stem + "_MANIFEST.json")).write_text(json.dumps(meta, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
