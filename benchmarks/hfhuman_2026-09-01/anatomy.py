#!/usr/bin/env python3
"""Dataset anatomy + geometry gates for the JPEG-AIC study family.

Establishes, from bytes only, the facts the ingestion lane rests on:

  G1  every PTC crop is a PIXEL-EXACT crop of the AIC-3 CTC full-resolution
      source, at a recovered (y, x) offset -- so a PTC stimulus can be scored
      as-is and its provenance chains back to the CTC encode.
  G2  every BTC crop is a 2x MAGNIFIED, distortion-AMPLIFIED rendering of a
      310x400 region of the CTC source: its residual against its own reference
      is a scaled copy of the CTC encode's residual on that region (corr > 0.8)
      with gain > 1.5, i.e. NOT native amplitude.  The gain is reported by two
      estimators that bracket it and is deliberately NOT pinned to a single
      number: the least-squares gain is ATTENUATED (the regressor is the
      native-grid residual, the response is a 2x-upsample-then-subsample of it,
      which loses high-frequency energy) and the MAE ratio is BIASED UP at
      level 1 (residuals near the 8-bit quantisation floor).  The arm design
      only needs "magnified and amplified", which is what the gate asserts.
  G3  the SDR25 `crops_sources/PTC_*` refs are PIXEL-identical to the AIC-3
      PTC zip's refs (one stimulus family, not two). Byte identity is NOT the
      test and does NOT hold -- the two distributions re-encoded the same
      pixels with different PNG writers; both shas are recorded.
  G4  the raw response CSVs' `is_bias` flag is exactly `img_left == img_right`,
      and `is_trap` rows always pair the original against dlevel 10.

Writes one JSON. No statistics are computed here (see `zenstats`); this file
only measures pixels and counts rows.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, sys
from pathlib import Path
import numpy as np
from PIL import Image

SDR25 = Path("/mnt/v/datasets/jpeg-ai-sdr25/dataset-JPEG-AI-SDR25")
CTC = Path("/mnt/v/dataset/aic3_ctc_epfl")
AIC3_CSV_DIR = Path("/mnt/v/datasets/aic3-btc-ptc")
SDR25_CSV_DIR = Path("/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data")

# img_num -> CTC full-resolution basename (the 5 images the crop studies used)
CROP_IMGS = {
    2: "00002_853x945", 6: "00006_2048x1536", 7: "00007_1600x1200",
    9: "00009_2048x1536", 10: "00010_2592x1946",
}
# response-CSV codec id -> stimulus-filename codec token / CTC decode token
CODEC_STIM = {1: "AVIF", 2: "JPEG-1", 3: "JPEG-2000", 4: "JPEG-XL", 5: "VVC", 6: "JPEG-AI"}
CODEC_CTC = {1: "AVIF", 2: "JPEG-1", 3: "JPEG-2000", 4: "JPEGXL", 5: "VVC"}


def load(p) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB")).astype(np.float64)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def find_exact_crop(full: np.ndarray, crop: np.ndarray):
    """Exhaustive exact-match offset search (cheap: search space <= 233x145)."""
    h, w, _ = crop.shape
    H, W, _ = full.shape
    tile = crop[:8, :8]
    for y in range(H - h + 1):
        for x in range(W - w + 1):
            if np.array_equal(full[y:y + 8, x:x + 8], tile) and np.array_equal(
                full[y:y + h, x:x + w], crop
            ):
                return (y, x)
    return None


def find_best_offset(full: np.ndarray, crop: np.ndarray, stride: int = 8):
    """Min-MAE offset for an inexact (resampled) crop."""
    h, w, _ = crop.shape
    H, W, _ = full.shape
    fg, g = full.mean(axis=2), crop.mean(axis=2)
    best = None
    for y in range(H - h + 1):
        for x in range(W - w + 1):
            d = float(np.abs(fg[y:y + h:stride, x:x + w:stride] - g[::stride, ::stride]).mean())
            if best is None or d < best[0]:
                best = (d, y, x)
    d, y, x = best
    return (y, x), float(np.abs(full[y:y + h, x:x + w] - crop).mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--btc-dir", required=True, help="extracted BTC_images.zip root (…/BTC2)")
    ap.add_argument("--ptc-dir", required=True, help="extracted PTC_images.zip root (…/PTC3_images)")
    ap.add_argument("--amp-levels", default="1,5,10")
    a = ap.parse_args()
    btc_root, ptc_root = Path(a.btc_dir), Path(a.ptc_dir)
    levels = [int(x) for x in a.amp_levels.split(",")]
    out: dict = {"gates": {}, "geometry": {}, "amplification": [], "responses": {}}

    # ---- G1 / G3 : PTC crops --------------------------------------------
    g1, g3 = [], []
    for num, name in CROP_IMGS.items():
        full = load(CTC / "original" / f"{name}.png")
        ptc = ptc_root / name / f"PTC_{num:05d}_0ref_00.png"
        crop = load(ptc)
        off = find_exact_crop(full, crop)
        g1.append({"img_num": num, "ctc_source": name, "offset": off,
                   "exact": off is not None, "crop_hw": list(crop.shape[:2])})
        sdr = SDR25 / "crops_sources" / f"PTC_{num:05d}_0ref_00.png"
        same_px = bool(np.array_equal(load(ptc), load(sdr)))
        g3.append({"img_num": num, "sha_aic3_zip": sha256(ptc), "sha_sdr25": sha256(sdr),
                   "bytes_identical": sha256(ptc) == sha256(sdr),
                   "pixels_identical": same_px})
    out["geometry"]["ptc_crop_offsets"] = g1
    out["gates"]["G1_ptc_is_exact_crop_of_ctc_source"] = all(r["exact"] for r in g1)
    out["geometry"]["ptc_ref_sha_vs_sdr25"] = g3
    out["gates"]["G3_sdr25_crops_sources_pixel_identical_to_aic3_ptc"] = all(
        r["pixels_identical"] for r in g3)

    # ---- G2 : BTC boosting ----------------------------------------------
    btc_geo = []
    for num, name in CROP_IMGS.items():
        full = load(CTC / "original" / f"{name}.png")
        b = load(btc_root / f"{num:05d}" / f"BTC_{num:05d}_0ref_00.png")
        sub = b[0::2, 0::2]
        (y, x), mae = find_best_offset(full, sub)
        btc_geo.append({"img_num": num, "ctc_source": name, "offset_2x_region": [y, x],
                        "region_hw": list(sub.shape[:2]), "resample_mae": round(mae, 4),
                        "btc_hw": list(b.shape[:2])})
    out["geometry"]["btc_2x_regions"] = btc_geo

    amps = []
    for g in btc_geo:
        num = g["img_num"]; name = g["ctc_source"]; y, x = g["offset_2x_region"]
        h, w = g["region_hw"]
        orig_reg = load(CTC / "original" / f"{name}.png")[y:y + h, x:x + w]
        bref = load(btc_root / f"{num:05d}" / f"BTC_{num:05d}_0ref_00.png")[0::2, 0::2]
        for cid, tok in CODEC_CTC.items():
            for lvl in levels:
                bp = btc_root / f"{num:05d}" / f"BTC_{num:05d}_{CODEC_STIM[cid]}_{lvl:02d}.png"
                cp = CTC / "decoded" / name / f"{tok}_{name}_{lvl}.png"
                if not (bp.exists() and cp.exists()):
                    amps.append({"img_num": num, "codec": CODEC_STIM[cid], "level": lvl,
                                 "status": "MISSING", "btc": bp.exists(), "ctc": cp.exists()})
                    continue
                bd = load(bp)[0::2, 0::2]
                cd = load(cp)[y:y + h, x:x + w]
                rb = (bd - bref).ravel()
                rc = (cd - orig_reg).ravel()
                res_btc = float(np.abs(rb).mean())
                res_ctc = float(np.abs(rc).mean())
                # Least-squares gain of the BTC residual on the CTC residual --
                # the unbiased estimator. The MAE ratio is kept beside it
                # because it is what a reader can check by hand, but it is
                # biased upward at level 1 where the residual is near the
                # 8-bit quantisation floor.
                den = float(rc @ rc)
                amps.append({"img_num": num, "codec": CODEC_STIM[cid], "level": lvl,
                             "status": "OK", "btc_residual_mae": round(res_btc, 4),
                             "ctc_residual_mae": round(res_ctc, 4),
                             "amplification_mae_ratio": round(res_btc / res_ctc, 4) if res_ctc else None,
                             "amplification_ls": round(float(rb @ rc) / den, 4) if den else None,
                             "residual_corr": round(float((rb @ rc) / (np.linalg.norm(rb) * np.linalg.norm(rc))), 4)
                             if den else None})
    out["amplification"] = amps
    ls = [r["amplification_ls"] for r in amps if r.get("amplification_ls")]
    mr = [r["amplification_mae_ratio"] for r in amps if r.get("amplification_mae_ratio")]
    cr = [r["residual_corr"] for r in amps if r.get("residual_corr")]
    out["gates"]["G2_btc_amplification"] = {
        "n": len(ls),
        "ls_min": round(min(ls), 4) if ls else None,
        "ls_max": round(max(ls), 4) if ls else None,
        "ls_mean": round(float(np.mean(ls)), 4) if ls else None,
        "mae_ratio_min": round(min(mr), 4) if mr else None,
        "mae_ratio_max": round(max(mr), 4) if mr else None,
        "mae_ratio_mean": round(float(np.mean(mr)), 4) if mr else None,
        "residual_corr_min": round(min(cr), 4) if cr else None,
    }
    out["gates"]["G2_btc_is_amplified_not_native"] = bool(ls) and all(k > 1.5 for k in ls) and all(
        c > 0.8 for c in cr)

    # ---- G4 : response-flag semantics -----------------------------------
    csvs = {
        "aic3_btc": AIC3_CSV_DIR / "JPEG-AIC_BTC_final_response_data_2024.01.10.csv",
        "aic3_iptc": AIC3_CSV_DIR / "JPEG-AIC_IPTC_final_response_data_2024_06_28 (1).csv",
        "sdr25_btc": SDR25_CSV_DIR / "JPEG_AIC_SDR_BTC_JPEG_AI_responses_2025.02.28_v1.csv",
        "sdr25_ptc": SDR25_CSV_DIR / "JPEG_AIC_SDR_PTC_JPEG_AI_responses_2025.02.28_v1.csv",
    }
    g4_ok = True
    for key, p in csvs.items():
        n = 0; bias_mismatch = 0; trap_mismatch = 0; pivot_not_orig = 0
        resp = {}; stim = set(); qs = set(); workers = set()
        with open(p) as f:
            for r in csv.DictReader(f):
                n += 1
                same_img = r["img_left"] == r["img_right"]
                if (r["is_bias"] == "1") != same_img:
                    bias_mismatch += 1
                if r["is_trap"] == "1":
                    lv, rv = int(r["dlevel_left"]), int(r["dlevel_right"])
                    if sorted((lv, rv)) != [0, 10]:
                        trap_mismatch += 1
                if not (r["codec_pivot"] == "0" and r["dlevel_pivot"] == "0"):
                    pivot_not_orig += 1
                resp[r["response"]] = resp.get(r["response"], 0) + 1
                stim.update((r["img_left"], r["img_right"], r["img_pivot"]))
                qs.add(r["question_id"]); workers.add(r["worker"])
        ok = bias_mismatch == 0 and trap_mismatch == 0 and pivot_not_orig == 0
        g4_ok &= ok
        out["responses"][key] = {
            "path": str(p), "sha256": sha256(p), "n_responses": n,
            "n_questions": len(qs), "n_workers": len(workers), "n_stimuli": len(stim),
            "responses": resp, "is_bias_equals_same_stimulus_mismatches": bias_mismatch,
            "is_trap_is_0_vs_10_mismatches": trap_mismatch,
            "pivot_not_original_rows": pivot_not_orig, "flag_semantics_ok": ok,
        }
    out["gates"]["G4_response_flag_semantics"] = bool(g4_ok)

    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    with open(a.json, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(json.dumps(out["gates"], indent=2))
    print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
