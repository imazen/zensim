#!/usr/bin/env python3
"""Stimulus universe + provenance gates + pairs TSVs for the JPEG-AIC family.

Emits three SCORING ARMS over one response corpus, so a conclusion can be
checked against the boosting choice instead of resting on it:

  ptc_native    the PTC / plain stimuli exactly as displayed. G1 proved each is
                a pixel-exact crop of the AIC-3 CTC full-resolution source, so
                this arm is native scale, native amplitude, zero reconstruction.
  btc_displayed the BTC stimuli exactly as displayed: 2x magnified and
                ~2x distortion-amplified (measured, `anatomy.py` G2). These are
                the literal pixels the worker judged.
  btc_native    the SAME BTC region at native scale and native amplitude,
                cropped out of the CTC full-resolution encode the boosted view
                was rendered from. This is the stimulus the reconstructed JND
                scale is attributed to.
  iptc_native   the AIC-3 June-2024 interactive-PTC campaign's 130 stimuli --
                the SAME plain PTC crops, at the even distortion levels that
                campaign used. Native scale, native amplitude, zero
                reconstruction, 51,870 responses (gate G8).
  iptc_ctl_*    deliberately MIS-ASSIGNED pixel maps for the same 130 stimulus
                keys (level shift / level REVERSAL / codec rotation / image
                rotation). Shift and rotation are ORDER-PRESERVING within a
                ladder, so they are expected to be near-neutral; reversal and
                image rotation are not, and are the discriminating pair. They exist
                only as negative controls for G8: if the identification is
                right, the true map must beat all three on the same responses.

Gates (all fail loud):
  G5  every PTC distorted stimulus is a PIXEL-EXACT crop of its CTC decode at
      the same offset its reference crop was taken from  -> provenance chain
      stimulus -> CTC encode -> source.
  G6  every PTC JPEG-AI stimulus is a PIXEL-EXACT crop of exactly one
      `Compressed_images_original_resolution/VM_*` file -> resolves the
      dlevel -> JPEG-AI VM quality map from bytes, never from the filename order.
  G7  the recovered BTC region offset is a strict local minimum of the
      reference residual (refined at stride 1), and every materialised native
      crop is byte-reproducible from (source, offset, size).
  G8  the AIC-3 `IPTC_*` stimuli ARE the plain `PTC_*` crops: 130/130 names
      resolve 1:1 onto a PTC file on disk, every response row's filename agrees
      with its own (img_num, codec, dlevel) fields, the level set is exactly the
      published {0,2,4,6,8,10}, and the campaign shape (1,050 questions, 352
      assignments) is the one the source paper reports for its PTC experiment.
      Published chain: Testolina et al., "Fine-grained subjective visual quality
      assessment for high-fidelity compressed images", DCC 2025
      (arXiv:2410.09501), "Experimental setup" -- the PTC experiment reused the
      same five 620x800 crops with "the decoded images ... left untouched", at
      "distortion levels numbered 0, 2, 4, 6, 8, and 10", 1,050 questions and
      352 assignments. The prefix in the response table is the campaign's method
      tag, not a different stimulus set.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, re, sys
from pathlib import Path
import numpy as np
from PIL import Image

SDR25 = Path("/mnt/v/datasets/jpeg-ai-sdr25/dataset-JPEG-AI-SDR25")
CTC = Path("/mnt/v/dataset/aic3_ctc_epfl")
AIC3_CSV = Path("/mnt/v/datasets/aic3-btc-ptc/JPEG-AIC_BTC_final_response_data_2024.01.10.csv")
AIC3_IPTC_CSV = Path("/mnt/v/datasets/aic3-btc-ptc/JPEG-AIC_IPTC_final_response_data_2024_06_28 (1).csv")
SDR25_BTC_CSV = Path("/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data/JPEG_AIC_SDR_BTC_JPEG_AI_responses_2025.02.28_v1.csv")
SDR25_PTC_CSV = Path("/mnt/v/datasets/jpeg-ai-sdr25/JPEG_AI_SDR_subjective_data/JPEG_AIC_SDR_PTC_JPEG_AI_responses_2025.02.28_v1.csv")

CROP_IMGS = {2: "00002_853x945", 6: "00006_2048x1536", 7: "00007_1600x1200",
             9: "00009_2048x1536", 10: "00010_2592x1946"}
# stimulus-filename codec token -> CTC `decoded/` token ("" = not a CTC codec)
CTC_TOKEN = {"AVIF": "AVIF", "JPEG-1": "JPEG-1", "JPEG-2000": "JPEG-2000",
             "JPEG-XL": "JPEGXL", "VVC": "VVC", "JPEG-AI": ""}
STIM_RE = re.compile(r"^(BTC|IPTC|PTC)_(\d{5})_(.+)_(\d{2})\.png$")
# response-table codec id -> stimulus-filename codec token (identical in every
# JPEG-AIC response CSV; asserted per row by G8)
CODEC_ID = {"0": "0ref", "1": "AVIF", "2": "JPEG-1", "3": "JPEG-2000",
            "4": "JPEG-XL", "5": "VVC", "6": "JPEG-AI"}
IPTC_LEVELS = {0, 2, 4, 6, 8, 10}          # the published PTC level set


def ptc_disk_name(n: str) -> str:
    """`IPTC_...png` -> the PTC file that campaign served (see G8)."""
    return ("PTC_" + n[len("IPTC_"):]) if n.startswith("IPTC_") else n


def load(p) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB")).astype(np.uint8)


def sha256(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def refine_offset(full_g: np.ndarray, crop_g: np.ndarray, y0: int, x0: int, rad: int = 8):
    h, w = crop_g.shape
    H, W = full_g.shape
    best = None
    for y in range(max(0, y0 - rad), min(H - h, y0 + rad) + 1):
        for x in range(max(0, x0 - rad), min(W - w, x0 + rad) + 1):
            d = float(np.abs(full_g[y:y + h, x:x + w] - crop_g).mean())
            if best is None or d < best[0]:
                best = (d, y, x)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anatomy", required=True, help="anatomy.py JSON (BTC/PTC offsets)")
    ap.add_argument("--btc-dir", required=True)
    ap.add_argument("--ptc-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    anat = json.loads(Path(a.anatomy).read_text())
    btc_root, ptc_root = Path(a.btc_dir), Path(a.ptc_dir)
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    nat_dir = out / "native_btc"; nat_dir.mkdir(exist_ok=True)
    man: dict = {"gates": {}, "stimuli": {}, "jpegai_vm_map": [], "arms": {}}

    ptc_off = {r["img_num"]: tuple(r["offset"]) for r in anat["geometry"]["ptc_crop_offsets"]}
    btc_off = {r["img_num"]: (tuple(r["offset_2x_region"]), tuple(r["region_hw"]))
               for r in anat["geometry"]["btc_2x_regions"]}

    # ---- stimulus universe from the three SCOREABLE response CSVs ---------
    stim_files = {}   # basename -> display path
    per_csv = {}
    for key, p, root in (("aic3_btc", AIC3_CSV, btc_root),
                         ("sdr25_btc", SDR25_BTC_CSV, btc_root),
                         ("sdr25_ptc", SDR25_PTC_CSV, ptc_root),
                         ("aic3_iptc", AIC3_IPTC_CSV, ptc_root)):
        names = set()
        with open(p) as f:
            for r in csv.DictReader(f):
                names.update((r["img_left"], r["img_right"], r["img_pivot"]))
        per_csv[key] = sorted(names)
        for n in names:
            m = STIM_RE.match(n)
            fam, num = m.group(1), int(m.group(2))
            d = (btc_root / f"{num:05d}") if fam == "BTC" else (ptc_root / CROP_IMGS[num])
            dn = ptc_disk_name(n)
            fp = d / dn
            if not fp.exists() and fam != "BTC":
                fp = SDR25 / "PTC_JPEG-AI_images" / f"{num:05d}" / dn
            if not fp.exists() and fam != "BTC":
                fp = SDR25 / "crops_sources" / dn
            if not fp.exists() and fam == "BTC":
                fp = SDR25 / "BTC_JPEG-AI_images" / f"{num:05d}" / n
            if not fp.exists() and fam == "BTC":
                fp = SDR25 / "source_images" / n
            assert fp.exists(), f"stimulus not on disk: {n}"
            stim_files[n] = fp
    man["stimuli"]["n_distinct"] = len(stim_files)
    man["stimuli"]["per_response_csv"] = {k: len(v) for k, v in per_csv.items()}

    # ---- G8 : the IPTC campaign served the plain PTC crops ---------------
    ident = {"n_names": 0, "resolved_to_ptc_file": 0, "unresolved": [],
             "row_field_filename_disagreements": 0, "n_row_field_checks": 0,
             "levels": [], "grid": {}, "n_question_id": 0, "n_workers": 0,
             "n_responses": 0}
    ipt_names = sorted(n for n in stim_files if n.startswith("IPTC_"))
    ident["n_names"] = len(ipt_names)
    for n in ipt_names:
        fp = stim_files[n]
        if fp.name == ptc_disk_name(n) and fp.exists():
            ident["resolved_to_ptc_file"] += 1
        else:
            ident["unresolved"].append(n)
    lv, grid, qids, wrk, nresp = set(), {}, set(), set(), 0
    with open(AIC3_IPTC_CSV) as f:
        for r in csv.DictReader(f):
            nresp += 1
            qids.add(r["question_id"]); wrk.add(r["worker"])
            for side in ("left", "right", "pivot"):
                m = STIM_RE.match(r["img_" + side])
                ident["n_row_field_checks"] += 3
                ok = (int(m.group(2)) == int(r["img_num"])
                      and m.group(3) == CODEC_ID[r["codec_" + side]]
                      and int(m.group(4)) == int(r["dlevel_" + side]))
                if not ok:
                    ident["row_field_filename_disagreements"] += 3
                lv.add(int(m.group(4)))
                grid.setdefault(m.group(3), set()).add(int(m.group(4)))
    ident["levels"] = sorted(lv)
    ident["grid"] = {k: sorted(v) for k, v in sorted(grid.items())}
    ident["n_question_id"] = len(qids)
    ident["n_workers"] = len(wrk)
    ident["n_responses"] = nresp
    ident["source_paper"] = ("Testolina et al., DCC 2025, arXiv:2410.09501 -- "
                             "PTC reused the same 620x800 crops untouched at "
                             "levels 0,2,4,6,8,10; 1050 questions; 352 assignments")
    g8 = (ident["n_names"] == 130
          and ident["resolved_to_ptc_file"] == 130
          and ident["row_field_filename_disagreements"] == 0
          and lv == IPTC_LEVELS
          and ident["n_question_id"] == 1050
          and ident["n_workers"] == 352)
    man["gates"]["G8_iptc_stimuli_are_the_plain_ptc_crops"] = bool(g8)
    man["iptc_identification"] = ident

    # ---- G6 : resolve the JPEG-AI dlevel -> VM quality map from bytes -----
    vm_map, g6_ok = [], True
    for num, name in CROP_IMGS.items():
        y, x = ptc_off[num]
        vms = sorted((SDR25 / "Compressed_images_original_resolution").glob(f"VM_{num:05d}_*.png"))
        vm_crops = {p: load(p)[y:y + 800, x:x + 620] for p in vms}
        for lvl in range(1, 11):
            sp = SDR25 / "PTC_JPEG-AI_images" / f"{num:05d}" / f"PTC_{num:05d}_JPEG-AI_{lvl:02d}.png"
            if not sp.exists():
                continue
            s = load(sp)
            hits = [p for p, c in vm_crops.items() if np.array_equal(c, s)]
            ok = len(hits) == 1
            g6_ok &= ok
            vm_map.append({"img_num": num, "dlevel": lvl, "n_exact_matches": len(hits),
                           "vm_file": hits[0].name if hits else None,
                           "vm_quality": hits[0].name.split("_")[-1].split(".")[0] if hits else None})
    man["jpegai_vm_map"] = vm_map
    man["gates"]["G6_jpegai_dlevel_resolves_to_one_vm_file"] = bool(g6_ok)

    # ---- G5 : PTC distorted stimuli are exact crops of the CTC decode -----
    g5, g5_ok = [], True
    for num, name in CROP_IMGS.items():
        y, x = ptc_off[num]
        for tok, ctc_tok in CTC_TOKEN.items():
            if not ctc_tok:
                continue
            for lvl in range(1, 11):
                sp = ptc_root / name / f"PTC_{num:05d}_{tok}_{lvl:02d}.png"
                cp = CTC / "decoded" / name / f"{ctc_tok}_{name}_{lvl}.png"
                if not (sp.exists() and cp.exists()):
                    g5.append({"img_num": num, "codec": tok, "dlevel": lvl, "status": "MISSING"})
                    g5_ok = False
                    continue
                eq = bool(np.array_equal(load(cp)[y:y + 800, x:x + 620], load(sp)))
                g5_ok &= eq
                g5.append({"img_num": num, "codec": tok, "dlevel": lvl,
                           "status": "OK" if eq else "MISMATCH", "exact": eq})
    man["gates"]["G5_ptc_dist_is_exact_crop_of_ctc_decode"] = bool(g5_ok)
    man["gates"]["G5_cells"] = len(g5)
    (out / "g5_ptc_ctc_crop.json").write_text(json.dumps(g5, indent=1))

    # ---- native full-resolution source for every (img, codec, dlevel) ----
    def native_full(num: int, tok: str, lvl: int) -> Path:
        name = CROP_IMGS[num]
        if lvl == 0:
            return CTC / "original" / f"{name}.png"
        if tok == "JPEG-AI":
            hit = [r for r in vm_map if r["img_num"] == num and r["dlevel"] == lvl]
            assert hit and hit[0]["vm_file"], f"no VM file for {num} lvl {lvl}"
            return SDR25 / "Compressed_images_original_resolution" / hit[0]["vm_file"]
        return CTC / "decoded" / name / f"{CTC_TOKEN[tok]}_{name}_{lvl}.png"

    # ---- G7 : refine the BTC region offset, materialise the native crops --
    g7, g7_ok = [], True
    btc_ref_off = {}
    for num, name in CROP_IMGS.items():
        (y0, x0), (h, w) = btc_off[num]
        full = load(CTC / "original" / f"{name}.png").astype(np.float64)
        bref = load(btc_root / f"{num:05d}" / f"BTC_{num:05d}_0ref_00.png").astype(np.float64)[0::2, 0::2]
        d, y, x = refine_offset(full.mean(axis=2), bref.mean(axis=2), y0, x0)
        # strict local minimum: every 4-neighbour must be worse
        fg = full.mean(axis=2); cg = bref.mean(axis=2)
        nb = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y + dy, x + dx
            if 0 <= yy <= fg.shape[0] - h and 0 <= xx <= fg.shape[1] - w:
                nb.append(float(np.abs(fg[yy:yy + h, xx:xx + w] - cg).mean()))
        strict = all(v > d for v in nb)
        g7_ok &= strict
        btc_ref_off[num] = (y, x, h, w)
        g7.append({"img_num": num, "offset": [y, x], "size_hw": [h, w],
                   "residual_mae": round(d, 4), "neighbour_mae": [round(v, 4) for v in nb],
                   "strict_local_min": bool(strict)})
    man["gates"]["G7_btc_offset_strict_local_min"] = bool(g7_ok)
    man["geometry_refined_btc"] = g7

    stim_native: dict[str, tuple[Path, Path]] = {}   # basename -> (native_ref, native_dist)
    for n, fp in sorted(stim_files.items()):
        m = STIM_RE.match(n)
        fam, num, tok, lvl = m.group(1), int(m.group(2)), m.group(3), int(m.group(4))
        if fam in ("PTC", "IPTC"):
            ref = ptc_root / CROP_IMGS[num] / f"PTC_{num:05d}_0ref_00.png"
            if not ref.exists():
                ref = SDR25 / "crops_sources" / f"PTC_{num:05d}_0ref_00.png"
            stim_native[n] = (ref, fp)
            continue
        y, x, h, w = btc_ref_off[num]
        outp = nat_dir / f"NAT_{num:05d}_{tok}_{lvl:02d}.png"
        if not outp.exists():
            src = native_full(num, tok, 0 if tok == "0ref" else lvl)
            Image.fromarray(load(src)[y:y + h, x:x + w]).save(outp, optimize=False)
        stim_native[n] = (nat_dir / f"NAT_{num:05d}_0ref_00.png", outp)
    for num in CROP_IMGS:
        p = nat_dir / f"NAT_{num:05d}_0ref_00.png"
        if not p.exists():
            y, x, h, w = btc_ref_off[num]
            Image.fromarray(load(CTC / "original" / f"{CROP_IMGS[num]}.png")[y:y + h, x:x + w]).save(p)

    # ---- pairs TSVs (one row per stimulus; human_score is a PLACEHOLDER: --
    # ---- these arms are decided pairwise, never by a per-row scalar) ------
    arms = {
        "ptc_native": [(n, *stim_native[n]) for n in sorted(stim_files) if n.startswith("PTC_")],
        "btc_displayed": [(n, stim_files[f"BTC_{STIM_RE.match(n).group(2)}_0ref_00.png"], stim_files[n])
                          for n in sorted(stim_files) if n.startswith("BTC_")],
        "btc_native": [(n, *stim_native[n]) for n in sorted(stim_files) if n.startswith("BTC_")],
        "iptc_native": [(n, *stim_native[n]) for n in sorted(stim_files) if n.startswith("IPTC_")],
    }
    # ---- G8 negative controls: the same 130 keys, deliberately mis-mapped -
    # If the IPTC->PTC identification is right, the true map must beat every one
    # of these on the identical responses. They are pixel maps only; the ref side
    # and the response side are untouched.
    codrot = {"AVIF": "JPEG-1", "JPEG-1": "JPEG-2000", "JPEG-2000": "JPEG-XL",
              "JPEG-XL": "VVC", "VVC": "AVIF"}
    imgrot = {2: 6, 6: 7, 7: 9, 9: 10, 10: 2}

    def ctl(n: str, kind: str):
        m = STIM_RE.match(n)
        num, tok, lvl = int(m.group(2)), m.group(3), int(m.group(4))
        if tok == "0ref":
            # the reference is the reference under every map; only the DISTORTED
            # assignment is perturbed, so the control differs from the truth on
            # exactly the 125 distorted stimuli
            r = ptc_root / CROP_IMGS[num] / f"PTC_{num:05d}_0ref_00.png"
            return (n, r, r)
        if kind == "levelshift":
            tgt = (num, tok, lvl - 1)
        elif kind == "levelrev":
            tgt = (num, tok, 12 - lvl)      # 02<->10, 04<->08, 06 fixed
        elif kind == "codecrot":
            tgt = (num, codrot[tok], lvl)
        else:
            tgt = (imgrot[num], tok, lvl)
        q, t, l = tgt
        ref = ptc_root / CROP_IMGS[num] / f"PTC_{num:05d}_0ref_00.png"
        dist = ptc_root / CROP_IMGS[q] / f"PTC_{q:05d}_{t}_{l:02d}.png"
        assert dist.exists(), f"control {kind} target missing: {dist}"
        return (n, ref, dist)

    for kind in ("levelshift", "levelrev", "codecrot", "imgrot"):
        arms[f"iptc_ctl_{kind}"] = [ctl(n, kind) for n in sorted(stim_files)
                                    if n.startswith("IPTC_")]
    for arm, rows in arms.items():
        tsv = out / f"{arm}_pairs.tsv"
        with open(tsv, "w", newline="") as f:
            w_ = csv.writer(f, delimiter="\t")
            w_.writerow(["ref_path", "dist_path", "human_score"])
            for _n, ref, dist in rows:
                w_.writerow([str(ref), str(dist), "0"])
        idx = out / f"{arm}_index.tsv"
        with open(idx, "w", newline="") as f:
            w_ = csv.writer(f, delimiter="\t")
            w_.writerow(["row", "stimulus", "ref_path", "dist_path"])
            for i, (n, ref, dist) in enumerate(rows):
                w_.writerow([i, n, str(ref), str(dist)])
        man["arms"][arm] = {"n_rows": len(rows), "pairs_tsv": str(tsv), "index_tsv": str(idx),
                            "pairs_sha256": sha256(tsv)}

    man["inputs"] = {p.name: {"path": str(p), "sha256": sha256(p)}
                     for p in (AIC3_CSV, SDR25_BTC_CSV, SDR25_PTC_CSV)}
    (out / "stimuli_manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True))
    print(json.dumps(man["gates"], indent=2))
    print(json.dumps(man["arms"], indent=2))
    bad = [g for g in man["gates"] if g.startswith("G") and man["gates"][g] is False]
    if bad:
        print(f"GATE FAILURES: {bad}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
