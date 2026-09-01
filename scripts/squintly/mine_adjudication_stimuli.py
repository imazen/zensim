#!/usr/bin/env python3
"""Mine the pre-registered squintly `zensim-adjudication` stimulus set and
stage it as a squintly coefficient SplitStore + pair-list TSV.

Method is the gMAD (group Maximum Differentiation) competition per Ma et al.
2016/2020 and its IQA instantiation in Mikhailiuk, Perez-Ortiz, Yue, Suen,
Mantiuk (2021) UPIQ, arXiv:2012.10758 — see
`benchmarks/squintly_literature_basis_2026-09-01.md` Q4 for the full review
this implements. Criterion (UPIQ supplementary VII), for test metric M and
benchmark metric B:

    argmax over pairs (i, j) OF THE SAME REFERENCE of |M_i - M_j|
    subject to |B_i - B_j| < tau_sim   ("M attacks B" — B calls them similar,
                                         M calls them very different)

Run BOTH directions (M attacks B, B attacks M) per the review's protocol
implication #1: "Run it both directions... report the 2x2 attack/defence
cell, not a single pooled number." B = SSIMULACRA2 throughout (the
already-computed benchmark score in every row of the joined encode table);
M ranges over the two adjudication candidates in candidates_2026-09-01.json.

Four strata, matching the mandate:

  s1_gmad        gMAD disagreement pairs, per (candidate, direction, q-zone)
                 cell, budget ~100/cell (UPIQ's own budget precedent — "the
                 only published budget for this operation").
  s1_random      Random-pair control, same reference-restricted sampling,
                 NOT selected by disagreement. Review implication #4: "a
                 selected-hard set does not generalize... report the
                 disagreement-set result BESIDE a random-set control."
  s2_ladder      Contested ladder steps: adjacent-quality-rung pairs (same
                 reference + codec) where SSIM2 and the candidate disagree
                 about the SIGN of the step — the ladder-inversion concept
                 from benchmarks/failure_profiles_2026-08-31.md, applied here
                 to the bigcodec pool (which has real bytes on disk) rather
                 than the 39-image dial grid (whose bitstreams are not
                 materialized on this box, and which is excluded from
                 candidate pools by build_encode_table.py's own DIAL_CLASSES
                 guard against feeding the dial instrument back into a human
                 corpus).
  s3_calibration Golden / attention-check pairs: SSIM2 and BOTH candidates
                 agree on direction by a wide margin. expected_choice is set;
                 these are the calibration rows squintly's grading.rs already
                 knows how to score.

Repeats (a fraction of s1/s2 rows, `repeat_of_pair` set at a later `seq`) are
added last, per the review's test-retest guidance and squintly's own
study-pairs design (planned rows, not `p_repeat`).

tau_sim = 5.0 SSIM2/model points, reusing squintly's OWN already-validated
threshold (CHANGELOG.md, "Changed" section, 2026-08 ladder-spacing entry:
"human agreement with ssim2 hits 100% by a 5-point gap") as the "benchmark
calls them similar" cutoff, rather than inventing a new one.

Reads: the joined encode table from build_encode_table.py (encoded_filename,
ref_basename, ref_path, encode_path, decoded_path, codec, q, ssim2, m_<name>
per candidate). Reads: the crop-holdout audit TSV to exclude any flagged
reference (belt-and-suspenders; the 2026-09-01 audit found zero genuine
matches, see benchmarks/squintly_adjudication_protocol_2026-09-01.md, but the
exclusion is mechanical so a future re-run with a different corpus stays
protected without code changes).

Writes: a coefficient SplitStore (meta/{sources,encodings}/*.json,
blobs/{sources,encodings}/*) under --out-corpus, and a pairs TSV in
`pair_manifest::parse_delimited`'s expected shape under --out-pairs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

TAU_SIM = 5.0  # squintly's own validated "ssim2 calls them similar" gap
LADDER_MATERIAL = 0.5  # failure_profiles_2026-08-31.md's material-inversion floor
QZONES = (("low", 0, 50), ("mid", 50, 85), ("high", 85, 101))
PER_REF_CAP_S1 = 3  # max gMAD pairs from one reference within one cell
PER_REF_CAP_S2 = 3
CODEC_EXT = {"zenjpeg": "jpg", "zenwebp": "webp", "zenjxl": "jxl", "zenavif": "avif"}


def qzone(q: float) -> str:
    for name, lo, hi in QZONES:
        if lo <= q < hi:
            return name
    return "high"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rows(encodes_parquet: Path, exclude_refs: set[str]) -> dict:
    t = pq.read_table(encodes_parquet).to_pydict()
    n = len(t["encoded_filename"])
    by_ref = defaultdict(list)
    for i in range(n):
        rb = t["ref_basename"][i]
        if rb in exclude_refs:
            continue
        by_ref[rb].append(i)
    cols = {k: v for k, v in t.items()}
    return cols, by_ref


def row(cols: dict, i: int) -> dict:
    return {k: cols[k][i] for k in cols}


def pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a < b else (b, a)


def mine_gmad(cols, by_ref, candidates: list[str], rng: random.Random) -> list[dict]:
    """Stratum 1: gMAD disagreement, both directions, both candidates, 3 q-zones."""
    out = []
    for cand in candidates:
        mcol = f"m_{cand}"
        for direction in ("m_attacks_b", "b_attacks_m"):
            for zname, _, _ in QZONES:
                # Score every within-reference pair in this zone, then take the
                # top PER_REF_CAP_S1 per reference, then the top ~100 overall —
                # this is what keeps content diversity (protocol implication #4:
                # never let one reference dominate a cell) while still reaching
                # deep enough into the ranked list to hit ~100/cell.
                per_ref_best: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
                for rb, idxs in by_ref.items():
                    zoned = [i for i in idxs if qzone(float(cols["q"][i])) == zname]
                    if len(zoned) < 2:
                        continue
                    scored = []
                    for a in range(len(zoned)):
                        for b in range(a + 1, len(zoned)):
                            ia, ib = zoned[a], zoned[b]
                            dm = abs(cols[mcol][ia] - cols[mcol][ib])
                            db = abs(cols["ssim2"][ia] - cols["ssim2"][ib])
                            if direction == "m_attacks_b":
                                if db >= TAU_SIM:
                                    continue
                                score = dm
                            else:
                                if dm >= TAU_SIM:
                                    continue
                                score = db
                            scored.append((score, ia, ib))
                    scored.sort(key=lambda s: -s[0])
                    per_ref_best[rb] = scored[:PER_REF_CAP_S1]
                pool = [x for lst in per_ref_best.values() for x in lst]
                pool.sort(key=lambda s: -s[0])
                for score, ia, ib in pool[:100]:
                    out.append(
                        {
                            "stratum": f"s1_gmad_{cand}_{direction}_{zname}",
                            "a": ia,
                            "b": ib,
                            "sort_score": score,
                            "meta": {
                                "mining": "gmad",
                                "candidate": cand,
                                "direction": direction,
                                "qzone": zname,
                                "tau_sim": TAU_SIM,
                                f"score_{cand}_a": cols[mcol][ia],
                                f"score_{cand}_b": cols[mcol][ib],
                                "score_ssim2_a": cols["ssim2"][ia],
                                "score_ssim2_b": cols["ssim2"][ib],
                            },
                        }
                    )
    return out


def mine_random_control(cols, by_ref, n_target: int, rng: random.Random, seen: set) -> list[dict]:
    """Stratum 1b: random within-reference pairs, unconditional on disagreement."""
    refs = [rb for rb, idxs in by_ref.items() if len(idxs) >= 2]
    out = []
    tries = 0
    while len(out) < n_target and tries < n_target * 50:
        tries += 1
        rb = rng.choice(refs)
        idxs = by_ref[rb]
        ia, ib = rng.sample(idxs, 2)
        key = pair_key(ia, ib)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "stratum": "s1_random_control",
                "a": ia,
                "b": ib,
                "sort_score": 0.0,
                "meta": {
                    "mining": "random_control",
                    "score_ssim2_a": cols["ssim2"][ia],
                    "score_ssim2_b": cols["ssim2"][ib],
                },
            }
        )
    return out


def mine_ladder(cols, by_ref, candidates: list[str], rng: random.Random) -> list[dict]:
    """Stratum 2: adjacent-rung sign disagreements within one (ref, codec) ladder."""
    out = []
    ladders: dict[tuple[str, str], list[int]] = defaultdict(list)
    for rb, idxs in by_ref.items():
        by_codec = defaultdict(list)
        for i in idxs:
            by_codec[cols["codec"][i]].append(i)
        for codec, ii in by_codec.items():
            ladders[(rb, codec)] = sorted(ii, key=lambda i: float(cols["q"][i]))
    for cand in candidates:
        mcol = f"m_{cand}"
        for zname, _, _ in QZONES:
            per_ref_best: dict[str, list[tuple[float, int, int]]] = defaultdict(list)
            for (rb, codec), ii in ladders.items():
                scored = []
                for k in range(len(ii) - 1):
                    ia, ib = ii[k], ii[k + 1]  # ib = next rung up in q
                    qmid = (float(cols["q"][ia]) + float(cols["q"][ib])) / 2.0
                    if qzone(qmid) != zname:
                        continue
                    d_ssim2 = cols["ssim2"][ib] - cols["ssim2"][ia]
                    d_m = cols[mcol][ib] - cols[mcol][ia]
                    if abs(d_ssim2) < LADDER_MATERIAL or abs(d_m) < LADDER_MATERIAL:
                        continue
                    if (d_ssim2 > 0) == (d_m > 0):
                        continue  # agree on direction, not contested
                    contest = min(abs(d_ssim2), abs(d_m))
                    scored.append((contest, ia, ib))
                scored.sort(key=lambda s: -s[0])
                per_ref_best[rb].extend(scored[:PER_REF_CAP_S2])
            pool = [x for lst in per_ref_best.values() for x in lst]
            pool.sort(key=lambda s: -s[0])
            for contest, ia, ib in pool[:100]:
                out.append(
                    {
                        "stratum": f"s2_ladder_{cand}_{zname}",
                        "a": ia,
                        "b": ib,
                        "sort_score": contest,
                        "meta": {
                            "mining": "ladder_inversion",
                            "candidate": cand,
                            "qzone": zname,
                            "q_a": cols["q"][ia],
                            "q_b": cols["q"][ib],
                            "codec": cols["codec"][ia],
                            f"score_{cand}_a": cols[mcol][ia],
                            f"score_{cand}_b": cols[mcol][ib],
                            "score_ssim2_a": cols["ssim2"][ia],
                            "score_ssim2_b": cols["ssim2"][ib],
                        },
                    }
                )
    return out


def mine_calibration(cols, by_ref, candidates: list[str], n_target: int, rng: random.Random, seen: set) -> list[dict]:
    """Stratum 3: obvious pairs, all scorers agree on direction by a wide margin."""
    candidates_scored = []
    for rb, idxs in by_ref.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                ia, ib = idxs[a], idxs[b]
                key = pair_key(ia, ib)
                if key in seen:
                    continue
                d_ssim2 = cols["ssim2"][ib] - cols["ssim2"][ia]
                if abs(d_ssim2) < 25.0:  # wide, unambiguous gap
                    continue
                agree = True
                for cand in candidates:
                    d_m = cols[f"m_{cand}"][ib] - cols[f"m_{cand}"][ia]
                    if (d_m > 0) != (d_ssim2 > 0) or abs(d_m) < 15.0:
                        agree = False
                        break
                if agree:
                    candidates_scored.append((abs(d_ssim2), ia, ib, d_ssim2 > 0))
    rng.shuffle(candidates_scored)
    # Spread across references: at most 1 calibration row per reference so the
    # attention checks aren't all trivially the same source image.
    used_refs = set()
    out = []
    for _, ia, ib, b_is_better in candidates_scored:
        rb = cols["ref_basename"][ia]
        if rb in used_refs:
            continue
        used_refs.add(rb)
        seen.add(pair_key(ia, ib))
        out.append(
            {
                "stratum": "s3_calibration",
                "a": ia,
                "b": ib,
                "sort_score": 0.0,
                "expected_choice": "b" if b_is_better else "a",
                "meta": {
                    "mining": "calibration",
                    "score_ssim2_a": cols["ssim2"][ia],
                    "score_ssim2_b": cols["ssim2"][ib],
                },
            }
        )
        if len(out) >= n_target:
            break
    return out


def add_repeats(rows: list[dict], frac: float, min_distance: int, rng: random.Random) -> list[dict]:
    """Append literal repeats of a sample of s1/s2 rows, `min_distance` seq
    positions after the original (spaced, not back-to-back — pwcmp's own
    practice per the literature review)."""
    eligible = [i for i, r in enumerate(rows) if r["stratum"].startswith(("s1_gmad", "s2_ladder"))]
    n_repeat = max(1, round(len(eligible) * frac))
    chosen = rng.sample(eligible, min(n_repeat, len(eligible)))
    out = list(rows)
    for idx in sorted(chosen):
        orig = rows[idx]
        out.append(
            {
                "stratum": orig["stratum"] + "_repeat",
                "a": orig["a"],
                "b": orig["b"],
                "sort_score": 0.0,
                "repeat_of_seq": idx,  # resolved to pair_id after seq assignment
                "meta": {"mining": "repeat", "repeat_of_stratum": orig["stratum"]},
            }
        )
    return out


def stable_pair_id(cols, a: int, b: int, stratum: str) -> str:
    key = f"{cols['encoded_filename'][a]}|{cols['encoded_filename'][b]}|{stratum}"
    return "pair_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def stage_corpus(cols, needed_idx: set[int], out_corpus: Path) -> dict[int, tuple[str, str]]:
    """Copy the bytes each planned row needs into a coefficient SplitStore.
    Returns {row_index: (source_hash, encoding_id)}."""
    (out_corpus / "meta" / "sources").mkdir(parents=True, exist_ok=True)
    (out_corpus / "meta" / "encodings").mkdir(parents=True, exist_ok=True)
    (out_corpus / "blobs" / "sources").mkdir(parents=True, exist_ok=True)
    (out_corpus / "blobs" / "encodings").mkdir(parents=True, exist_ok=True)

    ref_hash_cache: dict[str, str] = {}
    result: dict[int, tuple[str, str]] = {}
    n_sources = 0
    n_encodings = 0
    for i in needed_idx:
        ref_path = Path(cols["ref_path"][i])
        rb = cols["ref_basename"][i]
        if rb not in ref_hash_cache:
            h = sha256_file(ref_path)
            ref_hash_cache[rb] = h
            dst = out_corpus / "blobs" / "sources" / f"{h}.png"
            if not dst.exists():
                shutil.copyfile(ref_path, dst)
                w, hgt = Image.open(ref_path).size
                (out_corpus / "meta" / "sources" / f"{h}.json").write_text(
                    json.dumps(
                        {
                            "hash": h,
                            "width": w,
                            "height": hgt,
                            "size_bytes": ref_path.stat().st_size,
                            "corpus": "clean-picker-corpus-2026-06-26",
                            "filename": rb,
                        }
                    )
                )
                n_sources += 1
        source_hash = ref_hash_cache[rb]

        ef = cols["encoded_filename"][i]
        eid = hashlib.sha256(ef.encode()).hexdigest()[:24]
        codec = cols["codec"][i]
        ext = CODEC_EXT.get(codec, "bin")
        enc_dst = out_corpus / "blobs" / "encodings" / f"{eid}.{ext}"
        enc_path = Path(cols["encode_path"][i])
        if not enc_dst.exists():
            shutil.copyfile(enc_path, enc_dst)
            (out_corpus / "meta" / "encodings" / f"{eid}.json").write_text(
                json.dumps(
                    {
                        "id": eid,
                        "source_hash": source_hash,
                        "codec": codec,
                        "quality": float(cols["q"][i]),
                        "effort": None,
                        "encoded_size": enc_path.stat().st_size,
                    }
                )
            )
            n_encodings += 1
        result[i] = (source_hash, eid)
    print(f"[stage] {n_sources} source blobs, {n_encodings} encoding blobs written")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encodes-parquet", required=True, type=Path)
    ap.add_argument("--candidates-spec", required=True, type=Path)
    ap.add_argument("--crop-holdout-tsv", type=Path, default=None)
    ap.add_argument("--crop-holdout-threshold", type=int, default=10)
    ap.add_argument("--out-corpus", required=True, type=Path)
    ap.add_argument("--out-pairs", required=True, type=Path)
    ap.add_argument("--random-control-n", type=int, default=400)
    ap.add_argument("--calibration-n", type=int, default=120)
    ap.add_argument("--repeat-frac", type=float, default=0.12)
    ap.add_argument("--repeat-min-distance", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260901)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    spec = json.loads(a.candidates_spec.read_text())
    candidates = [c["name"] for c in spec["candidates"]]

    exclude_refs = set()
    if a.crop_holdout_tsv and a.crop_holdout_tsv.exists():
        with open(a.crop_holdout_tsv, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if int(r["min_hamming"]) <= a.crop_holdout_threshold:
                    exclude_refs.add(r["source_file"])
        print(f"[mine] crop-holdout excludes {len(exclude_refs)} refs at d<={a.crop_holdout_threshold}")

    t0 = time.time()
    cols, by_ref = load_rows(a.encodes_parquet, exclude_refs)
    print(f"[mine] loaded {len(cols['encoded_filename'])} rows, {len(by_ref)} refs, {time.time()-t0:.0f}s")

    rows = mine_gmad(cols, by_ref, candidates, rng)
    seen = {pair_key(r["a"], r["b"]) for r in rows}
    print(f"[mine] s1_gmad: {len(rows)} pairs")

    rc = mine_random_control(cols, by_ref, a.random_control_n, rng, seen)
    rows.extend(rc)
    print(f"[mine] s1_random_control: {len(rc)} pairs")

    lad = mine_ladder(cols, by_ref, candidates, rng)
    for r in lad:
        seen.add(pair_key(r["a"], r["b"]))
    rows.extend(lad)
    print(f"[mine] s2_ladder: {len(lad)} pairs")

    cal = mine_calibration(cols, by_ref, candidates, a.calibration_n, rng, seen)
    rows.extend(cal)
    print(f"[mine] s3_calibration: {len(cal)} pairs")

    rng.shuffle(rows)  # planned SEQ order is randomized once, then fixed forever
    rows = add_repeats(rows, a.repeat_frac, a.repeat_min_distance, rng)
    n_repeats = sum(1 for r in rows if r["stratum"].endswith("_repeat"))
    print(f"[mine] repeats: {n_repeats}")
    print(f"[mine] TOTAL planned rows: {len(rows)}")

    needed_idx = {r["a"] for r in rows} | {r["b"] for r in rows}
    print(f"[mine] staging corpus for {len(needed_idx)} unique encode rows...")
    idmap = stage_corpus(cols, needed_idx, a.out_corpus)

    pair_ids = []
    for seq, r in enumerate(rows):
        sh_a, eid_a = idmap[r["a"]]
        sh_b, eid_b = idmap[r["b"]]
        assert sh_a == sh_b, "a/b must share one source"
        pid = stable_pair_id(cols, r["a"], r["b"], r["stratum"] + f"_{seq}")
        pair_ids.append(pid)

    a.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out_pairs, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "pair_id",
                "seq",
                "source_hash",
                "a_encoding_id",
                "b_encoding_id",
                "stratum",
                "repeat_of_pair",
                "expected_choice",
                "meta_json",
            ]
        )
        for seq, r in enumerate(rows):
            sh_a, eid_a = idmap[r["a"]]
            _, eid_b = idmap[r["b"]]
            repeat_of = pair_ids[r["repeat_of_seq"]] if "repeat_of_seq" in r else ""
            w.writerow(
                [
                    pair_ids[seq],
                    seq,
                    sh_a,
                    eid_a,
                    eid_b,
                    r["stratum"],
                    repeat_of,
                    r.get("expected_choice", ""),
                    json.dumps(r["meta"]),
                ]
            )

    strata_counts = defaultdict(int)
    for r in rows:
        strata_counts[r["stratum"]] += 1
    manifest = {
        "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": a.seed,
        "candidates": candidates,
        "tau_sim": TAU_SIM,
        "ladder_material": LADDER_MATERIAL,
        "qzones": QZONES,
        "encodes_parquet": str(a.encodes_parquet),
        "crop_holdout_excluded_refs": len(exclude_refs),
        "n_rows_total": len(rows),
        "n_repeats": n_repeats,
        "n_unique_encode_rows_staged": len(needed_idx),
        "n_unique_references": len({cols["ref_basename"][i] for i in needed_idx}),
        "strata_counts": dict(sorted(strata_counts.items())),
        "out_corpus": str(a.out_corpus),
        "out_pairs": str(a.out_pairs),
    }
    (a.out_pairs.parent / "_MINING_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["strata_counts"], indent=1))
    print(f"[mine] DONE {time.time()-t0:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
