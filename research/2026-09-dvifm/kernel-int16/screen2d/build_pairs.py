#!/usr/bin/env python3
"""Phase-2d Part D — build the registered pairs TSVs.

Emits `ref_path\tdist_path\thuman_score` TSVs under $OUT/pairs/ plus a
manifest per file. Score conventions follow the lineage pairs files
(CID22 MCOS/100, TID MOS/9, KonFiG human_score as in konfig_pairs.tsv);
the fitter's recorded per-domain normalisation maps them to the 0..100
quality axis (prereg §5.1).

KADID CORRECTION (prereg Amendment 1, 2026-09-19): the lineage
`kadid_pairs_ab.tsv` stores `(5 - dmos)/4`, but kadid10k's `dmos.csv`
column is QUALITY-oriented (mean DCR falls 4.43 -> 1.38 across levels
1->5), so that convention was exactly inverted (corr -1.0 vs the source
column). All KADID rows here are re-derived from `dmos.csv` as the
canonical `(dmos - 1)/4`; the upstream value is asserted to equal
`(5 - dmos)/4` per row so the inversion is documented, not propagated.

CID22-B is emitted with `human_score` = 0.0 — a placeholder: B MCOS is
SEALED until the single frozen read (prereg §5.4). The extractor does
not consume the score column for block caches.

Splits (committed in docs/DATASET_HISTORY.md 2026-09-19 + DATA_SPLITS.md
exposure ledger, seed 20260919):
  CID22-A = 25 refs listed in the ledger; CID22-B = the other 24.
  KADID train refs = last digit in {0,2,4,6,8}; dev = {1,3,5};
  {7,9} untouched.
  TID2013: all 25 refs, types 10 (JPEG) + 11 (JPEG2000) only.
  KADID: types 10 (JPEG) + 09 (JPEG2000) only.
  KonFiG originsplit-val: sources {SRC01,SRC03,SRC31,SRC45} (the 2b eval
  view; TEST sources {SRC07,SRC09,SRC17} untouched).
"""
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

OUT = Path("/mnt/v/output/zensim/dvifm-screen2d-2026-09-19")
PAIRS = OUT / "pairs"

CID22_VAL = Path("/mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv")
CID22_CSV = Path("/mnt/v/dataset/cid22/CID22_validation_set/CID22_validation_set.csv")
TID_PAIRS = Path("/mnt/v/dataset/tid2013/tid_pairs_ab.tsv")
KADID_PAIRS = Path("/mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv")
KONFIG_PAIRS = Path("/mnt/v/output/zensim/konfig944/build/konfig_pairs.tsv")

# Committed ledger lists (docs/DATASET_HISTORY.md, 2026-09-19, seed 20260919).
CID22_A = {
    "1189261.png", "1531677.png", "159550.png", "1624487.png", "162520.png",
    "164595.png", "2079234.png", "21169144185_3f7977cb5a_o.png", "225228.png",
    "2389166.png", "2936831.png", "3316926.png", "3653963.png", "373965.png",
    "3762075.png", "4215100.png", "6078297.png", "6292444.png", "70497.png",
    "7062219.png", "844297.png", "pexels-photo-2686358.png",
    "pexels-photo-2802032.png", "pexels-photo-4210863.png",
    "ularapi_Semarang_City_Logo.png",
}
CID22_B = {
    "1025469.png", "1044329.png", "1279330.png", "1418519.png", "1420710.png",
    "1475938.png", "1544947.png", "2190188.png", "2253934.png", "2670327.png",
    "2736139.png", "2775196.png", "2887497.png", "297394.png", "3156482.png",
    "3316926_opo25u.png", "3637739.png", "382297.png", "5055743.png",
    "5458393.png", "7552578.png", "792079.png", "adriankierman-report-page.png",
    "pexels-photo-1933873.png",
}
KONFIG_VAL_SOURCES = {"SRC01", "SRC03", "SRC31", "SRC45"}
KONFIG_TEST_SOURCES = {"SRC07", "SRC09", "SRC17"}  # never emitted


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_tsv(name, rows, extra_cols=(), note=""):
    out = PAIRS / f"{name}.tsv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score", *extra_cols])
        for r in rows:
            w.writerow([r["ref_path"], r["dist_path"], r["human_score"],
                        *(r[c] for c in extra_cols)])
    manifest = {
        "file": str(out),
        "sha256": sha(out),
        "rows": len(rows),
        "note": note,
    }
    (PAIRS / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{name}: {len(rows)} rows -> {out.name}")
    return manifest


def load_tsv(p):
    with open(p) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def ref_base(r):
    return Path(r["ref_path"]).name


def dist_base(r):
    return Path(r["dist_path"]).name


def main():
    PAIRS.mkdir(parents=True, exist_ok=True)
    prov = {"built_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "sources": {}}

    # --- CID22 A/B -----------------------------------------------------
    cid = load_tsv(CID22_VAL)
    prov["sources"]["cid22val_pairs_ab.tsv"] = sha(CID22_VAL)
    refs_in = {ref_base(r) for r in cid}
    assert CID22_A | CID22_B == refs_in, (
        f"ledger refs != TSV refs: only-ledger={sorted((CID22_A|CID22_B)-refs_in)} "
        f"only-tsv={sorted(refs_in-(CID22_A|CID22_B))}")
    assert not (CID22_A & CID22_B)
    rows_a = [r for r in cid if ref_base(r) in CID22_A]
    rows_b = [dict(r, human_score="0") for r in cid if ref_base(r) in CID22_B]
    write_tsv("cid22a", rows_a,
              note="CID22-A (25 refs, ledger seed 20260919); human_score=MCOS/100 "
                   "from cid22val_pairs_ab.tsv. FIT-ALLOWED for Part D constants.")
    write_tsv("cid22b", rows_b,
              note="CID22-B (24 refs) — human_score BLANKED to 0 (MCOS sealed "
                   "until the single frozen read, prereg 5.4). Extraction only.")
    prov["cid22a_refs"] = sorted(CID22_A)
    prov["cid22b_refs"] = sorted(CID22_B)

    # --- TID2013 JPEG(10)+JP2K(11) ------------------------------------
    tid = load_tsv(TID_PAIRS)
    prov["sources"]["tid_pairs_ab.tsv"] = sha(TID_PAIRS)
    tid_rows = [r for r in tid
                if re.search(r"_(10|11)_\d\.png$", dist_base(r), re.IGNORECASE)]
    assert len(tid_rows) == 250, len(tid_rows)
    write_tsv("tid_jp2kjpeg", tid_rows,
              note="TID2013 types 10(JPEG)+11(JPEG2000), all 25 refs x 5 levels; "
                   "human_score=MOS/9 from tid_pairs_ab.tsv. TRAIN-only per "
                   "2026-08-29 ruling.")

    # --- KADID JPEG(10)+JP2K(09) --------------------------------------
    kad = load_tsv(KADID_PAIRS)
    prov["sources"]["kadid_pairs_ab.tsv"] = sha(KADID_PAIRS)
    # Amendment-1 correction: re-derive from dmos.csv (quality-oriented
    # mean-DCR column). Assert the upstream (5-dmos)/4 inversion per row.
    with open("/mnt/v/dataset/kadid10k/dmos.csv") as _f:
        kadid_dmos = {r["dist_img"]: float(r["dmos"])
                      for r in csv.DictReader(_f)}
    prov["sources"]["kadid10k/dmos.csv"] = sha(
        "/mnt/v/dataset/kadid10k/dmos.csv")

    def kad_fix(r):
        b = dist_base(r)
        d = kadid_dmos[b]
        assert abs(float(r["human_score"]) - (5.0 - d) / 4.0) < 1e-9, b
        return dict(r, human_score=(d - 1.0) / 4.0)

    def kad_sel(r, digits):
        m = re.match(r"I(\d\d)_(\d\d)_\d\d\.png$", dist_base(r))
        if not m or m.group(2) not in {"09", "10"}:
            return False
        return int(m.group(1)) % 10 in digits

    ktr = [kad_fix(r) for r in kad if kad_sel(r, {0, 2, 4, 6, 8})]
    kdv = [kad_fix(r) for r in kad if kad_sel(r, {1, 3, 5})]
    write_tsv("kadid_train", ktr,
              note="KADID-10k types 09(JP2K)+10(JPEG), refs last-digit "
                   "{0,2,4,6,8}; human_score=(dmos-1)/4 re-derived from "
                   "dmos.csv (quality-oriented source column; Amendment-1 "
                   "corrects the upstream (5-dmos)/4 inversion). TRAIN.")
    write_tsv("kadid_dev", kdv,
              note="KADID-10k types 09(JP2K)+10(JPEG), refs last-digit "
                   "{1,3,5}; human_score=(dmos-1)/4 re-derived from "
                   "dmos.csv (Amendment-1 correction). TRAIN-side "
                   "development eval. Digits {7,9} untouched.")

    # --- KonFiG originsplit-val ----------------------------------------
    kon = load_tsv(KONFIG_PAIRS)
    prov["sources"]["konfig_pairs.tsv"] = sha(KONFIG_PAIRS)
    kv = [r for r in kon if r["source"] in KONFIG_VAL_SOURCES]
    assert not any(r["source"] in KONFIG_TEST_SOURCES for r in kv)
    assert len(kv) == 436, len(kv)
    write_tsv("konfig_val", kv, extra_cols=("source",),
              note="KonFiG originsplit VAL sources {SRC01,SRC03,SRC31,SRC45} "
                   "(the 2b eval view); human_score as konfig_pairs.tsv. "
                   "TRAIN-side development. TEST sources untouched.")

    (OUT / "pairs" / "_MANIFEST.json").write_text(
        json.dumps(prov, indent=2) + "\n")
    print("manifest -> pairs/_MANIFEST.json")


if __name__ == "__main__":
    sys.exit(main())
