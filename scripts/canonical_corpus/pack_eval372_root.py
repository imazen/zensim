#!/usr/bin/env python3
"""Pack the fresh 372-col CSVs into a DATED eval root + `_MANIFEST.json`.

Companion to `build_eval372_root.sh`. For every corpus in `bake_verdict`'s
default CORPORA list it either (a) writes the fresh HEAD extraction, or
(b) copies the stored file when the corpus cannot be re-extracted on this box —
and in BOTH cases records the era, the row accounting and the sha256 in
`_MANIFEST.json`, so no reader has to guess which extractor produced a table.

File names are deliberately the OLD ones: `bake_verdict` hardcodes each
corpus's filename, so a drop-in root must reuse them. The ROOT carries the date.

Usage: pack_eval372_root.py [OUT_ROOT] [WORK_CSV_DIR]
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

OUT = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/zen/zensim-training/2026-08-30-full-features-372"
WORK = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/tmp/eval372root")
OLD = "/mnt/v/zen/zensim-training/2026-05-15-full-features"

# (corpus, bake_verdict filename, source): source = "fresh:<csv stem>" or "stored"
JOBS = [
    ("cid22",  "cid22_features_372col_2026-05-15.parquet",   "fresh:cid22"),
    ("kadid",  "kadid_features_372col_2026-05-15.parquet",   "fresh:kadid"),
    ("tid",    "tid_features_372col_2026-05-15.parquet",     "fresh:tid"),
    ("csiq",   "csiq_features_372col_2026-07-18.parquet",    "fresh:csiq"),
    ("live",   "live_features_372col_2026-07-18.parquet",    "fresh:live"),
    ("pipal",  "pipal_features_372col_2026-07-18.parquet",   "fresh:pipal"),
    ("konjnd", "konjnd_features_372col_2026-05-15.parquet",  "fresh:konjnd"),
    ("aic3",   "aic3_features_372col_2026-05-15.parquet",    "fresh:aic3"),
    # NOT re-extractable on this box — copied, era-stamped, never silently mixed.
    ("aic4",     "aic4_features_372col_2026-05-20.parquet",     "stored"),
    ("nonphoto", "nonphoto_features_372col_2026-07-15.parquet", "stored"),
    ("imazen26", "imazen26_test_120k_2026-07-16.parquet",       "stored"),
    ("sdr25",    "ext_sdr25.parquet",                           "stored"),
    ("hfnlproxy", "ext_hfnlproxy.parquet",                      "stored"),
    ("hf_nearlossless", "hf_nearlossless_val.parquet",          "stored"),
]

EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".avif", ".jxl")


def norm(r):
    rl = r.lower()
    for e in EXTS:
        if rl.endswith(e):
            return r[: -len(e)]
    return r


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_fresh_csv(path):
    t = pacsv.read_csv(path)
    # everything is f64 (the extractors emit plain decimal columns)
    return t.cast(pa.schema([(n, pa.string() if n == "ref_basename" else pa.float64())
                             for n in t.schema.names]))


def graft_extras(fresh: pa.Table, stored_path: str, corpus: str, report: dict) -> pa.Table:
    """Carry non-feature columns the fresh extraction does not produce
    (pipal's 22 metric/mix targets) over from the stored table, keyed on
    (normalized ref_basename, round(human_score, 9)). Refuses on any miss."""
    st = pq.read_table(stored_path)
    feat = {f"f{i}" for i in range(372)}
    extras = [n for n in st.schema.names if n not in feat and n not in ("ref_basename", "human_score")]
    if not extras:
        return fresh
    sd = st.to_pydict()
    idx = {}
    for i, (r, h) in enumerate(zip(sd["ref_basename"], sd["human_score"])):
        idx.setdefault((norm(r), round(float(h), 9)), i)
    fd = fresh.to_pydict()
    rows, miss = [], 0
    for r, h in zip(fd["ref_basename"], fd["human_score"]):
        k = (norm(r), round(float(h), 9))
        if k in idx:
            rows.append(idx[k])
        else:
            miss += 1
            rows.append(None)
    if miss:
        raise SystemExit(f"{corpus}: {miss} fresh rows have no stored row to graft extras from")
    cols = {n: fd[n] for n in fresh.schema.names}
    for e in extras:
        c = sd[e]
        cols[e] = [c[i] for i in rows]
    order = ["ref_basename", "human_score"] + extras + [f"f{i}" for i in range(372)]
    report["grafted_columns"] = extras
    return pa.table({k: cols[k] for k in order})


def drift(stored_path, new_path, tag, outdir):
    """Per-slot era comparison via the drift lane's instrument (NOT re-implemented)."""
    cmp_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drift_cmp.py")
    if not os.path.exists(cmp_py) or not os.path.exists(stored_path):
        return None
    os.makedirs(outdir, exist_ok=True)
    j = os.path.join(outdir, f"{tag}_stored_vs_new_positional.json")
    # POSITIONAL: both tables come from the same loader over the same label file,
    # so row order IS the alignment. Key-based pairing is wrong here — KADID has
    # 64.8 % of rows in repeated (ref, human_score) groups, AIC-3 100 %, TID
    # 24.2 %, and collapsing a group onto its first member compares an image
    # with a different one (and duplicates rows in the aligned table).
    r = subprocess.run([sys.executable, cmp_py, stored_path, new_path, "stored", "new", j,
                        "--positional"], capture_output=True, text=True)
    if r.returncode != 0:
        return {"error": (r.stderr or r.stdout).strip()[-500:]}
    with open(j) as f:
        return json.load(f)


def main():
    os.makedirs(OUT, exist_ok=True)
    man = {
        "description": "DATED 372-col eval root re-extracted with the CURRENT (post-2dab8f30) "
                       "extractor. The 2026-05-15 root's masked/IW block was a function of "
                       "RAYON_NUM_THREADS and does not reproduce at its own build commit — see "
                       "benchmarks/v1_extractor_drift_2026-08-30.md + docs/DATASET_HISTORY.md §3.27. "
                       "The old root is untouched. FILE NAMES ARE THE OLD ONES ON PURPOSE: "
                       "bake_verdict hardcodes each corpus's filename, so a drop-in root must "
                       "reuse them; the ROOT directory carries the date.",
        "built_utc": datetime.now(timezone.utc).isoformat(),
        "build_commit": subprocess.run(["git", "-C", os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "regime": "v1-372 (extended+iw, num_scales=4, blur_passes=1, blur_radius=5, "
                  "masking/iw strength 4.0, Box2x2 downscale)",
        "supersedes": OLD,
        "corpora": {},
    }
    for corpus, fn, src in JOBS:
        dst = os.path.join(OUT, fn)
        rep = {"filename": fn, "source": src}
        stored = os.path.join(OLD, fn)
        if src.startswith("fresh:"):
            csv_path = os.path.join(WORK, src.split(":", 1)[1] + ".csv")
            t = read_fresh_csv(csv_path)
            if corpus == "pipal":
                t = graft_extras(t, stored, corpus, rep)
            pq.write_table(t, dst, compression="zstd")
            rep["era"] = "current (re-extracted at build_commit)"
            rep["rows"] = t.num_rows
            rep["source_csv"] = csv_path
        else:
            shutil.copy2(stored, dst)
            rep["rows"] = pq.ParquetFile(dst).metadata.num_rows
            rep["era"] = "COPIED FROM THE OLD ROOT — not re-extractable on this box"
        rep["sha256"] = sha256(dst)
        rep["bytes"] = os.path.getsize(dst)
        if os.path.exists(stored):
            rep["stored_sha256"] = sha256(stored)
            rep["stored_rows"] = pq.ParquetFile(stored).metadata.num_rows
            if src.startswith("fresh:"):
                d = drift(stored, dst, corpus, os.path.join(OUT, "drift"))
                if d:
                    rep["drift_vs_stored"] = d
        man["corpora"][corpus] = rep
        print(f"{corpus:16s} rows={rep['rows']:6d} {rep['era'][:40]}")
    # kon504 — the JPEG half of KonJND, the registered cross-class kon ruler
    # (`konjnd-372-full-file-dilution-2026-08-29`). Derived from the FRESH
    # konjnd by the stored 504's own keys (KonJND keys are unique: verified).
    kon_stored_504 = os.path.join(OLD, "konjnd_jpeg504_372_2026-08-29.parquet")
    if os.path.exists(kon_stored_504):
        st = pq.read_table(kon_stored_504).to_pydict()
        want = [(norm(r), round(float(h), 9))
                for r, h in zip(st["ref_basename"], st["human_score"])]
        fr = pq.read_table(os.path.join(OUT, "konjnd_features_372col_2026-05-15.parquet"))
        fd = fr.to_pydict()
        idx = {}
        for i, (r, h) in enumerate(zip(fd["ref_basename"], fd["human_score"])):
            k = (norm(r), round(float(h), 9))
            if k in idx:
                raise SystemExit("kon504: fresh konjnd key is NOT unique — refusing")
            idx[k] = i
        rows = [idx[k] for k in want if k in idx]
        if len(rows) != len(want):
            raise SystemExit(f"kon504: only {len(rows)} of {len(want)} stored rows found in fresh")
        sub = fr.take(rows)
        pq.write_table(sub, os.path.join(OUT, "konjnd_jpeg504_372_2026-08-30.parquet"),
                       compression="zstd")
        # ...and as a one-file SIDE ROOT so `bake_verdict --corpora konjnd
        # --features-root <root>/kon504` reads the 504 ruler unchanged.
        os.makedirs(os.path.join(OUT, "kon504"), exist_ok=True)
        pq.write_table(sub, os.path.join(OUT, "kon504",
                                         "konjnd_features_372col_2026-05-15.parquet"),
                       compression="zstd")
        man["kon504"] = {
            "rows": sub.num_rows,
            "derived_from": "konjnd_features_372col_2026-05-15.parquet (this root) by the "
                            "stored 504's (ref_basename, human_score) keys",
            "stored_reference": kon_stored_504,
            "files": {
                "konjnd_jpeg504_372_2026-08-30.parquet":
                    sha256(os.path.join(OUT, "konjnd_jpeg504_372_2026-08-30.parquet")),
                "kon504/konjnd_features_372col_2026-05-15.parquet":
                    sha256(os.path.join(OUT, "kon504",
                                        "konjnd_features_372col_2026-05-15.parquet")),
            },
        }
        print(f"kon504            rows={sub.num_rows} (fresh subset + kon504/ side root)")

    with open(os.path.join(OUT, "_MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("wrote", os.path.join(OUT, "_MANIFEST.json"))


if __name__ == "__main__":
    main()
