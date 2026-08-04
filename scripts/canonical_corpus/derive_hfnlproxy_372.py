#!/usr/bin/env python3
"""derive_hfnlproxy_372.py — the 372-col (era-native) hfnlproxy eval slice for
era-bridge bakes, derived by EXACT KEY JOIN — no extraction, no decode.

Board-integrity pass 2026-08-04 (campaign doc §8 CORRECTIONS / fill note): the
hfnlproxy corpus is 944-root-only, so every era-bridge fulleval has
`rank.hfnlproxy` absent (absent-not-failed in benchmarks/eval_annotations.json).
The SAME cells exist at 372 features: the 944 TEST views were built keyed
`encoded_filename` from the canonical-picker-2026-06-27 TEST splits
(match_rate 1.0000), which carry `feat_0..feat_371` (the v1-372 with-iw space —
exactly the era bakes' native regime; 720 = v1-372 ++ v2-348, so any bake with
n_inputs <= 372 reads a 372-wide table under the width-dynamic loader).

Derivation (all gates hard-fail):
 1. Rebuild the hfnlproxy row selection EXACTLY as build_eval_slices_944.py
    (same VIEWS order, ssim2>=91 mask, refs with >=6 cells, global stride to
    ~10k rows) — but keeping `encoded_filename`.
 2. IDENTITY GATE: (ref_basename, human_score) must equal the committed 944
    slice row-for-row (float-exact).
 3. Join encoded_filename -> canonical-2026-06-27/<view>/test.parquet, pull
    feat_0..feat_371; GATE: per-row score_ssim2 float-exact on both sides.
 4. Write ext_hfnlproxy.parquet (ref_basename, human_score, f0..f371, f64,
    zstd) into the 720 features root + a _MANIFEST sidecar with build commit +
    source shas.

human_score is copied from the 944 view (score_ssim2 x 0.01) — the TARGET is
identical across eras; only the feature regime differs. per_ref SROCC on this
slice is therefore era-comparable by construction.
"""

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc  # noqa: F401  (parity with the 944 builder's imports)
import pyarrow.parquet as pq

VIEWS = ["zenavif_lossy", "zenjpeg_lossy", "zenjxl_lossy", "zenwebp_lossy"]
TARGET_ROWS = 10000
V944_ROOT = Path("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01")
CANON_0627 = Path("/mnt/v/output/canonical-picker-2026-06-27")
OUT_ROOT = Path("/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22")
COMMITTED_944_SLICE = V944_ROOT / "ext_hfnlproxy.parquet"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    # 1. Rebuild the selection over the 4 views, carrying encoded_filename.
    cols = ["ref_filename", "encoded_filename", "score_ssim2"]
    tabs, prov = [], []
    for ds in VIEWS:
        p = V944_ROOT / "bigcodec" / ds / "test_944.parquet"
        t = pq.read_table(p, columns=cols)
        tabs.append(t)
        prov.append({"view": str(p), "sha256": sha256_file(p), "rows": t.num_rows})
        print(f"  {ds}: test rows {t.num_rows}", flush=True)
    full = pa.concat_tables(tabs)
    refs = np.array(full["ref_filename"].to_pylist())
    ssim2 = np.asarray(full["score_ssim2"].combine_chunks(), dtype=np.float64)

    m_hf = ssim2 >= 91.0
    import collections
    cnt = collections.Counter(refs[m_hf].tolist())
    good_refs = {r for r, c in cnt.items() if c >= 6}
    m_hf &= np.isin(refs, sorted(good_refs))
    idx_all = np.nonzero(m_hf)[0]
    step = max(1, len(idx_all) // TARGET_ROWS)
    idx = idx_all[::step]
    print(f"hfnlproxy pool: {len(idx_all)} rows / {len(good_refs)} refs; stride {step} -> {len(idx)} rows")

    sel = full.take(pa.array(idx))
    sel_refs = sel["ref_filename"].to_pylist()
    sel_enc = sel["encoded_filename"].to_pylist()
    sel_hs = (np.asarray(sel["score_ssim2"].combine_chunks(), dtype=np.float64) * 0.01)

    # 2. IDENTITY GATE vs the committed 944 slice.
    ref944 = pq.read_table(COMMITTED_944_SLICE, columns=["ref_basename", "human_score"])
    assert ref944.num_rows == len(idx), \
        f"row count {len(idx)} != committed {ref944.num_rows}"
    r944 = ref944["ref_basename"].to_pylist()
    h944 = np.asarray(ref944["human_score"].combine_chunks(), dtype=np.float64)
    assert r944 == sel_refs, "ref_basename sequence mismatch vs committed 944 slice"
    assert np.array_equal(h944, sel_hs), "human_score sequence mismatch (float-exact) vs committed"
    print(f"IDENTITY GATE PASS: {len(idx)} rows == committed ext_hfnlproxy.parquet (refs + human_score float-exact)")

    # 3. Join encoded_filename -> 06-27 canonical test splits; pull feat_0..371.
    feat_cols = [f"feat_{i}" for i in range(372)]
    lut = {}
    for ds in VIEWS:
        p = CANON_0627 / ds / "test.parquet"
        t = pq.read_table(p, columns=["encoded_filename", "score_ssim2"] + feat_cols)
        enc = t["encoded_filename"].to_pylist()
        pos = {k: i for i, k in enumerate(enc)}
        lut[ds] = (t, pos)
        prov.append({"canon": str(p), "sha256": sha256_file(p), "rows": t.num_rows})
        print(f"  canon {ds}: {t.num_rows} rows", flush=True)

    take_ds, take_i = [], []
    for k in sel_enc:
        for ds in VIEWS:
            i = lut[ds][1].get(k)
            if i is not None:
                take_ds.append(ds)
                take_i.append(i)
                break
        else:
            raise SystemExit(f"JOIN GATE FAIL: {k} not found in any 06-27 test split")
    assert len(take_i) == len(idx)

    feats = np.empty((len(idx), 372), dtype=np.float64)
    canon_ssim2 = np.empty(len(idx), dtype=np.float64)
    for ds in VIEWS:
        rows = [j for j, d in enumerate(take_ds) if d == ds]
        if not rows:
            continue
        t = lut[ds][0]
        sub = t.take(pa.array([take_i[j] for j in rows]))
        canon_ssim2[rows] = np.asarray(sub["score_ssim2"].combine_chunks(), dtype=np.float64)
        for c in range(372):
            feats[rows, c] = np.asarray(sub[feat_cols[c]].combine_chunks(), dtype=np.float64)
    view_ssim2 = np.asarray(sel["score_ssim2"].combine_chunks(), dtype=np.float64)
    assert np.array_equal(canon_ssim2, view_ssim2), \
        "SSIM2 GATE FAIL: 06-27 score_ssim2 != 944-view score_ssim2 (float-exact required)"
    print("JOIN + SSIM2 GATES PASS: all rows joined, per-row score_ssim2 float-exact")

    # 4. Write the 372-col slice (ext-leg convention: ref_basename, human_score, f0..fN).
    out_cols = {"ref_basename": pa.array(sel_refs), "human_score": pa.array(sel_hs)}
    for c in range(372):
        out_cols[f"f{c}"] = pa.array(feats[:, c])
    out = OUT_ROOT / "ext_hfnlproxy.parquet"
    pq.write_table(pa.table(out_cols), out, compression="zstd", compression_level=7)

    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent.parent),
         "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or "unknown"
    manifest = {
        "description": "372-col (era-native) hfnlproxy slice — SAME cells + SAME "
                       "human_score as the committed 944 slice (identity-gated row-for-row), "
                       "features exact-key-joined (encoded_filename) from "
                       "canonical-picker-2026-06-27 TEST splits. Built for era-bridge "
                       "bake_verdict hfnlproxy fill (board-integrity pass; campaign doc §8 "
                       "CORRECTIONS). n_inputs<=372 era bakes only.",
        "build_commit": commit,
        "committed_944_slice": {"path": str(COMMITTED_944_SLICE),
                                "sha256": sha256_file(COMMITTED_944_SLICE)},
        "rows": len(idx),
        "distinct_refs": len(set(sel_refs)),
        "human_score": "score_ssim2 x 0.01 (copied from the 944 view — identical target)",
        "sources": prov,
        "output": {"path": str(out), "sha256": sha256_file(out)},
    }
    mp = OUT_ROOT / "ext_hfnlproxy._MANIFEST.json"
    mp.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out}\nwrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
