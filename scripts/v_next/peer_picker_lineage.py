#!/usr/bin/env python3
"""Extend the peer rows to the picker-lineage axes (imazen26 / nonphoto /
hfnlproxy) from STORED sidecars — no image is rescored.

Method: reproduce build_eval_slices_944's row selection EXACTLY (same views,
concat order, masks, stride) while carrying `encoded_filename`; apply the
family-aware reslice keep-rule (the same sets apply_d1_exclusion documents);
IDENTITY-GATE the reproduction against the shipped slice bytes (row count AND
human_score row-for-row); then join score_butteraugli / score_cvvdp /
score_iwssim / score_dssim from the fill4 7-metric sidecar (+ the jxl hqfill
top-up) by encoded_filename.

Semantics (stated, per the 2026-08-28 column audit): these axes' target
(human_score) is SSIM2-DERIVED (imazen26 = real-codec ssim2 ×1.0, nonphoto/
hfnlproxy = ssim2/100). So the ssim2 peer is TARGET-IDENTICAL (srocc 1.0 by
construction — included with that footnote, per "don't skip any"), and the
butter/cvvdp/iwssim peers measure agreement with the axis's ssim2 target —
which is what the axis IS (the broad ssim2-agreement axis).
"""
import importlib.util, json, os, sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "lib"))
from zen_stats import panel, panel_batch  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "bes944", os.path.join(os.path.dirname(__file__), "..", "canonical_corpus",
                           "build_eval_slices_944.py"))
bes = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bes)

VROOT = "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec"
SROOT = "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01"
MANIFEST = "/mnt/v/output/imazen-26-features/imazen26_manifest.tsv"
SIDECARS = ["/mnt/v/datasets/fill4-6codec-2026-07-01/fill4metrics_sidecar_patched_2026-07-02.parquet",
            "/mnt/v/output/zensim-multicodec-probe/hqfill_7metric_sidecar_2026-07-02.parquet",
            "/mnt/v/datasets/fill4-6codec-2026-07-01/hfnl_avifgap_4metric_sidecar_2026-08-05.parquet"]
OUT = "/mnt/v/output/zensim/reports/fulleval"
FAMILY = "/home/lilith/work/imazen-26/manifests/split_map_family.tsv"
CHA_TSV = "/mnt/v/output/zensim/imazen26-dhash-2026-08-27/prov_poolshare.tsv"

def main():
    import csv
    cls_of = {}
    for row in csv.DictReader(open(MANIFEST), delimiter="\t"):
        cls_of[int(row["stem"])] = row["content_class"]
    fam = {r["id"]: r["split"] for r in csv.DictReader(open(FAMILY), delimiter="\t")}
    cha = {r["id"] for r in csv.DictReader(open(CHA_TSV), delimiter="\t")}

    cols = ["ref_filename", "score_ssim2", "encoded_filename"]
    tabs = [pq.read_table(f"{VROOT}/{ds}/test_944.parquet", columns=cols) for ds in bes.VIEWS]
    full = pa.concat_tables(tabs)
    refs = full["ref_filename"].to_pylist()
    encs = full["encoded_filename"].to_pylist()
    ssim2 = np.asarray(full["score_ssim2"].combine_chunks(), dtype=np.float64)
    origins = np.array([bes.origin_of(r) for r in refs])
    classes = np.array([cls_of.get(o, "?") for o in origins])

    def keep_reslice(o):
        so = f"{o:04d}"
        if so in cha:
            return False
        b = fam.get(so)
        return b is None or b == "test"

    def select(mask, scale, slice_file):
        idx_all = np.nonzero(mask)[0]
        step = max(1, len(idx_all) // bes.TARGET_ROWS)
        idx = idx_all[::step]
        keep = np.array([keep_reslice(origins[i]) for i in idx])
        idx = idx[keep]
        t = pq.read_table(os.path.join(SROOT, slice_file), columns=["ref_basename", "human_score"])
        assert t.num_rows == len(idx), f"{slice_file}: repro {len(idx)} != slice {t.num_rows}"
        hs = np.asarray(t["human_score"].combine_chunks(), dtype=np.float64)
        rep = ssim2[idx] * scale
        assert np.max(np.abs(rep - hs)) < 1e-9, f"{slice_file}: human_score mismatch"
        return idx

    m_np = np.isin(classes, sorted(bes.NONPHOTO_CLASSES))
    cnt = {}
    m_hf = ssim2 >= 91.0
    for r, m in zip(refs, m_hf):
        if m: cnt[r] = cnt.get(r, 0) + 1
    good = {r for r, c in cnt.items() if c >= 6}
    m_hf &= np.isin(np.array(refs), sorted(good))

    axes = {
        "imazen26": select(np.ones(len(refs), bool), 1.0, "ext_imazen26.parquet"),
        "nonphoto": select(m_np, 0.01, "ext_nonphoto.parquet"),
        "hfnlproxy": select(m_hf, 0.01, "ext_hfnlproxy.parquet"),
    }
    print("identity gates PASS on all three axes (row count + human_score row-for-row)")

    side = {}
    COLMAPS = [  # (butteraugli, cvvdp, iwssim) per sidecar schema generation
        ("score_butteraugli", "score_cvvdp", "score_iwssim"),
        ("score_butteraugli_max_gpu", "score_cvvdp_imazen_v0_0_1", "score_iwssim_gpu"),
    ]
    for sp in SIDECARS:
        names = set(pq.ParquetFile(sp).schema_arrow.names)
        cm = next((m for m in COLMAPS if set(m) <= names), None)
        assert cm, f"no known column map for {sp}"
        st = pq.read_table(sp, columns=["encoded_filename", *cm])
        for e, b, c, iw in zip(st["encoded_filename"].to_pylist(), st[cm[0]].to_pylist(),
                               st[cm[1]].to_pylist(), st[cm[2]].to_pylist()):
            side[e] = (b, c, iw)
    print(f"sidecar entries: {len(side)}")

    PEERS = {"butteraugli": (0, -1), "cvvdp": (1, +1), "iwssim": (2, +1)}
    for ax, idx in axes.items():
        tgt = ssim2[idx]
        found = [side.get(encs[i]) for i in idx]
        cov = sum(1 for f in found if f is not None)
        print(f"{ax}: sidecar coverage {cov}/{len(idx)} ({100*cov/len(idx):.1f}%)")
        for peer, (fi, sign) in PEERS.items():
            xs, ys = [], []
            for f, t in zip(found, tgt):
                if f is None or f[fi] is None: continue
                xs.append(sign * float(f[fi])); ys.append(float(t))
            if len(xs) < 100:
                print(f"   {peer}: insufficient ({len(xs)}) — skipped"); continue
            st = panel(xs, ys)
            sb = panel_batch([("p", xs, ys)])[0]
            jp = os.path.join(OUT, f"peer_{peer}.fulleval.json")
            doc = json.load(open(jp))
            import random as _r
            _r.seed(11)
            k = min(5000, len(xs))
            samp = sorted(_r.sample(range(len(xs)), k))
            doc.setdefault("per_pair", {})[ax] = {"pred": [xs[i] for i in samp], "mos": [ys[i] for i in samp]}
            doc["rank"][ax] = {"srocc": st["srocc"], "srocc_signed": sb.get("srocc_signed"),
                               "plcc": st["plcc"], "krocc": st["krocc"], "or": st["or"],
                               "pwrc": st["pwrc"], "z_rmse": st["z_rmse"], "n": st["n"]}
            doc.setdefault("peer_provenance", {})[ax] = {
                "source": "fill4 7-metric sidecar join via reproduced slice selection (identity-gated)",
                "n": st["n"], "coverage": f"{cov}/{len(idx)}",
                "target_semantics": "ssim2-derived (the axis's definition)",
                "oriented": "negated" if sign < 0 else "as-is"}
            json.dump(doc, open(jp, "w"), indent=1)
            print(f"   {peer}: srocc {st['srocc']:.4f} (n={st['n']}) -> written")
        # ssim2 self-target row: 1.0 by construction, included per "don't skip any"
        jp = os.path.join(OUT, "peer_ssim2.fulleval.json")
        doc = json.load(open(jp))
        doc["rank"][ax] = {"srocc": 1.0, "srocc_signed": 1.0, "plcc": 1.0, "n": int(len(idx))}
        doc.setdefault("peer_provenance", {})[ax] = {
            "self_target": True,
            "note": "this axis's target IS ssim2-derived — srocc 1.0 by construction, not a measurement"}
        json.dump(doc, open(jp, "w"), indent=1)
        print(f"   ssim2: self-target 1.0 (footnoted)")

if __name__ == "__main__":
    main()
