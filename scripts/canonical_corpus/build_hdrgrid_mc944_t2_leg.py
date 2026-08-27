#!/usr/bin/env python3
"""Build the HDR-944 L1-T2 training leg: hdrgrid MULTI-CODEC x 944 feats x
era-B zensim target (wave registration + scheduling amendment:
benchmarks/hdr944_bake_wave_2026-08-27.md — T2's inputs are complete+final).

- features: the three harvested feat944 parquets (zenjxl/svt/gm; feat_0..feat_943,
  encode_sha-keyed) — hdrfeat944-* runs, Folded720Append2, HdrEncoding::Linear.
- target: human_score = era-B zensim/100 from zensim_scores_by_judge_era.parquet
  (judge era B-9dffa5ca ONLY — never the mixed harvest column).
- join: on bare encode_sha; duplicate keys hard-error per side.
- census hygiene: the 9 HDR-instrument SCENES excluded entirely (all scales).
- split: origin_split.split_of on ref_basename (THE owner); train+val written,
  test digits held out (counted only).
usage: build_hdrgrid_mc944_t2_leg.py [--out DIR]
"""
import argparse, hashlib, json, os, sys, datetime, subprocess
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

sys.path.insert(0, os.path.expanduser("~/work/zen/zenmetrics/scripts/picker"))
from origin_split import split_of

FEAT = "/mnt/v/output/hdrgrid-2026-08-06/harvest-feat944-2026-08-26/%s/features_folded720append2.parquet"
ERA = "/mnt/v/output/hdrgrid-2026-08-06/zensim_scores_by_judge_era.parquet"
CENSUS_TSV = os.path.expanduser("~/work/zen/zensim/benchmarks/hdr_instrument_refs_2026-08-27.tsv")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/mnt/v/zen/zensim-training/hdrgrid-mc944-t2-2026-08-27")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    census_scenes = set()
    for line in open(CENSUS_TSV):
        line = line.strip()
        if not line or line.startswith(("#", "scene\t")):
            continue
        census_scenes.add(line.split("\t")[0])

    era = pq.read_table(ERA)
    bare = lambda s: s.rsplit("/", 1)[-1]
    tgt = {}
    for sha, cod, z, e in zip(*[era[c].to_pylist() for c in ["encode_sha", "codec", "zensim_score", "judge_era"]]):
        if not e.startswith("B-"):
            continue
        k = bare(sha)
        if k in tgt:
            raise SystemExit(f"duplicate era-B encode_sha {k}")
        tgt[k] = z

    parts, counts = [], {}
    for cdir in ["zenjxl", "svt", "gm"]:
        t = pq.read_table(FEAT % cdir)
        shas = [bare(s) for s in t["encode_sha"].to_pylist()]
        # Byte-identical encodes fan many cells onto one sha (content-address
        # design); their features are identical by construction. Dedup per sha
        # with an equality ASSERT (the avifgen-fill pattern), keep first.
        z = t["zensim_score"].to_pylist()
        f0 = t["feat_0"].to_pylist()
        seen = {}
        dedup_keep = []
        for i, sh in enumerate(shas):
            if sh in seen:
                j = seen[sh]
                if not (z[i] == z[j] and f0[i] == f0[j]):
                    raise SystemExit(f"{cdir}: sha {sh} rows DIFFER (i={i},j={j}) — not a byte-fan")
                continue
            seen[sh] = i
            dedup_keep.append(i)
        t = t.take(dedup_keep)
        shas = [bare(s) for s in t["encode_sha"].to_pylist()]
        ips = t["image_path"].to_pylist()
        keep, hs = [], []
        for i, (s, ip) in enumerate(zip(shas, ips)):
            scene = bare(ip).split(".scale")[0]
            if scene in census_scenes or s not in tgt:
                continue
            keep.append(i)
            hs.append(tgt[s] / 100.0)
        sub = t.take(keep)
        sub = sub.append_column("human_score", pa.array(hs, pa.float64()))
        sub = sub.append_column("ref_basename", pc.binary_join_element_wise(
            pa.array([bare(ip) for ip in pc.take(t["image_path"], pa.array(keep)).to_pylist()]), ""))
        parts.append(sub)
        counts[cdir] = sub.num_rows
    full = pa.concat_tables(parts)

    split = [split_of(rb) for rb in full["ref_basename"].to_pylist()]
    masks = {"train": [s == "train" for s in split], "val": [s in ("val", "validate") for s in split]}
    n_test = sum(1 for s in split if s not in ("train", "val", "validate"))
    outs = {}
    for name, m in masks.items():
        tt = full.filter(pa.array(m))
        p = os.path.join(a.out, f"hdrgrid_mc944_t2_{name}.parquet")
        pq.write_table(tt, p, compression="zstd")
        outs[name] = {"path": p, "rows": tt.num_rows,
                      "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
        print(name, tt.num_rows, "->", p)

    build_commit = subprocess.run(["git", "-C", os.path.expanduser("~/work/zen/zensim"),
                                   "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = {
        "leg": "hdrgrid-mc944-T2 (HDR-944 L1, era-B zensim target)",
        "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "build_commit": build_commit,
        "features": "hdrfeat944-{zenjxl,svt,gm} harvested parquets (Folded720Append2, HdrEncoding::Linear)",
        "target": "human_score = era-B zensim/100 (judge era B-9dffa5ca ONLY; era table sha recorded in plan)",
        "census_scenes_excluded": sorted(census_scenes),
        "per_codec_rows": counts,
        "test_rows_held_out": n_test,
        "outputs": outs,
        "split_owner": "origin_split.split_of(ref_basename)",
    }
    with open(os.path.join(a.out, "_MANIFEST.json"), "w") as f:
        json.dump(man, f, indent=1)
    print("manifest written; per-codec:", counts, "test held:", n_test)

if __name__ == "__main__":
    main()
