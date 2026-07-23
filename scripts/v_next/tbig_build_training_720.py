#!/usr/bin/env python3
"""Assemble the T-big TRAINING corpus at 720: join bigcodec's human_score (ssim2
target) onto the fleet's 720-feature corpus (tbig_720_full).

bigcodec_hqdedup (2.32M rows) = (ref_basename, human_score, f0..f371) — the target
+ v1 features. tbig_720_full (5.74M) = (image_path, f0..f719) — the full features.
No shared encode_sha, so match on (ref, f0..f371). At this scale a full-tuple
fingerprint OOMs, so we use a COMPACT hash key: hash(ref, rounded f0..f10) + keep
f0..f2 for collision verification. Memory-bounded: bigcodec keys are a dict of
ints; tbig is streamed row-group by row-group. Output = (ref_basename,
human_score, f0..f719) training-ready parquet. Unmatched bigcodec rows are dropped
(reported), never fabricated.

Usage: python3 tbig_build_training_720.py <bigcodec.parquet> <tbig_720_full.parquet> <out.parquet>
"""
import sys
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

R = 6            # round decimals for the key
KEYF = 11        # features in the hash key (f0..f10)


def stem(x):
    return str(x).replace(".png", "")


def key_of(ref, feats):
    return hash((ref, tuple(np.round(feats[:KEYF], R).tolist())))


def main():
    bc_p, tb_p, out_p = sys.argv[1], sys.argv[2], sys.argv[3]

    # --- bigcodec side: compact key -> (bc_row_idx, verify f0..f2) ---
    bc = pq.read_table(bc_p, columns=["ref_basename", "human_score"] + [f"f{k}" for k in range(KEYF)])
    brefs = [stem(x) for x in bc.column("ref_basename").to_pylist()]
    bhuman = bc.column("human_score").to_pylist()
    bmat = np.column_stack([bc.column(f"f{k}").to_numpy() for k in range(KEYF)]).astype("f8")
    keymap = {}
    for i in range(len(brefs)):
        keymap.setdefault(key_of(brefs[i], bmat[i]), []).append(i)
    print(f"bigcodec: {len(brefs):,} rows, {len(keymap):,} distinct keys")

    # --- stream tbig, match, collect the 720 for matched bigcodec rows ---
    pf = pq.ParquetFile(tb_p)
    fcols = [f"f{k}" for k in range(720)]
    matched = {}  # bc_idx -> f0..f719 (from tbig, self-consistent)
    seen = 0
    for bi in range(pf.num_row_groups):
        rg = pf.read_row_group(bi, columns=["image_path"] + fcols)
        refs = [stem(x) for x in rg.column("image_path").to_pylist()]
        allf = np.column_stack([rg.column(c).to_numpy() for c in fcols]).astype("f8")
        for r in range(len(refs)):
            cands = keymap.get(key_of(refs[r], allf[r]))
            if not cands:
                continue
            for bi_idx in cands:
                if bi_idx in matched:
                    continue
                # verify f0..f2 close (guard against hash collision)
                if np.abs(allf[r][:3] - bmat[bi_idx][:3]).max() < 1e-4:
                    matched[bi_idx] = allf[r]
                    break
        seen += rg.num_rows
        if bi % 20 == 0:
            print(f"  scanned {seen:,} tbig rows, matched {len(matched):,}/{len(brefs):,}")
    n = len(brefs)
    print(f"matched {len(matched):,}/{n:,} ({100*len(matched)/n:.1f}%)")
    if not matched:
        raise SystemExit("0 matched")

    keep = sorted(matched)
    out = {"ref_basename": pa.array([bc.column("ref_basename")[i].as_py() for i in keep]),
           "human_score": pa.array([bhuman[i] for i in keep])}
    F = np.vstack([matched[i] for i in keep])
    for k in range(720):
        out[f"f{k}"] = pa.array(F[:, k])
    pq.write_table(pa.table(out), out_p, compression="zstd")
    print(f"  wrote {len(keep):,} x720 (+ human_score target) -> {out_p}")


if __name__ == "__main__":
    main()
