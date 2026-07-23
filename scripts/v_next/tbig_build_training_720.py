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

    brefs_full = bc.column("ref_basename").to_pylist()  # keep original (with .png) for output

    # --- stream tbig; write matched rows to the output writer immediately
    #     (bounded memory: keymap + one row-group + a small flush batch). ---
    pf = pq.ParquetFile(tb_p)
    fcols = [f"f{k}" for k in range(720)]
    schema = pa.schema([("ref_basename", pa.string()), ("human_score", pa.float64())]
                       + [(f"f{k}", pa.float64()) for k in range(720)])
    writer = pq.ParquetWriter(out_p, schema, compression="zstd")
    done = set()          # bc_idx already emitted (dedupe: emit each bigcodec row once)
    buf_ref, buf_hum, buf_feat = [], [], []   # flush batch
    seen = written = 0

    def flush():
        nonlocal buf_ref, buf_hum, buf_feat
        if not buf_ref:
            return
        F = np.vstack(buf_feat)
        cols = [pa.array(buf_ref), pa.array(buf_hum)] + [pa.array(F[:, k]) for k in range(720)]
        writer.write_table(pa.table(cols, schema=schema))
        buf_ref, buf_hum, buf_feat = [], [], []

    for bi in range(pf.num_row_groups):
        rg = pf.read_row_group(bi, columns=["image_path"] + fcols)
        refs = [stem(x) for x in rg.column("image_path").to_pylist()]
        allf = np.column_stack([rg.column(c).to_numpy() for c in fcols]).astype("f8")
        for r in range(len(refs)):
            cands = keymap.get(key_of(refs[r], allf[r]))
            if not cands:
                continue
            for bi_idx in cands:
                if bi_idx in done:
                    continue
                if np.abs(allf[r][:3] - bmat[bi_idx][:3]).max() < 1e-4:  # collision guard
                    done.add(bi_idx)
                    buf_ref.append(brefs_full[bi_idx]); buf_hum.append(bhuman[bi_idx]); buf_feat.append(allf[r])
                    written += 1
                    break
        if len(buf_ref) >= 50000:
            flush()
        seen += rg.num_rows
        if bi % 20 == 0:
            print(f"  scanned {seen:,} tbig rows, matched {written:,}/{len(brefs):,}", flush=True)
    flush()
    writer.close()
    n = len(brefs)
    print(f"matched {written:,}/{n:,} ({100*written/n:.1f}%)  -> {out_p}")
    if written == 0:
        raise SystemExit("0 matched")


if __name__ == "__main__":
    main()
