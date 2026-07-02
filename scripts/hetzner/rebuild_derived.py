#!/usr/bin/env python3
"""Rebuild the derived training parquets on a Hetzner train box, byte-equal to
the workstation builds (manifest sha256 gates verify). Usage: rebuild_derived.py /data

Produces under <root>/derived/:
  bigcodec_5p7M_2026-07-02.parquet            all 7 canonical datasets x all splits
  bigcodec_traindigits_2026-07-02.parquet     LSD train digits {0,2,4,6,8}
  bigcodec_valdigits_2026-07-02.parquet       LSD val digits {1,3,5}, 1/12 sample
  kadis_cvvdp_train.parquet / kadis_cvvdp_val.parquet   source_id%10 <8 / ==8

Split rules per docs/DATA_SPLITS.md (§2a LSD via origin_split logic inlined
below to keep the box dependency-free; MUST match
zenmetrics/scripts/picker/origin_split.py — change there first).
"""
import sys, os, re, time
import pyarrow.parquet as pq, pyarrow as pa, pyarrow.compute as pc

ROOT = sys.argv[1] if len(sys.argv) > 1 else "/data"
CANON = f"{ROOT}/canonical-2026-06-27"
KADIS = f"{ROOT}/kadis/kadis700k_canonical_gpu_2026-07-01.parquet"
OUT = f"{ROOT}/derived"
os.makedirs(OUT, exist_ok=True)

# --- LSD origin rule (mirror of origin_split.py; keep in sync) ---
_STEM = re.compile(r"^(?:o_|v2_src)?0*(\d+)")
TRAIN_D, VAL_D, TEST_D = frozenset("02468"), frozenset("135"), frozenset("79")
def split_of(name: str):
    m = _STEM.match(os.path.basename(name))
    if not m: return None
    d = m.group(1)[-1]
    return "train" if d in TRAIN_D else "val" if d in VAL_D else "test" if d in TEST_D else None

def build_bigcodec():
    out = f"{OUT}/bigcodec_5p7M_2026-07-02.parquet"
    if os.path.exists(out): print("bigcodec exists, skip"); return out
    writer = None; tot = 0; t0 = time.time()
    for ds in sorted(os.listdir(CANON)):
        p = f"{CANON}/{ds}"
        if not os.path.isdir(p): continue
        for sp in ("train", "validate", "test"):
            f = f"{p}/{sp}.parquet"
            if not os.path.exists(f): continue
            pf = pq.ParquetFile(f)
            featcols = sorted((c for c in pf.schema_arrow.names
                               if c.startswith("feat_") and c[5:].isdigit()),
                              key=lambda c: int(c[5:]))
            cols = ["variant_name", "score_ssim2"] + featcols
            for batch in pf.iter_batches(batch_size=131072, columns=cols):
                t = pa.Table.from_batches([batch])
                hs = pc.min_element_wise(pc.max_element_wise(
                    pc.divide(pc.cast(t["score_ssim2"], pa.float64()), 100.0), 0.0), 1.0)
                arrays = [t["variant_name"], hs] + [pc.cast(t[c], pa.float64()) for c in featcols]
                names = ["ref_basename", "human_score"] + [f"f{i}" for i in range(len(featcols))]
                o = pa.table(dict(zip(names, arrays)))
                if writer is None:
                    writer = pq.ParquetWriter(out, o.schema, compression="zstd")
                writer.write_table(o); tot += o.num_rows
        print(f"{ds}: total {tot:,} ({time.time()-t0:.0f}s)", flush=True)
    writer.close(); print(f"bigcodec DONE {tot:,} rows", flush=True)
    return out

def build_digitsplits(big):
    otr, ova = f"{OUT}/bigcodec_traindigits_2026-07-02.parquet", f"{OUT}/bigcodec_valdigits_2026-07-02.parquet"
    if os.path.exists(otr) and os.path.exists(ova): print("digitsplits exist, skip"); return
    pf = pq.ParquetFile(big); wtr = wva = None; ntr = nva = 0; vstride = 0
    for batch in pf.iter_batches(batch_size=131072):
        t = pa.Table.from_batches([batch]); names = t["ref_basename"].to_pylist()
        tr, va = [], []
        for i, n in enumerate(names):
            s = split_of(n)
            if s == "train": tr.append(i)
            elif s == "val":
                vstride += 1
                if vstride % 12 == 0: va.append(i)
        if tr:
            tt = t.take(tr)
            if wtr is None: wtr = pq.ParquetWriter(otr, tt.schema, compression="zstd")
            wtr.write_table(tt); ntr += len(tr)
        if va:
            tv = t.take(va)
            if wva is None: wva = pq.ParquetWriter(ova, tv.schema, compression="zstd")
            wva.write_table(tv); nva += len(va)
    wtr.close(); wva.close()
    print(f"digitsplits DONE train={ntr:,} valsample={nva:,}", flush=True)

def build_kadis():
    otr, ova = f"{OUT}/kadis_cvvdp_train.parquet", f"{OUT}/kadis_cvvdp_val.parquet"
    if os.path.exists(otr) and os.path.exists(ova): print("kadis exists, skip"); return
    t = pq.read_table(KADIS)
    featcols = sorted((c for c in t.schema.names if c.startswith("feat_") and c[5:].isdigit()),
                      key=lambda c: int(c[5:]))
    sid = pc.cast(t["source_id"], pa.int64())
    hs = pc.divide(pc.cast(t["score_cvvdp_cpu_imazen_v0_1_0"], pa.float64()), 10.0)
    cols = {"source_id": t["source_id"], "severity_level": t["severity_level"],
            "dist_type": t["dist_type"], "human_score": hs}
    for i, c in enumerate(featcols): cols[f"f{i}"] = pc.cast(t[c], pa.float64())
    full = pa.table(cols)
    # ladder-sort exactly like the workstation build (TV-pair index compat)
    full = full.sort_by([("source_id", "ascending"), ("dist_type", "ascending"),
                         ("severity_level", "ascending")])
    mod = pc.mod(pc.cast(full["source_id"], pa.int64()), 10)
    pq.write_table(full.filter(pc.less(mod, 8)), otr, compression="zstd")
    pq.write_table(full.filter(pc.equal(mod, 8)), ova, compression="zstd")
    print("kadis DONE", flush=True)

if __name__ == "__main__":
    big = build_bigcodec()
    build_digitsplits(big)
    build_kadis()
    print("REBUILD_DERIVED DONE", flush=True)
