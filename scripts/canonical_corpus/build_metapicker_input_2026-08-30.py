#!/usr/bin/env python3
"""Build the criterion-8 meta-picker input parquets.

Rows = every config row of every family (7 canonical datasets), with:
  feat_0..N   = SOURCE features (clean_features.tsv, zenanalyze 0.2.0, per rendition)
  codec       = family label; knob_tuple_json = {"cell": family}  -> cell == family
  score_zensim = the 2026-08-30 Profile-B rescore (era-correct judge)
  encoded_bytes, q, image_path (rendition basename)
Splits kept separate (train/validate/test) per the origin-digit rule already
baked into the canonical views.
"""
import csv, json, os, hashlib
import pyarrow.parquet as pq, pyarrow as pa
SRC="/mnt/v/output/clean-picker-corpus-2026-06-26/clean_features.tsv"
RS="/mnt/v/output/zensim/picker-rescore-B-2026-08-30"
OUT="/mnt/v/output/zensim/metapicker-2026-08-30"
os.makedirs(OUT, exist_ok=True)
# source features keyed by rendition basename (strip trailing .png once)
rows=list(csv.reader(open(SRC), delimiter="\t"))
hdr=rows[0]
meta_cols=8
# 36 size-gated columns (aq_map/noise_floor families) are empty on the same
# 636 small renditions — drop the COLUMNS, keep every rendition.
allcols=list(range(meta_cols, len(hdr)))
empty_cols=set()
for r in rows[1:]:
    for i in allcols:
        if i < len(r) and r[i]=="": empty_cols.add(i)
keep=[i for i in allcols if i not in empty_cols]
fnames=[hdr[i] for i in keep]
def norm(b):
    b=os.path.basename(str(b))
    return b[:-4] if b.endswith(".png") else b
feats={}
for r in rows[1:]:
    feats[norm(r[0])]=[float(r[i]) for i in keep]
print(f"source-feature renditions: {len(feats)}  n_feat kept: {len(fnames)} (dropped {len(empty_cols)} size-gated cols)")
FAMS=["zenjpeg_lossy","zenwebp_lossy","zenjxl_lossy","zenavif_lossy","zenpng_lossless","zenjxl_lossless","zenwebp_lossless"]
man={"built":"2026-08-30","source_features":SRC,"n_feat":len(fnames),"judge":"Profile B rescore 2026-08-30","files":{}}
for split in ["train","validate","test"]:
    cols={"image_path":[], "codec":[], "knob_tuple_json":[], "q":[], "score_zensim":[], "encoded_bytes":[]}
    fcols=[[] for _ in fnames]
    miss=set()
    for fam in FAMS:
        p=f"{RS}/{fam}_{split}_B.parquet"
        t=pq.read_table(p, columns=["variant_name","q","score_zensim_b","encoded_bytes"])
        vn=[norm(v) for v in t.column(0).to_pylist()]
        qs=t.column(1).to_pylist(); sc=t.column(2).to_pylist(); by=t.column(3).to_pylist()
        kj=json.dumps({"cell":fam})
        for i,v in enumerate(vn):
            fv=feats.get(v)
            if fv is None: miss.add(v); continue
            cols["image_path"].append(v); cols["codec"].append(fam)
            cols["knob_tuple_json"].append(kj)
            cols["q"].append(float(qs[i]) if qs[i] is not None else 0.0)
            cols["score_zensim"].append(float(sc[i])); cols["encoded_bytes"].append(int(by[i]))
            for j,x in enumerate(fv): fcols[j].append(x)
    data={k:pa.array(v) for k,v in cols.items()}
    for j,n in enumerate(fnames): data[f"feat_{j}"]=pa.array(fcols[j])
    tbl=pa.table(data)
    p=f"{OUT}/meta_{split}.parquet"
    pq.write_table(tbl,p,compression="zstd")
    man["files"][f"meta_{split}.parquet"]={"rows":tbl.num_rows,"missing_renditions":len(miss),"sha256":hashlib.sha256(open(p,'rb').read()).hexdigest()}
    print(split, tbl.num_rows, "rows; missing-rendition keys:", len(miss), list(miss)[:3])
json.dump(man, open(f"{OUT}/_MANIFEST.json","w"), indent=1)
print("done")
