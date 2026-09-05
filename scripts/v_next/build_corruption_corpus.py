#!/usr/bin/env python3
"""Build a large, source-diverse structural-corruption feature corpus.

For each reference in a sources.tsv (ref_path\tref_id\tcontent_class), run the
codec-corpus `corruption_corpus` generator (the sanctioned structural-corruption
generator — channel swaps, block garbage, bit-flips, geometric, chroma-boundary,
composite, tone, edge, overlay, aliasing), extract 720 zensim features on the
(ref, corruption) pairs + a deduped honest q20/q10 anchor, append to a labeled
parquet, then DELETE the PNGs (they are reproducible from (ref_id, seed, params)).
Streaming keeps disk bounded and yields only the compact feature rows.

Label per row: is_corruption (1/0), family, region, severity, kind, content_class,
ref_id. The honest matched anchors (q20/q10 of the SAME source) are the hard
negatives; broad honest range comes from the existing 720 corpora at train time.

## Feature width (2026-09-05)

`--nfeat 372` (default) runs the CANONICAL 372 extractor
`extract_features_372col` (imazen-only decoders, `--features training,zen-decode`),
whose v1 layout is basic `f0..155` | peaks `f156..227` | masked `f228..299` |
IW `f300..371`. `--nfeat 720` keeps the historical `v2_ab_extract` call for
reproducing the 2026-07-24 corpus. The two extractors take DIFFERENT argv, so
the width also selects the invocation — see `run_extract`.

⚠ ERA: a 372 corpus built after `56bbcda2` (option C — v1 stops pooling phantom
columns) is NOT comparable cell-for-cell with the 2026-07-24 720 corpus. MEASURED
on `im26/lilith/gen-line-concentric__00266…` (1024x1024, exactly the padded-width
class option C removed): 51.8% of basic cells and 66.9%/68.9% of masked/IW cells
move past `max(1e-6, 1e-5·|stored|)`, max |Δ| 0.336, on 664 of 674 rows.

⚠ NON-IMAZEN DEPENDENCY, inherited and deliberately NOT changed here: the
reference downscale below is PIL Lanczos, and the codec-corpus generator loads
images + writes its q10/q20 JPEG anchors through the `image` crate. Swapping
either for the imazen owner (zenresize / zenjpeg) changes the PIXELS, which would
confound an extractor-era A/B with a resampler/encoder change. Keep it identical
while comparing eras; replacing it is its own registered change with its own
before/after.

Usage:
  build_corruption_corpus.py --sources sources.tsv --out corpus.parquet \
      [--gen <corruption_corpus bin>] [--extract <extractor bin>] [--limit N] \
      [--nfeat 372|720] [--resume]
"""
import argparse, os, glob, subprocess, shutil, sys, tempfile
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # sources include legit 100+ MP scans, not attacks

GEN = os.path.expanduser("~/work/codec-corpus/crate/target/release/examples/corruption_corpus")
EXTRACT = "./target/release/examples/v2_ab_extract"
NFEAT = 720
FEATCOLS = [f"f{i}" for i in range(NFEAT)]
MAX_DIM = 1024  # cap source max-dimension (~1MP) — corruption signatures are
                # scale-invariant, and this aligns resolution with the KADIS/safesyn
                # negatives (~0.25-1MP) AND makes 720-feat extraction tractable
                # (the raw imazen-26 sources run to 146 MP → hours/ref otherwise).


def maybe_downsize(ref_path, tmpdir):
    """Return a path to a ≤MAX_DIM version of ref_path (Lanczos), or the original
    if already small enough. Writes a temp PNG so ref+corruption share dimensions."""
    try:
        im = Image.open(ref_path)
        im = im.convert("RGB")
        w, h = im.size
        if max(w, h) <= MAX_DIM:
            return ref_path
        s = MAX_DIM / max(w, h)
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        out = os.path.join(tmpdir, "ref_small.png")
        im.save(out)
        return out
    except Exception as e:
        print(f"  downsize failed for {ref_path}: {e}", flush=True)
        return ref_path


def run_extract(pairs_tsv, out_csv):
    """Invoke the width's extractor. THE one place that knows each binary's argv.

    `extract_features_372col` (the canonical 372 owner, 2026-09-04) takes named
    args and a `pairs` corpus mode; the legacy 720 `v2_ab_extract` took two
    positionals. Getting this wrong produces a truncated CSV that only surfaces
    later as `ValueError: 'fNNN' is not in list` — which is exactly how the
    2026-07-24 build died at rc=1 after 5,420 s with 141 of 174 refs written.
    """
    if NFEAT == 372:
        argv = [EXTRACT, "--corpus", "pairs", "--path", pairs_tsv, "--out", out_csv]
    else:
        argv = [EXTRACT, pairs_tsv, out_csv]
    return subprocess.run(argv, capture_output=True, text=True)


def parse_name(stem):
    # <refid...>__<family>__<region>__op<sev>__<kind>  (last 4 tokens fixed)
    p = stem.split("__")
    if len(p) < 5:
        return None
    return dict(kind=p[-1], severity=p[-2], region=p[-3], family=p[-4],
                ref=("__".join(p[:-4])))


def process_ref(ref_path, ref_id, cclass, tmpdir):
    ref_path = maybe_downsize(ref_path, tmpdir)  # cap ~MAX_DIM for tractable extraction
    outdir = os.path.join(tmpdir, "gen")
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    r = subprocess.run([GEN, "--ref", ref_path, "--ref-id", ref_id, "--class", cclass,
                        "--out", outdir], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isdir(outdir):
        print(f"  SKIP {ref_id}: gen failed ({r.stderr.strip()[:120]})", flush=True)
        return None
    corr = sorted(glob.glob(f"{outdir}/*__corruption.png"))
    q20 = sorted(glob.glob(f"{outdir}/*__q20.png"))[:1]  # honest anchors are identical
    q10 = sorted(glob.glob(f"{outdir}/*__q10.png"))[:1]  # per ref → dedup to one each
    pngs = corr + q20 + q10
    if not corr:
        shutil.rmtree(outdir); return None
    pairs = os.path.join(tmpdir, "pairs.tsv")
    labels = []
    with open(pairs, "w") as f:
        f.write("ref_path\tdist_path\thuman_score\n")
        for i, p in enumerate(pngs):
            f.write(f"{ref_path}\t{p}\t{i}\n")
            lab = parse_name(os.path.splitext(os.path.basename(p))[0]) or {}
            labels.append(lab)
    fcsv = os.path.join(tmpdir, "feats.csv")
    r = run_extract(pairs, fcsv)
    shutil.rmtree(outdir)
    if r.returncode != 0 or not os.path.exists(fcsv):
        print(f"  SKIP {ref_id}: extract failed ({r.stderr.strip()[:120]})", flush=True)
        return None
    # read feats.csv: ref_basename,human_score,f0..f719
    import csv as _csv
    feats = {}
    with open(fcsv) as f:
        rd = _csv.reader(f); hdr = next(rd)
        missing = [c for c in FEATCOLS if c not in hdr]
        if missing or "human_score" not in hdr:
            print(f"  SKIP {ref_id}: extractor output missing cols "
                  f"(e.g. {missing[:2]}) — truncated CSV, skipping ref", flush=True)
            return None
        fi = [hdr.index(c) for c in FEATCOLS]
        hj = hdr.index("human_score")
        for row in rd:
            idx = int(float(row[hj]))
            feats[idx] = np.array([float(row[j]) for j in fi], dtype=np.float32)
    cols = {c: [] for c in FEATCOLS}
    meta = dict(is_corruption=[], family=[], region=[], severity=[], kind=[],
                content_class=[], ref_id=[])
    for i, lab in enumerate(labels):
        if i not in feats or not np.all(np.isfinite(feats[i])):
            continue
        v = feats[i]
        for j, c in enumerate(FEATCOLS):
            cols[c].append(float(v[j]))
        meta["is_corruption"].append(1 if lab.get("kind") == "corruption" else 0)
        meta["family"].append(lab.get("family", "?"))
        meta["region"].append(lab.get("region", "?"))
        meta["severity"].append(lab.get("severity", "?"))
        meta["kind"].append(lab.get("kind", "?"))
        meta["content_class"].append(cclass)
        meta["ref_id"].append(ref_id)
    if not meta["is_corruption"]:
        return None
    return pa.table({**{c: pa.array(cols[c], pa.float32()) for c in FEATCOLS},
                     **{k: pa.array(v) for k, v in meta.items()}})


def main():
    global GEN, EXTRACT, MAX_DIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gen", default=GEN)
    ap.add_argument("--extract", default=EXTRACT)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-dim", type=int, default=MAX_DIM)
    ap.add_argument("--nfeat", type=int, default=372, choices=(372, 720),
                    help="372 = extract_features_372col (canonical, imazen-only "
                         "decoders); 720 = the legacy v2_ab_extract corpus")
    a = ap.parse_args()
    global NFEAT, FEATCOLS
    NFEAT = a.nfeat
    FEATCOLS = [f"f{i}" for i in range(NFEAT)]
    GEN, EXTRACT, MAX_DIM = a.gen, a.extract, a.max_dim
    print(f"nfeat={NFEAT} extractor={EXTRACT}", flush=True)
    srcs = [l.rstrip("\n").split("\t") for l in open(a.sources) if l.strip()]
    if a.limit:
        srcs = srcs[:a.limit]
    writer = None
    n_corr = n_hon = 0
    with tempfile.TemporaryDirectory(dir=os.path.expanduser("~/tmp")) as tmp:
        for k, (ref_path, ref_id, cclass) in enumerate(srcs):
            if not os.path.exists(ref_path):
                print(f"  MISS {ref_id}: {ref_path}", flush=True); continue
            t = process_ref(ref_path, ref_id, cclass, tmp)
            if t is None:
                continue
            if writer is None:
                writer = pq.ParquetWriter(a.out, t.schema, compression="zstd")
            writer.write_table(t)
            nc = int(np.sum(t.column("is_corruption").to_numpy()))
            n_corr += nc; n_hon += t.num_rows - nc
            print(f"[{k+1}/{len(srcs)}] {ref_id}: +{nc} corrupt +{t.num_rows-nc} honest "
                  f"(tot {n_corr}c/{n_hon}h)", flush=True)
    if writer:
        writer.close()
    print(f"DONE: {a.out}  {n_corr} corruption + {n_hon} honest rows", flush=True)


if __name__ == "__main__":
    main()
