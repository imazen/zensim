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
import csv, hashlib, importlib.util, json
from pathlib import Path
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


def run_extract(pairs_tsv, out_csv, env=None):
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
    return subprocess.run(argv, capture_output=True, text=True, env=env)


def file_sha(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_sources(source_path, family_path):
    """Validate the frozen source manifest against both canonical split owners."""
    zen = Path(__file__).resolve().parents[3]
    split_path = zen / "zenmetrics/scripts/picker/origin_split.py"
    spec = importlib.util.spec_from_file_location("origin_split", split_path)
    split = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(split)
    manifest = json.loads(Path(source_path).read_text())
    require(file_sha(family_path) == manifest["split_manifest_sha256"], "family manifest SHA mismatch")
    with open(family_path) as f:
        families = {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
    sources = manifest["sources"]
    require(bool(sources), "empty source manifest")
    origins, pixels, seen_families = set(), set(), set()
    for s in sources:
        origin = s["origin"]
        require(origin not in origins and s["sha256"] not in pixels, "duplicate source")
        require(s["family"] not in seen_families, "duplicate source family")
        origins.add(origin); pixels.add(s["sha256"]); seen_families.add(s["family"])
        require(split.origin_id(s["path"]) == origin, "path/origin mismatch")
        role = {"train": "train", "validate": "val"}.get(s["split"])
        require(role is not None and split.split_of(origin) == role,
                "source is terminal or disagrees with origin split")
        family = families[origin]
        require(family["split"] == s["split"],
                "family split mismatch")
        require((family["family"] or f"origin:{origin}") == s["family"], "family identity mismatch")
        require(s["content_class"] in ("photo", "screen", "graphic", "document"), "unknown class")
        require(file_sha(s["path"]) == s["sha256"], "source SHA mismatch")
    require(len({s["split"] for s in sources}) == 1, "mixed source roles")
    return manifest, file_sha(split_path)


def canonical_corpus(a):
    """Native-only, retained-pixel, complete-row path registered September 8."""
    require(a.nfeat == 372 and a.limit == 0, "canonical mode requires full 372-column extraction")
    require(a.artifacts_dir and a.family_manifest and a.producer_json,
            "canonical mode requires --artifacts-dir, --family-manifest, --producer-json")
    root, output = Path(a.artifacts_dir), Path(a.out)
    require(not root.exists() and not output.exists(), "canonical outputs must be fresh")
    manifest, split_owner_sha = canonical_sources(a.sources_json, a.family_manifest)
    source_manifest_sha = file_sha(a.sources_json)
    producer = json.loads(Path(a.producer_json).read_text())
    require(producer["generator_sha256"] == file_sha(GEN), "generator binary SHA mismatch")
    require(producer["extractor_sha256"] == file_sha(EXTRACT), "extractor binary SHA mismatch")
    require(producer["feature_ids"] == list(range(372)), "unexpected feature contract")
    require(producer["formula_revision"] == 1 and producer["root_form"] == "libm",
            "this registered corpus requires revision 1 / libm")
    require(bool(producer["source_files_sha256"]) and bool(producer["codec_revisions"]),
            "missing producer source provenance")
    for path, expected in producer["source_files_sha256"].items():
        require(file_sha(path) == expected, f"producer source changed: {path}")
    env = dict(os.environ, ZENSIM_FORMULA_REV="1", ZENSIM_ROOT_FORM="libm", RAYON_NUM_THREADS="8")
    root.mkdir(parents=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    (root / "SOURCES.json").write_text(Path(a.sources_json).read_text())
    (root / "PRODUCER.json").write_text(Path(a.producer_json).read_text())
    summaries = []
    writer = None
    try:
        for source in manifest["sources"]:
            origin = source["origin"]
            dest = root / origin
            cmd = [GEN, "corruption", "--in", source["path"], "--out", str(dest),
                   "--ref-id", origin, "--class", source["content_class"], "--seed", "1"]
            result = subprocess.run(cmd, capture_output=True, text=True, env=env)
            (root / f"{origin}.generator.log").write_text(result.stdout + result.stderr)
            require(result.returncode == 0, f"generator failed: {origin}")
            done = json.loads((dest / "COMPLETE.json").read_text())
            generated = json.loads((dest / "_MANIFEST.json").read_text())
            require(file_sha(dest / "_MANIFEST.json") == done["manifest_sha256"], "generator manifest mismatch")
            require(generated["origin"] == origin and generated["content_class"] == source["content_class"],
                    "generated source identity mismatch")
            require(generated["source_sha256"] == done["source_sha256"] == source["sha256"],
                    "generated source SHA mismatch")
            require(generated["generator_revision"] == producer["generator_revision"], "generator revision mismatch")
            require(file_sha(dest / "reference.png") == generated["reference_png_sha256"], "reference PNG mismatch")
            records = generated["records"]
            require(len(records) == done["catalog_entries"] + 2 and done["full_encodes"] == 2
                    and done["independent_anchor_decodes"] == 2, "incomplete generator counts")
            require(len({r["filename"] for r in records}) == len(records), "duplicate generated filename")
            require(sorted(r["quality"] for r in records if r["kind"] == "honest_anchor") == [10, 20],
                    "missing or duplicate anchors")
            require(sum(r.get("inert", False) for r in records) == done["inert_entries"], "inert count mismatch")
            require(len(list(dest.glob("*.png"))) == done["png_files"] == len(records) + 1,
                    "incomplete PNG outputs")
            pairs = dest / "pairs.tsv"
            labels = []
            with pairs.open("w") as f:
                f.write("ref_path\tdist_path\thuman_score\n")
                for i, r in enumerate(records):
                    name = r["filename"]
                    require(Path(name).name == name and name.endswith(".png"), "invalid PNG filename")
                    p = dest / name
                    entry = r.get("entry", {})
                    params = entry.get("params", {})
                    corrupt = r["kind"] == "corruption"
                    require(corrupt or r["kind"] == "honest_anchor", "unknown generated kind")
                    require(type(r["is_corruption"]) is bool, "invalid corruption label")
                    require(file_sha(p) == r["png_sha256"], "generated PNG SHA mismatch")
                    if corrupt:
                        require(entry["ref_id"] == origin and r["pixels_changed"] >= 0,
                                "corruption entry identity mismatch")
                        require(r["is_corruption"] == (r["pixels_changed"] > 0)
                                and r["inert"] == (r["pixels_changed"] == 0), "inert label mismatch")
                    else:
                        require(not r["is_corruption"] and r["quality"] in (10, 20), "anchor label mismatch")
                        require(file_sha(p.with_suffix(".jpg")) == r["encoded_sha256"], "anchor bytes mismatch")
                    f.write(f"{dest / 'reference.png'}\t{p}\t{i}\n")
                    labels.append(dict(
                        ref_id=origin, ref_basename=origin, source_family=source["family"],
                        split=source["split"], content_class=source["content_class"],
                        is_corruption=int(r["is_corruption"]), kind=r["kind"],
                        family=entry.get("family_name", "honest_anchor"),
                        region=json.dumps(params.get("region", "whole"), sort_keys=True),
                        severity=json.dumps(params.get("severity", r.get("quality")), sort_keys=True),
                        params_json=json.dumps(entry, sort_keys=True), inert=r.get("inert", False),
                        pixels_sha256=r["pixels_sha256"], png_sha256=file_sha(p),
                        source_sha256=source["sha256"], filename=str(p), row_id=f"{origin}:{i}"))
            fcsv = dest / "features.csv"
            result = run_extract(str(pairs), str(fcsv), env=env)
            (dest / "extraction.log").write_text(result.stdout + result.stderr)
            require(result.returncode == 0, f"extraction failed: {origin}")
            rows = {}
            with fcsv.open() as f:
                reader = csv.DictReader(f)
                require(reader.fieldnames == ["ref_basename", "human_score", *FEATCOLS], "feature header mismatch")
                for row in reader:
                    require(None not in row and None not in row.values(), "malformed feature row")
                    key = float(row["human_score"])
                    require(np.isfinite(key) and key.is_integer() and 0 <= key < len(labels), "invalid feature key")
                    key = int(key)
                    require(key not in rows, "duplicate feature key")
                    values = np.array([float(row[c]) for c in FEATCOLS], dtype=np.float32)
                    require(np.all(np.isfinite(values)), "nonfinite feature row")
                    rows[key] = values
            require(set(rows) == set(range(len(labels))), "missing feature rows")
            values = np.stack([rows[i] for i in range(len(labels))])
            table = pa.table({**{c: pa.array(values[:, i]) for i, c in enumerate(FEATCOLS)},
                              **{k: pa.array([r[k] for r in labels]) for k in labels[0]}})
            if writer is None:
                writer = pq.ParquetWriter(output, table.schema, compression="zstd")
            writer.write_table(table)
            summaries.append(dict(origin=origin, rows=len(labels),
                                  positives=sum(r["is_corruption"] for r in labels),
                                  inert=sum(r["inert"] for r in labels),
                                  unique_pixels=len({r["pixels_sha256"] for r in labels}),
                                  manifest_sha256=done["manifest_sha256"], features_sha256=file_sha(fcsv)))
            print(f"{origin}: {summaries[-1]}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    # A partial parquet is diagnostic output, never an accepted dataset.
    require(file_sha(a.sources_json) == source_manifest_sha, "source manifest changed")
    canonical_sources(a.sources_json, a.family_manifest)
    require(file_sha(GEN) == producer["generator_sha256"] and file_sha(EXTRACT) == producer["extractor_sha256"],
            "tool binary changed during generation")
    for path, expected in producer["source_files_sha256"].items():
        require(file_sha(path) == expected, f"producer source changed: {path}")
    require(len(summaries) == len(manifest["sources"]), "incomplete sources")
    complete = dict(schema="canonical-corruption-corpus-v1", sources=summaries,
                    rows=sum(s["rows"] for s in summaries), full_encodes=2 * len(summaries),
                    independent_anchor_decodes=2 * len(summaries),
                    source_manifest_sha256=source_manifest_sha, split_owner_sha256=split_owner_sha,
                    producer=producer, parquet=str(output), parquet_sha256=file_sha(output))
    (root / "COMPLETE.json").write_text(json.dumps(complete, indent=2) + "\n")


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
    sources = ap.add_mutually_exclusive_group(required=True)
    sources.add_argument("--sources", help="historical TSV mode, unchanged pixels/anchors")
    sources.add_argument("--sources-json", help="canonical frozen JSON manifest, native IO/anchors")
    ap.add_argument("--artifacts-dir")
    ap.add_argument("--family-manifest")
    ap.add_argument("--producer-json")
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
    if a.sources_json:
        canonical_corpus(a)
        return
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
