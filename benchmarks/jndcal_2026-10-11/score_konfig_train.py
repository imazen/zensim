#!/usr/bin/env python3
"""Score the KonFiG-IQA F-scale TRAIN-source stimuli with each calibrated zensim model.

Reads the reconstructed JND table (TRAIN+VAL sources; keeps only the TRAIN role from its
manifest's origin split), scores every (reference, distorted) pair with
`score_pair_with_bake --bake-post raw` per model, and writes one TSV row per
(model, stimulus) with the JND target and its uncertainty. No statistics are computed here.

    score_konfig_train.py --bin <score_pair_with_bake> --out <tsv> --model NAME=REV:PATH ...
"""
import argparse, concurrent.futures as cf, hashlib, json, os, subprocess, sys
import pyarrow.parquet as pq

KONFIG = "/mnt/v/dataset/konfig-iqa/KonFiG-IQA"
FSCALE = "/mnt/v/dataset/konfig-iqa/derived/konfig_fscale_trainval_2026-10.parquet"
MANIFEST = "/mnt/v/dataset/konfig-iqa/derived/konfig_fscale_trainval_2026-10.manifest.json"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", required=True, help="NAME=REV:PATH (REV = formula revision or 'default')")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()

    split = json.load(open(MANIFEST))["split"]["map"]
    train = sorted(s for s, role in split.items() if role == "train")
    rows = [r for r in pq.read_table(FSCALE).to_pylist() if split[r["source"]] == "train"]
    models = []
    for spec in a.model:
        name, rest = spec.split("=", 1)
        rev, path = rest.split(":", 1)
        models.append((name, rev, path, sha256(path)))

    def score(model, row):
        name, rev, path, _ = model
        env = dict(os.environ)
        if rev != "default":
            env["ZENSIM_FORMULA_REV"] = rev
        ref = f"{KONFIG}/IMAGES/reference_images/{row['source']}_0.png"
        dist = f"{KONFIG}/{row['image_path']}"
        out = subprocess.run([a.bin, "--bake", path, "--bake-post", "raw", "--ref", ref, "--dist", dist],
                             capture_output=True, text=True, env=env)
        if out.returncode != 0:
            raise SystemExit(f"{name} {row['image_name']}: {out.stderr.strip()}")
        return float(out.stdout.strip().splitlines()[-1])

    jobs = [(m, r) for m in models for r in rows]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        scores = list(ex.map(lambda mr: score(*mr), jobs))

    with open(a.out, "w") as f:
        f.write(f"# konfig fscale TRAIN sources {','.join(train)}; fscale sha256 {sha256(FSCALE)}; "
                f"binary sha256 {sha256(a.bin)}; bake-post raw\n")
        for name, rev, path, digest in models:
            f.write(f"# model {name} rev {rev} sha256 {digest} path {path}\n")
        f.write("model\tsource\tdistortion\tlevel\timage_name\tscore\tjnd\tjnd_se\tci95_lo\tci95_hi\n")
        for (m, r), s in zip(jobs, scores):
            f.write(f"{m[0]}\t{r['source']}\t{r['distortion']}\t{r['level']}\t{r['image_name']}\t{s!r}\t"
                    f"{r['jnd_f']!r}\t{r['jnd_se']!r}\t{r['ci95_lo']!r}\t{r['ci95_hi']!r}\n")
    print(f"wrote {len(jobs)} rows ({len(rows)} stimuli x {len(models)} models) to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
