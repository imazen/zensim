#!/usr/bin/env python3
"""Score the KonFiG-IQA F-scale TRAIN-source stimuli with each calibrated zensim model.

Reads the reconstructed JND table (TRAIN+VAL sources; keeps only the TRAIN role from its
manifest's origin split), scores every (reference, distorted) pair with
`score_pair_with_bake --bake-post raw` per model (`--model`) and/or with a served-profile
scorer (`--served`: a program called as `PROG ref.png dist.png` that prints the served score,
e.g. the `zensim_score` example, which serves `ZensimProfile::codec_target()`), and writes one
TSV row per (model, stimulus) with the JND target and its uncertainty. No statistics are
computed here.

    score_konfig_train.py --bin <score_pair_with_bake> --out <tsv> --model NAME=REV:PATH ...
    score_konfig_train.py --out <tsv> --served NAME=<zensim_score> ...
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
    ap.add_argument("--bin")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", action="append", default=[], help="NAME=REV:PATH (REV = formula revision or 'default')")
    ap.add_argument("--served", action="append", default=[], help="NAME=PROG (PROG ref dist prints the served score)")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    if not a.model and not a.served:
        raise SystemExit("give --model and/or --served")
    if a.model and not a.bin:
        raise SystemExit("--model needs --bin")

    split = json.load(open(MANIFEST))["split"]["map"]
    train = sorted(s for s, role in split.items() if role == "train")
    # filter at read time: VAL rows (labels included) are never materialised
    rows = pq.read_table(FSCALE, filters=[("source", "in", train)]).to_pylist()
    if {r["source"] for r in rows} != set(train):
        raise SystemExit(f"TRAIN sources {train} not all present in {FSCALE}")
    models = []
    for spec in a.model:
        name, rest = spec.split("=", 1)
        rev, path = rest.split(":", 1)
        models.append((name, rev, path, sha256(path)))
    for spec in a.served:
        name, prog = spec.split("=", 1)
        models.append((name, "served", prog, sha256(prog)))

    def score(model, row):
        name, rev, path, _ = model
        env = dict(os.environ)
        if rev not in ("default", "served"):
            env["ZENSIM_FORMULA_REV"] = rev
        ref = f"{KONFIG}/IMAGES/reference_images/{row['source']}_0.png"
        dist = f"{KONFIG}/{row['image_path']}"
        cmd = ([path, ref, dist] if rev == "served" else
               [a.bin, "--bake", path, "--bake-post", "raw", "--ref", ref, "--dist", dist])
        out = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if out.returncode != 0:
            raise SystemExit(f"{name} {row['image_name']}: {out.stderr.strip()}")
        return float(out.stdout.strip().splitlines()[-1])

    jobs = [(m, r) for m in models for r in rows]
    with cf.ThreadPoolExecutor(a.jobs) as ex:
        scores = list(ex.map(lambda mr: score(*mr), jobs))

    with open(a.out, "w") as f:
        if a.model:
            f.write(f"# konfig fscale TRAIN sources {','.join(train)}; fscale sha256 {sha256(FSCALE)}; "
                    f"binary sha256 {sha256(a.bin)}; bake-post raw\n")
        else:
            f.write(f"# konfig fscale TRAIN sources {','.join(train)}; fscale sha256 {sha256(FSCALE)}; "
                    f"served-profile scorers only\n")
        for name, rev, path, digest in models:
            if rev == "served":
                f.write(f"# model {name} served-profile scorer sha256 {digest} path {path}\n")
            else:
                f.write(f"# model {name} rev {rev} sha256 {digest} path {path}\n")
        f.write("model\tsource\tdistortion\tlevel\timage_name\tscore\tjnd\tjnd_se\tci95_lo\tci95_hi\n")
        for (m, r), s in zip(jobs, scores):
            f.write(f"{m[0]}\t{r['source']}\t{r['distortion']}\t{r['level']}\t{r['image_name']}\t{s!r}\t"
                    f"{r['jnd_f']!r}\t{r['jnd_se']!r}\t{r['ci95_lo']!r}\t{r['ci95_hi']!r}\n")
    print(f"wrote {len(jobs)} rows ({len(rows)} stimuli x {len(models)} models) to {a.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
