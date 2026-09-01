#!/usr/bin/env python3
"""Rank axis on the AIC-4 native-scale PTC set, with the opponent in the run.

300 stimuli = 5 references x 6 codecs (AVIF / JPEG-1 / JPEG-2000 / JPEG-XL /
VVC / JPEG-AI) x 10 distortion levels, all pixel-exact crops of the AIC-3 CTC
full-resolution sources (gate G5), target = the AIC-4 dataset's own
reconstructed JND (`distortion`, higher = more distorted).

Orientation: the target passed to `panel` is NEGATED (`-distortion`) so a
correct metric scores POSITIVE. `zenstats::panel` returns `spearman(..).abs()`
(the registered `band-srocc-absolute` trap), so this file reads `srocc_signed`
from `--batch --stats full` and never the absolute column.

Statistics: `panel --input --per-group` (pooled panel + per-reference SROCC =
`zenstats::per_group_srocc`) and `panel --batch` for the reference-clustered
paired bootstrap. Nothing is computed here.
"""
from __future__ import annotations
import argparse, csv, json, random, subprocess, sys
from pathlib import Path


def run(cmd, stdin=None):
    p = subprocess.run(cmd, capture_output=True, text=True, input=stdin)
    if p.returncode != 0:
        raise SystemExit(f"{cmd[0]} failed:\n{p.stderr[-4000:]}")
    return p.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--panel-bin", required=True)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260901)
    a = ap.parse_args()
    d = Path(a.dir); scratch = Path(a.scratch); scratch.mkdir(parents=True, exist_ok=True)

    tgt = list(csv.DictReader(open(d / "aic4_all_target.tsv"), delimiter="\t"))
    sco = list(csv.DictReader(open(d / "aic4_all_scores.tsv"), delimiter="\t"))
    assert len(tgt) == len(sco)
    for t, s in zip(tgt, sco):
        assert t["stimulus"] == s["stimulus"], "row order differs between target and scores"
    y = [-float(t["jnd"]) for t in tgt]           # higher = better quality
    band = [t["img_num"] for t in tgt]
    scorers = [c for c in sco[0] if c not in ("row", "stimulus")]
    # the published SSIMULACRA2 column from the dataset authors, as a control
    pub = [float(t["published_ssim2"]) for t in tgt]

    imgs = sorted(set(band))
    by_img = {i: [j for j, v in enumerate(band) if v == i] for i in imgs}
    rng = random.Random(a.seed)
    resamples = []
    for _ in range(a.boot):
        pick = []
        for _k in range(len(imgs)):
            pick.extend(by_img[imgs[rng.randrange(len(imgs))]])
        resamples.append(pick)

    # ---- pooled panel + per-reference SROCC, one call per scorer --------
    rows = []
    series = {"peer_ssim2_published": pub}
    for sc in scorers:
        series[sc] = [float(r[sc]) for r in sco]
    for name, x in series.items():
        f = scratch / f"aic4_rank_{name.replace('/', '_').replace('#', '_')}.tsv"
        with open(f, "w") as fh:
            fh.write("predicted\ttarget\tband\n")
            for xv, yv, b in zip(x, y, band):
                fh.write(f"{xv:.10g}\t{yv:.10g}\t{b}\n")
        js = json.loads(run([a.panel_bin, "--input", str(f), "--per-group", "--json"]))
        rows.append((name, js, f))

    # ---- reference-clustered paired bootstrap on SROCC vs ssim2 --------
    man = ["#def Y\t" + ",".join(f"{v:.10g}" for v in y)]
    for name, x in series.items():
        man.append(f"#def X_{len(man)}\t" + ",".join(f"{v:.10g}" for v in x))
    name_to_def = {}
    i = 1
    for name in series:
        name_to_def[name] = f"X_{i}"
        i += 1
    lines = list(man)
    for name in series:
        lines.append(f"{name}|POINT\t@{name_to_def[name]}:@Y\t*")
        for b, pick in enumerate(resamples):
            lines.append(f"{name}|B{b}\t@{name_to_def[name]}:@Y\t" + ",".join(map(str, pick)))
    out = run([a.panel_bin, "--batch", "-", "--stats", "srocc"], stdin="\n".join(lines) + "\n")
    got: dict[str, dict[str, float]] = {}
    for line in out.strip().split("\n")[1:]:
        f = line.split("\t")
        nm, lab = f[0].split("|", 1)
        got.setdefault(nm, {})[lab] = float(f[4])   # srocc_signed

    base = got["peer_ssim2"]
    result = []
    for name, js, _f in rows:
        g = got[name]
        deltas = sorted(g[f"B{b}"] - base[f"B{b}"] for b in range(a.boot))
        n = len(deltas)
        pg = js.get("per_group", {})
        result.append({
            "scorer": name,
            "n": js["groups"][0]["n"] if "groups" in js else len(y),
            "srocc_signed": g["POINT"],
            "per_ref_mean": pg.get("mean"),
            "per_ref_median": pg.get("median"),
            "per_ref_n": pg.get("n_groups"),
            "frac_negative": pg.get("frac_negative"),
            "d_ssim2": g["POINT"] - base["POINT"],
            "d_ssim2_lo": deltas[int(0.025 * n)],
            "d_ssim2_hi": deltas[min(n - 1, int(0.975 * n))],
            "d_ssim2_p_gt0": sum(1 for v in deltas if v > 0) / n,
        })
    outp = d / "aic4_rank.tsv"
    with open(outp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result[0]), delimiter="\t")
        w.writeheader()
        for r in result:
            w.writerow(r)
    for r in result:
        print(f"{r['scorer']:36s} srocc={r['srocc_signed']:+.4f} per_ref={r['per_ref_mean']:.4f} (n={r['per_ref_n']}) "
              f"d={r['d_ssim2']:+.4f} [{r['d_ssim2_lo']:+.4f},{r['d_ssim2_hi']:+.4f}] P={r['d_ssim2_p_gt0']:.3f}")
    print(f"-> {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
