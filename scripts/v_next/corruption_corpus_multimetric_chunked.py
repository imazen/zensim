#!/usr/bin/env python3
"""Multi-metric scoring of the corruption corpus, batched + incremental progress.

Scores every (ref, variant) pair in the gb82_dog corruption-gate corpus
across 4 GPU metrics (ssim2-gpu, butteraugli-gpu, cvvdp, dssim-gpu).
Writes a TSV incrementally so partial results survive crashes.

Output: /tmp/corruption_multimetric_2026-05-28.tsv

Usage:
  python3 scripts/v_next/corruption_corpus_multimetric_chunked.py
"""
import os, re, subprocess, sys, time
from pathlib import Path

CORPUS = Path("/mnt/v/output/zensim/corruption_gate")
REF = CORPUS / "gb82_dog__reference.png"
ZEN_METRICS = Path("/home/lilith/work/zen/zenmetrics/target/release/zenmetrics")
OUT = Path("/tmp/corruption_multimetric_2026-05-28.tsv")
CHUNK_SIZE = 50


def parse_meta(fname):
    base = fname[:-4]  # strip .png
    parts = base.split("__")
    if len(parts) < 5:
        return None
    # ref__family__region__sev__kind, but family may contain underscores
    kind = parts[-1]
    sev = parts[-2]
    region = parts[-3]
    family = "_".join(parts[1:-3])
    name = "__".join(parts[:-1])
    return name, family, region, sev, kind


def main():
    variants = sorted(
        f for f in os.listdir(CORPUS)
        if f.endswith("__corruption.png") or f.endswith("__q20.png") or f.endswith("__q10.png")
    )
    print(f"[{time.strftime('%H:%M:%S')}] {len(variants)} variants to score "
          f"({(len(variants) + CHUNK_SIZE - 1) // CHUNK_SIZE} chunks of {CHUNK_SIZE})", file=sys.stderr)

    with open(OUT, "w") as f:
        f.write("name\tfamily\tregion\tsev\tkind\tssim2_gpu\tbutter_max_gpu\tbutter_pnorm3_gpu\tcvvdp\tdssim_gpu\n")

    n_done = 0
    t0 = time.time()
    for chunk_idx in range(0, len(variants), CHUNK_SIZE):
        chunk = variants[chunk_idx:chunk_idx + CHUNK_SIZE]
        args = [str(ZEN_METRICS), "compare", "--reference", str(REF),
                "--metric", "ssim2-gpu", "--metric", "butteraugli-gpu",
                "--metric", "cvvdp", "--metric", "dssim-gpu"]
        for v in chunk:
            args.append("--variant"); args.append(str(CORPUS / v))
        try:
            res = subprocess.run(args, capture_output=True, text=True, timeout=300)
        except subprocess.TimeoutExpired:
            print(f"  chunk {chunk_idx // CHUNK_SIZE}: TIMEOUT after 300s; skipping", file=sys.stderr)
            continue
        if res.returncode != 0:
            print(f"  chunk {chunk_idx // CHUNK_SIZE}: returncode={res.returncode}, "
                  f"stderr={res.stderr[:300]}", file=sys.stderr)
            continue
        groups = re.split(r"\n\s*\n", res.stdout.strip())
        rows = []
        for g in groups:
            lines = [l.strip() for l in g.splitlines() if l.strip()]
            if not lines:
                continue
            m = re.match(r"^(.+?) vs (.+?):$", lines[0])
            if not m:
                continue
            variant_fname = os.path.basename(m.group(2))
            meta = parse_meta(variant_fname)
            if not meta:
                continue
            name, family, region, sev, kind = meta
            scores = {}
            for line in lines[1:]:
                parts = line.split(None, 1)
                if len(parts) != 2:
                    continue
                k, vraw = parts[0], parts[1].strip()
                try:
                    scores[k] = float(vraw) if not vraw.startswith("ERROR") else None
                except ValueError:
                    scores[k] = None

            def fmt(key):
                v = scores.get(key)
                return f"{v:.4f}" if v is not None else "nan"

            rows.append("\t".join([name, family, region, sev, kind,
                                   fmt("ssim2_gpu"),
                                   fmt("butteraugli_max_gpu"),
                                   fmt("butteraugli_pnorm3_gpu"),
                                   fmt("cvvdp_imazen_v0_0_1"),
                                   fmt("dssim_gpu")]))
        with open(OUT, "a") as f:
            f.write("\n".join(rows) + ("\n" if rows else ""))
        n_done += len(rows)
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = len(variants) - n_done
        eta_min = remaining / rate / 60 if rate > 0 else 0
        print(f"[{time.strftime('%H:%M:%S')}] chunk {chunk_idx // CHUNK_SIZE + 1}/"
              f"{(len(variants) + CHUNK_SIZE - 1) // CHUNK_SIZE}: "
              f"{n_done}/{len(variants)} rows  rate={rate:.1f}/s  eta={eta_min:.1f}min",
              file=sys.stderr, flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE — {n_done} rows in {(time.time() - t0) / 60:.1f}min",
          file=sys.stderr)


if __name__ == "__main__":
    main()
