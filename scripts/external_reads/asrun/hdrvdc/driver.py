#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/hdrvdc-conditions-2026-07-29/driver.py
# sha256(source): c7185ad2dd00144fbe5c10318a29baa63e175d3348758043c6f7b5e0bd46bcd4
# build_commit:  6b3505a57174
# Protocol doc:  benchmarks/hdrvdc_conditions_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""HDR-VDC viewing-condition study — decode + extraction driver.

Per PROTOCOL.md (/mnt/v/output/zensim/hdrvdc-conditions-2026-07-29/):
per content batch: ffprobe frame counts -> registered 8 uniform frame
indices from the REF stream -> single ffmpeg pass per video producing the
4K display frame (Lanczos a=3 upscale of PQ-coded R'G'B') and the
1920x1080 far-viewing frame -> extractor manifest over the 5 registered
configs -> per-content feature CSV -> delete PNGs.

Progress streams to stderr continuously (tee to ~/tmp/hdrvdc-extract.log).
"""
import csv
import os
import shutil
import subprocess
import sys
import time

WORK = "/home/lilith/tmp/hdrvdc-work"
VID = f"{WORK}/videos/HDR-VDC"
OUTDIR = f"{WORK}/feats"
LABELS = "/mnt/v/datasets/hdr-vdc/HDR_VDC_JOD_Scores.csv"
EXTRACTOR = "/home/lilith/work/zen/zensim/target/release/examples/hdrvdc_features_extract"
NFRAMES = 8
FULL_W, FULL_H = 3840, 2160
HALF_W, HALF_H = 1920, 1080

# Registered configs: (tag, dim, peak_nits, scale)
CONFIGS = [
    ("A", 0, 1000.0, "full"),
    ("B", 0, 700.0, "full"),
    ("C", 1, 700.0, "full"),
    ("D", 0, 700.0, "half"),
    ("E", 1, 700.0, "half"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def refresh_marker(activity):
    # Study-local heartbeat only — the zensim repo marker is RELEASED
    # during the long decode/extract phase (nothing touches the repo).
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"{WORK}/heartbeat", "w") as f:
        f.write(f"{ts} claude-hdrvdc {activity}\n")


def nb_frames(path):
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=nb_frames", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True).stdout.strip()
    return int(out)


def decode_video(src, dstdir, prefix):
    """One ffmpeg pass: select 8 frames, csc to full-range rgb48 at coded
    res, Lanczos to 4K display frame + 1080p half frame, PNG out."""
    n = nb_frames(src)
    idx = [int((j + 0.5) * n / NFRAMES) for j in range(NFRAMES)]
    sel = "+".join(f"eq(n\\,{k})" for k in idx)
    fc = (
        f"[0:v]select={sel},"
        f"scale=iw:ih:in_color_matrix=bt2020:in_range=tv:out_range=full:"
        f"flags=accurate_rnd+full_chroma_int,format=rgb48le,"
        f"scale={FULL_W}:{FULL_H}:flags=lanczos,split=2[fu][h0];"
        f"[h0]scale={HALF_W}:{HALF_H}:flags=lanczos[ha]"
    )
    cmd = [
        "nice", "-n19", "ionice", "-c3",
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", src,
        "-filter_complex", fc,
        "-map", "[fu]", "-vsync", "0", "-start_number", "0",
        "-compression_level", "30", f"{dstdir}/{prefix}_full_%d.png",
        "-map", "[ha]", "-vsync", "0", "-start_number", "0",
        "-compression_level", "30", f"{dstdir}/{prefix}_half_%d.png",
    ]
    subprocess.run(cmd, check=True)
    for j in range(NFRAMES):
        for s in ("full", "half"):
            p = f"{dstdir}/{prefix}_{s}_{j}.png"
            if not os.path.exists(p):
                raise RuntimeError(f"missing decoded frame {p}")
    return n, idx


def main():
    only = sys.argv[1:] or None  # optional content_id filter (pilot)
    rows = list(csv.DictReader(open(LABELS)))
    # unique distorted videos per content
    contents = {}
    for r in rows:
        cid = r["content_id"]
        c = contents.setdefault(cid, {
            "ref": r["ref_path"], "content": r["content"], "tests": {}})
        if r["is_reference"] == "True":
            continue
        key = (r["crf"], r["resolution"])
        c["tests"][key] = r["test_path"]
    os.makedirs(OUTDIR, exist_ok=True)
    total_t0 = time.time()
    order = sorted(contents, key=int)
    if only:
        order = [c for c in order if c in only]
    for cid in order:
        c = contents[cid]
        out_csv = f"{OUTDIR}/content_{int(cid):02d}.csv"
        if os.path.exists(out_csv):
            log(f"content {cid}: SKIP (exists: {out_csv})")
            continue
        refresh_marker(f"decode+extract content {cid}/{len(order)}")
        t0 = time.time()
        px = f"{WORK}/px_{cid}"
        shutil.rmtree(px, ignore_errors=True)
        os.makedirs(px)
        ref_src = f"{VID}/{c['ref']}"
        n_ref, idx = decode_video(ref_src, px, "ref")
        log(f"content {cid} ({c['content']}): ref {c['ref']} n={n_ref} idx={idx}")
        manifest = []
        dropped = []
        for (crf, res), tpath in sorted(c["tests"].items()):
            n_t = nb_frames(f"{VID}/{tpath}")
            if n_t != n_ref:
                dropped.append((tpath, f"frame count {n_t} != ref {n_ref}"))
                log(f"  DROP {tpath}: frame count {n_t} != ref {n_ref}")
                continue
            vp = f"{crf}_{res}"
            decode_video(f"{VID}/{tpath}", px, vp)
            log(f"  decoded {tpath}")
            for j in range(NFRAMES):
                for tag, dim, peak, scale in CONFIGS:
                    manifest.append("\t".join([
                        f"{cid}|{vp}|f{j}|{tag}", str(dim), str(peak),
                        f"{px}/ref_{scale}_{j}.png",
                        f"{px}/{vp}_{scale}_{j}.png"]))
        man_path = f"{px}/manifest.tsv"
        with open(man_path, "w") as f:
            f.write("\n".join(manifest) + "\n")
        t1 = time.time()
        log(f"content {cid}: decode done {t1-t0:.0f}s, {len(manifest)} manifest rows; extracting")
        refresh_marker(f"extract content {cid}/{len(order)} ({len(manifest)} rows)")
        env = dict(os.environ, RAYON_NUM_THREADS="14")
        subprocess.run(
            ["nice", "-n19", "ionice", "-c3", EXTRACTOR,
             "--manifest", man_path, "--out", out_csv],
            check=True, env=env)
        if dropped:
            with open(f"{OUTDIR}/drops.tsv", "a") as f:
                for p, why in dropped:
                    f.write(f"{cid}\t{p}\t{why}\n")
        shutil.rmtree(px)
        log(f"content {cid}: DONE {time.time()-t0:.0f}s total "
            f"({time.time()-total_t0:.0f}s cumulative)")
    log(f"ALL DONE {time.time()-total_t0:.0f}s")


if __name__ == "__main__":
    main()
