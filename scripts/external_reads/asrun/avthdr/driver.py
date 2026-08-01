#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/avthdr-validation-2026-07-29/driver.py
# sha256(source): 035dff245f213b89714cc17e989e37f902a993e76f492a59ce9d689f7dd44d7f
# build_commit:  1f0f92d5075d
# Protocol doc:  benchmarks/avthdr_validation_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""AVT-VQDB-UHD-1-HDR validation study — stage + decode + extraction driver.

Per PROTOCOL.md (/mnt/v/output/zensim/avthdr-validation-2026-07-29/):
per content batch: copy src + 39 segments from Tower -> local packet-count
gate (== ref N) -> registered 8 uniform frame indices from the REF stream
-> decode via the registered chain (av1/hevc/ffvhuff: system ffmpeg 4.4.2
single pass; vvc: ffmpeg n7.1.5 native decode -> rawvideo pipe -> same
system-ffmpeg filter chain) to 4K display-frame rgb48 PNGs -> extractor
manifest (ONE config: dim=0, peak=1000) -> per-content feature CSV ->
delete staged bytes + PNGs.

Progress streams to stderr continuously (tee to ~/tmp/avthdr-extract.log).
"""
import csv
import os
import re
import shutil
import subprocess
import sys
import time

WORK = "/home/lilith/tmp/avthdr-work"
TOWER = "/mnt/tower/input/datasets/avt-vqdb-uhd-1-hdr"
OUTDIR = f"{WORK}/feats"
LABELS = "/mnt/v/datasets/avt-vqdb-uhd-1-hdr/subjective_scores/mos_ci.csv"
EXTRACTOR = "/home/lilith/work/zen/zensim/target/release/examples/hdrvdc_features_extract"
FF7 = "/home/lilith/tmp/tools/ffmpeg-n7.1-latest-linux64-gpl-7.1/bin/ffmpeg"
FFPROBE7 = "/home/lilith/tmp/tools/ffmpeg-n7.1-latest-linux64-gpl-7.1/bin/ffprobe"
NFRAMES = 8
FULL_W, FULL_H = 3840, 2160
NICE = ["nice", "-n19", "ionice", "-c3"]

CONTENTS = ["Center_Panorama", "DevilMayCry5_P2", "Fireworks", "Flowers",
            "PES2019v2_P2"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def heartbeat(activity):
    # Study-local heartbeat only — the zensim repo marker is RELEASED
    # during this phase (nothing touches the repo; template convention).
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"{WORK}/heartbeat", "w") as f:
        f.write(f"{ts} claude-avthdr {activity}\n")


def packet_count(path):
    if path.endswith(".266"):
        cmd = [FFPROBE7, "-framerate", "60", "-v", "error",
               "-select_streams", "v:0", "-count_packets", "-show_entries",
               "stream=nb_read_packets", "-of", "csv=p=0", path]
    else:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0",
               "-count_packets", "-show_entries", "stream=nb_read_packets",
               "-of", "csv=p=0", path]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return int(out.stdout.strip())


def chain_filter(idx):
    sel = "+".join(f"eq(n\\,{k})" for k in idx)
    return (
        f"[0:v]select={sel},"
        f"scale=iw:ih:in_color_matrix=bt2020:in_range=tv:out_range=full:"
        f"flags=accurate_rnd+full_chroma_int,format=rgb48le,"
        f"scale={FULL_W}:{FULL_H}:flags=lanczos[f]"
    )


def decode_std(src, dstdir, prefix, idx):
    """System ffmpeg 4.4.2 single pass (av1 / hevc / ffvhuff)."""
    cmd = NICE + [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-i", src,
        "-filter_complex", chain_filter(idx),
        "-map", "[f]", "-vsync", "0", "-start_number", "0",
        "-compression_level", "30", f"{dstdir}/{prefix}_%d.png",
    ]
    subprocess.run(cmd, check=True)


def decode_vvc(src, dstdir, prefix, idx, w, h):
    """ffmpeg n7.1.5 native vvc decode -> rawvideo pipe -> system ffmpeg
    SAME filter chain (registered two-step; entropy decode spec-normative)."""
    p1 = subprocess.Popen(
        NICE + [FF7, "-nostdin", "-loglevel", "error", "-framerate", "60",
                "-i", src, "-f", "rawvideo", "-pix_fmt", "yuv420p10le", "-"],
        stdout=subprocess.PIPE)
    p2 = subprocess.Popen(
        NICE + ["ffmpeg", "-nostdin", "-loglevel", "error",
                "-f", "rawvideo", "-pix_fmt", "yuv420p10le",
                "-video_size", f"{w}x{h}", "-framerate", "60", "-i", "-",
                "-filter_complex", chain_filter(idx),
                "-map", "[f]", "-vsync", "0", "-start_number", "0",
                "-compression_level", "30", f"{dstdir}/{prefix}_%d.png"],
        stdin=p1.stdout)
    p1.stdout.close()
    rc2 = p2.wait()
    rc1 = p1.wait()
    if rc1 != 0 or rc2 != 0:
        raise RuntimeError(f"vvc two-step failed rc1={rc1} rc2={rc2} {src}")


def check_pngs(dstdir, prefix):
    for j in range(NFRAMES):
        p = f"{dstdir}/{prefix}_{j}.png"
        if not os.path.exists(p):
            raise RuntimeError(f"missing decoded frame {p}")


def main():
    only = sys.argv[1:] or None
    # segment inventory from the label file (registered 195 rows)
    segs = {}  # content -> list of (codec, w, h, br, towerfile)
    tower_files = os.listdir(f"{TOWER}/videosegments")
    by_stem = {os.path.splitext(f)[0]: f for f in tower_files}
    pat = re.compile(r"^(\d+)_(\d+)_(\w+)_(av1|hevc|vvc)_(.+)\.mkv$")
    n_rows = 0
    for r in csv.DictReader(open(LABELS)):
        m = pat.match(r["stimuli_file"])
        if not m:
            assert "original" in r["stimuli_file"], r["stimuli_file"]
            continue
        w, h, br, codec, content = m.groups()
        stem = f"{w}_{h}_{br}_{codec}_{content}"
        assert stem in by_stem, f"no tower file for {stem}"
        segs.setdefault(content, []).append(
            (codec, int(w), int(h), br, by_stem[stem]))
        n_rows += 1
    assert n_rows == 195 and sorted(segs) == sorted(CONTENTS)
    os.makedirs(OUTDIR, exist_ok=True)
    total_t0 = time.time()
    order = [c for c in CONTENTS if (not only or c in only)]
    for content in order:
        out_csv = f"{OUTDIR}/content_{content}.csv"
        if os.path.exists(out_csv):
            log(f"{content}: SKIP (exists: {out_csv})")
            continue
        t0 = time.time()
        heartbeat(f"stage {content}")
        st = f"{WORK}/stage_{content}"
        px = f"{WORK}/px_{content}"
        for d in (st, px):
            shutil.rmtree(d, ignore_errors=True)
            os.makedirs(d)
        ref_name = f"3840_2160_original_{content}.mkv"
        subprocess.run(NICE + ["cp", f"{TOWER}/srcs/{ref_name}", st],
                       check=True)
        for _, _, _, _, tf in segs[content]:
            subprocess.run(NICE + ["cp", f"{TOWER}/videosegments/{tf}", st],
                           check=True)
        log(f"{content}: staged {len(segs[content])+1} files "
            f"({time.time()-t0:.0f}s)")
        heartbeat(f"decode {content}")
        n_ref = packet_count(f"{st}/{ref_name}")
        idx = [int((j + 0.5) * n_ref / NFRAMES) for j in range(NFRAMES)]
        decode_std(f"{st}/{ref_name}", px, "ref", idx)
        check_pngs(px, "ref")
        log(f"{content}: ref n={n_ref} idx={idx}")
        manifest = []
        dropped = []
        cov = []
        for codec, w, h, br, tf in sorted(segs[content]):
            n_t = packet_count(f"{st}/{tf}")
            cov.append((content, tf, n_t))
            if n_t != n_ref:
                dropped.append((tf, f"packet count {n_t} != ref {n_ref}"))
                log(f"  DROP {tf}: packet count {n_t} != ref {n_ref}")
                continue
            vp = f"{codec}_{w}_{h}_{br}"
            td = time.time()
            try:
                if codec == "vvc":
                    decode_vvc(f"{st}/{tf}", px, vp, idx, w, h)
                else:
                    decode_std(f"{st}/{tf}", px, vp, idx)
                check_pngs(px, vp)
            except (RuntimeError, subprocess.CalledProcessError) as e:
                dropped.append((tf, f"decode failure: {e}"))
                log(f"  DROP {tf}: decode failure: {e}")
                continue
            log(f"  decoded {tf} ({time.time()-td:.0f}s)")
            for j in range(NFRAMES):
                manifest.append("\t".join([
                    f"{content}|{vp}|f{j}|P", "0", "1000",
                    f"{px}/ref_{j}.png", f"{px}/{vp}_{j}.png"]))
        man_path = f"{px}/manifest.tsv"
        with open(man_path, "w") as f:
            f.write("\n".join(manifest) + "\n")
        with open(f"{OUTDIR}/coverage_counts.tsv", "a") as f:
            for c, tf, n in cov:
                f.write(f"{c}\t{tf}\t{n}\n")
        if dropped:
            with open(f"{OUTDIR}/drops.tsv", "a") as f:
                for p, why in dropped:
                    f.write(f"{content}\t{p}\t{why}\n")
        t1 = time.time()
        log(f"{content}: decode done {t1-t0:.0f}s, {len(manifest)} manifest "
            f"rows, {len(dropped)} drops; extracting")
        heartbeat(f"extract {content} ({len(manifest)} rows)")
        env = dict(os.environ, RAYON_NUM_THREADS="14")
        subprocess.run(
            NICE + [EXTRACTOR, "--manifest", man_path, "--out", out_csv],
            check=True, env=env)
        shutil.rmtree(px)
        shutil.rmtree(st)
        log(f"{content}: DONE {time.time()-t0:.0f}s total "
            f"({time.time()-total_t0:.0f}s cumulative)")
    log(f"ALL DONE {time.time()-total_t0:.0f}s")


if __name__ == "__main__":
    main()
