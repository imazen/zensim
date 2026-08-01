#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/avthdr-validation-2026-07-29/chug_driver.py
# sha256(source): 815a916df14588fa2d85d8a90fa5010195047196d9526ff89203fca4d2243d48
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
"""CHUG sampled FR leg — decode + extraction driver (per PROTOCOL.md).

Reads chug_sample.tsv (registered seeded sample). Per content: copy ref +
its sampled transcodes from Tower -> decode ref ONCE at coded resolution
(registered chain: select 8 uniform frames from ref count -> swscale
bt2020 tv->full rgb48 accurate_rnd+full_chroma_int at coded res; ref
passes through unscaled) -> per transcode: same chain + Lanczos a=3
upscale to the ref's coded resolution -> `-noautorotate` both sides
(coded-orientation FR; rotation metadata is display-side common-mode) ->
manifest rows (dim=0, peak=1000) -> batch extraction -> delete pixels.
"""
import csv
import os
import shutil
import subprocess
import sys
import time
from collections import defaultdict

WORK = "/home/lilith/tmp/avthdr-work"
VID = "/mnt/tower/input/datasets/chug/videos"
OUTDIR = f"{WORK}/chug_feats"
SAMPLE = f"{WORK}/chug_sample.tsv"
EXTRACTOR = "/home/lilith/work/zen/zensim/target/release/examples/hdrvdc_features_extract"
NFRAMES = 8
NICE = ["nice", "-n19", "ionice", "-c3"]
BATCH_PAIRS = 40


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def heartbeat(activity):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(f"{WORK}/heartbeat", "w") as f:
        f.write(f"{ts} claude-avthdr {activity}\n")


def decode(src, dstdir, prefix, idx, out_w, out_h):
    sel = "+".join(f"eq(n\\,{k})" for k in idx)
    fc = (
        f"[0:v]select={sel},"
        f"scale=iw:ih:in_color_matrix=bt2020:in_range=tv:out_range=full:"
        f"flags=accurate_rnd+full_chroma_int,format=rgb48le,"
        f"scale={out_w}:{out_h}:flags=lanczos[f]"
    )
    cmd = NICE + [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-noautorotate",
        "-i", src, "-filter_complex", fc,
        "-map", "[f]", "-vsync", "0", "-start_number", "0",
        "-compression_level", "30", f"{dstdir}/{prefix}_%d.png",
    ]
    subprocess.run(cmd, check=True)
    for j in range(NFRAMES):
        p = f"{dstdir}/{prefix}_{j}.png"
        if not os.path.exists(p):
            raise RuntimeError(f"missing decoded frame {p}")


def main():
    pairs = list(csv.DictReader(open(SAMPLE), delimiter="\t"))
    bycontent = defaultdict(list)
    for p in pairs:
        bycontent[p["content"]].append(p)
    os.makedirs(OUTDIR, exist_ok=True)
    px = f"{WORK}/chug_px"
    total_t0 = time.time()
    manifest = []
    batch_i = 0
    dropped = []
    done_pairs = 0

    def flush(force=False):
        nonlocal manifest, batch_i
        if not manifest or (len(manifest) < BATCH_PAIRS * NFRAMES and not force):
            return
        man_path = f"{px}/manifest_{batch_i}.tsv"
        with open(man_path, "w") as f:
            f.write("\n".join(m[0] for m in manifest) + "\n")
        out_csv = f"{OUTDIR}/chug_batch_{batch_i:03d}.csv"
        heartbeat(f"chug extract batch {batch_i} ({len(manifest)} rows)")
        env = dict(os.environ, RAYON_NUM_THREADS="14")
        subprocess.run(NICE + [EXTRACTOR, "--manifest", man_path,
                               "--out", out_csv], check=True, env=env)
        # delete only pixel files referenced by this batch
        for _, files in manifest:
            for fp in files:
                if os.path.exists(fp):
                    os.remove(fp)
        log(f"batch {batch_i}: extracted {len(manifest)} rows -> {out_csv}")
        manifest = []
        batch_i += 1

    shutil.rmtree(px, ignore_errors=True)
    os.makedirs(px)
    contents = sorted(bycontent)
    for ci, content in enumerate(contents):
        plist = bycontent[content]
        ref_id = plist[0]["ref_id"]
        ref_w, ref_h = int(plist[0]["ref_w"]), int(plist[0]["ref_h"])
        n_ref = int(plist[0]["n_frames"])
        idx = [int((j + 0.5) * n_ref / NFRAMES) for j in range(NFRAMES)]
        heartbeat(f"chug content {ci+1}/{len(contents)}")
        st = f"{WORK}/chug_stage"
        shutil.rmtree(st, ignore_errors=True)
        os.makedirs(st)
        subprocess.run(NICE + ["cp", f"{VID}/{ref_id}.mp4", st], check=True)
        rp = f"c{ci:03d}_ref"
        try:
            decode(f"{st}/{ref_id}.mp4", px, rp, idx, ref_w, ref_h)
        except (RuntimeError, subprocess.CalledProcessError) as e:
            for p in plist:
                dropped.append((p["rung"], content, f"ref decode failure: {e}"))
            log(f"  DROP all pairs of {content}: ref decode failure: {e}")
            shutil.rmtree(st)
            continue
        ref_files = [f"{px}/{rp}_{j}.png" for j in range(NFRAMES)]
        used_ref = False
        for p in plist:
            # orientation guard (coded dims, both from probe)
            po = (int(p["tr_w"]) < int(p["tr_h"])) != (ref_w < ref_h)
            if po:
                dropped.append((p["rung"], content, "orientation mismatch"))
                log(f"  DROP {p['rung']} {content}: orientation mismatch")
                continue
            subprocess.run(NICE + ["cp", f"{VID}/{p['tr_id']}.mp4", st],
                           check=True)
            tp = f"c{ci:03d}_{p['rung'].rstrip('_')}"
            try:
                decode(f"{st}/{p['tr_id']}.mp4", px, tp, idx, ref_w, ref_h)
            except (RuntimeError, subprocess.CalledProcessError) as e:
                dropped.append((p["rung"], content, f"decode failure: {e}"))
                log(f"  DROP {p['rung']} {content}: decode failure: {e}")
                continue
            tr_files = [f"{px}/{tp}_{j}.png" for j in range(NFRAMES)]
            for j in range(NFRAMES):
                key = f"{content}|{p['rung']}|f{j}|P"
                line = "\t".join([key, "0", "1000", ref_files[j], tr_files[j]])
                # ref pngs deleted with the LAST batch that references them:
                # attach them only to this pair's rows for cleanup bookkeeping
                manifest.append((line, [tr_files[j]]))
            used_ref = True
            done_pairs += 1
            if done_pairs % 10 == 0:
                log(f"  {done_pairs} pairs decoded "
                    f"({time.time()-total_t0:.0f}s cumulative)")
        # ref cleanup after this content's pairs are all in manifest:
        # defer deletion until after next flush by tagging ref files onto
        # the last row of this content
        if used_ref and manifest:
            manifest[-1] = (manifest[-1][0], manifest[-1][1] + ref_files)
        elif not used_ref:
            for fp in ref_files:
                os.remove(fp)
        shutil.rmtree(st)
        flush(force=False)
    flush(force=True)
    if dropped:
        with open(f"{OUTDIR}/chug_drops.tsv", "a") as f:
            for rung, content, why in dropped:
                f.write(f"{rung}\t{content}\t{why}\n")
    log(f"ALL DONE {time.time()-total_t0:.0f}s; {done_pairs} pairs, "
        f"{len(dropped)} drops")


if __name__ == "__main__":
    main()
