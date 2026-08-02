#!/usr/bin/env python3
"""LYB two-arm re-extraction for the BANDVIS dst-activity adjudication
(benchmarks/bandvis_dst_activity_2026-08-02.md): same frame-sampling
protocol as the July run (~/tmp/lyb_pipeline.py), but each batch's frames
are scored TWICE — ZENSIM_APPEND2_DSTACT=0 and =1 — from the SAME fresh
frames, so the arms are perfectly paired. Fresh-OFF vs the July master is
the ffmpeg-determinism drift check (toggle-off math is byte-stable, F10).
"""
import csv, os, re, subprocess, shutil, sys, time

VID = "/mnt/tower/input/datasets/live-yt-banding/videos"
STAGE = os.path.expanduser("~/tmp/bandvis-dst/lyb-stage")
FR = os.path.expanduser("~/tmp/bandvis-dst/lyb-frames")
OUT_OFF = os.path.expanduser("~/tmp/bandvis-dst/lyb-off")
OUT_ON = os.path.expanduser("~/tmp/bandvis-dst/lyb-on")
DRIVER = os.environ.get("ZENSIM_AB_BIN", os.path.expanduser("~/work/zen/zensim/target/release/examples/v2_ab_extract"))
META = "/mnt/v/datasets/live-yt-banding/metadata/LIVE_Banding_metadata.csv"
N_FRAMES = 8
for d in (STAGE, FR, OUT_OFF, OUT_ON):
    os.makedirs(d, exist_ok=True)

rows = list(csv.DictReader(open(META)))
def content(fn): return re.sub(r'_(ref_qp\d+|cq\d+)\.(mp4|webm)$', '', fn)
groups = {}
for r in rows:
    groups.setdefault(content(r['Filename']), []).append(r)
contents = sorted(groups)

def run(cmd, env=None):
    subprocess.run(cmd, check=True, env=env)

def dur_of(path):
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                        "-of", "csv=p=0", path], capture_output=True, text=True)
    return float(p.stdout.strip())

manifest = open(os.path.join(OUT_OFF, "pairs_manifest.csv"), "w")
manifest.write("row_id,content,dist_file,frame_idx,t\n")
master = {"off": [], "on": []}
header = {}
row_id = 0
BATCH = 5
t_start = time.time()
for bi in range(0, len(contents), BATCH):
    batch = contents[bi:bi + BATCH]
    tsv_path = os.path.join(OUT_OFF, f"batch_{bi}.tsv")
    tsv = open(tsv_path, "w")
    tsv.write("ref_path\tdist_path\thuman_score\n")
    for c in batch:
        g = groups[c]
        ref = [r for r in g if '_ref_' in r['Filename']][0]
        dsts = sorted([r for r in g if '_cq' in r['Filename']], key=lambda r: r['Filename'])
        files = [ref['Filename']] + [d['Filename'] for d in dsts]
        for f in files:
            shutil.copy(os.path.join(VID, f), os.path.join(STAGE, f))
        dur = dur_of(os.path.join(STAGE, ref['Filename']))
        ts = [dur * (i + 0.5) / N_FRAMES for i in range(N_FRAMES)]
        for f in files:
            tag = 'ref' if '_ref_' in f else f.split('_')[-1].split('.')[0]
            for i, t in enumerate(ts):
                run(["nice", "-n19", "ffmpeg", "-hide_banner", "-loglevel", "error",
                     "-ss", f"{t:.3f}", "-i", os.path.join(STAGE, f),
                     "-frames:v", "1", "-y", os.path.join(FR, f"{c}__{tag}__f{i}.png")])
        for d in dsts:
            tag = d['Filename'].split('_')[-1].split('.')[0]
            for i, t in enumerate(ts):
                rp = os.path.join(FR, f"{c}__ref__f{i}.png")
                dp = os.path.join(FR, f"{c}__{tag}__f{i}.png")
                tsv.write(f"{rp}\t{dp}\t{row_id}\n")
                manifest.write(f"{row_id},{c},{d['Filename']},{i},{ts[i]:.3f}\n")
                row_id += 1
        for f in files:
            os.remove(os.path.join(STAGE, f))
    tsv.close()
    manifest.flush()
    # Score the SAME frames with both arms before deleting them.
    for arm, out_dir, dstact in (("off", OUT_OFF, "0"), ("on", OUT_ON, "1")):
        env = dict(os.environ, ZENSIM_AB_MODE="foldapp2",
                   ZENSIM_APPEND2_DSTACT=dstact, RAYON_NUM_THREADS="6")
        csv_path = os.path.join(out_dir, f"batch_{bi}.csv")
        run(["nice", "-n19", "ionice", "-c3", DRIVER, tsv_path, csv_path], env=env)
        with open(csv_path) as f:
            lines = f.read().splitlines()
        master[arm].extend(lines[1:])
        if bi == 0:
            header[arm] = lines[0]
    for fn in os.listdir(FR):
        os.remove(os.path.join(FR, fn))
    print(f"batch {bi}: {len(master['off'])} pair rows/arm total, "
          f"{time.time() - t_start:.0f}s elapsed", flush=True)
manifest.close()
for arm, out_dir in (("off", OUT_OFF), ("on", OUT_ON)):
    with open(os.path.join(out_dir, "lyb_foldapp2_master.csv"), "w") as f:
        f.write(header[arm] + "\n")
        f.write("\n".join(master[arm]) + "\n")
shutil.copy(os.path.join(OUT_OFF, "pairs_manifest.csv"),
            os.path.join(OUT_ON, "pairs_manifest.csv"))
print("DONE", len(master["off"]), "pair rows per arm")
