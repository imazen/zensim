#!/usr/bin/env python3
"""Build ref/dist/human_score pairs manifests for FR-IQA corpora → feed
`extract_features_372col --corpus pairs-tsv`. One reproducible builder per dataset so the
FR-corpus expansion (CSIQ/LIVE/TID2008/...) isn't amnesiac.

Convention (MUST match kadid/tid): human_score is QUALITY-oriented in [0,1] (higher = better).
Datasets whose native label is a distortion score (DMOS higher=worse) are flipped to 1−norm.
⚠ A column NAMED `dmos` is not automatically distortion-oriented — KADID's is a MOS in
disguise (raw DCR falls with severity), and blindly flipping it inverted the target for six
weeks (2026-08-04, see build_kadid). CHECK the native orientation against the corpus's raw
human labels before choosing a transform, and gate the output with
`scripts/canonical_corpus/check_target_orientation.py`.
Output TSV columns: ref_path, dist_path, human_score  (the tool derives ref_basename).

Usage: python3 scripts/canonical_corpus/build_fr_corpus_pairs.py <csiq|live|tid2008>
Then:  extract_features_372col --corpus pairs-tsv --path <out.tsv> --out <corpus>_features_372col.csv
       (convert csv→parquet, add a Corpus entry to bake_verdict CORPORA, rebuild)
"""
import os, sys, csv
from pathlib import Path


class MissingCorpusFiles(RuntimeError):
    """A pairs manifest could not name every (ref, dist) the label file lists."""


def _require_all_resolved(label: str, missing: list[str], total: int) -> None:
    """NO GRACEFUL SKIPS. A pairs builder that silently drops rows produces a
    manifest that looks complete and is not — the downstream extractor then
    emits fewer rows than the corpus has and, if anything catches it at all, it
    is a row-count guard several steps later.

    MEASURED CASE (2026-08-30): `build_tid` upper-cased every reference stem
    while TID2013's own reference dir carries `i25.png` LOWERCASE (its source
    BMP is `i25.bmp` too). 120 of 3,000 rows named a path that does not exist.
    The old code counted them into a `skipped` total and printed it; nothing
    failed, and the broken manifest sat on disk for six weeks until an
    unrelated re-extraction hit the row-count guard.

    Set `FRPAIRS_ALLOW_MISSING=1` to downgrade to a loud warning — the decision
    then belongs to the CALLER and is visible in the invocation, which is the
    only form of skip the project allows.
    """
    if not missing:
        return
    head = "\n  ".join(missing[:10])
    more = f"\n  ... and {len(missing) - 10} more" if len(missing) > 10 else ""
    msg = (f"{label}: {len(missing)} of {total} label rows name a file that does "
           f"not exist:\n  {head}{more}\n"
           f"Refusing to write a manifest that silently omits them.")
    if os.environ.get("FRPAIRS_ALLOW_MISSING") == "1":
        print(f"WARNING (FRPAIRS_ALLOW_MISSING=1): {msg}", file=sys.stderr)
        return
    raise MissingCorpusFiles(msg)


def _case_insensitive_index(d: Path) -> dict:
    """name.lower() -> real path. Corpora with mixed-case filenames (TID2013
    ships both `I01_..` and `i01_..`, and its 25th reference as `i25.png`) must
    be resolved through this, never by forcing a case."""
    return {q.name.lower(): q for q in d.iterdir()}


def build_csiq():
    """CSIQ: 30 refs × 6 distortions. DMOS in [0,1] (0=best). human = 1 − DMOS."""
    import openpyxl
    # CONSOLIDATED 2026-07-22: csiq lives entirely under the SINGULAR
    # /mnt/v/dataset/csiq (source PNGs + the per-distortion dirs awgn/blur/...
    # AND the DMOS xlsx + pairs tsv, moved here from the old plural
    # /mnt/v/datasets/csiq). The plural distorted dst_imgs/ was a duplicate of
    # the singular per-distortion dirs; its archive is on tower. See
    # benchmarks/dataset_path_audit_2026-07-22.md.
    SRC = "/mnt/v/dataset/csiq"
    DST = "/mnt/v/dataset/csiq"  # per-distortion dirs live directly under csiq/
    OUT = "/mnt/v/dataset/csiq/csiq_pairs.tsv"
    # dst_type (xlsx) -> (folder, filename_token)
    M = {"noise": ("awgn", "AWGN"), "blur": ("blur", "BLUR"), "contrast": ("contrast", "contrast"),
         "fnoise": ("fnoise", "fnoise"), "jpeg": ("jpeg", "JPEG"), "jpeg 2000": ("jpeg2000", "jpeg2000")}
    ws = openpyxl.load_workbook("/mnt/v/dataset/csiq/csiq.DMOS.xlsx")["all_by_image"]
    rows = [r for r in ws.iter_rows(values_only=True)]
    hi = next(i for i, r in enumerate(rows) if r and "image" in [str(x) for x in r])
    hdr = [str(x) for x in rows[hi]]
    I = hdr.index
    out, miss = [], 0
    for r in rows[hi + 1:]:
        if not r or r[I("image")] is None or r[I("dmos")] is None:
            continue
        img, dt = str(r[I("image")]), str(r[I("dst_type")])
        lev = str(r[I("dst_lev")]).split(".")[0]
        if dt not in M:
            continue
        folder, tok = M[dt]
        ref = f"{SRC}/{img}.png"
        dist = f"{DST}/{folder}/{img}.{tok}.{lev}.png"
        if not (Path(ref).exists() and Path(dist).exists()):
            miss += 1
            continue
        out.append((ref, dist, 1.0 - float(r[I("dmos")])))  # 1 − DMOS → quality-oriented
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"CSIQ: {len(out)} pairs → {OUT}  (skipped {miss} missing)")


def build_live():
    """LIVE IQA Release 2: 29 refs × {jp2k,jpeg,wn,gblur,fastfading}, 779 real distortions.

    readme.txt fixes the concat order EXACTLY:
      dmos=[jp2k(1:227) jpeg(1:233) wn(1:174) gblur(1:174) fastfading(1:174)]  (982 total)
    orgs(i)==1 marks a reference-copy placed in-sequence (dmos~0) -> skip (982-779=203).
    We use the REALIGNED dmos (dmos_new, Sheikh 2006 recommended) + refnames_all for the
    ref join. human_score = 1 - dmos_new/100  (quality-oriented [0,1], higher=better, to
    match kadid/tid/csiq). dmos_std (per-sample sigma) is emitted as a 4th column for
    later Z-RMSE; the feature extractor reads only the first 3 (ref/dist/human_score)."""
    import scipy.io as sio
    import numpy as np
    BASE = "/mnt/v/datasets/LIVE/databaserelease2"
    OUT = "/mnt/v/datasets/LIVE/live_r2_pairs.tsv"
    # (folder, global offset, count) — offsets are the readme concat order.
    SEG = [("jp2k", 0, 227), ("jpeg", 227, 233), ("wn", 460, 174),
           ("gblur", 634, 174), ("fastfading", 808, 174)]
    rea = sio.loadmat(f"{BASE}/dmos_realigned.mat")
    dmos = np.asarray(rea["dmos_new"]).flatten()          # realigned DMOS, ~[-3,112]
    std = np.asarray(rea["dmos_std"]).flatten()           # per-sample sigma
    orgs = np.asarray(rea["orgs"]).flatten().astype(int)  # 1 == reference-copy -> skip
    refs = [str(x[0]) for x in np.asarray(sio.loadmat(f"{BASE}/refnames_all.mat")["refnames_all"]).flatten()]
    out, miss = [], 0
    for folder, off, cnt in SEG:
        for k in range(1, cnt + 1):
            gi = off + (k - 1)
            if orgs[gi] == 1:            # reference-copy in-sequence, not a real distortion
                continue
            ref = f"{BASE}/refimgs/{refs[gi]}"
            dist = f"{BASE}/{folder}/img{k}.bmp"
            if not (Path(ref).exists() and Path(dist).exists()):
                miss += 1
                continue
            out.append((ref, dist, 1.0 - float(dmos[gi]) / 100.0, float(std[gi])))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score", "sigma"])
        w.writerows(out)
    print(f"LIVE R2: {len(out)} pairs -> {OUT}  (skipped {miss} missing)")


def build_kadid():
    """KADID-10k for the v2 trainability A/B: quality-oriented q=(dmos-1)/4 in [0,1].

    ⚠ SIGN FIXED 2026-08-04 — this emitted `(5 - dmos)/4` from its first commit until
    now, which is DISTORTION-oriented and the exact inverse of what its own docstring
    (and this module's stated convention, above) claimed. The trap: KADID's column is
    NAMED `dmos`, so it got the invert-a-DMOS treatment that CSIQ (`1 - DMOS`) and LIVE
    (`1 - dmos_new/100`) genuinely need — but **KADID's `dmos` is a MOS in disguise**.
    Measured from the raw crowdsourced DCR (349,800 ratings,
    `/mnt/v/dataset/kadid10k/raw_crowdsource_data.csv`), mean DCR FALLS with severity:
    L1 4.0789 -> L5 2.0072. So the flip inverted a label that was already correct, and
    the canonical lineage's `(dmos - 1)/4`
    (`build_canonical_parquets.py:288`, `fix_kadid_tid_build_pairs.py:15`) is the right
    transform. TID here uses `mos/9` and was never affected.

    Cost of the six weeks this went unnoticed: every `ext720`/`ext924`/`ext944`
    `ext_kadid.parquet` stores the target backwards, so every KADID number in the
    SOTA-944 campaign was an unsigned magnitude of a sign-flipped quantity, 110 of 188
    board bakes trained/scored ANTI-CORRELATED with KADID's real human MOS, and a
    registered gate (`KADID >= 0.70`) was passed by the three most-inverted arms and
    failed by the only correctly-oriented one. Determination + evidence:
    `benchmarks/sota944_campaign_2026-08-03.md` REGISTERED APPENDIX F.

    ⚠ THE EXISTING ext TABLES ARE NOT REGENERATED BY THIS FIX. Rebuilding them changes
    the target that ~110 existing bakes were trained against, so the rebuild is a
    deliberate act: rebuild, re-verdict, and re-annotate — do not assume a stale ext
    table matches this builder. Gate every rebuild with
    `scripts/canonical_corpus/check_target_orientation.py`, which fails on exactly this
    class of defect and currently reports the three ext KADID tables INVERTED.
    """
    SRC = "/mnt/v/dataset/kadid10k"
    OUT = f"{SRC}/kadid_pairs_ab.tsv"
    out, miss = [], 0
    with open(f"{SRC}/dmos.csv") as f:
        for r in csv.DictReader(f):
            ref = f"{SRC}/images/{r['ref_img']}"
            dist = f"{SRC}/images/{r['dist_img']}"
            if not (Path(ref).exists() and Path(dist).exists()):
                miss += 1
                continue
            # (dmos - 1)/4, NOT (5 - dmos)/4 — KADID's `dmos` is quality-oriented
            # (raw DCR falls 4.0789 -> 2.0072 across severity). See the docstring.
            out.append((ref, dist, (float(r["dmos"]) - 1.0) / 4.0))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"KADID: {len(out)} pairs -> {OUT}  (skipped {miss})")


def build_tid():
    """TID2013 for the A/B: q=mos/9 in [0,1]. Uses the pre-converted PNGs.

    BOTH sides are mixed-case and BOTH are resolved case-insensitively. The
    distorted side always was; the REFERENCE side was not — it forced
    `stem.upper() + ".png"`, which silently lost every row of the one reference
    TID ships lowercase (`i25.png`, matching its source `i25.bmp`): 120 of
    3,000 pairs. Fixed 2026-08-30; a miss is now fatal (`_require_all_resolved`).
    """
    SRC = "/mnt/v/dataset/tid2013"
    OUT = f"{SRC}/tid_pairs_ab.tsv"
    dist_by_lower = _case_insensitive_index(Path(f"{SRC}/distorted_images_png"))
    ref_by_lower = _case_insensitive_index(Path(f"{SRC}/reference_images_png"))
    out, missing, total = [], [], 0
    with open(f"{SRC}/mos_with_names.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) != 2:
                continue
            total += 1
            mos, bmp = float(parts[0]), parts[1]
            png = dist_by_lower.get(bmp.lower().replace(".bmp", ".png"))
            ref = ref_by_lower.get(bmp.split("_")[0].lower() + ".png")
            if png is None:
                missing.append(f"distorted {bmp} (as .png)")
                continue
            if ref is None:
                missing.append(f"reference for {bmp}")
                continue
            out.append((str(ref), str(png), mos / 9.0))
    _require_all_resolved("TID2013", missing, total)
    prev = Path(OUT)
    if prev.is_file():
        bak = prev.with_suffix(prev.suffix + ".pre-i25case-2026-08-30.bak")
        if not bak.exists():
            prev.rename(bak)
            print(f"TID2013: preserved previous manifest as {bak.name}")
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"TID2013: {len(out)}/{total} pairs -> {OUT}  (0 skipped — a miss is fatal)")


def build_cid22val():
    """CID22 validation set (HOLDOUT-ONLY, eval side of the A/B): human = MCOS/100."""
    SRC = "/mnt/v/dataset/cid22/CID22_validation_set"
    OUT = f"{SRC}/cid22val_pairs_ab.tsv"
    out, miss = [], 0
    with open(f"{SRC}/CID22_validation_set.csv") as f:
        for r in csv.DictReader(f):
            if r["encoder"] == "Reference":
                continue
            ref = f"{SRC}/{r['reference_img']}"
            dist = f"{SRC}/{r['distorted_img']}"
            if not (Path(ref).exists() and Path(dist).exists()):
                miss += 1
                continue
            out.append((ref, dist, float(r["MCOS"]) / 100.0))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"CID22-val: {len(out)} pairs -> {OUT}  (skipped {miss})")


def build_aic3():
    """AIC-3 CTC held-out gate (v2 backfill): ref=original/{img}.png,
    dist=decoded/{img}/{codec}_{img}_{q}.png, human=score.jnd (signed JND;
    verdict uses |SROCC| so orientation is fine). HOLDOUT-ONLY."""
    root = Path("/mnt/v/dataset/aic3_ctc_epfl")
    OUT = "/mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv"
    out, miss = [], 0
    with open(root / "decoded/info.csv") as f:
        for r in csv.DictReader(f):
            img, codec, q = r["img.name"], r["codec"], r["quality"]
            ref = root / "original" / f"{img}.png"
            dist = root / "decoded" / img / f"{codec}_{img}_{q}.png"
            if ref.exists() and dist.exists():
                out.append((str(ref), str(dist), r["score.jnd"]))
            else:
                miss += 1
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"AIC-3: {len(out)} pairs -> {OUT}  (skipped {miss})")


def build_safesyn_jpeg_full():
    """Full safesyn JPEG training mass (v2 backfill, ALL 3,218 sources — up
    from the 1,100-source lab slice). decoded_path is the .jpg BITSTREAM
    (zenjpeg-decodable; the deleted PNG cache is irrelevant to JPEG rows).
    human = gpu_ssimulacra2/100 (production ssim2-shaped target)."""
    OUT = "/mnt/v/output/zensim/v2-ab-2026-07-19/safesyn_jpeg_FULL_pairs_ab.tsv"
    out = []
    with open("/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv") as f:
        for r in csv.DictReader(f):
            if r["decoded_path"].endswith(".jpg"):
                out.append((r["source_path"], r["decoded_path"],
                            float(r["gpu_ssimulacra2"]) / 100.0))
    srcs = len({s for s, _, _ in out})
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"safesyn JPEG FULL: {len(out)} pairs from {srcs} sources -> {OUT}")


def build_aic4():
    """AIC-4 sample dataset (v2 backfill H-aic4, HOLDOUT-ONLY): 5 source
    images x 6 codecs x 10 distortion levels ~= 300 pairs. Mirrors the v1
    Rust loader `load_aic4` in `zensim-bench/examples/extract_features_372col.rs`
    exactly (same CSV columns, same PTC_images path convention, same
    human_score = signed JND `distortion` column).

    CSV: img_num,codec,dlevel,img_source,img_distorted,distortion,CI_min,CI_max
    Images: <aic4_root>/PTC_images/<img_num zero-padded to 5>/<img_source|img_distorted>
    human_score = distortion (signed JND; verdict uses |SROCC| so orientation is fine,
    matching AIC-3's score.jnd convention)."""
    CSV = "/mnt/v/backups/home/work/JPEG-AIC-4-datasets/JPEG_AIC_reconstructed_jnd_scores.csv"
    AIC4_ROOT = Path("/mnt/v/dataset/aic4_sample/JPEG_AIC-4_Sample_Dataset")
    PTC_ROOT = AIC4_ROOT / "PTC_images"
    OUT = "/mnt/v/output/zensim/v2-backfill-2026-07-20/aic4_pairs.tsv"
    out, miss = [], 0
    with open(CSV) as f:
        for r in csv.DictReader(f):
            img_num = int(r["img_num"])
            img_dir = PTC_ROOT / f"{img_num:05d}"
            ref = img_dir / r["img_source"]
            dist = img_dir / r["img_distorted"]
            if not (ref.exists() and dist.exists()):
                miss += 1
                continue
            out.append((str(ref), str(dist), float(r["distortion"])))
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"AIC-4: {len(out)} pairs -> {OUT}  (skipped {miss} missing)")


def build_konjnd_jpeg_val():
    """KonJND-1k JPEG-half val corpus (v2 backfill H-konjnd, HOLDOUT-ONLY,
    near-threshold). Mirrors the v1 Rust loader `load_konjnd` in
    `zensim-bench/examples/extract_features_372col.rs` EXACTLY, filtered to
    `Compression type == JPEG` (BPG has no local decoder -- documented gap,
    see docs/V2_EXPERIMENT_PLAN_2026-07-20.md).

    subjective_ratings.csv columns: image_id, Compression type, No. of
    ratings, mean, std, ratings. human_score = raw `mean` PJND threshold
    (NOT normalized -- matches the documented val/konjnd.parquet convention,
    range ~[6,90], nominal [22,70] per project memory
    feedback_konjnd_human_score_two_columns.md). dist filename convention:
    `{stem}_JPEG_{round(mean):03}.jpg` under `<base>/jpeg/`.

    NOTE: this is NOT the same corpus as `konjnd-dense` (the CVVDP+IWSSIM
    'active-mix' 20-samples-per-ref TRAIN corpus referenced as T-konjnd in
    the v2 plan) -- that corpus has no raw pixel path columns in any local
    parquet (only ref_basename, duplicated 20x with no per-row quality-level
    discriminator) and its target requires zenmetrics-side CVVDP scoring.
    See benchmarks/v2_backfill_local_2026-07-20.md for the full flag."""
    BASE = Path("/mnt/v/datasets/KonJND-1k/KonJND-1k")
    OUT = "/mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv"
    out, miss = [], 0
    with open(BASE / "subjective_ratings.csv") as f:
        for r in csv.DictReader(f):
            if r["Compression type"] != "JPEG":
                continue
            image_id = r["image_id"]
            stem = image_id[:-4] if image_id.endswith(".png") else image_id
            mean_threshold = float(r["mean"])
            level = max(1, min(100, round(mean_threshold)))
            dist = BASE / "jpeg" / f"{stem}_JPEG_{level:03d}.jpg"
            ref = BASE / "source_image" / image_id
            if not (ref.exists() and dist.exists()):
                miss += 1
                continue
            out.append((str(ref), str(dist), mean_threshold))
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"KonJND-1k JPEG val: {len(out)} pairs -> {OUT}  (skipped {miss} missing)")


def build_sdr25():
    """JPEG-AI-SDR25 scoreable subset (v2 backfill H-sdr25, HOLDOUT-ONLY,
    HQ zone q75-100 weak zone). The dataset's ~95k rows are TRIPLET
    COMPARISON RESPONSES (which-side-is-more-distorted judgments against a
    pivot original), NOT independent labeled pairs -- they were already
    reconstructed into an absolute per-stimulus JND scale via ordered-probit
    joint MLE by `scripts/v_next/reconstruct_sdr25_jnd.py` (T0 eval anchor,
    BUILT 2026-07-02, docs/DATA_SPLITS.md:108). This builder does NOT
    re-derive that reconstruction (owner = reconstruct_sdr25_jnd.py) -- it
    only builds the (ref,dist,human_score) pairs layer on top of its output.

    Of the 116 reconstructed stimuli (5 images x up to 6 codecs), only the
    JPEG-AI codec (codec==6) subset has locally-available pixels -- 5 images
    x 10 distortion levels = 50 pairs. The other 5 codecs (AVIF/JPEG-1/
    JPEG-2000/JPEG-XL/VVC) are 'anchor' stimuli whose bitstreams are NOT in
    the public zip (sparse/inconsistent dlevel coverage per image confirms
    this -- verified 2026-07-20 via direct parquet inspection). Reproduces
    (byte-for-byte path convention) the pre-existing
    /mnt/v/output/zensim-multicodec-probe/sdr25_eval_pairs.tsv (50 rows,
    verified identical 2026-07-20) -- always uses the PTC crop file for a
    given (img_num, dlevel) regardless of whether the reconstructed row's
    `filename` says BTC_ or PTC_ (a deliberate prior-session choice: PTC is
    the single crop file used for feature extraction; BTC/PTC only
    distinguish which DISPLAY condition a given triplet response came from).

    ref  = crops_sources/PTC_{img:05d}_0ref_00.png
    dist = PTC_JPEG-AI_images/{img:05d}/PTC_{img:05d}_JPEG-AI_{dlevel:02d}.png
    human_score = q_jnd (ordered-probit JND-scale distortion magnitude,
    0=original/best, higher=more distorted -- matches AIC-3/AIC-4 convention
    in this file; downstream eval uses |SROCC|)."""
    import pyarrow.parquet as pq
    RECON = "/mnt/v/output/zensim-multicodec-probe/sdr25_jnd_reconstructed_2026-07-02.parquet"
    ROOT = Path("/mnt/v/datasets/jpeg-ai-sdr25/dataset-JPEG-AI-SDR25")
    OUT = "/mnt/v/output/zensim/v2-backfill-2026-07-20/sdr25_pairs.tsv"
    df = pq.read_table(RECON).to_pandas()
    jai = df[df["codec"] == 6]  # codec 6 = JPEG-AI -- the only pixel-available codec
    out, miss = [], 0
    for _, r in jai.iterrows():
        img, dlevel = int(r["img_num"]), int(r["dlevel"])
        ref = ROOT / "crops_sources" / f"PTC_{img:05d}_0ref_00.png"
        dist = ROOT / "PTC_JPEG-AI_images" / f"{img:05d}" / f"PTC_{img:05d}_JPEG-AI_{dlevel:02d}.png"
        if not (ref.exists() and dist.exists()):
            miss += 1
            continue
        out.append((str(ref), str(dist), float(r["q_jnd"])))
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"SDR25 (JPEG-AI scoreable subset): {len(out)} pairs -> {OUT}  (skipped {miss} missing)")


def build_cid22_train201():
    """CID22 train-only 201-ref subset (v2 backfill T-cid201, TRAINABLE --
    ssim2-anchored, NEVER human MCOS per CLAUDE.md 'CID22 is VALIDATION-ONLY').

    Reuses the ALREADY-BUILT raw-pixel-path workspace TSV
    `canonical-2026-05-21/_workspace/cid22_train_ssim2.tsv` (produced by
    `scripts/canonical_corpus/v11_extract_cid22_train.py` +
    `v11_cid22_train_backfill_cvvdp_iwssim.py`; this builder does NOT
    re-derive the 201-ref split or re-score ssim2, it only reformats the
    existing (ref_path, dist_path, ..., ssim2_gpu) table into this file's
    (ref_path, dist_path, human_score) convention). human_score =
    ssim2_gpu / 100, matching the documented invariant
    `cid22_train_norm.human_score == ssim2_gpu/100 exactly` (docs/DATA_SPLITS.md:99).

    SAFETY: verified 2026-07-20 -- the 201 unique ref_basenames here have
    ZERO overlap with the 49-ref CID22_validation_set.csv `reference_img`
    basenames (the sacred human-MOS holdout). Source images live at
    /mnt/v/dataset/cid22/CID22/ (the broader CID22 library), NOT
    /mnt/v/dataset/cid22/CID22_validation_set/."""
    SRC = "/mnt/v/zen/zensim-training/canonical-2026-05-21/_workspace/cid22_train_ssim2.tsv"
    OUT = "/mnt/v/output/zensim/v2-backfill-2026-07-20/cid22_train201_pairs.tsv"
    out, miss = [], 0
    with open(SRC) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ref, dist = r["ref_path"], r["dist_path"]
            if not (Path(ref).exists() and Path(dist).exists()):
                miss += 1
                continue
            out.append((ref, dist, float(r["ssim2_gpu"]) / 100.0))
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["ref_path", "dist_path", "human_score"])
        w.writerows(out)
    print(f"CID22-train-201: {len(out)} pairs -> {OUT}  (skipped {miss} missing)")


BUILDERS = {
    "csiq": build_csiq,
    "live": build_live,
    "kadid": build_kadid,
    "tid": build_tid,
    "cid22val": build_cid22val,
    "aic3": build_aic3,
    "safesyn_jpeg_full": build_safesyn_jpeg_full,
    "aic4": build_aic4,
    "konjnd_jpeg_val": build_konjnd_jpeg_val,
    "sdr25": build_sdr25,
    "cid22_train201": build_cid22_train201,
}

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else ""
    if name not in BUILDERS:
        print(f"builders: {list(BUILDERS)}")
        sys.exit(2)
    BUILDERS[name]()
