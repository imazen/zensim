#!/usr/bin/env python3
"""rev4 featbank-extract: set registry, path constants and row-meta helpers.

Split out of convert_cache.py (committed-file 30 KB limit). Holds no pipeline
logic — only configuration, corpus-name derivations and per-source-row meta
extraction used by both the parquet-convert and fresh-extract paths.

Path note (review correction 4): pairs/raw files that carry a held-out
`human_score` were moved under /var/tmp/rev4-featbank/_sealed/ on 2026-09-23
(move, not regenerate — sha256s unchanged). Sealed sets reference them there.
"""
import json
import re
from functools import cache
from pathlib import Path

BANK = Path("/var/tmp/rev4-featbank/bank")
RAW = Path("/var/tmp/rev4-featbank/raw")
PAIRS = Path("/var/tmp/rev4-featbank/pairs")
SEALED_RAW = Path("/var/tmp/rev4-featbank/_sealed/raw")
SEALED_PAIRS = Path("/var/tmp/rev4-featbank/_sealed/pairs")
RECOVERY = Path("/var/tmp/zensim-validation-2026-09-14/baseline-recovery")
CEILING = Path.home() / "work/zensim-validation-2026-09-13/ceiling/final"
DERIVED_AUDIT = Path("/var/tmp/rev4-featbank/ceiling_human_audit.jsonl")
CEIL_FEATS = Path("/var/tmp/rev4-featbank/ceiling_human_feats.csv")

INPUT_CONTRACT = "legacy-rgb8"
ERA = "ceiling_rev3"
TOKENS = "basic+peaks+masked+iw+v2+append+append2"
FEATURE_SET_ID = f"{TOKENS}@w944/{ERA}#b782e349"
SIDECAR_NAME = f"features__{TOKENS}__{ERA}__b782e349.parquet"
EXTRACTOR_SHA256 = "7c7ffbbfa033e8ca1a8f103d472b61ccde061c2394b03d519af2852ee8eeda87"
ROW_GROUP = 65536

# Registered structural-zero feature ids for the w944 ceiling_rev3 era
# (39 int64 all-zero columns in the Rev3 parquet caches).
STRUCTURAL_ZERO_IDS = [
    720, 721, 754, 755, 756, 757, 758, 759, 760, 761, 762, 763, 764, 765,
    766, 767, 768, 769, 770, 771, 772, 805, 806, 822, 823, 856, 857, 873,
    874, 907, 908, 927, 928, 932, 933, 937, 938, 942, 943,
]
POPULATED_IDS = [i for i in range(944) if i not in set(STRUCTURAL_ZERO_IDS)]

CID22_A_REFS = {
    "1189261", "1531677", "159550", "1624487", "162520", "164595", "2079234",
    "21169144185_3f7977cb5a_o", "225228", "2389166", "2936831", "3316926",
    "3653963", "373965", "3762075", "4215100", "6078297", "6292444", "70497",
    "7062219", "844297", "pexels-photo-2686358", "pexels-photo-2802032",
    "pexels-photo-4210863", "ularapi_Semarang_City_Logo",
}
CID22_B_REFS = {
    "1025469", "1044329", "1279330", "1418519", "1420710", "1475938",
    "1544947", "2190188", "2253934", "2670327", "2736139", "2775196",
    "2887497", "297394", "3156482", "3316926_opo25u", "3637739", "382297",
    "5055743", "5458393", "7552578", "792079", "adriankierman-report-page",
    "pexels-photo-1933873",
}
# select() compares basename_norm() = lowercased stem; one A ref is mixed case
# (ularapi_Semarang_City_Logo) so membership tests use the lowered sets.
CID22_A_REFS_L = {x.lower() for x in CID22_A_REFS}
CID22_B_REFS_L = {x.lower() for x in CID22_B_REFS}
KONFIG_ORIGIN_SPLIT = {
    "SRC01": "val", "SRC03": "val", "SRC06": "train", "SRC07": "test",
    "SRC09": "test", "SRC17": "test", "SRC28": "train", "SRC31": "val",
    "SRC45": "val", "SRC50": "train",
}


def basename_norm(p):
    return Path(p).stem.lower()


def audit_row_id(rec):
    for k, v in rec.get("extra_targets", []):
        if k == "row_id":
            return int(v)
    return None


def derivations(set_name, dist_path):
    """(codec, knob) derived from the distorted path, per-corpus conventions."""
    p = Path(dist_path)
    stem = p.stem
    if set_name.startswith("kadid"):
        parts = stem.split("_")  # I07_01_03 -> type, level
        return f"kadid_{parts[1]}", parts[2]
    if set_name.startswith("tid"):
        parts = stem.split("_")  # i01_04_2 -> type, level
        return f"tid_{parts[1]}", parts[2]
    if set_name.startswith("konfig"):
        parts = stem.split("_")  # SRC01_colordiffusion_0
        return parts[1], parts[-1]
    if set_name.startswith("konjnd_bpg"):
        q = stem.rsplit("_", 1)[1]  # SRC0505_BPG_051 -> rung
        return "bpg", q
    if set_name.startswith("konjnd_jpeg"):
        q = stem.rsplit("_", 1)[1]  # SRC0001_JPEG_031 -> rung
        return "jpeg", q
    if set_name.startswith("cid22"):
        return p.parent.name, stem  # compressed/<ref>/<codec>/<knob>.<ext>
    if set_name == "aic3":
        parts = stem.split("_")  # AVIF_00001_1192x832_1
        return parts[0].lower(), parts[-1]
    if set_name == "aic4":
        parts = stem.split("_")  # PTC_00002_AVIF_02
        return parts[2].lower(), parts[3]
    if set_name == "csiq":
        # 1600.AWGN.1.png -> codec = distortion dir, knob = level after last dot
        return p.parent.name, stem.rsplit(".", 1)[1]
    if set_name == "mcljci":
        return "jpeg", stem.rsplit("_", 1)[1]
    return p.parent.name, stem


def meta_from_parquet(src, corpus):
    rb = src.column("ref_basename").to_pylist()
    rid = src.column("row_id").to_pylist()
    has_ck = "codec" in src.schema.names
    codecs = src.column("codec").to_pylist() if has_ck else [None] * len(rb)
    knobs = src.column("knob").to_pylist() if has_ck else [None] * len(rb)
    return [
        {"row_id": rid[i], "ref_group": f"{corpus}:{Path(str(rb[i])).stem}",
         "codec": codecs[i], "knob": knobs[i]}
        for i in range(len(rb))
    ]


def ceiling_meta(src, inputs_rows):
    inp = {r["row_id"]: r for r in inputs_rows}
    meta = []
    for i in src.column("row_id").to_pylist():
        r = inp[i]
        stem = Path(r["distorted"]).stem
        meta.append({
            "row_id": i,
            "ref_group": r["origin"],
            "codec": r["family"],
            "knob": stem.split("_")[-1],
        })
    return meta


@cache
def _ceiling_sets():
    inputs_rows = json.loads((CEILING / "INPUTS.json").read_text())["rows"]
    sel = {
        "kadid_train": sorted(
            r["row_id"] for r in inputs_rows
            if r["task"] == "human" and r["corpus"] == "kadid"
            and r["role"] in ("fit", "dev")
        ),
        "tid2013": sorted(
            r["row_id"] for r in inputs_rows
            if r["task"] == "human" and r["corpus"] == "tid"
        ),
        "kadid_select": sorted(
            r["row_id"] for r in inputs_rows
            if r["task"] == "human" and r["corpus"] == "kadid"
            and r["role"] == "test"
        ),
    }
    return inputs_rows, sel


_BIN_COMMIT = f"unknown(binary {EXTRACTOR_SHA256})"
_CEIL_NOTE = "re-derived: extract-native-admission 7c7ffbbf… --audit-jsonl; verified 25/25 vs recorded probe audit; 11,125 rows bit-exact vs ceiling parquets"
_FRESH_NOTE = "extract-native-admission --audit-jsonl (canonical zen_decode; parity vs konfig_pairs.tsv dist_px_sha256 verified 5/5)"

_SET_COMMON_FRESH = {
    "build_commit": _BIN_COMMIT,
    "binary_sha256": EXTRACTOR_SHA256,
    "pixel_hash_source": _FRESH_NOTE,
}

_SET_COMMON_CEILING = {
    "path": "parquet",
    "audit": DERIVED_AUDIT,
    "pairs": Path("/var/tmp/rev4-featbank/ceiling_human_pairs.tsv"),
    "build_commit": _BIN_COMMIT,
    "binary_sha256": EXTRACTOR_SHA256,
    "admission_sha256": None,
    "input_files_note": "per-input file sha256s in ceiling/final/INPUTS.json (sha256 44a574b7…)",
    "pixel_hash_source": _CEIL_NOTE,
}

SETS = {
    # ---- parquet-convert (existing Rev3 caches) ----
    "cid22_train": {
        "path": "parquet",
        "parquets": [RECOVERY / "cid22-train944.parquet"],
        "audit": RECOVERY / "cid22-train-audit.jsonl",
        "pairs": None,
        "rows": 17611,
        "role": "train",
        "labels": "ssim2_oracle",
        "label_scale": "ssim2 raw (peer-SSIM2 oracle; observed 3.01–94.11)",
        "label_note": "stored human_score = peer-SSIM2 oracle target; copied to labels__ssim2_oracle.parquet",
        "row_meta": lambda src: meta_from_parquet(src, "cid22"),
        "build_commit": _BIN_COMMIT,
        "binary_sha256": EXTRACTOR_SHA256,
        "admission_sha256": "0a64ad74ce5946bff3ef8abb72137f26339d480541df598e38f818a65c4ae2b9",
        "pairs_sha256": "0ec4576a41415424b99dcacd0f45ed5a543a72d32410f4e88f289b392da3bc2b",
        "input_files_note": "per-input file sha256s in CID22_ADMISSION.json (sha256 0a64ad74…)",
        "pixel_hash_source": "cid22-train-audit.jsonl (canonical-feature-audit-v1)",
    },
    "safesyn": {
        "path": "parquet",
        "parquets": [RECOVERY / "safesyn-train944.parquet"],
        "audit": RECOVERY / "safesyn-train-audit.jsonl",
        "pairs": None,
        "rows": 196086,
        "role": "train",
        "labels": "ssim2_oracle",
        "label_scale": "ssim2 raw (peer-SSIM2 oracle; observed −743.9…100.0)",
        "label_note": "stored human_score = peer-SSIM2 oracle; original_oracle carried alongside",
        "row_meta": lambda src: meta_from_parquet(src, "safesyn"),
        "build_commit": _BIN_COMMIT,
        "binary_sha256": EXTRACTOR_SHA256,
        "admission_sha256": "9ced0f04adecfcabebf05c136c05f1c4849bfbc9f798734af642c4d4ee70fd4b",
        "pairs_sha256": "5a53976070a5e21b2bb7fe0d05f58b510b93e141dd9207dd3e2cd337fd15cd3b",
        "input_files_note": "per-input file sha256s in SAFESYN_ADMISSION.json (sha256 9ced0f04…)",
        "pixel_hash_source": "safesyn-train-audit.jsonl (canonical-feature-audit-v1)",
    },
    "kadid_train": {
        "parquets": [CEILING / "human_fit.parquet", CEILING / "human_dev.parquet"],
        "rows": 5000,
        "role": "train",
        "labels": "human",
        "label_scale": "human (KADID dmos-derived, source units)",
        "label_note": "TRAIN-role human scores copied through unanalysed",
        "row_meta": lambda src: ceiling_meta(src, _ceiling_sets()[0]),
        "select_row_ids": lambda: _ceiling_sets()[1]["kadid_train"],
        **_SET_COMMON_CEILING,
    },
    "tid2013": {
        "parquets": [CEILING / "human_fit.parquet"],
        "rows": 3000,
        "role": "train",
        "labels": "human",
        "label_scale": "human (TID2013 mos, source units)",
        "label_note": "TRAIN-role human scores copied through unanalysed",
        "row_meta": lambda src: ceiling_meta(src, _ceiling_sets()[0]),
        "select_row_ids": lambda: _ceiling_sets()[1]["tid2013"],
        **_SET_COMMON_CEILING,
    },
    "kadid_select": {
        "parquets": [CEILING / "human_test.parquet"],
        "rows": 3125,
        "role": "fold+potential",
        "labels": "human",
        "label_scale": "human (KADID dmos-derived, source units)",
        "label_note": "KADID SELECT is potential-exposed (ruling D1): labels usable by fit lanes; copy-through only",
        "row_meta": lambda src: ceiling_meta(src, _ceiling_sets()[0]),
        "select_row_ids": lambda: _ceiling_sets()[1]["kadid_select"],
        **_SET_COMMON_CEILING,
    },
    # ---- fresh-extract ----
    "kadid_terminal": {
        "path": "extract",
        "pairs": SEALED_PAIRS / "kadid_terminal.tsv",
        "feats": SEALED_RAW / "kadid_terminal.feats.csv",
        "audit": SEALED_RAW / "kadid_terminal.audit.jsonl",
        "rows": 2000,
        "role": "confirmation",
        "labels": "none",
        "label_scale": "none (pixels only)",
        "label_note": "KADID TERMINAL refs {7,9}: confirmation-only, pixels only; no labels emitted. Sources sealed at /var/tmp/rev4-featbank/_sealed/ (held-out human_score replicas, review corr. 4)",
        **_SET_COMMON_FRESH,
    },
    "konfig_train": {
        "path": "extract",
        "pairs": PAIRS / "konfig_all.tsv",
        "feats": RAW / "konfig_all.feats.csv",
        "audit": RAW / "konfig_all.audit.jsonl",
        "rows": 327,
        "role": "train",
        "labels": "human",
        "label_scale": "human (KonFiG q_jnd-derived, source units)",
        "label_note": "TRAIN-role (origin split SRC06/28/50); q_jnd-derived quality score copied through",
        "select": lambda i, pr, a: KONFIG_ORIGIN_SPLIT[
            re.search(r"(SRC\d+)", a["reference"]).group(1)] == "train",
        **_SET_COMMON_FRESH,
    },
    "konfig_val": {
        "path": "extract",
        "pairs": PAIRS / "konfig_all.tsv",
        "feats": RAW / "konfig_all.feats.csv",
        "audit": RAW / "konfig_all.audit.jsonl",
        "rows": 436,
        "role": "fold+potential",
        "labels": "human",
        "label_scale": "human (KonFiG q_jnd-derived, source units)",
        "label_note": "originsplit_val (SRC01/03/31/45): potential-exposed (ruling D1); labels usable",
        "select": lambda i, pr, a: KONFIG_ORIGIN_SPLIT[
            re.search(r"(SRC\d+)", a["reference"]).group(1)] == "val",
        **_SET_COMMON_FRESH,
    },
    "konjnd_bpg_train": {
        "path": "extract",
        "pairs": PAIRS / "konjnd_bpg_train.tsv",
        "feats": RAW / "konjnd_bpg_train.feats.csv",
        "audit": RAW / "konjnd_bpg_train.audit.jsonl",
        "rows": 8060,
        "role": "train",
        "labels": "ssim2_oracle",
        "label_scale": "ssim2/100 (gpu_ssimulacra2/100; observed −0.649…0.962)",
        "label_note": "human_score = gpu_ssimulacra2/100 (TRAIN oracle target); copied through",
        **_SET_COMMON_FRESH,
    },
    "konjnd_bpg_val": {
        "path": "extract",
        "pairs": PAIRS / "konjnd_bpg_val.tsv",
        "feats": RAW / "konjnd_bpg_val.feats.csv",
        "audit": RAW / "konjnd_bpg_val.audit.jsonl",
        "rows": 2020,
        "role": "fold",
        "labels": "ssim2_oracle",
        "label_scale": "ssim2/100 (gpu_ssimulacra2/100; observed −0.649…0.962)",
        "label_note": "within-BPG-leg val refs (srcnum%10 in {8,9}): LODO-exposed at fold eval; oracle labels usable",
        **_SET_COMMON_FRESH,
    },
    "cid22_a25": {
        "path": "extract",
        "pairs": SEALED_PAIRS / "cid22val.tsv",
        "feats": SEALED_RAW / "cid22val.feats.csv",
        "audit": SEALED_RAW / "cid22val.audit.jsonl",
        "rows": 2192,
        "role": "fold+potential",
        "labels": "human",
        "label_scale": "CID22 human MCOS/100",
        "label_note": "CID22 human MCOS/100 (build_cid22val sets human_score = MCOS/100); potential-exposed, labels usable. Source file mixes A+B rows -> sealed at /var/tmp/rev4-featbank/_sealed/",
        "select": lambda i, pr, a: basename_norm(a["reference"]) in CID22_A_REFS_L,
        **_SET_COMMON_FRESH,
    },
    "cid22_b": {
        "path": "extract",
        "pairs": SEALED_PAIRS / "cid22val.tsv",
        "feats": SEALED_RAW / "cid22val.feats.csv",
        "audit": SEALED_RAW / "cid22val.audit.jsonl",
        "rows": 2100,
        "role": "confirmation",
        "labels": "none",
        "label_scale": "none (pixels only)",
        "label_note": "CID22-B 24 refs (ruling D4): pixels only; NO labels emitted or read. Sources sealed at /var/tmp/rev4-featbank/_sealed/",
        "select": lambda i, pr, a: basename_norm(a["reference"]) in CID22_B_REFS_L,
        **_SET_COMMON_FRESH,
    },
    "aic3": {
        "path": "extract",
        "pairs": PAIRS / "aic3.tsv",
        "feats": RAW / "aic3.feats.csv",
        "audit": RAW / "aic3.audit.jsonl",
        "rows": 600,
        "role": "fold+potential",
        "labels": "human",
        "label_scale": "human (AIC-3 CTC, source units)",
        "label_note": "AIC-3 CTC: fold set + potential-exposed (ruling D1/D2); labels usable",
        **_SET_COMMON_FRESH,
    },
    "aic4": {
        "path": "extract",
        "pairs": SEALED_PAIRS / "aic4.tsv",
        "feats": SEALED_RAW / "aic4.feats.csv",
        "audit": SEALED_RAW / "aic4.audit.jsonl",
        "rows": 300,
        "role": "confirmation",
        "labels": "none",
        "label_scale": "none (pixels only)",
        "label_note": "AIC-4 sample (confirmation): pixels only; NO labels emitted or read. Sources sealed at /var/tmp/rev4-featbank/_sealed/",
        **_SET_COMMON_FRESH,
    },
    "konjnd_jpeg_select": {
        "path": "extract",
        "pairs": SEALED_PAIRS / "konjnd_jpeg_val.tsv",
        "feats": SEALED_RAW / "konjnd_jpeg_val.feats.csv",
        "audit": SEALED_RAW / "konjnd_jpeg_val.audit.jsonl",
        "rows": 404,
        "role": "confirmation",
        "labels": "none",
        "label_scale": "none (pixels only)",
        "label_note": "KonJND JPEG SELECT (confirmation): pixels only; NO labels emitted or read. Sources sealed at /var/tmp/rev4-featbank/_sealed/",
        "select": lambda i, pr, a: int(re.search(r"SRC(\d+)", a["reference"]).group(1)) % 10 not in (7, 9),
        **_SET_COMMON_FRESH,
    },
    "konjnd_jpeg_terminal": {
        "path": "extract",
        "pairs": SEALED_PAIRS / "konjnd_jpeg_val.tsv",
        "feats": SEALED_RAW / "konjnd_jpeg_val.feats.csv",
        "audit": SEALED_RAW / "konjnd_jpeg_val.audit.jsonl",
        "rows": 100,
        "role": "confirmation",
        "labels": "none",
        "label_scale": "none (pixels only)",
        "label_note": "KonJND JPEG TERMINAL {7,9} (confirmation): pixels only; NO labels emitted or read. Sources sealed at /var/tmp/rev4-featbank/_sealed/",
        "select": lambda i, pr, a: int(re.search(r"SRC(\d+)", a["reference"]).group(1)) % 10 in (7, 9),
        **_SET_COMMON_FRESH,
    },
    "csiq": {
        "path": "extract",
        "pairs": SEALED_PAIRS / "csiq.tsv",
        "feats": SEALED_RAW / "csiq.feats.csv",
        "audit": SEALED_RAW / "csiq.audit.jsonl",
        "rows": 866,
        "role": "confirmation",
        "labels": "none",
        "label_scale": "none (pixels only)",
        "label_note": "CSIQ (confirmation): pixels only; NO labels emitted or read. Sources sealed at /var/tmp/rev4-featbank/_sealed/",
        **_SET_COMMON_FRESH,
    },
    "mcljci": {
        "path": "extract",
        "pairs": SEALED_PAIRS / "mcljci.tsv",
        "feats": SEALED_RAW / "mcljci.feats.csv",
        "audit": SEALED_RAW / "mcljci.audit.jsonl",
        "rows": 5000,
        "role": "confirmation",
        "labels": "none",
        "label_scale": "none (pixels only)",
        "label_note": "MCL-JCI (confirmation-only per ruling D3 default): pixels only; NO labels emitted or read. Sources sealed at /var/tmp/rev4-featbank/_sealed/. Pairs built from /var/tmp/datasets/mcl-jci/pairs_mcljci_src.tsv (5000 rows, true-source refs; sha256 b7f4667d…, row_id column appended 2026-09-23)",
        **_SET_COMMON_FRESH,
    },
}
