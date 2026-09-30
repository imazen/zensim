"""Instrument v2 wide tables (R915 sampling) and the teacher-leg input pin.

Governing record: benchmarks/rev4_featpot_v2_amendment_2026-09-30.md, revision R1.

  python v2_wide.py pin            # hash every teacher-leg input into TEACHER_PIN (reads no label)
  python v2_wide.py build          # all variants; or --variant real|p1|p2|p3

Legs: five human sources (full table for evaluation, plus fit/dev by a label-free reference hash, plus the
merged human training tables for each held-out source) and two R915 teacher legs (SafeSyn, CID22-train;
fit/dev = R915's own reference split). pixels_identical keys are dropped everywhere. Column layout and
variants: see v2_common. Human targets: per-source affine q0.001/q0.999 -> [0, 100]; teacher targets: raw
signed SSIMULACRA2 (the bank's ssim2_oracle), as R915 trained with --target-scale 1.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import p2_data
import restore_data
from admit_bank import PROMOTED_BANK
from data import load as load_bank
from linear_probe import panel_batch
from v2_common import (COL_GMSD, COL_GMSM, COL_ORACLE, ORACLE_SEED_BASE, ORACLE_SIGMA, PERM_SEED_BASE,
                       R915_TABLES, SOURCE_ORDER, SOURCES, TEACHER_PIN, TEACHERS, V2, VARIANTS, WIDTH,
                       human_dev, sha)

RESEARCH_IDS = list(range(944, 1825))
RESTORE_BANK = Path("/var/tmp/restore-cuts/bank")
PARTB_FILES = {"csfw_dvifm": "features__csfw_dvifm.parquet", "c1c4": "features__rev4c1c4.parquet",
               "gmsbank": "features__gmsbank.parquet"}
RESTORE_FILES = {f: f"features__restore_{f}.parquet" for f in ("mapdev", "z1max", "gmsnative", "dvifmgate")}
FEATURES = [f"f{i}" for i in range(WIDTH)]
BANK_ID = "basic+peaks+masked+iw+v2+append+append2@w944/ceiling_rev3#b782e349"


# ---------------------------------------------------------------- teacher pin (no label read)
def teacher_files(bank_name: str, leg: str) -> dict:
    d = PROMOTED_BANK / bank_name
    manifest = json.loads((d / "_MANIFEST.json").read_text())
    base = next(f for f in manifest["files"] if f.startswith("features__basic"))
    label = next(f for f in manifest["files"] if f.startswith("labels__"))
    files = {"manifest": d / "_MANIFEST.json", "keys": d / "keys.parquet", "base": d / base, "labels": d / label,
             **{f"partb_{k}": d / v for k, v in PARTB_FILES.items()},
             "restore_manifest": RESTORE_BANK / bank_name / "_MANIFEST_restore.json",
             **{f"restore_{k}": RESTORE_BANK / bank_name / v for k, v in RESTORE_FILES.items()},
             "peer": p2_data.PEER / f"{bank_name}.parquet",
             "r915_fit": R915_TABLES / f"{leg}_fit.parquet", "r915_dev": R915_TABLES / f"{leg}_development.parquet"}
    return files


def write_pin() -> None:
    pin = {"schema": "rev4-featpot-v2-teacher-pin-v1", "label": "POTENTIAL — ceiling, not a model score",
           "note": "hashes only; no label value read when this pin was written",
           "r915_dedup_manifest_sha256": sha(R915_TABLES / "_MANIFEST.json"),
           "peer_manifest_sha256": p2_data.PEER_MANIFEST_SHA, "legs": {}}
    for leg, (bank_name, r915_leg, _) in TEACHERS.items():
        files = teacher_files(bank_name, r915_leg)
        pin["legs"][leg] = {"bank_set": bank_name, "r915_leg": r915_leg,
                            "files": {k: {"path": str(p), "sha256": sha(p)} for k, p in files.items()}}
    TEACHER_PIN.write_text(json.dumps(pin, indent=1) + "\n")
    print(json.dumps({"pin": str(TEACHER_PIN), "sha256": sha(TEACHER_PIN)}))


def pinned(leg: str) -> dict:
    pin = json.loads(TEACHER_PIN.read_text())
    files = {}
    for key, rec in pin["legs"][leg]["files"].items():
        path = Path(rec["path"])
        if sha(path) != rec["sha256"]:
            raise ValueError(f"{leg}: {key} changed after the teacher pin")
        files[key] = path
    return files


# ---------------------------------------------------------------- loaders
def human_set(name: str) -> pd.DataFrame:
    """Admitted bank rows (f0-f943 + target) joined to every research column and the peer pair."""
    frame, _ = load_bank(name)
    side = restore_data.side_join(name, RESEARCH_IDS)          # verifies every registered pin
    peer, _ = restore_data.peer_side(name)
    peer = peer.rename(columns={"gmsd": f"f{COL_GMSD}", "gmsm": f"f{COL_GMSM}"})
    out = frame.merge(side, on="pair_key", how="left", sort=False, validate="many_to_one", indicator=True)
    if not (out._merge == "both").all():
        raise ValueError(f"{name}: research sidecar coverage mismatch")
    out = out.drop(columns="_merge").merge(peer, on="pair_key", how="left", sort=False,
                                           validate="many_to_one", indicator=True)
    if not (out._merge == "both").all():
        raise ValueError(f"{name}: peer coverage mismatch")
    out = out.drop(columns="_merge")
    flags = restore_data.identical_flags(name)
    keep = ~out.pair_key.map(flags).astype(bool).to_numpy()
    out = out.loc[keep].reset_index(drop=True)
    out["member_set"] = name
    return out


def teacher_set(leg: str) -> pd.DataFrame:
    bank_name, r915_leg, _ = TEACHERS[leg]
    files = pinned(leg)
    manifest = json.loads(files["manifest"].read_text())
    ids, zeros = manifest["populated_feature_ids"], manifest["structural_zero_feature_ids"]
    if sorted(ids + zeros) != list(range(944)):
        raise ValueError(f"{leg}: feature-ID partition mismatch")
    keys = pq.read_table(files["keys"], columns=["pair_key", "ref_group", "pixels_identical"]).to_pandas()
    base = pq.read_table(files["base"]).to_pandas()
    if base.columns.tolist() != ["pair_key"] + [f"f{i}" for i in ids]:
        raise ValueError(f"{leg}: base schema mismatch")
    labels = pq.read_table(files["labels"], columns=["pair_key", "source_row_id", "ssim2_oracle"]).to_pandas()
    out = labels.merge(keys, on="pair_key", how="left", sort=False, validate="many_to_one")  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
    out = out.merge(base, on="pair_key", how="left", sort=False, validate="many_to_one")  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
    for z in zeros:
        out[f"f{z}"] = np.float32(0.0)
    for key in (*(f"partb_{k}" for k in PARTB_FILES), *(f"restore_{k}" for k in RESTORE_FILES)):
        side = pq.read_table(files[key]).to_pandas()
        if side.pair_key.duplicated().any():
            raise ValueError(f"{leg}: duplicate keys in {key}")
        out = out.merge(side, on="pair_key", how="left", sort=False, validate="many_to_one")  # joinsafety-ok: pair_key-keyed sidecar join, coverage asserted below
    peer = pq.read_table(files["peer"], columns=["pair_key", "gmsd", "gmsm"]).to_pandas().drop_duplicates("pair_key")
    peer = peer.rename(columns={"gmsd": f"f{COL_GMSD}", "gmsm": f"f{COL_GMSM}"})
    out = out.merge(peer, on="pair_key", how="left", sort=False, validate="many_to_one")  # joinsafety-ok: pair_key-keyed peer join, coverage asserted below
    need = [f"f{i}" for i in range(1827)]
    if out[need].isna().any().any() or not np.isfinite(out[need].to_numpy(np.float32)).all():
        raise ValueError(f"{leg}: missing or nonfinite feature after joins")
    out = out.loc[~out.pixels_identical.astype(bool)].copy()
    fit = set(pq.read_table(files["r915_fit"], columns=["ref_basename"])["ref_basename"].to_pylist())
    dev = set(pq.read_table(files["r915_dev"], columns=["ref_basename"])["ref_basename"].to_pylist())
    stripped = out.ref_group.astype(str).str.split(":", n=1).str[1]
    out["split"] = np.where(stripped.isin(fit), "fit", np.where(stripped.isin(dev), "dev", "excluded"))
    out = out.loc[out.split != "excluded"].rename(columns={"ref_group": "ref_basename",
                                                           "ssim2_oracle": "target"}).reset_index(drop=True)
    out["member_set"] = bank_name
    return out


# ---------------------------------------------------------------- build
def add_oracles(out: pd.DataFrame, y01: np.ndarray, leg_index: int) -> dict:
    rng = np.random.default_rng(ORACLE_SEED_BASE + leg_index)
    sd = float(np.std(y01))
    info = {}
    for name in ("oracle_lo", "oracle_hi"):
        out[f"f{COL_ORACLE[name]}"] = (y01 + rng.normal(0.0, ORACLE_SIGMA[name] * sd, len(out))).astype(np.float32)
    return info


def write(frame: pd.DataFrame, path: Path, human_score: np.ndarray, note: str) -> dict:
    view = frame[["ref_basename"] + FEATURES].copy()
    view.insert(1, "human_score", human_score)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(view, preserve_index=False), path, compression="zstd")
    Path(f"{path}.manifest.json").write_text(json.dumps({
        "source_bank_feature_set_id": BANK_ID,
        "composite": "Rev4 POTENTIAL Instrument v2 wide table: bank f0-f943, research f944-f1824, peers "
                     "f1825-f1826, calibration f1827-f1828; diagnostic only; " + note,
        "formula_revision": 3}) + "\n")
    return {"path": str(path), "sha256": sha(path), "manifest_sha256": sha(Path(f"{path}.manifest.json")),
            "rows": len(view), "references": int(view.ref_basename.nunique())}


def build_variant(variant: str) -> None:
    k = 0 if variant == "real" else int(variant[1:])
    dest = V2 / "wide" / variant
    receipt = {"schema": "rev4-featpot-v2-wide-v1", "label": "POTENTIAL — ceiling, not a model score",
               "variant": variant, "width": WIDTH, "legs": {}}
    legs = list(SOURCE_ORDER) + list(TEACHERS)
    human_parts = {}
    for leg_index, leg in enumerate(legs):
        if leg in SOURCES:
            frames = [human_set(n) for n in SOURCES[leg]]
            refsets = [set(f.ref_basename.astype(str)) for f in frames]
            if len(frames) > 1 and refsets[0] & refsets[1]:
                raise ValueError(f"{leg}: member sets share references")
            out = pd.concat(frames, ignore_index=True)
        else:
            out = teacher_set(leg)
        target = out.target.to_numpy(dtype=np.float64)
        lo, hi = (float(v) for v in np.quantile(target, [0.001, 0.999]))
        y01 = np.clip((target - lo) / (hi - lo), 0.0, 1.0)
        add_oracles(out, y01, leg_index)
        if k:
            rng = np.random.default_rng(PERM_SEED_BASE + k + 1000 * leg_index)
            restore_data._permute_within_reference(out, FEATURES[944:], np.ones(len(out), dtype=bool), rng)
        rec = {"bounds": [lo, hi]}
        if leg in SOURCES:
            score = 100.0 * y01
            dev_mask = out.ref_basename.astype(str).map(human_dev).to_numpy(dtype=bool)
            rec["full"] = write(out, dest / f"{leg}.parquet", score, f"human source {leg} (evaluation)")
            keys = out[["pair_key", "source_row_id", "ref_basename", "member_set", "target"]]
            pq.write_table(pa.Table.from_pandas(keys, preserve_index=False), dest / f"{leg}.keys.parquet",
                           compression="zstd")
            rec["keys_sha256"] = sha(dest / f"{leg}.keys.parquet")
            human_parts[leg] = (out, score, dev_mask)
            if k == 0:
                rec["oracle_standalone_srocc"] = {
                    n: panel_batch([(leg, out[f"f{COL_ORACLE[n]}"].to_numpy(np.float64), target)],
                                   stats="srocc")[0]["srocc"] for n in COL_ORACLE}
        else:
            score = target  # raw signed SSIMULACRA2, --target-scale 1
            for split in ("fit", "dev"):
                m = (out.split == split).to_numpy()
                rec[split] = write(out.loc[m], dest / f"{leg}_{split}.parquet", score[m], f"teacher {leg} {split}")
        receipt["legs"][leg] = rec
        print(json.dumps({"variant": variant, "leg": leg, "rows": len(out)}), flush=True)
    for held in SOURCE_ORDER:
        train = [s for s in SOURCE_ORDER if s != held]
        rec = {}
        for split, want_dev in (("fit", False), ("dev", True)):
            parts = [(o.loc[d == want_dev], s[d == want_dev]) for o, s, d in (human_parts[t] for t in train)]
            frame = pd.concat([p[0] for p in parts], ignore_index=True)
            score = np.concatenate([p[1] for p in parts])
            rec[split] = write(frame, dest / f"human_without_{held}_{split}.parquet", score,
                               f"human training leg without {held}, {split}")
        receipt["legs"][f"human_without_{held}"] = rec
    (dest / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    print(json.dumps({"receipt": str(dest / "receipt.json"), "sha256": sha(dest / "receipt.json")}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["pin", "build"])
    ap.add_argument("--variant", choices=VARIANTS, action="append")
    args = ap.parse_args()
    if args.action == "pin":
        write_pin()
        return
    for variant in args.variant or VARIANTS:
        build_variant(variant)


if __name__ == "__main__":
    sys.exit(main())
