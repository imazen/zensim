"""Pinned C1-C4 sidecar join for Rev4 POTENTIAL candidate arms.

Canonical f986-f1321 are packed after bank f0-f943 for the diagnostic fit;
the mapping is explicit and the intervening DVIFM f944-f985 are never read.
No label is opened until the Part B completion report and a committed input
pin have been verified. The baseline loader owns the role-allowed label read.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from admit_bank import PROMOTED_BANK, load_pinned_set
from data import FEATURES as BANK_FEATURES, load as load_bank
import restore_data


ROOT = Path("/var/tmp/rev4-featpot")
REPORT = (Path.home() / "tmp/zensim-paper/rev4/PARTB_C1C4_DONE.md")
PIN = ROOT / "candidates" / "input_pin.json"
COMMITTED = ROOT / "candidates" / "input_pin_committed.json"
FAMILIES = {
    "c1": tuple(range(986, 1082)),
    "c2": tuple(range(1082, 1154)),
    "c3": tuple(range(1154, 1298)),
    "c4": tuple(range(1298, 1322)),
}
ARMS = {**FAMILIES, "all": tuple(range(986, 1322)),
        **{arm: tuple(restore_data.arm_ids(arm)) for arm in restore_data.ARM_KEYS}}
SIDECAR = "features__rev4c1c4.parquet"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def columns(arm: str) -> tuple[list[str], list[int]]:
    if arm.removesuffix("_perm") in restore_data.ARM_KEYS:
        return restore_data.columns(arm)
    family = arm.removesuffix("_perm")
    if family not in ARMS:
        raise ValueError(arm)
    ids = list(ARMS[family])
    return BANK_FEATURES + [f"f{i}" for i in range(944, 944 + len(ids))], ids


def preflight(name: str) -> dict:
    if not REPORT.is_file() or not PIN.is_file() or not COMMITTED.is_file():
        raise FileNotFoundError("Part B report, input pin and its prereg commit required")
    committed = json.loads(COMMITTED.read_text())
    if (committed.get("schema") != "rev4-featpot-c1c4-pin-committed-v1" or
            committed.get("pin_sha256") != sha(PIN) or
            len(committed.get("amendment_commit", "")) != 40):
        raise ValueError("candidate pin has no matching prereg commit receipt")
    pin = json.loads(PIN.read_text())
    if pin["schema"] != "rev4-featpot-c1c4-input-v1" or sha(REPORT) != pin["report_sha256"]:
        raise ValueError("Part B report changed after candidate pin")
    if name not in pin["sets"]:
        raise ValueError(f"{name}: absent from candidate input pin")
    info = pin["sets"][name]
    set_dir = PROMOTED_BANK / name
    sidecar = set_dir / SIDECAR
    if (sha(sidecar) != info["sidecar_sha256"] or
            sha(set_dir / "_MANIFEST.json") != info["manifest_sha256"] or
            sha(set_dir / "keys.parquet") != info["keys_sha256"]):
        raise ValueError(f"{name}: candidate sidecar or bank pin changed")
    manifest = load_pinned_set(name)
    promoted = json.loads((set_dir / "_MANIFEST.json").read_text())
    for filename, component in manifest["files"].items():
        if promoted["files"].get(filename, {}).get("sha256") != component["sha256"]:
            raise ValueError(f"{name}: Part B changed baseline component {filename}")
    if promoted["files"].get(SIDECAR, {}).get("sha256") != info["sidecar_sha256"]:
        raise ValueError(f"{name}: Part B sidecar absent from promoted manifest")
    if info["rows"] != manifest["row_count"] or info["unique_pair_keys"] != manifest["unique_pair_keys"]:
        raise ValueError(f"{name}: pinned row/key counts differ from promoted bank")
    schema = pq.ParquetFile(sidecar).schema_arrow
    expected = ["pair_key"] + [f"f{i}" for i in range(986, 1322)]
    if schema.names != expected:
        raise ValueError(f"{name}: candidate column order/schema mismatch")
    for field in schema[1:]:
        if str(field.type) != "float":
            raise ValueError(f"{name}: {field.name} is not f32")
    return info


def load(name: str, arm: str, features: bool = True):
    if arm.removesuffix("_perm") in restore_data.ARM_KEYS:
        return restore_data.load(name, arm, features)
    cols, ids = columns(arm)
    info = preflight(name)
    if not features:
        base, meta = load_bank(name, features=False)
        meta = dict(meta)
        meta.update({"candidate_arm": arm, "candidate_ids": ids,
                     "candidate_report_sha256": sha(REPORT),
                     "candidate_sidecar_sha256": info["sidecar_sha256"]})
        return base, meta
    side = pq.read_table(PROMOTED_BANK / name / SIDECAR,
                         columns=["pair_key"] + [f"f{i}" for i in ids]).to_pandas()
    if len(side) != info["unique_pair_keys"] or side.pair_key.nunique() != info["unique_pair_keys"]:
        raise ValueError(f"{name}: candidate sidecar key cardinality mismatch")
    if not np.isfinite(side[[f"f{i}" for i in ids]].to_numpy()).all():
        raise ValueError(f"{name}: nonfinite candidate feature")
    unique = side.drop_duplicates("pair_key", keep="first")
    if len(unique) != info["unique_pair_keys"]:
        raise ValueError(f"{name}: duplicate key mismatch")
    for feature in [f"f{i}" for i in ids]:
        if not side[feature].eq(side.pair_key.map(unique.set_index("pair_key")[feature])).all():
            raise ValueError(f"{name}: collapsed-key feature disagreement in {feature}")
    base, meta = load_bank(name)
    out = base.merge(unique, on="pair_key", how="left", sort=False,  # joinsafety-ok: pair_key-keyed feature/label join with validate= and full-coverage asserts, not a metric-table join
                     validate="many_to_one", indicator=True)
    if len(out) != len(base) or not (out._merge == "both").all():
        raise ValueError(f"{name}: sidecar join coverage mismatch")
    out = out.drop(columns="_merge")
    out.index = np.arange(len(out))
    packed = {f"f{canonical}": f"f{944 + j}" for j, canonical in enumerate(ids)}
    out = out.rename(columns=packed)
    out, convention = restore_data.apply_identical_convention(
        out, cols[944:], name, permute=arm.endswith("_perm"))
    meta = dict(meta)
    meta.update({"candidate_arm": arm, "candidate_ids": ids,
                 "candidate_report_sha256": sha(REPORT),
                 "candidate_sidecar_sha256": info["sidecar_sha256"],
                 "candidate_packed_columns": cols[944:],
                 "identical_convention": convention})
    return out, meta
