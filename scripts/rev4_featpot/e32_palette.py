"""E32 consumer admission and named column projection; no fleet or fit owner."""
import json
from pathlib import Path

from v2_common import REPO, sha
from v2_human_role import MEMBER_SOURCE
from v2_teacher import key_path, row_keys_sha

CONTRACT = json.loads((REPO / "benchmarks/e32_palette_training_transport_2026-10-07.json").read_text())
IDS_PATH = REPO / "benchmarks/e32_palette_feature_ids_2026-10-07.json"
ARM_IDS = json.loads(IDS_PATH.read_text())["arm_ids"]
PALETTE_IDS = list(range(1825, 1867))
BASE_ID = CONTRACT["inherited_feature_set_id"]
FEATURE_SET_ID = CONTRACT["feature_set_id"]


def admit_declaration(d):
    """All roles, pins and the named map precede any label-bearing payload open."""
    p = d.get("research_palette")
    if not isinstance(p, dict):
        raise ValueError("E32 explicit research projection required")
    identity_keys = ("schema", "inherited_feature_set_id", "palette_feature_set_id", "feature_ids_sha256",
                     "bank_manifest_sha256", "instrument_manifest_sha256", "producer_binary_sha256",
                     "build_commit", "serving_allowed", "cast")
    if (any(p.get(k) != CONTRACT[k] for k in identity_keys)
            or p.get("serving_allowed") is not False
            or p.get("columns") != {str(i): f"palette_f{i}" for i in PALETTE_IDS}
            or d.get("feature_set_id") != FEATURE_SET_ID or d.get("formula_revision") != 5
            or not d.get("decoder_era")):
        raise ValueError("E32 palette identity/era/map/producer pin mismatch")
    for value in [*(d.get(k) for k in ("table_sha256", "keys_sha256", "row_keys_sha256", "row_selection_sha256")),
                  p.get("inherited_table_sha256")]:
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise ValueError("E32 byte-bound table/key/selection ancestry required")
    members = p.get("member_sets", [])
    allowed = {*MEMBER_SOURCE, "safesyn", "cid22_train", "coverage_pool"}
    if not members or len(set(members)) != len(members) or not set(members) <= allowed:
        raise ValueError("E32 unapproved/AIC members refused before payload access")
    role = p.get("role")
    if role in ("D1-fit", "D1-development"):
        sources = d.get("human_sources", [])
        valid = (d.get("data_role") == "design-released-human"
                 and d.get("data_role_decision_required") == "SHIPPATH-human-production-role"
                 and set(members) <= MEMBER_SOURCE.keys()
                 and sources and len(sources) == len(set(sources))
                 and {MEMBER_SOURCE[m] for m in members} == set(sources))
    elif role in ("TRAIN-oracle-fit", "TRAIN-oracle-development"):
        valid = d.get("data_role") == "TRAIN oracle teacher" and len(members) == 1 and members[0] in ("safesyn", "cid22_train")
    elif role == "TRAIN-ordinal":
        valid = (d.get("data_role") == "TRAIN ordinal KADIS source_id%10<8; no human labels"
                 and members == ["coverage_pool"] and p.get("key_domain") == "coverage-selection-ordinal")
    else:
        valid = False
    if not valid or (role != "TRAIN-ordinal" and p.get("key_domain") != "member-pair-observation"):
        raise ValueError("E32 original D1/TRAIN role and key domain required")
    return p


def validate_keep(keep):
    if sha(IDS_PATH) != CONTRACT["feature_ids_sha256"] or list(keep) != ARM_IDS:
        raise ValueError("E32 requires exactly the registered ordered 462 IDs")


def admit_recipe(receipt, recipe, keep, head, strict, train_only):
    from v2_common import recipe_of
    expected = recipe_of("sel:59f0bbc2f290@h32:H128:cv16:cf98")
    if (receipt.get("research_palette") != CONTRACT
            or receipt.get("research_palette", {}).get("serving_allowed") is not False
            or type(receipt.get("width")) is not int or receipt.get("width") != 1867
            or recipe != expected or head != "N" or not strict or not train_only):
        raise ValueError("E32 needs its explicit frozen transport and unchanged E30 recipe, strict train-only head N")
    validate_keep(keep)


def feature_columns(path, keep):
    d = json.loads(Path(f"{path}.manifest.json").read_text())
    if "research_palette" not in d:
        return [f"f{i}" for i in keep]
    p = admit_declaration(d)
    validate_keep(keep)
    return [p["columns"][str(i)] if i in PALETTE_IDS else f"f{i}" for i in keep]


def admit_keys(path, d, keys):
    import pyarrow as pa
    p = admit_declaration(d)
    if sha(key_path(path)) != d["keys_sha256"] or row_keys_sha(keys) != d["row_keys_sha256"]:
        raise ValueError("E32 original ordered key bytes changed")
    if p["role"] == "TRAIN-ordinal":
        allowed = {"ladder", "source_filename", "type", "family", "severity_level", "severity", "sign", "__index_level_0__"}
        if not set(keys.column_names) <= allowed or "__index_level_0__" not in keys.column_names:
            raise ValueError("E32 coverage needs original selection ordinals, no pixel-key fallback")
    else:
        allowed = {"pair_key", "source_row_id", "row_id", "ref_basename", "member_set"}
        if (not set(keys.column_names) <= allowed or "member_set" not in keys.column_names
                or keys["member_set"].type not in (pa.string(), pa.large_string()) or keys["member_set"].null_count
                or set(keys["member_set"].to_pylist()) != set(p["member_sets"])):
            raise ValueError("E32 unapproved/AIC or label-bearing key population")


def coverage_extension(pool, keep):
    d = json.loads(Path(f"{pool}.manifest.json").read_text())
    p = admit_declaration(d)
    validate_keep(keep)
    if p["role"] != "TRAIN-ordinal":
        raise ValueError("E32 ordinal coverage role required")
    return p
