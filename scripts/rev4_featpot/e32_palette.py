"""E32 consumer admission and named column projection; no fleet or fit owner."""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2_common import REPO, sha
from v2_human_role import MEMBER_SOURCE
from v2_teacher import key_path, row_keys_sha

CONTRACT = json.loads(
    (REPO / "benchmarks/e32_palette_training_transport_2026-10-07.json").read_text()
)
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
    identity_keys = (
        "schema",
        "inherited_feature_set_id",
        "palette_feature_set_id",
        "feature_ids_sha256",
        "bank_manifest_sha256",
        "instrument_manifest_sha256",
        "producer_binary_sha256",
        "build_commit",
        "serving_allowed",
        "cast",
    )
    if (
        any(p.get(k) != CONTRACT[k] for k in identity_keys)
        or p.get("serving_allowed") is not False
        or p.get("columns") != {str(i): f"palette_f{i}" for i in PALETTE_IDS}
        or d.get("feature_set_id") != FEATURE_SET_ID
        or d.get("formula_revision") != 5
        or not d.get("decoder_era")
    ):
        raise ValueError("E32 palette identity/era/map/producer pin mismatch")
    for value in [
        *(
            d.get(k)
            for k in (
                "table_sha256",
                "keys_sha256",
                "row_keys_sha256",
                "row_selection_sha256",
            )
        ),
        p.get("inherited_table_sha256"),
    ]:
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)
        ):
            raise ValueError("E32 byte-bound table/key/selection ancestry required")
    members = p.get("member_sets", [])
    allowed = {*MEMBER_SOURCE, "safesyn", "cid22_train", "coverage_pool"}
    if not members or len(set(members)) != len(members) or not set(members) <= allowed:
        raise ValueError("E32 unapproved/AIC members refused before payload access")
    role = p.get("role")
    if role in ("D1-fit", "D1-development"):
        sources = d.get("human_sources", [])
        valid = (
            d.get("data_role") == "design-released-human"
            and d.get("data_role_decision_required") == "SHIPPATH-human-production-role"
            and set(members) <= MEMBER_SOURCE.keys()
            and sources
            and len(sources) == len(set(sources))
            and {MEMBER_SOURCE[m] for m in members} == set(sources)
        )
    elif role in ("TRAIN-oracle-fit", "TRAIN-oracle-development"):
        valid = (
            d.get("data_role") == "TRAIN oracle teacher"
            and len(members) == 1
            and members[0] in ("safesyn", "cid22_train")
        )
    elif role == "TRAIN-ordinal":
        valid = (
            d.get("data_role") == "TRAIN ordinal KADIS source_id%10<8; no human labels"
            and members == ["coverage_pool"]
            and p.get("key_domain") == "coverage-selection-ordinal"
        )
    else:
        valid = False
    if not valid or (
        role != "TRAIN-ordinal" and p.get("key_domain") != "member-pair-observation"
    ):
        raise ValueError("E32 original D1/TRAIN role and key domain required")
    return p


def validate_keep(keep):
    if sha(IDS_PATH) != CONTRACT["feature_ids_sha256"] or list(keep) != ARM_IDS:
        raise ValueError("E32 requires exactly the registered ordered 462 IDs")


def admit_recipe(receipt, recipe, keep, head, strict, train_only):
    from v2_common import recipe_of

    expected = recipe_of("sel:59f0bbc2f290@h32:H128:cv16:cf98")
    if (
        receipt.get("research_palette") != CONTRACT
        or receipt.get("research_palette", {}).get("serving_allowed") is not False
        or type(receipt.get("width")) is not int
        or receipt.get("width") != 1867
        or recipe != expected
        or head != "N"
        or not strict
        or not train_only
    ):
        raise ValueError(
            "E32 needs its explicit frozen transport and unchanged E30 recipe, strict train-only head N"
        )
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
    if (
        sha(key_path(path)) != d["keys_sha256"]
        or row_keys_sha(keys) != d["row_keys_sha256"]
    ):
        raise ValueError("E32 original ordered key bytes changed")
    from v2_teacher import admit_ordered_keys

    admit_ordered_keys(path, d, keys)
    if p["role"] == "TRAIN-ordinal":
        allowed = {
            "ladder",
            "source_filename",
            "type",
            "family",
            "severity_level",
            "severity",
            "sign",
            "__index_level_0__",
        }
        if (
            not set(keys.column_names) <= allowed
            or "__index_level_0__" not in keys.column_names
        ):
            raise ValueError(
                "E32 coverage needs original selection ordinals, no pixel-key fallback"
            )
    else:
        allowed = {"pair_key", "source_row_id", "row_id", "ref_basename", "member_set"}
        if (
            not set(keys.column_names) <= allowed
            or "member_set" not in keys.column_names
            or keys["member_set"].type not in (pa.string(), pa.large_string())
            or keys["member_set"].null_count
            or set(keys["member_set"].to_pylist()) != set(p["member_sets"])
        ):
            raise ValueError("E32 unapproved/AIC or label-bearing key population")


def coverage_extension(pool, keep):
    d = json.loads(Path(f"{pool}.manifest.json").read_text())
    p = admit_declaration(d)
    validate_keep(keep)
    if p["role"] != "TRAIN-ordinal":
        raise ValueError("E32 ordinal coverage role required")
    return p


def append_named_projection(inherited, keys, measured):
    """Join byte-bound observations to the bank's explicit row identities."""
    import hashlib
    import numpy as np
    import pyarrow as pa

    if len(inherited) != len(keys):
        raise ValueError("E32 inherited table/key row count differs")
    offsets, lookup, chunks, offset = {}, {}, [], 0
    for member in sorted(measured):
        bank = measured[member]
        pairs = bank["pair_key"].to_pylist()
        rows = bank["row_id"].to_pylist()
        if rows != list(range(len(bank))) or len(set(pairs)) != len(pairs):
            raise ValueError("E32 bank pair/row identity differs")
        offsets[member] = offset
        lookup[member] = dict(zip(pairs, rows))
        chunks.append(bank)
        offset += len(bank)
    if "member_set" in keys.column_names:
        source_rows = keys[
            "source_row_id" if "source_row_id" in keys.column_names else "row_id"
        ].to_pylist()
        bindings = []
        for member, pair, original_row in zip(
            keys["member_set"].to_pylist(), keys["pair_key"].to_pylist(), source_rows
        ):
            if member not in lookup or pair not in lookup[member] or original_row < 0:
                raise ValueError("E32 missing original observation join")
            bindings.append([member, pair, original_row, lookup[member][pair]])
        joined = pa.concat_tables(chunks).take(
            pa.array([offsets[m] + bank_row for m, _, _, bank_row in bindings])
        )
    else:
        bank = measured["coverage_pool"]
        if len(bank) != len(keys):
            raise ValueError("E32 original ordinal population differs")
        joined = bank
        bindings = [
            ["coverage_pool", pair, original_row, bank_row]
            for pair, original_row, bank_row in zip(
                bank["pair_key"].to_pylist(),
                keys["__index_level_0__"].to_pylist(),
                bank["row_id"].to_pylist(),
            )
        ]
    result = inherited
    for i in PALETTE_IDS:
        values = joined[f"f{i}"].to_numpy()
        cast = values.astype(np.float32)
        if not np.isfinite(values).all() or not np.isfinite(cast).all():
            raise ValueError("E32 palette primary values must be finite")
        if f"palette_f{i}" in result.column_names:
            raise ValueError("E32 projection already present")
        result = result.append_column(
            f"palette_f{i}", pa.array(cast, type=pa.float32())
        )
    digest = hashlib.sha256(
        json.dumps(bindings, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return result, digest


def prepare_projection(source, palette_root, out, fleet_root, build_commit):
    """Extend the admitted D1 population through the existing palette bank keys.

    Repeated observations retain order. Every inherited feature/target bit and
    label-free key byte is checked; numeric auxiliary names remain untouched.
    """
    import copy
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq
    from lib.assessment_identity import safe_path
    from v2_common import refuse_immutable_output, admission_input_roots
    from v2_human_role import PRODUCTION_SOURCES, preflight_recipe
    from rev5_bank import validate_palette_instrument_manifest

    source, palette_root, out, fleet_root = map(
        safe_path, (source, palette_root, out, fleet_root)
    )
    refuse_immutable_output(out, (*admission_input_roots(source), palette_root))
    if out.exists():
        raise ValueError("E32 projection requires a fresh root")
    preflight_recipe(source, source / "human_role_decision.json")
    bank = palette_root / "bank"
    if sha(bank / "_MANIFEST.json") != CONTRACT["bank_manifest_sha256"]:
        raise ValueError("E32 original palette bank changed")
    top = json.loads((bank / "_MANIFEST.json").read_text())
    validate_palette_instrument_manifest(
        palette_root / "instrument/_MANIFEST.json",
        CONTRACT["build_commit"],
        CONTRACT["bank_manifest_sha256"],
        CONTRACT["instrument_manifest_sha256"],
    )
    original = json.loads((source / "wide/main/real/receipt.json").read_text())
    receipt = copy.deepcopy(original)
    receipt["legs"].pop("hdr_consensus", None)
    receipt.update(width=1867, feature_set_id=FEATURE_SET_ID, research_palette=CONTRACT)
    selected = []
    for name, leg in receipt["legs"].items():
        if name not in {
            *PRODUCTION_SOURCES,
            "safesyn",
            "cid22",
            "human_all",
            *(f"human_without_{s}" for s in PRODUCTION_SOURCES),
        }:
            raise ValueError("E32 unapproved original leg")
        for split in ("full", "fit", "dev"):
            if split in leg:
                selected.append((name, split, leg[split], source / leg[split]["rel"]))
    selected.append(
        ("coverage_pool", "fit", None, source / "e15/coverage_pool.parquet")
    )
    populations = {}
    # All declaration/key/producer bindings precede table and target reads.
    for name, split, rec, path in selected:
        d = json.loads(safe_path(Path(f"{path}.manifest.json")).read_text())
        keys = pq.read_table(safe_path(key_path(path)))
        if (
            sha(key_path(path)) != d["keys_sha256"]
            or row_keys_sha(keys) != d["row_keys_sha256"]
        ):
            raise ValueError("E32 inherited ordered keys changed")
        if name == "coverage_pool":
            members = ["coverage_pool"]
            if (
                sha(key_path(path))
                != "bc225a115ab8505738a5c17ced6d4fc592a9a38ac6d9ec98661f4e8f0898addf"
            ):
                raise ValueError("E32 original coverage selection key pin changed")
        else:
            members = sorted(set(keys["member_set"].to_pylist()))
        if not set(members) <= {
            *MEMBER_SOURCE,
            "safesyn",
            "cid22_train",
            "coverage_pool",
        }:
            raise ValueError("E32 unauthorized palette key member")
        populations[str(path)] = (d, keys, members)
    measured, lookup, producer_pins = {}, {}, {}
    for member in sorted(
        {m for _, _, members in populations.values() for m in members}
    ):
        mp = bank / member / "_MANIFEST.json"
        m = json.loads(safe_path(mp).read_text())
        fp = safe_path(bank / member / "features__palette.parquet")
        if (
            sha(mp) != top["sets"][member]["manifest_sha256"]
            or m["build_commit"] != CONTRACT["build_commit"]
            or m["feature_set_id"] != CONTRACT["palette_feature_set_id"]
            or m["feature_ids"] != PALETTE_IDS
            or m["binary_sha256"] != CONTRACT["producer_binary_sha256"]
            or m["labels_read"] is not False
            or m["serving_allowed"] is not False
            or sha(fp) != m["features_sha256"]
        ):
            raise ValueError("E32 palette producer/content identity changed")
        t = pq.read_table(
            fp, columns=["pair_key", "row_id", *(f"f{i}" for i in PALETTE_IDS)]
        )
        if len(t) != m["rows"] or t["row_id"].to_pylist() != list(range(len(t))):
            raise ValueError("E32 palette original row order changed")
        pair = t["pair_key"].to_pylist()
        if len(pair) != len(set(pair)):
            raise ValueError("E32 duplicate palette bank key")
        if member == "coverage_pool" and (
            m["sources"][1]["sha256"]
            != "bc225a115ab8505738a5c17ced6d4fc592a9a38ac6d9ec98661f4e8f0898addf"
            or m["sources"][1]["columns_read"] != ["__index_level_0__"]
        ):
            raise ValueError(
                "E32 coverage original selection-domain producer binding changed"
            )
        kp = safe_path(bank / member / "keys.parquet")
        if (
            sha(kp) != m["keys_sha256"]
            or pq.read_table(kp, columns=["pair_key"])["pair_key"].to_pylist() != pair
        ):
            raise ValueError("E32 palette key/feature row binding changed")
        measured[member] = t
        lookup[member] = {k: j for j, k in enumerate(pair)}
        producer_pins[member] = dict(manifest_sha256=sha(mp), features_sha256=sha(fp))
    (out / "wide/main/real").mkdir(parents=True)
    (out / "e15").mkdir()
    checks = []
    for name, split, rec, path in selected:
        d, keys, members = populations[str(path)]
        if sha(path) != d["table_sha256"] or (
            rec is not None
            and sha(Path(f"{path}.manifest.json")) != rec["manifest_sha256"]
        ):
            raise ValueError("E32 inherited table/manifest byte pin changed")
        dest = out / path.relative_to(source)
        t = pq.read_table(path)
        t, join_sha = append_named_projection(
            t, keys, {m: measured[m] for m in members}
        )
        pq.write_table(t, dest, compression="zstd", compression_level=1)
        shutil.copyfile(key_path(path), key_path(dest))
        role = (
            "TRAIN-ordinal"
            if name == "coverage_pool"
            else ("D1-" if name not in ("safesyn", "cid22") else "TRAIN-oracle-")
            + ("development" if split == "dev" else "fit")
        )
        projected = {
            **d,
            "rows": len(keys),
            "feature_set_id": FEATURE_SET_ID,
            "table_sha256": sha(dest),
            "immutable_input_roots": [
                *d["immutable_input_roots"],
                str(out.resolve()),
                str(fleet_root),
            ],
            "research_palette": {
                **{
                    k: CONTRACT[k]
                    for k in (
                        "schema",
                        "inherited_feature_set_id",
                        "palette_feature_set_id",
                        "feature_ids_sha256",
                        "bank_manifest_sha256",
                        "instrument_manifest_sha256",
                        "producer_binary_sha256",
                        "build_commit",
                        "serving_allowed",
                        "cast",
                    )
                },
                "columns": {str(i): f"palette_f{i}" for i in PALETTE_IDS},
                "inherited_table_sha256": d["table_sha256"],
                "member_sets": members,
                "role": role,
                "key_domain": "coverage-selection-ordinal"
                if name == "coverage_pool"
                else "member-pair-observation",
                "producer_bindings": {m: producer_pins[m] for m in members},
            },
        }
        Path(f"{dest}.manifest.json").write_text(json.dumps(projected, indent=2) + "\n")
        if rec is not None:
            rec.update(
                sha256=sha(dest), manifest_sha256=sha(Path(f"{dest}.manifest.json"))
            )
        reread = pq.read_table(dest, columns=pq.read_schema(path).names)
        original_table = t.select(reread.column_names)
        for col in reread.column_names:
            if pa.types.is_floating(reread[col].type):
                if (
                    reread[col].to_numpy().tobytes()
                    != original_table[col].to_numpy().tobytes()
                ):
                    raise ValueError("E32 inherited IEEE feature/target bits changed")
            elif not reread[col].equals(original_table[col]):
                raise ValueError("E32 inherited observation order changed")
        if sha(key_path(dest)) != d["keys_sha256"]:
            raise ValueError("E32 original key bytes changed")
        checks.append(
            dict(
                table=str(dest.relative_to(out)),
                rows=len(t),
                all_inherited_columns_bit_equal=True,
                ordered_keys_byte_identical=True,
                observation_to_bank_row_sha256=join_sha,
                palette_columns=42,
                role=role,
            )
        )
        print(json.dumps(checks[-1]), flush=True)
    for rel in (
        "wide/keep_lists.json",
        "human_role_decision.json",
        "source_bindings/receipt.json",
        "source_bindings/frozen.json",
    ):
        dest = out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / rel, dest)
    rp = out / "wide/main/real/receipt.json"
    rp.write_text(json.dumps(receipt, indent=2) + "\n")
    frozen = json.loads((source / "wide/frozen.json").read_text())
    frozen.update(build_commit=build_commit, wide_receipts={"main/real": sha(rp)})
    frozen["auxiliary_files"] = {
        rel: sha(out / rel)
        for rel in frozen["auxiliary_files"]
        if (out / rel).is_file()
    }
    (out / "wide/frozen.json").write_text(json.dumps(frozen, indent=2) + "\n")
    for fold in (None, *PRODUCTION_SOURCES):
        preflight_recipe(out, out / "human_role_decision.json", fold)
    proof = dict(
        schema="e32-inherited-population-projection-v1",
        build_commit=build_commit,
        original_receipt_sha256=sha(source / "wide/main/real/receipt.json"),
        palette_contract=CONTRACT,
        source_root=str(source),
        prepared_root=str(out),
        checks=checks,
        protected_label_payloads_opened=0,
    )
    (out / "PROJECTION_PROOF.json").write_text(json.dumps(proof, indent=2) + "\n")
    return proof


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Prepare the registered palette projection; no fitting or assessment"
    )
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--palette-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fleet-root", type=Path, required=True)
    ap.add_argument("--build-commit", required=True)
    args = ap.parse_args()
    prepare_projection(
        args.source, args.palette_root, args.out, args.fleet_root, args.build_commit
    )
