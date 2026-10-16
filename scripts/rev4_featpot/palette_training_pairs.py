"""Build source-balanced synthetic palette legs from persisted keyed captures.

No corpus discovery, generation, training or launch. Encoding aliases remain in
provenance; each (reference pixels, decoded pixels, feature era) contributes once.
The caller supplies the admitted source inventory and immutable capture hashes.
"""
from collections import defaultdict
import math

from v2_common import acceptance_weight

CONTENT_CLASSES = ("tiny", "photo", "screen", "line", "mixed")


def inventory(sources):
    by_id = {s["source_id"]: s for s in sources}
    if len(by_id) != 320 or len(sources) != 320:
        raise ValueError("320 distinct palette sources required")
    for cls in CONTENT_CLASSES:
        for role, count in (("fit", 50), ("dev", 14)):
            if sum(s["class"] == cls and s["role"] == role for s in sources) != count:
                raise ValueError("palette role/class inventory differs")
    if len({s["family"] for s in sources}) != 320:
        raise ValueError("palette family overlaps roles/sources")
    return by_id


def prepare(sources, captures, feature_contract_sha256, feature_ids):
    """Return trainer groups and immutable keyed rows, without opening any file.

    Captures include the independent decode/persistence proof, teacher owner,
    all metric map hashes, and the direct Rev5 feature vector. No quality-based
    selection or palette-count ranking is performed.
    """
    source_by_id = inventory(sources)
    if (len(feature_contract_sha256) != 64 or len(feature_ids) < 410
            or len(set(feature_ids)) != len(feature_ids)):
        raise ValueError("explicit difference/product-factor read contract required")
    unique, aliases = {}, defaultdict(list)
    by_source = defaultdict(list)
    for row in captures:
        src = source_by_id[row["source_id"]]
        if row["family"] != src["family"] or row["role"] != src["role"]:
            raise ValueError("palette capture source role mismatch")
        if (row["formula_revision"] != 5 or row["feature_contract_sha256"] != feature_contract_sha256
                or row["feature_ids"] != feature_ids or row["teacher_owner"] != "imazen-ssimulacra2"
                or row["colour"] != "rgb8-srgb-opaque" or not row["independent_decode_equal"]):
            raise ValueError("palette capture producer/feature/decoder contract differs")
        if (not row["metric_maps"] or not row["metric_outputs"]
                or any(len(row[k]) != 64 for k in ("encode_sha256", "reference_pixels_sha256",
                                                  "decoded_pixels_sha256", "feature_era_sha256"))):
            raise ValueError("palette bytes/metrics/maps must be persisted")
        features = row["features"]
        teacher = float(row["ssimulacra2"])
        if len(features) != len(feature_ids) or not all(math.isfinite(v) for v in (*features, teacher)):
            raise ValueError("palette nonfinite/missing selected feature or teacher")
        key = (row["reference_pixels_sha256"], row["decoded_pixels_sha256"], row["feature_era_sha256"])
        semantic = (row["source_id"], row["role"], features, teacher)
        if key in unique:
            if unique[key] != semantic:
                raise ValueError("palette pixel alias disagrees on features/teacher/source")
        else:
            unique[key] = semantic
            by_source[row["source_id"]].append(dict(
                reference_key=key[0], distorted_key=key[1], human_score=teacher/100.,
                features=features, feature_era_sha256=key[2]))
        aliases[key].append(dict(encode_sha256=row["encode_sha256"], requested=row["requested"],
                                 metric_outputs=row["metric_outputs"], metric_maps=row["metric_maps"]))
    if set(by_source) != set(source_by_id):
        raise ValueError("palette capture omits declared source")
    groups = []
    for source_id in sorted(by_source):
        src, rows = source_by_id[source_id], by_source[source_id]
        refs = [r["reference_key"] for r in rows]
        if not any(refs.count(r) >= 2 for r in set(refs)):
            raise ValueError("palette source has no eligible within-reference pairs")
        # Equal class, then source mass. R915 corrects rejected same-row draws.
        nominal = 2 / (5 * (50 if src["role"] == "fit" else 14))
        corrected = acceptance_weight(nominal, refs)
        groups.append(dict(name=f"palette_{source_id}", source_id=source_id, role=src["role"],
                           mode="withinref,both", nominal_weight=nominal,
                           train_weight=corrected if src["role"] == "fit" else 0.,
                           val_weight=corrected if src["role"] == "dev" else 0., rows=rows))
    return dict(schema="palette-training-pairs-v1", groups=groups,
                feature_contract_sha256=feature_contract_sha256, feature_ids=feature_ids,
                aliases=[dict(pixel_key=list(key), captures=value) for key, value in sorted(aliases.items())])
