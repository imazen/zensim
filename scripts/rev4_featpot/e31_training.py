"""Fit-only admission for the registered native HDR leg; never opens development."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq

from e21_cheap_recipe import columns
from upiq380 import RULE, TRANSFORM, LABEL_SHA, sha

MANIFEST_SHA = "2da346bb17e08a4a63aae9ed89b159b36edd331199a9dd1c42b4a05b53ac939e"
TABLE_SHA = "7f09debedc591e7dd3494846ada9fe0c93b01779918b8358e2cf8053f5f1a6c4"
KEYS_SHA = "c47ca1c12d1e8e206464938884ff9d05baa8274785826060604d16caf0bff49e"


def disposition(path):
    d = json.loads(Path(path).read_text())
    expected = dict(schema="e31-upiq-label-disposition-v1", state="approved",
                    decision_id="E31-legacy-HDR-label-producer-gap",
                    allowed_use="registered-E31-research-training",
                    manifest_sha256=MANIFEST_SHA, legacy_label_sha256=LABEL_SHA,
                    accept_unresolved_producer=True)
    if any(d.get(k) != v for k, v in expected.items()):
        raise ValueError("E31 requires the owner's bound legacy-label disposition")
    return d


def fit_metadata(path, decision, keep):
    """Metadata and label-free keys only; every payload pin stays immutable."""
    path = Path(path)
    sp = Path(f"{path}.manifest.json")
    d = json.loads(sp.read_text())
    # Semantic checks precede even hashing the manifest, useful for tripwires.
    expected = dict(schema="upiq380-v2-leg-v1", arm="uh4", source="UPIQ-380",
                    split="fit", role="train", tier="T2", authority="D3-2026-10-07",
                    rows=330, references=26, formula_revision=5, requested_ids=columns("by_v2fy"),
                    input_contract="upiq-exr-bt709-nits-v1", split_rule=RULE,
                    target_transform=TRANSFORM, feature_dtype="float64", absent_slots="NaN",
                    table_sha256=TABLE_SHA, keys_sha256=KEYS_SHA, qualified_provenance=False)
    if any(d.get(k) != v for k, v in expected.items()) or list(keep) != columns("by_v2fy"):
        raise ValueError("E31 fit role/population/native-slot declaration mismatch")
    approval = disposition(decision)
    if sha(sp) != MANIFEST_SHA:
        raise ValueError("E31 fit manifest pin changed")
    kp = path.with_suffix(".keys.parquet")
    if sha(kp) != KEYS_SHA:
        raise ValueError("E31 fit key pin changed")
    keys = pq.read_table(kp).to_pylist()
    if len(keys) != 330 or [r["condition_id"] for r in keys] != d["member_set"]:
        raise ValueError("E31 fit ordered membership mismatch")
    refs = []
    for i, r in enumerate(keys):
        binding = "upiq380-original-byte-pair-v1\0" + r["condition_id"] + "\0" + r["reference_sha256"] + "\0" + r["distorted_sha256"]
        if (r["row_id"] != i or r["role"] != "train" or r["source"] != "UPIQ-380"
                or r["authority"] != "D3-2026-10-07" or r["split"] != "fit"
                or int(r["reference_sha256"], 16) % 5 == 0
                or r["pair_key"] != hashlib.sha256(binding.encode()).hexdigest()):
            raise ValueError("E31 fit source/split/key binding mismatch")
        refs.append(r["reference_sha256"])
    if len(set(refs)) != 26:
        raise ValueError("E31 fit reference membership mismatch")
    return dict(name="upiq380", table_sha256=TABLE_SHA, manifest_sha256=MANIFEST_SHA,
                keys_sha256=KEYS_SHA, qualified_provenance=False,
                label_source=d["label_source"], label_disposition=approval,
                label_disposition_sha256=sha(decision), references=refs)


def fit_group(path, decision, keep):
    from v2_common import acceptance_weight
    receipt = fit_metadata(path, decision, keep)
    weight = acceptance_weight(4.0, receipt.pop("references"))
    return ("upiq380", Path(path), weight, 0, "rank"), receipt
