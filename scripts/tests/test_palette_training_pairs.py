import copy
from pathlib import Path
import sys
import unittest
import hashlib
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import palette_training_pairs as p


class PalettePairs(unittest.TestCase):
    def fixture(self):
        sources = [
            dict(
                source_id=f"{c}{i}",
                family=f"{c}{i}",
                **{"class": c},
                role="fit" if i < 50 else "dev",
            )
            for c in p.CONTENT_CLASSES
            for i in range(64)
        ]
        rows = []
        for i, s in enumerate(sources):
            for j in range(2):
                rows.append(
                    s
                    | dict(
                        feature_contract_sha256="a" * 64,
                        feature_ids=list(range(410)),
                        formula_revision=5,
                        teacher_owner="imazen-ssimulacra2",
                        colour="rgb8-srgb-opaque",
                        independent_decode_equal=True,
                        metric_maps={"invented": "b" * 64},
                        metric_outputs={"invented": 0},
                        encode_sha256=f"{i * 2 + j:064x}",
                        reference_pixels_sha256=f"{i:064x}",
                        decoded_pixels_sha256=f"{j:064x}",
                        feature_era_sha256="c" * 64,
                        features=[float(j)] * 410,
                        ssimulacra2=-40.0 if j == 0 else 80.0,
                        requested={"colors": j + 2},
                    )
                )
        return sources, rows

    def test_negative_teacher_aliases_and_equal_source_mass(self):
        sources, rows = self.fixture()
        duplicate = copy.deepcopy(rows[0])
        duplicate["encode_sha256"] = "d" * 64
        result = p.prepare(
            sources, rows + [duplicate], "a" * 64, list(range(410)), pool_for(sources)
        )
        self.assertAlmostEqual(
            sum(g["nominal_weight"] for g in result["groups"] if g["role"] == "fit"), 2
        )
        self.assertEqual(result["groups"][0]["rows"][0]["human_score"], -0.4)
        self.assertEqual(sum(len(g["rows"]) for g in result["groups"]), 640)
        self.assertTrue(all(g["mode"] == "withinref,both" for g in result["groups"]))
        self.assertTrue(
            all(g["train_weight"] == 0 for g in result["groups"] if g["role"] == "dev")
        )
        self.assertEqual(sum(len(a["captures"]) for a in result["aliases"]), 641)

    def test_alias_disagreement_or_missing_capture_refuses(self):
        sources, rows = self.fixture()
        duplicate = copy.deepcopy(rows[0])
        duplicate["ssimulacra2"] = 0
        with self.assertRaises(ValueError):
            p.prepare(
                sources,
                rows + [duplicate],
                "a" * 64,
                list(range(410)),
                pool_for(sources),
            )
        with self.assertRaises(ValueError):
            p.prepare(sources, rows[2:], "a" * 64, list(range(410)), pool_for(sources))

    def test_short_class_weights_preserve_equal_nominal_mass(self):
        sources, rows = self.fixture()
        selected = []
        for cls, n in zip(p.CONTENT_CLASSES, [64, 64, 4, 25, 61]):
            dev = max(1, min(n - 1, (14 * n + 32) // 64))
            members = [s for s in sources if s["class"] == cls][:n]
            for i, s in enumerate(members):
                s["role"] = "fit" if i < n - dev else "dev"
            selected += members
        ids = {s["source_id"]: s for s in selected}
        captures = [r | ids[r["source_id"]] for r in rows if r["source_id"] in ids]
        result = p.prepare(
            selected, captures, "a" * 64, list(range(410)), pool_for(selected)
        )
        for cls in p.CONTENT_CLASSES:
            for role in ("fit", "dev"):
                self.assertAlmostEqual(
                    sum(
                        g["nominal_weight"]
                        for g in result["groups"]
                        if ids[g["source_id"]]["class"] == cls and g["role"] == role
                    ),
                    0.4,
                )
        self.assertTrue(
            all(g["train_weight"] == 0 for g in result["groups"] if g["role"] == "dev")
        )
        with self.assertRaises(ValueError):
            p.inventory(selected[:-1], pool_for(selected))


def pool_for(sources):
    pool = dict(
        schema="e35-palette-pool-v2",
        certified_clean=True,
        available_families={
            c: [s["family"] for s in sources if s["class"] == c]
            for c in p.CONTENT_CLASSES
        },
    )
    pool["authority_sha256"] = hashlib.sha256(
        json.dumps(pool, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    ).hexdigest()
    return pool


if __name__ == "__main__":
    unittest.main()
