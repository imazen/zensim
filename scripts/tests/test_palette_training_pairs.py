import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import palette_training_pairs as p


class PalettePairs(unittest.TestCase):
    def fixture(self):
        sources = [dict(source_id=f"{c}{i}", family=f"{c}{i}", **{"class":c},
                        role="fit" if i < 50 else "dev") for c in p.CONTENT_CLASSES for i in range(64)]
        rows = []
        for i, s in enumerate(sources):
            for j in range(2):
                rows.append(s | dict(feature_contract_sha256="a"*64, feature_ids=list(range(410)),
                    formula_revision=5, teacher_owner="imazen-ssimulacra2", colour="rgb8-srgb-opaque",
                    independent_decode_equal=True, metric_maps={"invented":"b"*64}, metric_outputs={"invented":0},
                    encode_sha256=f"{i*2+j:064x}", reference_pixels_sha256=f"{i:064x}",
                    decoded_pixels_sha256=f"{j:064x}", feature_era_sha256="c"*64,
                    features=[float(j)]*410, ssimulacra2=-40. if j == 0 else 80., requested={"colors":j+2}))
        return sources, rows

    def test_negative_teacher_aliases_and_equal_source_mass(self):
        sources, rows = self.fixture()
        duplicate = copy.deepcopy(rows[0])
        duplicate["encode_sha256"] = "d"*64
        result = p.prepare(sources, rows+[duplicate], "a"*64, list(range(410)))
        self.assertAlmostEqual(sum(g["nominal_weight"] for g in result["groups"] if g["role"] == "fit"), 2)
        self.assertEqual(result["groups"][0]["rows"][0]["human_score"], -.4)
        self.assertEqual(sum(len(g["rows"]) for g in result["groups"]), 640)
        self.assertTrue(all(g["mode"] == "withinref,both" for g in result["groups"]))
        self.assertTrue(all(g["train_weight"] == 0 for g in result["groups"] if g["role"] == "dev"))
        self.assertEqual(sum(len(a["captures"]) for a in result["aliases"]), 641)

    def test_alias_disagreement_or_missing_capture_refuses(self):
        sources, rows = self.fixture()
        duplicate = copy.deepcopy(rows[0])
        duplicate["ssimulacra2"] = 0
        with self.assertRaises(ValueError):
            p.prepare(sources, rows+[duplicate], "a"*64, list(range(410)))
        with self.assertRaises(ValueError):
            p.prepare(sources, rows[2:], "a"*64, list(range(410)))


if __name__ == "__main__":
    unittest.main()
