"""HDRVID external video report: label join census, leg mapping and refusals.

Synthetic labels only; no dataset file is opened.
"""

import io
import csv
import unittest

import numpy as np

import v40_panels as owner


def hdrvdc_csv(drop=0, rename=None):
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["content_id", "content", "crf", "resolution", "test_path", "is_reference",
                "luminance_level", "viewing_distance", "jod"])
    n = 0
    for c in range(16):
        name = f"C{c}"
        tests = [("H", "3840x2160")] + [(q, r) for q in "HML" for r in ("1920x1080", "1280x720", "3840x2160")
                                          if (q, r) != ("H", "3840x2160")]
        tests = tests[:9] if c < 12 else [("H", "1920x1080"), ("H", "1280x720"), ("M", "1920x1080"),
                                          ("M", "1280x720"), ("L", "1920x1080"), ("L", "1280x720")]
        for k, (q, r) in enumerate(tests):
            for lum in ("bright", "dim"):
                for dist in ("near", "far"):
                    ref = k == 0
                    if not ref:
                        n += 1
                        if n <= drop:
                            continue
                    content = rename if (rename and not ref and n == 1) else name
                    w.writerow([c + 1, content, q, r, f"test/{name}/{name}_{q}_{r}.mp4", str(ref), lum, dist,
                                10.0 if ref else 9 - k / 3])
    return out.getvalue().encode()


class E31Video(unittest.TestCase):
    def test_hdrvdc_join_excludes_reference_rows_and_keeps_464(self):
        labels = owner.e31video_labels("hdrvdc", hdrvdc_csv())
        self.assertEqual(len(labels), 464)
        self.assertEqual({r["group"] for r in labels}, {"bright-near", "bright-far", "dim-near", "dim-far"})
        self.assertEqual(len({r["video"] for r in labels}), 116)
        self.assertTrue(all(r["video"].startswith("hdrvdc/C") for r in labels))

    def test_hdrvdc_census_and_identity_refusals(self):
        with self.assertRaisesRegex(ValueError, "census"):
            owner.e31video_labels("hdrvdc", hdrvdc_csv(drop=1))
        with self.assertRaisesRegex(ValueError, "registered test video"):
            owner.e31video_labels("hdrvdc", hdrvdc_csv(rename="Other"))

    def test_avt_join_skips_originals_and_refuses_foreign_rows(self):
        rows = ["stimuli_file,mos,ci,std"]
        for content in ("Alpha", "Beta_P2", "Gamma", "Delta", "Eps"):
            rows.append(f"3840_2160_original_{content}.mkv,4.5,0.1,0.5")
            for codec in ("av1", "hevc", "vvc"):
                for w, h, br in [(1280, 720, "500K"), (1280, 720, "3000K"), (1280, 720, "8000K"),
                                 (1920, 1080, "1000K"), (1920, 1080, "5000K"), (1920, 1080, "12000K"),
                                 (2560, 1440, "1000K"), (2560, 1440, "5000K"), (2560, 1440, "12000K"),
                                 (3840, 2160, "3000K"), (3840, 2160, "8000K"), (3840, 2160, "17000K"),
                                 (3840, 2160, "40000K")]:
                    rows.append(f"{w}_{h}_{br}_{codec}_{content}.mkv,3.0,0.1,0.5")
        labels = owner.e31video_labels("avt", "\n".join(rows).encode())
        self.assertEqual(len(labels), 195)
        self.assertEqual(labels[0]["video"], "avt/Alpha/av1_1280x720_500K")
        self.assertEqual({r["group"] for r in labels}, {"av1", "hevc", "vvc"})
        with self.assertRaisesRegex(ValueError, "unregistered AVT"):
            owner.e31video_labels("avt", "\n".join(rows + ["foo.mkv,1,1,1"]).encode())

    def test_legs_follow_the_july_configuration_map(self):
        self.assertEqual(set(owner.HDRVDC_LEGS["i"].values()), {"A"})
        self.assertEqual(owner.HDRVDC_LEGS["iii"], {"bright-near": "B", "bright-far": "D",
                                                    "dim-near": "C", "dim-far": "E"})

    def test_report_shape_and_nonfinite_refusal(self):
        labels = owner.e31video_labels("hdrvdc", hdrvdc_csv())
        target = np.array([r["human"] for r in labels])
        rng = np.random.default_rng(0)
        pred = target + rng.normal(0, 0.1, len(target))
        report = owner.e31video_report(pred, labels)
        self.assertEqual(len(report["per_study"]), 4)
        self.assertEqual(len(report["within_reference"]), 16)
        self.assertEqual(len(report["scatter"]["prediction"]), 464)
        self.assertGreater(report["pooled"]["srocc_signed"], 0.5)
        pred[3] = np.nan
        with self.assertRaisesRegex(ValueError, "INCOMPLETE"):
            owner.e31video_report(pred, labels)


if __name__ == "__main__":
    unittest.main()
