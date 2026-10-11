"""The readers of stored `rank.<corpus>.srocc_ci` label each interval's kind and never mix
the two (STATS review P2, 2026-10-11).

`bake_verdict` wrote pair-bootstrap intervals until 2026-10-11 (no `srocc_ci_method`) and
reference-clustered ones since (`srocc_ci_method.design == "one-way: reference"`). These
tests drive the real readers on one record of each kind: the shared classifier, the
gauntlet's stamping and its page-side `ciHalf` (run under node, extracted verbatim from
`gauntlet.py`), the speed/accuracy page's bars and axis note, and the replication-wave
summary.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1]
for sub in ("", "v_next", "demos", "replication_wave"):
    sys.path.insert(0, str(SCRIPTS / sub))

from lib import srocc_ci  # noqa: E402

LEGACY = {"n": 4292, "srocc": 0.8927, "srocc_signed": 0.8927, "srocc_ci": [0.8865, 0.8988]}
REF = {
    "n": 4292,
    "srocc": 0.8927,
    "srocc_signed": 0.8927,
    "srocc_ci": [0.8690, 0.9177],
    "srocc_ci_signed": [0.8690, 0.9177],
    "srocc_ci_se": 0.0126,
    "srocc_ci_method": {"design": "one-way: reference", "units": 49, "resamples": 1000},
}
ROWS = dict(REF, srocc_ci_method={"design": "rows", "units": 4292, "resamples": 1000})


class ClassifierTest(unittest.TestCase):
    def test_kinds(self):
        self.assertEqual(srocc_ci.kind(LEGACY), srocc_ci.PAIR)
        self.assertEqual(srocc_ci.kind(REF), srocc_ci.REF)
        self.assertEqual(srocc_ci.kind(ROWS), srocc_ci.PAIR)
        self.assertIsNone(srocc_ci.kind({"srocc": 0.9}))
        self.assertIsNone(srocc_ci.kind(dict(LEGACY, srocc_ci=[float("nan"), 0.9])))
        self.assertIsNone(srocc_ci.kind(dict(LEGACY, srocc_ci=None)))
        with self.assertRaises(ValueError):
            srocc_ci.kind(dict(REF, srocc_ci_method={"design": "two-way"}))

    def test_common_kind_refuses_a_mix(self):
        self.assertEqual(srocc_ci.common_kind(["ref", None, "ref"]), "ref")
        self.assertIsNone(srocc_ci.common_kind(["ref", "pair"]))
        self.assertIsNone(srocc_ci.common_kind([]))


class GauntletTest(unittest.TestCase):
    def setUp(self):
        import gauntlet

        self.gauntlet = gauntlet
        self.src = Path(gauntlet.__file__).read_text()

    def test_stamp_labels_each_block(self):
        rank = {"cid22": dict(LEGACY), "kadid": dict(REF), "nonphoto": {"srocc": 0.9}}
        self.gauntlet.stamp_srocc_ci_kinds(rank)
        self.assertEqual(rank["cid22"]["srocc_ci_kind"], "pair")
        self.assertEqual(rank["kadid"]["srocc_ci_kind"], "ref")
        self.assertNotIn("srocc_ci_kind", rank["nonphoto"])

    def test_columns_split_by_kind_and_legend_names_both(self):
        self.assertIn("['cid22_ci_ref','CID22 95%CI± ref'", self.src)
        self.assertIn("['cid22_ci_pair','CID22 95%CI± pair'", self.src)
        self.assertNotIn("['cid22_ci','CID22 95%CI±'", self.src)
        self.assertNotIn("the board’s own srocc_ci understates it", self.src)
        self.assertIn("Compare half-widths only", self.src)

    def test_page_ci_half_returns_only_the_requested_kind(self):
        self.assertIsNotNone(shutil.which("node"), "node is required for the page-side check")
        m = re.search(r"^function ciHalf\(r,kind\)\{.*?^\}", self.src, re.S | re.M)
        self.assertIsNotNone(m, "ciHalf not found in gauntlet.py")
        rank = {"a": dict(LEGACY), "b": dict(REF)}
        self.gauntlet.stamp_srocc_ci_kinds(rank)
        js = m.group(0) + (
            f"\nconst rank={json.dumps(rank)};"
            "\nconsole.log(JSON.stringify([ciHalf(rank.a,'pair'),ciHalf(rank.a,'ref'),"
            "ciHalf(rank.b,'ref'),ciHalf(rank.b,'pair'),ciHalf(null,'ref')]));"
        )
        out = subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True)
        got = json.loads(out.stdout)
        self.assertAlmostEqual(got[0], (0.8988 - 0.8865) / 2)
        self.assertIsNone(got[1])
        self.assertAlmostEqual(got[2], (0.9177 - 0.8690) / 2)
        self.assertIsNone(got[3])
        self.assertIsNone(got[4])


class SpeedAccuracyPageTest(unittest.TestCase):
    def setUp(self):
        import speed_accuracy_page

        self.page = speed_accuracy_page

    def point(self, kind):
        rec = {"label": "x", "cls": "zensim", "ms": 1.0, "cv": 0.01, "srocc": 0.89, "hollow": False}
        block = REF if kind == "ref" else LEGACY
        rec["srocc_ci"] = block["srocc_ci"]
        rec["srocc_ci_kind"] = srocc_ci.kind(block)
        return rec

    def test_bars_carry_their_kind(self):
        ch = self.page.Chart([0.5, 2.0], [0.85, 0.95])
        ch.point(self.point("ref"))
        ch.point(self.point("pair"))
        svg = "".join(ch.parts)
        self.assertIn('class="ci ci-ref"', svg)
        self.assertIn('class="ci ci-pair"', svg)
        self.assertIn("reference-clustered bootstrap", svg)
        self.assertIn("pair (row) bootstrap", svg)

    def test_axis_note_names_the_kind_and_flags_a_mix(self):
        self.assertIn("reference-clustered", self.page.ci_kind_note(["ref", "ref"]))
        self.assertIn("MIXED", self.page.ci_kind_note(["ref", "pair"]))
        self.assertEqual(self.page.ci_kind_note([]), "")

    def test_records_joined_before_the_kind_existed_are_pair(self):
        self.assertEqual(self.page.stored_kind({"srocc_ci": [0.1, 0.2]}), "pair")


class ReplicationWaveTest(unittest.TestCase):
    def test_ci_half_never_pools_kinds(self):
        import analyze_wave as aw

        ref = (0.80, 0.90, "ref")
        pair = (0.84, 0.86, "pair")
        self.assertEqual(aw.ci_half([ref, ref])[1], "ref")
        self.assertAlmostEqual(aw.ci_half([ref, ref])[0], 0.05)
        self.assertEqual(aw.ci_half([ref, pair]), (None, "mixed"))
        self.assertEqual(aw.ci_half([]), (None, None))
        self.assertEqual(aw.fmt_half(None, "mixed"), "mixed CI kinds")
        self.assertEqual(aw.axis({"rank": {"cid22": REF}}, "cid22")[3], "ref")
        self.assertEqual(aw.axis({"rank": {"cid22": LEGACY}}, "cid22")[3], "pair")


if __name__ == "__main__":
    unittest.main()
