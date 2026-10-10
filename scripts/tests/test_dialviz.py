"""dialviz: readers parse the live sources, agree with independent sources, and refuse changed shapes.

Runs against the repository and the sibling zenanalyze checkout. Reads no label
table. Negative controls mutate a source in memory and require SourceShapeError.
"""
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialviz import build, catalogue, experiments_src, feature_defs_src, featuresets_src, model, quotes, sources, zenanalyze_src  # noqa: E402
from dialviz.mdparse import SourceShapeError  # noqa: E402
from dialviz.sources import REPO, Ctx  # noqa: E402

GATES_JSON = os.environ.get("DIALVIZ_INTEGRITY_GATES")


class MutatedCtx(Ctx):
    """A Ctx whose text for one path is transformed before readers see it."""

    def __init__(self, path, fn):
        super().__init__()
        self._path, self._fn = path, fn

    def text(self, rel, reader):
        t = super().text(rel, reader)
        return self._fn(t) if rel == self._path else t


def sub_once(pattern, repl):
    def f(t):
        out, n = re.subn(pattern, repl, t, count=1, flags=re.M)
        assert n == 1, f"mutation pattern {pattern!r} did not match"
        return out
    return f


class LiveSources(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ctx = Ctx()
        cls.m = model.build(cls.ctx, Path(GATES_JSON) if GATES_JSON else None)

    def test_minimum_entity_counts(self):
        m = self.m
        self.assertGreaterEqual(len(m["gates"]["gates"]), 15)
        self.assertGreaterEqual(len(m["bugs"]), 10)
        self.assertGreaterEqual(len(m["peers"]["scorers"]), 15)
        self.assertGreaterEqual(len(m["splits"]["datasets"]), 15)
        self.assertGreaterEqual(len(m["splits"]["ledger"]), 30)
        self.assertGreaterEqual(len(m["experiments"]), 25)
        self.assertGreaterEqual(sum(len(r["arms"]) for r in m["results"].values()), 10)
        self.assertGreaterEqual(len(m["za"]["features"]), 100)
        self.assertEqual(len(m["wanted"]), len(catalogue.WANTED))

    def test_feature_layout_matches_registry_json(self):
        fam = {b["family"]: b for b in self.m["features"]["blocks"]}
        tokens = self.m["featuresets"]["tokens"]
        checked = 0
        for t, v in tokens.items():
            if t in fam and v["ids"]:
                self.assertEqual(v["ids"], list(range(fam[t]["lo"], fam[t]["hi"] + 1)), t)
                checked += 1
        self.assertGreaterEqual(checked, 10)
        fd = self.m["features"]
        self.assertIn(fd["full_width"], fd["registered_widths"])
        self.assertEqual([s["id"] for s in fd["slots"]], list(range(fd["full_width"])))

    def test_feature_slots_match_e33_fx1_table(self):
        slot = {s["id"]: s for s in self.m["features"]["slots"]}
        table = self.m["named_sets"]["fx1_table"]
        self.assertEqual(len(table), 410)
        for row in table:
            s = slot[row["id"]]
            self.assertEqual((s["signal"], s["scale"], s["channel"]), (row["signal"], row["scale"], row["channel"].lower()), row)
            self.assertEqual(slot[row["fragility_id"]]["signal"], "pjnd_fragility")
        frag = self.m["named_sets"]["fragility_ids"]
        self.assertTrue(all(slot[f]["form"] == "reference_only" for f in frag))
        self.assertEqual(sorted(self.m["named_sets"]["e33_direct"] + frag), self.m["named_sets"]["by_v2fy"])

    def test_registry_hashes_reproduce(self):
        sets = [s for s in self.m["featuresets"]["sets"] if s["hash_ok"] is not None]
        self.assertGreater(len(sets), 40)
        self.assertTrue(all(s["hash_ok"] for s in sets), [s["id"] for s in sets if not s["hash_ok"]])
        self.assertNotEqual(featuresets_src.slots_hash8([1, 2, 3]), featuresets_src.slots_hash8([1, 2, 4]))

    def test_gate_crosswalk_covers_every_instrument(self):
        cw = self.m["crosswalk"]
        for key, names in (("terminal", self.m["terminal"]["gates"]), ("freeze", self.m["freeze"]["gates"]),
                           ("contract", [r["requirement"] for r in self.m["scorecard"]["contract"]])):
            mapped = {n for v in cw.values() for n in v[key]}
            self.assertEqual(mapped, set(names), key)

    def test_every_locator_resolves(self):
        for p in catalogue.WANTED + catalogue.UNWANTED:
            for k in ("rules", "owners", "state", "evidence"):
                for loc in p.get(k, []):
                    quotes.resolve(self.ctx, loc)
        for loc in catalogue.EVALUATION.values():
            quotes.resolve(self.ctx, loc)

    def test_router_drift_is_read_from_the_artifacts(self):
        routers = [p for p in self.m["pickers"] if p["kind"].startswith("ZNPR")]
        self.assertGreaterEqual(len(routers), 3)
        for r in routers:
            self.assertGreaterEqual(len(r["pins"]), 50)
            states = {x["state"] for x in r["pins"]}
            self.assertTrue(states <= {"current", "hash-changed", "missing"}, states)


class NegativeControls(unittest.TestCase):
    def refuse(self, path, mutation, reader):
        with self.assertRaises(SourceShapeError):
            reader(MutatedCtx(path, mutation))

    def test_release_gate_map_column_renamed(self):
        self.refuse(sources.RELEASE_GATE_MAP, sub_once(r"\| Status for production \|", "| Status |"), sources.release_gates)

    def test_release_gate_status_unclassifiable(self):
        with self.assertRaises(SourceShapeError):
            sources.classify_gate_status("pending review", "")

    def test_paper_gates_schema_changed(self):
        self.refuse(sources.PAPER_GATES, sub_once(r'"paper-gates-2026-09-23-v1"', '"paper-gates-v2"'), sources.paper_gates)

    def test_known_bugs_without_dates(self):
        self.refuse("CLAUDE.md", lambda t: re.sub(r"^\* \*\*\d{4}-\d{2}-\d{2} — ", "* **", t, flags=re.M), sources.known_bugs)

    def test_feature_defs_signal_removed(self):
        self.refuse(feature_defs_src.FEATURE_DEFS, sub_once(r'^\s*v1\(F, 9, "mse", Mean, K\),\n', ""), feature_defs_src.read)

    def test_feature_defs_unknown_constructor(self):
        self.refuse(feature_defs_src.FEATURE_DEFS, sub_once(r'v1\(F, 9, "mse", Mean, K\)', 'v9(F, 9, "mse", Mean, K)'),
                    feature_defs_src.read)

    def test_feature_defs_reader_follows_the_source(self):
        # positive control: a changed form in the source changes the parsed slots (the reader is not a lookup table)
        ctx = MutatedCtx(feature_defs_src.FEATURE_DEFS,
                         sub_once(r'(21,\s*"pjnd_fragility",\s*Mean,\s*)ReferenceOnly', r"\1Difference"))
        d = feature_defs_src.read(ctx)
        self.assertEqual(next(s for s in d["slots"] if s["id"] == 422)["form"], "difference")

    def test_e33_fx1_columns_changed(self):
        self.refuse(featuresets_src.E33, sub_once(r'"fragility_id"\s*\]', '"frag"]'), featuresets_src.named_sets)

    def test_registry_set_missing_field(self):
        self.refuse(featuresets_src.REGISTRY, sub_once(r'"slots_hash8": "62adfc93",', ""), featuresets_src.registry)

    def test_zenanalyze_row_malformed(self):
        self.refuse(zenanalyze_src.FEATURE_RS, sub_once(r"^(\s*)Variance = 0 : f32 => variance,", r"\1Variance = zero : f32 -> variance,"),
                    zenanalyze_src.catalogue)

    def test_terminal_gates_renamed(self):
        self.refuse(model.TERMINAL_OWNER, sub_once(r"^GATES = \(", "GATE_NAMES = ("), model.terminal_gates)

    def test_freeze_check_function_renamed(self):
        self.refuse(model.FREEZE_CHECK, sub_once(r"^fn qualification_report\(", "fn qualification_report_v2("),
                    model.freeze_check_gates)

    def test_experiment_result_schema_changed(self):
        rel = experiments_src.RESULTS["E28"][0]
        self.refuse(rel, sub_once(r'"e28-decision-v1"', '"e28-decision-v2"'), experiments_src.results)

    def test_catalogue_locator_missing(self):
        loc = catalogue.WANTED[1]["state"][0]
        self.refuse(loc["row"], lambda t: t.replace("N1 near-identity ceiling", "N1 ceiling"), lambda c: quotes.resolve(c, loc))


class SiteBuild(unittest.TestCase):
    def test_build_writes_linked_pages(self):
        base = Path(os.environ.get("TMPDIR", str(Path.home() / "tmp")))
        base.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base) as d:
            out = Path(d) / "site"
            self.assertEqual(build.main(["--out", str(out)]), 0)
            pages = sorted(out.rglob("*.html"))
            self.assertGreater(len(pages), 60)
            broken = []
            for p in pages:
                html = p.read_text()
                self.assertNotIn("{BASE}", html, p)
                for href in re.findall(r'href="([^"#:]+\.html)(?:#[^"]*)?"', html):
                    if not (p.parent / href).resolve().is_file():
                        broken.append(f"{p.relative_to(out)} -> {href}")
            self.assertEqual(broken, [])
            self.assertTrue((out / "search.js").read_text().startswith("window.DIALVIZ_INDEX="))


if __name__ == "__main__":
    unittest.main()
