"""Synthetic placement amendments preserve training and coordinator authority."""

import json
from pathlib import Path
import tempfile
import unittest

import v40_caps_fix as owner


class CapsFix(unittest.TestCase):
    def fixture(self, root):
        bundle = root / "packet"
        bundle.mkdir()
        caps = {
            job: dict(memory="6g", hosts={}, build_commit="b" * 40)
            for job in owner.JOBSETS
        }
        original = owner.encoded(caps)
        (bundle / "jobset_caps.json").write_bytes(original)
        (bundle / "launch.py").write_text(
            'runpy.run_path(str(r / "scripts/rev4_featpot/v40_launch.py"), run_name="__main__")\n'
        )
        identities = dict(
            program_sha="c" * 64, files={"jobset_caps.json": owner.digest(original)}
        )
        for job in owner.JOBSETS:
            (bundle / f"AUTHORIZATION_REQUIRED-{job}.json").write_bytes(
                owner.encoded(dict(identities=identities))
            )
            template = dict(
                identities=identities,
                coordinator_message="",
                **{
                    k: False
                    for k in (
                        "reviewed",
                        "source_landed",
                        "pins_pushed",
                        "E30_completed",
                        "control_choice_frozen",
                    )
                },
            )
            (bundle / f"AUTHORIZATION_TEMPLATE-{job}.json").write_bytes(
                owner.encoded(template)
            )
        reference = root / "coordinator-caps.json"
        reference.write_bytes(
            owner.encoded(
                {"approved": dict(memory="6g", hosts={"i265": 3, "r3500": 2})}
            )
        )
        return bundle, reference

    def test_amendment_pins_all_four_entries_and_preserves_original_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, reference = self.fixture(Path(tmp))
            originals = {p.name: p.read_bytes() for p in bundle.iterdir()}
            authority = bundle / f"LAUNCH_AUTHORIZATION-{owner.JOBSETS[0]}.json"
            authority.write_bytes(b"original coordinator authority")
            untouched = (
                "program.tar.gz",
                "PACKAGE_PINNED.json",
                "assessment-program.tar.gz",
                "postfit.sh",
                "harvest_driver_v40.py",
            )
            for name in untouched:
                (bundle / name).write_bytes(b"unchanged " + name.encode())
            record = owner.amend(bundle, reference, "approved", "a" * 40)
            self.assertEqual(record["source_commit"], "a" * 40)
            self.assertEqual(record["fleet_hosts"], ["i265", "r3500"])
            self.assertEqual(authority.read_bytes(), b"original coordinator authority")
            for name in untouched:
                self.assertEqual(
                    (bundle / name).read_bytes(), b"unchanged " + name.encode()
                )
            for name, raw in originals.items():
                self.assertEqual(
                    (bundle / "capsfix-before-2026-10-09" / name).read_bytes(), raw
                )
            entries = json.loads((bundle / "jobset_caps.json").read_bytes())
            for job in owner.JOBSETS:
                self.assertEqual(
                    entries[job],
                    dict(
                        memory="6g",
                        hosts={"i265": 3, "r3500": 2},
                        build_commit="b" * 40,
                    ),
                )
                ids = json.loads(
                    (bundle / f"AUTHORIZATION_REQUIRED-{job}.json").read_bytes()
                )["identities"]
                template = json.loads(
                    (bundle / f"AUTHORIZATION_TEMPLATE-{job}.json").read_bytes()
                )
                self.assertEqual(template["identities"], ids)
                self.assertEqual(ids["program_sha"], "c" * 64)
                self.assertEqual(ids["caps_fix_source_commit"], "a" * 40)
                self.assertEqual(
                    set(ids["files"]),
                    {
                        "jobset_caps.json",
                        "launch.py",
                        "launch-runtime/v40_launch.py",
                        "CAPS_FIX.json",
                    },
                )
                for name, pin in ids["files"].items():
                    self.assertEqual(owner.digest((bundle / name).read_bytes()), pin)
                self.assertFalse(template["reviewed"])
            self.assertIn(
                'b / "launch-runtime/v40_launch.py"', (bundle / "launch.py").read_text()
            )
            with self.assertRaisesRegex(
                ValueError,
                "original caps authorization pins differ|original frozen launcher",
            ):
                owner.amend(bundle, reference, "approved", "a" * 40)

    def test_bad_reference_or_pins_refuse_without_creating_anything(self):
        for bad in ("empty", "nonfleet", "pins"):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as tmp:
                bundle, reference = self.fixture(Path(tmp))
                if bad == "pins":
                    (bundle / "jobset_caps.json").write_bytes(b"{}")
                else:
                    hosts = {} if bad == "empty" else {"not-a-fleet-host": 2}
                    reference.write_bytes(
                        owner.encoded({"approved": dict(hosts=hosts)})
                    )
                before = {p.name: p.read_bytes() for p in bundle.iterdir()}
                with self.assertRaises(ValueError):
                    owner.amend(bundle, reference, "approved", "a" * 40)
                self.assertEqual(
                    {p.name: p.read_bytes() for p in bundle.iterdir()}, before
                )

    def test_reference_cannot_be_the_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            bundle, _ = self.fixture(Path(tmp))
            with self.assertRaisesRegex(ValueError, "reference/live"):
                owner.amend(
                    bundle, bundle / "jobset_caps.json", owner.JOBSETS[0], "a" * 40
                )


if __name__ == "__main__":
    unittest.main()
