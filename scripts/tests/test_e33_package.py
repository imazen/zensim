"""E33 packet: caps always list hosts, the filler rehearsal refuses the V40 empty-map failure, fx1 is pinned."""

import json
from pathlib import Path
import tempfile
import unittest

import e33_package as owner
import v2_common

LAUNCHER = """#!/usr/bin/env bash
if [ -f "$CAPS" ]; then
  envelope=$(python3 - "$CAPS" "$JOBSET" "$HOST" "$N" <<'PY_CAP'
import json, subprocess, sys
caps,js,host,want=sys.argv[1:]
r=json.load(open(caps)).get(js)
if r is None:
    print('- -')
else:
    if host not in r['hosts']:
        raise SystemExit('jobset placement refused by resource envelope')
    p=subprocess.run(['ssh',host,'docker ps'],capture_output=True,text=True,check=True)
    active=sum(line==f'ZEN_RUN=jobs/{js}' for line in p.stdout.splitlines())
    free=max(0,r['hosts'][host]-active)
    print(r['memory'],min(int(want),free))
PY_CAP
  ) || exit 1
fi
"""


class Packet(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home() / "tmp")
        self.root = Path(self.tmp.name)
        self.launcher = self.root / "launch_v2.sh"
        self.launcher.write_text(LAUNCHER)
        self.approved = self.root / "approved.json"
        self.approved.write_text(json.dumps({"fitv40-control-20261007": {
            "memory": "6g", "hosts": {"i265": 3, "tower": 5}}}))

    def tearDown(self):
        self.tmp.cleanup()

    def test_caps_list_hosts_and_rehearsal_places_every_jobset(self):
        caps = owner.caps(self.root, self.approved, "7g")
        self.assertEqual(sorted(caps), sorted(owner.jobset(n) for n in ("control", "a", "c", "full")))
        self.assertEqual(caps[owner.jobset("c")]["memory"], "7g")
        self.assertEqual(caps[owner.jobset("a")]["memory"], "6g")
        report = owner.rehearse(self.root, self.launcher, ["i265", "tower"])
        for js in caps:
            self.assertEqual(report["jobsets"][js]["total_first_wave"], 8)
            self.assertEqual(report["jobsets"][js]["refused"], [])

    def test_empty_host_map_refuses(self):
        self.approved.write_text(json.dumps({"fitv40-control-20261007": {"memory": "6g", "hosts": {}}}))
        with self.assertRaises(ValueError):
            owner.caps(self.root, self.approved, "6g")
        (self.root / "jobset_caps.json").write_text(json.dumps({owner.jobset("a"): {"memory": "6g", "hosts": {}}}))
        with self.assertRaises(ValueError):
            owner.rehearse(self.root, self.launcher, ["i265"])

    def test_unregistered_host_refuses(self):
        (self.root / "jobset_caps.json").write_text(json.dumps({owner.jobset("a"): {"memory": "6g", "hosts": {"elsewhere": 2}}}))
        with self.assertRaises(ValueError):
            owner.rehearse(self.root, self.launcher, ["i265", "tower"])

    def test_fx1_declaration_is_pinned_and_matches_the_registered_arms(self):
        decl = v2_common.fx1_declaration()
        self.assertEqual(len(decl["direct"]), 410)
        self.assertEqual(len(decl["products"]), 410)
        self.assertEqual(sorted({b for _, b in decl["products"]}),
                         [422, 480, 509, 538, 567, 596, 625, 654, 683, 712])
        arms = owner.arms()
        self.assertEqual(sorted(set(decl["direct"]) | {b for _, b in decl["products"]}), arms["control"][1])
        self.assertEqual(v2_common.recipe_of(arms["c"][0])["derived_inputs"], "fx1")
        self.assertNotIn("derived_inputs", v2_common.recipe_of(arms["a"][0]))


    def test_wall_caps_are_the_registered_values(self):
        self.assertEqual(owner.WALL_CAPS, {"control": 4200, "a": 4200, "c": 7600, "full-a": 4200, "full-c": 7600})
        report = None
        owner.caps(self.root, self.approved, "6g")
        report = owner.rehearse(self.root, self.launcher, ["i265", "tower"])
        self.assertEqual(report["all_jobsets_concurrent"]["tower"], dict(cells=20, nominal_gib=120))

    def test_declared_destinations_follow_the_executor_layout(self):
        by_arm = owner.arms()
        spec, cols = by_arm["c"]
        lodo = owner.cell_argv("v2_lodo_mlp.py", spec, cols, 0, "/var/tmp/rev4-featpot/e33-c-results/cells/x__N/without_kadid_s0", ["--heldout", "kadid"])
        full = owner.cell_argv("v2_confirm_fit.py", spec, cols, 0, "/var/tmp/rev4-featpot/e33-full-results/confirm/cells/x__N/full_s0", ["--pack-production"])
        for argv in (lodo, full):
            dest = Path(argv[argv.index("--dest") + 1])
            suffix = ("cells",) if argv[0] == "v2_lodo_mlp.py" else ("confirm", "cells")
            rel = dest.relative_to("/var/tmp/rev4-featpot")
            self.assertEqual(rel.parts[1:-2], suffix)

    def test_fx1_token_reaches_the_trainer_argv(self):
        import v2_lodo_mlp
        decl = v2_common.fx1_declaration()
        keep = self.root / "keep.txt"
        keep.write_text("\n".join(map(str, decl["direct"])) + "\n")
        cmd = v2_lodo_mlp.train_command([], 1, 2, 1853, keep, "N", self.root / "out.bin",
                                        {"derived_inputs": "fx1"})
        self.assertIn("--derived-inputs", cmd)
        self.assertEqual(cmd[cmd.index("--derived-inputs") + 1], str(v2_common.FX1_DECLARATION))
        self.assertIn("--nonneg-distance", cmd)
        plain = v2_lodo_mlp.train_command([], 1, 2, 1853, keep, "N", self.root / "out.bin", {})
        self.assertNotIn("--derived-inputs", plain)
        keep.write_text("\n".join(map(str, decl["direct"][:-1])) + "\n")
        with self.assertRaises(ValueError):
            v2_lodo_mlp.train_command([], 1, 2, 1853, keep, "N", self.root / "out.bin",
                                      {"derived_inputs": "fx1"})


if __name__ == "__main__":
    unittest.main()
