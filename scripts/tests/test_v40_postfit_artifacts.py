"""Failure injections through the real controller and artifact boundary."""
import copy
import contextlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import v40_score as owner
import v40_postfit_fixture as fixture

REPO = Path(__file__).resolve().parents[2]
METRICS = REPO.parent / "zenmetrics--v40"


def scratch(name):
    retained = os.environ.get("V40_POSTFIT_TEST_OUTPUT")
    if retained:
        path = Path(retained) / name
        path.mkdir(parents=True, exist_ok=False)
        return contextlib.nullcontext(str(path))
    return tempfile.TemporaryDirectory()


class Artifacts(unittest.TestCase):
    def test_control_and_all_arms_require_exact_structured_outputs(self):
        with scratch("artifact-boundary") as tmp:
            b = Path(tmp)
            grids, root = fixture.fixture(b)
            pins = fixture.freeze(b, grids)
            stdout = b / "stdout.json"
            fixture.write(stdout, pins)
            def complete(*args, only_control=False, **kwargs):
                return fixture.complete_fixture(grids, args[1], only_control)
            with patch.object(owner, "complete", complete):
                self.assertEqual(owner.validate_artifacts(b, "e29", b, b, b, b, b / "V40_CONTROL_PINS.json", stdout, freeze_control=True)["cells"], 40)
                for name in ("V40_CONTROL_PINS.json", "E29_CONTROL_PINS.json"):
                    path = b / name
                    original = path.read_bytes()
                    for bad in ({}, {**pins, "cells": {}}):
                        fixture.write(path, bad)
                        with self.assertRaises(ValueError):
                            owner.validate_artifacts(b, "e29", b, b, b, b, b / "V40_CONTROL_PINS.json", stdout, freeze_control=True)
                    path.write_bytes(original)
                for study in fixture.ARMS:
                    out = b / study
                    report = fixture.assessment(b, study, grids, out)
                    fixture.write(stdout, report["decisions"])
                    verify = lambda: owner.validate_artifacts(b, study, b, b, b, out, b / "V40_CONTROL_PINS.json", stdout, root=root)
                    self.assertEqual(verify()["status"], "PASS")
                    for key, value in (("schema", "wrong"), ("decisions", {}), ("program_sha256", "0" * 64), ("control_pins_sha256", "0" * 64), ("cells", {}), ("observation_counts", {}), ("artifacts", {})):
                        changed = {**report, key: value}
                        fixture.write(out / "decision.json", changed)
                        with self.assertRaises(ValueError): verify()
                    changed = copy.deepcopy(report)
                    changed["panels"]["control"].pop("kadid_s0")
                    fixture.write(out / "decision.json", changed)
                    with self.assertRaises(ValueError): verify()
                    changed = copy.deepcopy(report)
                    changed["panels"]["control"]["kadid_s0"]["signed"] = None
                    fixture.write(out / "decision.json", changed)
                    with self.assertRaises((ValueError, TypeError)): verify()
                    fixture.write(out / "decision.json", report)
                    for name in ("control/kadid_s0/pred.tsv", "control/kadid_s0/result.json", *( ["e29_sdr_decision.json"] if study == "e29" else [])):
                        path = out / name
                        original = path.read_bytes()
                        path.unlink()
                        with self.assertRaises((ValueError, OSError)): verify()
                        path.write_bytes(original)
                    for value in ("", "{}", '{"bad":NaN}'):
                        stdout.write_text(value)
                        with self.assertRaises(ValueError): verify()
                    fixture.write(stdout, report["decisions"])
                    self.assertEqual(verify()["status"], "PASS")

    def test_actual_postfit_controller_empty_and_valid_artifacts(self):
        with scratch("controller-boundary") as tmp:
            base = Path(tmp)
            env_script = base / "env.sh"
            env_script.write_text("""function .() {
    if [[ "$1" == "$HOME/.config/zen/s3env.sh" ]]; then return 0; fi
    builtin . "$@"
}
function python3() {
    if [[ "$1" == */harvest_driver_v40.py ]]; then printf '{"done":40,"total":40,"installed":40}\n'; return 0; fi
    command python3 "$@"
}
""")
            wrapper = """import argparse,json,os,sys
from pathlib import Path
from unittest.mock import patch
sys.path[:0] = PATHS
import v40_score as owner
import v40_postfit_fixture as fixture
b = Path(__file__).resolve().parent
mode = os.environ['V40_SYNTHETIC_MODE']
if '--verify-artifacts' in sys.argv:
    def complete(*args, only_control=False, **kwargs):
        return fixture.complete_from_disk(b, args[1], only_control)
    with patch.object(owner, 'complete', complete): owner.main()
else:
    study = sys.argv[sys.argv.index('--study') + 1]
    control = '--freeze-control' in sys.argv
    if mode == 'fail': raise SystemExit(9)
    if mode == 'empty': raise SystemExit(0)
    if mode == 'empty-object': print('{}'); raise SystemExit(0)
    if control:
        value = fixture.freeze(b, fixture.complete_from_disk(b, study))
    else:
        value = fixture.assessment(b, study, fixture.complete_from_disk(b, study), Path(sys.argv[sys.argv.index('--out') + 1]))['decisions']
    if mode == 'missing-artifact':
        (b / ('E29_CONTROL_PINS.json' if control else f'assessment-{study}/control/kadid_s0/pred.tsv')).unlink()
    if mode == 'incomplete':
        path = b / ('V40_CONTROL_PINS.json' if control else f'assessment-{study}/decision.json')
        value2 = json.loads(path.read_text())
        if control: value2['cells'].pop('kadid_s0')
        else: value2['panels']['control'].pop('kadid_s0')
        fixture.write(path, value2)
    if mode == 'nonfinite':
        path = b / ('V40_CONTROL_PINS.json' if control else f'assessment-{study}/decision.json')
        value2 = json.loads(path.read_text())
        if control: value2['cells']['kadid_s0']['bake_sha256'] = float('nan')
        else: value2['panels']['control']['kadid_s0']['signed'] = float('nan')
        path.write_text(json.dumps(value2))
    print(json.dumps(value))
""".replace('PATHS', repr([str(REPO / 'scripts'), str(REPO / 'scripts/rev4_featpot'), str(REPO / 'scripts/tests')]))
            for study in ('control', 'e29', 'e31', 'e32'):
                for mode in ('fail', 'empty', 'empty-object', 'missing-artifact', 'incomplete', 'nonfinite', 'valid'):
                    b = base / f'{study}-{mode}'
                    grids, root = fixture.fixture(b)
                    (b / 'v2e29').symlink_to(root, target_is_directory=True)
                    for alias in ('v2d1', 'v2e32'): (b / alias).symlink_to(root, target_is_directory=True)
                    if study != 'control': fixture.freeze(b, grids)
                    (b / 'score.py').write_text(wrapper)
                    env = {**os.environ, 'BASH_ENV': str(env_script), 'V40_SYNTHETIC_MODE': mode, 'TMPDIR': str(base)}
                    with (b / 'controller.log').open('w') as log:
                        result = subprocess.run(['bash', str(METRICS / 'scripts/jobsys/v40_postfit.sh'), str(b), study], env=env, stdout=log, stderr=subprocess.STDOUT, timeout=30)
                    text = (b / 'controller.log').read_text()
                    success = 'SDR RESULT:' in text or 'CONTROL FROZEN:' in text
                    self.assertEqual((result.returncode == 0, success), (mode == 'valid', mode == 'valid'), (study, mode, text))


if __name__ == '__main__': unittest.main()
