"""Report closure refuses incomplete grids and changed raw target order."""

import copy
import unittest
from pathlib import Path
import tempfile
from unittest.mock import patch

import v40_e31_report_evidence as owner


def fixture():
    report = dict(
        schema="e31-v40-upiq-training-report-v1",
        report_only=True,
        independent_test=False,
        shipping_adoption_authorized=False,
        control_pins_sha256="pin",
        panels={},
    )
    for split, n, refs in (("fit", 330, 26), ("development", 50, 4)):

        def rank(count):
            return dict(n=count, n_dropped=0, srocc_signed=-0.4)

        counts = [n // refs + int(i < n % refs) for i in range(refs)]
        panel = dict(
            pooled=rank(n),
            per_study={"korshunov": rank(n // 2), "narwaria": rank(n - n // 2)},
            within_reference={str(i): rank(c) for i, c in enumerate(counts)},
            scatter=dict(target=list(range(n)), prediction=list(range(n))),
        )
        entry = dict(
            schema="e31-upiq-training-report-v1",
            independent_test=False,
            shipping_adoption_authorized=False,
            panels={split: panel},
        )
        report["panels"][split] = {
            arm: {
                f"{f}_s{s}": copy.deepcopy(entry)
                for f in owner.FOLDS
                for s in range(10)
            }
            for arm in ("control", "uh4")
        }
    exposure = dict(
        mode="upiq",
        label_read_authorized=True,
        coordinator_message="approved",
        control_pins_sha256="pin",
    )
    return report, exposure


class Closure(unittest.TestCase):
    def test_complete_negative_rank_is_retained(self):
        result = owner.verify(*fixture())
        self.assertEqual(result["pooled_signed_srocc"]["fit"]["uh4"]["kadid_s0"], -0.4)
        self.assertFalse(result["shipping_adoption_authorized"])

    def test_protected_direct_and_symlink_report_refuse_before_open(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for name in ("holdout", "_sealed", "terminal", "aic3"):
                protected = root / name
                protected.mkdir()
                sentinel = protected / "sentinel.json"
                sentinel.write_text("{}")
                alias = root / (name + "-alias")
                alias.symlink_to(sentinel)
                for path in (sentinel, alias):
                    with patch(
                        "v40_panels.os.open",
                        side_effect=AssertionError("sentinel open"),
                    ):
                        with self.assertRaises(PermissionError):
                            owner.bound_bytes(path)

    def test_missing_cell_refuses(self):
        report, exposure = fixture()
        del report["panels"]["development"]["uh4"]["kadid_s0"]
        with self.assertRaisesRegex(ValueError, "forty-cell"):
            owner.verify(report, exposure)

    def test_changed_target_order_refuses(self):
        report, exposure = fixture()
        target = report["panels"]["fit"]["uh4"]["kadid_s0"]["panels"]["fit"]["scatter"][
            "target"
        ]
        target[0], target[1] = target[1], target[0]
        with self.assertRaisesRegex(ValueError, "targets/order"):
            owner.verify(report, exposure)

    def test_missing_reference_refuses(self):
        report, exposure = fixture()
        del report["panels"]["fit"]["uh4"]["kadid_s0"]["panels"]["fit"][
            "within_reference"
        ]["0"]
        with self.assertRaisesRegex(ValueError, "reference census"):
            owner.verify(report, exposure)


if __name__ == "__main__":
    unittest.main()
