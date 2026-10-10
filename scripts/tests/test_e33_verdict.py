"""E33 9.4 verdict rules on synthetic inputs."""
import unittest

from e33_verdict import verdict


def gates(a_ok=True, c_ok=True, runtime=True):
    arm = lambda ok: {k: {"pass_" if k not in ("N1", "N2", "N3") else "pass": ok}
                      for k in ("N1", "N2", "N3", "C2", "C5", "G-STEER", "output_stage")} | {"runtime_pass": True}
    return {"arms": {"a": arm(a_ok), "c": arm(c_ok)}, "runtime": {"slower": []} if runtime else "INCOMPLETE"}


def e21(a=True, c=True, beats=False):
    return {"as_good": {"a": a, "c": c}, "c_vs_a": {"c_beats_a": beats}}


class Verdict(unittest.TestCase):
    def test_rules(self):
        self.assertEqual(verdict(e21(), gates())["adopt"], "a")
        self.assertEqual(verdict(e21(beats=True), gates())["adopt"], "c")
        self.assertEqual(verdict(e21(a=False), gates())["adopt"], "c")
        self.assertEqual(verdict(e21(), gates(c_ok=False))["adopt"], "a")
        self.assertEqual(verdict(e21(a=False), gates(c_ok=False))["adopt"], "control")
        self.assertEqual(verdict(e21(), gates(runtime=False))["status"], "INCOMPLETE")
        g = gates()
        g["arms"]["a"]["runtime_pass"] = False
        out = verdict(e21(), g)
        self.assertEqual((out["adopt"], out["arms"]["a"]["failed_gates"]), ("c", ["runtime"]))


if __name__ == "__main__":
    unittest.main()
