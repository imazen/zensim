"""STEER command admission on invented argv only; no table opens."""
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import v2_common as c
import v2_lodo_mlp as m

class SteerRecipe(unittest.TestCase):
    def test_s_only_adds_flag_to_existing_n_argv(self):
        args = ([], 1101, 101, 1825, Path("invented.keep"), "N", Path("invented.bin"))
        legacy = m.train_command(*args)
        smooth = m.train_command(*args, recipe=c.recipe_of("r0@h32:sh1"))
        self.assertEqual(smooth.count("--smooth-hinge"), 1)
        self.assertEqual([v for v in smooth if v != "--smooth-hinge"], legacy)
        self.assertEqual(c.recipe_of("r0@h32"), {})
        with self.assertRaises(ValueError):
            m.train_command([], 1101, 101, 1825, Path("invented.keep"), "P", Path("invented.bin"), recipe={"smooth_hinge":True})
        with self.assertRaises(ValueError):
            c.recipe_of("r0@h32:sh1:sh1")

    def test_registered_seed_rotation_and_full_data(self):
        for fold, source in enumerate(("kadid", "tid2013", "konfig", "cid22_a25")):
            for i in range(10):
                self.assertEqual(c.seeds(source, i), (c.INIT_SEEDS[i], c.SAMPLE_SEEDS[(i + fold) % 10]))
        for i in range(3):
            self.assertEqual(c.confirm_seeds(i), (c.INIT_SEEDS[i], c.SAMPLE_SEEDS[i]))

if __name__ == "__main__":
    unittest.main()
