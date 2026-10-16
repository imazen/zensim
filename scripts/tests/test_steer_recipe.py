"""STEER command admission on invented argv only; no table opens."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import v2_common as c
import v2_lodo_mlp as m
import v2_teacher as teacher

class SteerRecipe(unittest.TestCase):
    def test_complete_coverage_path_binding_preserves_legacy_default(self):
        with patch.dict('os.environ', {}, clear=True):
            a=teacher.coverage_output_path(0x98,Path('/scratch/tmp'))
            b=teacher.coverage_output_path(0x98,Path('/scratch/tmp'))
            self.assertNotEqual(a,b)
            self.assertNotIn('_steer_',str(a))
        with patch.dict('os.environ', {'ZEN_STEER_ARGV_SHA':'a'*64}):
            self.assertEqual(teacher.coverage_output_path(0x98,Path('/scratch/tmp')),Path('/scratch/tmp')/('coverage_cf98_steer_'+'a'*64+'.parquet'))
        for token in ['', 'a'*63, '../escape', 'A'*64]:
            with patch.dict('os.environ', {'ZEN_STEER_ARGV_SHA':token}):
                with self.assertRaises(ValueError):teacher.coverage_output_path(0x98,Path('/scratch/tmp'))
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
