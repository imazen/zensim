"""Synthetic extraction checks; no external datasets are opened."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
import zipfile

SPEC = importlib.util.spec_from_file_location('inspect_metric_dataset', Path(__file__).parents[1] / 'inspect_metric_dataset.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class InspectDatasetTests(unittest.TestCase):
    def setUp(self):
        root = Path(os.environ.get('TMPDIR', Path.home() / 'tmp'))
        root.mkdir(exist_ok=True, parents=True)
        self.scratch = tempfile.TemporaryDirectory(dir=root)
        self.root = Path(self.scratch.name)
        self.addCleanup(self.scratch.cleanup)

    def archive(self, name, data):
        path = self.root / 'data.zip'
        with zipfile.ZipFile(path, 'w') as z:
            z.writestr(name, data)
        return path

    def test_extract_and_exact_resume_preserve_original(self):
        archive = self.archive('nested/data.csv', 'id,score\na,1\n')
        original = MODULE.digest(archive)
        MODULE.extract(archive, self.root / 'unpacked')
        MODULE.extract(archive, self.root / 'unpacked')
        self.assertEqual(original, MODULE.digest(archive))
        self.assertEqual((self.root / 'unpacked/nested/data.csv').read_text(), 'id,score\na,1\n')

    def test_existing_different_content_is_never_overwritten(self):
        archive = self.archive('data.csv', 'a,1\n')
        out = self.root / 'unpacked'
        out.mkdir()
        (out / 'data.csv').write_text('b,2\n')
        with self.assertRaisesRegex(ValueError, 'CRC'):
            MODULE.extract(archive, out)
        self.assertEqual((out / 'data.csv').read_text(), 'b,2\n')

    def test_archive_traversal_refuses_before_member_write(self):
        for name in ['../escape', '/absolute', 'a\\b', 'C:drive']:
            with self.subTest(name=name):
                archive = self.archive(name, 'bad')
                with self.assertRaisesRegex(ValueError, 'unsafe'):
                    MODULE.extract(archive, self.root / 'unpacked')
        self.assertFalse((self.root / 'escape').exists())

    def test_inventory_keeps_markdown_links_and_csv_missingness(self):
        (self.root / 'README.md').write_text('[source](https://example.org/dataset)\n')
        (self.root / 'data.csv').write_text('id,score\n"a,b",\nc,2\n')
        report = MODULE.inspect(self.root)
        self.assertEqual(report['csv'][0]['rows'], 2)
        self.assertEqual(report['csv'][0]['missing'], {'score': 1})
        self.assertIn({'source': 'README.md', 'text': 'source', 'href': 'https://example.org/dataset'}, report['links'])


if __name__ == '__main__':
    unittest.main()
