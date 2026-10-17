"""Invented native ledger publication proofs, with strict lost-row refusals."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).parents[1] / "rev4_featpot"))
sys.path.insert(0, str(Path(__file__).parent))
from e36_native_ledger_admit import admit
from e36_store import sha
import test_e36_census_stream as fixtures


class Publication(unittest.TestCase):
    def fixture(self, root):
        directory = fixtures.Census().fixture(root)
        schedule = root / "scheduling-v7"
        schedule.mkdir()
        old = directory / "manifest.json"
        jobs = json.loads(old.read_text())
        new = schedule / "fallback.json"
        changed = copy.deepcopy(jobs)
        for job in changed:
            job["hint"] = dict(peak_mem_bytes=1, threads=1, vram_bytes=None)
        new.write_text(json.dumps(changed))
        worker = root / "invented-worker"
        worker.write_bytes(b"never executed")
        (root / "bulk-plan-v6.json").write_text(
            json.dumps(dict(phases=dict(fallback=dict(run="invented"))))
        )

        def bind(p):
            return dict(path=str(p), sha256=sha(p))

        (schedule / "amendment.json").write_text(
            json.dumps(
                dict(
                    worker=bind(worker),
                    manifests=dict(fallback=dict(**bind(new), original=bind(old))),
                )
            )
        )
        ids = ["1" * 64, "2" * 64]
        for label in ["original", "scheduled"]:
            (schedule / ("fallback." + label + ".ids.tsv")).write_text(
                "".join(
                    f"{i}\t{job_id}\timage{i}.png\n" for i, job_id in enumerate(ids)
                )
            )
        files = [bind(directory / f"blobs/{i}") for i in range(2)]
        rows = [
            dict(
                job_id=job_id,
                status="done",
                attempts=1,
                output_sha=files[i]["sha256"],
                kind_json=json.dumps(jobs[i]["kind"]),
            )
            for i, job_id in enumerate(ids)
        ]
        ledger = directory / "ledger.chunk-invented.parquet"
        pq.write_table(pa.Table.from_pylist(rows), ledger)
        (root / "GENERATION-v7.log").write_text("")
        return directory, ledger, rows, [j["kind"]["view"] for j in jobs], files

    def test_native_publication_matches_every_census_output_and_missing_row_refuses(
        self,
    ):
        with tempfile.TemporaryDirectory(
            dir=Path.home() / "tmp", prefix="e36-publish-"
        ) as d:
            root = Path(d)
            directory, ledger, rows, views, files = self.fixture(root)
            proof = admit(root, "invented", views, files)
            self.assertTrue(json.loads(Path(proof["path"]).read_text())["admitted"])
            ledger.rename(directory / "ledger.retained.bak")
            with self.assertRaisesRegex(ValueError, "complete native ledger"):
                admit(root, "invented", views, files)

    def test_attempt_or_output_changes_refuse(self):
        for key, value, pattern in [
            ("attempts", 2, "max-attempts-one"),
            ("output_sha", "3" * 64, "differs from native durable"),
        ]:
            with (
                self.subTest(key=key),
                tempfile.TemporaryDirectory(
                    dir=Path.home() / "tmp", prefix="e36-publish-"
                ) as d,
            ):
                root = Path(d)
                directory, ledger, rows, views, files = self.fixture(root)
                ledger.rename(directory / "ledger.retained.bak")
                rows[0][key] = value
                pq.write_table(pa.Table.from_pylist(rows), ledger)
                with self.assertRaisesRegex(ValueError, pattern):
                    admit(root, "invented", views, files)


if __name__ == "__main__":
    unittest.main()
