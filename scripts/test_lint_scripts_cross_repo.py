"""Exact V40 allowances; synthetic repositories, no data or network access."""

import contextlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import lint_scripts as owner


class CrossRepositoryScripts(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        scratch = Path(os.environ.get("TMPDIR", str(Path.home() / "tmp")))
        self.base = Path(
            self.stack.enter_context(tempfile.TemporaryDirectory(dir=scratch))
        )
        self.root = self.base / "zensim"
        self.root.mkdir()
        self.stack.enter_context(patch.object(owner, "ROOT", self.root))
        self.stack.enter_context(patch.object(owner, "ZEN", self.base))
        self.ref = "/".join(("scripts", "jobsys", "v40_postfit.sh"))
        self.other_caller = "/".join(("scripts", "tests", "unlisted_caller.py"))
        self.callers = (
            "scripts/tests/test_v40_postfit_artifacts.py",
            "scripts/tests/v40_r4_prepare.py",
        )

    def script(self, caller, literal):
        path = self.root / caller
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("target = " + repr(literal) + "\n")
        return path

    def freeze(self, with_target):
        sibling = self.base / "zenmetrics"
        subprocess.run(
            ["jj", "git", "init", "--colocate", str(sibling)],
            check=True,
            capture_output=True,
        )
        path = sibling / (self.ref if with_target else "README.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic frozen target\n")
        subprocess.run(
            ["jj", "describe", "-m", "test: synthetic frozen lint target"],
            cwd=sibling,
            check=True,
            capture_output=True,
        )
        return subprocess.check_output(
            ["jj", "log", "-r", "@", "--no-graph", "-T", "commit_id"],
            cwd=sibling,
            text=True,
        ).strip()

    def bind(self, revision):
        entries = {
            (caller, self.ref): ("zenmetrics", revision, "synthetic target")
            for caller in self.callers
        }
        return patch.dict(owner.CROSS_REPO_SCRIPT_REFS, entries)

    def test_allowlist_has_exactly_the_authorized_two_references(self):
        self.assertEqual(
            set(owner.CROSS_REPO_SCRIPT_REFS),
            {(caller, self.ref) for caller in self.callers},
        )
        for repo, revision, reason in owner.CROSS_REPO_SCRIPT_REFS.values():
            self.assertEqual(repo, "zenmetrics")
            self.assertEqual(revision, "66c0a961876bc582fb73484ff0cca7f2ffed673b")
            self.assertTrue(reason)

    def test_registered_refs_use_frozen_objects_even_when_worktree_file_is_gone(self):
        revision = self.freeze(with_target=True)
        (self.base / "zenmetrics" / self.ref).unlink()
        with self.bind(revision):
            for caller in self.callers:
                self.assertEqual(owner.check(self.script(caller, self.ref)), [])

    def test_worktree_and_local_files_cannot_replace_missing_frozen_target(self):
        revision = self.freeze(with_target=False)
        for repo in (self.base / "zenmetrics", self.root):
            path = repo / self.ref
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("unfrozen substitute\n")
        with self.bind(revision):
            for caller in self.callers:
                self.assertEqual(
                    owner.check(self.script(caller, self.ref)),
                    [f"DEAD-SCRIPT missing frozen sibling script: {self.ref}"],
                )

    def test_present_nonrepository_sibling_refuses(self):
        path = self.base / "zenmetrics" / self.ref
        path.parent.mkdir(parents=True)
        path.write_text("not a frozen object\n")
        self.assertEqual(
            owner.check(self.script(self.callers[0], self.ref)),
            [f"DEAD-SCRIPT missing frozen sibling script: {self.ref}"],
        )

    def test_missing_frozen_revision_refuses(self):
        self.freeze(with_target=True)
        with self.bind("0" * 40):
            self.assertEqual(
                owner.check(self.script(self.callers[0], self.ref)),
                [f"DEAD-SCRIPT missing frozen sibling script: {self.ref}"],
            )

    def test_unlisted_caller_and_literal_still_fail(self):
        other = "/".join(("scripts", "jobsys", "unlisted_missing.sh"))
        cases = (
            (self.other_caller, self.ref, self.ref),
            (self.callers[0], other, other),
            (self.callers[0], "python3 " + self.ref, self.ref),
        )
        for caller, literal, missing in cases:
            with self.subTest(caller=caller, literal=literal):
                self.assertEqual(
                    owner.check(self.script(caller, literal)),
                    [f"DEAD-SCRIPT calls missing script: {missing}"],
                )

    def test_without_sibling_only_exact_registered_refs_are_allowed(self):
        for caller in self.callers:
            self.assertEqual(owner.check(self.script(caller, self.ref)), [])
        self.assertEqual(
            owner.dead_script_refs(self.ref),
            [f"DEAD-SCRIPT calls missing script: {self.ref}"],
        )

    def test_ordinary_local_script_references_still_pass(self):
        path = self.root / self.ref
        path.parent.mkdir(parents=True)
        path.write_text("local script\n")
        self.assertEqual(owner.check(self.script(self.other_caller, self.ref)), [])


if __name__ == "__main__":
    unittest.main()
