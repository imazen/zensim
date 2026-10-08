# RELEASEGATE6 landing preparation — 2026-10-08

The reviewed RELEASEGATE5 + RELEASEGATE6 chain is rebased onto
`6a37f4e45af4f15e8076b82e45d4e8d805cfe02a` in an isolated jj workspace.
The eight source commits were duplicated onto this base, preserving the
original review bookmarks and working copies. Rebased reviewed tip before
the follow-up: `acf83433769eee5b3df822a113d7c34c61715bc7`.

A lock-respecting writer could previously rewrite the ledger in place,
remove the spent reservation, and still get a successful run. The result
transaction now requires the exact `KADID-TERMINAL-SPENT:<design>` token
in the retained canonical ledger descriptor under its exclusive flock,
before appending the result. The existing failure boundary maps a missing
reservation to `exposure-refused`; the CLI returns 2 without printing PASS.
The spent journal continues to prevent another read. An assessment JSON alone
is not a successful transaction: a result-append refusal can leave that file,
as the reviewed round-six behavior already documents.

Two regressions exercise actual orchestration with synthetic labels and the
pinned native panel: a lock-respecting same-inode rewrite that deletes only
the reservation line must refuse, and a cooperating append-only writer must
preserve both the reservation and result. The operational requirements now
state that cooperating editors may only append and must preserve every prior
line. No real protected payload, authorization, exposure ledger, model or
acceptance threshold was opened or changed. No actual terminal read or
preparation filesystem qualification is authorized by these tests.

Evidence and final test/commit pins:
`/mnt/v/output/zensim/releasegate6-land-2026-10-08/`, mirrored to
`/mnt/tower/output/zensim/releasegate6-land-2026-10-08/`.
The borrowed panel remains pinned in the prior RELEASEGATE6 archive;
no duplicate native panel or image is created. The newly created Cargo target
is removed after evidence verification. All work is local; no push and no
shared-main movement. The original round-five/six worklogs and evidence retain
their historical source and artifact pins; this record owns the new landing tip.

Verification: the full RELEASEGATE suite passes 66/66 with no skips. The same
new deletion regression on original reviewed tip `9a67a47d` fails exactly once
(`AssertionError: 0 != 2`): that CLI reports success after the spent line is
removed, while the new tree returns `exposure-refused` without printing PASS.
The full `cargo test -p zensim --all-features` run passes 900 tests with zero
failures and 27 existing ignores. All eight rebased per-commit diffs are
byte-identical to the reviewed originals, recorded in `REBASE_MAP.json`.
CI-exact `just clippy`, `just lint-scripts` (866 runnable scripts), scoped
Ruff F, and whitespace validation all pass. No expectations, numerical bars
or existing ignores were weakened. Runtime source `_terminal_owner.py` is
29,998 bytes; every newly committed non-Rust file stays below 30,000 bytes.
Full commands, source pins and run-heavy completion/resource lines are retained
in `CHECK_VERIFICATION.json`, `FOLLOWUP_SOURCE_HASHES.json` and the logs.
