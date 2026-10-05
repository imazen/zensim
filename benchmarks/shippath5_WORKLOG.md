# SHIPPATH5 worklog — lossless native checkpoint paths, 2026-10-05

One narrow correction to SHIPPATH4's reviewed parent f3b3fc3c6ff80a104e12ac080b92fae85d396817. Existing shippath jj workspace; local commit only. The public train_mlp API still accepts native PathBuf dump directories, including non-UTF-8 Unix names. No API or training arithmetic changed.

The shared checkpoint writer now serializes a valid UTF-8 path as the existing JSON string. Otherwise it uses Serde's native OsStr encoding: a tagged Unix raw-byte array or Windows native-code-unit array. The stamping reader accepts both representations and reconstructs the exact OsString/PathBuf. No replacement/display string identifies ownership. Successful fs::write still precedes the receipt, and stamping still targets only receipts from this invocation's log.

Evidence and owner locations:

- zensim-validate/src/mlp_train/mod.rs:1034 — both CPU lanes' shared writer.
- zensim-validate/src/bin/zensim_mlp_train.rs:2879 — lossless ownership reader.
- zensim-validate/src/mlp_train/mod.rs:12008 — actual train_mlp regression using the reviewer's 48 synthetic rows, four features, H8, three epochs x 128 draws. Invalid bytes ff and fe form distinct Unix names with identical lossy display. Both fits and a UTF-8 fit return a valid final model and all three dumps; exact receipt paths, epochs and byte counts agree, and entire model/dump bytes match. This test failed with rc101 on reviewed production code after adding only the test, reproducing the epoch000 serialization panic; it passes after the fix.
- zensim-validate/src/bin/zensim_mlp_train.rs:5281 — actual stamp owner changes only the recorded non-UTF-8 path; the foreign file with the same lossy display remains byte-identical.

Recompiled the reviewer's unchanged path_control.rs against the new public library and ran both path variants. Compared best plus epochs000/001/002 against the retained reviewer parent UTF-8/non-UTF-8 and reviewed UTF-8 controls: every full file is byte-identical across all five cases. Epoch000 SHA-256 f2a53f0ef708c809eb21942746dde208a1a98c7b1ca809785ebcc642585d9d88. This is raw byte parity, without metadata normalization. Reproducer/source/binary hashes and commands are retained in /var/tmp/shippath5 and benchmarks/shippath5_checks_2026-10-05.json.

Passed: new public-owner regression (1), checkpoint admission/stamping tests (4), both actual writer-lane tests (2), failed-write receipt test (1), retained actual CLI surviving-file and p2 first-fit regressions (2, including seven artifact cases), trainer/library/harness builds, CI-exact just clippy, cargo fmt --all --check, just lint-scripts (816 scripts). Every heavy command used the required --mem 16G --jobs 8 wrapper and task TMPDIR; numeric fits used one Rayon/OMP thread.

Unix behavior is tested; the lossless Windows representation uses Serde's existing platform implementation but no Windows execution is claimed. No packing/inference owner, admission/root guard, role decision, recipe or selected-epoch policy changed. No protected/holdout or real training input read, fleet use, push, model qualification or human-role approval. The pending decision remains pending. SHIPPATH5_DONE.md is written only after the local commit and evidence verification.
