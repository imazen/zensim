# V40 reviewed package: landing preparation

The coordinator requested merges preserving the reviewed producer and all frozen
source commits. Both merges are conflict-free; no bound source or lockfile was
resolved or rewritten. Publication and launch remain coordinator actions.

- zensim merge: `2b3875fce60ee4a2d1e1ce23ff74611090d8a7e5`, parents current
  `main@origin=c3235f2a47f07beaf75b48ce440da7ab43256c5e` and reviewed
  `16927483a6eafc23af81150d91b8112f800fcc28`.
- zenmetrics merge: `b2d5690380029b73c8499ce39c9cd706bf16dbfc`, parents current
  `master@origin=e056defcc113e81e55207704dff910a7a0471d2d` and reviewed
  `f2f1544a59ce35c45ef08a64da9cc95757a7a0c4`.
- Retained binary producer: `05621a03094bcc1646d92874586b94551a5e36f2`.
- Frozen assessment source: `f0584d00fab0bc38bce236ca386e34f9e8b63308`.
- Frozen zenmetrics program owner: `66c0a961876bc582fb73484ff0cca7f2ffed673b`.

## Bound-byte gate

**The merged-tree `FINAL_LOCAL_CHECK` refuses.** Six native files differ from the
160-file producer inventory: `zensim/src/{blur.rs,feature_defs.rs,
feature_v2_stream.rs,fused.rs}`, `zensim-validate/src/bin/bake_verdict.rs` and
`zensim-validate/src/dial_addressability.rs`. Seven assessment files differ from
the 619-file frozen inventory: `scripts/demos/{speed_matrix_report.py,
speedq_run.py,test_speedq.py}`, `scripts/rev4_featpot/{_terminal_bound_io.py,
kadid_terminal_read.py,v2c_labels.py}` and
`scripts/tests/test_kadid_terminal_read.py`.

This matches the independent review's landing warning. No conflict caused the
drift: the merge retains newer main changes. The reviewed packet is still bound
to its original source ancestors. The stricter requested condition that merged
source bytes equal those frozen bytes is unsatisfied. Keeping the original
packet as the execution authority needs coordinator disposition; choosing merged
tools or assessment owners requires rebuild, rebinding, parity and review. These
instructions do not supply that disposition or claim merged-tool parity.

Exact per-file hashes and test command/status/log receipts are retained under
`/home/lilith/tmp/v40-land/`, in `MERGE_BOUND_DRIFT.json`,
`TEST_RECEIPTS.json`, `final-local-check.log` and the individual suite logs.
The frozen packet and its previous PASS receipt are unchanged.

## Merge verification receipts

The merged zensim tree passes `cargo test --locked -p zensim --all-features`
(including doctests), the zensim-validate library/trainer suites, the existing
E32/SHIPPATH/E29/E26/import-guard/E31 suites, V40 projection/statistics/launch/panel
tests, postfit artifact/controller tests, producer-metadata tests, all four
shipped-binary admission suites, CI-exact `just clippy` and scoped format checks.
The statistics invocation initially lacked `ZEN_PANEL_BIN`; its retry explicitly
uses the retained reviewed panel and passes unchanged expectations.

`just lint-scripts` **fails** on two cross-repository false positives:
`scripts/tests/test_v40_postfit_artifacts.py` and
`scripts/tests/v40_r4_prepare.py` resolve
`METRICS / "scripts/jobsys/v40_postfit.sh"` into zenmetrics, while the linter
checks that string relative to zensim. The actual zenmetrics owner exists and
the postfit/controller suites pass. Both referencing files belong to the frozen
assessment inventory; they were preserved rather than edited during landing
preparation. This required gate remains unresolved; no exemption or relaxed
expectation was introduced.

The zenmetrics merge passes its 48 fit-tool tests and postfit checks. Initial
worker invocations against live sibling checkouts fail dependency resolution
(`ultrahdr-rs` asks for unpublished `zenjpeg ^0.9.0`). The canonical
`scripts/ci/lock.sh --check --rev b2d5690380029b73c8499ce39c9cd706bf16dbfc`
exports the exact merge and passes with all 23 committed CI sibling pins.
In that isolated snapshot, worker claim regressions, full worker tests including
doctests, and worker clippy pass. No live sibling or committed lockfile changed.

The supplemental `FROZEN_ANCESTOR_CHECK.json` passes all 160 native and 619
assessment source records against their preserved ancestors, retained tools,
frozen assessment runtime and actual local image ID. It explicitly retains
`merged_tree_equality=FAIL` and `coordinator_disposition=PENDING`; it does not
replace the requested merged-tree check. Its first invocation failed because
a jj workspace lacks a colocated `.git`; the retry resolves `jj git root` and
checks the same objects. Both logs are retained.

No scientific fits or new parity fits ran during this preparation. The reviewed
packet's full-budget parity still covers 2 of 40 controls; the 40 fresh controls
and 160 arm fits remain future scientific work.

## Commands to list for the coordinator

**Not executed.** Resolve the bound-byte gate and any reported test failures
before publishing or authorizing. If the remote tips advance, preserve bound
ancestors with another merge rather than rebasing them. Safe-push refuses a
non-descendant target. These commands publish the named local merge commits;
additional local documentation commits are listed separately in the completion
record.

```bash
cd /home/lilith/work/zen/zensim
TMPDIR=$HOME/tmp/v40-land flock ~/tmp/zensim-paper/rev4/heavy.lock \
  ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  bash scripts/safe_push.sh -b main -r 2b3875fce60ee4a2d1e1ce23ff74611090d8a7e5
cd /home/lilith/work/zen/zenmetrics
TMPDIR=$HOME/tmp/v40-land flock ~/tmp/zensim-paper/rev4/heavy.lock \
  ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  bash scripts/safe_push.sh -b master -r b2d5690380029b73c8499ce39c9cd706bf16dbfc
```

The zenmetrics merge includes the reviewed `v40_program_pins.json`. The frozen
fit program is baked into the image; the launcher publishes data to the existing
jobset input keys. It has no separate program-upload key. Before image push,
resolve the actual local identity and require the frozen image ID:

```bash
image=ghcr.io/imazen/zenfleet-worker:fit-v40r4-05621a03094b-wc581bdb55f88
test "$(docker image inspect -f '{{.Id}}' "$image")" = \
  sha256:a8d415a0000dd53866592f0b5ad89c619c6dba8f6e57ac44ed1182efb5cf81ff
docker push "$image"
```

Exact manifest/data publication commands, independent of queue insertion, are:

```bash
set -euo pipefail
bundle=/mnt/v/output/zensim/v40r4-2026-10-08
. "$HOME/.config/zen/s3env.sh" >/dev/null 2>&1
for study in control e29 e32 e31; do
  jobset="fitv40-$study-20261007"
  read -r data_file data_sha < <(python3 - "$bundle" "$jobset" <<'PY'
import json,sys
from pathlib import Path
ids=json.loads((Path(sys.argv[1])/f'AUTHORIZATION_REQUIRED-{sys.argv[2]}.json').read_text())['identities']
print(ids['data_file'],ids['data_sha'])
PY
  )
  s5cmd --endpoint-url "$EP" cp "$bundle/fit-manifest-$jobset.json" \
    "s3://zentrain/jobs/$jobset/manifest.json"
  s5cmd --endpoint-url "$EP" cp "$bundle/$data_file" \
    "s3://zentrain/jobs/$jobset/inputs/$data_sha"
done
```

`launch.py` performs these same publications, plus a jobset control object and
queue insertion, only after its authorization gate. Do not create an unpaused
control object separately during preparation. The unchanged TRAIN archive pins
are E29/control `9c3eff1b740d2b7a77a235a16a3cd3521746af55a72bb92cddd0536674963008`,
E32 `66c3963ae08189402d94897403a4beb4b432f2da8e65f976754878c462c1809c`, and
E31 `d31ab76022ba22dcc2477e6e8833aee4865abfa4901f29509a4b6d2fcf00edca`.

## Caps and authorization contents

Merge these four exact entries from the packet's `jobset_caps.json` into the
live `/var/tmp/fitv2/jobset_caps.json`, preserving unrelated entries:

```json
{
  "fitv40-control-20261007": {"memory":"6g","hosts":{},"build_commit":"f0584d00fab0bc38bce236ca386e34f9e8b63308","reason":"V40 local preparation; one CPU per cell, 6 GiB/no swap; coordinator authorization required"},
  "fitv40-e29-20261007": {"memory":"6g","hosts":{},"build_commit":"f0584d00fab0bc38bce236ca386e34f9e8b63308","reason":"V40 local preparation; one CPU per cell, 6 GiB/no swap; coordinator authorization required"},
  "fitv40-e32-20261007": {"memory":"6g","hosts":{},"build_commit":"f0584d00fab0bc38bce236ca386e34f9e8b63308","reason":"V40 local preparation; one CPU per cell, 6 GiB/no swap; coordinator authorization required"},
  "fitv40-e31-20261007": {"memory":"6g","hosts":{},"build_commit":"f0584d00fab0bc38bce236ca386e34f9e8b63308","reason":"V40 local preparation; one CPU per cell, 6 GiB/no swap; coordinator authorization required"}
}
```

Each launcher requires `<bundle>/LAUNCH_AUTHORIZATION-<jobset>.json`. Its content
must equal the corresponding frozen `AUTHORIZATION_TEMPLATE-<jobset>.json`
with only these approval fields supplied:

```json
{
  "schema": "v40-coordinator-launch-v1",
  "coordinator_message": "THE COORDINATOR'S EXPLICIT APPROVAL OF THIS EXACT JOBSET AND PACKET",
  "source_landed": true,
  "pins_pushed": true,
  "reviewed": true,
  "E30_completed": true,
  "control_choice_frozen": true,
  "identities": "THE EXACT OBJECT FROM THE MATCHING AUTHORIZATION_REQUIRED FILE"
}
```

The last field is a full JSON object, not the explanatory string above: preserve
all hashes, file pins, program, data, image/config ID, worker build, frozen source
and zenmetrics identities. Full concrete objects for all four jobsets are in
`/home/lilith/tmp/v40-land/AUTHORIZATION_CONTENTS_REFERENCE.json`, outside the
packet and under a filename no launcher accepts. That reference has a message
placeholder; it is not authorization. Existing packet templates remain false.
The gate also requires all 40 E30 cells, live caps exactly equal to the reviewed
entries, and, for every arm, all 40 fresh V40 controls plus exact frozen hashes.

## Control-first launch and postfit

The frozen packet's `POSTFIT_COMMANDS.md` supplies the same commands. Listed here
without executing any of them:

```bash
set -euo pipefail
bundle=/mnt/v/output/zensim/v40r4-2026-10-08
python3 "$bundle/launch.py" --bundle "$bundle" --jobset fitv40-control-20261007
TMPDIR=$HOME/tmp/v40 flock ~/tmp/zensim-paper/rev4/heavy.lock \
  ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
  bash "$bundle/postfit.sh" "$bundle" control
```

Only after the control chain freezes all 40 fresh controls in
`V40_CONTROL_PINS.json` and `E29_CONTROL_PINS.json`, with each arm's separate
coordinator authorization present:

```bash
set -euo pipefail
bundle=/mnt/v/output/zensim/v40r4-2026-10-08
for study in e29 e32 e31; do
  python3 "$bundle/launch.py" --bundle "$bundle" --jobset "fitv40-$study-20261007"
  TMPDIR=$HOME/tmp/v40 flock ~/tmp/zensim-paper/rev4/heavy.lock \
    ~/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 -- \
    bash "$bundle/postfit.sh" "$bundle" "$study"
done
```

Use a shell with `set -euo pipefail`; stop on any failed launch or postfit. The
chains serialize incremental trusted harvest and SDR scoring, retain heartbeat
logs, refuse incomplete/empty results, and verify fresh artifacts before their
success markers. Harvest installs only full-budget cells into
`/var/tmp/rev4-featpot/v40-<study>-results/cells`. No optional protected population
is part of these commands. HDR/external/UPIQ panels keep their separate exposure
authorization; this preparation supplies none.
