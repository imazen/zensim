# SHIPPATH6 — companion qualification, 2026-10-05

Implemented independently of the pending human-role decision on parent
0eb019058148532e2f2d540c3adbe8b721893088, in the existing shippath jj workspace.
Implementation commit: 5483be4c87ec0ebfee0c64bf5926b0ced241ef2e. Local commits only.
Preregistration: `benchmarks/shippath6_REGISTRATION.md`.

The existing Python exporter opts into ZCTH v4 only with a verified TRAIN
admission. Defaults emit the unchanged v1/v2/v3 formats. V4 extends the header
from120 to152 bytes; SHA-256 covers bytes0..120 plus bytes152..EOF. This binds
all numeric sections, schema/revision/IDs and admission together. Rust checks
this digest, contiguous complete sections and header-matching admission before
exposing the new feature-gated/doc-hidden getter. Existing scoring math is
unchanged. The digest is an integrity binding, not a signature or qualification.
The sole API snapshot delta is that getter and generated counts; supported
surface/defaults remain unchanged.

Owner locations:

- `scripts/v_next/train_corruption_head.py:213`: extended existing `emit_zcth`.
- `scripts/v_next/train_corruption_head.py:345`: complete protected-path/TRAIN
  preflight, hashes, per-table sidecars/row roles and pinned producer/content
  bindings. Original source paths also preflight before hashing.
- `scripts/v_next/train_corruption_head.py:1060`: registered admitted-preparation
  route reuses `fit_canonical_hgb`; validates the pinned rows, exact source
  selections and cached f32 feature bytes, with fresh output outside input roots.
- `zensim/src/corruption_head.rs:440`: content-bound v4 loading; :781 getter.
- `zensim-validate/src/bin/bake_verdict.rs:4240`: actual composition owner verifies
  every pin, registered producer revision, TRAIN/scoring features and decoder,
  exact table declarations, TRAIN roles and bound row selections. Unadmitted
  legacy or changed/unknown/mismatched records cannot qualify.

Refit inputs and role authority:

Reused original registered R5INTEG3 Rev5/by_v2fy-420 TRAIN preparation, width720,
seed4101, fit5499/calibrate2706, 8205 unique source/pixel keys, equal origin
weights, fit-only weighted StandardScaler, clip8, HGB100/max31/no early stopping,
class balancing and separate weighted isotonic, P>0.9. No new source/row choice,
tuning, EVAL, human fit or new extraction. Two fresh admitted views copy original
TRAIN Parquets byte-identically; sidecars record the original producer receipt
and decoder executable, plus ordered fit/calibrate source-row selections.
All 16 explicit original/view/metadata/producer pins remain unchanged.

The append-only producer registration is
`basic+peaks+v2@w1825/r5integ_rev5#36c3f3af` (576 producer slots). Its formula is
Rev5. The decoder era is `legacy-rgb8/executable-sha256:` plus the original
01ed2897b7f4b07203d172a614a47dbd4c87b4d39dd42998f25cc131f29a7740 producer hash.
This is executable-bound historical decoding evidence, not an invented source
commit, independent codec commit, or native-color admission. Original corrected
serving handoff and current test extractor are separately pinned.

Historical `canonical_main` rehashes the protected-reference list behind its
content screen. This replay instead pins that already-completed receipt and
verifies only the original TRAIN payloads/preparation/row selectors. It does not
reopen or traverse protected payloads. Original receipt/catalog limitations are
retained; no new protected-screen or scientific selection claim.

Final artifact and proofs:

`/var/tmp/shippath6/refit/head.zcth` is 210994 bytes, v4, Rev5, 420 IDs, 100 trees,
6100 nodes; SHA-256
568380f1bd2fe6202ebcc5a194078d7885035f466abe93ada3a4b3ea8f4a26f2.
Original v3 control is
8e6ed23eca6257b9d6911c4032b21ec0d16385b6879c03a1a587a93fe8944f74.
All numeric header bytes16..56 and all seven predictive sections are raw
byte-identical. Actual Rust parity on8205 rows has zero margin/probability error
and zero fire disagreements (7713 fires). Final fit-time exporter source is
retained and matches the head's trainer hash.

The actual extractor completed all8510 registered TRAIN pairs with zero failures;
pixel/cache/literal composition scores and consumed features agree. All seven
unchanged TRAIN calibration gates pass, and the entire calibration summary
matches the registered control: detection2534/2546, honest activation/lowering0/160,
real bugs148/155, composed belowq20 2536/2546. This is the existing TRAIN screen,
not an EVAL/product integrity result.

The actual `bake_verdict::composition_feature_sets` owner reports qualified
Table provenance for the companion member on both admitted TRAIN views. A
clearly scoped companion-only projection passes the actual `freeze_check`
Table provenance row. Full composition retains the original historical primary
and still fails that row: "Explicit historical replay in the scorer cannot
qualify a new model". Both freeze commands return1 because full qualification
is intentionally incomplete. No gate or primary was waived. Evidence names
`COMPANION_LEG_ONLY.json` and `FULL_COMPOSITION.json` make the scope explicit.

Runtime: exact private ComputeSet equality for all six registered Rev5 base
compositions; no extra fields/work enabled. Five registered prepared probes per
composition (30 rows) have exact feature/pixel/cache parity, correct integrity
accept/refuse decisions and no prepared-map queries on rejects. All12 registered
TRAIN-origin identities score100; both head/base cross-revision directions fail
with the typed revision error. No benchmark/performance claim.

Tests/checks passed: corruption35; revision36 (+3 ignored external fixtures);
verdict38 (+1 explicitly run external owner proof); freeze41; Python18 including
retained integrity admission tests. Negative controls cover every numeric and
admission section/digest, properly rehashed revision mismatch, changed TRAIN
bytes/declarations/roles/decoder/selection/pins, protected/relative paths before
payload hashing, unknown legacy admission and missing scoring tables. Synthetic
fixture metadata is not scientific admission. Exporter byte parity covers v1,
v2 and all v3 revisions1..5, with seven Rust synthetic parity cases and three
invalid-contract refusals. CI-exact `just clippy`, `just api-doc-check`,
`cargo fmt --all --check` and `just lint-scripts` pass.

All heavy jobs used the required --mem16G/--jobs8 wrapper and task TMPDIR;
numeric fits used one thread. The first nested extractor build encountered a
full /home filesystem. Only this checkout's build caches were relocated, with
bytes preserved and original target paths retained as symlinks to
/var/tmp/shippath6/target-root and target-bench; no source/data/evidence moved.
Earlier harness retries are retained in scratch; final successful commands,
return codes and hashes are indexed in `shippath6_checks_2026-10-05.json`.

Full evidence lives under /var/tmp/shippath6; review bundle under
~/tmp/zensim-paper/rev4/SHIPPATH6_assets. Bound paths are preserved: relocating
files alone does not relocate/rewrite the content-bound admission. Refit and
proof replay commands are in the bundle. No original frozen roots, previous
reports/assets, protected/holdout labels or pending role decision changed. No
fleet, push, scientific promotion or product qualification. The full human
recipe remains blocked on SHIPPATH-human-production-role. DONE is written last.
