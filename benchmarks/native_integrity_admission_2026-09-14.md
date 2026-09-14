# Native integrity training admission — September 14, 2026

Registered against `03154f1b` before implementation. Native input audit-v2
cannot be accepted by changing a schema whitelist: equal sample bytes with
different primaries are different scoring inputs, and historical RGB8 hashes
are not native-depth identities. The existing strict trainer and integrity
assessment remain the owners; no parallel scorer or model-fitting path.

Add exact presented `scoring_identity` metadata from the Rust ImageSource:
format, primaries, alpha, gamut mapping, dimensions and byte order. Python
validates these explicit fields; it does not infer them from Debug descriptions
of decoded codec metadata or rerun color conversion.

An explicit `integrity-head-train-v3` manifest admits Rev3 D228 with
`sdr-native-clip-v1`, a matching `integrity-train-diagnostic-v2` admission,
reviewed native pixel/interpretation bindings and complete structural-feature
audit coverage. Bind roles, audits, model/tool identities and pair lists before
feature payloads or fitting. Never mix native and legacy rows. Legacy recipes
remain legacy and must refuse native receipts without the new admission.
Preserve fit/calibration/development separation and all existing thresholds.
The final served audit must use the same native contract.

Negative controls cover equal codes/different primaries, metadata changes,
mixed eras, incomplete feature checks, mismatched roles and source-family
leakage. Replay the existing 214 TRAIN input-audit cases with fresh receipts;
they remain diagnostic and are not relabelled clean negatives or quality
training targets. No EVAL, TEST/TERMINAL, HDR qualification or model promotion.

Implementation also binds each canonical row's 372 f32 little-endian bytes to
its Rust audit hash, including the candidate/producer formula revisions.
The native recipe explicitly requires Rev3's nested-square-root calculation;
its final scoring command uses header-driven `pairs-tsv` and the same native
input contract. Legacy recipes retain their original root/decoder settings.
The exact public alpha flag is recorded: `Unknown` is explicitly treated as
straight alpha by ImageSource and is not an unknown color interpretation.

## Results and corrections

The trainer's actual final-audit command completes on all214 admitted TRAIN
pairs, using the existing frozen Rev3 base/head as instruments. All214 f32
feature-row hashes and native interpretation bindings verify. The feature CSV
is byte-identical to the preceding complete-feature audit; every previous
audit field, score, head decision and spatial result is unchanged. New fields
bind interpretation, formula/root settings and the canonical feature bytes.

Two real failures were caught without weakening the checks:

- The trainer's inherited `libm` root override differed from the Rev3 default
  by up to1.11e-16 in f64 features on this packet, with no f32 differences.
  The new native recipe explicitly uses `sqrt`; legacy recipes retain their
  prior settings. The unsuccessful comparison and outputs are preserved.
- Default pandas CSV parsing changed47 f32 entries across29 rows, including36
  D228 entries. Round-trip parsing restores all214 canonical row hashes.
  The decimal `0.008768686559051275` now exercises this boundary in the
  trainer test. Updating a CSV's manifest hash cannot bypass the row binding.

The integrity assessment admits33 exact identities as valid and retains181
nonidentity pairs as unresolved. One unresolved pair activates the head (q85);
none of the identities does. These are not corruption recall/false-positive
estimates on adjudicated encodes. The strict trainer rejects this diagnostic
packet for `unreviewed binary disposition` before opening feature payloads;
no fit output is created. P3/q85 labels are not repaired by decoder metadata.

Nine synthetic admission tests pass, including native fitting/calibration masks,
source-role refusals, metadata changes, hidden CSV changes and mixed-era reads.
The23 Rust native input/codec tests pass. The actual trainer scoring command,
public-score replay, root/bench clippy, script lint and gauntlet gates complete.
No new model is fitted or promoted; no EVAL or TEST/TERMINAL data is accessed.

Next resolve P3 encoder input conversion and q85 through their original codec
owners, restore mobile corruption coverage, and create a matching reviewed
TRAIN packet for the new native recipe. Historical RGB8 manifests and models
remain separate. HDR still requires its own input/display contract.
