# Rev4 feature-bank definition revision preregistration (2026-09-23 UTC)

This freezes the calibration rule before reading the diagnostic distributions.
No human labels, scores, or held-out pixels are inputs. The old candidate
definitions remain the control; the new definitions retain C1/C3 IDs and width.

## Inputs and roles

The existing 144-pair implementation gate supplies pixel paths only:

| TRAIN input | Pairs | SHA-256 |
|---|---:|---|
| `/var/tmp/featbank-impl/evidence/gate/cid22-64-pairs.tsv` | 64 | `36ef7cee8ec03de944c440ecf1d502e0ff540b2c922bcb6578de95031d33aef9` |
| `/var/tmp/featbank-impl/evidence/gate/safesyn-64-pairs.tsv` | 64 | `396ca563a146c00da1e450b3e424967d8b17432f66ad4702d47413be4b505236` |
| `/var/tmp/featbank-impl/evidence/gate/kadid-train16-pairs.tsv` | 16 | `4d780829d691a2963a19474482a37428b9b20b370a6572f3f616e072b3a1befa` |

The definition selection uses only the first two TSV path columns. The C3
criterion uses SafeSyn and CID22 TRAIN separately. KADID is included for the
C1 dead-bin criterion but does not choose a C3 edge. Decoding, formula revision,
tier and threading match the existing native serial `--full-rev4` gate.

## Deterministic statistics and decision

- C3: use the diagnostic per-cell 64-bin log counts for each of four maps,
  separately by corpus. The 32-bin registered histogram retains 31 fixed
  log-spaced interior edges. Choose the smallest simple top decade endpoint
  among `1`, `2`, `4` for which at most 1% of eligible nonempty cells per map
  and corpus have a p99 falling in the top bin. If the coarse diagnostic cannot
  certify an endpoint, extract a finer TRAIN diagnostic and document it. Pin
  each resulting edge as an IEEE-754 bit literal; no runtime `powf`.
- C1: use the pooled TRAIN on-grid `|ẽ|` diagnostic. Choose six increasing,
  round decimal hat centres from the occupied distribution, preserving its
  low end and moving the dead upper centre into supported mass. The release
  criterion is that every hat has a nonzero emitted value in at least one cell
  of the 144-pair corpus. Signed cancellation is checked on the emitted slots,
  not inferred from membership alone.
- Re-run the 144-pair × 2-thread × 3-tier f0–f985 identity matrix and the
  C1–C4 family tests after changing definitions. Any nonzero old-slot bit
  difference fails. No bootstrap or CI applies to this deterministic census;
  no random seed or human outcome is used.

These rules select implementation ranges, not predictive quality. The resulting
registry revision is recorded in the design note before any extraction that
could consume the C1/C3 values.
