# Registered factorial statistics

The internal `panel e34 registered-evaluate` command accepts the
`registered-factorial-paired-v1` input and emits twelve comparisons, ordered
P, H, PH versus control and H/P, PH/P, PH/H for A, then C. It uses ten paired
seed panels, margin .0065 for eligibility, zero for improvements, 10,000
attempted draws and the maximum of directional-bootstrap and Student p-values.
The base seed is 20261016; the existing per-test and guard offsets are retained.
An incomplete slot stays in Holm with decision p=1. Any incomplete slot blocks
a scientific decision. Numeric completion does not establish gate eligibility.

The numerical implementation calls the existing E34 row, statistic, interval,
W2, seed and pair-class owners. References are globally dataset-qualified;
the pooled two-way reference/seed draw is unstratified. Expanded copy labels
remain distinct. Missing datasets, fewer than three reference copies in any
required dataset, or undefined reductions produce nonfinite draws, without
redrawing. Finite draws alone enter the type-7 interval and sample SE.
Zero/one finite draws and zero primary SE preserve the owner's incomplete
outcome. A finite zero-SE W2 retains the strict `delta > -2*SE` guard.

Floating reduction is also inherited: a constant statistic shifted by .0065
can have a tiny positive computed SE. There is no epsilon repair or identical
arm exception. An exact-zero primary SE remains incomplete. All returned
values are checked against the pinned executable, rather than assuming that
mathematically constant data always produces floating zero variance.

`just registered-statistics-unit` checks actual finite-draw reduction and
copy expansion with invented data. Explicit qualification additionally runs:

```
just registered-statistics-build
just registered-statistics-parity <new-private-receipt-path>
```

Run these commands through the shared resource wrapper. The build's explicit
`E34_OWNER_ORACLE=1` mode extracts the unmodified E34 source at
`81a545a3ff3457c9fabae93bc9d262b19b50952e` from repository history and compiles
it as a separate executable. The normal build does not require jj or extract
historical source. The helper dependency remains pinned at
`48665c8fb1bb3de5c99d792ae4ed4f3f1e854629`.

The synthetic parity command compares float bits and returned null/state
semantics for pooled, dataset, W2 and reference-only intervals, seed summaries,
both decision probabilities and incomplete cases. Unequal dataset sizes,
colliding local reference IDs and multiple views exercise the pooled design.
Other fixtures cover identical panels, absent models/seeds, undefined W2,
all-dropped draws and hard errors. Separate production-inventory fixtures
exercise all twelve slots and incomplete-packet refusal. The receipt binds
source, executable, harness, input and output hashes. Its parity-only command
uses E34's inventory and seeds and cannot issue a registered verdict.

These commands use invented observations only. They neither open a corpus nor
launch jobs. Actual input archives, program closures, gates and measured
preflight remain separately bound by the coordinator.

`registered_verdict.py` consumes this twelve-test packet and measured packets
from the inherited gate owners. All eight model packets, every gate, four
dataset guards, and paired steering counts at seeds 0/1/2 are required. It
retains P → H → PH replacement, the strict practical improvement bar, and the
report-only mean guard. Its fixed A-before-C seed-0 acceptance candidate is a
binding for a later reader; the adapter never authorizes an acceptance read.
`just registered-verdict-check` exercises replacement comparisons, all gates,
paired steering, incomplete slots and no-eligible refusal with invented data.
