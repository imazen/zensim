# STEER launch runtime repairs

The approved registration, 172 fit identities, literal fit argv, data and
scientific program remain frozen. The image remains
`ghcr.io/imazen/zenfleet-worker@sha256:36e86e70334698f06971d01b67383a0185430bbc8c0b8f6ed068081dbc21c0fa`.
These launch repairs change transport and worker admission only. No scientific
assessment has been run, and no gate or verdict rule changes.

The first registered cell passed its archive-bound/native verifier, but its
install rename crossed separate Docker bind mounts and failed with EXDEV.
Commit `254d5939` mounts the common parent once. The same preserved blob then
verified and installed in the same image, without another fit. All nine
installed first-cell files passed NAS byte hashes before expansion.

After expansion, completed blobs appeared before their ledger rows. The actual
pinned worker groups four fit cells per chunk at the original
`ZEN_CHUNK_WALL_SEC=7663` setting. It flushes ledger rows between chunks. It consults pause only at the start
of a worker pass; changing the control object does not stop claims during an
already-running pass. Its SIGTERM handler exits immediately and does not flush
an unfinished chunk. That parameter is a packing target; the scientific program
separately enforces the frozen 7,663-second cell cap.

The executable oracle is `just steer-worker-scheduling-probe <immutable-image>
<new-output-directory>`. Forty unique invented jobs, a capability-reporting
stub executor, no trainer and no network reproduce ten four-row sidecars at
target 7663 and forty one-row sidecars at target 1. Both runs must execute all
40 distinct jobs successfully. Existing output directories refuse reuse.

The prepared scheduling repair uses target 1 for one-cell chunks. Each new
worker also receives the existing absolute claim guard set to its host's UTC
epoch plus 30 seconds at enrollment. This is a rolling worker allocation,
not a host cutoff: permanent hosts continue through clean re-enrollment.
The existing limits remain one trainer thread, 6 GiB RAM with zero swap,
7,663 seconds per cell, 8,000 seconds per worker pass and 150 minutes per
worker allocation. The budget is 30 + 7,663 + 300 = 7,993 seconds, retaining
the 300-second setup/persistence reserve within the existing pass timeout.
A conditional host also retains its earlier loan deadline, taking the minimum
of that deadline and the rolling worker claim deadline.

Do not mix chunk schemes. Set pause control to block later passes, then wait
for each active original chunk to publish its durable ledger. Stop only that
STEER container after publication. If it has already started another fit,
that incomplete fit gets a full retry; preserve its partial artifacts. The
original completed chunk must verify, install and back up before transition.
The old worker may exit nonzero from this intentional stop; do not describe
it as a clean drain. Then admit one remaining registered
cell on the repaired scheduling. Verify its one-cell ledger, clean drain,
pinned-image installation and NAS copy before reenrolling the other hosts.
No extra fit or changed fit identity is used for this admission.

On one admitted worker, Docker reports swap-limit support absent and
ignores its requested combined limit. While holding before claims, explicitly
set that container's kernel `memory.swap.max` to zero and verify
`memory.max=6442450944`, `memory.swap.max=0` and `memory.swap.current=0`.
Its disk-backed scratch/cache supplies the owner-approved disk floor. Preserve
the initial failed envelope receipt and the final kernel-limit receipt.

Live application state, raw attempt logs, the scheduling amendment and receipts
remain under the [frozen packet's artifact root](../benchmarks/steer_freeze_packet_2026-10-16.pointer.md).
The launch status distinguishes raw completed blobs, published ledger rows and
verified installed cells. In-flight snapshots are preservation, not optimizer
resume or installed results. Keep new claims paused on any unresolved defect.

The canonical allocation owner is `scripts/rev4_featpot/steer_worker_lifecycle.py`.
It refuses enrollment before the original-chunk transition and the repaired
first-cell native/NAS admission proofs. It calls the packet-hashed LAN launcher
and leaves job selection, leases and reconciliation to the pinned worker.
`just steer-worker-watch <packet> <packet-sha>` re-enrolls the three active hosts
in the frozen jobset order, using canonical harvest metadata only. Every new
allocation rechecks capacity and its paused envelope. A failed allocation or
harvest stops the supervisor; completed evidence is retained.

To hold enrollment, use `just steer-worker-pause <packet> <packet-sha>`.
It creates `launch/PAUSE_WORKERS` beside the packet before waiting on the
shared enrollment lock, then publishes native pause for all eight jobsets.
Waiting for the enrollment critical section prevents a late release from
overwriting that pause. The owner checks that flag before allocation
and again before releasing its paused envelope. The rolling native deadline
prevents an admitted worker from claiming another cell after its window closes.
An observed coordinator change to the admission control object prevents
release. In contrast,
the original workers lack that rolling guard and must use the publication-bound
stop described above.
