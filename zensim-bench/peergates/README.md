# Peer engineering packet

This unpublished measurement tool calls pinned butteraugli and SSIMULACRA2
on the existing, label-free E34 pixels. It verifies original and decoded
pixel hashes, scores each independent block repair, retains native maps,
and reuses the unchanged coherence statistics owner. It does not fit models.

SSIMULACRA2 uses the native fused `Ssimulacra2Maps::diffmap` at the
zenmetrics registered pin `54fabca9`; its scalar/map score bits are checked
on every pair. Butteraugli uses registry 0.9.3 and its `pnorm_3` scalar.
The older SSIMULACRA2 0.8.2 timing peer has no spatial-map surface.

Use the root `peergates-*` just recipes. `peergates-inputs` projects only
explicit engineering metadata columns. The caller must supply the shared
heavy lock and `run-heavy` for building and scoring. Output directories must
be fresh; a missing input, changed hash, mismatched block or non-finite
score fails. Maps are little-endian f32 compressed with zstd; result rows
carry their dimensions and order.

The frozen plan and evidence pins accompany the dated benchmark report.
Throughput logs are not quiet-gated timing evidence. The separate native
codec intervention harness needs an external-map interface before these
maps can enter its bitrate gate.
