# Coordinator decision: shared fresh matched control for E29 / E31 / E32 (2026-10-07, before any arm fit)

Measured: E30's pinned trainer (sha 9954507d…) refuses both the E31 UPIQ table (`malformed feature_set_id`) and the
E32 palette identity/IDs before fitting (E31_READY.md, E32_READY.md). E29's new legs likewise need trainer changes. So no
arm can run on E30's exact program, and the registrations' exact-parity reuse of E30's nA3 cells cannot be established.

Each registration's registered fallback applies: "the coordinator chooses and freezes one complete fresh matched control
before any fit; no outcome-based choice". Decision, made now with no arm result in existence:

* One extended fit program ("v40") carries the E29, E31 and E32 trainer/admission extensions together.
* **One fresh matched control**: the E30 nA3 recipe exactly (by_v2fy, head N, Rev5, D1 four sources, kadid / tid2013 /
  konfig / cid22_a25 × seeds 0–9, 120 × 50,000, final epoch 119) run under v40 — 40 cells — serves as the control for all
  three experiments. Each experiment's statistics stay its own (no pooling across experiments).
* Before the control runs, the extension must be shown not to change the baseline path: a v40 control cell is expected to
  be bit-identical to the matching E30 nA3 cell (same seeds, one Rayon thread, v3 tier). A mismatch is recorded and the fresh
  control still runs (it is the registered comparison either way); it is never replaced by E30's cells after the fact.
* Arms (E29 hb4 + hc4, E31 uh4, E32 palette) and the control are declared in one package, reviewed before launch.
