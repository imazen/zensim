# Profile B: SDR/HDR backend routing — design decision + rationale

**Status:** shipped 2026-07-04 (commits `e3438a71`, `6c81f67e`, `18e0dbd9`).
This doc exists because the routing was a *user-directed* correction to a wrong
default, and a design that rests on one real-time catch is fragile until it's
written down. If a future session is tempted to "simplify" B by removing the
dispatch — read this first.

## The decision

`ZensimProfile::B` is a **single handle that dispatches to a different backend
by the declared input domain**:

- SDR entry paths (`compute` / `compute_extended_features` on an SDR source) →
  the **SDR linear weights** (`b_sdr_linear_cid80_winsor`, `ens-Pline-cid80`).
- HDR entry paths (`compute_pu_linear*`, i.e. nits / `LinearF32Rgba` with
  `is_hdr()`) → the **BHdr PU-linear weights** (`bhdr_linear_shaped_anchored2`).

`ZensimProfile::BHdr` is the **explicit, unrouted** HDR handle (forces the HDR
weights regardless of entry path). So B = "the current best, for whatever valid
content you hand me"; BHdr = "the HDR model, explicitly."

Dispatch keys on the **declared descriptor** (`is_hdr()` flag + source type),
never on pixel values.

## Why this is the right design — independent of who asked for it

zensim is a **user-facing quality dial**: a caller types a target score and the
codec stack binary-searches an encode that hits it. That contract only holds if
the metric *works on the caller's content*. A caller with HDR content who picks
"the current best zensim" (B) must get a valid HDR score — not an error, and
absolutely not a garbage score from SDR-tuned weights run on PU-linear features.
So the single "best" handle has to cover both domains. That is a property of the
*contract*, not a preference — it would be the right design no matter who
proposed it.

## What would have shipped WITHOUT the routing directive (the counterfactual)

The honest answer, from the git history: **B would have been SDR-only, and HDR
input would have hit the issue-#38 guard and ERRORED** (`6c81f67e`: "issue #38
guard now routes instead of erroring"). Before the guard it was worse — SDR
weights on PU-linear features is a silent invalid pairing (wrong-shape features
→ garbage score). So the default trajectory was *domain-incomplete B*: either a
hard error on HDR, or (earlier) silently wrong numbers.

That the correct behavior only exists because of a real-time catch is the
process risk this doc addresses. The fix that makes the design robust to a wrong
default is **writing the decision + its rejected alternatives down** — so the
next session extends it instead of re-deriving (or regressing) it.

## Rejected alternatives (and why)

| Alternative | Why rejected |
|---|---|
| **Error on HDR** (the original #38 guard) | Breaks the dial contract for HDR callers who pick "the best". A metric that refuses valid content isn't "the best" for that content. |
| **Value-sniffing** (inspect pixel magnitudes to guess SDR vs HDR) | Measured **threshold-seam risk**: 5–10 pt cross-model score scatter right at whatever luminance cutoff you pick, because two different backends meet at the seam. Dispatch must key on the *declared* domain, which is unambiguous, not on a guessed one. Explicitly rejected in `e3438a71`. |
| **SDR weights run on PU-linear features** (no separate HDR backend) | Invalid feature/weight pairing → garbage. Made **unrepresentable** via B by construction (`e3438a71`). |
| **Fully explicit only** (B=SDR, BHdr=HDR, caller must choose) | Pushes domain detection onto every caller; a wrong pick is an error or garbage. We keep BHdr as the explicit override, but B auto-routing is the safe default. |

## Invariants (pinned by tests)

- **`is_hdr()` alone routes; the container type does not.** `LinearF32Rgba` is a
  container, not an HDR signal — SDR linear-f32 flows the SDR pipeline unchanged
  (`18e0dbd9`, test `flag_not_format_decides_the_pipeline`: same bytes, flag
  flipped → different pipeline).
- **`B == BHdr` on nits** (same weights on HDR entry paths) — routing test in
  `e3438a71` / parity tests in `6c81f67e`.
- **Contradictory descriptors still error** (a mixed/ambiguous pair is not
  silently coerced) — `6c81f67e`.
- **Identity contract holds** through the routed path (`score(x,x)=100`).

## Reproduction

Both backends are reproducible from committed code:
- B (SDR): `scripts/reproduce_b.sh` — ensemble → anchor spline → winsorize,
  asserts byte-identity to the shipped bake (`b92b0b7a`).
- BHdr: `scripts/v_next/hdr_anchor_dense_refit.py` (shaped PU-linear, already
  winsorized by its shaping transforms).

Full lineage + pinned input shas: `benchmarks/provenance_best_results_2026-07-04.md`.
