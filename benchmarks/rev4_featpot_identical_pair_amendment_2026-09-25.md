# Rev4 POTENTIAL amendment — identical-pair convention: fabricated zeros everywhere (2026-09-25)

**POTENTIAL — ceiling, not a model score.** Supplements the preregistration chain through the restore-cuts amendment `72b35b43439b` / pin `c56290c10f13`. Written at 2026-09-25T16:27:39Z by Claude Sonnet lane `featbank-potential`. **No C-arm and no restore-arm result has been read for reporting under this amendment; see "Disclosure" for the eight cells fitted before it.**

## Authority and finding

Coordinator decision (2026-09-25) on `REVIEW_PARTB.md` section 2. The bank's old-family (f0-f943) zeros on the 88 `pixels_identical` keys come from the product identity short-circuit (`BakeScorer::compute` returns score 100 and a zero vector); the research extractor computes the features instead. R0 + C1-C4, csfw/DVIFM and gmsbank are consistently all-zero on those rows. Restore-cuts `mapdev` is **not**: it is nonzero on identical rows (48 cells each; ReferenceOnly `hfsq_src_dev`/`hfabs_src_dev` and Undeclared `*_dst_dev`, max 0.12), so A1 and A1m would mix the two conventions with R0 on those rows. z1max, gmsnative and dvifmgate are 0 on identity.

## Decision (option b)

**One convention for the whole run: fabricated zeros on `pixels_identical` keys** — the runtime convention, and what R0, the C-arms, and the running P0/P2 fleet cells already use.

- The adapter zero-masks **every candidate/restore column** of an arm on rows whose `pair_key` is `pixels_identical` (today only mapdev differs, but the mask is applied to every added family so the convention is uniform). The flag is read from the admitted table's `pixels_identical` column (feature-side metadata, not a label), verified consistent per `pair_key`. **No file is changed**; the mask lives in `restore_data.py` and `candidate_data.py`.
- **Permuted controls:** the permutation runs jointly within each reference over the **non-identical** `pair_key`s only; identical rows stay zero in the control exactly as in the real arm, so the fabricated zeros never move and real and control share one convention.
- The bank R0 columns are untouched (already zero on those rows).
- The adapter records, label-free and per arm/set, how many added-column cells on identical rows were nonzero before the mask, in the receipt metadata.

## Recorded, not applied in this run

Option (a) — excluding `pixels_identical` keys from every fit, selection and evaluation table — is the **recommendation for future Rev4 training tables**: the runtime never scores identical pairs with a model (both `Zensim::compute` and `BakeScorer::compute` return 100), so they carry no serving signal, and each KADID identical key carries 4 differently labelled stimuli with identical pixels. It is **not applied here** because it would change the data of the in-flight P0/P2 fleet cells and break their comparability. Affected stimuli (from the review): kadid_train 156/5,000, kadid_select 100/3,125, konfig_train 3, konfig_val 4 (safesyn 1 and kadid_terminal 64 are outside this run's sets).

## Disclosure (eight pre-amendment cells)

Before this decision reached the lane, the local deterministic runner had fitted BVLS D1 cells of arm `a1` (mapdev, unmasked) on all eight D1 sets (result sha256 prefixes: kadid_train f50b70c6, konfig_val f9608df1, tid2013 5dc373e2, konfig_train bdae7204, cid22_a25 8488e4be, aic3 5fb4d911, kadid_select ac3f743e, konjnd_bpg_train 2c334685) and started `a1_perm` — i.e. labels were read for restore-arm fits under the mixed convention. I saw the per-fold and nested SROCC lines of one of them (kadid_train a1 BVLS) in a log tail; no other value was viewed and nothing was compared or recorded. The runner was stopped at 16:26Z and all outputs were **moved, not deleted,** to `/var/tmp/rev4-featpot/candidates_VOID_mixed_identical_convention/`. They are **VOID**: never reported, never aggregated, never compared. Every restore-arm and C-arm cell is rerun from scratch after the adapter applies this convention.
