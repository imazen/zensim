# KADID TERMINAL — single confirmatory read of the final qualified model (registered 2026-10-07, owner decision D2)

Owner, verbatim: "D2 (KADID TERMINAL): B — register now, read once on the final qualified model". Ledger: `docs/DATA_SPLITS.md`,
exposure entry 2026-10-07. No KADID TERMINAL label has been read for this design line; none may be read before the conditions below.

* **Population:** the KADID-10k reference-level TERMINAL split (2,000 pairs), exactly as frozen in the bank (`bank/kadid_terminal`);
  rows, keys and pixel hashes pinned at read time from the existing receipts.
* **When:** once, after the qualified production model exists (strict-admission training per D1) and has passed its release gates
  (`benchmarks/shippath_gate_map_2026-10-05.md`). The exact model bytes (sha256) and the scorer binary are committed in a pre-read pin
  before any label file is opened.
* **Metrics (all reported):** signed SROCC, KROCC and PLCC; within-reference SROCC; per-distortion-type SROCC; scatter geometry from
  `zenstats::scatter`.
* **Comparators:** the shipped profile B (`codec_target()`) and the Rev4 research by_v2fy bake, both scored on the same pairs.
* **Confirmation rule:** the qualified model's signed SROCC is not worse than profile B's by more than 2 SE (paired bootstrap over
  references, 10,000 resamples, seed fixed in the pin), and not worse than the research by_v2fy's by more than 0.005. Failing either
  is recorded as a failed confirmation; the set is then spent and the result stands.
* **Exposure:** the read is logged in the DATA_SPLITS exposure ledger with the pin's sha; KADID TERMINAL is spent for this design
  line afterwards.
