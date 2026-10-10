"""The authored part of dialviz: what each property means and why it matters.

Only prose that no source states in extractable form lives here. Every rule,
bar, owner, number and state shown beside it is a *locator* that a reader
resolves against the committed sources at build time (`quotes.resolve`), so a
moved or reworded source fails the build instead of going stale.

Locator kinds:
  {"quote": path, "match": regex}                 paragraph/bullet starting at the first matching line
  {"row": path, "headers": [...], "key": regex}   one row of the first table with those headers
  {"code": path, "start": regex, "end": regex}    verbatim code excerpt (end line inclusive)
  {"symbol": path, "match": regex}                owner code reference (path:line)
State quotes may carry {"pass": regex, "fail": regex}: the chip shown beside the quote is
`fail` if the fail pattern matches the quoted text, else `pass` if the pass pattern matches,
else `info`. The full quoted text is always shown.
"""

GATE_MAP = "benchmarks/release_gate_map_2026-10-07.md"
SCORECARD = "docs/MODEL_SELECTION_SCORECARD.md"
E33MD = "benchmarks/e33_registration_2026-10-09.md"
STATUS = "benchmarks/rev5_research_status_2026-10-05.md"
GADDR = "zensim-validate/src/dial_addressability.rs"
GADDR_DOC = "benchmarks/dial_addressability_gate_2026-09-04.md"
BV = "zensim-validate/src/bin/bake_verdict.rs"
FC = "zensim-validate/src/bin/freeze_check.rs"

E33_GATES = ["Gate", "Rule", "Seed-0 production today"]

WANTED = [
    {
        "id": "identity", "title": "Identity scores exactly 100",
        "definition": "A byte-identical copy of the reference scores exactly 100, on every scoring path, and no distorted image "
                      "scores above it. On cached feature vectors, an identical pair lands inside the identity band [97.5, 100].",
        "why": "100 is the dial's anchor. Codecs and users read it as \"no loss\"; a copy that scores 96 makes every "
               "near-lossless target unreachable, and a distortion that beats a copy breaks ordering at the top.",
        "gates": ["negative-tails-identity", "rust-surface-final-identity"],
        "rules": [
            {"row": GATE_MAP, "headers": ["Gate / owner command", "Required inputs and pass rule"], "key": r"Negative tails / identity",
             "col": "Required inputs and pass rule"},
            {"row": E33MD, "headers": E33_GATES, "key": r"C5 identity", "col": "Rule"},
        ],
        "owners": [
            {"symbol": GADDR, "match": r"IDENTITY_IS_THE_ZERO_VECTOR", "label": "C5 identity band"},
            {"symbol": "zensim/src/metric.rs", "match": r"fn identical_result_at", "label": "identity short-circuit"},
        ],
        "state": [
            {"row": E33MD, "headers": E33_GATES, "key": r"C5 identity", "col": "Seed-0 production today", "fail": r"fails"},
            {"quote": STATUS, "match": r"^\*\*ADJUDICATE", "fail": r"below 97\.5"},
            {"quote": "benchmarks/nearid_2026-10-09.md", "match": r"^All 1,944 rows completed", "pass": r"served exactly 100"},
        ],
        "charts": ["nearid"],
    },
    {
        "id": "near-identity", "title": "Smooth approach to 100 (near-identity N1–N3)",
        "definition": "Imperceptible changes score just below 100, and scores fall smoothly as the change grows: a one-pixel "
                      "±1 change serves ≥ 99.0 (N1), every reference reaches ≥ 99.0 with some nonidentical rung (N2), and "
                      "nested distortion ladders do not increase (N3).",
        "why": "A target of 98 or 99 must be reachable by a near-lossless encode. A gap between 100 and the best nonidentical "
               "score makes the top of the dial unusable for exactly the settings photographers care about.",
        "gates": ["negative-tails-identity"],
        "rules": [
            {"row": E33MD, "headers": E33_GATES, "key": r"N1 near-identity", "col": "Rule"},
            {"row": E33MD, "headers": E33_GATES, "key": r"N2 no gap", "col": "Rule"},
            {"row": E33MD, "headers": E33_GATES, "key": r"N3 ladder order", "col": "Rule"},
            {"quote": E33MD, "match": r"^The 99\.0 bar in N1/N2 is a registered choice"},
        ],
        "owners": [
            {"symbol": "scripts/prodqual_label_free.py", "match": r"def nearid_summary", "label": "NEARID summary"},
            {"symbol": "scripts/prodqual_label_free.py", "match": r"def nearid_register", "label": "NEARID panel registration"},
        ],
        "state": [
            {"row": E33MD, "headers": E33_GATES, "key": r"N1 near-identity", "col": "Seed-0 production today", "fail": r"fails"},
            {"row": E33MD, "headers": E33_GATES, "key": r"N2 no gap", "col": "Seed-0 production today", "fail": r"fails"},
            {"row": E33MD, "headers": E33_GATES, "key": r"N3 ladder order", "col": "Seed-0 production today", "pass": r"122/144"},
        ],
        "charts": ["nearid"],
    },
    {
        "id": "monotone", "title": "Monotone ladders",
        "definition": "Along any codec's quality ladder for one image, a better setting never scores lower: strict order on "
                      "≥ 93% of adjacent pairs (C1 / G-DIAL G3), with every reversal reported.",
        "why": "Targeting is a search over a codec setting. A reversal makes the search oscillate or pick a worse setting "
               "that scores higher.",
        "gates": ["g-dial", "negative-tails-identity", "g-addr-five-codec-floors"],
        "rules": [
            {"row": GATE_MAP, "headers": ["Gate / owner command", "Required inputs and pass rule"], "key": r"^\*\*G-DIAL",
             "col": "Required inputs and pass rule"},
            {"quote": GADDR_DOC, "match": r"^\| C1 "},
        ],
        "owners": [{"symbol": GADDR, "match": r"\"C1\"", "label": "C1 row"},
                   {"symbol": BV, "match": r"G3: monotonicity = 1 − material-inversion rate", "label": "G-DIAL G1/G3 in bake_verdict"}],
        "state": [{"row": E33MD, "headers": E33_GATES, "key": r"N3 ladder order", "col": "Seed-0 production today",
                   "pass": r"122/144"}],
        "charts": ["gaddr_c"],
    },
    {
        "id": "no-ties", "title": "No ties (C2)",
        "definition": "Distinct encodes get distinct scores: at most 5% of adjacent grid pairs tie, on both the flat standard "
                      "grid and the ladder.",
        "why": "A tie is a flat spot on the dial: the target search cannot tell two settings apart and wastes bytes or passes.",
        "gates": ["negative-tails-identity", "g-dial"],
        "rules": [{"row": E33MD, "headers": E33_GATES, "key": r"C2 ties", "col": "Rule"}],
        "owners": [{"symbol": GADDR, "match": r"\"C2\"", "label": "C2 row"}],
        "state": [
            {"row": E33MD, "headers": E33_GATES, "key": r"C2 ties", "col": "Seed-0 production today", "pass": r"0\.0065"},
            {"quote": STATUS, "match": r"^- \*\*C2 tie check fix\*\*", "pass": r"passes C2"},
        ],
        "charts": ["gaddr_c"],
    },
    {
        "id": "addressability", "title": "Dial addressability (G-DIAL, G-ADDR C1–C6)",
        "definition": "The dial covers its range on a standard grid (p5 ≤ 25, p95 ≥ 85, monotone), and the G-ADDR CONTRACT "
                      "tier holds: monotone, few ties, a negative tail exists, identity in band, nothing above identity. "
                      "A1–A6 mentor value pins are report-only.",
        "why": "Per the owner's 2026-09-04 rule, \"any model that limits dial range cannot ship\": a dial that cannot "
               "reach 25 or 85 cannot serve the targets codecs request.",
        "gates": ["g-dial", "g-addr-five-codec-floors"],
        "rules": [
            {"quote": SCORECARD, "match": r"^\*\*G-DIAL asks"},
            {"row": GATE_MAP, "headers": ["Gate / owner command", "Required inputs and pass rule"], "key": r"^\*\*G-ADDR",
             "col": "Required inputs and pass rule"},
        ],
        "owners": [{"symbol": GADDR, "match": r"pub fn evaluate_full", "label": "G-ADDR evaluate_full"},
                   {"symbol": GADDR, "match": r"fn shippable", "label": "Verdict::shippable"}],
        "state": [{"quote": STATUS, "match": r"^\*\*Seed 0 does not qualify yet", "fail": r"does not qualify"}],
        "charts": ["gaddr_matrix", "gaddr_bars"],
    },
    {
        "id": "codec-floors", "title": "Codec floors and cross-codec consistency at PJND",
        "definition": "Each codec's lowest configurable settings stay distinguishable (A7r floor representability, judged "
                      "against the mentor on identical cells for avif-rav1e, avif-svt, jpeg, jxl and webp), and equal "
                      "perceived quality maps to equal scores across codecs, including at the just-noticeable difference.",
        "why": "A user picks one number for every codec. If the floor of one codec is unreachable, or the same visible "
               "quality scores differently per codec, the number means different things per format.",
        "gates": ["g-addr-five-codec-floors"],
        "rules": [
            {"quote": GADDR_DOC, "match": r"A7r"},
            {"row": "docs/CODEC_TARGET_GOALS.md", "headers": ["Measure", "Threshold"], "key": r"KonJND PJND pairs", "col": "Threshold",
             "note": "Historical G2 anchor (document carries a 2026-07-18 supersession banner); no current release row restates it."},
            {"quote": "docs/CODEC_TARGET_GOALS.md", "match": r"^## G4"},
        ],
        "owners": [{"symbol": GADDR, "match": r"A7r", "label": "A7r rows"},
                   {"symbol": FC, "match": r"Codec floor: \{codec\}", "label": "freeze_check codec floors"}],
        "state": [{"row": GATE_MAP, "headers": ["Gate / owner command", "Status for production"], "key": r"^\*\*G-ADDR",
                   "col": "Status for production"}],
        "rule_pending_note": "Cross-codec consistency at PJND has no row in the current release contract; the historical "
                             "goals document is superseded. Shown as rule pending.",
        "charts": ["a7r"],
    },
    {
        "id": "negative-tails", "title": "Negative tails (C3, C4)",
        "definition": "Severe distortions score below zero: some all-negative-truth probe cell scores < 0 (C3) and the deepest "
                      "probe is < 0 (C4). The dial is not clamped at 0.",
        "why": "A floor at 0 hides how bad a catastrophic encode is and ties every severe failure; codecs exploring low "
               "qualities need the ordering to continue.",
        "gates": ["negative-tails-identity"],
        "rules": [{"row": GATE_MAP, "headers": ["Gate / owner command", "Required inputs and pass rule"],
                   "key": r"Negative tails / identity", "col": "Required inputs and pass rule"}],
        "owners": [{"symbol": GADDR, "match": r"\"C3\"", "label": "C3 row"}, {"symbol": GADDR, "match": r"\"C4\"", "label": "C4 row"}],
        "state": [{"quote": STATUS, "match": r"^\*\*Seed 0 does not qualify yet", "pass": r"negative tails"}],
        "charts": ["gaddr_c"],
    },
    {
        "id": "target-accuracy", "title": "Target accuracy (G-TARGET)",
        "definition": "Given a requested score on an attainable target, the codec loop lands close in 1, 2 or 3 shots: "
                      "one-shot median |error| ≤ 2 and p95 ≤ 8; two-shot ≤ 1 / ≤ 3; three-shot ≤ 0.5 / ≤ 1 / max ≤ 3, with "
                      "bounded undershoot.",
        "why": "This is the product: users ask for a number and expect it. Each extra encode pass costs latency.",
        "gates": ["g-target"],
        "rules": [{"row": SCORECARD, "headers": ["Requirement", "Release bar / measurement"], "key": r"One-shot targeting",
                   "col": "Release bar / measurement"},
                  {"row": SCORECARD, "headers": ["Requirement", "Release bar / measurement"], "key": r"Two-shot targeting",
                   "col": "Release bar / measurement"},
                  {"row": SCORECARD, "headers": ["Requirement", "Release bar / measurement"], "key": r"Three-shot targeting",
                   "col": "Release bar / measurement"},
                  {"row": SCORECARD, "headers": ["Requirement", "Release bar / measurement"], "key": r"Target coverage",
                   "col": "Release bar / measurement"}],
        "owners": [{"symbol": "zensim-target/examples/demo_matrix.rs", "match": r"fn main", "label": "demo_matrix"}],
        "state": [{"row": GATE_MAP, "headers": ["Gate / owner command", "Status for production"], "key": r"^\*\*G-TARGET",
                   "col": "Status for production"}],
        "charts": [],
    },
    {
        "id": "steering", "title": "Steering maps (G-STEER M2 / M3f)",
        "definition": "The per-block diffmap points a codec at the blocks whose repair raises the score most: the ceiling "
                      "M2 ≥ 0.99 and the deployable map M3/M3f ≥ 0.70 on 135 registered cases, with exact neighbour replay.",
        "why": "Spatial steering is how a codec spends bytes where they are visible. A map that points elsewhere wastes "
               "bytes or games the score.",
        "gates": ["g-steer", "steercodec-chromaq"],
        "rules": [{"row": GATE_MAP, "headers": ["Gate / owner command", "Required inputs and pass rule"], "key": r"^\*\*G-STEER",
                   "col": "Required inputs and pass rule"},
                  {"row": E33MD, "headers": E33_GATES, "key": r"G-STEER", "col": "Rule"}],
        "owners": [{"symbol": "zensim/examples/diffmap_block_coherence.rs", "match": r"M2", "label": "diffmap_block_coherence"},
                   {"symbol": "scripts/m3a_sweep.sh", "match": r".", "label": "m3a_sweep.sh"}],
        "state": [{"row": E33MD, "headers": E33_GATES, "key": r"G-STEER", "col": "Seed-0 production today", "fail": r"128/135"},
                  {"quote": "benchmarks/steerfix_2026-10-09.md", "match": r"Full G-STEER remains", "fail": r"128/135"}],
        "charts": ["steerfix"],
    },
    {
        "id": "spatial-value", "title": "Spatial value in real codecs (G-RD)",
        "definition": "Map-guided encoding saves bytes at equal quality as judged by independent metrics: ≥ 0% geometric-mean "
                      "savings on every judge and ≥ 1% on at least one, against a strong scalar controller, per codec.",
        "why": "A map that only improves zensim's own score is grading its own homework; independent judges catch gaming.",
        "gates": ["g-rd-spatial-value"],
        "rules": [{"row": SCORECARD, "headers": ["Requirement", "Release bar / measurement"], "key": r"Spatial value",
                   "col": "Release bar / measurement"}],
        "owners": [{"symbol": "scripts/v_next/rd_probe_analyze_2026-07-18.py", "match": r"interventions", "label": "rd_probe_analyze"}],
        "state": [{"quote": SCORECARD, "match": r"^Current disposition:", "fail": r"spatial RD\s+FAIL"}],
        "charts": [],
    },
    {
        "id": "hdr", "title": "HDR behaviour",
        "definition": "Native PQ/HLG/cICP inputs are interpreted in absolute nits on common primaries, scored by a model "
                      "trained or aligned for HDR, and judged by HDR judges on the same release rows as SDR.",
        "why": "An SDR fit applied to HDR pixels says nothing about HDR quality; HDR claims need HDR evidence.",
        "gates": ["hdr-scope"],
        "rules": [{"row": GATE_MAP, "headers": ["Gate / owner command", "Required inputs and pass rule"], "key": r"^\*\*HDR scope",
                   "col": "Required inputs and pass rule"}],
        "owners": [{"symbol": "zensim/src/metric/bake.rs", "match": r"fn compute_hdr", "label": "BakeScorer::compute_hdr"}],
        "state": [{"row": GATE_MAP, "headers": ["Gate / owner command", "Status for production"], "key": r"^\*\*HDR scope",
                   "col": "Status for production"}],
        "charts": ["hdr_e27", "v40_hdr"],
    },
    {
        "id": "runtime", "title": "Runtime and memory",
        "definition": "Complete scoring is fast and bounded: uncached p95 ≤ 50 ms at 1024² and ≤ 200 ms at 2048² on one pinned "
                      "worker, cached score+map ≤ 3× uncached, peak incremental RSS ≤ 128 B/pixel + 64 MiB per worker.",
        "why": "zensim runs inside encode loops several times per image; its cost multiplies.",
        "gates": ["runtime-memory-work-census"],
        "rules": [{"row": SCORECARD, "headers": ["Requirement", "Release bar / measurement"], "key": r"Scalar performance",
                   "col": "Release bar / measurement"},
                  {"row": SCORECARD, "headers": ["Requirement", "Release bar / measurement"], "key": r"Spatial and memory cost",
                   "col": "Release bar / measurement"}],
        "owners": [{"symbol": "zensim/benches/extract_paths_bench.rs", "match": r".", "label": "extract_paths_bench"}],
        "state": [{"quote": "benchmarks/rev5_speedq3_2026-10-08.md", "match": r"^Rev5 is not established", "fail": r"not established"},
                  {"quote": "CLAUDE.md", "match": r"^At 1024²/v4x/1T, production/A/main medians"}],
        "charts": ["speedq3"],
    },
    {
        "id": "integrity", "title": "Integrity head (ZCTH)",
        "definition": "A companion classifier flags corrupt decodes (catastrophic bugs, swapped channels) and lowers their "
                      "score, without ever activating on honest native codec output.",
        "why": "A perceptual metric can rate a broken decode as merely low quality; codecs need a hard signal that the "
               "output is wrong, and no false alarms on valid low-quality encodes.",
        "gates": ["integrity-zcth-v4"],
        "rules": [{"row": GATE_MAP, "headers": ["Gate / owner command", "Required inputs and pass rule"], "key": r"^\*\*Integrity",
                   "col": "Required inputs and pass rule"},
                  {"quote": SCORECARD, "match": r"^The later \[activation and severity contract\]"}],
        "owners": [{"symbol": "zensim/src/corruption_head.rs", "match": r"ZCTH", "label": "ZCTH format"},
                   {"symbol": "scripts/v_next/corruption_gate_eval.py", "match": r"def main", "label": "corruption_gate_eval"}],
        "state": [{"row": GATE_MAP, "headers": ["Gate / owner command", "Status for production"], "key": r"^\*\*Integrity",
                   "col": "Status for production"}],
        "charts": ["integrity"],
    },
    {
        "id": "ranking", "title": "Ranking against human data (G-RANK)",
        "definition": "On frozen, authorized human evaluation sets, the model orders stimuli like people do: per corpus signed "
                      "SROCC with KROCC, PLCC, PWRC, bands and within-reference means, at least SSIMULACRA2 on the SDR "
                      "aggregate and each content class, and no holdout collapse.",
        "why": "The dial must agree with people, not only with itself; rank is necessary but not sufficient.",
        "gates": ["g-rank-board-axes", "kadid-terminal-d2"],
        "rules": [{"row": SCORECARD, "headers": ["Requirement", "Release bar / measurement"], "key": r"Human ranking",
                   "col": "Release bar / measurement"}],
        "owners": [{"symbol": BV, "match": r"fn main", "label": "bake_verdict"},
                   {"symbol": "zensim-validate/src/panel.rs", "match": r"thin re-export shim", "label": "panel statistics (re-export of zenstats::panel)"}],
        "state": [{"row": GATE_MAP, "headers": ["Gate / owner command", "Status for production"], "key": r"^\*\*G-RANK",
                   "col": "Status for production"}],
        "charts": ["experiments_overview"],
    },
]

UNWANTED = [
    {
        "id": "identity-gaps", "title": "Identity gaps",
        "definition": "A band below 100 that no nonidentical input reaches, or identity that differs by scoring path "
                      "(feature path below the band while pixel path is exactly 100).",
        "catches": ["negative-tails-identity", "rust-surface-final-identity"],
        "bugs": r"identity|pixel-identical",
        "evidence": [{"row": E33MD, "headers": E33_GATES, "key": r"N2 no gap", "col": "Seed-0 production today", "fail": r"fails"},
                     {"quote": STATUS, "match": r"^\*\*ADJUDICATE", "fail": r"below 97\.5"}],
    },
    {
        "id": "floor-ties", "title": "Floor ties",
        "definition": "Distinct low-quality encodes collapse to the same score at a codec's floor, so the lowest settings "
                      "cannot be told apart or targeted.",
        "catches": ["g-addr-five-codec-floors", "g-steer"],
        "bugs": r"floor|tie",
        "evidence": [{"quote": "CLAUDE.md", "match": r"Rev5 steering discards signal beneath the output floor", "fail": r"OPEN"}],
    },
    {
        "id": "non-monotone", "title": "Non-monotone ladders",
        "definition": "A better codec setting scores lower than a worse one on the same image.",
        "catches": ["g-dial", "negative-tails-identity", "g-addr-five-codec-floors"],
        "bugs": r"monoton|ladder",
        "evidence": [{"quote": "benchmarks/nearid_2026-10-09.md", "match": r"^Seed-0 has reversals"}],
    },
    {
        "id": "saturation", "title": "Saturation and clamping",
        "definition": "Scores pile up at a ceiling or floor (an intrinsic top below 100, a clamp at 0), hiding differences "
                      "the dial should express.",
        "catches": ["g-dial", "negative-tails-identity", "g-rank-board-axes"],
        "bugs": r"saturat|clamp|ceiling",
        "evidence": [{"quote": "CLAUDE.md", "match": r"Full-population scatter diagnostics come from"}],
    },
    {
        "id": "unidentified-tables", "title": "Unidentified tables",
        "definition": "A data table with no recorded builder, provenance or orientation is read by an evaluation, so its "
                      "results cannot be interpreted.",
        "catches": ["table-provenance"],
        "bugs": r"unidentified|provenance",
        "evidence": [{"quote": "CLAUDE.md", "match": r"is an unidentified table; 63 board cells read it", "fail": r"OPEN"}],
    },
    {
        "id": "feature-drift", "title": "Feature drift",
        "definition": "Features change meaning under a consumer: a formula revision, SIMD-tier divergence or renamed feature "
                      "reaches a model trained on the old values, or tables from different eras are mixed.",
        "catches": ["table-provenance", "rev5-correctness", "input-serving-bake-format-api"],
        "bugs": r"bits|divergen|drift|revision|tier",
        "evidence": [{"quote": "CLAUDE.md", "match": r"bulk sRGB→XYB gives different bits", "fail": r"OPEN"}],
        "links": ["zenanalyze.html#drift"],
    },
    {
        "id": "leakage", "title": "Data leakage and exposure",
        "definition": "Evaluation labels influence training, selection or calibration, or a protected set is read more than "
                      "its registration allows.",
        "catches": ["kadid-terminal-d2", "g-rank-board-axes", "table-provenance"],
        "bugs": r"admission|exposure|VAL|holdout|leak",
        "evidence": [{"quote": "docs/DATA_SPLITS.md", "match": r"^## September 14 clarification"}],
        "links": ["splits.html#ledger"],
    },
]

# Code excerpts and quotes for the evaluation page.
EVALUATION = {
    "as_good": {"code": "scripts/rev4_featpot/e24_rev5.py", "start": r"^def as_good\(", "end": r"^\s+and w2\[\"mean\"\]"},
    "e21_floors": {"code": "scripts/rev4_featpot/e21_cheap_recipe.py", "start": r"^MEAN_FLOOR, SOURCE_FLOOR", "end": r"^MEAN_FLOOR"},
    "sdr_decision": {"code": "scripts/rev4_featpot/v40_score.py", "start": r"^def sdr_decision\(", "end": r"^\s+as_good=all"},
    "worst_case": {"code": "scripts/rev4_featpot/e13_teacher.py", "start": r"^def worst_case\(", "end": r"^\s+return out"},
    "consensus": {"code": "scripts/rev4_featpot/e29_consensus.py", "start": r"^def consensus\(", "end": r"^\s+return"},
    "e33_rule": {"quote": E33MD, "match": r"^### 9\.1"},
    "panel_table": {"code": "zensim-validate/src/bin/panel.rs", "start": r"^//! \| Stat / op", "end": r"^//! \| rank vector"},
    "selection_vs_qual": {"quote": SCORECARD, "match": r"^\*\*Qualification clarification, 2026-09-07:\*\*"},
    "e28_decision": {"code": "scripts/rev4_featpot/e24_rev5.py", "start": r"^def e28_arm_decisions\(", "end": r"^\s+adopted = "},
    "split_clarification": {"quote": "docs/DATA_SPLITS.md", "match": r"^## September 14 clarification"},
    "ftest_floor": {"code": FC, "start": r"^mod balanced \{", "end": r"pub const M3A_GOLD"},
}
