# Corruption D228 readout

Readout of the frozen September 8 battery, using the October 16 native audits. No threshold, label, gate or verdict changed. Both candidates use the same authorized head `568380f1`, threshold 0.9, and have the same three misses. Evidence root on dev: `/mnt/v/output/zensim/shipgates-2026-10-16/`; exact case records and pixel measurements: `/mnt/v/output/zensim/gtarget-2026-10-18/CORRUPTION_CASES.json`.

## Honest AVIF: a detector defect, not a threshold near miss

Validation record **14958**, source row **202**, origin **5315**, document, CQ **1**. Reference: `/mnt/v/output/imazen-26-variants/cleanpicker-ladder11@2026-08-23/o_5315.png.scale198x256.png`. Bitstream: `/mnt/v/output/zensim/avif-native-target-2026-09-08/validate-pinned/5315-scalar-0.avif` (SHA256 `9984c21a10520f4f4eaca17fc7ac37b0b078dc39ac59b904df596148891f3499`); adjacent `.png` contains exactly the distorted RGB hash in both native audits (`7efc3821…fe47c`).

I viewed both 198×256 images: the hurricane-pressure table, text, lines and logos remain in place. Of 50,688 pixels, 12,150 differ; maximum channel error is **5**, mean absolute channel error **0.14672111742424243**. The head raw value **4.521377576509678** maps to probability **1.0**, well above 0.9. diff+frag-820’s base score **95.21991684637607** and W-C’s **95.14129573182652** both become **0.0**. The retained file and exact audited pixels agree. This is a severe false positive of the shared detector on a valid, small-error codec reconstruction; the evidence does not identify the responsible feature. It is not explained by a borderline threshold or a mismatched PNG.

## RB swap: real operation, missed detector activation

Validation record **11201**, source row **6823:20**, document. Reference: `/mnt/v/output/zensim/canonical-corruption-2026-09-08/validate/6823/reference.png`; distorted: adjacent `corruption-0018.png` (SHA256 `75951090…23538`). Generator metadata says **whole-image, opaque swap_rb**. Independent pixel comparison proves exact R↔B permutation: all **44,544** pixels change, maximum channel error **89**. Visually the aged paper turns blue while text and layout survive.

Head probability **0.7075219167472319** (raw **−1.1660561625591164**) misses 0.9. Base/composed scores stay **39.9690089560871** (diff+frag-820) and **41.606162744526536** (W-C). This is a true detector false negative against the registered swap contract, with a substantial threshold gap. The operation is neither inert nor mislabeled; a layout-preserving tint shift is the observed failure mode. Its pixels alone do not establish a catastrophic structural failure, a distinction the current operation-based gate does not make.

## GB swap: blended operation within the registered denominator

Validation record **11238**, source row **6823:57**, same reference; distorted: adjacent `corruption-0055.png` (SHA256 `3f0e7d4b…72cfd`). Generator metadata says **whole-image swap_gb, opacity 0.5**, not an opaque channel permutation. Every pixel changes; maximum channel error **28**. The result matches the halfway blend of original and G↔B-swapped RGB to within the expected **0.5** rounding unit. Paper becomes pink; text and layout remain recognizable.

Head probability **0.8175943322297282** (raw **0.12214674771117542**) misses 0.9; unchanged base/composed scores are **44.394217872114844** (diff+frag-820) and **44.36287238256422** (W-C). This is a detector false negative under the registered non-inert-swap rule. There is also a labeling-scope nuance: `params_json` marks `expected_below_lq=true` but `expected_negative=false`; the battery includes it as channel corruption, not as a severity-adjudicated catastrophic case. That does not invalidate its measured gate failure. Calling it a full opaque GB swap would be inaccurate.

The operative failures remain **1/304** honest native lowered, **116/117 GB** and **119/120 RB** detected. The shared head explains identical activation failures across the two candidates; their different scalar models do not repair those decisions. Broader passed rates and the separate unfinished severity qualification remain unchanged. Source records: `corruption-authorized-bars/REGISTERED_BARS.json`; full native values: `corruption-attempt2/{diff-frag-820,w-c}/full.jsonl`; operation metadata: the bound `canonical-corruption-2026-09-08/validate.parquet`, columns `row_id`, `params_json`, `filename` only. No human labels read.
