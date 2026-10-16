# Versioned STEER evaluation

`panel e34 steer-evaluate --registration PIN --input REQUEST --output FRESH`
selects the separate STEER schema. The existing E34 evaluate/noise/report paths
retain their frozen nine-test inventory and numerical owners.

PIN has schema `steer-registration-v1`, fixed fits172, tau0.125, the inherited
rank-data digest, identical_tests[2,5,8,11], alias_policy (`q12-a-v1` or
`e34-strict-b-v1`) and q12_approved. The coordinator must bind the owner answer
before setting approval. Without approval, the command writes PENDING_OWNER_Q12
**before opening REQUEST**. No fit or scientific exposure is authorized here.
REQUEST schema `steer-evaluation-v1` binds the exact PIN SHA256 and embeds paired
schema `steer-paired-seed-v1`, four D1 datasets and the fixed twelve tests.

Test order: S/R/SR against base-A; S/R/SR against base-C; R−S, SR−S, SR−R for A,
then those three for C. Labels use s-a,r-a,sr-a,base-a and corresponding c names.
Ordinary predictions must prove R=base and SR=S bit-for-bit across all rows/seeds.
Only comparisons2/5/8/11 are declared pairwise identical before fitting.

Under Q12-A, validated identical comparisons have the distinct state
IDENTICAL_BY_CONSTRUCTION and decision p=1. Their canonical bootstrap difference
is zero (theta is shifted by the unchanged margin), and their expected zero seed
SE is recorded before calling the refusing seed owner. Student inferential fields
are null with the construction reason. These are not statistical passes. All12
Holm slots remain. R's strict W2 guard is still false: 0>−2*0 is false.

Ordinary comparisons use the existing unstratified reference×seed bootstrap,
max(bootstrap,corrected Student) p and guards. A missing/invalid component or an
ordinary seed_stat refusal produces INCOMPLETE/p1 and blocks the whole S verdict.
Constant constituent rank inputs cannot supply a valid correlation. All source
and type coverage, finite inputs and alias bits are checked before the exception.
Under Q12-B, those aliases remain incomplete and the packet is diagnostic-only.

`steer_verdict` in the existing e34_verdict owner admits only S-A/S-C under A;
R/SR never become fallbacks. All required gate records, including diagnostic
variants and controls, are required; a measured diagnostic failure is not missing
evidence. No thresholds change. Adoption remains a separate owner decision.

`panel e34 steer-pack-check` requires exactly twelve named full-data bindings:
base-a,base-c,s-a,s-c each suffixed -s0,-s1,-s2. The inherited packed-spline
checker validates them; R/SR are mode aliases rather than extra fitted weights.

`just steer-statistics-check` exercises actual bootstrap, Student, Holm, seed
refusal and admission on invented predictions, plus frozen E34 tests and Clippy.
`just steer-verdict-tests` checks S-only verdicts and incomplete cases. To connect
the owners, explicitly set STEER_INVENTED_RESULT to a fresh receipt when running
the Rust tests, then run `just steer-statistics-verdict-integration` with that same
receipt. This is invented-data engineering evidence, not a qualification result.
