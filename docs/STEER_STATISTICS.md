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
max(bootstrap,corrected Student) p and guards. Invalid admission, primary inference
(including zero/nonfinite primary SE), or missing/nonfinite required numeric
guard evidence produces INCOMPLETE/p1 and blocks both S verdicts. Ordinary
primary-seed and W2-seed refusals are nested INCOMPLETE diagnostics with reasons,
not hypothesis/packet vetoes; the frozen E34 owners already catch those errors.
Constant constituent rank inputs cannot supply a valid correlation. All source
and type coverage, finite inputs and alias bits are checked before the exception.
Dataset/W2 intervals are required on all twelve comparisons, including ordinary
improvements7/9/10/12. Improvement aliases8/11 retain analytic zero-delta/zero-SE
guard records under A; their strict W2 booleans are false. Finite false diagnostic
point guards are nondecisive for S eligibility. Under Q12-B the ordinary primary
zero-SE alias comparisons keep the packet incomplete and diagnostic-only.

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

`just steer-fix2-check` adds real invented-prediction cases with zero seed-only
SE and nonzero primary reference uncertainty. It also runs the four reviewed
boundary mutations on actual canonical primary/guard receipts, with the real
seed_stat and Holm owners, then connects them to the S-only verdict. Set
STEER_FIX2_RESULT to a fresh explicit receipt path. No statistical values are
injected into the primary bootstrap; the receipt labels the diagnostic/guard
mutations at the packet-completion boundary.
