# E13 SafeSyn strata (pointer)

Per-row codec and quality for the v2-canon SafeSyn fit table, used by the `ts<rule>` teacher-curation arms
(`scripts/rev4_featpot/v2_teacher.py`, design log E13). Not committed: 53 KB binary, rebuilt deterministically.

- File: `/var/tmp/rev4-featpot/v2c/e13/safesyn_fit_strata.npz`, packed into fit program v18 as `data/e13/safesyn_fit_strata.npz`
- sha256: `6baf5d1b2cb012963cdfa19d94ea2717bca17066a49cf9660979669d5662929c` (53,449 bytes, 141,054 rows)
- Aligned to `safesyn_fit.keys.parquet` sha256 `receipt.legs.safesyn.fit.keys_sha256` (checked at build and at every load)
- Built by: `python3 scripts/rev4_featpot/e13_teacher.py strata --root /var/tmp/rev4-featpot/v2c`
  (codec/knob from `/var/tmp/rev4-featbank/bank/safesyn/keys.parquet` by pair_key)
- Codecs: mozjpeg-rs-420-e4, zenavif-s5-e6, zenjpeg-420-e2, zenjpeg-420-xyb-e2, zenjxl-e7, zenwebp-default-m4; q5–q100
