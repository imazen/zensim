These two ZCTH fixtures are synthetic format/ownership controls, not trained
scientific companions. Both use the existing Python `emit_zcth` owner on the
same one-stump binary classifier (128 synthetic rows, one HGB iteration,
max_leaf_nodes=2, seed6). V3 has no admitted capability. V4 was emitted with
the synthetic record factory in `scripts/tests/test_zcth_admission.py`.

The verdict regression replaces only the v4 metadata section, updating its
length and content digest, to bind its own temporary synthetic input files.
It deliberately exercises rehashed invalid declarations/roles as well as
changed pinned bytes. The fixtures' original temporary paths are not evidence
of scientific admission and are never opened by the fixture loader.
