"""Explicit local smoke argv changes its job identity; never a registered fit."""

def apply_smoke_budget(value, owner):
    import v2_common
    import v2_lodo_mlp
    pieces = value.split(":")
    if len(pieces) != 2:
        raise ValueError("local smoke budget must be epochs:pairs")
    epochs, pairs = map(int, pieces)
    if not (1 <= epochs < v2_common.EPOCHS and 1 <= pairs < v2_common.PAIRS_PER_EPOCH):
        raise ValueError("local smoke budget must be positive and below registered budget")
    for target in (vars(v2_common), vars(v2_lodo_mlp), owner):
        target.update(EPOCHS=epochs, PAIRS_PER_EPOCH=pairs, LOG_EVERY=1)
