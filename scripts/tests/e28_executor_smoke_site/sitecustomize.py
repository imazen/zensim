"""Opt-in bounded smoke constants for the unmodified installed fit script.

The executor and declared argv are unchanged. The hook runs at owner.main()
entry after the registered module constants have passed their normal checks.
"""
import os
import sys

OWNER = "/opt/fleet-fits/program/scripts/rev4_featpot/v2_lodo_mlp.py"
if os.environ.get("E28_BOUNDED_SMOKE") == "1" and sys.argv[0] == OWNER:
    def bounded_entry(frame, event, _arg):
        if event == "call" and frame.f_code.co_filename == OWNER and frame.f_code.co_name == "main":
            globals_ = frame.f_globals
            assert (globals_["EPOCHS"], globals_["PAIRS_PER_EPOCH"], globals_["LOG_EVERY"]) == (120, 50_000, 17)
            globals_.update(EPOCHS=2, PAIRS_PER_EPOCH=128, LOG_EVERY=1)
            sys.settrace(None)
            print("E28_BOUNDED_SMOKE: installed owner.main; epochs=2 pairs=128 log=1; argv unchanged", file=sys.stderr, flush=True)
        return None

    sys.settrace(bounded_entry)
