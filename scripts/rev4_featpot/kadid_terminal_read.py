#!/usr/bin/env python3
"""Canonical terminal entry: execute and pin retained source, never cached pyc."""
import sys
from pathlib import Path
import _terminal_acceptance as acceptance

acceptance.capture(__file__, sys._getframe().f_code)
_owner = acceptance.load(Path(__file__).with_name("_terminal_owner.py"), "_terminal_owner")
if __name__ == "__main__":
    sys.exit(_owner.main())
else:
    sys.modules[__name__] = _owner
