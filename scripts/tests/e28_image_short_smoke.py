"""Bounded image smoke through the installed v2 owner; registration/grid unchanged."""
import json
from pathlib import Path
import sys

# Arguments are the normal v2 owner CLI. The caller imports the actual immutable
# installed program, not source from the editable checkout.
program = Path('/opt/fleet-fits/program')
sys.path.insert(0, str(program / 'scripts/rev4_featpot'))
import v2_lodo_mlp as owner
owner.EPOCHS, owner.PAIRS_PER_EPOCH, owner.LOG_EVERY = 2, 128, 1
owner.main()
