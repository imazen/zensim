"""Explicit local budget override through the real strict owners, never a fleet cell."""
import json
import sys
from pathlib import Path

root, dest, bin_dir, route = sys.argv[1:]
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "rev4_featpot"))
import os
os.environ['REV4_V2_BIN_DIR'] = bin_dir
import e21_cheap_recipe as e21
from e30_four_source import SPEC
# v2_common binds its data root from argv at import; this script imported it
# for recipe resolution, so explicitly set the same root in both real owners.
import v2_common as common
common.V2 = Path(root)
import v2_lodo_mlp as lodo
common.EPOCHS, common.PAIRS_PER_EPOCH = 2, 128
lodo.V2, lodo.EPOCHS, lodo.PAIRS_PER_EPOCH, lodo.LOG_EVERY = Path(root), 2, 128, 1
owner = lodo
argv = ["--spec", SPEC, "--head", "N", "--seed-index", "0", "--root", root,
        "--dest", dest, "--columns", ','.join(map(str, e21.columns('by_v2fy'))),
        "--strict-admission", "--train-only", "--data-role-decision", str(Path(root)/'human_role_decision.json')]
if route == 'production':
    import v2_confirm_fit as owner
    owner.V2, owner.EPOCHS, owner.PAIRS_PER_EPOCH = Path(root), 2, 128
    argv += ['--pack-production']
elif route == 'e30':
    argv += ['--heldout', 'kadid']
else:
    raise ValueError(route)
sys.argv = [owner.__file__, *argv]
owner.main()
record = json.loads((Path(dest)/'result.json').read_text())
assert record['selection']['selected_epoch'] == 1
assert len(record['selection']['strict_table_admission']) == 7
print(json.dumps({'status':'PASS', 'route':route, 'epochs':2, 'pairs_per_epoch':128,
                  'selected_epoch':1, 'selected_bake_sha256':record['selected_bake_sha256']}))
