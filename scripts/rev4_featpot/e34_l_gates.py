"""Bind E34's explicit L smoke model to existing packing and serving owners."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from v2_common import sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['pack', 'identity', 'served-identity', 'nearid', 'verdict-standard', 'verdict-ladder', 'prodqual', 'steer'])
    p.add_argument('--smoke', required=True, type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--bin-dir', required=True, type=Path)
    a = p.parse_args()
    r = json.loads((a.smoke / 'result.json').read_text())
    assert r['execution_contract'] == 'local-smoke' and r['epochs'] == 1
    assert r['spec'] == 'sel:3b7bd5ebe929@h32:H1:cv16:cf98' and r['head'] == 'N'
    model = Path(r['selected_bake'])
    assert sha(model) == r['selected_bake_sha256']
    a.out.mkdir(parents=True, exist_ok=True)
    if a.mode == 'pack':
        from v2_production_pack import pack_production
        dest = a.out / 'packed'
        dest.mkdir()
        (dest / 'refit').mkdir()
        packed = pack_production(model, dest, a.root / 'wide/main/real/cid22_fit.parquet', e33_output_stage=True)
        with (a.out / 'packing.json').open('x') as f:
            json.dump(packed, f, indent=2)
        with (a.out / 'models.json').open('x') as f:
            json.dump({'l': dict(path=packed['packed_model'], sha256=packed['packed_model_sha256'], cell=str(dest))}, f)
        print(json.dumps(packed))
        return
    packed = json.loads((a.out / 'packing.json').read_text())
    model = Path(packed['packed_model'])
    assert sha(model) == packed['packed_model_sha256']
    # Existing gate dispatcher accepts an explicit model set and output root.
    gatebin = a.out / 'bin'
    gatebin.mkdir(exist_ok=True)
    binaries = {'serve_custom_bake': a.bin_dir / 'examples/serve_custom_bake',
                'bake_verdict': a.bin_dir / 'bake_verdict',
                'steer_instrument': a.bin_dir / 'deps/zensim-6fc55070ad1b5213'}
    for name, binary in binaries.items():
        link = gatebin / name
        if not link.exists():
            link.symlink_to(binary.resolve(strict=True))
        assert link.resolve() == binary.resolve()
    if a.mode == 'prodqual':
        with (a.out / 'prodqual.json').open('x') as out, (a.out / 'prodqual.log').open('x') as log:
            subprocess.run([str(binaries['serve_custom_bake']), '--prodqual', str(model)],
                           stdout=out, stderr=log, check=True)
        return
    mode = 'verdict' if a.mode.startswith('verdict-') else a.mode
    cmd = [sys.executable, str(Path(__file__).with_name('e33_gates.py')), mode,
           '--root', str(a.out), '--models', str(a.out / 'models.json')]
    if mode in ('verdict', 'nearid', 'steer'):
        cmd += ['--arm', 'l']
    if mode == 'verdict':
        cmd += ['--grid', a.mode.split('-')[1]]
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    main()
