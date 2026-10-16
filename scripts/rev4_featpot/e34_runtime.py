"""Bind E34 full-data seed-0 models to SPEEDQ's unchanged collectors.

The family control is the paired runtime baseline. Timing uses the existing
32 clean paired rounds, quiet gate and frozen paired-round analyzer.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'demos'))
import speedq_run as owner

ROOT = Path('/mnt/v/output/zensim/e34-launch-2026-10-15/evaluation')
ANALYZER = Path('/mnt/v/output/zensim/costcmp-2026-10-09/provenance/paired-rounds-analyzer')
ANALYZER_SHA = '798d938626e31c54e524da56687c468f7178d1629b4ee4f268692a290ccbba82'
ARMS = ['e34_control', 'e34_candidate', 'e34_candidate_map', 'e34_candidate_baseline']
GRID = [(g, t, 1) for t in ['v4x', 'v3'] for g in ['64x64', '256x256', '1024x1024']]
SPATIAL = [('2048x2048', 'v4x', 1)]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def models(arm):
    full = json.loads((ROOT / 'full-models.json').read_text())
    family = arm[-1]
    for label, name in [('CONTROL', f'base-{family}'), ('CANDIDATE', arm)]:
        record = full[f'{name}-s0']
        assert sha(record['path']) == record['sha256']
        hidden = 32 if name.startswith('w-') else 1 if name.startswith('l-') else 128
        os.environ[f'ZEN_S2_E34_BAKE_{label}'] = record['path']
        os.environ[f'ZEN_S2_E34_SHA_{label}'] = record['sha256']
        os.environ[f'ZEN_S2_E34_HIDDEN_{label}'] = str(hidden)
    os.environ['SPEEDQ_LOCK_OWNER'] = 'codex-e34-eval'
    os.environ['ZENSIM_NEIGHBOUR_EXACT'] = '1'
    os.environ.pop('ZENSIM_MAX_TIER', None)


def ready(rows, name, geometry, tier, threads, record):
    old = rows[geometry, tier, threads, name]
    assert (record['width'], record['height']) == tuple(map(int, geometry.split('x')))
    assert record['tier'] == tier and int(record['threads']) == threads
    assert record['arm'] == name
    assert (record['input_sha256'], record['score_bits'], record['model']) == (
        old['input_sha256'], old['score_bits'], old['model'])
    label = 'CONTROL' if name == 'e34_control' else 'CANDIDATE'
    assert record['model']['source_sha256'] == os.environ[f'ZEN_S2_E34_SHA_{label}']
    assert record['model']['revision'] == '5'


def load(path):
    data = json.loads(Path(path).read_text())
    assert data['schema'] == 'e34-runtime-preflight-v1' and data['status'] == 'PASS'
    rows = {(f"{r['width']}x{r['height']}", r['tier'], int(r['threads']), r['arm']): r
            for r in data['records']}
    assert len(rows) == len(data['records']) and all(math.isfinite(r['score']) for r in rows.values())
    assert set(rows) == {(g, t, n, a) for g, t, n in GRID + SPATIAL for a in ARMS}
    for geometry in ['64x64', '256x256', '1024x1024']:
        for arm in ARMS:
            assert len({rows[geometry, tier, 1, arm]['score_bits'] for tier in ['v4x', 'v3']}) == 1
    for g, t, n in GRID + SPATIAL:
        assert rows[g, t, n, 'e34_candidate']['score_bits'] == rows[g, t, n, 'e34_candidate_map']['score_bits']
        assert len({rows[g, t, n, a]['input_sha256'] for a in ARMS}) == 1
        for a in ARMS:
            ready(rows, a, g, t, n, rows[g, t, n, a])
    return rows


def main():
    global ROOT, ANALYZER
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['ready', 'parity', 'timing', 'rss'])
    p.add_argument('--arm', choices=['w-a', 'k-a', 'l-a', 'w-c', 'k-c'], required=True)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--root', type=Path, default=ROOT,
                   help='assessment artifact root on the owner-selected measurement host')
    p.add_argument('--analyzer', type=Path, default=ANALYZER,
                   help='relocated copy of the unchanged SHA-pinned analyzer')
    a = p.parse_args()
    ROOT, ANALYZER = a.root, a.analyzer
    assert sha(ANALYZER) == ANALYZER_SHA, 'registered paired-round analyzer changed'
    models(a.arm)
    root = ROOT / 'runtime' / a.arm
    root.mkdir(parents=True, exist_ok=True)
    if a.mode == 'ready':
        owner.quiet_gate(root / 'startup-quiet-waits.jsonl')
        return
    parity = root / 'PREFLIGHT_PASS.json'
    if a.mode == 'parity':
        assert not parity.exists()
        records = []
        for g, t, n in GRID + SPATIAL:
            for arm in ARMS:
                rec = owner.worker_run(a.binary, arm, 5, g, t, n, root / f'{t}-{g}-{arm}.log')
                records.append(rec)
        owner.write(parity, dict(schema='e34-runtime-preflight-v1', status='PASS',
                                binary_sha256=sha(a.binary), records=records))
        try:
            load(parity)
        except Exception:
            parity.rename(root / 'PREFLIGHT_FAILED.json')
            raise
    else:
        assert sha(a.binary) == json.loads(parity.read_text())['binary_sha256']
        if a.mode == 'timing':
            owner.timing(a.binary, root / 'timing', 32, parity, ANALYZER,
                         arms=ARMS[:3], baseline='e34_control', cells=GRID + SPATIAL,
                         parity_loader=load, ready_check=lambda *args: ready(load(parity), *args),
                         lock=Path.home() / 'tmp/zensim-paper/rev4/heavy.lock')
        else:
            owner.rss(a.binary, root / 'rss', parity, arms=ARMS[1:], parity_loader=load,
                      ready_check=lambda *args: ready(load(parity), *args),
                      geometries=['1024x1024', '2048x2048'], thread_counts=[1])


if __name__ == '__main__':
    main()
