"""E34 frozen assessment adapters; statistics stay in the Rust panel owner.

The preread inventory and committed exposure ledger must precede target access.
Original fitted cells and frozen packets are never modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

R = Path('/mnt/v/output/zensim/e34-launch-2026-10-15')
OUT = R / 'evaluation'
ROOT = Path('/mnt/v/output/zensim/v40r4-2026-10-08/v2e29')
FOLDS = ['kadid', 'tid2013', 'konfig', 'cid22_a25']
COUNTS = [7869, 3000, 756, 2192]
GROUPS = [65, 25, 7, 25]
LABELS = {'controla': 'base-a', 'controlc': 'base-c', 'wa': 'w-a',
          'wc': 'w-c', 'k1a': 'k-a', 'k10c': 'k-c', 'la': 'l-a'}
os.environ.setdefault('REV4_V2_BIN_DIR', str(R / 'bin'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def capture():
    import pandas as pd
    import pyarrow.parquet as pq
    from v2_lodo_mlp import predict
    ledger = Path(__file__).resolve().parents[2] / 'docs/DATA_SPLITS.md'
    assert 'E34 registered evaluation preread' in ledger.read_text()
    inventory = OUT / 'preread.json'
    assert sha(inventory) == 'e3aae632f8cc4537bb91c48c3de88ce039b4c691c8bd0e77c7025715590d1909'
    cells = json.loads(inventory.read_text())['cells']
    grids = {label: {} for label in LABELS.values()}
    for c in cells:
        name = c['jobset'].split('-')[1]
        if name in LABELS and c['fold'] != 'full':
            assert sha(c['selected_model']) == c['selected_sha256']
            key = (c['fold'], c['seed'])
            assert key not in grids[LABELS[name]]
            grids[LABELS[name]][key] = c
    expected = {(f, s) for f in FOLDS for s in range(10)}
    assert all(set(grid) == expected for grid in grids.values())
    receipt = json.loads((ROOT / 'wide/main/real/receipt.json').read_text())
    admitted = []
    for fold, count, groups in zip(FOLDS, COUNTS, GROUPS, strict=True):
        rec = receipt['legs'][fold]['full']
        table = ROOT / rec['rel']
        keys = table.with_suffix('.keys.parquet')
        assert sha(table) == rec['sha256']
        assert sha(str(table) + '.manifest.json') == rec['manifest_sha256']
        assert sha(keys) == rec['keys_sha256']
        meta = pq.read_table(keys).to_pandas()
        assert len(meta) == count
        names = meta.ref_basename.tolist()
        if fold == 'konfig':
            names = [n.removesuffix('_parta').removesuffix('_partb') for n in names]
        ids = {v: i for i, v in enumerate(sorted(set(names)))}
        assert len(ids) == groups
        members = sorted(set(meta.member_set))
        joins = []
        for member in members:
            path = Path('/var/tmp/rev4-featbank/bank') / member / 'keys.parquet'
            assert sha(path) == receipt['bank'][member]['keys_sha256']
            data = pq.read_table(path, columns=['pair_key', 'codec', 'dist_path']).to_pandas()
            if fold in ('kadid', 'tid2013'):
                data['codec'] = [Path(p).stem.split('_')[1] for p in data.dist_path]
            joins.append(data[['pair_key', 'codec']])
        joined = pd.concat(joins).drop_duplicates()
        assert not joined.pair_key.duplicated().any()
        codec = joined.set_index('pair_key').codec.reindex(meta.pair_key)
        assert codec.notna().all()
        codec_ids = {v: i for i, v in enumerate(sorted(set(codec)))}
        admitted.append((fold, table, [ids[v] for v in names], [codec_ids[v] for v in codec]))
    write(OUT / 'label-read-started.json', dict(ledger_sha256=sha(ledger),
          preread_sha256=sha(inventory), script_sha256=sha(__file__),
          scope='four frozen D1 target columns; seven paired ten-seed grids'))
    datasets = []
    for fold, table, reference, codec in admitted:
        target = pq.read_table(table, columns=['human_score'])['human_score'].to_pylist()
        predictions = {}
        for label, grid in grids.items():
            predictions[label] = []
            for seed in range(10):
                dest = OUT / 'predictions' / label / f'{fold}_s{seed}'
                dest.mkdir(parents=True)
                pred = predict(Path(grid[fold, seed]['selected_model']), table, dest / 'pred.tsv')
                predictions[label].append(pred.tolist())
                print(f'captured {label} {fold} seed {seed}', flush=True)
        datasets.append(dict(name=fold, target=target, reference=reference, codec=codec,
                             orientation=1., predictions=predictions))
    tests = [('w-a', 'base-a', True), ('k-a', 'base-a', True), ('l-a', 'base-a', True),
             ('w-c', 'base-c', True), ('k-c', 'base-c', True), ('k-a', 'w-a', False),
             ('l-a', 'w-a', False), ('l-a', 'k-a', False), ('k-c', 'w-c', False)]
    write(OUT / 'panel-input.json', dict(schema='e34-paired-seed-v1', datasets=datasets,
          tests=[dict(index=i + 1, arm=a, control=c, eligibility=e)
                 for i, (a, c, e) in enumerate(tests)]))
    write(OUT / 'capture-complete.json', dict(input_sha256=sha(OUT / 'panel-input.json'),
          prediction_panels=280, observations=sum(COUNTS), references=sum(GROUPS)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['capture'])
    parser.parse_args()
    capture()
