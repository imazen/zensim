"""Select a private source budget and pin COSTCMP's frozen candidate builds."""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import costcmp_run as cmp
import speedq_run as owner


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['candidate', 'inventory', 'select'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--mib', type=int, choices=[64, 128, 256])
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    runtime = repo / 'zensim/src/feature_v2.rs'
    provenance = args.root / 'provenance'
    if args.mode in ('candidate', 'select'):
        assert args.mib is not None
        text, count = re.subn(r'const REV5_JOB_BUDGET_BYTES: usize = \d+ \* 1024 \* 1024;',
                             f'const REV5_JOB_BUDGET_BYTES: usize = {args.mib} * 1024 * 1024;', runtime.read_text())
        assert count == 1, 'one private budget constant required'
        if args.mode == 'select':
            assert hashlib.sha256(text.encode()).hexdigest() == sha(provenance/f'runtime-cap{args.mib}.rs'), 'selected source differs from the measured candidate'
        runtime.write_text(text)
        if args.mode == 'candidate':
            with (provenance/f'runtime-cap{args.mib}.rs').open('x') as stream:
                stream.write(text)
        print(f'private queue budget: {args.mib} MiB', flush=True)
        return
    arms = {}
    for arm in cmp.BUDGET_ARMS:
        label = 'uncapped' if arm.endswith('_before') else 'cap'+arm.rsplit('b', 1)[1]
        binary = provenance / f'instrument-{label}'
        artifact = provenance / (binary.name+'.artifact.json')
        source = provenance / f'runtime-{label}.rs'
        arms[arm] = dict(binary=str(binary), artifact=str(artifact), binary_sha256=sha(binary),
                         runtime_source=str(source), runtime_source_sha256=sha(source))
    owner.write(provenance/'budget-binaries.json', dict(schema='rev5perf5-binaries-v1', arms=arms))
    cmp.budget_inventory(provenance/'budget-binaries.json')
    analyzer = provenance/'paired-rounds-analyzer'
    origin = Path('/mnt/v/output/zensim/costcmp-2026-10-09/provenance/paired-rounds-analyzer')
    assert sha(origin) == '798d938626e31c54e524da56687c468f7178d1629b4ee4f268692a290ccbba82'
    with analyzer.open('xb') as stream:
        stream.write(origin.read_bytes())
    analyzer.chmod(origin.stat().st_mode & 0o777)
    model = Path(owner.BAKE)
    assert sha(model) == owner.SOURCE_SHA
    with (provenance/'production-f16.bin').open('xb') as stream:
        stream.write(model.read_bytes())
    owner.write(provenance/'budget-source.json', dict(task='REV5PERF5',
                base_commit=subprocess.check_output(['jj','log','-r','ecc87ac0','--no-graph','-T','commit_id'],cwd=repo,text=True),
                model_sha256=owner.SOURCE_SHA,paired_analyzer_sha256=sha(analyzer),
                host=subprocess.check_output(['hostname'],text=True).strip(),
                rustc=subprocess.check_output(['rustc','--version','--verbose'],text=True),
                grid=dict(tiers=['v4x'],threads=cmp.BUDGET_THREADS,geometries=cmp.BUDGET_GEOMETRIES,budgets_mib=[64,128,256]),
                commands=dict(timing='just -f benchmarks/rev5perf5.just rev5perf5-budget timing ROOT (internal --lock)',
                              rss='just -f benchmarks/rev5perf5.just rev5perf5-budget rss ROOT (internal --lock)'),
                target_cpu_native=False))
    for source, dest in [('Cargo.lock','workspace-Cargo.lock'),('zensim-bench/Cargo.lock','bench-Cargo.lock')]:
        with (provenance/dest).open('xb') as stream:
            stream.write((repo/source).read_bytes())
    print('budget executable/source/model/analyzer inventory pinned', flush=True)


if __name__ == '__main__':
    main()
