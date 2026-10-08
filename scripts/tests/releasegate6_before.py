"""Run the three independent round-six ledger probes on an old source export."""
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import unittest


def main():
    source = Path(sys.argv[1]).resolve()
    if source.exists():
        raise FileExistsError('preserve the original-tip control export')
    git_dir = subprocess.check_output(
        ['jj', '--ignore-working-copy', 'git', 'root'], text=True
    ).strip()
    files = [
        'scripts/rev4_featpot/' + name for name in
        ('kadid_terminal_read.py', '_terminal_acceptance.py', '_terminal_owner.py',
         '_terminal_bound_io.py', 'v2c_labels.py')
    ] + ['scripts/lib/zen_stats.py',
         'benchmarks/shippath_qualified_fit_contract_2026-10-07.json',
         'scripts/tests/test_kadid_terminal_read.py',
         'scripts/tests/test_kadid_terminal_bound_payloads.py']
    archive = subprocess.check_output([
        'git', '--git-dir', git_dir, 'archive',
        '42972b5a3281342fac6a747f31beacf6a7a15781', *files,
    ])
    source.mkdir(parents=True)
    with tarfile.open(fileobj=io.BytesIO(archive)) as tree:
        tree.extractall(source, filter='data')
    sys.path.insert(0, str(source / 'scripts/tests'))
    path = Path(__file__).with_name('test_kadid_terminal_ledger.py')
    spec = importlib.util.spec_from_file_location('round6_probes_on_old_source', path)
    probes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probes)
    assert probes.owner.acceptance.__file__.startswith(str(source) + '/')
    cases = (
        'test_rename_over_after_preflight_refuses_before_journal_or_labels',
        'test_rename_over_after_reservation_refuses_result_append',
        'test_other_writer_can_lock_during_assessment_and_after_append',
    )
    suite = unittest.TestSuite(probes.LedgerTransactions(name) for name in cases)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.testsRun == 3 and len(result.failures) == 2 and len(result.errors) == 1
    assert not result.skipped and not result.unexpectedSuccesses
    print('PASS negative control: original tip fails both rename-over refusals and blocks the assessment-time writer')


if __name__ == '__main__':
    main()
