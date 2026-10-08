set working-directory := '..'
heavy := 'flock /home/lilith/tmp/zensim-paper/rev4/heavy.lock /home/lilith/work/claudehints/scripts/run-heavy --mem 16G --jobs 8 --'
evidence := '/mnt/v/output/zensim/releasegate6-2026-10-08'
scratch := '/home/lilith/tmp/releasegate6'
panel := '/mnt/v/output/zensim/releasegate2-2026-10-07/bin/panel'

suite:
    {{heavy}} env TMPDIR={{scratch}} OPENBLAS_NUM_THREADS=1 ZEN_PANEL_BIN={{panel}} just releasegate-tests > {{evidence}}/tests.log 2>&1

lint:
    ruff check --select F scripts/rev4_featpot/_terminal_acceptance.py scripts/rev4_featpot/_terminal_bound_io.py scripts/rev4_featpot/_terminal_owner.py scripts/tests/test_kadid_terminal_read.py scripts/tests/test_kadid_terminal_hardening.py scripts/tests/test_kadid_terminal_ledger.py scripts/tests/kadid_terminal_fixture_devices.py scripts/tests/releasegate6_before.py > {{evidence}}/ruff.log 2>&1

before:
    {{heavy}} env TMPDIR={{scratch}} OPENBLAS_NUM_THREADS=1 ZEN_PANEL_BIN={{panel}} python3 scripts/tests/releasegate6_before.py {{scratch}}/base > {{evidence}}/before.log 2>&1
