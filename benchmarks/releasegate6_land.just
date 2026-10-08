set working-directory := '..'

suite:
    just releasegate-tests

all-features:
    cargo test -p zensim --all-features

lint-scripts:
    just lint-scripts

# Caller runs this through its shared heavy lock and run-heavy wrapper.
[positional-arguments]
archive receipt:
    python3 "$1"
