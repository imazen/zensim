//! Explicit pinned-owner oracle. Does not discover or open any corpus.
#[cfg(e34_owner_oracle)]
include!(concat!(env!("OUT_DIR"), "/e34-owner-entry.rs"));

#[cfg(not(e34_owner_oracle))]
fn main() -> std::process::ExitCode {
    eprintln!(
        "owner oracle was not materialized; build with E34_OWNER_ORACLE=1 through registered-statistics-build"
    );
    std::process::ExitCode::from(2)
}
