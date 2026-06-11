//! Public-API surface snapshots for this workspace's published crates
//! (`zensim` + `zensim-regress`, auto-discovered from `publish` flags),
//! regenerated into `docs/public-api/<crate>.txt` on every `cargo test`.
//!
//! The implementation is the shared `zenutils-apidoc` crate — see its docs
//! for the snapshot format (taxonomy summary + delta features section) and
//! the `ZEN_API_DOC=off|check|regen` / `ZEN_API_DOC_TOOLCHAIN` protocol.

#[test]
fn public_api_surface_docs_are_current() {
    zenutils_apidoc::run();
}
