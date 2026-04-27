//! Architecture detection for cross-platform checksum management.
//!
//! Produces canonical architecture tags like `"x86_64"`, `"x86_64-avx2"`,
//! `"x86_64-avx512"`, or `"aarch64"`. These tags are stored in checksum
//! entries to track which architectures produce each exact output.

/// Detect the current architecture tag at runtime.
///
/// Returns the most specific tag available:
/// - `"x86_64-avx512"` — x86_64 with AVX-512F detected
/// - `"x86_64-avx2"` — x86_64 with AVX2 detected (no AVX-512)
/// - `"x86_64"` — x86_64 without AVX2
/// - `"aarch64"` — ARM 64-bit
/// - `"unknown"` — anything else
///
/// Detection uses `is_x86_feature_detected!` on x86_64 targets, which
/// queries CPUID at runtime. On other platforms, this is compile-time only.
pub fn detect_arch_tag() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f") {
            return "x86_64-avx512";
        }
        if is_x86_feature_detected!("avx2") {
            return "x86_64-avx2";
        }
        "x86_64"
    }

    #[cfg(target_arch = "aarch64")]
    {
        "aarch64"
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "unknown"
    }
}

/// Check whether two architecture tags are compatible.
///
/// An entry's arch tag matches the current platform if:
/// - The entry's tag equals the current tag exactly, or
/// - The entry's tag is a prefix of the current tag (e.g., `"x86_64"` matches
///   `"x86_64-avx2"` — the output was produced on some x86_64 and should work
///   on any x86_64 variant)
///
/// This is intentionally asymmetric: `"x86_64"` matches `"x86_64-avx2"` but
/// `"x86_64-avx2"` does NOT match plain `"x86_64"`, since AVX2 may produce
/// different rounding than scalar code.
pub fn arch_matches(entry_tag: &str, current_tag: &str) -> bool {
    if entry_tag == current_tag {
        return true;
    }
    // entry_tag is a prefix match: "x86_64" matches "x86_64-avx2"
    current_tag.starts_with(entry_tag) && current_tag.as_bytes().get(entry_tag.len()) == Some(&b'-')
}

/// All known architecture tags, for validation.
#[cfg(test)]
pub(crate) const KNOWN_ARCH_TAGS: &[&str] = &[
    "x86_64",
    "x86_64-avx2",
    "x86_64-avx512",
    "aarch64",
    "unknown",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_returns_known_tag() {
        let tag = detect_arch_tag();
        assert!(
            KNOWN_ARCH_TAGS.contains(&tag),
            "detect_arch_tag() returned unknown tag: {tag:?}"
        );
    }

    #[test]
    fn arch_matches_exact() {
        assert!(arch_matches("x86_64-avx2", "x86_64-avx2"));
        assert!(arch_matches("aarch64", "aarch64"));
    }

    #[test]
    fn arch_matches_prefix() {
        assert!(arch_matches("x86_64", "x86_64-avx2"));
        assert!(arch_matches("x86_64", "x86_64-avx512"));
    }

    #[test]
    fn arch_does_not_match_reverse() {
        // More specific does NOT match less specific
        assert!(!arch_matches("x86_64-avx2", "x86_64"));
    }

    #[test]
    fn arch_does_not_match_cross_platform() {
        assert!(!arch_matches("x86_64", "aarch64"));
        assert!(!arch_matches("aarch64", "x86_64-avx2"));
    }

    #[test]
    fn arch_no_false_prefix() {
        // "x86_64" should not match "x86_64_custom" (no dash separator)
        assert!(!arch_matches("x86_64", "x86_64_custom"));
    }
}
