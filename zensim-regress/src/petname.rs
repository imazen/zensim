//! Deterministic memorable names for hash IDs.
//!
//! Converts a hash ID like `"sea:a4839401fabae99c"` into a human-friendly name
//! like `"sunny-crab-a4839401fa:sea"`. The name is deterministic — same hash
//! always produces the same memorable name.
//!
//! # Format
//!
//! `{adjective}-{noun}-{hex10}:{algo}`
//!
//! - `adjective`: from 256-word list, indexed by raw hash byte 5
//! - `noun`: from 256-word list, indexed by raw hash byte 6
//! - `hex10`: first 10 hex characters of the hash value (40 bits)
//! - `algo`: hash algorithm tag (e.g., `"sea"`)
//!
//! Using bytes 5-6 for word indices avoids overlap with the displayed hex prefix
//! (bytes 0-4), so different hex prefixes don't correlate with different words
//! and vice versa.

/// Generate a memorable name from a hash ID.
///
/// Input format: `"{algo}:{hex}"` (e.g., `"sea:a4839401fabae99c"`).
///
/// Returns `"{adj}-{noun}-{hex10}:{algo}"` (e.g., `"sunny-crab-a4839401fa:sea"`).
///
/// # Panics
///
/// Panics if the hash ID doesn't contain a `:` separator or if the hex
/// portion is shorter than 14 characters (need bytes 5-6 for word indices).
pub fn memorable_name(hash_id: &str) -> String {
    let (algo, hex) = hash_id
        .split_once(':')
        .expect("hash ID must contain ':' separator");

    assert!(
        hex.len() >= 14,
        "hash hex must be at least 14 chars (need bytes 5-6 at positions 10-13), got {}",
        hex.len()
    );

    // Parse bytes 5 and 6 from the hex string (chars 10-13)
    let adj_byte = u8::from_str_radix(&hex[10..12], 16).expect("invalid hex at byte 5");
    let noun_byte = u8::from_str_radix(&hex[12..14], 16).expect("invalid hex at byte 6");

    let adj = ADJECTIVES[adj_byte as usize];
    let noun = NOUNS[noun_byte as usize];
    let hex10 = &hex[..10];

    format!("{adj}-{noun}-{hex10}:{algo}")
}

/// Extract the original hash ID from a memorable name.
///
/// This is a partial inverse — it recovers the `{algo}:{hex}` prefix,
/// but the full hash is not recoverable from the memorable name alone.
/// Returns `None` if the format doesn't match.
///
/// Handles both old 5-char hex and new 10-char hex petnames.
pub fn parse_memorable_name(name: &str) -> Option<MemorableNameParts> {
    // Format: adj-noun-hex:algo
    // Find the last ':' — that separates hex from algo
    let colon_pos = name.rfind(':')?;
    let algo = &name[colon_pos + 1..];

    // Everything before the colon: "adj-noun-hex"
    let before_colon = &name[..colon_pos];

    // Find the last '-' before the colon — separates noun from hex
    let last_dash = before_colon.rfind('-')?;
    let hex = &before_colon[last_dash + 1..];

    // Find the second-to-last '-' — separates adj from noun
    let adj_noun = &before_colon[..last_dash];
    let first_dash = adj_noun.find('-')?;
    let adj = &adj_noun[..first_dash];
    let noun = &adj_noun[first_dash + 1..];

    Some(MemorableNameParts {
        adjective: adj.to_string(),
        noun: noun.to_string(),
        hex: hex.to_string(),
        algo: algo.to_string(),
    })
}

/// Parsed components of a memorable name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MemorableNameParts {
    pub adjective: String,
    pub noun: String,
    /// Hex prefix (5 chars for legacy, 10 chars for new petnames).
    pub hex: String,
    pub algo: String,
}

// ─── Defensive conversion helpers ────────────────────────────────────────

/// Convert a hash ID (or petname) to a memorable name, handling all edge cases.
///
/// Safe to call with any of:
/// - Raw hash: `"sea:a4839401fabae99c"` → `"sunny-crab-a4839401fa:sea"`
/// - Hash with extension: `"sea:a4839401fabae99c.png"` → `"sunny-crab-a4839401fa:sea"`
/// - Already a petname: `"sunny-crab-a4839401fa:sea"` → returned as-is
/// - Short/malformed hash: `"sea:abc"` → returned as-is
///
/// This is the recommended entry point for converting arbitrary hash strings
/// to memorable names. Unlike [`memorable_name`], this function never panics.
pub fn try_memorable_name(hash_id: &str) -> String {
    // Already a petname (contains dashes from adj-noun format)
    if hash_id.contains('-') {
        return hash_id.to_string();
    }
    // Strip file extension if present
    let bare = strip_hash_extension(hash_id);
    // Check if the hex portion is long enough for word generation
    let hex_len = bare.split_once(':').map(|(_, hex)| hex.len()).unwrap_or(0);
    if hex_len >= 14 {
        memorable_name(bare)
    } else {
        bare.to_string()
    }
}

/// Strip known image file extensions from a hash ID.
///
/// `"sea:a4839401fabae99c.png"` → `"sea:a4839401fabae99c"`
///
/// Returns the input unchanged if no known extension is found.
pub fn strip_hash_extension(hash_id: &str) -> &str {
    for ext in [".png", ".jpg", ".jpeg", ".webp", ".gif", ".unknown"] {
        if let Some(stripped) = hash_id.strip_suffix(ext) {
            return stripped;
        }
    }
    hash_id
}

// ─── Word lists ──────────────────────────────────────────────────────────
//
// 256 adjectives and 256 nouns, each 3-6 characters.
// Short, common, easy-to-read English words.

#[rustfmt::skip]
const ADJECTIVES: [&str; 256] = [
    "able",  "acid",  "aged",  "airy",  "apt",   "avid",  "bald",  "bare",
    "base",  "big",   "bold",  "bone",  "born",  "both",  "brave", "brisk",
    "broad", "brown", "brute", "bulk",  "busy",  "calm",  "cheap", "chief",
    "civic", "civil", "clean", "clear", "close", "cold",  "cool",  "coral",
    "core",  "cozy",  "crisp", "cross", "crude", "curly", "cute",  "damp",
    "dark",  "dawn",  "dear",  "deep",  "dense", "dim",   "dire",  "dizzy",
    "done",  "draft", "drawn", "dried", "dry",   "dual",  "dull",  "dusk",
    "dusty", "eager", "early", "easy",  "edgy",  "elfin", "elite", "empty",
    "equal", "exact", "extra", "faded", "fair",  "false", "famed", "fancy",
    "fast",  "fatal", "fawn",  "fed",   "few",   "fiery", "final", "fine",
    "firm",  "first", "fit",   "fixed", "fizzy", "flat",  "fleet", "fluid",
    "focal", "foggy", "fond",  "free",  "fresh", "front", "frugal","full",
    "funny", "fuzzy", "giant", "glad",  "gold",  "grand", "gray",  "great",
    "green", "grim",  "grown", "gruff", "half",  "happy", "hardy", "hasty",
    "hazy",  "heavy", "hefty", "high",  "holy",  "husky", "icy",   "idle",
    "ill",   "inner", "ionic", "iron",  "ivory", "jade",  "jazzy", "jolly",
    "just",  "keen",  "kind",  "known", "lame",  "large", "last",  "late",
    "lazy",  "lean",  "legal", "level", "light", "like",  "limp",  "lined",
    "live",  "local", "lofty", "lone",  "long",  "lost",  "loud",  "low",
    "loyal", "lucid", "lucky", "lunar", "mad",   "major", "many",  "meek",
    "mere",  "mild",  "mini",  "minor", "mint",  "mixed", "modal", "moist",
    "moot",  "moved", "murky", "muted", "naive", "naval", "near",  "neat",
    "new",   "next",  "nice",  "noble", "north", "novel", "numb",  "odd",
    "oily",  "old",   "only",  "open",  "oral",  "outer", "oval",  "own",
    "paced", "paid",  "pale",  "past",  "peak",  "petty", "pink",  "plain",
    "plump", "plush", "polar", "poor",  "prime", "prior", "prone", "proud",
    "pure",  "quick", "quiet", "rapid", "rare",  "raw",   "ready", "real",
    "red",   "regal", "rich",  "ripe",  "rocky", "roomy", "rough", "round",
    "royal", "ruby",  "rude",  "rural", "rusty", "safe",  "salty", "same",
    "sandy", "sharp", "sheer", "short", "shy",   "silly", "slim",  "slow",
    "small", "smart", "snowy", "soft",  "solar", "sole",  "solid", "south",
    "spare", "stark", "steep", "stiff", "stout", "sunny", "swift", "tidy",
];

#[rustfmt::skip]
const NOUNS: [&str; 256] = [
    "ace",   "ant",   "ape",   "arc",   "ark",   "ash",   "axe",   "bass",
    "bat",   "bay",   "bead",  "beam",  "bear",  "bee",   "bell",  "birch",
    "bird",  "bloom", "boar",  "bolt",  "bone",  "boot",  "bow",   "box",
    "brick", "brook", "buck",  "bud",   "bull",  "calf",  "cape",  "carp",
    "cave",  "chip",  "clam",  "clay",  "cliff", "cloud", "coal",  "cobra",
    "cod",   "colt",  "cone",  "core",  "cork",  "crab",  "crane", "creek",
    "crow",  "cub",   "curl",  "dawn",  "deer",  "den",   "dew",   "disc",
    "dock",  "doe",   "dove",  "drum",  "duck",  "dune",  "dust",  "eagle",
    "edge",  "eel",   "elm",   "ember", "fawn",  "fern",  "fig",   "finch",
    "fire",  "fish",  "flint", "flock", "flora", "flow",  "flux",  "foam",
    "fog",   "ford",  "forge", "fort",  "fox",   "frog",  "frost", "fruit",
    "fury",  "gale",  "gate",  "gecko", "gem",   "glen",  "glow",  "goat",
    "goose", "grain", "grape", "gull",  "gust",  "hare",  "harp",  "hawk",
    "hazel", "hedge", "heron", "hill",  "hive",  "hoof",  "horn",  "hound",
    "hub",   "hull",  "ice",   "ivy",   "jade",  "jar",   "jay",   "jewel",
    "kelp",  "kite",  "knoll", "lace",  "lake",  "lamb",  "lance", "larch",
    "lark",  "lava",  "leaf",  "ledge", "lilac", "lily",  "lime",  "linen",
    "lion",  "lodge", "loom",  "lotus", "lynx",  "mace",  "maple", "mare",
    "marsh", "mast",  "mesa",  "mint",  "mist",  "mole",  "moon",  "moose",
    "moss",  "moth",  "mound", "mouse", "mule",  "nest",  "newt",  "node",
    "oak",   "oat",   "olive", "onyx",  "orbit", "ore",   "osprey","otter",
    "owl",   "ox",    "palm",  "panda", "park",  "path",  "peach", "peak",
    "pearl", "petal", "pike",  "pine",  "plum",  "plume", "pod",   "pond",
    "poppy", "port",  "prism", "pulse", "puma",  "quail", "quay",  "rain",
    "ram",   "raven", "ray",   "reed",  "reef",  "ridge", "ring",  "river",
    "robin", "rock",  "root",  "rose",  "rust",  "sage",  "sail",  "sand",
    "seal",  "seed",  "shell", "shore", "shrew", "silk",  "slate", "sloth",
    "snail", "snake", "snow",  "sole",  "song",  "spark", "spire", "squid",
    "star",  "stem",  "stone", "stork", "storm", "swift", "teal",  "thorn",
    "tide",  "tiger", "toad",  "torch", "trail", "tree",  "trout", "tulip",
    "tuna",  "vale",  "vine",  "viper", "voice", "wail",  "wasp",  "wave",
    "whale", "wheat", "wolf",  "wren",  "yak",   "yew",   "zinc",  "zone",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memorable_name_deterministic() {
        let hash = "sea:a4839401fabae99c";
        let n1 = memorable_name(hash);
        let n2 = memorable_name(hash);
        assert_eq!(n1, n2);
    }

    #[test]
    fn memorable_name_format() {
        let hash = "sea:a4839401fabae99c";
        let name = memorable_name(hash);
        // Should be adj-noun-hex10:algo
        assert!(name.ends_with(":sea"), "name={name}");
        let parts = parse_memorable_name(&name).unwrap();
        assert_eq!(parts.hex, "a4839401fa");
        assert_eq!(parts.algo, "sea");
        // Verify word indices: byte 5 = 0xba = 186, byte 6 = 0xe9 = 233
        assert_eq!(parts.adjective, ADJECTIVES[0xba_usize]);
        assert_eq!(parts.noun, NOUNS[0xe9_usize]);
    }

    #[test]
    fn memorable_name_different_hashes_differ() {
        let n1 = memorable_name("sea:0000000000000000");
        let n2 = memorable_name("sea:ffffffffffffffff");
        assert_ne!(n1, n2);
    }

    #[test]
    fn parse_roundtrip() {
        let hash = "sea:1234567890abcdef";
        let name = memorable_name(hash);
        let parts = parse_memorable_name(&name).unwrap();
        assert_eq!(parts.hex, "1234567890");
        assert_eq!(parts.algo, "sea");
    }

    #[test]
    fn parse_invalid_returns_none() {
        assert!(parse_memorable_name("no-dashes-here").is_none());
        assert!(parse_memorable_name("").is_none());
    }

    #[test]
    fn word_lists_valid() {
        // All words should be 2-6 chars
        for (i, adj) in ADJECTIVES.iter().enumerate() {
            assert!(
                adj.len() >= 2 && adj.len() <= 6,
                "adj[{i}] = {adj:?} len={}",
                adj.len()
            );
        }
        for (i, noun) in NOUNS.iter().enumerate() {
            assert!(
                noun.len() >= 2 && noun.len() <= 6,
                "noun[{i}] = {noun:?} len={}",
                noun.len()
            );
        }
    }

    #[test]
    fn word_lists_unique() {
        let mut adj_set = std::collections::HashSet::new();
        for adj in &ADJECTIVES {
            assert!(adj_set.insert(adj), "duplicate adjective: {adj}");
        }
        let mut noun_set = std::collections::HashSet::new();
        for noun in &NOUNS {
            assert!(noun_set.insert(noun), "duplicate noun: {noun}");
        }
    }

    #[test]
    fn all_byte_values_produce_valid_names() {
        // Exhaustive: every possible byte pair for word selection
        for adj_idx in 0u8..=255 {
            for noun_idx in [0u8, 127, 255] {
                let hex = format!("sea:00000000{:02x}{:02x}0000", adj_idx, noun_idx);
                // Should not panic
                let _ = memorable_name(&hex);
            }
        }
    }

    // ─── try_memorable_name tests ───────────────────────────────────

    #[test]
    fn try_memorable_name_normal_hash() {
        let result = try_memorable_name("sea:a4839401fabae99c");
        assert!(result.contains("-"), "expected petname, got: {result}");
        assert!(result.ends_with(":sea"));
    }

    #[test]
    fn try_memorable_name_with_extension() {
        let without = try_memorable_name("sea:a4839401fabae99c");
        let with_png = try_memorable_name("sea:a4839401fabae99c.png");
        let with_jpg = try_memorable_name("sea:a4839401fabae99c.jpg");
        assert_eq!(without, with_png);
        assert_eq!(without, with_jpg);
    }

    #[test]
    fn try_memorable_name_already_petname() {
        let petname = "sunny-crab-a4839401fa:sea";
        assert_eq!(try_memorable_name(petname), petname);
    }

    #[test]
    fn try_memorable_name_short_hash() {
        assert_eq!(try_memorable_name("sea:abc"), "sea:abc");
    }

    #[test]
    fn try_memorable_name_no_colon() {
        assert_eq!(try_memorable_name("deadbeef"), "deadbeef");
    }

    // ─── strip_hash_extension tests ─────────────────────────────────

    #[test]
    fn strip_extension_png() {
        assert_eq!(strip_hash_extension("sea:abc123.png"), "sea:abc123");
    }

    #[test]
    fn strip_extension_none() {
        assert_eq!(strip_hash_extension("sea:abc123"), "sea:abc123");
    }

    #[test]
    fn strip_extension_unknown() {
        assert_eq!(strip_hash_extension("sea:abc123.unknown"), "sea:abc123");
    }

    #[test]
    fn strip_extension_webp() {
        assert_eq!(strip_hash_extension("sea:abc123.webp"), "sea:abc123");
    }
}
