//! Per-codec picker training data prep.
//!
//! Binary: `picker_sweep` — encode → decode → zensim_Tuner score per (source,
//! codec, q) cell, writing one parquet per codec.
//!
//! See `~/.claude/projects/-home-lilith-work-zen/memory/project_per_codec_picker_design.md`
//! for design.
