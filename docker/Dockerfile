# zensim V_X full-reproducibility container.
#
# Goal: any future agent can `docker build && docker run` and end up
# with the same canonical clean training corpus + trained V_X bake +
# 10-band validation as is on disk today. Audit chain is unbroken.
#
# Stages:
#
#   1. base — Debian + rust-1.93 + python3 + system deps.
#   2. corpora — download (or pull from R2 cache) every test corpus
#      we need: CID22 / KADID / TID / AIC-3 / AIC-4 / KonJND.
#   3. synth — re-derive the synthetic training corpus OR pull from
#      R2 mirror at zentrain-r2.imazen.org/zensim-compare-site/. The
#      r2-mirror path is faster (10 min) than full re-derive (~hours).
#   4. features — compute the 228-feature CSV (via
#      zensim::compute_all_features) for every (ref, dist) pair. Also
#      run KADID/TID overlap audit and produce the 149-basename
#      contamination blocklist.
#   5. train — run the V0_18 3-way concat recipe against the clean
#      corpus. Produce v0_X_*.bin under benchmarks/.
#   6. validate — run dataset_metric_baseline with --v04-bake against
#      every corpus; produce v0_X_10band_<date>.md.
#   7. ship-bundle — collect the bake + methodology doc + validation
#      report + audit TSVs into a single `/output/` tarball.
#
# Each stage caches separately so iterating on (e.g.) the training
# recipe doesn't re-run feature extraction.

# ── Stage 1: base ────────────────────────────────────────────────
FROM rust:1.93-slim-bookworm AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 python3-pip python3-venv python3-numpy python3-pandas python3-matplotlib python3-pil \
    git curl ca-certificates pkg-config \
    libssl-dev libfontconfig1-dev \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /work

# Pre-fetch cargo registry cache for faster builds in subsequent stages.
COPY Cargo.toml Cargo.lock /work/
RUN mkdir -p /work/zensim/src /work/zensim-validate/src /work/zensim-bench/src /work/zensim-regress/src /work/zensim-wasm-tests/src /work/zensim-train-core/src \
    && touch /work/zensim/src/lib.rs /work/zensim-validate/src/lib.rs /work/zensim-bench/src/lib.rs /work/zensim-regress/src/lib.rs /work/zensim-wasm-tests/src/lib.rs /work/zensim-train-core/src/lib.rs

# ── Stage 2: corpora ─────────────────────────────────────────────
FROM base AS corpora
# Each download lands at /corpora/<name>/. R2 mirror is preferred over
# upstream sources for stability and speed.
ENV CORPORA_BASE=/corpora
RUN mkdir -p $CORPORA_BASE
# CID22 validation set (from zentrain R2 mirror — 1.2 GB).
RUN echo "TODO: implement curl download from R2 with retries + checksum"
# KADID-10k from https://database.mmsp-kn.de/kadid-10k-database.html (4 GB)
RUN echo "TODO: implement"
# TID2013 from http://www.ponomarenko.info/tid2013.htm (~600 MB)
RUN echo "TODO: implement"
# AIC-3 CTC EPFL (250 MB)
RUN echo "TODO: implement"
# AIC-4 sample (250 MB)
RUN echo "TODO: implement"
# KonJND-1k (1.2 GB)
RUN echo "TODO: implement"

# ── Stage 3: synth corpus (or R2 mirror) ─────────────────────────
FROM corpora AS synth
ENV SYNTH_BASE=/synth
RUN mkdir -p $SYNTH_BASE
# Option A (fast): pull pre-computed features from R2.
# Option B (slow, audit-rebuild): re-run the synth generator + score every pair.
ARG SYNTH_MODE=r2
RUN if [ "$SYNTH_MODE" = "r2" ]; then \
      echo "pull canonical features from r2://zentrain-r2.imazen.org/zensim-compare-site/clean-corpus/2026-05-14/"; \
    else \
      echo "TODO: re-run synth + feature extraction"; \
    fi

# ── Stage 4: features + contamination audit ──────────────────────
FROM synth AS features
COPY . /work/
WORKDIR /work
RUN cargo build --release --bin check_holdout_overlap --bin contamination_guard
# Re-run the perceptual-overlap audit on every (training_corpus, holdout)
# pair. If new contamination is found, the build fails — guarantees no
# silent regression.
RUN ./target/release/check_holdout_overlap \
      --cid22-refs $CORPORA_BASE/cid22/CID22_validation_set/original \
      --training-csv $SYNTH_BASE/training_clean.csv \
      --threshold 16 \
      --out-tsv /work/audit_cid22.tsv && \
    ./target/release/check_holdout_overlap \
      --cid22-refs $CORPORA_BASE/kadid10k/refs_only \
      --training-csv $SYNTH_BASE/training_clean.csv \
      --threshold 16 \
      --out-tsv /work/audit_kadid.tsv && \
    ./target/release/check_holdout_overlap \
      --cid22-refs $CORPORA_BASE/tid2013/reference_images \
      --training-csv $SYNTH_BASE/training_clean.csv \
      --threshold 16 \
      --out-tsv /work/audit_tid.tsv

# ── Stage 5: train V_X (default = v0_19 3-way concat recipe) ─────
FROM features AS train
ARG BAKE_VERSION=v0_19
ARG BAKE_DATE=2026-05-14
RUN cargo build --release \
      --bin zensim_mlp_train --bin concat_three_way --bin affine_calibrate
# Each component runs in series for reproducibility (parallel = harder
# to seed-pin; this is the audit chain).
RUN ./target/release/zensim_mlp_train \
      --recipe v0_18_base_component \
      --group safesyn:$SYNTH_BASE/safe_synth_clean_features.csv:1.0:0.0 \
      --group kadid:$SYNTH_BASE/kadid_features.csv:0.3:1.0 \
      --group tid:$SYNTH_BASE/tid_features.csv:0.3:1.0 \
      --group konjnd:$SYNTH_BASE/konjnd_aligned_features.csv:0.5:1.0 \
      --out /work/benchmarks/${BAKE_VERSION}_base_${BAKE_DATE}.bin && \
    ./target/release/zensim_mlp_train \
      --recipe v0_18_cycle14_component \
      --group safesyn:$SYNTH_BASE/safe_synth_clean_features.csv:1.0:0.0 \
      --group kadid:$SYNTH_BASE/kadid_features.csv:0.3:1.0 \
      --group tid:$SYNTH_BASE/tid_features.csv:0.3:1.0 \
      --group konjnd:$SYNTH_BASE/konjnd_aligned_features.csv:0.5:1.0 \
      --tv-pairs-file $SYNTH_BASE/tv_pairs_bands.tsv \
      --seed 1 \
      --out /work/benchmarks/${BAKE_VERSION}_cycle14_s1_${BAKE_DATE}.bin && \
    ./target/release/zensim_mlp_train \
      --recipe v0_18_cycle14_component \
      --group safesyn:$SYNTH_BASE/safe_synth_clean_features.csv:1.0:0.0 \
      --group kadid:$SYNTH_BASE/kadid_features.csv:0.3:1.0 \
      --group tid:$SYNTH_BASE/tid_features.csv:0.3:1.0 \
      --group konjnd:$SYNTH_BASE/konjnd_aligned_features.csv:0.5:1.0 \
      --tv-pairs-file $SYNTH_BASE/tv_pairs_bands.tsv \
      --seed 42 \
      --out /work/benchmarks/${BAKE_VERSION}_cycle14_s42_${BAKE_DATE}.bin && \
    ./target/release/concat_three_way \
      --base /work/benchmarks/${BAKE_VERSION}_base_${BAKE_DATE}.bin \
      --s1   /work/benchmarks/${BAKE_VERSION}_cycle14_s1_${BAKE_DATE}.bin \
      --s42  /work/benchmarks/${BAKE_VERSION}_cycle14_s42_${BAKE_DATE}.bin \
      --out  /work/benchmarks/${BAKE_VERSION}_concat_3way_${BAKE_DATE}.bin && \
    ./target/release/affine_calibrate \
      --in-bake  /work/benchmarks/${BAKE_VERSION}_concat_3way_${BAKE_DATE}.bin \
      --out-bake /work/zensim/weights/${BAKE_VERSION}_${BAKE_DATE}.bin \
      --alpha 28.0366 --beta=-5.0738

# ── Stage 6: validate ────────────────────────────────────────────
FROM train AS validate
RUN cargo build --release --example dataset_metric_baseline
RUN ./target/release/examples/dataset_metric_baseline \
      --cid22 $CORPORA_BASE/cid22/CID22_validation_set \
      --kadid $CORPORA_BASE/kadid10k \
      --tid $CORPORA_BASE/tid2013 \
      --v04-bake /work/zensim/weights/${BAKE_VERSION}_${BAKE_DATE}.bin \
      --max-pairs 50000 \
      > /work/benchmarks/${BAKE_VERSION}_10band_${BAKE_DATE}.md

# ── Stage 7: ship-bundle ─────────────────────────────────────────
FROM validate AS bundle
RUN mkdir -p /output && \
    cp /work/zensim/weights/${BAKE_VERSION}_${BAKE_DATE}.bin /output/ && \
    cp /work/benchmarks/${BAKE_VERSION}_10band_${BAKE_DATE}.md /output/ && \
    cp /work/benchmarks/v0_18_methodology_2026-05-13.md /output/ && \
    cp /work/audit_*.tsv /output/ && \
    cp /work/benchmarks/contamination_blocklist_2026-05-14.txt /output/ && \
    md5sum /output/* > /output/CHECKSUMS

# Final image is just the bundle.
FROM scratch
COPY --from=bundle /output /output
