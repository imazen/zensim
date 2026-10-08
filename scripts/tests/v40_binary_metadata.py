"""Reconcile current tool producers with actual bytes; history is descriptive only."""

import hashlib


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(bundle, metadata, bindings, archived_hashes):
    producer = bindings["binary_producer_commit"]
    assert metadata["build_commit"] == producer
    assert metadata["trainer_producer_commit"] == producer
    assert metadata["zensim_source_commit"] == producer
    assert set(metadata["binary_mix"]) == set(bindings["binaries"])
    assert {p for p in archived_hashes if p.startswith("bin/")} == {
        "bin/" + name
        for name in bindings["binaries"]
        if name != "predict_features_with_bake"
    }
    for name, binding in bindings["binaries"].items():
        record = metadata["binary_mix"][name]
        actual = digest(bundle / "bin" / name)
        assert record["build_commit"] == binding["producer_commit"] == producer, name
        assert record["sha256"] == binding["sha256"] == actual, name
        if name != "predict_features_with_bake":
            assert archived_hashes["bin/" + name] == actual, name
        assert record["producer_record"] == "BUILD_LOG.txt", name
        assert record["producer_record_sha256"] == digest(bundle / "BUILD_LOG.txt"), (
            name
        )
