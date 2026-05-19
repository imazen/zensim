#!/usr/bin/env python3
"""Affine-calibrate a per-sample-α head bake's metadata payload (2026-05-19).

For a bake whose final score is `y = α·y_rank + (1-α)·y_pool`, the affine
`y' = α0 + β·y` is equivalent to scaling rank_w / reducer_w by β AND
shifting rank_b / reducer_b by `β·b_old + (α0)` each — since the mix is
linear in y_rank / y_pool and α is sample-dependent.

This script reads the bake's `zentrain.per_sample_alpha_head` metadata
payload (192 + 4 + 192 + 4 + 16 + 4 + 4 = 416 bytes for n_hidden=128
... wait, that's wrong. Let me re-derive from the source.

The payload from `bake_per_sample_alpha_head_v3` is:
  W_α[n_hidden] | b_α | rank_w[n_hidden] | rank_b | reducer_w[4] |
    reducer_b | p_norm   = 4 · (2·n_hidden + 8) bytes

For n_hidden=128: 4·(256 + 8) = 1056 bytes payload.

We modify rank_w, rank_b, reducer_w, reducer_b in-place:
  rank_w'    = β · rank_w
  rank_b'    = β · rank_b + α0
  reducer_w' = β · reducer_w
  reducer_b' = β · reducer_b + α0

This makes the runtime-computed y_rank and y_pool both shift by α0 and
scale by β; the per-sample-α mix `y = α·y_rank + (1-α)·y_pool` then has
y' = α0 + β·y. W_α, b_α, p_norm are left unchanged.

Direct byte-rewrite — we don't re-serialize the bake's wire format,
just patch the metadata block's payload bytes.
"""

import argparse
import struct
import sys
from pathlib import Path

METADATA_KEY = b"zentrain.per_sample_alpha_head"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-bake", required=True)
    ap.add_argument("--out-bake", required=True)
    ap.add_argument("--alpha", type=float, required=True, help="affine offset α0 in y' = α0 + β·y")
    ap.add_argument("--beta", type=float, required=True, help="affine scale β in y' = α0 + β·y")
    ap.add_argument("--n-hidden", type=int, default=128)
    args = ap.parse_args()

    data = bytearray(Path(args.in_bake).read_bytes())
    n = args.n_hidden
    payload_floats = 2 * n + 8
    payload_bytes = payload_floats * 4

    # Locate the metadata key in the file. The wire format follows the
    # key with a kind byte + 4-byte little-endian length + the payload.
    # See zenpredict-bake's emit for exact layout — here we just scan for
    # the key and locate the payload via the length prefix.
    idx = data.find(METADATA_KEY)
    if idx < 0:
        print(f"FATAL: metadata key {METADATA_KEY!r} not found", file=sys.stderr)
        return 1
    # After the key, there's some framing. In ZNPR v3, metadata entries
    # are typically packed as [key_len:u32 | key | kind:u8 | val_len:u32 | val].
    # The payload (val) is `payload_bytes` long and starts at some offset
    # AFTER the key. Probe by looking for `payload_bytes` length in the
    # next 16 bytes.
    found_payload_at = None
    for probe in range(idx + len(METADATA_KEY), idx + len(METADATA_KEY) + 32):
        if probe + 4 > len(data):
            break
        length = struct.unpack_from("<I", data, probe)[0]
        if length == payload_bytes:
            found_payload_at = probe + 4
            break
    if found_payload_at is None:
        # Fallback: the payload SHOULD be at the end of the bake (since it
        # ends the metadata section in our layout, and the bake we emit
        # always has exactly one metadata entry).
        candidate = len(data) - payload_bytes
        # Verify by checking the first 128 W_α floats are reasonable (near
        # zero or tiny — they are by construction in our bake).
        first_floats = struct.unpack_from(f"<{n}f", data, candidate)
        if all(abs(v) < 1.0 for v in first_floats):
            found_payload_at = candidate
        else:
            print(
                f"FATAL: could not locate payload; tried both prefix-length scan "
                f"and end-of-file. first float at fallback={first_floats[0]}",
                file=sys.stderr,
            )
            return 2
    print(f"payload offset: {found_payload_at} (payload size = {payload_bytes} bytes)", file=sys.stderr)

    # Layout: [W_α(n) | b_α | rank_w(n) | rank_b | reducer_w(4) | reducer_b | p_norm]
    off = found_payload_at
    floats = list(struct.unpack_from(f"<{payload_floats}f", data, off))
    w_alpha = floats[:n]
    b_alpha = floats[n]
    rank_w_start = n + 1
    rank_w = floats[rank_w_start : rank_w_start + n]
    rank_b = floats[rank_w_start + n]
    reducer_w_start = rank_w_start + n + 1
    reducer_w = floats[reducer_w_start : reducer_w_start + 4]
    reducer_b = floats[reducer_w_start + 4]
    p_norm = floats[reducer_w_start + 5]

    print(f"BEFORE: rank_b={rank_b:.4f} reducer_b={reducer_b:.4f}", file=sys.stderr)

    # Apply affine y' = α0 + β·y.
    new_rank_w = [r * args.beta for r in rank_w]
    new_rank_b = args.beta * rank_b + args.alpha
    new_reducer_w = [r * args.beta for r in reducer_w]
    new_reducer_b = args.beta * reducer_b + args.alpha

    # Pack back into floats.
    new_floats = (
        list(w_alpha)
        + [b_alpha]
        + new_rank_w
        + [new_rank_b]
        + new_reducer_w
        + [new_reducer_b]
        + [p_norm]
    )
    assert len(new_floats) == payload_floats
    struct.pack_into(f"<{payload_floats}f", data, off, *new_floats)

    print(
        f"AFTER:  rank_b={new_rank_b:.4f} reducer_b={new_reducer_b:.4f} "
        f"(α0={args.alpha:.3f}, β={args.beta:.3f})",
        file=sys.stderr,
    )

    Path(args.out_bake).write_bytes(bytes(data))
    print(f"wrote {args.out_bake}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
