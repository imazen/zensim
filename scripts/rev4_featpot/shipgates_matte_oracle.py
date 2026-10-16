"""Independent float64 IEC sRGB source-over oracle; never writes images."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["input", "prepared", "output"]:
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    with Image.open(args.input) as source, Image.open(args.prepared) as prepared:
        assert source.size == prepared.size and prepared.mode == "RGB"
        rgba = np.asarray(source.convert("RGBA"))
        actual = np.asarray(prepared)
    encoded = rgba[:, :, :3].astype(np.float64) / 255.
    alpha = rgba[:, :, 3:].astype(np.float64) / 255.
    linear = np.where(encoded <= .04045, encoded / 12.92,
                      ((encoded + .055) / 1.055) ** 2.4)
    composed = linear * alpha + (1. - alpha)
    expected = np.where(composed <= .0031308, composed * 12.92,
                        1.055 * composed ** (1. / 2.4) - .055)
    expected = np.floor(expected * 255. + .5).astype(np.uint8)
    delta = actual.astype(np.int16) - expected.astype(np.int16)
    result = dict(reference="IEC61966-2-1 sRGB float64 source-over white; half-up8bit",
        input_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        prepared_sha256=hashlib.sha256(args.prepared.read_bytes()).hexdigest(),
        prepared_rgb_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),
        changed_channels=int(np.count_nonzero(delta)), maximum_abs_channel_error=int(np.abs(delta).max()),
        nonopaque_pixels=int(np.count_nonzero(rgba[:, :, 3] != 255)),
        opaque_pixels_unchanged=bool(np.array_equal(actual[rgba[:, :, 3] == 255], rgba[:, :, :3][rgba[:, :, 3] == 255])),
        transparent_pixels_white=bool(np.all(actual[rgba[:, :, 3] == 0] == 255)))
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, allow_nan=False)
        out.write("\n")
    print(json.dumps(result, indent=2), flush=True)
    assert result["changed_channels"] == 0 and result["opaque_pixels_unchanged"] and result["transparent_pixels_white"]


if __name__ == "__main__":
    main()
