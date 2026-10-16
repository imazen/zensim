"""Independent read-only alpha/crop oracle; no image writing or tolerance."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['receipt', 'output']:
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    recipe = json.loads(args.receipt.read_text())
    crop = recipe['crop']
    source_path, prepared_path = Path(recipe['input']), Path(recipe['output'])
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == recipe['input_sha256']
    assert hashlib.sha256(prepared_path.read_bytes()).hexdigest() == recipe['output_sha256']
    with Image.open(source_path) as source, Image.open(prepared_path) as prepared:
        rgba = np.asarray(source.convert('RGBA'))
        assert prepared.mode == 'RGB'
        actual = np.asarray(prepared)
    x, y, w, h = [crop[k] for k in ['x', 'y', 'width', 'height']]
    roi = rgba[y:y+h, x:x+w]
    exact = bool(np.array_equal(actual, roi[:, :, :3]))
    opaque = bool(np.all(roi[:, :, 3] == 255))
    result = dict(recipe_sha256=hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
        input_sha256=recipe['input_sha256'], prepared_sha256=recipe['output_sha256'],
        crop=crop, dimensions_match=actual.shape == (h, w, 3),
        opaque=opaque, all_rgb_bytes_identical=exact,
        changed_channels=int(np.count_nonzero(actual != roi[:, :, :3])),
        pixels_written_by='first-party zenpng; independent oracle only reads')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    assert result['dimensions_match'] and opaque and exact, result
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
