"""Read-only independent PNG byte-container and logical float-map oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--judges', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--lossless-png-expansion', action='store_true')
    args = parser.parse_args()
    def rgb_image(image):
        if args.lossless_png_expansion:
            assert image.mode in ['RGB','RGBA','L','LA','P','1'], 'unsupported precision'
            rgba=image.convert('RGBA')
            assert rgba.getchannel('A').getextrema()==(255,255)
            return rgba.convert('RGB')
        assert image.mode=='RGB'
        return image
    checked = {}
    rows = [json.loads(line) for line in (args.judges/'judges.jsonl').read_text().splitlines()]
    assert json.loads((args.judges/'COMPLETE.json').read_text())['pairs'] == len(rows)
    for row in rows:
        assert {a['metric_id'].split(';')[0] for a in row['maps']} == {'butteraugli','zensim-B-default-diffmap'}
        with Image.open(row['ref_path']) as image:
            assert image.size == (row['width'], row['height'])
            reference = rgb_image(image).tobytes()
        with Image.open(row['dist_path']) as image:
            assert image.size == (row['width'], row['height'])
            distorted = rgb_image(image).tobytes()
        assert hashlib.sha256(reference).hexdigest() == row['reference_pixels_sha256']
        assert hashlib.sha256(distorted).hexdigest() == row['distorted_pixels_sha256']
        for artifact in row['maps']:
            path = Path(artifact['path'])
            dedup = hashlib.sha256(reference + distorted + artifact['metric_id'].encode()).hexdigest()
            assert dedup == artifact['dedup_key']
            if path in checked:
                physical_fields = ['format','width','height','storage_rgb_width','stored_bytes','raw_smap1_bytes','logical_smap1_sha256','map_sha256']
                assert all(artifact[k] == checked[path][k] for k in physical_fields), 'inconsistent physical map metadata'
                continue
            stored = path.read_bytes()
            assert len(stored) == artifact['stored_bytes']
            assert hashlib.sha256(stored).hexdigest() == artifact['map_sha256']
            w, h = artifact['width'], artifact['height']
            with Image.open(path) as image:
                assert image.size == (artifact['storage_rgb_width'], h)
                packed = np.asarray(rgb_image(image)).reshape(h, -1)
            assert np.all(packed[:, 4*w:] == 0)
            payload = packed[:, :4*w].copy().tobytes()
            assert np.isfinite(np.frombuffer(payload, dtype='<f4')).all()
            logical = b'SMAP1\0' + struct.pack('<QQ', w, h) + payload
            assert len(logical) == artifact['raw_smap1_bytes']
            assert hashlib.sha256(logical).hexdigest() == artifact['logical_smap1_sha256']
            checked[path] = artifact
    result = dict(oracle_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        png_decode_policy='lossless PNG sample expansion to RGB8' if args.lossless_png_expansion else 'strict RGB storage',
        judges_jsonl_sha256=hashlib.sha256((args.judges/'judges.jsonl').read_bytes()).hexdigest(),
        judge_rows=len(rows), unique_maps=len(checked),
        all_physical_hashes_verified=True, all_logical_hashes_verified=True,
        all_float_values_finite=True, all_padding_zero=True,
        all_input_pixel_hashes_verified=True, all_map_dedup_keys_verified=True,
        independent_reader='Pillow/NumPy, reads only')
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
