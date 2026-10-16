"""Lossless PNG representation adapter; first-party writers, independent readers."""
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pixels(path):
    with Image.open(path) as image:
        assert image.mode in ['RGB','RGBA','L','LA','P','1'], 'unsupported precision'
        rgba=image.convert('RGBA')
        assert rgba.getchannel('A').getextrema()==(255,255), 'opaque pixels required'
        return image.size,rgba.convert('RGB').tobytes()


def canonical_pair(source,native_reference,decoded,destination,binary):
    """Preserve every delivered RGB byte, including palette/gray PNG storage."""
    source,native_reference,decoded=map(Path,[source,native_reference,decoded])
    reference_pixels=pixels(source)
    assert pixels(native_reference)==reference_pixels, 'native reference differs from admitted source'
    metadata=dict(reference_png_sha256=sha(source),native_reference=str(native_reference),
        native_reference_png_sha256=sha(native_reference))
    with Image.open(decoded) as image:
        needs_conversion=image.mode!='RGB'
    if needs_conversion:
        original_pixels=pixels(decoded)
        target=Path(destination)/'decoded-canonical-png8.png'
        recipe=Path(destination)/'decoded-canonical-png8.json'
        with recipe.open('x') as stream:
            subprocess.run([str(binary),'--expand-png8',str(decoded),str(target)],stdout=stream,check=True)
        result=json.loads(recipe.read_text())
        assert result['source_png_bit_depth'] in [1,2,4,8]
        assert result['output_png_bit_depth']==8 and result['output_png_color_type'] in [0,2]
        assert result['crop']==dict(x=0,y=0,width=original_pixels[0][0],height=original_pixels[0][1])
        assert pixels(target)==original_pixels, 'PNG canonicalization changed delivered pixels'
        metadata.update(native_decoded=str(decoded),native_decoded_png_sha256=sha(decoded),
            png8_recipe=str(recipe),png8_recipe_sha256=sha(recipe),png8_binary_sha256=sha(binary))
        decoded=target
    return source,decoded,metadata
