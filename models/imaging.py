"""Image loading helpers shared by every model.

Browsers draw an <img> with its EXIF orientation applied. Models, however, see the raw pixel grid.
If the two disagree (typical for phone photos) normalised boxes land in the wrong place. All analysis
therefore runs on the EXIF-transposed RGB image; the stored ORIGINAL file is never modified.
"""
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 100_000_000  # decompression-bomb guard (PIL raises above 2x this)


def load_rgb(path: str) -> Image.Image:
    with Image.open(path) as im:
        im.load()
        return ImageOps.exif_transpose(im).convert("RGB")
