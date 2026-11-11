"""Pixel-art specific quality metrics. Only needs NumPy and Pillow.

Idea: real pixel art is made of flat square cells. If we average every cell and
paint the average back, a true pixel-art image doesn't change at all, while a
blurry "pixel-ish" image changes a lot. That difference is the grid error.
"""

import numpy as np
from PIL import Image

from .config import CELL


def _to_array(img: Image.Image, cell: int) -> np.ndarray:
    """RGB float array in [0, 1], cropped so both sides are a multiple of `cell`."""
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    h, w = (arr.shape[0] // cell) * cell, (arr.shape[1] // cell) * cell
    return arr[:h, :w]


def _cell_means(arr: np.ndarray, cell: int) -> np.ndarray:
    """Average colour of every cell: (H, W, 3) -> (H/cell, W/cell, 3)."""
    h, w, c = arr.shape
    return arr.reshape(h // cell, cell, w // cell, cell, c).mean(axis=(1, 3))


def grid_error(img: Image.Image, cell: int = CELL) -> float:
    """Mean absolute difference between the image and its 'blockified' version.

    0.0 means every cell is one flat colour (perfect pixel grid). Higher is worse.
    Limitation: a completely blank image also scores 0, so read this next to
    CLIP score, never on its own.
    """
    arr = _to_array(img, cell)
    means = _cell_means(arr, cell)
    blocky = means.repeat(cell, axis=0).repeat(cell, axis=1)
    return float(np.abs(arr - blocky).mean())


def palette_size(img: Image.Image, cell: int = CELL, bits: int = 4) -> int:
    """Number of distinct colours once the image is reduced to one colour per cell.

    Colours are rounded to `bits` bits per channel so near-identical shades count
    as one. Pixel art uses small palettes, so lower is more sprite-like.
    """
    means = _cell_means(_to_array(img, cell), cell)
    quantised = (means * 255).astype(np.uint8) >> (8 - bits)
    return int(len(np.unique(quantised.reshape(-1, 3), axis=0)))


def snap_to_grid(img: Image.Image, cell: int = CELL) -> Image.Image:
    """Post-process: force the image onto the pixel grid.

    Shrinks to one pixel per cell, then scales back up without smoothing, so the
    result is a true sprite (grid error exactly 0).
    """
    w, h = img.size
    small = img.convert("RGB").resize((w // cell, h // cell), Image.Resampling.NEAREST)
    return small.resize((w, h), Image.Resampling.NEAREST)
