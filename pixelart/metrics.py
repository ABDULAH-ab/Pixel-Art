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


def _best_grid_offset(arr: np.ndarray, cell: int) -> tuple:
    """Grid position (dy, dx) in pixels that gives the lowest grid error.

    Works on a 4x smaller copy so trying many positions stays fast.
    """
    shrink = 4 if cell % 4 == 0 else 1
    small = arr[::shrink, ::shrink].astype(np.float32)
    c = cell // shrink

    best_error, best = None, (0, 0)
    for dy in range(0, c, 2):
        for dx in range(0, c, 2):
            shifted = np.roll(small, (-dy, -dx), axis=(0, 1))
            means = _cell_means(shifted, c)
            error = np.abs(shifted - means.repeat(c, axis=0).repeat(c, axis=1)).mean()
            # Only move away from the default position for a clear improvement.
            if best_error is None or error < best_error * 0.95:
                best_error, best = error, (dy * shrink, dx * shrink)
    return best


def snap_to_grid(img: Image.Image, cell: int = CELL) -> Image.Image:
    """Post-process: force the image onto the pixel grid.

    Two steps:
    1. Find where the model's blocks actually sit. The model draws blocks of the
       right size but not always starting at pixel 0, so we slide the grid and
       keep the position where the image is most block-like (lowest grid error).
    2. Fill every cell with its most common colour. Unlike averaging, this never
       invents blended in-between colours.

    The result is a true sprite (grid error exactly 0).
    """
    arr = np.asarray(img.convert("RGB"))
    h, w = (arr.shape[0] // cell) * cell, (arr.shape[1] // cell) * cell
    arr = arr[:h, :w]

    dy, dx = _best_grid_offset(arr, cell)
    arr = np.roll(arr, (-dy, -dx), axis=(0, 1))  # shift the image so its blocks line up with the grid

    # (rows, cols, pixels-in-cell, 3)
    cells = arr.reshape(h // cell, cell, w // cell, cell, 3).transpose(0, 2, 1, 3, 4)
    cells = cells.reshape(h // cell, w // cell, cell * cell, 3)

    # Group near-identical shades by rounding to 5 bits per channel.
    coarse = cells >> 3
    keys = (coarse[..., 0].astype(np.int32) << 10) | (coarse[..., 1].astype(np.int32) << 5) | coarse[..., 2]

    small = np.zeros((h // cell, w // cell, 3), dtype=np.uint8)
    for r in range(h // cell):
        for c in range(w // cell):
            values, counts = np.unique(keys[r, c], return_counts=True)
            winner = keys[r, c] == values[counts.argmax()]
            small[r, c] = cells[r, c][winner].mean(axis=0).round()

    return Image.fromarray(small).resize((w, h), Image.Resampling.NEAREST)
