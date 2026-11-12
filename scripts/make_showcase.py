"""Build the README showcase image: are the outputs usable as 16x16 game sprites?

The benchmark's full-size images look very different from how a game would use
them. A game needs a tiny sprite with a transparent background that sits in an
inventory slot. This script puts the base model's output and the LoRA's output
through the same two steps and shows the result:

    1. snap to a 16x16 grid          (pixelart.metrics.snap_to_grid)
    2. remove the background colour  (remove_background, below)

No GPU needed. It reads the images scripts/benchmark.py saved.

Usage:  python scripts/make_showcase.py
        python scripts/make_showcase.py --skip "an iron sword" "a pink frosted donut"
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixelart.config import CELL, EVAL_SUBJECTS  # noqa: E402
from pixelart.metrics import snap_to_grid  # noqa: E402

SLOT = 96     # size of one inventory slot in the output image
SCALE = 5     # each sprite pixel is drawn as 5x5, so a 16x16 sprite is 80px
GAP = 4
PAD = 8
LABEL_H = 28


def to_sprite(img: Image.Image, cell: int = CELL) -> Image.Image:
    """Full-size output -> real low-resolution sprite (one pixel per grid cell)."""
    snapped = snap_to_grid(img, cell)
    return snapped.resize((snapped.width // cell, snapped.height // cell), Image.Resampling.NEAREST)


def remove_background(sprite: Image.Image, tolerance: float = 60.0) -> Image.Image:
    """Make the background transparent.

    The background colour is taken to be the typical colour along the sprite's
    border. Starting from the border, we spread inwards through every pixel close
    to that colour (a flood fill) and make those transparent. Pixels of a similar
    colour inside the object are kept, because the fill cannot reach them.
    """
    rgb = np.asarray(sprite.convert("RGB")).astype(np.float32)
    h, w, _ = rgb.shape
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    background = np.median(border, axis=0)
    similar = np.linalg.norm(rgb - background, axis=2) < tolerance

    transparent = np.zeros((h, w), dtype=bool)
    stack = [(r, c) for r in range(h) for c in (0, w - 1)] + [(r, c) for c in range(w) for r in (0, h - 1)]
    while stack:
        r, c = stack.pop()
        if 0 <= r < h and 0 <= c < w and similar[r, c] and not transparent[r, c]:
            transparent[r, c] = True
            stack += [(r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]

    alpha = np.where(transparent, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), "RGBA")


def inventory_bar(sprites: list) -> Image.Image:
    """Dark game-style inventory bar with one sprite per slot."""
    width = GAP + len(sprites) * (SLOT + GAP)
    bar = Image.new("RGB", (width, SLOT + 2 * GAP), (24, 24, 34))
    draw = ImageDraw.Draw(bar)
    for i, sprite in enumerate(sprites):
        x = GAP + i * (SLOT + GAP)
        draw.rounded_rectangle((x, GAP, x + SLOT - 1, GAP + SLOT - 1), radius=8, fill=(48, 48, 66), outline=(92, 92, 120), width=2)
        big = sprite.resize((sprite.width * SCALE, sprite.height * SCALE), Image.Resampling.NEAREST)
        bar.paste(big, (x + (SLOT - big.width) // 2, GAP + (SLOT - big.height) // 2), big)  # third argument = use alpha
    return bar


def build_showcase(base_images: list, lora_images: list, cell: int = CELL) -> Image.Image:
    rows = [
        ("Base SDXL: what it draws for each prompt",
         _strip([im.resize((SLOT, SLOT), Image.Resampling.LANCZOS) for im in base_images])),
        ("Base SDXL as 16x16 game sprites",
         inventory_bar([remove_background(to_sprite(im, cell)) for im in base_images])),
        ("Pixel-art LoRA (6 steps) as 16x16 game sprites",
         inventory_bar([remove_background(to_sprite(im, cell)) for im in lora_images])),
    ]
    try:
        font = ImageFont.load_default(size=16)
    except TypeError:  # older Pillow
        font = ImageFont.load_default()

    width = max(row.width for _, row in rows) + 2 * PAD
    height = PAD + sum(LABEL_H + row.height + PAD for _, row in rows)
    out = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(out)
    y = PAD
    for label, row in rows:
        draw.text((PAD, y + 4), label, fill=(30, 30, 30), font=font)
        y += LABEL_H
        out.paste(row, (PAD, y))
        y += row.height + PAD
    return out


def _strip(images: list) -> Image.Image:
    """Images side by side with the same spacing as the inventory bar."""
    strip = Image.new("RGB", (GAP + len(images) * (SLOT + GAP), SLOT + 2 * GAP), "white")
    for i, im in enumerate(images):
        strip.paste(im, (GAP + i * (SLOT + GAP), GAP))
    return strip


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("outputs/benchmark/images"))
    parser.add_argument("--base", default="base_30step")
    parser.add_argument("--lora", default="lora_lcm_6step_taesd")
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark/showcase.png"))
    parser.add_argument("--skip", nargs="*", default=[], help="prompts to leave out of the image")
    args = parser.parse_args()

    unknown = [s for s in args.skip if s not in EVAL_SUBJECTS]
    if unknown:
        parser.error(f"not in EVAL_SUBJECTS: {unknown}")

    # One image per distinct prompt (the prompt list may contain repeats), minus the skipped ones.
    # The benchmark saved its images by position in EVAL_SUBJECTS, so that list must not be
    # edited to choose what appears here: use --skip instead.
    indices = [EVAL_SUBJECTS.index(s) for s in dict.fromkeys(EVAL_SUBJECTS) if s not in args.skip]

    def load(config: str) -> list:
        return [Image.open(args.images / config / f"{i:02d}.png").convert("RGB") for i in indices]

    build_showcase(load(args.base), load(args.lora)).save(args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
