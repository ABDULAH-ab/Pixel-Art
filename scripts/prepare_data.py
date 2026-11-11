"""Build a small training set from the Pixel Art Atlas dataset.

The full dataset has 1.4M sprites; a style LoRA needs a few hundred. This script
streams the dataset (no full download), keeps one sprite per subject for variety,
and writes them in the "imagefolder" layout the diffusers training script expects:

    data/train/00000.png
    data/train/00001.png
    data/train/metadata.jsonl   <- one {"file_name": ..., "text": ...} line per image

Usage:  python scripts/prepare_data.py --n 500
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixelart.config import DATASET, IMAGE_SIZE, lora_prompt  # noqa: E402


def sprite_to_training_image(sprite: Image.Image, size: int = IMAGE_SIZE) -> Image.Image:
    """16x16 transparent sprite -> 1024x1024 RGB image SDXL can train on.

    Two things matter here:
    1. SDXL has no transparency, so the sprite is pasted on a white background.
    2. Upscaling uses NEAREST so each sprite pixel becomes a crisp 64x64 block.
       Any smoothing filter (bilinear, bicubic) would blur the edges and the
       model would learn blurry pixel art.
    """
    sprite = sprite.convert("RGBA")
    background = Image.new("RGBA", sprite.size, (255, 255, 255, 255))
    flat = Image.alpha_composite(background, sprite).convert("RGB")
    return flat.resize((size, size), Image.Resampling.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500, help="number of training images")
    parser.add_argument("--out", type=Path, default=Path("data/train"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    from datasets import load_dataset  # imported here so the function above works without it

    args.out.mkdir(parents=True, exist_ok=True)
    rows = load_dataset(DATASET, split="train", streaming=True)
    rows = rows.shuffle(seed=args.seed, buffer_size=20_000)

    seen_subjects = set()
    with open(args.out / "metadata.jsonl", "w", encoding="utf-8") as meta:
        for row in rows:
            if row["subject"] in seen_subjects:
                continue  # one sprite per subject -> the LoRA learns the style, not one object
            seen_subjects.add(row["subject"])

            name = f"{len(seen_subjects) - 1:05d}.png"
            sprite_to_training_image(row["image"]).save(args.out / name)
            meta.write(json.dumps({"file_name": name, "text": lora_prompt(row["caption"])}) + "\n")

            if len(seen_subjects) >= args.n:
                break

    print(f"Wrote {len(seen_subjects)} images to {args.out}")


if __name__ == "__main__":
    main()
