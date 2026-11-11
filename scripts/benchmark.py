"""Benchmark every configuration on the same prompts and seeds.

For each configuration it records:
  - latency      seconds per image (after a warm-up run)
  - peak VRAM    highest GPU memory used while generating
  - CLIP score   does the image match the prompt? (higher is better)
  - grid error   is it on a clean pixel grid? (lower is better)
  - palette      number of distinct colours (lower is more sprite-like)

Outputs go to outputs/benchmark/: results.json, results.md and one image grid per config.

Usage:  python scripts/benchmark.py --lora outputs/pixel-lora
"""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pixelart.config import EVAL_SUBJECTS, IMAGE_SIZE, base_prompt, lora_prompt  # noqa: E402
from pixelart.metrics import grid_error, palette_size  # noqa: E402
from pixelart.pipeline import CONFIGS, build_pipeline  # noqa: E402


class ClipScorer:
    """CLIP embeds images and text in the same space; cosine similarity says how well they match."""

    def __init__(self, device: str):
        name = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(name).to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(name)
        self.device = device

    @torch.no_grad()
    def score(self, image: Image.Image, text: str) -> float:
        inputs = self.processor(text=[text], images=image, return_tensors="pt", padding=True).to(self.device)
        out = self.model(**inputs)
        img = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
        txt = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
        return float((img * txt).sum()) * 100


def make_grid(images: list, thumb: int = 256) -> Image.Image:
    """Put images side by side in one row. NEAREST keeps pixel edges crisp when shrinking."""
    grid = Image.new("RGB", (thumb * len(images), thumb), "white")
    for i, img in enumerate(images):
        grid.paste(img.resize((thumb, thumb), Image.Resampling.NEAREST), (i * thumb, 0))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora", required=True, help="local folder or Hub repo id of the trained LoRA")
    parser.add_argument("--out", type=Path, default=Path("outputs/benchmark"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda"
    args.out.mkdir(parents=True, exist_ok=True)
    clip = ClipScorer(device)
    results = []

    for cfg in CONFIGS:
        print(f"== {cfg.name}")
        pipe = build_pipeline(cfg, args.lora, device)
        make_prompt = lora_prompt if cfg.use_lora else base_prompt

        def generate(subject: str, seed: int) -> Image.Image:
            return pipe(
                make_prompt(subject),
                num_inference_steps=cfg.steps,
                guidance_scale=cfg.guidance,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE,
                generator=torch.Generator(device).manual_seed(seed),
            ).images[0]

        # The first call is slow (CUDA setup, memory allocation), so it isn't timed.
        generate(EVAL_SUBJECTS[0], args.seed)
        torch.cuda.reset_peak_memory_stats()

        images, latencies = [], []
        for i, subject in enumerate(EVAL_SUBJECTS):
            # GPU calls are asynchronous: synchronize() waits for them to finish,
            # otherwise the timer would stop before the work is actually done.
            torch.cuda.synchronize()
            start = time.perf_counter()
            image = generate(subject, args.seed + i)
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - start)
            images.append(image)

        results.append({
            "config": cfg.name,
            "steps": cfg.steps,
            "latency_s": sum(latencies) / len(latencies),
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            # CLIP is scored against a plain description so every config is judged on the same text.
            "clip_score": sum(clip.score(im, f"pixel art of {s}") for im, s in zip(images, EVAL_SUBJECTS)) / len(images),
            "grid_error": sum(grid_error(im) for im in images) / len(images),
            "palette_size": sum(palette_size(im) for im in images) / len(images),
        })
        make_grid(images).save(args.out / f"grid_{cfg.name}.png")

        # Free the GPU before loading the next configuration.
        del pipe
        gc.collect()
        torch.cuda.empty_cache()

    baseline = results[0]["latency_s"]
    lines = [
        "| Config | Steps | Latency (s) | Speed-up | Peak VRAM (GB) | CLIP score | Grid error | Palette size |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | {r['steps']} | {r['latency_s']:.2f} | {baseline / r['latency_s']:.1f}x "
            f"| {r['peak_vram_gb']:.1f} | {r['clip_score']:.1f} | {r['grid_error']:.4f} | {r['palette_size']:.0f} |"
        )
    table = "\n".join(lines)

    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    (args.out / "results.md").write_text(table + "\n")
    print(table)


if __name__ == "__main__":
    main()
