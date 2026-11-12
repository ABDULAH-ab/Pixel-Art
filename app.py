"""Gradio demo. Runs locally with a GPU, or on Hugging Face Spaces.

Set the LORA_REPO environment variable to your LoRA's Hub repo id (or a local folder).
"""

import os

import gradio as gr
import torch

from pixelart.config import IMAGE_SIZE, lora_prompt
from pixelart.metrics import snap_to_grid
from pixelart.pipeline import CONFIGS, build_pipeline

try:
    import spaces  # only exists on Hugging Face Spaces; gives the app a GPU on demand
    gpu = spaces.GPU
    ON_SPACES = True
except ImportError:
    ON_SPACES = False

    def gpu(fn):
        return fn

LORA = os.environ.get("LORA_REPO", "unsupervised-abdullah/pixel-art-sdxl-lora")
FAST = next(c for c in CONFIGS if c.name == "lora_lcm_6step")  # the fast configuration from the benchmark

# On Spaces the GPU is only attached while generate() runs, but the model must still be
# placed on "cuda" at start-up; the `spaces` library handles the hand-over.
DEVICE = "cuda" if ON_SPACES or torch.cuda.is_available() else "cpu"
pipe = build_pipeline(FAST, LORA, DEVICE)


@gpu
def generate(subject: str, steps: int, seed: int, snap: bool):
    image = pipe(
        lora_prompt(subject),
        num_inference_steps=int(steps),
        guidance_scale=FAST.guidance,
        height=IMAGE_SIZE,
        width=IMAGE_SIZE,
        generator=torch.Generator(DEVICE).manual_seed(int(seed)),
    ).images[0]
    return snap_to_grid(image) if snap else image


demo = gr.Interface(
    fn=generate,
    inputs=[
        gr.Textbox(label="What should the sprite be?", value="a red apple"),
        gr.Slider(2, 12, value=FAST.steps, step=1, label="Steps"),
        gr.Number(value=0, precision=0, label="Seed"),
        gr.Checkbox(value=True, label="Snap to 16x16 pixel grid"),
    ],
    outputs=gr.Image(label="Sprite"),
    title="Pixel Art LoRA (fast)",
    description="SDXL with a pixel-art LoRA, sampled in a few steps with LCM.",
    examples=[["a wooden treasure chest", 6, 0, True], ["a green slime monster", 6, 1, True]],
    cache_examples=False,
)

if __name__ == "__main__":
    demo.launch()
