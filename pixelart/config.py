"""Shared settings used by data prep, benchmarking and the demo app."""

# Models (all downloaded from the Hugging Face Hub)
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
VAE_FP16 = "madebyollin/sdxl-vae-fp16-fix"   # SDXL's own VAE produces NaNs in half precision; this one doesn't
TINY_VAE = "madebyollin/taesdxl"             # tiny distilled decoder: much faster, slightly lower quality
LCM_LORA = "latent-consistency/lcm-lora-sdxl"  # distillation LoRA that lets SDXL sample in 4-8 steps

# Data
DATASET = "Angais/pixel-art-atlas-1.4m"  # 16x16 RGBA sprites with captions, CC0
SPRITE_SIZE = 16                         # sprites are 16x16 pixels
IMAGE_SIZE = 1024                        # SDXL's native resolution
CELL = IMAGE_SIZE // SPRITE_SIZE         # one sprite pixel = a 64x64 block in the 1024px image

# A made-up word the LoRA learns to associate with the style.
# Using a rare token keeps the style from leaking into ordinary prompts.
TRIGGER = "pxlsprite"


def lora_prompt(subject: str) -> str:
    """Prompt format the LoRA was trained on."""
    return f"{TRIGGER}, {subject}"


def base_prompt(subject: str) -> str:
    """Best-effort prompt for the un-tuned base model (it doesn't know the trigger word)."""
    return f"pixel art sprite of {subject}, 16x16, white background"


# Fixed prompts used for every benchmark configuration, so results are comparable.
EVAL_SUBJECTS = [
    "a red apple",
    "a wooden treasure chest",
    "a blue potion bottle",
    "a golden key",
    "a green slime monster",
    "an iron sword",
    "a small pine tree",
    "a pink frosted donut",
    "a green slime monster",
]
