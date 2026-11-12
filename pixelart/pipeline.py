"""Builds the SDXL pipeline in each configuration we benchmark."""

from dataclasses import dataclass

import torch
from diffusers import AutoencoderKL, AutoencoderTiny, LCMScheduler, StableDiffusionXLPipeline

from .config import BASE_MODEL, LCM_LORA, TINY_VAE, VAE_FP16


@dataclass
class RunConfig:
    name: str
    use_lora: bool      # load our pixel-art LoRA?
    use_lcm: bool       # add the LCM distillation LoRA + scheduler for few-step sampling?
    tiny_vae: bool      # swap the VAE decoder for the tiny one?
    steps: int          # denoising steps
    guidance: float     # classifier-free guidance scale. 1.0 turns CFG off, which
                        # halves the UNet work per step (one forward pass instead of two)


# Each row changes one thing, so the table shows what every optimisation buys.
CONFIGS = [
    RunConfig("base_30step",         use_lora=False, use_lcm=False, tiny_vae=False, steps=30, guidance=7.0),
    RunConfig("lora_30step",         use_lora=True,  use_lcm=False, tiny_vae=False, steps=30, guidance=7.0),
    RunConfig("lora_lcm_6step",      use_lora=True,  use_lcm=True,  tiny_vae=False, steps=6,  guidance=1.0),
    RunConfig("lora_lcm_6step_taesd", use_lora=True, use_lcm=True,  tiny_vae=True,  steps=6,  guidance=1.0),
]


def build_pipeline(cfg: RunConfig, lora: str, device: str = "cuda") -> StableDiffusionXLPipeline:
    """`lora` is a local folder or a Hub repo id containing the trained LoRA weights."""
    dtype = torch.float16  # half precision: half the memory, faster on GPU

    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL,
        vae=AutoencoderKL.from_pretrained(VAE_FP16, torch_dtype=dtype),
        torch_dtype=dtype,
        variant="fp16",
        use_safetensors=True,
    )
    if cfg.tiny_vae:
        pipe.vae = AutoencoderTiny.from_pretrained(TINY_VAE, torch_dtype=dtype)

    # LoRAs are small add-on weight matrices. Several can be active at once:
    # ours supplies the style, the LCM one supplies few-step sampling.
    adapters = []
    if cfg.use_lora:
        # Name the file explicitly: the Hub repo also holds intermediate checkpoints.
        pipe.load_lora_weights(lora, weight_name="pytorch_lora_weights.safetensors", adapter_name="pixel")
        adapters.append("pixel")
    if cfg.use_lcm:
        pipe.load_lora_weights(LCM_LORA, adapter_name="lcm")
        adapters.append("lcm")
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    if adapters:
        pipe.set_adapters(adapters, adapter_weights=[1.0] * len(adapters))

    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)
