"""Krea2 demo 在 launcher、校验器和 ComfyUI runner 间共享的固定配置。"""

from pathlib import Path


KREA2_MODEL_NAME = r"krea2\[GPT逼真版]krea2GPTGrandPUSSYTruth_krea2GPT.safetensors"
KREA2_MODEL_PATH = Path(
    r"E:\Documents\ComfyUI\models\diffusion_models\krea2\[GPT逼真版]krea2GPTGrandPUSSYTruth_krea2GPT.safetensors"
)
KREA2_TEXT_ENCODER_NAME = "qwen3vl_4b_fp8_scaled.safetensors"
KREA2_VAE_NAME = r"krea2\qwen_image_vae.safetensors"
