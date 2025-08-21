import os
import torch

from diffusers import QwenImageEditPipeline

from huggingface_hub import hf_hub_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32


def fetch_lora():
    """
    Скачивает LoRA
    """
    hf_hub_download(
        repo_id='flymy-ai/qwen-image-edit-inscene-lora',
        filename='flymy_qwen_image_edit_inscene_lora.safetensors',
        local_dir='./',
        local_dir_use_symlinks=False
    )


# ------------------------- пайплайн -------------------------
def get_pipeline():
    QwenImageEditPipeline.from_pretrained(
        "Qwen/Qwen-Image-Edit"
    )


if __name__ == "__main__":
    fetch_lora()
    get_pipeline()
