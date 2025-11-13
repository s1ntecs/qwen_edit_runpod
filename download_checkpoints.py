import os
import torch

from diffusers import QwenImageEditPlusPipeline

from huggingface_hub import hf_hub_download, snapshot_download

# ------------------------- каталоги -------------------------
os.makedirs("loras", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

# Новая модель с поддержкой multi-image input
MODEL_REPO = "Qwen/Qwen-Image-Edit-2509"


def fetch_lora():
    """
    Скачивает LoRA (опционально, если понадобится)
    """
    print("[~] Downloading LoRA weights...")
    hf_hub_download(
        repo_id='flymy-ai/qwen-image-edit-inscene-lora',
        filename='flymy_qwen_image_edit_inscene_lora.safetensors',
        local_dir='./',
        local_dir_use_symlinks=False
    )
    print("[+] LoRA downloaded")


def fetch_model():
    """
    Скачивает Qwen-Image-Edit-2509 (новая модель с multi-image support)
    """
    print(f"[~] Downloading model: {MODEL_REPO}")
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=f"checkpoints/{MODEL_REPO.split('/')[-1]}",
        local_dir_use_symlinks=False
    )
    print("[+] Model downloaded")


# ------------------------- пайплайн -------------------------
def get_pipeline():
    """
    Загружает pipeline для тестирования
    """
    print(f"[~] Loading pipeline: {MODEL_REPO}")
    pipe = QwenImageEditPlusPipeline.from_pretrained(
        MODEL_REPO,
        torch_dtype=torch.bfloat16,
        cache_dir="checkpoints"
    )
    print("[+] Pipeline loaded successfully")
    return pipe


if __name__ == "__main__":
    print("=" * 60)
    print("Qwen-Image-Edit-2509 Checkpoint Downloader")
    print("=" * 60)

    # Скачиваем основную модель
    fetch_model()

    # Опционально: скачиваем LoRA (закомментировано по умолчанию)
    # fetch_lora()

    # Тестируем загрузку pipeline
    print("\n[~] Testing pipeline loading...")
    try:
        get_pipeline()
        print("[+] All checks passed!")
    except Exception as e:
        print(f"[ERROR] Failed to load pipeline: {e}")
