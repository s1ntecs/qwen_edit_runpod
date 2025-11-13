# import cv2
import base64, io, random, time, numpy as np, torch
from typing import Any, Dict, List, Union
from PIL import Image

from diffusers import QwenImageEditPlusPipeline

import runpod
from runpod.serverless.utils.rp_download import file as rp_file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --------------------------- КОНСТАНТЫ ----------------------------------- #
MAX_SEED = np.iinfo(np.int16).max
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_STEPS = 250
TARGET_RES = 1024
DEFAULT_NUM_INFERENCE_STEPS = 40  # Qwen-Image-Edit-2509 рекомендует 40


DTYPE = torch.bfloat16


logger = RunPodLogger()


# ------------------------- ФУНКЦИИ-ПОМОЩНИКИ ----------------------------- #
def filter_items(colors_list, items_list, items_to_remove):
    keep_c, keep_i = [], []
    for c, it in zip(colors_list, items_list):
        if it not in items_to_remove:
            keep_c.append(c)
            keep_i.append(it)
    return keep_c, keep_i


def resize_dimensions(dimensions, target_size):
    w, h = dimensions
    if w < target_size and h < target_size:
        return dimensions
    if w > h:
        ar = h / w
        return target_size, int(target_size * ar)
    ar = w / h
    return int(target_size * ar), target_size


def url_to_pil(url: str) -> Image.Image:
    info = rp_file(url)
    return Image.open(info["file_path"]).convert("RGB")


def urls_to_pil_list(urls: Union[str, List[str]]) -> List[Image.Image]:
    """Конвертирует URL или список URLs в список PIL изображений"""
    if isinstance(urls, str):
        urls = [urls]
    return [url_to_pil(url) for url in urls]


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def round_to_multiple(x, m=8):
    return (x // m) * m


def compute_work_resolution(w, h, max_side=1024):
    # масштабируем так, чтобы большая сторона <= max_side
    scale = min(max_side / max(w, h), 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    # выравниваем до кратных 8
    new_w = round_to_multiple(new_w, 8)
    new_h = round_to_multiple(new_h, 8)
    return max(new_w, 8), max(new_h, 8)


# ------------------------- ЗАГРУЗКА МОДЕЛЕЙ ------------------------------ #
repo_id = "Qwen/Qwen-Image-Edit-2509"
PIPELINE = QwenImageEditPlusPipeline.from_pretrained(
    repo_id,
    torch_dtype=torch.bfloat16
)

PIPELINE.to(DEVICE)
PIPELINE.set_progress_bar_config(disable=True)
# lora_path = "./flymy_qwen_image_edit_inscene_lora.safetensors"
# PIPELINE.load_lora_weights(lora_path)


# ------------------------- ОСНОВНОЙ HANDLER ------------------------------ #
def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    try:
        payload = job.get("input", {})

        # Поддержка как старого API (image_url), так и нового (image_urls)
        image_urls = payload.get("image_urls") or payload.get("image_url")
        if not image_urls:
            return {"error": "'image_urls' or 'image_url' is required"}

        prompt = payload.get("prompt")
        if not prompt:
            return {"error": "'prompt' is required"}

        negative_prompt = payload.get("negative_prompt", " ")

        # Параметры Qwen-Image-Edit-2509
        true_cfg_scale = float(payload.get("true_cfg_scale", 4.0))
        guidance_scale = float(payload.get("guidance_scale", 1.0))

        steps = min(
            int(payload.get("num_inference_steps") or payload.get("steps", DEFAULT_NUM_INFERENCE_STEPS)),
            MAX_STEPS
        )

        num_images_per_prompt = int(payload.get("num_images_per_prompt", 1))

        seed = int(payload.get("seed", random.randint(0, MAX_SEED)))
        generator = torch.Generator(device=DEVICE).manual_seed(seed)

        # Загружаем изображения (поддержка multi-image input)
        image_pils = urls_to_pil_list(image_urls)
        logger.info(f"Loaded {len(image_pils)} input image(s)")

        # Определяем рабочее разрешение на основе первого изображения
        orig_w, orig_h = image_pils[0].size
        work_w, work_h = compute_work_resolution(orig_w, orig_h, TARGET_RES)

        # Resize всех изображений до одинакового размера
        resized_images = []
        for img in image_pils:
            resized = img.resize((work_w, work_h), Image.Resampling.LANCZOS)
            resized_images.append(resized.convert("RGB"))

        # Для совместимости: если только одно изображение, передаем его напрямую
        # Если несколько - передаем список
        pipeline_input = resized_images if len(resized_images) > 1 else resized_images[0]

        # ------------------ генерация ---------------- #
        with torch.inference_mode():
            output = PIPELINE(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=pipeline_input,
                num_inference_steps=steps,
                true_cfg_scale=true_cfg_scale,
                guidance_scale=guidance_scale,
                generator=generator,
                num_images_per_prompt=num_images_per_prompt,
            )
            result_images = output.images

        # Добавляем входные изображения для сравнения
        result_images.extend(resized_images)

        return {
            "images_base64": [pil_to_b64(i) for i in result_images],
            "time": round(time.time() - job["created"], 2) if "created" in job else None,
            "num_input_images": len(image_pils),
            "steps": steps,
            "seed": seed,
            "true_cfg_scale": true_cfg_scale,
            "guidance_scale": guidance_scale,
        }

    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "CUDA out of memory" in str(exc):
            return {"error": "CUDA OOM — уменьшите 'steps' или размер изображения."} # noqa
        return {"error": str(exc)}
    except Exception as exc:
        import traceback
        return {"error": str(exc), "trace": traceback.format_exc(limit=5)}


# ------------------------- RUN WORKER ------------------------------------ #
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
