# Qwen-Image-Edit-2509 RunPod Serverless

Production-ready serverless container for **Qwen-Image-Edit-2509** with **multi-image input** support for **RunPod Serverless**.

## Features

✨ **Multi-Image Input Support** - Process 1-3 images simultaneously for optimal results
- Person + Person combinations
- Person + Product editing
- Person + Scene composition
- Enhanced identity preservation across multiple subjects

🚀 **RunPod Serverless**
- REST API endpoint
- Auto-scaling GPU workers
- Fast cold start optimization

🎯 **Optimized Parameters**
- Uses latest `Qwen-Image-Edit-2509` model
- Recommended: 40 inference steps
- `true_cfg_scale=4.0` for better quality
- `guidance_scale=1.0` for stability

## Model Information

- **Model**: `Qwen/Qwen-Image-Edit-2509`
- **Pipeline**: `QwenImageEditPlusPipeline`
- **Precision**: bfloat16
- **Optimal Input**: 1-3 images
- **Resolution**: Auto-scaled to max 1024px

## Installation & Setup

### Быстрый старт

```bash
# 1. Клонируйте репозиторий
git clone <your-repo-url>
cd qwen_edit_runpod

# 2. Соберите Docker образ
./build_runpod.sh --hf-token YOUR_HF_TOKEN --push --registry dockerhub.io/username/qwen-edit

# 3. Деплой на RunPod
# Следуйте инструкциям в RUNPOD_DEPLOY.md
```

Полное руководство: **[RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md)**

## Usage

### Single Image Input

```python
import requests

payload = {
    "input": {
        "image_url": "https://example.com/image.jpg",
        "prompt": "Add sunglasses to the person",
        "negative_prompt": " ",
        "true_cfg_scale": 4.0,
        "guidance_scale": 1.0,
        "num_inference_steps": 40,
        "seed": 42
    }
}

response = requests.post("https://your-runpod-endpoint.com/run", json=payload)
result = response.json()
```

#### Multi-Image Input (NEW!)

```python
payload = {
    "input": {
        "image_urls": [
            "https://example.com/person1.jpg",
            "https://example.com/person2.jpg"
        ],
        "prompt": "The first person is on the left, the second person is on the right, they are facing each other in a park",
        "negative_prompt": " ",
        "true_cfg_scale": 4.0,
        "guidance_scale": 1.0,
        "num_inference_steps": 40,
        "num_images_per_prompt": 1,
        "seed": 42
    }
}

response = requests.post("https://your-runpod-endpoint.com/run", json=payload)
result = response.json()

# Response includes:
# - images_base64: list of base64 encoded images
# - num_input_images: number of input images processed
# - steps, seed, true_cfg_scale, guidance_scale
```

## API Parameters

### Input Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_url` / `image_urls` | str / list | **required** | Single URL or list of image URLs (1-3 optimal) |
| `prompt` | str | **required** | Text description of desired edit |
| `negative_prompt` | str | `" "` | What to avoid in generation |
| `num_inference_steps` | int | `40` | Number of denoising steps (1-100) |
| `guidance_scale` | float | `1.0` | Guidance strength (1.0-20.0) |
| `true_cfg_scale` | float | `4.0` | Qwen-specific CFG parameter (1.0-10.0) |
| `num_images_per_prompt` | int | `1` | Number of outputs per prompt (1-4) |
| `seed` | int | random | Random seed for reproducibility |

### Output Format

```json
{
  "images_base64": ["base64_string1", "base64_string2", ...],
  "num_input_images": 2,
  "steps": 40,
  "seed": 42,
  "true_cfg_scale": 4.0,
  "guidance_scale": 1.0,
  "time": 2.34
}
```

## Example Use Cases

### 1. Person + Person Composition
```python
{
    "image_urls": ["person1.jpg", "person2.jpg"],
    "prompt": "Two friends standing together at a beach sunset"
}
```

### 2. Person + Product Advertisement
```python
{
    "image_urls": ["model.jpg", "product.jpg"],
    "prompt": "Fashion model elegantly presenting the luxury handbag"
}
```

### 3. Person + Scene Integration
```python
{
    "image_urls": ["person.jpg", "background.jpg"],
    "prompt": "Person exploring the futuristic cityscape"
}
```

### 4. Single Image Text Edit
```python
{
    "image_url": "sign.jpg",
    "prompt": "Change the text to 'OPEN 24/7' in bold red letters"
}
```

## Docker Build

Используйте включенный build скрипт для сборки:

```bash
./build_runpod.sh --hf-token YOUR_HF_TOKEN
```

Подробные инструкции: [RUNPOD_DEPLOY.md](RUNPOD_DEPLOY.md)

## Requirements

- Python 3.10+
- CUDA 11.8+ / CUDA 12.1+
- GPU with 16GB+ VRAM (24GB recommended for multi-image)
- diffusers (latest from git)
- torch 2.0+
- transformers, accelerate

## Performance Tips

1. **Optimal Image Count**: Use 1-3 images for best results
2. **Resolution**: Images auto-scaled to 1024px max dimension
3. **Steps**: 40 steps recommended (balance of quality/speed)
4. **Batch Size**: Keep `num_images_per_prompt=1` for stability
5. **CFG Scales**:
   - `true_cfg_scale=4.0` for quality
   - `guidance_scale=1.0` for stability

## Troubleshooting

### CUDA Out of Memory
- Reduce image resolution
- Decrease `num_inference_steps`
- Process fewer images at once

### Poor Quality Results
- Increase `num_inference_steps` to 50-60
- Adjust `true_cfg_scale` (try 3.5-5.0)
- Use more descriptive prompts
- Ensure input images are clear and high quality

### Model Loading Issues
```bash
# Re-download model
python download_checkpoints.py

# Or manually:
from diffusers import QwenImageEditPlusPipeline
pipe = QwenImageEditPlusPipeline.from_pretrained("Qwen/Qwen-Image-Edit-2509")
```

## Credits

- Model: [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509) by Alibaba Qwen Team
- Based on Qwen-Image architecture with multi-image training
- RunPod Serverless implementation

## License

This implementation follows the model's license from Hugging Face.

## Changelog

### v2.0 (2025)
- ✨ Added multi-image input support (1-3 images)
- 🔄 Updated to Qwen-Image-Edit-2509
- 🚀 Migrated to QwenImageEditPlusPipeline
- 📝 Enhanced API with new parameters
- 🎯 Optimized default parameters (40 steps, cfg scales)

### v1.0
- Initial implementation with Qwen-Image-Edit (single image only)
