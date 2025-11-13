# RunPod Serverless Deployment Guide
## Qwen-Image-Edit-2509 Multi-Image Support

Полное руководство по сборке и деплою контейнера на RunPod Serverless.

---

## 📋 Требования

### Локальная сборка
- Docker Desktop установлен
- 50GB+ свободного места
- Hugging Face токен (для скачивания модели)

### RunPod Deployment
- RunPod аккаунт
- Docker Hub / другой registry
- GPU: RTX 4090, A100, или аналог (24GB+ VRAM рекомендуется)

---

## 🔨 Сборка Docker образа

### 1. Получите Hugging Face токен

Зайдите на https://huggingface.co/settings/tokens и создайте токен с правами `read`.

### 2. Соберите образ

```bash
# Базовая сборка
./build_runpod.sh --hf-token YOUR_HF_TOKEN

# С тегом и push в registry
./build_runpod.sh \
  --hf-token YOUR_HF_TOKEN \
  --tag v1.0 \
  --registry dockerhub.io/username/qwen-edit \
  --push
```

#### Параметры build скрипта:

| Параметр | Описание |
|----------|----------|
| `--hf-token` | Hugging Face токен для скачивания модели |
| `--tag` | Тег образа (default: `latest`) |
| `--registry` | URL registry для push |
| `--push` | Автоматически push в registry после сборки |

### 3. Ручная сборка (опционально)

```bash
# Экспортируйте токен
export HF_TOKEN=hf_xxxxxxxxxxxxx

# Соберите образ
docker build \
  --build-arg HF_TOKEN=$HF_TOKEN \
  --platform linux/amd64 \
  -t qwen-edit-2509:latest \
  .

# Push в registry
docker tag qwen-edit-2509:latest your-registry/qwen-edit-2509:latest
docker push your-registry/qwen-edit-2509:latest
```

---

## 🚀 Деплой на RunPod

### Шаг 1: Подготовка образа

После успешной сборки, убедитесь что образ доступен в публичном registry:

```bash
# Проверьте образ локально
docker images | grep qwen-edit

# Push в Docker Hub или другой registry
docker push YOUR_REGISTRY/qwen-edit-2509:latest
```

### Шаг 2: Создание Serverless Endpoint

1. Войдите в RunPod Console
2. Перейдите в **Serverless** → **New Endpoint**
3. Настройте endpoint:

**Container Configuration:**
```
Image: YOUR_REGISTRY/qwen-edit-2509:latest
Docker Command: (оставьте пустым)
Container Disk: 20GB минимум
```

**GPU Configuration:**
```
GPU Type: RTX 4090, A100 (рекомендуется 24GB VRAM)
Max Workers: 3
Idle Timeout: 5 секунд
Execution Timeout: 600 секунд (10 минут)
```

**Environment Variables:**
```
PYTHONUNBUFFERED=1
HF_HOME=/app/checkpoints
```

4. Нажмите **Deploy**

### Шаг 3: Тестирование

После деплоя получите endpoint URL из панели RunPod.

```python
import requests
import base64

# Ваш RunPod endpoint
ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
API_KEY = "your_runpod_api_key"

headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# Single image test
payload = {
    "input": {
        "image_url": "https://example.com/image.jpg",
        "prompt": "Add sunglasses to the person",
        "num_inference_steps": 40,
        "true_cfg_scale": 4.0,
        "guidance_scale": 1.0,
        "seed": 42
    }
}

response = requests.post(ENDPOINT_URL, json=payload, headers=headers)
result = response.json()

# Результат в result["output"]["images_base64"]
```

---

## 🎯 API примеры

### Single Image Input

```python
payload = {
    "input": {
        "image_url": "https://example.com/person.jpg",
        "prompt": "Add wizard hat and robes",
        "negative_prompt": "blurry, low quality",
        "num_inference_steps": 40,
        "true_cfg_scale": 4.0,
        "guidance_scale": 1.0,
        "seed": 42
    }
}
```

### Multi-Image Input (2-3 изображения)

```python
payload = {
    "input": {
        "image_urls": [
            "https://example.com/person1.jpg",
            "https://example.com/person2.jpg"
        ],
        "prompt": "The first person on the left wearing red, the second person on the right wearing blue, standing together in a park",
        "num_inference_steps": 40,
        "true_cfg_scale": 4.0,
        "guidance_scale": 1.0,
        "num_images_per_prompt": 1,
        "seed": 123
    }
}
```

### Response Format

```json
{
  "output": {
    "images_base64": ["base64_string1", "base64_string2"],
    "num_input_images": 2,
    "steps": 40,
    "seed": 123,
    "true_cfg_scale": 4.0,
    "guidance_scale": 1.0,
    "time": 3.45
  }
}
```

---

## 📊 Параметры API

| Параметр | Тип | Default | Описание |
|----------|-----|---------|----------|
| `image_url` | string | **required*** | URL одного изображения |
| `image_urls` | array | **required*** | Массив URL изображений (1-3) |
| `prompt` | string | **required** | Описание редактирования |
| `negative_prompt` | string | `" "` | Что избегать |
| `num_inference_steps` | int | `40` | Шаги диффузии (1-100) |
| `true_cfg_scale` | float | `4.0` | CFG scale Qwen (1.0-10.0) |
| `guidance_scale` | float | `1.0` | Guidance scale (1.0-20.0) |
| `num_images_per_prompt` | int | `1` | Количество выходных изображений |
| `seed` | int | random | Сид для воспроизводимости |

*Укажите либо `image_url`, либо `image_urls`

---

## 🔧 Troubleshooting

### Проблема: Контейнер не стартует

**Решение:**
```bash
# Проверьте логи в RunPod Console
# Убедитесь что HF_TOKEN был передан при сборке
# Проверьте что модель скачалась: docker run ... ls -la checkpoints/
```

### Проблема: CUDA Out of Memory

**Решение:**
- Увеличьте GPU до 24GB VRAM
- Уменьшите `num_inference_steps` до 30-35
- Процессируйте меньше изображений одновременно

### Проблема: Долгий cold start

**Решение:**
- Модель ~20GB, первый запуск может занять 2-3 минуты
- Установите `Idle Timeout = 60` секунд чтобы держать worker активным
- Используйте `Max Workers > 1` для параллельных запросов

### Проблема: Плохое качество результата

**Решение:**
```python
# Увеличьте steps
"num_inference_steps": 50

# Настройте CFG
"true_cfg_scale": 3.5  # или 5.0

# Более детальный prompt
"prompt": "Detailed description with specific positions, colors, and actions..."
```

---

## 📈 Производительность

| GPU | VRAM | Single Image | Multi-Image (2) | Multi-Image (3) |
|-----|------|--------------|-----------------|-----------------|
| RTX 4090 | 24GB | ~3-4s | ~5-6s | ~7-9s |
| A100 | 40GB | ~2-3s | ~4-5s | ~5-7s |
| RTX 3090 | 24GB | ~4-5s | ~6-8s | ~9-11s |

*Время на 40 inference steps, разрешение 1024px

---

## 💡 Best Practices

### 1. Оптимальные параметры
```python
{
    "num_inference_steps": 40,  # Баланс качество/скорость
    "true_cfg_scale": 4.0,      # Рекомендовано для Qwen-2509
    "guidance_scale": 1.0,      # Стабильность
}
```

### 2. Multi-image prompting
```python
# Хорошо: четкое описание позиций
"The wizard bear on the left, knight bear on the right, facing each other"

# Плохо: неясные позиции
"Two bears together"
```

### 3. Количество изображений
- **1 image**: Редактирование одного изображения
- **2 images**: Оптимально для композиции
- **3 images**: Сложные сцены (увеличивает VRAM usage)

### 4. Batch processing
```python
# Вместо цикла, используйте num_images_per_prompt
"num_images_per_prompt": 4  # Создаст 4 варианта за один запрос
```

---

## 📝 Структура проекта

```
qwen_edit_runpod/
├── Dockerfile              # RunPod Docker образ
├── rp_handler.py          # RunPod serverless handler
├── download_checkpoints.py # Скрипт загрузки модели
├── requirements.txt       # Python зависимости
├── build_runpod.sh       # Build скрипт
├── start_standalone.sh   # Entrypoint скрипт
├── .dockerignore         # Исключения для Docker
└── README.md             # Документация
```

---

## 🔐 Безопасность

### Secrets Management

**НЕ коммитьте:**
- HF_TOKEN в Dockerfile
- API ключи
- Приватные URL изображений

**Используйте:**
- Build args: `--build-arg HF_TOKEN=$HF_TOKEN`
- Environment variables в RunPod Console
- Secrets manager для production

---

## 📞 Поддержка

**Документация:**
- [RunPod Serverless Docs](https://docs.runpod.io/serverless/overview)
- [Qwen-Image-Edit-2509 Model Card](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)

**Issues:**
- GitHub Issues для багов в коде
- RunPod Discord для проблем с платформой

---

## 📄 Changelog

### v2.0 - Multi-Image Support
- ✨ Поддержка 1-3 изображений
- 🔄 Qwen-Image-Edit-2509 model
- 🚀 Оптимизированный Dockerfile
- 📝 Автоматический build скрипт

### v1.0 - Initial Release
- Single image support
- RunPod serverless integration
