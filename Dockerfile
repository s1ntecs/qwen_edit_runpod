FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=off \
    SHELL=/bin/bash
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# 1) System deps
RUN apt update && \
    apt install -y --no-install-recommends \
      python3-dev python3-pip python3.10-venv \
      fonts-dejavu-core git git-lfs jq wget curl \
      libglib2.0-0 libsm6 libgl1 libxrender1 libxext6 \
      ffmpeg procps && \
    rm -rf /var/lib/apt/lists/* && \
    git lfs install

WORKDIR /workspace

# 2) Copy requirements
COPY requirements.txt .

# 3) Install torch + torchvision + torchaudio + xformers (all pinned) with retries
RUN pip3 install --upgrade pip && \
    pip3 install \
      --no-cache-dir \
      --timeout=120 \
      --retries=5 \
      torch==2.5.0 torchvision==0.20.0 torchaudio==2.5.0 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install \
      --no-cache-dir \
      --timeout=120 \
      --retries=5 \
      xformers==0.0.28.post1 \
        --index-url https://download.pytorch.org/whl/cu124 && \
    pip3 install --no-cache-dir -r requirements.txt

RUN pip3 uninstall -y transformers diffusers || true && \
    pip3 install --no-cache-dir \
      "transformers>=4.53.3" \
      "accelerate>=0.33.0" \
      "qwen-vl-utils>=0.0.8" && \
    pip3 install --no-cache-dir "git+https://github.com/huggingface/diffusers@main"

# Быстрая проверка наличия класса ещё на этапе сборки
RUN python - <<'PY'
import importlib, transformers, diffusers
print("Transformers:", transformers.__version__)
print("Diffusers:", diffusers.__version__)
m = importlib.import_module("transformers")
assert hasattr(m, "Qwen2_5_VLForConditionalGeneration"), "Нет Qwen2_5_VLForConditionalGeneration"
print("OK: класс найден")
PY
# 4) Copy rest of the code
COPY . .

ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN huggingface-cli login --token $HF_TOKEN --add-to-git-credential

# 5) Prepare dirs & checkpoints
RUN mkdir -p loras checkpoints && \
    python3 download_checkpoints.py

# 6) Entry point
COPY --chmod=755 start_standalone.sh /start.sh
ENTRYPOINT ["/start.sh"]
