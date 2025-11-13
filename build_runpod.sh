#!/bin/bash
set -e

# ===================================================================
# RunPod Serverless - Qwen-Image-Edit-2509 Docker Build Script
# ===================================================================

echo "=========================================="
echo "Building Qwen-Image-Edit-2509 for RunPod"
echo "=========================================="

# Configuration
IMAGE_NAME="qwen-image-edit-2509-runpod"
IMAGE_TAG="latest"
REGISTRY_URL=""  # Добавьте ваш registry URL (например: your-username/qwen-edit)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --registry)
            REGISTRY_URL="$2"
            shift 2
            ;;
        --push)
            PUSH_IMAGE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--hf-token TOKEN] [--tag TAG] [--registry URL] [--push]"
            exit 1
            ;;
    esac
done

# Check for HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "⚠️  Warning: HF_TOKEN not provided. Model download may fail."
    echo "   Use: --hf-token YOUR_TOKEN or set HF_TOKEN env var"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Build full image name
if [ -n "$REGISTRY_URL" ]; then
    FULL_IMAGE_NAME="${REGISTRY_URL}:${IMAGE_TAG}"
else
    FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
fi

echo ""
echo "Build configuration:"
echo "  Image name: $FULL_IMAGE_NAME"
echo "  HF Token: ${HF_TOKEN:+***set***}"
echo ""

# Build Docker image
echo "🔨 Building Docker image..."
docker build \
    --build-arg HF_TOKEN="${HF_TOKEN}" \
    --platform linux/amd64 \
    -t "${FULL_IMAGE_NAME}" \
    -f Dockerfile \
    .

if [ $? -eq 0 ]; then
    echo "✅ Build successful: ${FULL_IMAGE_NAME}"
else
    echo "❌ Build failed!"
    exit 1
fi

# Push to registry if requested
if [ "$PUSH_IMAGE" = true ]; then
    if [ -z "$REGISTRY_URL" ]; then
        echo "❌ Cannot push: --registry not specified"
        exit 1
    fi

    echo ""
    echo "📤 Pushing image to registry..."
    docker push "${FULL_IMAGE_NAME}"

    if [ $? -eq 0 ]; then
        echo "✅ Push successful!"
        echo ""
        echo "🚀 Deploy to RunPod:"
        echo "   1. Go to RunPod Serverless"
        echo "   2. Create new endpoint"
        echo "   3. Use image: ${FULL_IMAGE_NAME}"
        echo "   4. GPU: RTX 4090 / A100 (24GB+ VRAM recommended)"
    else
        echo "❌ Push failed!"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Build complete!"
echo "=========================================="
echo ""
echo "Test locally:"
echo "  docker run --gpus all -p 8000:8000 ${FULL_IMAGE_NAME}"
echo ""
echo "Image size:"
docker images "${FULL_IMAGE_NAME}" --format "{{.Size}}"
