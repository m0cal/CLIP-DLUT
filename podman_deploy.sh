#!/bin/bash

# Exit on error
set -e

POD_NAME="clip-dlut-pod"
IMAGE_NAME="clip-dlut-app"
REDIS_CONTAINER="clip-dlut-redis"
BACKEND_CONTAINER="clip-dlut-backend"
WORKER_CONTAINER="clip-dlut-worker"
PROXY_URL_CONTAINER="${PROXY_URL_CONTAINER:-}"
NO_PROXY_CONTAINER="localhost,127.0.0.1,::1,host.containers.internal"
HF_HOME_CONTAINER="${HF_HOME_CONTAINER:-/app/storage/hf}"
HF_DISABLE_XET="${HF_DISABLE_XET:-1}"
HF_ENDPOINT_CONTAINER="${HF_ENDPOINT_CONTAINER:-}"
HF_ENABLE_HF_TRANSFER="${HF_ENABLE_HF_TRANSFER:-1}"
HUGGINGFACE_HUB_DISABLE_XET="${HUGGINGFACE_HUB_DISABLE_XET:-1}"
HUGGINGFACE_HUB_BASE_URL_CONTAINER="${HUGGINGFACE_HUB_BASE_URL_CONTAINER:-}"

echo "=== CLIP-DLUT Podman Deployment ==="

# 1. Check/Generate NVIDIA CDI config
if ! podman run --rm --device nvidia.com/gpu=all docker.io/library/ubuntu nvidia-smi > /dev/null 2>&1; then
    echo "Checking NVIDIA CDI configuration..."
    if command -v nvidia-ctk &> /dev/null; then
        echo "Generating CDI specification at /etc/cdi/nvidia.yaml..."
        sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml
        echo "CDI generated. Verifying..."
    else 
        echo "WARNING: nvidia-ctk not found. GPU access might fail if not configured."
    fi
fi

# 2. Build Image
echo "Building application image..."
podman build -t $IMAGE_NAME .

# 3. Create Pod
# We expose port 8000. In a pod, all containers share localhost.
if podman pod exists $POD_NAME; then
    echo "Pod $POD_NAME exists. Stopping and removing old containers..."
    podman pod stop $POD_NAME
    podman pod rm $POD_NAME
fi

echo "Creating Pod $POD_NAME..."
podman pod create --name $POD_NAME -p 8000:8000

# 4. Start Redis
echo "Starting Redis..."
podman run -d --pod $POD_NAME \
    --name $REDIS_CONTAINER \
    docker.io/library/redis:alpine

# 5. Start Backend
# Note: Since they are in the same pod, they communicate via localhost.
# The internal config defaults to localhost:6379, so no env overrides needed for BROKER/BACKEND URLs.
echo "Starting Backend..."
podman run -d --pod $POD_NAME \
    --name $BACKEND_CONTAINER \
    -v $(pwd)/storage:/app/storage \
    -v $(pwd)/backend:/app/backend \
    -v $(pwd)/model:/app/model \
    -e HF_HOME=$HF_HOME_CONTAINER \
    -e TRANSFORMERS_CACHE=$HF_HOME_CONTAINER/transformers \
    -e HF_HUB_DISABLE_XET=$HF_DISABLE_XET \
    -e HF_HUB_ENABLE_HF_TRANSFER=$HF_ENABLE_HF_TRANSFER \
    -e HUGGINGFACE_HUB_DISABLE_XET=$HUGGINGFACE_HUB_DISABLE_XET \
    ${HF_ENDPOINT_CONTAINER:+-e HF_ENDPOINT=$HF_ENDPOINT_CONTAINER} \
    ${HUGGINGFACE_HUB_BASE_URL_CONTAINER:+-e HUGGINGFACE_HUB_BASE_URL=$HUGGINGFACE_HUB_BASE_URL_CONTAINER} \
    ${PROXY_URL_CONTAINER:+-e http_proxy=$PROXY_URL_CONTAINER} \
    ${PROXY_URL_CONTAINER:+-e https_proxy=$PROXY_URL_CONTAINER} \
    ${PROXY_URL_CONTAINER:+-e all_proxy=$PROXY_URL_CONTAINER} \
    ${PROXY_URL_CONTAINER:+-e NO_PROXY=$NO_PROXY_CONTAINER} \
    $IMAGE_NAME \
    uvicorn backend.main:app --host 0.0.0.0 --port 8000

# 6. Start Worker with GPU
echo "Starting Worker (with GPU)..."
# Using --device nvidia.com/gpu=all (CDI syntax)
podman run -d --pod $POD_NAME \
    --device nvidia.com/gpu=all \
    --name $WORKER_CONTAINER \
    -v $(pwd)/storage:/app/storage \
    -v $(pwd)/backend:/app/backend \
    -v $(pwd)/model:/app/model \
    -e HF_HOME=$HF_HOME_CONTAINER \
    -e TRANSFORMERS_CACHE=$HF_HOME_CONTAINER/transformers \
    -e HF_HUB_DISABLE_XET=$HF_DISABLE_XET \
    -e HF_HUB_ENABLE_HF_TRANSFER=$HF_ENABLE_HF_TRANSFER \
    -e HUGGINGFACE_HUB_DISABLE_XET=$HUGGINGFACE_HUB_DISABLE_XET \
    ${HF_ENDPOINT_CONTAINER:+-e HF_ENDPOINT=$HF_ENDPOINT_CONTAINER} \
    ${HUGGINGFACE_HUB_BASE_URL_CONTAINER:+-e HUGGINGFACE_HUB_BASE_URL=$HUGGINGFACE_HUB_BASE_URL_CONTAINER} \
    ${PROXY_URL_CONTAINER:+-e http_proxy=$PROXY_URL_CONTAINER} \
    ${PROXY_URL_CONTAINER:+-e https_proxy=$PROXY_URL_CONTAINER} \
    ${PROXY_URL_CONTAINER:+-e all_proxy=$PROXY_URL_CONTAINER} \
    ${PROXY_URL_CONTAINER:+-e NO_PROXY=$NO_PROXY_CONTAINER} \
    $IMAGE_NAME \
    celery -A backend.core.celery_app worker --loglevel=info --concurrency=1 --pool=solo

echo "=== Deployment Complete ==="
echo "Access the API at: http://localhost:8000/docs"
echo "View logs: podman logs -f $WORKER_CONTAINER"
