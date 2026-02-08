# CLIP-DLUT

CLIP-DLUT is a deep learning-based image retouching system that leverages the CLIP model to optimize 3D Look-Up Tables (LUTs) based on text descriptions. Users can provide an input image and a text prompt (e.g., "Cyberpunk style night city"), and the system will generate a stylized image along with a compatible .cube 3D LUT file.

This project implements a complete backend pipeline including a REST API, an asynchronous task queue for GPU processing, and containerized deployment scripts.

## Features

- **Text-Driven Retouching**: Uses Chinese-CLIP to guide color grading based on natural language descriptions.
- **3D LUT Export**: Generates standard .cube files that can be used in professional post-processing software like Adobe Photoshop, Premiere Pro, or DaVinci Resolve.
- **Asynchronous Processing**: Decouples API requests from heavy GPU inference using Celery and Redis to ensure responsiveness.
- **Containerized Architecture**: Fully containerized using Podman (compatible with Docker) with NVIDIA GPU support.
- **Self-Healing Infrastructure**: Automatic resource cleanup (VRAM/RAM) after task execution.

## Architecture

The system consists of three main components running within a Podman Pod:

1.  **FastAPI Backend**: Handles HTTP requests, input validation, and task submission.
2.  **Redis**: Acts as the message broker and result backend for the task queue.
3.  **Celery Worker**: Executes the PyTorch-based optimization loop on the GPU, generating the optimized LUT and result image.

## Prerequisites

- Linux Operating System
- Python 3.14+ (if running locally without containers)
- **NVIDIA GPU** with appropriate drivers installed.
- **Podman** (recommended) or Docker.
- **NVIDIA Container Toolkit** (`nvidia-ctk` CLI) for generating CDI specifications to allow Podman to access the GPU.

## Installation and Deployment

### 1. Clone the Repository

```bash
git clone <repository-url>
cd CLIP-DLUT
```

### 2. Deploy with Podman

A convenience script is provided to build the image, configure NVIDIA CDI (if needed), and launch the service pod.

```bash
chmod +x podman_deploy.sh
./podman_deploy.sh
```

This script will:
- Check for NVIDIA CDI configuration.
- Build the `clip-dlut-app` container image.
- Create a Pod named `clip-dlut-pod`.
- Launch Redis, API Backend, and Celery Worker containers.
- Mount local `backend/` and `model/` directories for development convenience.

### 3. Verification

Check if the pod is running:

```bash
podman pod ps
podman logs -f clip-dlut-worker  # Monitor worker logs
```

The API will be available at `http://localhost:8000`.

## API Usage

### 1. Submit a Retouching Task

**Endpoint**: `POST /retouch`

**Payload**:
```json
{
  "image": "base64_encoded_string_of_image...",
  "target_prompt": "cyberpunk style night",
  "original_prompt": "night city",
  "iteration": 1000
}
```

**Response**:
```json
{
  "task_id": "uuid-string",
  "status": "pending"
}
```

### 2. Query Task Status

**Endpoint**: `POST /query_task`

**Payload**:
```json
{
  "task_id": "uuid-string",
  "include_image": true,
  "lut_format": "cube"
}
```

**Response** (when finished):
```json
{
  "status": "finished",
  "current_iteration": 1000,
  "image": "base64_encoded_result_preview...",
  "lut": "base64_encoded_cube_file..."
}
```

## Development

### Project Structure

- `backend/`: FastAPI application and Celery task definitions.
- `model/`: PyTorch implementation of the CLIP-DLUT model, loss functions, and LUT application logic.
- `storage/`: Directory for storing temporary uploads and results.
- `test_api.py`: Python script for end-to-end testing of the API.
- `podman_deploy.sh`: Deployment automation script.

### Running Tests

To verify the installation and the correctness of the LUT generation:

```bash
# Ensure the pod is running first
python test_api.py
```

This script performs the following actions:
1.  Submits a test task using a local image.
2.  Polls the server until completion.
3.  Downloads the result image and the .cube LUT.
4.  Applies the downloaded .cube LUT locally using Pillow to verify that the client-side rendering matches the server-side reference.
