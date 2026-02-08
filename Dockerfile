FROM python:3.14-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
# Install python dependencies
# Note: In a real setup, we'd use a requirements.txt or poetry/uv
# Here we extract from pyproject or just install manually known deps
RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
RUN pip install fastapi uvicorn[standard] redis celery[redis] python-multipart pillow pydantic-settings transformers ftfy regex tqdm

COPY . .

# Create storage directories
RUN mkdir -p storage/uploads storage/results

# Expose port
EXPOSE 8000

# Default command (overwritten in docker-compose)
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
