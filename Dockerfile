# Attention Safety Guard - RunPod Serverless Dockerfile
# Optimized for 24GB VRAM GPUs (RTX 4090, L4, A10G)

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade torch and torchvision to compatible versions
RUN pip install --no-cache-dir --upgrade torch torchvision

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY core.py .
COPY api.py .
COPY handler.py .

# Copy policy file
COPY data/policy.md /app/data/policy.md

# Pre-download the model during build (optional but recommended)
# This increases image size but eliminates cold start model download
RUN python -c "from core import load_model; load_model()" || echo "Model will be downloaded at runtime"

# Expose port for FastAPI (used in Pod mode)
EXPOSE 8000

# Default: RunPod Serverless mode (uses handler.py)
# For Pod mode (FastAPI): docker run ... python api.py
ENV RUNPOD_MODE=serverless

CMD ["python", "-u", "handler.py"]
